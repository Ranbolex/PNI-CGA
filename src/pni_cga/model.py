"""Joint configurations, predictive point channels and verified compatibility.

The prediction table scores plans. A separately supplied compatibility table
authorizes exclusions. Predictions with small or zero probability do not
authorize deleting a configuration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Mapping, Sequence

UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Cost:
    views: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    seconds: float = 0.0

    def __post_init__(self):
        for name in ("views", "input_tokens", "output_tokens", "calls"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not isfinite(self.seconds) or self.seconds < 0:
            raise ValueError("seconds must be finite and nonnegative")

    def __add__(self, other: Cost) -> Cost:
        return Cost(*(getattr(self, n) + getattr(other, n)
                      for n in self.__dataclass_fields__))


@dataclass(frozen=True)
class Budget:
    views: int = 4
    output_tokens: int = 65536
    calls: int = 32
    seconds: float = 600.0

    def __post_init__(self):
        Cost(views=self.views, output_tokens=self.output_tokens,
             calls=self.calls, seconds=self.seconds)

    def fits(self, cost: Cost) -> bool:
        return (cost.views <= self.views and cost.output_tokens <= self.output_tokens
                and cost.calls <= self.calls and cost.seconds <= self.seconds + 1e-10)

    def spend(self, cost: Cost) -> Budget:
        if not self.fits(cost):
            raise ValueError("Resource budget exceeded")
        return Budget(self.views - cost.views, self.output_tokens - cost.output_tokens,
                      self.calls - cost.calls, max(0.0, self.seconds - cost.seconds))


@dataclass(frozen=True)
class Configuration:
    id: str
    answer: str
    facts: tuple[tuple[str, str], ...] = ()
    mass: float = 1.0
    anonymous: bool = False

    def __post_init__(self):
        if not self.id or not self.answer or not isfinite(self.mass) or self.mass <= 0:
            raise ValueError("Configurations need IDs, answers and positive finite mass")
        if len(dict(self.facts)) != len(self.facts):
            raise ValueError("A joint configuration cannot assign one fact twice")


@dataclass(frozen=True)
class Action:
    id: str
    cost: Cost = field(default_factory=lambda: Cost(views=1))
    predicted_cost: float = 1.0
    primitive: str = ""
    anchor_ids: tuple[str, ...] = ()
    box: tuple[int, int, int, int] | None = None
    requires_scope: bool = False

    def __post_init__(self):
        if not self.id or self.cost.views < 1:
            raise ValueError("Each acquisition action needs an ID and at least one view")
        if not isfinite(self.predicted_cost) or self.predicted_cost < 0:
            raise ValueError("Invalid predicted tie-breaking cost")
        if self.box is not None:
            x0, y0, x1, y1 = self.box
            if not all(isinstance(v, int) for v in self.box) or x0 >= x1 or y0 >= y1:
                raise ValueError("Action box must have ordered integer root coordinates")


class AcquisitionModel:
    """A finite joint task model with one fixed latent configuration per run.

    Outcomes of distinct actions are conditionally independent given that
    configuration. Correlations must be encoded by augmenting configurations.
    The model contains possible worlds, never the environment's true-world ID.
    """

    def __init__(self, configurations: Sequence[Configuration], actions: Sequence[Action],
                 channels: Mapping[str, Mapping[str, Mapping[str, float]]],
                 compatibility: Mapping[str, Mapping[str, Sequence[str]]]):
        if not configurations or len({h.id for h in configurations}) != len(configurations):
            raise ValueError("Require nonempty, uniquely identified joint configurations")
        if len({a.id for a in actions}) != len(actions):
            raise ValueError("Action IDs must be unique")
        self.configurations = MappingProxyType({h.id: h for h in configurations})
        self.actions = MappingProxyType({a.id: a for a in actions})
        self.initial_state = tuple(sorted(self.configurations))
        ids, aids = set(self.configurations), set(self.actions)
        if set(channels) != aids or set(compatibility) != aids:
            raise ValueError("Every action needs prediction and verification tables")
        laws, allowed, alphabets = {}, {}, {}
        for aid in self.actions:
            if set(channels[aid]) != ids:
                raise ValueError("Every action needs a predictive row for every configuration")
            outcomes = tuple(sorted(compatibility[aid]))
            if UNKNOWN not in outcomes:
                raise ValueError("Every action must support conservative UNKNOWN")
            alphabets[aid] = outcomes
            laws[aid] = {}
            allowed[aid] = {}
            for hid in self.initial_state:
                row = dict(channels[aid][hid])
                if set(row) != set(outcomes) or any(not isfinite(p) or p < 0 or p > 1
                                                   for p in row.values()):
                    raise ValueError("Invalid predictive outcome row")
                if abs(sum(row.values()) - 1) > 1e-10:
                    raise ValueError("Predictive probabilities must sum to one")
                laws[aid][hid] = MappingProxyType(row)
            for outcome in outcomes:
                keep = frozenset(compatibility[aid][outcome])
                if not keep <= ids:
                    raise ValueError("Compatibility contains an unknown configuration")
                if outcome == UNKNOWN and keep != ids:
                    raise ValueError("UNKNOWN must retain every configuration")
                allowed[aid][outcome] = keep
            laws[aid] = MappingProxyType(laws[aid])
            allowed[aid] = MappingProxyType(allowed[aid])
        self.channels = MappingProxyType(laws)
        self.compatibility = MappingProxyType(allowed)
        self.outcomes = MappingProxyType(alphabets)

    def state(self, ids: Sequence[str] | None = None) -> tuple[str, ...]:
        result = self.initial_state if ids is None else tuple(sorted(ids))
        if len(set(result)) != len(result) or not set(result) <= self.configurations.keys():
            raise ValueError("Invalid retained configuration set")
        return result

    def answer(self, state: Sequence[str]) -> str | None:
        answers = {self.configurations[h].answer for h in self.state(state)}
        return next(iter(answers)) if len(answers) == 1 else None

    def update(self, state: Sequence[str], action: str, outcome: str) -> tuple[str, ...]:
        keep = self.compatibility[action][outcome]
        return tuple(h for h in self.state(state) if h in keep)

    def weights(self, state: Sequence[str] | None = None) -> tuple[float, ...]:
        """Answer-balanced mass; used for mean closure/cost tie-breaking only."""
        ids = self.state(state)
        totals: dict[str, float] = {}
        for hid in ids:
            h = self.configurations[hid]
            totals[h.answer] = totals.get(h.answer, 0.0) + h.mass
        return tuple(self.configurations[h].mass /
                     (len(totals) * totals[self.configurations[h].answer]) for h in ids)

    def pair_mass(self, state: Sequence[str], root: Sequence[str] | None = None) -> float:
        root_ids = self.state(root)
        mass = dict(zip(root_ids, self.weights(root_ids)))
        def cross(ids):
            return sum(mass[a] * mass[b] for i, a in enumerate(ids) for b in ids[i + 1:]
                       if self.configurations[a].answer != self.configurations[b].answer)
        normalizer = cross(root_ids)
        return cross(self.state(state)) / normalizer if normalizer else 0.0

    def reachable(self, state: Sequence[str], action: str, outcome: str) -> bool:
        return any(self.channels[action][h][outcome] > 0 for h in self.state(state))
