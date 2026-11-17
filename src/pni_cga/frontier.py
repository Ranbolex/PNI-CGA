"""Exact finite denotations preserve bindings and anonymous alternatives."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .model import AcquisitionModel


@dataclass(frozen=True)
class FrontierBlock:
    answer: str
    members: frozenset[str]


@dataclass(frozen=True)
class Counterexample:
    first: str
    second: str
    missing_predicates: tuple[str, ...]


class JointFrontier:
    def __init__(self, model: AcquisitionModel, state: Sequence[str] | None = None):
        self.model = model
        self.state = model.state(state)
        self.version = 0
        self.evidence: list[tuple[str, str]] = []

    def counterexample(self, weights: dict[str, float] | None = None) -> Counterexample | None:
        weights = weights or {}
        ids = sorted(self.state, key=lambda h: (-weights.get(h, 0.0), h))
        for a in ids:
            for b in ids:
                first, second = self.model.configurations[a], self.model.configurations[b]
                if first.answer == second.answer:
                    continue
                fa, fb = dict(first.facts), dict(second.facts)
                missing = tuple(sorted(k for k in fa.keys() | fb.keys() if fa.get(k) != fb.get(k)))
                return Counterexample(a, b, missing)
        return None

    def observe(self, action: str, outcome: str) -> tuple[str, ...]:
        self.state = self.model.update(self.state, action, outcome)
        self.evidence.append((action, outcome))
        self.version += 1
        return self.state

    def compress(self, max_blocks: int) -> tuple[FrontierBlock, ...]:
        groups: dict[str, set[str]] = {}
        for h in self.state:
            groups.setdefault(self.model.configurations[h].answer, set()).add(h)
        if max_blocks < len(groups):
            raise ValueError("Cannot drop an answer class to satisfy frontier capacity")
        return tuple(FrontierBlock(y, frozenset(ids)) for y, ids in sorted(groups.items()))

    def expand(self, blocks: Sequence[FrontierBlock]) -> tuple[str, ...]:
        result: set[str] = set()
        for block in blocks:
            members = self.model.state(tuple(block.members))
            if not members or result.intersection(members):
                raise ValueError("Frontier blocks must be nonempty and disjoint")
            if any(self.model.configurations[h].answer != block.answer for h in members):
                raise ValueError("Compression must preserve answer-homogeneous joint members")
            result.update(members)
        return tuple(sorted(result))
