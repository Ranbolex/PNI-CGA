"""Finite manuscript witness and complementary observation examples."""
from itertools import product

from .model import Action, AcquisitionModel, Configuration, UNKNOWN


def deterministic_model(configurations, actions, observations, unknown_probability=0.0):
    if not 0 <= unknown_probability <= 1:
        raise ValueError("Invalid unreadability probability")
    channels, compatibility = {}, {}
    ids = [h.id for h in configurations]
    for action in actions:
        aid = action.id
        if any(observations[aid][h] == UNKNOWN for h in ids):
            raise ValueError("Reserved UNKNOWN outcome cannot encode a readable value")
        outcomes = sorted(set(observations[aid].values()) | {UNKNOWN})
        channels[aid] = {h: {o: unknown_probability if o == UNKNOWN else
                            (1 - unknown_probability if observations[aid][h] == o else 0.0)
                            for o in outcomes} for h in ids}
        compatibility[aid] = {o: ids if o == UNKNOWN else
                              [h for h in ids if observations[aid][h] == o] for o in outcomes}
    return AcquisitionModel(configurations, actions, channels, compatibility)


def six_configuration_model() -> AcquisitionModel:
    bits = ("0011", "0101", "1011", "1100", "1101", "1111")
    labels = ("1", "1", "0", "0", "0", "1")
    hs = [Configuration(f"h{i}", y, tuple((f"region{j}", bit) for j, bit in enumerate(row)))
          for i, (row, y) in enumerate(zip(bits, labels))]
    actions = [Action(f"a{j}", primitive=f"region{j}", anchor_ids=(f"tile{j}",),
                      box=(j * 64, 0, (j + 1) * 64, 64)) for j in range(4)]
    observations = {a.id: {f"h{i}": row[j] for i, row in enumerate(bits)}
                    for j, a in enumerate(actions)}
    return deterministic_model(hs, actions, observations)


def complementary_model(unknown_probability=0.0) -> AcquisitionModel:
    hs = [Configuration(f"h{x}{y}", str(x ^ y), (("left", str(x)), ("right", str(y))))
          for x, y in product((0, 1), repeat=2)]
    actions = [Action("left", primitive="left"), Action("right", primitive="right")]
    observations = {a.id: {h.id: dict(h.facts)[a.id] for h in hs} for a in actions}
    return deterministic_model(hs, actions, observations, unknown_probability)
