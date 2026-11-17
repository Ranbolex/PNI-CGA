"""Budget-feasible contingent policies and fixed-world trajectory evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import prod
from typing import Sequence

import numpy as np

from .model import AcquisitionModel, Budget


@dataclass(frozen=True)
class PolicyTree:
    action: str | None = None
    children: tuple[tuple[str, PolicyTree], ...] = ()

    def to_dict(self) -> dict:
        return {"action": self.action,
                "children": {o: t.to_dict() for o, t in self.children}}

    @classmethod
    def from_dict(cls, value: dict) -> PolicyTree:
        return cls(value["action"], tuple((o, cls.from_dict(t))
                                         for o, t in sorted(value.get("children", {}).items())))


STOP = PolicyTree()


def validate_structure(model: AcquisitionModel, tree: PolicyTree,
                       used: frozenset[str] = frozenset()) -> None:
    """Validate outcome completeness and no repeated action on any branch."""
    if tree.action is None:
        if tree.children:
            raise ValueError("Terminal policy has children")
        return
    aid = tree.action
    if aid not in model.actions or aid in used:
        raise ValueError("Unknown action or repeated pixels in policy")
    branches = dict(tree.children)
    if len(branches) != len(tree.children) or set(branches) != set(model.outcomes[aid]):
        raise ValueError("Incomplete policy branches")
    for child in branches.values():
        validate_structure(model, child, used | {aid})


def validate_policy(model: AcquisitionModel, tree: PolicyTree, budget: Budget,
                    depth: int, used: frozenset[str] = frozenset()) -> None:
    if tree.action is None:
        if tree.children:
            raise ValueError("Terminal policies cannot have children")
        return
    if depth < 1 or tree.action in used or tree.action not in model.actions:
        raise ValueError("Invalid depth, repeated read, or unknown action")
    action = model.actions[tree.action]
    remaining = budget.spend(action.cost)
    branches = dict(tree.children)
    if len(branches) != len(tree.children) or set(branches) != set(model.outcomes[action.id]):
        raise ValueError("A contingent policy needs every modeled outcome exactly once")
    for child in branches.values():
        validate_policy(model, child, remaining, depth - 1, used | {action.id})


def enumerate_policies(model: AcquisitionModel, budget: Budget, depth: int,
                       state: Sequence[str] | None = None, max_trees: int = 100000) -> tuple[PolicyTree, ...]:
    """Enumerate all no-repeat trees, including early stopping and UNKNOWN.

    Stops at a single-answer frontier. Branches impossible under every root
    configuration use STOP. The explicit limit raises instead of silently
    returning an incomplete policy class as an exact one.
    """
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0 or max_trees < 1:
        raise ValueError("Invalid policy depth or enumeration limit")
    root = model.state(state)

    @lru_cache(maxsize=None)
    def visit(s, remaining, d, available):
        if not s or model.answer(s) is not None or d == 0:
            return (STOP,)
        result = [STOP]
        for aid in available:
            action = model.actions[aid]
            if not remaining.fits(action.cost):
                continue
            rest = tuple(a for a in available if a != aid)
            outcomes = model.outcomes[aid]
            subtrees = [visit(model.update(s, aid, o), remaining.spend(action.cost), d - 1, rest)
                        if model.reachable(root, aid, o) else (STOP,) for o in outcomes]
            if len(result) + prod(len(branch) for branch in subtrees) > max_trees:
                raise ValueError("Exact policy enumeration limit reached")
            for children in product(*subtrees):
                result.append(PolicyTree(aid, tuple(zip(outcomes, children))))
        return tuple(result)

    return visit(root, budget, depth, tuple(sorted(model.actions)))


@dataclass(frozen=True)
class PolicyEvaluation:
    closure: tuple[float, ...]
    residual: tuple[float, ...]
    predicted_cost: tuple[float, ...]


def evaluate_policy(model: AcquisitionModel, tree: PolicyTree,
                    state: Sequence[str] | None = None) -> PolicyEvaluation:
    """Take trajectory-level conjunction BEFORE expectation over trajectories.

    A leaf succeeds for h only if h survives and every retained answer equals
    y(h). False exclusion of h earns zero success even at a single-answer leaf.
    The same latent h is held fixed at all nodes of the policy tree.
    """
    root = model.state(state)
    if not root:
        raise ValueError("Cannot evaluate an empty task frontier")
    validate_structure(model, tree)

    @lru_cache(maxsize=None)
    def visit(h, node, s):
        if node.action is None:
            good = h in s and model.answer(s) == model.configurations[h].answer
            return float(good), model.pair_mass(s, root), 0.0
        aid = node.action
        result = np.array([0.0, 0.0, model.actions[aid].predicted_cost])
        for outcome, child in node.children:
            p = model.channels[aid][h][outcome]
            if p:
                result += p * np.asarray(visit(h, child, model.update(s, aid, outcome)))
        return tuple(float(v) for v in result)

    evaluated = [visit(h, tree, root) for h in root]
    return PolicyEvaluation(*(tuple(v[i] for v in evaluated) for i in range(3)))


def policy_matrices(model: AcquisitionModel, trees: Sequence[PolicyTree],
                    state: Sequence[str] | None = None) -> tuple[np.ndarray, ...]:
    records = [evaluate_policy(model, tree, state) for tree in trees]
    return tuple(np.asarray([getattr(r, key) for r in records], dtype=float)
                 for key in ("closure", "residual", "predicted_cost"))
