"""Mixed-policy LPs for complete closure and matched residual-edge controls."""
from __future__ import annotations

from dataclasses import dataclass, replace
from random import Random
from typing import Sequence

import numpy as np
from scipy.optimize import linprog

from .model import AcquisitionModel
from .policies import PolicyTree, policy_matrices

TOL = 1e-9


@dataclass(frozen=True)
class Pricing:
    """Reduced-cost oracle for a lexicographic stage of the policy game."""
    weights: tuple[float, ...]
    threshold: float
    cost_coefficient: float = 0.0
    cost_weights: tuple[float, ...] = ()


@dataclass(frozen=True)
class Plan:
    policies: tuple[PolicyTree, ...]
    probabilities: tuple[float, ...]
    per_configuration: tuple[float, ...]
    mean_closure: float
    worst_residual: float
    mean_predicted_cost: float
    lower_bound: float
    upper_bound: float
    dual_weights: tuple[float, ...]
    oracle_complete: bool = False
    rounds: int = 0
    expanded_nodes: int = 0
    counterexamples: tuple[tuple[str, str], ...] = ()
    tie_breaks_complete: bool = False
    mean_pricing: Pricing | None = None
    cost_pricing: Pricing | None = None
    objective: str = "closure"

    def sample(self, rng: Random) -> PolicyTree:
        return rng.choices(self.policies, weights=self.probabilities, k=1)[0]

    def summary(self) -> dict:
        return {"optimization_objective": self.objective,
                "worst_configuration_closure": self.lower_bound,
                "mean_closure": self.mean_closure,
                "worst_residual": self.worst_residual,
                "mean_predicted_cost": self.mean_predicted_cost,
                "lower_bound": self.lower_bound, "upper_bound": self.upper_bound,
                "certified_gap": max(0.0, self.upper_bound - self.lower_bound),
                "oracle_complete": self.oracle_complete, "rounds": self.rounds,
                "tie_breaks_complete": self.tie_breaks_complete,
                "expanded_nodes": self.expanded_nodes,
                "per_configuration_closure": list(self.per_configuration),
                "counterexamples": [list(p) for p in self.counterexamples],
                "mixture": [{"weight": w, "policy": t.to_dict()}
                            for t, w in zip(self.policies, self.probabilities) if w > 1e-8]}


def _lp(c, a=None, b=None, bounds=None):
    n = len(c)
    result = linprog(c, A_ub=a, b_ub=b, A_eq=np.ones((1, n)), b_eq=[1.0],
                     bounds=bounds or [(0, None)] * n, method="highs")
    if not result.success:
        raise RuntimeError("Policy mixture LP failed: " + result.message)
    return result


def solve_mixture(model: AcquisitionModel, trees: Sequence[PolicyTree],
                  state: Sequence[str] | None = None, objective: str = "closure") -> Plan:
    """Lexicographic optimization over one shared policy class.

    Closure: worst-world success, answer-balanced mean success, then cost.
    Edge: worst-world expected residual mass, best worst-world closure among
    edge-optimal mixtures, mean closure, then cost. No penalty is added to
    either primary objective.
    """
    if objective not in ("closure", "edge") or not trees:
        raise ValueError("Require a nonempty policy class and a known objective")
    trees = tuple(dict.fromkeys(trees))
    root = model.state(state)
    j, residual, costs = policy_matrices(model, trees, root)
    n, worlds = j.shape
    mean = j @ np.asarray(model.weights(root))
    cost = costs @ np.asarray(model.weights(root))
    a, b = [], []
    if objective == "edge":
        edge = linprog(np.r_[np.zeros(n), 1.0],
                       A_ub=np.c_[residual.T, -np.ones(worlds)], b_ub=np.zeros(worlds),
                       A_eq=[np.r_[np.ones(n), 0.0]], b_eq=[1.0],
                       bounds=[(0, None)] * n + [(0, 1)], method="highs")
        if not edge.success:
            raise RuntimeError("Residual-edge LP failed: " + edge.message)
        a.extend(residual.T)
        b.extend([edge.x[-1] + 1e-10] * worlds)
    primary_a = np.vstack([np.c_[-j.T, np.ones(worlds)],
                            np.c_[np.asarray(a), np.zeros(len(a))]]) if a else np.c_[-j.T, np.ones(worlds)]
    primary_b = np.r_[np.zeros(worlds), b]
    primary = linprog(np.r_[np.zeros(n), -1.0], A_ub=primary_a, b_ub=primary_b,
                      A_eq=[np.r_[np.ones(n), 0.0]], b_eq=[1.0],
                      bounds=[(0, None)] * n + [(0, 1)], method="highs")
    if not primary.success:
        raise RuntimeError("Closure LP failed: " + primary.message)
    gamma = float(primary.x[-1])
    a.extend(-j.T)
    b.extend([-gamma + 1e-10] * worlds)
    mean_solution = _lp(-mean, np.asarray(a), np.asarray(b))
    mean_value = float(mean @ mean_solution.x)
    a.append(-mean)
    b.append(-mean_value + 1e-10)
    final = _lp(cost, np.asarray(a), np.asarray(b))
    x = np.maximum(0.0, final.x)
    x /= x.sum()
    eta = np.maximum(0.0, -primary.ineqlin.marginals[:worlds])
    if eta.sum():
        eta /= eta.sum()
    else:
        eta[:] = 1.0 / worlds
    success = x @ j
    mean_pricing = cost_pricing = None
    if objective == "closure":
        # A missing policy improves a minimization LP iff its reduced cost is
        # negative. Convert each stage's dual to weighted closure minus cost.
        mean_q = np.asarray(model.weights(root)) - mean_solution.ineqlin.marginals
        mean_scale = max(1.0, float(sum(mean_q)))
        mean_pricing = Pricing(tuple(float(v / mean_scale) for v in mean_q),
                               float(-mean_solution.eqlin.marginals[0] / mean_scale))
        cost_q = (-final.ineqlin.marginals[:worlds]
                  - final.ineqlin.marginals[-1] * np.asarray(model.weights(root)))
        cost_scale = max(1.0, float(sum(cost_q)))
        cost_pricing = Pricing(tuple(float(v / cost_scale) for v in cost_q),
                               float(-final.eqlin.marginals[0] / cost_scale),
                               1.0 / cost_scale, model.weights(root))
    return Plan(trees, tuple(float(v) for v in x), tuple(float(v) for v in success),
                float(x @ mean), float(max(x @ residual)), float(x @ cost),
                float(min(success)), 1.0, tuple(float(v) for v in eta),
                mean_pricing=mean_pricing, cost_pricing=cost_pricing, objective=objective)


def mark_exact(plan: Plan) -> Plan:
    # An exact edge-optimal solution need not maximize closure over all trees.
    return replace(plan, upper_bound=plan.lower_bound if plan.objective == "closure" else 1.0,
                   oracle_complete=True,
                   tie_breaks_complete=True)
