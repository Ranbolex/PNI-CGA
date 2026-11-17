"""CGA: full-frontier leaf validation and counterexample-guided policy expansion."""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from math import isfinite
from time import monotonic
from typing import Sequence

from .frontier import JointFrontier
from .game import Plan, mark_exact, solve_mixture
from .model import AcquisitionModel, Budget
from .policies import STOP, PolicyTree, enumerate_policies


@dataclass(frozen=True)
class PlannerSettings:
    depth: int = 4
    rounds: int = 16
    max_nodes: int = 20000
    beam: int = 32
    seconds: float = 60.0

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in
               (self.depth, self.rounds, self.max_nodes, self.beam)):
            raise ValueError("Planner limits must be positive integers")
        if not isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError("Invalid planner time limit")


@dataclass(frozen=True)
class BestResponse:
    tree: PolicyTree
    value: float
    complete: bool
    expanded_nodes: int


def best_response(model: AcquisitionModel, budget: Budget, depth: int,
                  weights: Sequence[float], *, state: Sequence[str] | None = None,
                  max_nodes: int = 20000, beam: int = 32,
                  cost_coefficient: float = 0.0,
                  cost_weights: Sequence[float] | None = None,
                  deadline: float = float("inf")) -> BestResponse:
    """Find one common contingent policy against fixed-world dual weights.

    Counterexamples rank candidate reads by their ability to distinguish an
    unresolved opposite-answer pair. Recursion scores the complete remaining
    tree, so individually uninformative but complementary reads remain useful.
    Pruned/limited search returns a feasible policy with complete=False.
    Weights are nonnegative, not necessarily normalized. An optional separate
    expected-cost term supports pricing of the cost tie-breaking LP.
    """
    root = model.state(state)
    if len(weights) != len(root) or any(not isfinite(w) or w < 0 for w in weights):
        raise ValueError("Invalid fixed-world weights")
    cost_weights = tuple(weights if cost_weights is None else cost_weights)
    if (len(cost_weights) != len(root) or any(not isfinite(w) or w < 0 for w in cost_weights)
            or not isfinite(cost_coefficient) or cost_coefficient < 0):
        raise ValueError("Invalid expected-cost objective")
    expanded, complete = 0, True

    @lru_cache(maxsize=None)
    def search(s, remaining, d, available, world_weights, cost_mass):
        nonlocal expanded, complete
        answer = model.answer(s)
        if answer is not None:
            value = sum(w for h, w in zip(root, world_weights)
                        if h in s and model.configurations[h].answer == answer)
            return value, 0.0, STOP
        if not s or not d or not available or sum(world_weights) == 0:
            return 0.0, 0.0, STOP
        if expanded >= max_nodes or monotonic() >= deadline:
            complete = False
            return 0.0, 0.0, STOP
        expanded += 1
        cex = JointFrontier(model, s).counterexample(dict(zip(root, world_weights)))
        feasible = [a for a in available if remaining.fits(model.actions[a].cost)]
        def priority(a):
            if cex is None:
                return 0.0, a
            h, g = cex.first, cex.second
            separation = sum(model.channels[a][h][o] *
                             (g not in model.compatibility[a][o])
                             for o in model.outcomes[a])
            separation += sum(model.channels[a][g][o] *
                              (h not in model.compatibility[a][o])
                              for o in model.outcomes[a])
            return -separation, a
        feasible.sort(key=priority)
        if len(feasible) > beam:
            complete = False
            feasible = feasible[:beam]
        best = (0.0, 0.0, STOP)
        for aid in feasible:
            if monotonic() >= deadline or expanded >= max_nodes:
                complete = False
                break
            action = model.actions[aid]
            children, utility = [], 0.0
            cost = sum(cost_mass) * action.predicted_cost
            rest = tuple(a for a in available if a != aid)
            for o in model.outcomes[aid]:
                branch_weights = tuple(w * model.channels[aid][h][o]
                                       for h, w in zip(root, world_weights))
                branch_cost_mass = tuple(w * model.channels[aid][h][o]
                                         for h, w in zip(root, cost_mass))
                if sum(branch_weights):
                    u, c, child = search(model.update(s, aid, o), remaining.spend(action.cost),
                                         d - 1, rest, branch_weights, branch_cost_mass)
                    utility += u
                    cost += c
                else:
                    child = STOP
                children.append((o, child))
            score = utility - cost_coefficient * cost
            best_score = best[0] - cost_coefficient * best[1]
            if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and cost < best[1]):
                best = utility, cost, PolicyTree(aid, tuple(children))
        return best

    value, cost, tree = search(root, budget, depth, tuple(sorted(model.actions)),
                               tuple(weights), cost_weights)
    return BestResponse(tree, float(value - cost_coefficient * cost), complete, expanded)


def solve_exact(model: AcquisitionModel, budget: Budget, depth: int,
                 objective: str = "closure", max_trees: int = 100000) -> Plan:
    trees = enumerate_policies(model, budget, depth, max_trees=max_trees)
    return mark_exact(solve_mixture(model, trees, objective=objective))


class CGAPlanner:
    def __init__(self, settings: PlannerSettings | None = None):
        self.settings = settings or PlannerSettings()

    def plan(self, model: AcquisitionModel, budget: Budget,
             state: Sequence[str] | None = None) -> Plan:
        s = model.state(state)
        if not s:
            raise ValueError("An empty frontier is a model conflict")
        settings = self.settings
        deadline = monotonic() + min(settings.seconds, budget.seconds)
        trees = [STOP]
        seen = {STOP}
        cex_pairs: set[tuple[str, str]] = set()
        expanded = 0
        upper = 1.0
        oracle_complete = False
        tie_breaks_complete = False
        for round_id in range(1, settings.rounds + 1):
            plan = solve_mixture(model, trees, s)
            hard = dict(zip(s, plan.dual_weights))
            # Queries always use full retained configurations, including anonymous ones.
            for tree, weight in zip(plan.policies, plan.probabilities):
                if weight < 1e-8:
                    continue
                def inspect(node, ids):
                    if node.action is None:
                        cex = JointFrontier(model, ids).counterexample(hard)
                        if cex:
                            cex_pairs.add((cex.first, cex.second))
                        return
                    for outcome, child in node.children:
                        if model.reachable(ids, node.action, outcome):
                            inspect(child, model.update(ids, node.action, outcome))
                inspect(tree, s)
            response = best_response(model, budget, settings.depth, plan.dual_weights,
                                     state=s, max_nodes=settings.max_nodes, beam=settings.beam,
                                     deadline=deadline)
            expanded += response.expanded_nodes
            # Restricted-game values alone are not a full-model upper bound.
            if response.complete:
                upper = min(upper, max(plan.lower_bound, response.value))
                oracle_complete = True
            primary_closed = response.complete and response.value <= plan.lower_bound + 1e-8
            candidate = response.tree if response.tree not in seen else None
            if primary_closed or candidate is None:
                # Primary duals can put all weight on an impossible world.
                # Price the mean on the primary-optimal face, then the cost
                # on the primary-and-mean-optimal face; never sacrifice the
                # worst-world objective to obtain either improvement.
                stages_complete = primary_closed
                candidate = None
                for pricing in (plan.mean_pricing, plan.cost_pricing):
                    if pricing is None or monotonic() >= deadline:
                        stages_complete = False
                        break
                    extra = best_response(model, budget, settings.depth, pricing.weights,
                                          state=s, max_nodes=settings.max_nodes,
                                          beam=settings.beam, deadline=deadline,
                                          cost_coefficient=pricing.cost_coefficient,
                                          cost_weights=pricing.cost_weights or None)
                    expanded += extra.expanded_nodes
                    if extra.value > pricing.threshold + 1e-8 and extra.tree not in seen:
                        candidate = extra.tree
                        stages_complete = False
                        break
                    if not extra.complete:
                        stages_complete = False
                        break
                tie_breaks_complete = stages_complete
            if candidate is None:
                break
            seen.add(candidate)
            trees.append(candidate)
            if monotonic() >= deadline:
                break
        final = solve_mixture(model, trees, s)
        return replace(final, upper_bound=max(final.lower_bound, upper),
                       oracle_complete=oracle_complete, rounds=round_id,
                       expanded_nodes=expanded, counterexamples=tuple(sorted(cex_pairs)),
                       tie_breaks_complete=tie_breaks_complete)
