"""Committed acquisition, evidence updates and actual resource accounting."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from random import Random
from time import monotonic
from typing import Any, Callable, Protocol

from .frontier import JointFrontier
from .game import Plan
from .model import Action, AcquisitionModel, Budget, Cost, UNKNOWN
from .policies import validate_policy


@dataclass(frozen=True)
class AcquiredView:
    action_id: str
    pixels: Any
    root_box: tuple[int, int, int, int] | None = None
    root_size: tuple[int, int] | None = None
    cost: Cost = field(default_factory=lambda: Cost(views=1))


@dataclass(frozen=True)
class Verification:
    outcome: str
    cost: Cost = field(default_factory=Cost)
    reason: str = "verified"


class AcquisitionBackend(Protocol):
    def acquire(self, action: Action) -> AcquiredView: ...


class Verifier(Protocol):
    def verify(self, view: AcquiredView, action: Action) -> Verification: ...


class ExecutionFailure(RuntimeError):
    def __init__(self, message: str, cost: Cost | None = None, *, accounting_complete: bool = True):
        super().__init__(message)
        self.cost = cost or Cost()
        self.accounting_complete = accounting_complete


@dataclass(frozen=True)
class Episode:
    status: str
    answer: str | None
    retained: tuple[str, ...]
    trace: tuple[dict, ...]
    total_cost: Cost
    roles: tuple[dict, ...]
    views: tuple[AcquiredView, ...]
    plan: Plan
    accounting_complete: bool = True

    def summary(self) -> dict:
        return {"status": self.status, "answer": self.answer,
                "retained": list(self.retained), "trace": list(self.trace),
                "cost": asdict(self.total_cost), "roles": list(self.roles),
                "accounting_complete": self.accounting_complete,
                "plan": self.plan.summary()}


def run_episode(model: AcquisitionModel, backend: AcquisitionBackend, verifier: Verifier,
                 planner: Callable[[AcquisitionModel, Budget], Plan], budget: Budget,
                 *, seed: int = 11, depth: int = 4) -> Episode:
    """Execute one sampled tree through its outcome branches.

    Neither planner nor verifier receives the backend's latent world ID.
    The output contains raw acquired views for a separate downstream answerer.
    A model-revision workflow should start a new plan explicitly, retaining
    already accepted constraints and charging prior resources.
    """
    start = monotonic()
    spent, roles, trace, views = Cost(), [], [], []
    accounting_complete = True
    frontier = JointFrontier(model)
    def charge(role, cost, elapsed, complete=True):
        nonlocal spent, accounting_complete
        cost = replace(cost, seconds=max(cost.seconds, elapsed))
        spent = spent + cost
        accounting_complete = accounting_complete and complete
        roles.append({"role": role, **asdict(cost), "accounting_complete": complete})
    def available():
        current = replace(spent, seconds=max(spent.seconds, monotonic() - start))
        return budget.fits(current)
    def remaining():
        return budget.spend(replace(spent, seconds=max(spent.seconds, monotonic() - start)))
    def resource_stop():
        if spent.views > budget.views: return "unresolved_views"
        if spent.output_tokens > budget.output_tokens: return "unresolved_tokens"
        if spent.calls > budget.calls: return "unresolved_calls"
        return "unresolved_time"
    t = monotonic()
    plan = planner(model, budget)
    charge("planning", Cost(), monotonic() - t)
    tree = plan.sample(Random(seed))
    # Every branch is checked against the advertised hard resource cap.
    validate_policy(model, tree, budget, depth)
    status = "unresolved_no_test"
    used = set()
    while True:
        if not available():
            status = resource_stop()
            break
        if not frontier.state:
            status = "model_conflict"
            break
        if model.answer(frontier.state) is not None:
            status = "determined"
            break
        if tree.action is None:
            status = "unresolved_views" if spent.views >= budget.views else "unresolved_no_test"
            break
        aid = tree.action
        if aid in used:
            raise ValueError("A committed policy cannot reread identical pixels")
        action = model.actions[aid]
        if not remaining().fits(action.cost):
            status = ("unresolved_views" if action.cost.views > remaining().views else
                      "unresolved_tokens" if action.cost.output_tokens > remaining().output_tokens else
                      "unresolved_calls" if action.cost.calls > remaining().calls else "unresolved_time")
            break
        t = monotonic()
        try:
            view = backend.acquire(action)
        except ExecutionFailure as error:
            charge("acquisition_failure", error.cost, monotonic() - t, error.accounting_complete)
            status = "tool_failure"
            break
        if view.action_id != aid or view.cost.views < 1:
            raise ValueError("Backend returned a view for the wrong action or failed to meter it")
        charge("acquisition", view.cost, monotonic() - t)
        views.append(view)
        if not available():
            status = resource_stop()
            break
        t = monotonic()
        try:
            verified = verifier.verify(view, action)
        except ExecutionFailure as error:
            charge("verification_failure", error.cost, monotonic() - t, error.accounting_complete)
            status = "tool_failure"
            break
        charge("verification", verified.cost, monotonic() - t)
        if not available():
            status = resource_stop()
            break
        outcome = verified.outcome
        if outcome not in model.outcomes[aid]:
            outcome = UNKNOWN
        before = frontier.state
        frontier.observe(aid, outcome)
        trace.append({"action": aid, "outcome": outcome, "reason": verified.reason,
                      "retained": list(frontier.state),
                      "excluded": sorted(set(before) - set(frontier.state)),
                      "state_version": frontier.version})
        used.add(aid)
        tree = dict(tree.children)[outcome]
    total = replace(spent, seconds=max(spent.seconds, monotonic() - start))
    return Episode(status, model.answer(frontier.state) if status == "determined" else None,
                   frontier.state, tuple(trace), total, tuple(roles), tuple(views), plan,
                   accounting_complete)
