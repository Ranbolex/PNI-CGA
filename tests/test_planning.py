"""Behavioral tests of the current closure objective and matched controls."""
from dataclasses import replace
from fractions import Fraction
from importlib.resources import files
import json
from random import Random

import numpy as np
import pytest

from pni_cga.audit import exact_twin_audit, verify_separation_certificate
from pni_cga.examples import complementary_model, deterministic_model, six_configuration_model
from pni_cga.frontier import JointFrontier
from pni_cga.game import solve_mixture
from pni_cga.io import model_from_dict, model_to_dict
from pni_cga.model import Action, AcquisitionModel, Budget, Configuration, Cost, UNKNOWN
from pni_cga.planning import CGAPlanner, PlannerSettings, best_response, solve_exact
from pni_cga.policies import STOP, PolicyTree, enumerate_policies, evaluate_policy, validate_policy


def test_manuscript_separation_under_identical_policy_class():
    model, budget = six_configuration_model(), Budget(views=2)
    trees = enumerate_policies(model, budget, 2)
    cga = solve_mixture(model, trees)
    edge = solve_mixture(model, trees, objective="edge")
    assert cga.lower_bound == pytest.approx(1 / 2, abs=1e-8)
    assert cga.mean_closure == pytest.approx(7 / 12, abs=1e-8)
    assert edge.lower_bound == pytest.approx(1 / 3, abs=1e-8)
    assert edge.mean_closure == pytest.approx(11 / 18, abs=1e-8)
    assert edge.worst_residual == pytest.approx(2 / 27, abs=1e-8)
    assert cga.policies == edge.policies
    assert cga.mean_closure < edge.mean_closure  # Preserve the worst/mean tradeoff.


def test_independent_rational_certificate_checks_all_101_trees():
    data = json.loads(files("pni_cga").joinpath("fixtures/separation_certificate.json").read_text())
    result = verify_separation_certificate(data)
    assert result["all_passed"] and result["all_trees_checked"] == 101
    data["rational_certificate"]["cert_dual_world_weights"] = ["1"] * 6
    with pytest.raises(ValueError):
        verify_separation_certificate(data)


def test_exact_edge_control_does_not_report_its_closure_as_a_global_upper_bound():
    model, budget = six_configuration_model(), Budget(views=2)
    edge = solve_exact(model, budget, 2, objective="edge")
    closure = solve_exact(model, budget, 2)
    assert edge.objective == "edge" and edge.oracle_complete
    assert edge.lower_bound < closure.lower_bound <= edge.upper_bound


def test_counterexample_refinement_reaches_exact_objective():
    model, budget = six_configuration_model(), Budget(views=2)
    result = CGAPlanner(PlannerSettings(depth=2)).plan(model, budget)
    assert result.lower_bound == pytest.approx(.5, abs=1e-8)
    assert result.upper_bound == pytest.approx(.5, abs=1e-8)
    assert result.counterexamples and result.rounds > 1
    assert result.oracle_complete
    for tree in result.policies:
        validate_policy(model, tree, budget, 2)


def test_zero_worst_case_still_optimizes_mean_closure():
    hs = [Configuration("opaque", "0"), Configuration("left", "1"), Configuration("right", "1")]
    outcomes = {"opaque": UNKNOWN, "left": "YES", "right": "NO"}
    channels = {"read": {h.id: {o: float(outcomes[h.id] == o) for o in (UNKNOWN, "YES", "NO")}
                         for h in hs}}
    model = AcquisitionModel(hs, [Action("read")], channels,
                             {"read": {UNKNOWN: list(outcomes), "YES": ["left"], "NO": ["right"]}})
    plan = CGAPlanner(PlannerSettings(depth=1)).plan(model, Budget(views=1))
    assert plan.lower_bound == 0
    assert plan.mean_closure == pytest.approx(.5, abs=1e-8)
    assert plan.tie_breaks_complete
    assert any(t.action == "read" and p > .9 for t, p in zip(plan.policies, plan.probabilities))


def test_lexicographic_column_generation_matches_full_enumeration():
    rng = np.random.default_rng(31)
    for _ in range(8):
        hs = [Configuration(f"h{i}", str(i % 2)) for i in range(6)]
        actions = [Action(f"a{i}", predicted_cost=float(rng.uniform(.1, 3))) for i in range(4)]
        outcomes = {a.id: {h.id: str(int(rng.integers(2))) for h in hs} for a in actions}
        model = deterministic_model(hs, actions, outcomes)
        budget = Budget(views=2)
        exact = solve_exact(model, budget, 2)
        bounded = CGAPlanner(PlannerSettings(depth=2, rounds=64)).plan(model, budget)
        assert bounded.lower_bound == pytest.approx(exact.lower_bound, abs=1e-7)
        assert bounded.mean_closure == pytest.approx(exact.mean_closure, abs=1e-7)
        assert bounded.mean_predicted_cost == pytest.approx(exact.mean_predicted_cost, abs=1e-7)
        assert bounded.tie_breaks_complete


@pytest.mark.parametrize("unknown", [0, .2, .7, 1])
def test_complementary_reads_score_joint_success_before_expectation(unknown):
    model = complementary_model(unknown)
    plan = CGAPlanner(PlannerSettings(depth=2)).plan(model, Budget(views=2))
    assert plan.lower_bound == pytest.approx((1 - unknown) ** 2, abs=1e-8)
    one_view = solve_exact(model, Budget(views=1), 1)
    assert one_view.lower_bound == 0


def test_search_budget_exits_do_not_fabricate_a_global_certificate():
    model = six_configuration_model()
    plan = CGAPlanner(PlannerSettings(depth=2, max_nodes=1, rounds=1)).plan(model, Budget(views=2))
    assert not plan.oracle_complete and plan.upper_bound == 1
    assert plan.lower_bound <= .5 + 1e-8
    beam = CGAPlanner(PlannerSettings(depth=2, beam=1)).plan(model, Budget(views=2))
    assert not beam.oracle_complete and beam.upper_bound == 1


def test_empty_frontier_is_conflict_not_a_certificate():
    model = complementary_model()
    assert model.answer(()) is None
    with pytest.raises(ValueError, match="conflict"):
        CGAPlanner().plan(model, Budget(), state=())


def test_conservative_compression_keeps_joint_entities_and_anonymous_residuals():
    configs = [Configuration("same", "1", (("left_entity", "p"), ("right_entity", "p"))),
               Configuration("distinct", "2", (("left_entity", "p"), ("right_entity", "q"))),
               Configuration("unseen", "3", (("count_residual", "1"),), anonymous=True)]
    model = AcquisitionModel(configs, [], {}, {})
    frontier = JointFrontier(model)
    assert frontier.expand(frontier.compress(3)) == model.initial_state
    assert "unseen" in frontier.state
    with pytest.raises(ValueError, match="answer class"):
        frontier.compress(2)
    cex = frontier.counterexample()
    assert cex and cex.missing_predicates


def test_unknown_does_not_become_negative_evidence():
    model = complementary_model()
    assert model.update(model.initial_state, "left", UNKNOWN) == model.initial_state
    values = model_to_dict(model)
    values["compatibility"]["left"][UNKNOWN] = ["h00"]
    with pytest.raises(ValueError, match="UNKNOWN must retain"):
        model_from_dict(values)


def test_predictive_zero_does_not_authorize_exclusion():
    hs = [Configuration("a", "yes"), Configuration("b", "no")]
    action = Action("read")
    channels = {"read": {"a": {"YES": 1., "NO": 0., UNKNOWN: 0.},
                         "b": {"YES": 0., "NO": 1., UNKNOWN: 0.}}}
    compatibility = {"read": {o: ["a", "b"] for o in ("YES", "NO", UNKNOWN)}}
    model = AcquisitionModel(hs, [action], channels, compatibility)
    assert model.update(model.initial_state, "read", "YES") == model.initial_state
    assert solve_exact(model, Budget(views=1), 1).lower_bound == 0


def test_false_exclusion_of_truth_earns_zero_even_at_single_answer_leaf():
    hs = [Configuration("a", "yes"), Configuration("b", "no")]
    channels = {"read": {h.id: {"YES": 1., UNKNOWN: 0.} for h in hs}}
    model = AcquisitionModel(hs, [Action("read")], channels,
                             {"read": {"YES": ["a"], UNKNOWN: ["a", "b"]}})
    tree = PolicyTree("read", (("UNKNOWN", STOP), ("YES", STOP)))
    score = evaluate_policy(model, tree)
    assert score.closure == (1.0, 0.0)


@pytest.mark.parametrize("cost,budget", [
    (Cost(views=2), Budget(views=1)),
    (Cost(views=1, output_tokens=9), Budget(output_tokens=8)),
    (Cost(views=1, calls=2), Budget(calls=1)),
    (Cost(views=1, seconds=3), Budget(seconds=2)),
])
def test_policy_class_respects_every_resource_dimension(cost, budget):
    hs = [Configuration("a", "a"), Configuration("b", "b")]
    model = deterministic_model(hs, [Action("read", cost)],
                                {"read": {"a": "0", "b": "1"}})
    assert enumerate_policies(model, budget, 1) == (STOP,)


def test_committed_policy_is_not_replaced_by_world_specific_choices():
    model, budget = six_configuration_model(), Budget(views=2)
    result = solve_exact(model, budget, 2)
    sampled = result.sample(Random(11))
    scores = evaluate_policy(model, sampled).closure
    assert set(scores) <= {0., 1.}
    assert min(scores) == 0  # A single tree cannot guarantee every world.
    assert min(result.per_configuration) == pytest.approx(.5, abs=1e-8)


def test_best_response_matches_enumeration_for_multiple_dual_distributions():
    model, budget = six_configuration_model(), Budget(views=2)
    trees = enumerate_policies(model, budget, 2)
    matrix = np.array([evaluate_policy(model, t).closure for t in trees])
    rng = np.random.default_rng(13)
    for _ in range(8):
        eta = rng.dirichlet(np.ones(6))
        response = best_response(model, budget, 2, eta)
        assert response.complete
        assert response.value == pytest.approx(max(matrix @ eta), abs=1e-9)


def test_twin_audit_checks_full_trajectory_and_deterministic_assumptions():
    model = six_configuration_model()
    tree = PolicyTree("a0", tuple((o, STOP) for o in model.outcomes["a0"]))
    audit = exact_twin_audit(model, [(STOP, .4), (tree, .6)], "h0", "h2")
    assert audit["trajectory_tv"] == pytest.approx(.6)
    assert audit["coverage"] == pytest.approx(.6)
    assert audit["twin_accuracy_ceiling"] == pytest.approx(.8)
    with pytest.raises(ValueError, match="deterministic"):
        exact_twin_audit(complementary_model(.2), [(STOP, 1.)], "h00", "h01")


def test_model_serialization_round_trip_and_probability_validation():
    original = six_configuration_model()
    data = model_to_dict(original)
    restored = model_from_dict(data)
    assert restored.initial_state == original.initial_state
    assert dict(restored.channels) == dict(original.channels)
    data["channels"]["a0"]["h0"]["0"] = .3
    with pytest.raises(ValueError, match="sum to one"):
        model_from_dict(data)


def test_policy_enumeration_limit_is_explicit_and_never_labeled_exact():
    with pytest.raises(ValueError, match="enumeration limit"):
        solve_exact(six_configuration_model(), Budget(views=2), 2, max_trees=2)


def test_repeated_pixels_and_missing_branches_are_rejected():
    model = complementary_model()
    incomplete = PolicyTree("left", (("0", STOP),))
    with pytest.raises(ValueError, match="outcome"):
        validate_policy(model, incomplete, Budget(), 2)
    child = PolicyTree("left", tuple((o, STOP) for o in model.outcomes["left"]))
    repeated = PolicyTree("left", tuple((o, child) for o in model.outcomes["left"]))
    with pytest.raises(ValueError, match="repeated"):
        validate_policy(model, repeated, Budget(), 2)
