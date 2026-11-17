"""Actual image I/O, verification rejection, runtime metering and CLI behavior."""
from dataclasses import replace
import json

from PIL import Image
import pytest

from pni_cga.cli import main
from pni_cga.examples import deterministic_model, six_configuration_model
from pni_cga.execution import AcquiredView, ExecutionFailure, Verification, run_episode
from pni_cga.model import Action, Budget, Configuration, Cost, UNKNOWN
from pni_cga.planning import CGAPlanner, PlannerSettings
from pni_cga.vision import BinaryTileVerifier, JSONVerifier, LocalImageBackend, gate_verification


def single_read_model():
    actions = [Action("read", primitive="the marked tile is white", anchor_ids=("tile",), box=(0, 0, 32, 32))]
    hs = [Configuration("dark", "0"), Configuration("light", "1")]
    return deterministic_model(hs, actions, {"read": {"dark": "0", "light": "1"}})


@pytest.mark.parametrize("color,answer", [("black", "0"), ("white", "1")])
def test_actual_png_crops_determine_answer_without_a_latent_world_input(tmp_path, color, answer):
    path = tmp_path / "scene.png"
    Image.new("RGB", (32, 32), color).save(path)
    model = single_read_model()
    episode = run_episode(model, LocalImageBackend(path), BinaryTileVerifier(),
                          CGAPlanner(PlannerSettings(depth=1)).plan, Budget(views=1), depth=1)
    assert episode.status == "determined" and episode.answer == answer
    assert len(episode.views) == 1
    assert episode.total_cost.views == 1
    assert episode.total_cost.calls == 0
    assert sum(r["views"] for r in episode.roles) == episode.total_cost.views


def test_unreadable_pixels_preserve_alternatives(tmp_path):
    path = tmp_path / "ambiguous.png"
    Image.new("RGB", (32, 32), (128, 128, 128)).save(path)
    model = single_read_model()
    episode = run_episode(model, LocalImageBackend(path), BinaryTileVerifier(),
                          CGAPlanner(PlannerSettings(depth=1)).plan, Budget(views=1), depth=1)
    assert episode.status != "determined" and episode.answer is None
    assert episode.retained == model.initial_state
    assert episode.trace[0]["outcome"] == UNKNOWN
    assert episode.trace[0]["excluded"] == []


def payload():
    return {"primitive_id": "read", "entity_anchor_ids": ["tile"], "value": "no",
            "legibility_ordinal": 3, "binding_ordinal": 3, "scope_complete": True,
            "witness_regions": [[0, 0, 32, 32]]}


@pytest.mark.parametrize("change", [
    {"primitive_id": "other"}, {"entity_anchor_ids": []}, {"legibility_ordinal": 1},
    {"binding_ordinal": True}, {"scope_complete": False}, {"witness_regions": []},
    {"witness_regions": [[0, 0, 100, 100]]}, {"value": "made-up"},
])
def test_verifier_rejects_unsupported_no_and_unbound_or_unreadable_evidence(change):
    action = replace(single_read_model().actions["read"], requires_scope=True)
    view = AcquiredView("read", Image.new("RGB", (32, 32)), (0, 0, 32, 32), (32, 32))
    data = payload()
    data.update(change)
    assert gate_verification(data, action, view) == UNKNOWN


def test_json_verifier_accepts_readable_bound_negative_literal_and_meters_cost():
    action = replace(single_read_model().actions["read"], requires_scope=True)
    view = AcquiredView("read", Image.new("RGB", (32, 32)), (0, 0, 32, 32), (32, 32))
    verifier = JSONVerifier(lambda crop, primitive: (payload(), Cost(input_tokens=10, output_tokens=4, calls=1)))
    result = verifier.verify(view, action)
    assert result.outcome == "NO" and result.cost.output_tokens == 4


def test_budget_overrun_preserves_metering_and_stops_before_exclusion(tmp_path):
    path = tmp_path / "scene.png"
    Image.new("RGB", (32, 32), "white").save(path)
    class ExpensiveVerifier:
        def verify(self, view, action):
            return Verification("1", Cost(input_tokens=10, output_tokens=6, calls=1))
    model = single_read_model()
    episode = run_episode(model, LocalImageBackend(path), ExpensiveVerifier(),
                          CGAPlanner(PlannerSettings(depth=1)).plan,
                          Budget(views=1, output_tokens=5), depth=1)
    assert episode.status == "unresolved_tokens"
    assert episode.total_cost.output_tokens == 6
    assert episode.total_cost.input_tokens == 10
    assert episode.retained == model.initial_state


def test_failed_calls_are_charged_and_no_episode_is_dropped(tmp_path):
    class FailedBackend:
        def acquire(self, action):
            raise ExecutionFailure("test transport failure", Cost(calls=1, output_tokens=2))
    model = single_read_model()
    episode = run_episode(model, FailedBackend(), BinaryTileVerifier(),
                          CGAPlanner(PlannerSettings(depth=1)).plan, Budget(), depth=1)
    assert episode.status == "tool_failure"
    assert episode.total_cost.calls == 1 and episode.total_cost.output_tokens == 2
    assert episode.retained == model.initial_state


def test_local_crop_clips_root_coordinates_and_preserves_transform(tmp_path):
    path = tmp_path / "scene.png"
    Image.new("RGB", (40, 30), "white").save(path)
    action = Action("tile", box=(-4, -3, 60, 50))
    view = LocalImageBackend(path, max_edge=20).acquire(action)
    assert view.root_box == (0, 0, 40, 30)
    assert view.root_size == (40, 30) and view.pixels.size == (20, 15)


def test_offline_cli_emits_checkable_outputs(tmp_path, capsys):
    path = tmp_path / "certificate.json"
    main(["verify-certificate", "--output", str(path)])
    assert json.loads(path.read_text())["all_trees_checked"] == 101
    main(["image-demo", "--output-dir", str(tmp_path / "image")])
    summary = json.loads((tmp_path / "image/episode.json").read_text())
    assert summary["cost"]["views"] <= 2
    assert (tmp_path / "image/acquired-1.png").exists()
    assert capsys.readouterr().out
