"""Provider boundary tests use in-memory replies and never access the network."""
from io import BytesIO
import json
from urllib.error import URLError

from PIL import Image
import pytest

from pni_cga.execution import AcquiredView, ExecutionFailure, run_episode
from pni_cga.examples import six_configuration_model
from pni_cga.model import Action, Budget, Cost
from pni_cga.planning import CGAPlanner, PlannerSettings
from pni_cga.vision import OpenAICompatibleClient


def client(**kwargs):
    return OpenAICompatibleClient(model="fixture-model", base_url="https://example.invalid/v1",
                                  api_key="test-only", **kwargs)


def view():
    return AcquiredView("tile", Image.new("RGB", (16, 16), "white"), (0, 0, 16, 16), (16, 16))


def test_image_client_sends_only_crop_and_records_provider_usage(monkeypatch):
    def reply(request, timeout):
        body = json.loads(request.data)
        assert body["model"] == "fixture-model" and timeout == 90
        content = body["messages"][1]["content"]
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
        assert "answer" not in json.loads(content[0]["text"])
        return BytesIO(json.dumps({"usage": {"prompt_tokens": 42, "completion_tokens": 17},
                                  "choices": [{"message": {"content": '{"value":"yes"}'}}]}).encode())
    monkeypatch.setattr("pni_cga.vision.urlopen", reply)
    payload, cost = client()(view(), Action("tile"))
    assert payload == {"value": "yes"}
    assert cost.input_tokens == 42 and cost.output_tokens == 17 and cost.calls == 1


@pytest.mark.parametrize("raw", [[], {}, {"usage": []}, {"usage": {"prompt_tokens": True, "completion_tokens": 0}}])
def test_missing_accounting_never_becomes_a_zero_token_call(monkeypatch, raw):
    monkeypatch.setattr("pni_cga.vision.urlopen", lambda *a, **k: BytesIO(json.dumps(raw).encode()))
    with pytest.raises(ExecutionFailure, match="accounting") as caught:
        client()(view(), Action("tile"))
    assert caught.value.cost.calls == 1 and not caught.value.accounting_complete


def test_malformed_content_preserves_known_usage(monkeypatch):
    raw = {"usage": {"prompt_tokens": 4, "completion_tokens": 2}, "choices": []}
    monkeypatch.setattr("pni_cga.vision.urlopen", lambda *a, **k: BytesIO(json.dumps(raw).encode()))
    payload, cost = client()(view(), Action("tile"))
    assert payload == {} and cost.output_tokens == 2


def test_transport_failure_does_not_echo_headers_or_server_response(monkeypatch):
    def fail(*args, **kwargs):
        raise URLError("sensitive-provider-message")
    monkeypatch.setattr("pni_cga.vision.urlopen", fail)
    with pytest.raises(ExecutionFailure) as caught:
        client()(view(), Action("tile"))
    assert str(caught.value) == "Model call failed"
    assert not caught.value.accounting_complete


@pytest.mark.parametrize("url", ["http://localhost.example.invalid/v1", "https://name:pass@example.invalid/v1",
                                 "https://example.invalid/v1?key=placeholder"])
def test_endpoint_validation_uses_parsed_host(url):
    with pytest.raises(ValueError, match="HTTPS"):
        OpenAICompatibleClient(model="fixture", base_url=url, api_key="test-only")


def test_episode_reports_partial_accounting_for_failed_provider_call():
    model = six_configuration_model()
    class Backend:
        def acquire(self, action):
            return AcquiredView(action.id, Image.new("RGB", (16, 16)), cost=Cost(views=1))
    class Verifier:
        def verify(self, view, action):
            raise ExecutionFailure("missing usage", Cost(calls=1), accounting_complete=False)
    result = run_episode(model, Backend(), Verifier(), CGAPlanner(PlannerSettings(depth=2)).plan,
                         Budget(views=2), depth=2)
    assert result.status == "tool_failure" and not result.accounting_complete
    assert result.total_cost.calls == 1 and result.retained == model.initial_state
