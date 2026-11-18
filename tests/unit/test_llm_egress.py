"""Gemini egress: transparent geo-block fallback through the Modal fleet.

The sandbox/worker region can get 400 FAILED_PRECONDITION ("User location is
not supported") from the Generative Language API. `GeminiRestBackend` retries
the identical payload through the deployed `directoros-vlm.vlm_call` proxy
(US/EU egress) and the decision sticks for the backend's lifetime — the same
contract as `perception.vision.GeminiVisionClient`. CI stays hermetic: the
`modal` import is lazy behind `_modal_vlm_fn`, which tests monkeypatch.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import httpx
import pytest
from director_os.agent_runtime import llm as llm_mod
from director_os.agent_runtime.llm import GeminiRestBackend, LLMBackendError
from director_os.agent_runtime.registry import ToolRegistry
from director_os.core.config import get_settings
from pydantic import BaseModel

GEO_400 = httpx.Response(
    400,
    text=json.dumps(
        {
            "error": {
                "code": 400,
                "message": "User location is not supported for the API use.",
                "status": "FAILED_PRECONDITION",
            }
        }
    ),
)


class _NoopInput(BaseModel):
    x: int = 1


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register("noop", "Does nothing (egress tests).", _NoopInput, lambda **kw: {"ok": True})
    return reg


def _gemini_ok_response() -> dict[str, Any]:
    return {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"functionCall": {"name": "noop", "args": {"x": 1}}}],
                }
            }
        ],
        "usageMetadata": {"promptTokenCount": 42, "candidatesTokenCount": 7},
    }


class _FakeModalFn:
    """Stands in for modal.Function.from_name('directoros-vlm', 'vlm_call')."""

    def __init__(self, responses: list[dict[str, Any]]):
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def remote(self, model: str, payload_json: str) -> dict[str, Any]:
        self.calls.append({"model": model, "payload": json.loads(payload_json)})
        return self._responses.pop(0)


@pytest.fixture()
def modal_creds(monkeypatch):
    settings = dataclasses.replace(
        get_settings(), modal_token_id="ak-test", modal_token_secret="as-test"
    )
    monkeypatch.setattr("director_os.core.config.get_settings", lambda: settings)


def test_geo_block_falls_back_to_modal_and_sticks(modal_creds, monkeypatch):
    direct_calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        direct_calls["n"] += 1
        return GEO_400

    fake = _FakeModalFn(
        [{"status": 200, "body": json.dumps(_gemini_ok_response())} for _ in range(2)]
    )
    monkeypatch.setattr(llm_mod, "_modal_vlm_fn", lambda: fake)
    backend = GeminiRestBackend(
        "k", "gemini-3.8-flash", _registry(), transport=httpx.MockTransport(handler)
    )

    steps = backend.propose("state", "director")
    assert len(steps) == 1 and steps[0].tool == "noop"
    assert backend._egress_active == "modal"
    assert direct_calls["n"] == 1  # geo-blocked once, then switched
    assert fake.calls[0]["model"] == "gemini-3.8-flash"

    # second propose goes straight through the fleet — direct is never retried
    steps2 = backend.propose("state 2", "director")
    assert len(steps2) == 1
    assert direct_calls["n"] == 1
    assert len(fake.calls) == 2


def test_geo_block_without_modal_creds_raises(modal_creds, monkeypatch):
    settings = dataclasses.replace(get_settings(), modal_token_id="", modal_token_secret="")
    monkeypatch.setattr("director_os.core.config.get_settings", lambda: settings)

    def handler(request: httpx.Request) -> httpx.Response:
        return GEO_400

    backend = GeminiRestBackend(
        "k", "gemini-3.8-flash", _registry(), transport=httpx.MockTransport(handler)
    )
    with pytest.raises(LLMBackendError, match="MODAL_TOKEN_ID"):
        backend.propose("state", "director")


def test_transient_fleet_5xx_retries(modal_creds, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)  # backoff off in tests

    def handler(request: httpx.Request) -> httpx.Response:
        return GEO_400

    fake = _FakeModalFn(
        [
            {"status": 503, "body": "overloaded"},
            {"status": 429, "body": "rate limit"},
            {"status": 200, "body": json.dumps(_gemini_ok_response())},
        ]
    )
    monkeypatch.setattr(llm_mod, "_modal_vlm_fn", lambda: fake)
    backend = GeminiRestBackend(
        "k", "gemini-3.8-flash", _registry(), transport=httpx.MockTransport(handler)
    )
    steps = backend.propose("state", "director")
    assert len(steps) == 1
    assert len(fake.calls) == 3


def test_hard_fleet_error_raises(modal_creds, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return GEO_400

    fake = _FakeModalFn(
        [{"status": 400, "body": json.dumps({"error": {"message": "bad request"}})}]
    )
    monkeypatch.setattr(llm_mod, "_modal_vlm_fn", lambda: fake)
    backend = GeminiRestBackend(
        "k", "gemini-3.8-flash", _registry(), transport=httpx.MockTransport(handler)
    )
    with pytest.raises(LLMBackendError, match="via modal egress 400"):
        backend.propose("state", "director")


def test_direct_quota_429_downgrades_without_fleet(modal_creds, monkeypatch):
    """The exact e2e failure: direct 429 quota-exceeded walks the chain direct."""
    seen_models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = request.url.path.split("/models/")[1].split(":")[0]
        seen_models.append(model)
        if model in ("gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash"):
            return httpx.Response(
                429,
                text=json.dumps({"error": {"message": "You exceeded your current quota"}}),
            )
        return httpx.Response(200, text=json.dumps(_gemini_ok_response()))

    fake = _FakeModalFn([])
    monkeypatch.setattr(llm_mod, "_modal_vlm_fn", lambda: fake)
    backend = GeminiRestBackend(
        "k",
        "gemini-3.8-flash",
        _registry(),
        transport=httpx.MockTransport(handler),
        fallback_models=["gemini-3.7-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"],
    )
    steps = backend.propose("state", "director")
    assert len(steps) == 1
    assert backend.model == "gemini-3.5-flash-lite"  # sticky downgrade
    assert seen_models == [
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
    ]
    assert fake.calls == []  # fleet never needed — geo was fine, quota was not


def test_capacity_503_downgrades_through_fleet_stickily(modal_creds, monkeypatch):
    """Geo-switched egress + fleet-side capacity: chain walks via the fleet."""
    monkeypatch.setattr("time.sleep", lambda _s: None)

    def handler(request: httpx.Request) -> httpx.Response:
        return GEO_400

    fake = _FakeModalFn(
        [{"status": 503, "body": "high demand"} for _ in range(6)]
        + [{"status": 200, "body": json.dumps(_gemini_ok_response())}]
    )
    monkeypatch.setattr(llm_mod, "_modal_vlm_fn", lambda: fake)
    backend = GeminiRestBackend(
        "k",
        "gemini-3.8-flash",
        _registry(),
        transport=httpx.MockTransport(handler),
        fallback_models=["gemini-3.7-flash", "gemini-3.5-flash"],
    )
    steps = backend.propose("state", "director")
    assert len(steps) == 1
    assert backend.model == "gemini-3.5-flash"  # 3rd candidate after fleet 503s
    assert backend._egress_active == "modal"
    # 3 transient fleet 503s on 3.8, then walk: 3.7 (x3), 3.5 -> 200 on 1st try
    assert [c["model"] for c in fake.calls[:6]] == ["gemini-3.8-flash"] * 3 + [
        "gemini-3.7-flash"
    ] * 3
    assert fake.calls[6]["model"] == "gemini-3.5-flash"


def test_all_models_hot_raises_capacity_exhausted(modal_creds, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)

    def handler(request: httpx.Request) -> httpx.Response:
        return GEO_400

    fake = _FakeModalFn([{"status": 503, "body": "hot"} for _ in range(12)])
    monkeypatch.setattr(llm_mod, "_modal_vlm_fn", lambda: fake)
    backend = GeminiRestBackend(
        "k",
        "gemini-3.8-flash",
        _registry(),
        transport=httpx.MockTransport(handler),
        fallback_models=["gemini-3.7-flash", "gemini-3.5-flash"],
    )
    with pytest.raises(LLMBackendError, match="capacity exhausted across 3 model"):
        backend.propose("state", "director")
    assert [c["model"] for c in fake.calls][-1] == "gemini-3.5-flash"


def test_direct_non_capacity_error_does_not_touch_fleet(modal_creds, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return (
            httpx.Response(500, text="boom")
            if request.url.path.endswith("3.8-flash:generateContent")
            else httpx.Response(500, text="boom")
        )

    fake = _FakeModalFn([])
    monkeypatch.setattr(llm_mod, "_modal_vlm_fn", lambda: fake)
    backend = GeminiRestBackend(
        "k",
        "gemini-3.8-flash",
        _registry(),
        transport=httpx.MockTransport(handler),
        fallback_models=["gemini-3.7-flash"],
    )
    with pytest.raises(LLMBackendError, match="capacity exhausted"):
        backend.propose("state", "director")
    assert fake.calls == []
    assert backend._egress_active == "direct"


def test_multi_turn_history_always_starts_with_user(modal_creds):
    """Regression (Eustis autodrive round-2 400): without remembering the user
    turn, the replayed contents START with the bare model functionCall turn —
    Gemini rejects 'function call turn comes immediately after a user turn'."""
    seen_role_shapes: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen_role_shapes.append([c["role"] for c in body["contents"]])
        return httpx.Response(200, text=json.dumps(_gemini_ok_response()))

    backend = GeminiRestBackend(
        "k",
        "gemini-3.8-flash",
        _registry(),
        transport=httpx.MockTransport(handler),
        fallback_models=[],
    )
    steps = backend.propose("state 1", "director")
    backend.record_results(steps, ["ok"])
    backend.propose("state 2", "director")

    assert seen_role_shapes[0] == ["user"]
    # round 2 replays: user -> model(functionCall) -> user(fr) -> user(state)
    assert seen_role_shapes[1][0] == "user"
    assert seen_role_shapes[1][1] == "model"
    assert seen_role_shapes[1][2] == "user"
    assert seen_role_shapes[1][-1] == "user"
