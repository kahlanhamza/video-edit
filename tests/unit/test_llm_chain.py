"""Multi-turn replay + Phase A complete_json on the live-routing chain.

The routing chain itself (Groq ladder, Retry-After, model-not-found, tier
failover, geo egress) is pinned by test_routing_chain.py — this file pins the
M5 story-map additions on top: OpenAI-compat multi-turn history with
role="tool" replay, and complete_json walking the same capacity ladder for
the Phase A macro plan.
"""

from __future__ import annotations

import json

import httpx
from director_os.agent_runtime.agents import ToolStep
from director_os.agent_runtime.llm import (
    FailoverBackend,
    GeminiRestBackend,
    LLMBackendError,
    OpenAICompatBackend,
    strip_code_fences,
)
from director_os.agent_runtime.registry import ToolRegistry


def _registry() -> ToolRegistry:
    from director_os.agent_runtime.editing_tools import _TrimInput

    reg = ToolRegistry()
    reg.register(
        "timeline.trim_clip",
        "Move a master clip's in-point and duration (source seconds).",
        _TrimInput,
        lambda **kw: {"ok": True},
    )
    return reg


def _transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def _tool_call_response(call_id: str = "call_1") -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                # wire names are dot-mangled upstream (openrouter law)
                                "name": "timeline__trim_clip",
                                "arguments": (
                                    '{"item_index": 0, "new_start_s": 1.0, "new_dur_s": 2.0}'
                                ),
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }


# ------------------------------------------------------------------ multi-turn
def test_openai_compat_multi_turn_replays_tool_results():
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode()))
        return httpx.Response(200, json=_tool_call_response())

    backend = OpenAICompatBackend(
        "k", "m1", _registry(), base_url="https://groq.test/v1", transport=_transport(handler)
    )
    steps = backend.propose("state", "director")
    assert steps and steps[0].tool == "timeline.trim_clip", "wire name un-mangled"
    assert steps[0].provider_meta and steps[0].provider_meta.get("call_id") == "call_1"

    backend.record_results(steps, ["trimmed clip 0 to start=1.00s dur=2.00s"])
    backend.propose("state turn 2", "director")

    assert len(captured) == 2
    second = captured[1]
    kinds = [(m.get("role"), "tool_calls" in m) for m in second["messages"]]
    assert ("tool", False) in kinds, "tool result must be replayed as role=tool"
    tool_msg = next(m for m in second["messages"] if m.get("role") == "tool")
    assert tool_msg["tool_call_id"] == "call_1"
    assert any(m.get("tool_calls") for m in second["messages"]), "assistant turn replayed verbatim"


def test_gemini_backend_has_complete_json():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode())
        assert "tools" not in payload  # Phase A has no tools attached
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": '{"acts": []}'}]}}],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3},
            },
        )

    gem = GeminiRestBackend(
        "AQ.key", "gemini-3.5-flash-lite", _registry(), transport=_transport(handler)
    )
    assert gem.complete_json("sys", "user") == '{"acts": []}'


# ------------------------------------------------------------------ complete_json
def test_complete_json_walks_dead_models(monkeypatch):
    monkeypatch.setattr("director_os.agent_runtime.llm.time.sleep", lambda s: None)
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode())
        calls.append(payload["model"])
        if payload["model"] == "dead/model":
            return httpx.Response(404, text="model_not_found")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": '{"acts": []}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    backend = OpenAICompatBackend(
        "k",
        "dead/model",
        _registry(),
        base_url="https://groq.test/v1",
        transport=_transport(handler),
        fallback_models=["live/model"],
    )
    assert backend.complete_json("sys", "user") == '{"acts": []}'
    assert calls == ["dead/model", "live/model"], "dead model name skips to the next candidate"


def test_failover_complete_json_walks_tiers():
    class DeadTier:
        provider = "openai_compat"
        model = "groq/dead"

        def propose(self, state, role):
            raise LLMBackendError("429 rate limit exceeded")

        def complete_json(self, system, user, max_tokens=4096):
            raise LLMBackendError("429 capacity exhausted")

    class LiveTier:
        provider = "gemini"
        model = "gemini-3.5-flash-lite"

        def complete_json(self, system, user, max_tokens=4096):
            return '{"acts": []}'

    chain = FailoverBackend([DeadTier(), LiveTier()])
    assert chain.complete_json("sys", "user") == '{"acts": []}'
    assert chain.model == "gemini:gemini-3.5-flash-lite", "active tier label follows the winner"


def test_failover_model_property_and_reset():
    class Tier:
        provider = "p"
        model = "m"
        reset_calls = 0

        def reset_history(self):
            self.reset_calls += 1  # type: ignore[attr-defined]

    t = Tier()
    chain = FailoverBackend([t])
    assert chain.model == "p:m"
    chain.reset_history()
    assert t.reset_calls == 1  # type: ignore[attr-defined]


def test_strip_code_fences_extracts_outermost_object():
    assert strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_code_fences('sure! {"a": {"b": 2}} hope that helps') == '{"a": {"b": 2}}'
    assert strip_code_fences("no braces") == "no braces"


def test_record_results_fans_out_via_chain():
    class Recorder:
        provider = "p"
        model = "m"
        recorded = 0

        def record_results(self, steps, results):
            self.recorded += 1  # type: ignore[attr-defined]

    a, b = Recorder(), Recorder()
    chain = FailoverBackend([a, b])  # type: ignore[list-item]
    chain.record_results([ToolStep(tool="timeline.trim_clip", args={})], ["ok"])
    assert a.recorded == 1  # type: ignore[attr-defined]
    _ = b  # only the active tier records (upstream semantics)
