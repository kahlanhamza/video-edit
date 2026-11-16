"""Real LLM backends — provider wiring behind hermetic ports.

httpx.MockTransport pins the wire format (functionDeclarations, tool_calls,
usage accounting) without network. Provider choice logic is env-driven.

The Gemini tests pin the three known Gemini 3.x REST traps:
1. `thoughtSignature` replay across multi-step tool turns (assistant turns go
   back verbatim; normalized copies ride on ToolStep.provider_meta),
2. `parametersJsonSchema` (full JSON Schema) instead of the legacy `parameters`
   field (which 400s on additionalProperties / $defs / $ref),
3. 401 ACCESS_TOKEN_TYPE_UNSUPPORTED on x-goog-api-key -> ?key= retry.
"""

from __future__ import annotations

import json

import httpx
import pytest
from director_os.agent_runtime.agents import PlannerExecutor
from director_os.agent_runtime.llm import (
    GeminiRestBackend,
    LLMBackendError,
    OpenAICompatBackend,
    build_backend,
)
from director_os.agent_runtime.registry import ToolRegistry
from pydantic import BaseModel, ConfigDict


def mini_registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool("timeline.split_clip", description="split a clip")
    def split_clip(item_index: int, at_local_s: float) -> str:
        return "split"

    return reg


def messy_registry() -> ToolRegistry:
    """Registry whose schema carries exactly the keys the legacy `parameters`
    field rejects (`additionalProperties`, `$defs`/`$ref`) — trap #2 fixture."""

    class ClipRef(BaseModel):
        model_config = ConfigDict(extra="forbid")
        item_id: str

    class SplitArgs(BaseModel):
        model_config = ConfigDict(extra="forbid")
        target: ClipRef
        mode: str

    reg = ToolRegistry()
    reg.register(
        "timeline.split_clip",
        "split a clip",
        SplitArgs,
        lambda target, mode: "split",
    )
    return reg


# thoughtSignature sits on the first functionCall part, opaque + untouched.
GEMINI_OK = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {"text": "thinking"},
                    {
                        "thoughtSignature": "sig-opaque-abc123",
                        "functionCall": {
                            "id": "call_1",
                            "name": "timeline.split_clip",
                            "args": {"item_index": 0, "at_local_s": 1.5},
                        },
                    },
                    {"functionCall": {"name": "made_up_tool", "args": {}}},
                ]
            }
        }
    ],
    "usageMetadata": {"promptTokenCount": 120, "candidatesTokenCount": 34},
}

OPENAI_OK = {
    "choices": [
        {
            "message": {
                "tool_calls": [
                    {
                        "function": {
                            "name": "timeline.split_clip",
                            "arguments": '{"item_index": 0, "at_local_s": 1.5}',
                        }
                    },
                    {"function": {"name": "made_up_tool", "arguments": "{}"}},
                ]
            }
        }
    ],
    "usage": {"prompt_tokens": 90, "completion_tokens": 21},
}


def test_gemini_backend_parses_function_calls_and_usage():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("x-goog-api-key")
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json=GEMINI_OK)

    reg = mini_registry()
    be = GeminiRestBackend("k", "gemini-3.8-flash", reg, transport=httpx.MockTransport(handler))
    steps = be.propose("project=p master_items=[]", "director")

    assert seen["auth"] == "k"
    assert "/v1beta/models/gemini-3.8-flash:generateContent" in seen["url"]
    decls = seen["body"]["tools"][0]["functionDeclarations"]
    assert decls[0]["name"] == "timeline.split_clip"
    # trap #2: full JSON Schema channel, never the lossy legacy field
    assert "parametersJsonSchema" in decls[0]
    assert "parameters" not in decls[0]
    assert seen["body"]["systemInstruction"]["parts"][0]["text"]
    # unknown tool dropped; known one mapped with provider metadata
    assert [s.tool for s in steps] == ["timeline.split_clip"]
    assert steps[0].args == {"item_index": 0, "at_local_s": 1.5}
    assert steps[0].provider_meta == {
        "thoughtSignature": "sig-opaque-abc123",
        "call_id": "call_1",
    }
    assert be.last_usage == {"tokens_in": 120, "tokens_out": 34}


def test_gemini_passthrough_full_json_schema():
    """Trap #2: Pydantic v2 output ($defs/$ref, additionalProperties) rides in
    `parametersJsonSchema` untouched — stripping would lose constraints."""

    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"candidates": [{"content": {"parts": []}}]})

    be = GeminiRestBackend("k", "m", messy_registry(), transport=httpx.MockTransport(handler))
    be.propose("s", "director")

    decl = seen["body"]["tools"][0]["functionDeclarations"][0]
    assert "parameters" not in decl
    schema = decl["parametersJsonSchema"]
    assert schema["additionalProperties"] is False
    assert "$defs" in schema
    assert "$ref" in schema["properties"]["target"]


def test_gemini_replays_thought_signature_verbatim():
    """Trap #1: on turn 2 the assistant turn is replayed byte-for-byte — the
    opaque thoughtSignature stays inside its functionCall part."""

    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content.decode()))
        return httpx.Response(200, json=GEMINI_OK)

    be = GeminiRestBackend("k", "m", mini_registry(), transport=httpx.MockTransport(handler))
    be.propose("state one", "director")
    be.propose("state two", "director")

    contents = bodies[1]["contents"]
    model_turn = contents[0]
    assert model_turn["role"] == "model"
    # verbatim: every part, including the thought-summary text part
    assert model_turn["parts"][0] == {"text": "thinking"}
    call_part = next(p for p in model_turn["parts"] if "functionCall" in p)
    assert call_part["thoughtSignature"] == "sig-opaque-abc123"
    assert call_part["functionCall"] == {
        "id": "call_1",
        "name": "timeline.split_clip",
        "args": {"item_index": 0, "at_local_s": 1.5},
    }
    # the fresh state turn is appended after the replayed history
    assert contents[-1]["role"] == "user"


def test_gemini_record_results_appends_function_response():
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content.decode()))
        return httpx.Response(200, json=GEMINI_OK)

    be = GeminiRestBackend("k", "m", mini_registry(), transport=httpx.MockTransport(handler))
    steps = be.propose("state", "director")
    be.record_results(steps, ["split ok"])
    be.propose("state two", "director")

    contents = bodies[1]["contents"]
    assert contents[0]["role"] == "model"
    fr = contents[1]["parts"][0]["functionResponse"]
    assert contents[1]["role"] == "user"
    assert fr["name"] == "timeline.split_clip"
    assert fr["response"] == {"result": "split ok"}


def test_gemini_401_access_token_type_unsupported_falls_back_to_key_param():
    """Trap #3: some AI Studio AQ. keys reject the x-goog-api-key header with
    401 ACCESS_TOKEN_TYPE_UNSUPPORTED; ?key= resolves it."""

    attempts: list[tuple[str, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append((str(request.url), request.headers.get("x-goog-api-key")))
        if len(attempts) == 1:
            return httpx.Response(
                401,
                json={
                    "error": {
                        "code": 401,
                        "message": "Invalid credentials [ACCESS_TOKEN_TYPE_UNSUPPORTED]",
                        "status": "INVALID_ARGUMENT",
                    }
                },
            )
        return httpx.Response(200, json=GEMINI_OK)

    be = GeminiRestBackend(
        "k-ai", "m", mini_registry(), transport=httpx.MockTransport(handler)
    )
    steps = be.propose("s", "director")

    assert len(attempts) == 2  # header attempt, then ?key= retry
    assert attempts[0][1] == "k-ai"
    assert "key=k-ai" in attempts[1][0]
    assert [s.tool for s in steps] == ["timeline.split_clip"]


def test_planner_executor_feeds_results_to_gemini_backend(tmp_path):
    """PlannerExecutor.run duck-types record_results — the Gemini conversation
    stays replayable without importing any provider type in the runtime."""

    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content.decode()))
        return httpx.Response(200, json=GEMINI_OK)

    from director_os.agent_runtime.checkpoint import CheckpointStore
    from director_os.core.db import init_models, make_engine, make_session_factory

    be = GeminiRestBackend("k", "m", mini_registry(), transport=httpx.MockTransport(handler))
    engine = make_engine(f"sqlite:///{tmp_path}/plan.db")
    init_models(engine)
    store = CheckpointStore(make_session_factory(engine))

    import opentimelineio as otio

    executor = PlannerExecutor(
        mini_registry(), store, "p-plan", agent="director_agent", role="director"
    )
    run = executor.run(be, otio.schema.Timeline(name="plan-test"))

    assert run.ok
    assert [s.tool for s in run.steps] == ["timeline.split_clip"]
    # executor.run is ONE propose; the recorded results are replayed on the
    # NEXT propose: [replayed model turn, functionResponse turn, new user turn]
    be.propose("state two", "director")
    contents = bodies[1]["contents"]
    assert contents[0]["role"] == "model"
    fr = contents[1]["parts"][0]["functionResponse"]
    assert fr["name"] == "timeline.split_clip"
    assert fr["response"] == {"result": "split"}
    assert contents[-1]["role"] == "user"


def test_openai_compat_backend_parses_tool_calls():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer k2"
        assert "/chat/completions" in str(request.url)
        return httpx.Response(200, json=OPENAI_OK)

    reg = mini_registry()
    be = OpenAICompatBackend(
        "k2",
        "glm-4.6",
        reg,
        base_url="https://gw.example/v1",
        transport=httpx.MockTransport(handler),
    )
    steps = be.propose("state", "director")
    assert [s.tool for s in steps] == ["timeline.split_clip"]
    assert steps[0].args["at_local_s"] == 1.5
    assert steps[0].provider_meta is None
    assert be.last_usage == {"tokens_in": 90, "tokens_out": 21}


def test_backend_raises_on_http_error():
    be = GeminiRestBackend(
        "k",
        "m",
        mini_registry(),
        transport=httpx.MockTransport(lambda r: httpx.Response(429, text="rate")),
    )
    with pytest.raises(LLMBackendError, match="429"):
        be.propose("s", "director")


def test_build_backend_env_routing(tmp_path, monkeypatch):
    reg = mini_registry()
    monkeypatch.setenv("LLM_API_KEY", "abc")
    monkeypatch.setenv("LLM_MODEL", "gemini-3.8-flash")
    from director_os.core.config import get_settings

    get_settings.cache_clear()
    try:
        monkeypatch.setenv("LLM_BASE_URL", "")
        be = build_backend(reg)
        assert isinstance(be, GeminiRestBackend)

        monkeypatch.setenv("LLM_BASE_URL", "https://gw.example/v1")
        get_settings.cache_clear()
        be2 = build_backend(reg, llm_model="glm-4.6")
        assert isinstance(be2, OpenAICompatBackend)
        assert be2.base_url == "https://gw.example/v1"

        monkeypatch.delenv("LLM_API_KEY")
        get_settings.cache_clear()
        with pytest.raises(LLMBackendError, match="not configured"):
            build_backend(reg)
    finally:
        get_settings.cache_clear()
