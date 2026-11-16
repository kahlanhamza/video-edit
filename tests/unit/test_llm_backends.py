"""Real LLM backends + vision client — provider wiring behind hermetic ports.

httpx.MockTransport pins the wire format (functionDeclarations, tool_calls,
usage accounting) without network. Provider choice logic is env-driven.
"""

from __future__ import annotations

import json

import httpx
import pytest
from director_os.agent_runtime.llm import (
    GeminiRestBackend,
    LLMBackendError,
    OpenAICompatBackend,
    build_backend,
)
from director_os.agent_runtime.registry import ToolRegistry


def mini_registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool("timeline.split_clip", description="split a clip")
    def split_clip(item_index: int, at_local_s: float) -> str:
        return "split"

    return reg


GEMINI_OK = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {"text": "thinking"},
                    {
                        "functionCall": {
                            "name": "timeline.split_clip",
                            "args": {"item_index": 0, "at_local_s": 1.5},
                        }
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
    be = GeminiRestBackend("k", "gemini-2.5-flash", reg, transport=httpx.MockTransport(handler))
    steps = be.propose("project=p master_items=[]", "director")

    assert seen["auth"] == "k"
    assert "/v1beta/models/gemini-2.5-flash:generateContent" in seen["url"]
    decls = seen["body"]["tools"][0]["functionDeclarations"]
    assert decls[0]["name"] == "timeline.split_clip"
    assert "additionalProperties" not in json.dumps(decls)
    assert seen["body"]["systemInstruction"]["parts"][0]["text"]
    # unknown tool dropped; known one mapped
    assert [s.tool for s in steps] == ["timeline.split_clip"]
    assert steps[0].args == {"item_index": 0, "at_local_s": 1.5}
    assert be.last_usage == {"tokens_in": 120, "tokens_out": 34}


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
