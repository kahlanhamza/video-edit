"""Stage-6/Stage-4 routing chain — wire facts for the multi-provider law.

Pins (user-specified routing, 2026-11-19):
- Stage 6 Director: Groq openai-compat PRIMARY (openai/gpt-oss-120b ->
  qwen/qwen3.8-27b -> openai/gpt-oss-20b chain INSIDE the tier, honoring
  429 Retry-After), then OpenRouter (nvidia/nemotron-3-ultra:free), then
  native Gemini (gemini-3.5-flash-lite -> gemini-3.1-flash-lite).
- Stage 4 Vision: Gemini PRIMARY (gemini-3.5-flash-lite ->
  gemini-3.1-flash-lite), then OpenRouter (google/gemma-4-31b:free ->
  qwen/qwen3.8-27b:free). The bogus "openrouter/free" id must never appear.
- Capacity-only failover: a 400 contract error re-raises instead of
  silently hiding behind the next tier.

httpx.MockTransport keeps every test hermetic.
"""

from __future__ import annotations

import json

import httpx
import pytest
from director_os.agent_runtime.llm import (
    FailoverBackend,
    LLMBackendError,
    OpenAICompatBackend,
    _groq_prompt_budget,
    build_backend_chain,
)
from director_os.agent_runtime.registry import ToolRegistry
from director_os.core.config import get_settings
from director_os.perception.vision import (
    OPENROUTER_VISION_MODELS,
    FailoverVisionClient,
    FrameRef,
    GeminiVisionClient,
    OpenRouterVisionClient,
    VisionError,
    build_vision_client,
)


def mini_registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool("timeline.split_clip", description="split a clip")
    def split_clip(item_index: int, at_local_s: float) -> str:
        return "split"

    return reg


def groq_ok_body() -> dict:
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "timeline.split_clip",
                                "arguments": json.dumps({"item_index": 0, "at_local_s": 1.5}),
                            }
                        }
                    ]
                }
            }
        ],
        "usage": {"prompt_tokens": 120, "completion_tokens": 30},
    }


def openrouter_ok_body() -> dict:
    return groq_ok_body()


def gemini_ok_body() -> dict:
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "timeline.split_clip",
                                "args": {"item_index": 0, "at_local_s": 1.5},
                            }
                        }
                    ]
                }
            }
        ],
        "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 20},
    }


# --------------------------------------------------------------------- stage 6
def test_groq_429_honors_retry_after_then_succeeds(monkeypatch):
    """429 + Retry-After -> sleep the window (capped), retry SAME model, 200."""
    seen_models: list[str] = []
    calls = {"n": 0}
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        seen_models.append(json.loads(request.content)["model"])
        if calls["n"] == 1:
            return httpx.Response(
                429, json={"error": "rate limit exceeded"}, headers={"Retry-After": "8"}
            )
        return httpx.Response(200, json=groq_ok_body())

    monkeypatch.setattr("director_os.agent_runtime.llm.time.sleep", sleeps.append)
    backend = OpenAICompatBackend(
        "gsk_k",
        "openai/gpt-oss-120b",
        mini_registry(),
        base_url="https://api.groq.com/openai/v1",
        fallback_models=[],
        retry_after_max_s=20.0,
        transport=httpx.MockTransport(handler),
    )
    steps = backend.propose("state", "director")
    assert [s.tool for s in steps] == ["timeline.split_clip"]
    assert seen_models == ["openai/gpt-oss-120b", "openai/gpt-oss-120b"]  # retried, not downgraded
    assert sleeps == [8.0]  # Retry-After honored verbatim (under the 20s cap)
    assert backend.last_usage == {"tokens_in": 120, "tokens_out": 30}


def test_groq_429_twice_downgrades_to_next_model_in_chain():
    """Retry-After honored once, still 429 -> walk to the next candidate."""
    seen_models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        seen_models.append(model)
        if model == "openai/gpt-oss-120b":
            return httpx.Response(429, json={"error": "rate limit"}, headers={"Retry-After": "0"})
        return httpx.Response(200, json=groq_ok_body())

    backend = OpenAICompatBackend(
        "gsk_k",
        "openai/gpt-oss-120b",
        mini_registry(),
        base_url="https://api.groq.com/openai/v1",
        fallback_models=["qwen/qwen3.8-27b", "openai/gpt-oss-20b"],
        retry_after_max_s=0.01,
        transport=httpx.MockTransport(handler),
    )
    steps = backend.propose("state", "director")
    assert steps and backend.model == "qwen/qwen3.8-27b"
    assert set(seen_models) == {"openai/gpt-oss-120b", "qwen/qwen3.8-27b"}


def test_groq_dead_model_name_skips_to_next_candidate():
    """400 'model decommissioned' -> skip candidate, NOT a hard failure."""

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        if model == "openai/gpt-oss-120b":
            return httpx.Response(
                400, json={"error": {"message": "model openai/gpt-oss-120b is decommissioned"}}
            )
        if model == "qwen/qwen3.8-27b":
            return httpx.Response(
                400, json={"error": {"message": "model qwen/qwen3.8-27b does not exist"}}
            )
        return httpx.Response(200, json=groq_ok_body())

    backend = OpenAICompatBackend(
        "gsk_k",
        "openai/gpt-oss-120b",
        mini_registry(),
        base_url="https://api.groq.com/openai/v1",
        fallback_models=["qwen/qwen3.8-27b", "openai/gpt-oss-20b"],
        retry_after_max_s=0.01,
        transport=httpx.MockTransport(handler),
    )
    steps = backend.propose("state", "director")
    assert [s.tool for s in steps] == ["timeline.split_clip"]
    assert backend.model == "openai/gpt-oss-20b"


def test_groq_contract_400_raises_no_silent_skip():
    """A non-model 400 is a CONTRACT error: re-raise, never hide it."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "tools schema invalid"}})

    backend = OpenAICompatBackend(
        "gsk_k",
        "openai/gpt-oss-120b",
        mini_registry(),
        base_url="https://api.groq.com/openai/v1",
        fallback_models=["openai/gpt-oss-20b"],
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(LLMBackendError, match="tools schema invalid"):
        backend.propose("state", "director")


def test_groq_chain_exhaustion_raises_capacity_marked_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "rate limit"}, headers={"Retry-After": "0"})

    backend = OpenAICompatBackend(
        "gsk_k",
        "openai/gpt-oss-120b",
        mini_registry(),
        base_url="https://api.groq.com/openai/v1",
        fallback_models=["openai/gpt-oss-20b"],
        retry_after_max_s=0.01,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(LLMBackendError) as err:
        backend.propose("state", "director")
    assert "429" in str(err.value)  # capacity marker — cross-tier failover may proceed


def test_failover_backend_walks_tiers_on_capacity_only():
    class Boom429:
        provider = "openai_compat"
        model = "openai/gpt-oss-120b"

        def propose(self, state_summary, role):
            raise LLMBackendError("openai-compat capacity exhausted across 3 model(s): 429 x")

    class Boom429b:
        provider = "openai_compat"
        model = "nvidia/nemotron-3-ultra:free"

        def propose(self, state_summary, role):
            raise LLMBackendError("openai-compat capacity exhausted across 1 model(s): 429 x")

    class Good:
        provider = "gemini"
        model = "gemini-3.5-flash-lite"

        def __init__(self):
            self.last_usage = {"tokens_in": 1, "tokens_out": 2}

        def propose(self, state_summary, role):
            from director_os.agent_runtime.agents import ToolStep

            return [ToolStep(tool="timeline.split_clip", args={"item_index": 0, "at_local_s": 2.0})]

        def record_results(self, steps, results):
            self.recorded = (steps, results)

    chain = FailoverBackend([Boom429(), Boom429b(), Good()])
    steps = chain.propose("state", "director")
    assert [s.tool for s in steps] == ["timeline.split_clip"]
    assert chain.active.provider == "gemini"
    assert chain.last_usage == {"tokens_in": 1, "tokens_out": 2}
    chain.record_results(steps, ["split"])
    assert chain.active.recorded[1] == ["split"]


def test_failover_backend_reraises_contract_errors():
    class Bad400:
        provider = "openai_compat"
        model = "x"

        def propose(self, state_summary, role):
            raise LLMBackendError("openai-compat 400: tools schema invalid")

    chain = FailoverBackend([Bad400()])
    with pytest.raises(LLMBackendError, match="tools schema invalid"):
        chain.propose("state", "director")


def test_groq_prompt_budget_clamps_state_summary():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=groq_ok_body())

    backend = OpenAICompatBackend(
        "gsk_k",
        "openai/gpt-oss-120b",
        mini_registry(),
        base_url="https://api.groq.com/openai/v1",
        fallback_models=[],
        prompt_char_budget=80,
        transport=httpx.MockTransport(handler),
    )
    backend.propose("S" * 500, "director")
    user_text = captured["body"]["messages"][1]["content"]
    assert "clamped to the provider TPM budget" in user_text
    assert len(user_text) < 500


def test_groq_prompt_budget_only_on_groq_host(monkeypatch):
    monkeypatch.delenv("LLM_PROMPT_CHAR_BUDGET", raising=False)
    assert _groq_prompt_budget("https://api.groq.com/openai/v1") == 6000
    assert _groq_prompt_budget("https://openrouter.ai/api/v1") == 0
    assert _groq_prompt_budget("https://api.openai.com/v1") == 0


def test_build_backend_chain_full_stack_order(monkeypatch, tmp_path):
    """Groq -> OpenRouter -> Gemini, exactly the user-specified tier order."""
    monkeypatch.setenv("LLM_API_KEY", "gsk_k")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-120b")
    monkeypatch.setenv("LLM_FALLBACK_MODELS", "qwen/qwen3.8-27b,openai/gpt-oss-20b")
    monkeypatch.setenv("AGENT_BRAIN", "legacy")  # pins the legacy tier internals
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or_k")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra:free")
    monkeypatch.setenv("VISION_API_KEY", "AQ.k")
    monkeypatch.setenv("VISION_MODEL", "gemini-3.5-flash-lite")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "gemini-3.1-flash-lite")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        backend = build_backend_chain(mini_registry())
        assert isinstance(backend, FailoverBackend)
        assert backend.tier_names == [
            "openai_compat:openai/gpt-oss-120b",
            "openai_compat:nvidia/nemotron-3-ultra:free",
            "gemini:gemini-3.5-flash-lite",
        ]
        groq_tier = backend._backends[0]
        assert groq_tier._fallbacks == ["qwen/qwen3.8-27b", "openai/gpt-oss-20b"]
        assert groq_tier.prompt_char_budget == 6000
        gemini_tier = backend._backends[2]
        assert gemini_tier._fallbacks == ["gemini-3.1-flash-lite"]
    finally:
        get_settings.cache_clear()


def test_build_backend_chain_native_gemini_primary_no_duplicate(monkeypatch, tmp_path):
    """Blank LLM_BASE_URL -> native Gemini primary; OpenRouter still appended."""
    monkeypatch.setenv("LLM_API_KEY", "AQ.k")
    monkeypatch.setenv("LLM_BASE_URL", "")
    monkeypatch.setenv("LLM_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("AGENT_BRAIN", "legacy")  # pins the legacy tier internals
    monkeypatch.delenv("VISION_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or_k")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra:free")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        backend = build_backend_chain(mini_registry())
        assert isinstance(backend, FailoverBackend)
        providers = [b.provider for b in backend._backends]
        assert providers == ["gemini", "openai_compat"]  # no duplicate gemini tier
    finally:
        get_settings.cache_clear()


def test_build_backend_chain_single_tier_is_bare(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_API_KEY", "gsk_k")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-120b")
    monkeypatch.setenv("AGENT_BRAIN", "legacy")  # pins the legacy tier internals
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("VISION_API_KEY", raising=False)
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        backend = build_backend_chain(mini_registry())
        assert isinstance(backend, OpenAICompatBackend)  # no wrapper overhead
    finally:
        get_settings.cache_clear()


def test_build_backend_chain_nothing_configured_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("VISION_API_KEY", raising=False)
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        with pytest.raises(LLMBackendError, match="LLM not configured"):
            build_backend_chain(mini_registry())
    finally:
        get_settings.cache_clear()


# --------------------------------------------------------------------- stage 4
def _frame(tmp_path) -> FrameRef:
    p = tmp_path / "f0.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    return FrameRef(timestamp_s=3.2, image_path=p)


def test_gemini_vision_429_walks_to_fallback_model(tmp_path):
    asked: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append(request.url.path)
        if "gemini-3.5-flash-lite" in request.url.path:
            return httpx.Response(429, json={"error": {"status": "RESOURCE_EXHAUSTED"}})
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": json.dumps(
                                        [
                                            {
                                                "timestamp_s": 3.2,
                                                "kind": "observation",
                                                "description": "unit parked",
                                                "confidence": 0.9,
                                            }
                                        ]
                                    )
                                }
                            ]
                        }
                    }
                ]
            },
        )

    client = GeminiVisionClient(
        "AQ.k",
        model="gemini-3.5-flash-lite",
        fallback_models=["gemini-3.1-flash-lite"],
        modal_egress=False,
        transport=httpx.MockTransport(handler),
    )
    insights = client.analyze("what is parked?", [_frame(tmp_path)])
    assert insights[0].description == "unit parked"
    assert client.model == "gemini-3.1-flash-lite"
    assert len(asked) == 2


def test_openrouter_vision_parses_insights_and_never_free_id(tmp_path):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                [
                                    {
                                        "timestamp_s": 3.2,
                                        "kind": "risk",
                                        "description": "plate visible",
                                        "confidence": 0.8,
                                    }
                                ]
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 50, "completion_tokens": 12},
            },
        )

    client = OpenRouterVisionClient("sk-or_k", transport=httpx.MockTransport(handler))
    insights = client.analyze("any plates?", [_frame(tmp_path)])
    assert insights[0].kind == "risk"
    assert captured["body"]["model"] == "google/gemma-4-31b-it:free"  # live catalog id
    image_part = next(
        p for p in captured["body"]["messages"][0]["content"] if p["type"] == "image_url"
    )
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert client.last_usage == {"tokens_in": 50, "tokens_out": 12}


def test_openrouter_vision_rotated_catalog_ids_are_skipped(tmp_path):
    """A 404/400 dead model id walks to the next candidate — never blocks."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body["model"])
        if body["model"] == "google/gemma-4-31b-it:free":
            return httpx.Response(
                400,
                json={"error": {"message": "No endpoints found for google/gemma-4-31b-it:free."}},
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps([])}}]},
        )

    client = OpenRouterVisionClient(
        "sk-or_k",
        models=["google/gemma-4-31b-it:free", "nvidia/nemotron-3.5-super-vl:free"],
        transport=httpx.MockTransport(handler),
    )
    assert client.analyze("q", [_frame(tmp_path)]) == []
    assert seen == ["google/gemma-4-31b-it:free", "nvidia/nemotron-3.5-super-vl:free"]


def test_openrouter_vision_429_walks_to_second_model(tmp_path):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body["model"])
        if body["model"] == "google/gemma-4-31b-it:free":
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps([])}}],
            },
        )

    client = OpenRouterVisionClient(
        "sk-or_k",
        models=["google/gemma-4-31b-it:free", "nvidia/nemotron-3.5-super-vl:free"],
        transport=httpx.MockTransport(handler),
    )
    assert client.analyze("q", [_frame(tmp_path)]) == []
    assert seen == ["google/gemma-4-31b-it:free", "nvidia/nemotron-3.5-super-vl:free"]


def test_failover_vision_gemini_exhausted_openrouter_serves(tmp_path):
    def gemini_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "RESOURCE_EXHAUSTED"})

    def or_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                [
                                    {
                                        "timestamp_s": 1.0,
                                        "kind": "observation",
                                        "description": "via openrouter",
                                        "confidence": 0.7,
                                    }
                                ]
                            )
                        }
                    }
                ]
            },
        )

    gemini = GeminiVisionClient(
        "AQ.k",
        model="gemini-3.5-flash-lite",
        fallback_models=["gemini-3.1-flash-lite"],
        modal_egress=False,
        transport=httpx.MockTransport(gemini_handler),
    )
    openrouter = OpenRouterVisionClient("sk-or_k", transport=httpx.MockTransport(or_handler))
    chain = FailoverVisionClient([gemini, openrouter])
    insights = chain.analyze("q", [_frame(tmp_path)])
    assert insights[0].description == "via openrouter"
    assert chain.active.provider == "openrouter"


def test_failover_vision_reraises_contract_error(tmp_path):
    class Bad:
        provider = "gemini"

        def analyze(self, question, frames):
            raise VisionError("vision provider 401: invalid api key")

    chain = FailoverVisionClient([Bad()])
    with pytest.raises(VisionError, match="401"):
        chain.analyze("q", [_frame(tmp_path)])


def test_build_vision_client_wraps_failover_with_openrouter(monkeypatch, tmp_path):
    monkeypatch.setenv("VISION_PROVIDER", "gemini")
    monkeypatch.setenv("VISION_API_KEY", "AQ.k")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or_k")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        client = build_vision_client()
        assert isinstance(client, FailoverVisionClient)
        primary, fallback = client._clients
        assert isinstance(primary, GeminiVisionClient)
        assert primary.model == "gemini-3.5-flash-lite"
        assert isinstance(fallback, OpenRouterVisionClient)
        assert "openrouter/free" not in fallback._models  # bogus id must never appear
    finally:
        get_settings.cache_clear()


def test_build_vision_client_no_openrouter_keeps_bare_gemini(monkeypatch, tmp_path):
    monkeypatch.setenv("VISION_PROVIDER", "gemini")
    monkeypatch.setenv("VISION_API_KEY", "AQ.k")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        client = build_vision_client()
        assert isinstance(client, GeminiVisionClient)
    finally:
        get_settings.cache_clear()


def test_openrouter_vision_model_ids_pinned():
    """The routing law, pinned: live ids first, requested ids retained; never
    the bogus 'openrouter/free'."""
    assert OPENROUTER_VISION_MODELS == (
        "google/gemma-4-31b-it:free",
        "nvidia/nemotron-3.5-super-vl:free",
        "google/gemma-4-31b:free",
        "qwen/qwen3.8-27b:free",
    )
    assert all(m != "openrouter/free" for m in OPENROUTER_VISION_MODELS)


def test_openai_compat_tool_names_sanitized_on_wire():
    """Wire fact 2026-11-19 (Nvidia NIM): dots in function names 400 before the
    model runs. Schemas go out dot-free; proposals come back canonical."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["tools"] = json.loads(request.content)["tools"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "timeline__split_clip",
                                        "arguments": json.dumps(
                                            {"item_index": 0, "at_local_s": 2.5}
                                        ),
                                    }
                                }
                            ]
                        }
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5},
            },
        )

    backend = OpenAICompatBackend(
        "sk-or_k",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        mini_registry(),
        base_url="https://openrouter.ai/api/v1",
        fallback_models=[],
        transport=httpx.MockTransport(handler),
    )
    steps = backend.propose("state", "director")
    wire_names = [t["function"]["name"] for t in captured["tools"]]
    assert wire_names == ["timeline__split_clip"]  # no dots on the wire
    assert [s.tool for s in steps] == ["timeline.split_clip"]  # canonical on dispatch


def test_groq_dead_key_403_fails_over_to_openrouter():
    """Wire fact 2026-11-19: Groq 403s even GET /models — the tier must die
    QUIETLY (failover), not kill stage 6."""

    class Groq403:
        provider = "openai_compat"
        model = "openai/gpt-oss-120b"

        def propose(self, state_summary, role):
            raise LLMBackendError('openai-compat 403: {"error":{"message":"Forbidden"}}')

    class Ok:
        provider = "openai_compat"
        model = "nvidia/nemotron-3-ultra-550b-a55b:free"
        last_usage = {"tokens_in": 9, "tokens_out": 4}

        def propose(self, state_summary, role):
            from director_os.agent_runtime.agents import ToolStep

            return [ToolStep(tool="timeline.split_clip", args={"item_index": 0, "at_local_s": 1.0})]

    chain = FailoverBackend([Groq403(), Ok()])
    steps = chain.propose("state", "director")
    assert [s.tool for s in steps] == ["timeline.split_clip"]
    assert chain.active.model == "nvidia/nemotron-3-ultra-550b-a55b:free"


def test_gemini_geo_block_fails_over_to_openrouter():
    """Tape-1 wire fact 2026-11-25: the Gemini tier raises 'geo-blocked from
    this region' (a 403 egress condition, no status code in the message) —
    the chain must treat it as capacity and walk to the OpenRouter tier,
    never kill the Director (the 251s fallback-narration root cause)."""

    class GeoBlockedGemini:
        provider = "gemini"
        model = "gemini-3.5-flash-lite"

        def complete_json(self, system, user, max_tokens=4096):
            raise LLMBackendError(
                "gemini complete_json exhausted fallbacks: Gemini geo-blocked "
                "from this region and MODAL_TOKEN_ID/SECRET not configured for fleet egress"
            )

    class Ok:
        provider = "openai_compat"
        model = "nvidia/nemotron-3-ultra-550b-a55b:free"

        def complete_json(self, system, user, max_tokens=4096):
            return '{"ok":true}'

    chain = FailoverBackend([GeoBlockedGemini(), Ok()])
    out = chain.complete_json("system", "user")
    assert out == '{"ok":true}'
    assert chain.active.model == "nvidia/nemotron-3-ultra-550b-a55b:free"
