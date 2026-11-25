"""Meta Model API (Muse) backend — pricing math, budget guard, wire format.

Pins (docs: dev.meta.ai/docs):
- pricing table incl. the Ember <=200K / >200K tier switch and the
  cached-input discount (cached tokens are a SUBSET of prompt tokens),
- /chat/completions tool_calls parsing + ``prompt_tokens_details.cached_tokens``,
- persistent spend ledger + hard budget cap checked BEFORE each HTTP call,
- 400 on an optional extra field -> one vanilla-payload retry,
- contributor-mode models rejected unless LLM_ALLOW_TRAINING_ON_CONTENT,
- env routing: api.meta.ai base URL (or LLM_PROVIDER=meta) -> MetaModelBackend,
- PlannerExecutor writes the per-proposal USD cost into agent_tool_logs
  (feeds the M5 cost dashboard).
"""

from __future__ import annotations

import json

import httpx
import pytest
from director_os.agent_runtime.agents import PlannerExecutor
from director_os.agent_runtime.llm import LLMBackendError, build_backend
from director_os.agent_runtime.meta import (
    BudgetExceededError,
    MetaModelBackend,
    SpendLedger,
    estimate_cost_usd,
    price_for,
)
from director_os.agent_runtime.registry import ToolRegistry
from director_os.core.config import get_settings


def mini_registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool("timeline.split_clip", description="split a clip")
    def split_clip(item_index: int, at_local_s: float) -> str:
        return "split"

    return reg


# Meta chat-completions shape: OpenAI tool_calls + prompt cache details.
META_OK = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "timeline.split_clip",
                            "arguments": '{"item_index": 0, "at_local_s": 1.5}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {
        "prompt_tokens": 1200,
        "completion_tokens": 88,
        "total_tokens": 1288,
        "prompt_tokens_details": {"cached_tokens": 900},
    },
}


def make_backend(
    tmp_path,
    handler,
    *,
    budget: float | None = 15.0,
    ledger: SpendLedger | None = None,
    model: str = "muse-ember-1.0",
    effort: str = "minimal",
    base_url: str = "https://api.meta.ai/v1",
) -> MetaModelBackend:
    return MetaModelBackend(
        "LLM_k",
        model,
        mini_registry(),
        base_url=base_url,
        transport=httpx.MockTransport(handler),
        ledger=ledger if ledger is not None else SpendLedger(tmp_path / "spend.jsonl"),
        budget_usd=budget,
        reasoning_effort=effort,
    )


# ------------------------------------------------------------------ pricing
def test_ember_small_tier_uses_cached_discount():
    # 1000 uncached input + 2000 cached input + 500 output (<=200K tier)
    cost = estimate_cost_usd("muse-ember-1.0", 3000, 2000, 500)
    expected = 1000 * 0.10 / 1e6 + 2000 * 0.004 / 1e6 + 500 * 0.50 / 1e6
    assert abs(cost - expected) < 1e-12


def test_ember_over_200k_input_switches_tier():
    cost = estimate_cost_usd("muse-ember-1.0", 250_000, 240_000, 1000)
    expected = 10_000 * 0.30 / 1e6 + 240_000 * 0.012 / 1e6 + 1000 * 1.50 / 1e6
    assert abs(cost - expected) < 1e-12


def test_spark_pricing_and_conservative_unknown_fallback():
    spark = estimate_cost_usd("muse-spark-1.3", 1_000_000, 0, 1_000_000)
    assert abs(spark - 5.50) < 1e-9  # 1.25 in + 4.25 out
    # unknown future muse ids are over-estimated with standard Spark pricing
    unknown = estimate_cost_usd("muse-spark-9.9", 1_000_000, 0, 1_000_000)
    assert price_for("muse-spark-9.9") is price_for("muse-spark-1.3")
    assert unknown == spark


# ------------------------------------------------------- wire + accounting
def test_meta_backend_parses_tool_calls_cached_tokens_and_ledger(tmp_path):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json=META_OK)

    ledger_path = tmp_path / "spend.jsonl"
    be = make_backend(tmp_path, handler, ledger=SpendLedger(ledger_path))
    steps = be.propose("state", "director")

    assert seen["auth"] == "Bearer LLM_k"
    assert "/chat/completions" in seen["url"]
    # openai-compat wire law: tool names are sanitized ('.' -> '__') on the way
    # out (Nvidia NIM rejects dots) and un-mangled back on the way in.
    assert seen["body"]["tools"][0]["function"]["name"] == "timeline__split_clip"
    assert seen["body"]["reasoning_effort"] == "minimal"
    assert [s.tool for s in steps] == ["timeline.split_clip"]
    assert be.last_usage == {"tokens_in": 1200, "tokens_out": 88}
    assert be.last_tokens_cached == 900
    expected = 300 * 0.10 / 1e6 + 900 * 0.004 / 1e6 + 88 * 0.50 / 1e6
    assert be.last_cost_usd is not None
    assert abs(be.last_cost_usd - expected) < 1e-9
    rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["tokens_cached"] == 900
    assert abs(rows[0]["cost_usd"] - expected) < 1e-9
    assert be.spend_usd() == pytest.approx(expected, abs=1e-9)


def test_budget_cap_blocks_call_before_http(tmp_path):
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, json=META_OK)

    ledger = SpendLedger(tmp_path / "spend.jsonl")
    ledger.record({"cost_usd": 15.0, "model": "muse-ember-1.0"})
    be = make_backend(tmp_path, handler, ledger=ledger, budget=15.0)
    with pytest.raises(BudgetExceededError, match="budget exhausted"):
        be.propose("state", "director")
    assert calls == []  # no HTTP request was paid for


def test_ledger_survives_restart_and_torn_lines(tmp_path):
    path = tmp_path / "spend.jsonl"
    path.write_text(
        json.dumps({"cost_usd": 0.5, "model": "muse-ember-1.0"})
        + "\n"
        + '{"torn": '  # killed process mid-write
        + "\n"
        + json.dumps({"cost_usd": 0.25, "model": "muse-spark-1.3"})
        + "\n"
    )
    assert SpendLedger(path).total_usd() == pytest.approx(0.75)


def test_meta_400_on_optional_extra_retries_vanilla(tmp_path):
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content.decode()))
        if len(bodies) == 1:
            return httpx.Response(
                400, json={"error": {"message": "Unknown parameter: reasoning_effort"}}
            )
        return httpx.Response(200, json=META_OK)

    be = make_backend(tmp_path, handler)
    steps = be.propose("state", "director")
    assert "reasoning_effort" in bodies[0]
    assert "reasoning_effort" not in bodies[1]
    assert [s.tool for s in steps] == ["timeline.split_clip"]


def test_meta_http_error_surfaces(tmp_path):
    be = make_backend(
        tmp_path,
        lambda r: httpx.Response(404, json={"error": {"code": "model_not_found"}}),
    )
    with pytest.raises(LLMBackendError, match="404"):
        be.propose("s", "director")


# ------------------------------------------------------------------ routing
def test_build_backend_routes_meta_via_base_url(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_API_KEY", "LLM_k")
    monkeypatch.setenv("LLM_MODEL", "muse-ember-1.0")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.meta.ai/v1")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        be = build_backend(mini_registry())
        assert isinstance(be, MetaModelBackend)
        assert be.budget_usd == 15.0
        assert be.ledger is not None and be.ledger.path == tmp_path / "llm_spend.jsonl"
    finally:
        get_settings.cache_clear()


def test_build_backend_routes_meta_via_provider_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_API_KEY", "LLM_k")
    monkeypatch.setenv("LLM_MODEL", "muse-spark-1.3")
    monkeypatch.setenv("LLM_PROVIDER", "meta")
    monkeypatch.setenv("LLM_BASE_URL", "https://gw.example/v1")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        be = build_backend(mini_registry())
        assert isinstance(be, MetaModelBackend)
        assert be.base_url == "https://gw.example/v1"
    finally:
        get_settings.cache_clear()


def test_contributor_model_rejected_without_opt_in(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_API_KEY", "LLM_k")
    monkeypatch.setenv("LLM_MODEL", "muse-spark-1.3-contributor")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.meta.ai/v1")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        with pytest.raises(LLMBackendError, match="train on the content"):
            build_backend(mini_registry())
        monkeypatch.setenv("LLM_ALLOW_TRAINING_ON_CONTENT", "1")
        get_settings.cache_clear()
        be = build_backend(mini_registry())
        assert isinstance(be, MetaModelBackend)
    finally:
        get_settings.cache_clear()


def test_non_meta_gateway_unchanged(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "LLM_k")
    monkeypatch.setenv("LLM_MODEL", "glm-4.6")
    monkeypatch.setenv("LLM_BASE_URL", "https://gw.example/v1")
    monkeypatch.setenv("AGENT_BRAIN", "legacy")
    get_settings.cache_clear()
    try:
        be = build_backend(mini_registry())
        assert type(be).__name__ == "OpenAICompatBackend"
        assert not isinstance(be, MetaModelBackend)
    finally:
        get_settings.cache_clear()


# ------------------------------------------------------- M5 dashboard feed
def test_planner_executor_persists_usd_cost(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=META_OK)

    from director_os.agent_runtime.checkpoint import CheckpointStore
    from director_os.core.db import init_models, make_engine, make_session_factory

    be = make_backend(tmp_path, handler, ledger=SpendLedger(tmp_path / "spend.jsonl"))
    engine = make_engine(f"sqlite:///{tmp_path}/plan.db")
    init_models(engine)
    store = CheckpointStore(make_session_factory(engine))

    import opentimelineio as otio

    executor = PlannerExecutor(mini_registry(), store, "p-meta")
    run = executor.run(be, otio.schema.Timeline(name="meta-cost-test"))
    assert run.ok
    row = next(r for r in store.recent_logs("p-meta") if r["tool"] == "llm.propose")
    assert row["tokens_in"] == 1200
    assert row["tokens_out"] == 88
    expected = 300 * 0.10 / 1e6 + 900 * 0.004 / 1e6 + 88 * 0.50 / 1e6
    assert row["token_cost_usd"] == pytest.approx(expected, abs=1e-9)
