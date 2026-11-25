"""Pydantic AI brain — the framework-owned agent loop behind the LLMBackend port.

Hermetic: FunctionModel replays fixed model responses (no network, no model
downloads, same pattern as ScriptedBackend). Pins the contracts that make this
a real agent loop:

- structured output: PlanOutput schema validated by the framework, not by a
  hand-rolled JSON parser,
- fail-closed WITH feedback: a hallucinated tool name or a schema violation
  becomes a ModelRetry the model must answer (never a silently dropped step),
- FallbackModel: a capacity error (429) on the primary model transparently
  moves to the next candidate,
- message-history replay: tool results recorded via record_results() reach the
  next proposal turn,
- PlannerExecutor integration: the brain slots behind the unchanged port.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from director_os.agent_runtime.agents import PlannerExecutor
from director_os.agent_runtime.checkpoint import CheckpointStore
from director_os.agent_runtime.llm import GeminiRestBackend, build_backend
from director_os.agent_runtime.pydantic_brain import (
    PydanticAIBackend,
    brain_name,
)
from director_os.agent_runtime.registry import ToolRegistry
from director_os.core.config import get_settings
from director_os.core.db import init_models, make_engine, make_session_factory
from director_os.otio_engine.contract import RawClipMeta
from director_os.otio_engine.timeline import (
    MASTER_TRACK,
    ClipSpec,
    TimelineSpec,
    TrackSpec,
    build_timeline,
)
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    ModelResponse,
    TextPart,
)
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import FunctionModel

GOOD_STEPS = {"steps": [{"tool": "t.split", "args": {"item_index": 0, "at_local_s": 1.5}}]}
BAD_TOOL_STEPS = {"steps": [{"tool": "hallucinated_tool", "args": {}}]}
BAD_SCHEMA_STEPS = {"steps": [{"tool": 42, "args": {}}]}


def registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool("t.split", description="split a clip")
    def split_clip(item_index: int, at_local_s: float) -> str:
        return "split"

    return reg


def plan_response(payload: dict) -> ModelResponse:
    return ModelResponse(parts=[TextPart(json.dumps(payload))])


# ------------------------------------------------------------------- propose


def test_propose_returns_structured_steps() -> None:
    seen: list[list[str]] = []

    def fn(messages, info) -> ModelResponse:  # type: ignore[no-untyped-def]
        seen.append([str(getattr(p, "content", ""))[:20] for p in messages[-1].parts])
        return plan_response(GOOD_STEPS)

    backend = PydanticAIBackend(
        "AQ.test", "gemini-test", registry(), model_factory=lambda: FunctionModel(fn)
    )
    steps = backend.propose("project=p master_items=[]", "director")
    assert [(s.tool, s.args) for s in steps] == [("t.split", {"item_index": 0, "at_local_s": 1.5})]
    assert backend.last_usage is not None
    assert seen, "FunctionModel never reached"


def test_hallucinated_tool_retries_with_feedback() -> None:
    calls: list[int] = []

    def fn(messages, info) -> ModelResponse:  # type: ignore[no-untyped-def]
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            assert calls == [1]
            return plan_response(BAD_TOOL_STEPS)
        return plan_response(GOOD_STEPS)

    backend = PydanticAIBackend(
        "AQ.test", "gemini-test", registry(), model_factory=lambda: FunctionModel(fn)
    )
    steps = backend.propose("project=p", "director")
    assert len(calls) == 2
    assert [s.tool for s in steps] == ["t.split"]


def _is_retry_feedback(messages) -> bool:  # type: ignore[no-untyped-def]
    for msg in messages:
        for part in getattr(msg, "parts", []):
            content = str(getattr(part, "content", ""))
            if "Your previous response failed" in content or "retry_feedback" in content:
                return True
    return False


def test_unknown_tool_names_are_listed_in_retry_feedback() -> None:
    def fn(messages, info) -> ModelResponse:  # type: ignore[no-untyped-def]
        if _is_retry_feedback(messages):
            feedback = " ".join(
                str(getattr(part, "content", ""))
                for msg in messages
                for part in getattr(msg, "parts", [])
            )
            assert "hallucinated_tool" in feedback
            assert "t.split" in feedback
            return plan_response(GOOD_STEPS)
        return plan_response(BAD_TOOL_STEPS)

    backend = PydanticAIBackend(
        "AQ.test", "gemini-test", registry(), model_factory=lambda: FunctionModel(fn)
    )
    assert [s.tool for s in backend.propose("project=p", "director")] == ["t.split"]


def test_schema_violation_retries() -> None:
    calls: list[int] = []

    def fn(messages, info) -> ModelResponse:  # type: ignore[no-untyped-def]
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            return plan_response(BAD_SCHEMA_STEPS)
        return plan_response(GOOD_STEPS)

    backend = PydanticAIBackend(
        "AQ.test", "gemini-test", registry(), model_factory=lambda: FunctionModel(fn)
    )
    assert [s.tool for s in backend.propose("project=p", "director")] == ["t.split"]
    assert len(calls) == 2


def test_step_budget_enforced_by_retry() -> None:
    oversized = {
        "steps": [
            {"tool": "t.split", "args": {"item_index": 0, "at_local_s": 1.5}} for _ in range(13)
        ]
    }

    def fn(messages, info) -> ModelResponse:  # type: ignore[no-untyped-def]
        if _is_retry_feedback(messages):
            return plan_response(GOOD_STEPS)
        return plan_response(oversized)

    backend = PydanticAIBackend(
        "AQ.test", "gemini-test", registry(), model_factory=lambda: FunctionModel(fn)
    )
    assert len(backend.propose("project=p", "director")) == 1


# ------------------------------------------------------------- fallback tier


def test_fallback_moves_to_next_model_on_capacity_error() -> None:
    hits: list[str] = []

    def dead(messages, info) -> ModelResponse:  # type: ignore[no-untyped-def]
        hits.append("dead")
        raise ModelHTTPError(429, model_name="gemini-test-dead", body="quota exceeded")

    def alive(messages, info) -> ModelResponse:  # type: ignore[no-untyped-def]
        hits.append("alive")
        return plan_response(GOOD_STEPS)

    model = FallbackModel(FunctionModel(dead, model_name="dead"), FunctionModel(alive))
    backend = PydanticAIBackend("AQ.test", "gemini-test", registry(), model_factory=lambda: model)
    steps = backend.propose("project=p", "director")
    assert [s.tool for s in steps] == ["t.split"]
    assert hits == ["dead", "alive"]


# ------------------------------------------------------- multi-turn protocol


def test_record_results_reach_next_proposal() -> None:
    def fn(messages, info) -> ModelResponse:  # type: ignore[no-untyped-def]
        for msg in messages:
            for part in getattr(msg, "parts", []):
                content = getattr(part, "content", "")
                if isinstance(content, str) and content.startswith("Tool results:"):
                    assert "t.split OK" in content
                    return plan_response(GOOD_STEPS)
        return plan_response(GOOD_STEPS)

    backend = PydanticAIBackend(
        "AQ.test", "gemini-test", registry(), model_factory=lambda: FunctionModel(fn)
    )
    steps = backend.propose("project=p", "director")
    backend.record_results(steps, ["noop 2"])
    assert [s.tool for s in backend.propose("project=p round 2", "director")] == ["t.split"]


def test_history_trim_keeps_valid_conversation_start() -> None:
    backend = PydanticAIBackend(
        "AQ.test",
        "gemini-test",
        registry(),
        model_factory=lambda: FunctionModel(lambda m, i: plan_response(GOOD_STEPS)),
    )
    backend.HISTORY_MAX_ENTRIES = 4
    for round_no in range(3):
        steps = backend.propose(f"project=p round={round_no}", "director")
        backend.record_results(steps, ["ok"])
    kinds = [type(m).__name__ for m in backend._history]
    assert len(kinds) <= 4
    assert kinds[0] in ("ModelRequest",)


def test_reset_history_clears_replay() -> None:
    backend = PydanticAIBackend(
        "AQ.test",
        "gemini-test",
        registry(),
        model_factory=lambda: FunctionModel(lambda m, i: plan_response(GOOD_STEPS)),
    )
    steps = backend.propose("project=p", "director")
    backend.record_results(steps, ["ok"])
    backend.reset_history()
    assert backend._history == []


# ------------------------------------------------------------- complete_json


def test_complete_json_plain_text() -> None:
    payload = {"acts": [{"kind": "quote", "start_s": 0.0, "end_s": 4.0}]}

    def fn(messages, info) -> ModelResponse:  # type: ignore[no-untyped-def]
        return ModelResponse(parts=[TextPart(json.dumps(payload))])

    backend = PydanticAIBackend(
        "AQ.test", "gemini-test", registry(), model_factory=lambda: FunctionModel(fn)
    )
    assert json.loads(backend.complete_json("sys", "user")) == payload


# ------------------------------------------------------------ PlannerExecutor


def test_planner_executor_runs_the_brain(tmp_path: Path) -> None:
    executed: list[tuple[str, dict]] = []

    def fn(messages, info) -> ModelResponse:  # type: ignore[no-untyped-def]
        return plan_response(GOOD_STEPS)

    reg = registry()

    @reg.tool("t.split", description="split a clip")
    def split_clip(item_index: int, at_local_s: float) -> str:
        executed.append(("t.split", {"item_index": item_index, "at_local_s": at_local_s}))
        return "split done"

    backend = PydanticAIBackend(
        "AQ.test", "gemini-test", reg, model_factory=lambda: FunctionModel(fn)
    )
    engine = make_engine(f"sqlite:///{tmp_path}/brain.db")
    init_models(engine)
    executor = PlannerExecutor(reg, CheckpointStore(make_session_factory(engine)), "proj_brain")
    timeline = build_timeline(
        TimelineSpec(
            name="brain_case",
            fps=30,
            width=1920,
            height=1080,
            tracks=[
                TrackSpec(
                    name=MASTER_TRACK,
                    items=[
                        ClipSpec(
                            name="raw_1",
                            source_start_s=0.0,
                            dur_s=10.0,
                            meta=RawClipMeta(asset_id="a1"),
                        )
                    ],
                )
            ],
        )
    )
    run = executor.run(backend, timeline)
    assert run.ok
    assert run.steps_ok == 1
    assert executed == [("t.split", {"item_index": 0, "at_local_s": 1.5})]


# ------------------------------------------------------------ env gate wiring


@pytest.fixture()
def clean_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("AGENT_BRAIN", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "AQ.test")
    monkeypatch.setenv("LLM_MODEL", "gemini-test")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    get_settings.cache_clear()  # type: ignore[attr-defined]
    yield monkeypatch
    get_settings.cache_clear()  # type: ignore[attr-defined]


def test_build_backend_defaults_to_pydantic_brain(clean_env: pytest.MonkeyPatch) -> None:
    backend = build_backend(registry())
    assert isinstance(backend, PydanticAIBackend)


def test_build_backend_legacy_gate(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("AGENT_BRAIN", "legacy")
    backend = build_backend(registry())
    assert isinstance(backend, GeminiRestBackend)


def test_openrouter_tier_uses_brain(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("OPENROUTER_API_KEY", "sk-or-test")
    clean_env.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    clean_env.setenv("OPENROUTER_MODEL", "test/free")
    from director_os.agent_runtime.llm import FailoverBackend, build_backend_chain

    chain = build_backend_chain(registry())
    tiers = chain._backends if isinstance(chain, FailoverBackend) else [chain]
    openrouter_tiers = [t for t in tiers if isinstance(t, PydanticAIBackend) and t.base_url]
    assert len(openrouter_tiers) == 1
    assert openrouter_tiers[0].base_url == "https://openrouter.ai/api/v1"


def test_brain_name_warns_and_falls_back_without_package(
    clean_env: pytest.MonkeyPatch, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    real_import = builtins.__import__

    def no_pydantic_ai(name: str, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        if name.startswith("pydantic_ai"):
            raise ImportError("simulated absence")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.delenv("AGENT_BRAIN", raising=False)
    monkeypatch.setattr(builtins, "__import__", no_pydantic_ai)
    assert brain_name(None) == "legacy"


def test_chain_full_stack_pydantic_tiers(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("LLM_BASE_URL", "https://gw.example/v1")
    clean_env.setenv("LLM_FALLBACK_MODELS", "qwen/qwen3.8-27b,openai/gpt-oss-20b")
    clean_env.setenv("OPENROUTER_API_KEY", "sk-or-test")
    clean_env.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    clean_env.setenv("VISION_API_KEY", "AQ.test")
    clean_env.setenv("VISION_MODEL", "gemini-test-lite")
    get_settings.cache_clear()  # type: ignore[attr-defined]
    try:
        from director_os.agent_runtime.llm import FailoverBackend, build_backend_chain

        backend = build_backend_chain(registry())
        assert isinstance(backend, FailoverBackend)
        assert backend.tier_names == [
            "pydantic_ai:gemini-test",
            "pydantic_ai:nvidia/nemotron-3-ultra:free",
            "pydantic_ai:gemini-test-lite",
        ]
        openrouter = backend._backends[1]
        assert openrouter.base_url == "https://openrouter.ai/api/v1"
        native_gemini = backend._backends[2]
        assert native_gemini.base_url == ""
    finally:
        get_settings.cache_clear()  # type: ignore[attr-defined]
