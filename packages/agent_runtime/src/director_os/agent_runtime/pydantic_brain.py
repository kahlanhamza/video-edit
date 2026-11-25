"""Pydantic AI agent brain — framework-owned proposal loop behind the LLMBackend port.

Replaces the hand-rolled propose/parse/retry/fallback pipelines in llm.py with
pydantic-ai primitives (port/adapter law, SKILL.md §1 — the LLMBackend port and
its contract are unchanged):

  - FallbackModel               -> provider/model capacity chains (the hand-rolled
                                   tier walking in GeminiRestBackend/OpenAICompatBackend)
  - output_type + ModelRetry    -> structured ToolStep plans with automatic validation
                                   feedback (the hand-rolled JSON repair/re-ask loops)
  - message_history replay      -> multi-turn tool-result protocol (the hand-rolled
                                   _history dict shaping)

Env gate (config is data): AGENT_BRAIN=pydantic_ai (default) | legacy.
build_backend() tries this brain first and falls back to the legacy backends
when the pydantic-ai package is absent or the config covers a path the brain
does not (Meta spend ledger / training gate). The legacy backends remain the
transport for MODAL fleet egress: a geo-blocked region needs AGENT_BRAIN=legacy,
because the brain calls providers directly.

Tests stay hermetic via the `model_factory` seam (FunctionModel) — same pattern
as ScriptedBackend: no network, no model downloads.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from typing import Any

from director_os.agent_runtime.agents import AgentRole, ToolStep
from director_os.agent_runtime.llm import (
    MAX_STEPS_PER_PROPOSAL,
    LLMBackendError,
    _env_model_list,
)
from director_os.agent_runtime.prompting import compose_agent_prompt
from director_os.agent_runtime.registry import ToolRegistry
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.fallback import FallbackModel


#: structured proposal contract returned by the agent's single output turn
class PlanOutput(BaseModel):
    steps: list[ToolStep] = Field(default_factory=list, max_length=MAX_STEPS_PER_PROPOSAL)


class PydanticAIBackend:
    """LLMBackend implemented on pydantic-ai (Provider tier of the config law).

    propose() runs a pydantic-ai agent whose output is `PlanOutput`; schema
    violations and hallucinated tool names come back to the model as ModelRetry
    feedback (fail-closed with feedback, the "Secure AI Agents" law), never as
    silently dropped steps. Capacity failures (429/503/refusals) on the primary
    model transparently move to the next candidate via FallbackModel.
    """

    provider = "pydantic_ai"

    HISTORY_MAX_ENTRIES = 24

    def __init__(
        self,
        api_key: str,
        model: str,
        registry: ToolRegistry,
        niche: str = "bodycam_documentary",
        base_url: str = "",
        fallback_models: list[str] | None = None,
        prompt_char_budget: int = 0,
        model_factory: Callable[[], Model] | None = None,
    ):
        if not api_key:
            raise LLMBackendError("PydanticAIBackend requires LLM_API_KEY")
        if not model:
            raise LLMBackendError("LLM_MODEL is required when LLM_API_KEY is set")
        self.api_key = api_key
        self.model = model
        self.registry = registry
        self.niche = niche
        self.base_url = base_url.rstrip("/")
        self.prompt_char_budget = prompt_char_budget
        self.last_usage: dict[str, int] | None = None
        self.last_cost_usd: float | None = None
        self.last_system_prompt = ""
        self._history: list[ModelMessage] = []
        self._model = (
            model_factory()
            if model_factory is not None
            else self._resolve_model(api_key, model, fallback_models)
        )

    def _resolve_model(self, api_key: str, model: str, fallback_models: list[str] | None) -> Model:
        candidates = [model, *(m for m in (fallback_models or []) if m != model)]
        models = [self._one_model(api_key, m) for m in candidates]
        return models[0] if len(models) == 1 else FallbackModel(*models)

    @staticmethod
    def _one_model(api_key: str, model: str, base_url: str = "") -> Model:
        if base_url:
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider

            return OpenAIChatModel(
                model,  # type: ignore[arg-type]
                provider=OpenAIProvider(base_url=base_url, api_key=api_key),
            )
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        return GoogleModel(model, provider=GoogleProvider(api_key=api_key))

    def propose(self, state_summary: str, role: AgentRole) -> list[ToolStep]:
        if self.prompt_char_budget and len(state_summary) > self.prompt_char_budget:
            state_summary = state_summary[: self.prompt_char_budget]
        system = compose_agent_prompt(
            agent=f"{role}_agent",
            role=role,
            niche=self.niche,
            state_summary=state_summary,
            registry=self.registry,
        )
        self.last_system_prompt = system
        prompt = (
            "Timeline state:\n"
            + state_summary
            + "\n\nPropose the next tool calls. Respond with a steps list — at most "
            f"{MAX_STEPS_PER_PROPOSAL}, in execution order."
        )

        def agent_for(system_prompt: str) -> Agent[None, PlanOutput]:
            return Agent(
                self._model, output_type=PlanOutput, retries=3, system_prompt=system_prompt
            )

        agent = agent_for(system)

        @agent.output_validator
        def validate_known_tools(ctx: Any, output: PlanOutput) -> PlanOutput:
            known = self.registry.names()
            for i, step in enumerate(output.steps):
                if step.tool not in known:
                    raise ModelRetry(
                        f"step {i}: unknown tool '{step.tool}'. Known tools: {known}."
                        " Re-send a corrected steps list."
                    )
            return output

        try:
            result = agent.run_sync(prompt, message_history=list(self._history) or None)
        except Exception as e:
            raise LLMBackendError(f"pydantic_ai propose failed: {type(e).__name__}: {e}") from e
        output = result.output
        if not isinstance(output, PlanOutput):
            raise LLMBackendError("pydantic_ai returned an unexpected output type")
        usage = result.usage
        self.last_usage = {
            "tokens_in": int(getattr(usage, "request_tokens", 0)),
            "tokens_out": int(getattr(usage, "response_tokens", 0)),
        }
        self._remember(result.all_messages())
        return output.steps[:MAX_STEPS_PER_PROPOSAL]

    def complete_json(self, system: str, user: str, max_tokens: int = 4096) -> str:
        """Plain-text JSON completion (Phase A macro plan — no tools)."""
        agent: Agent[None, str] = Agent(self._model, retries=2, system_prompt=system)
        try:
            result = agent.run_sync(user)
        except Exception as e:
            raise LLMBackendError(
                f"pydantic_ai complete_json failed: {type(e).__name__}: {e}"
            ) from e
        usage = result.usage
        self.last_usage = {
            "tokens_in": int(getattr(usage, "request_tokens", 0)),
            "tokens_out": int(getattr(usage, "response_tokens", 0)),
        }
        self.last_system_prompt = system
        return result.output

    def record_results(self, steps: list[ToolStep], results: list[str]) -> None:
        """Feed tool results back into the replayed conversation (port contract).

        Protocol shape: results ride the message history as an explicit
        `tool results` user turn, so the next proposal sees an alternating
        user -> model -> user flow (the bare-model-turn trap documented in
        GeminiRestBackend applies to structured output just the same).
        """
        lines = "\n".join(
            f"{step.tool} OK: {detail}" if detail else f"{step.tool} OK"
            for step, detail in zip(steps, results, strict=False)
        )
        self._history.append(
            ModelRequest(parts=[UserPromptPart(content=f"Tool results:\n{lines}")])
        )

    def reset_history(self) -> None:
        self._history.clear()

    def _remember(self, messages: Any) -> None:
        msgs: list[ModelMessage] = list(messages)
        self._history.extend(msgs)
        if len(self._history) > self.HISTORY_MAX_ENTRIES:
            del self._history[: len(self._history) - self.HISTORY_MAX_ENTRIES]
            while self._history and not isinstance(self._history[0], ModelRequest):
                del self._history[0]


def build_pydantic_brain(
    registry: ToolRegistry,
    niche: str = "bodycam_documentary",
    *,
    llm_api_key: str = "",
    llm_base_url: str = "",
    llm_model: str = "",
    prompt_char_budget: int = 0,
    model_factory: Callable[[], Model] | None = None,
) -> PydanticAIBackend:
    """Env-driven PydanticAIBackend factory (mirrors build_backend()."""
    from director_os.core.config import get_settings

    s = get_settings()
    api_key = llm_api_key or s.llm_api_key
    base_url = (llm_base_url or s.llm_base_url).strip()
    model = llm_model or s.llm_model
    if not api_key:
        raise LLMBackendError(
            "LLM not configured: set LLM_API_KEY (and optionally LLM_BASE_URL/LLM_MODEL)"
        )
    if not model:
        raise LLMBackendError("LLM_MODEL is required when LLM_API_KEY is set")
    return PydanticAIBackend(
        api_key,
        model,
        registry,
        niche,
        base_url=base_url,
        fallback_models=_env_model_list(s.llm_fallback_models) or None,
        prompt_char_budget=prompt_char_budget,
        model_factory=model_factory,
    )


def brain_name(settings: Any) -> str:
    """Which brain build_backend() should assemble ('pydantic_ai' | 'legacy')."""
    raw = (os.getenv("AGENT_BRAIN", "pydantic_ai") or "").strip().lower()
    if raw == "legacy":
        return "legacy"
    try:
        import pydantic_ai  # noqa: F401
    except ImportError:
        warnings.warn(
            "AGENT_BRAIN=pydantic_ai but pydantic-ai is not installed; using legacy", stacklevel=1
        )
        return "legacy"
    return "pydantic_ai"


__all__ = ["PlanOutput", "PydanticAIBackend", "brain_name", "build_pydantic_brain"]
