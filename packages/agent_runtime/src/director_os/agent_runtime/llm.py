"""Real LLM backends for the native tool-calling loop (M2 close-out, PLAN §3.1).

Two providers behind the existing `LLMBackend` port (`agents.py`), so the
PlannerExecutor loop, checkpoint-first dispatch and token accounting are
provider-agnostic:

- `GeminiRestBackend`  — native Google AI Studio (Generative Language REST,
  `x-goog-api-key`). Function-calling via `functionDeclarations`; the registry's
  OpenAI schemas map 1:1.
- `OpenAICompatBackend` — any OpenAI-compatible `/chat/completions` endpoint
  (vLLM, OpenRouter, GLM, a Gemini-to-OpenAI gateway) via `LLM_BASE_URL`.

`build_backend()` picks per env: `LLM_BASE_URL` set -> OpenAI-compatible;
blank -> native Gemini. No heavyweight SDK: httpx only, so CI stays hermetic
(`ScriptedBackend` covers offline tests).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from director_os.agent_runtime.agents import AgentRole, ToolStep
from director_os.agent_runtime.prompting import compose_agent_prompt
from director_os.agent_runtime.registry import ToolRegistry

GEMINI_DEFAULT_BASE = "https://generativelanguage.googleapis.com"
PROPOSE_TIMEOUT_S = 90.0
MAX_STEPS_PER_PROPOSAL = 12


class LLMBackendError(RuntimeError):
    pass


def _clean_schema(parameters: dict[str, Any]) -> dict[str, Any]:
    """Strip JSON-Schema keys Gemini rejects ($defs, additionalProperties...)."""
    drop = {"additionalProperties", "$defs", "$schema", "title"}
    out: dict[str, Any] = {}
    for k, v in parameters.items():
        if k in drop:
            continue
        if isinstance(v, dict):
            out[k] = _clean_schema(v)
        elif isinstance(v, list):
            out[k] = [_clean_schema(i) if isinstance(i, dict) else i for i in v]
        else:
            out[k] = v
    return out


def _declarations(registry: ToolRegistry) -> list[dict[str, Any]]:
    return [
        {
            "name": f["function"]["name"],
            "description": f["function"]["description"],
            "parameters": _clean_schema(f["function"]["parameters"]),
        }
        for f in registry.openai_schemas()
    ]


class GeminiRestBackend:
    """Native Google AI Studio function-calling backend (LLM_API_KEY)."""

    provider = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str,
        registry: ToolRegistry,
        niche: str = "bodycam_documentary",
        base_url: str = GEMINI_DEFAULT_BASE,
        transport: httpx.BaseTransport | None = None,
    ):
        if not api_key:
            raise LLMBackendError("GeminiRestBackend requires LLM_API_KEY")
        self.api_key = api_key
        self._transport = transport
        self.model = model
        self.registry = registry
        self.niche = niche
        self.base_url = base_url.rstrip("/")
        self.last_usage: dict[str, int] | None = None
        self.last_system_prompt = ""

    def propose(self, state_summary: str, role: AgentRole) -> list[ToolStep]:
        self.last_system_prompt = compose_agent_prompt(
            agent=f"{role}_agent",
            role=role,
            niche=self.niche,
            state_summary=state_summary,
            registry=self.registry,
        )
        payload: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": self.last_system_prompt}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": "Timeline state:\n"
                            + state_summary
                            + "\n\nPropose the next tool calls. Respond ONLY with "
                            "function calls against the provided tools — at most "
                            f"{MAX_STEPS_PER_PROPOSAL}, in execution order."
                        }
                    ],
                }
            ],
            "tools": [{"functionDeclarations": _declarations(self.registry)}],
            "generationConfig": {"temperature": 0.4, "maxOutputTokens": 2048},
        }
        url = f"{self.base_url}/v1beta/models/{self.model}:generateContent"
        try:
            with httpx.Client(transport=self._transport, timeout=PROPOSE_TIMEOUT_S) as client:
                r = client.post(url, headers={"x-goog-api-key": self.api_key}, json=payload)
        except httpx.HTTPError as e:
            raise LLMBackendError(f"gemini request failed: {e}") from e
        if r.status_code != 200:
            raise LLMBackendError(f"gemini {r.status_code}: {r.text[:300]}")
        data = r.json()
        usage = data.get("usageMetadata", {})
        self.last_usage = {
            "tokens_in": int(usage.get("promptTokenCount", 0)),
            "tokens_out": int(usage.get("candidatesTokenCount", 0)),
        }
        steps: list[ToolStep] = []
        for part in data.get("candidates", [{}])[0].get("content", {}).get("parts", []):
            call = part.get("functionCall")
            if not call:
                continue
            name = call.get("name", "")
            if name in self.registry.names():
                steps.append(
                    ToolStep(
                        tool=name,
                        args=call.get("args") or {},
                        checkpoint_note=f"{self.provider}:{self.model} proposal",
                    )
                )
            if len(steps) >= MAX_STEPS_PER_PROPOSAL:
                break
        return steps


class OpenAICompatBackend:
    """OpenAI-compatible /chat/completions backend (LLM_BASE_URL route)."""

    provider = "openai_compat"

    def __init__(
        self,
        api_key: str,
        model: str,
        registry: ToolRegistry,
        niche: str = "bodycam_documentary",
        base_url: str = "https://api.openai.com/v1",
        transport: httpx.BaseTransport | None = None,
    ):
        self.api_key = api_key
        self._transport = transport
        self.model = model
        self.registry = registry
        self.niche = niche
        self.base_url = base_url.rstrip("/")
        self.last_usage: dict[str, int] | None = None
        self.last_system_prompt = ""

    def propose(self, state_summary: str, role: AgentRole) -> list[ToolStep]:
        self.last_system_prompt = compose_agent_prompt(
            agent=f"{role}_agent",
            role=role,
            niche=self.niche,
            state_summary=state_summary,
            registry=self.registry,
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.last_system_prompt},
                {
                    "role": "user",
                    "content": "Timeline state:\n"
                    + state_summary
                    + "\n\nPropose the next tool calls (max "
                    f"{MAX_STEPS_PER_PROPOSAL}) via tool_calls.",
                },
            ],
            "tools": self.registry.openai_schemas(),
            "tool_choice": "auto",
            "temperature": 0.4,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            with httpx.Client(transport=self._transport, timeout=PROPOSE_TIMEOUT_S) as client:
                r = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
        except httpx.HTTPError as e:
            raise LLMBackendError(f"openai-compat request failed: {e}") from e
        if r.status_code != 200:
            raise LLMBackendError(f"openai-compat {r.status_code}: {r.text[:300]}")
        data = r.json()
        usage = data.get("usage", {})
        self.last_usage = {
            "tokens_in": int(usage.get("prompt_tokens", 0)),
            "tokens_out": int(usage.get("completion_tokens", 0)),
        }
        msg = data.get("choices", [{}])[0].get("message", {})
        steps: list[ToolStep] = []
        known = self.registry.names()
        for call in msg.get("tool_calls") or []:
            fn = call.get("function", {})
            name = fn.get("name", "")
            if name not in known:
                continue
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            steps.append(
                ToolStep(
                    tool=name,
                    args=args,
                    checkpoint_note=f"{self.provider}:{self.model} proposal",
                )
            )
            if len(steps) >= MAX_STEPS_PER_PROPOSAL:
                break
        return steps


def build_backend(
    registry: ToolRegistry,
    niche: str = "bodycam_documentary",
    *,
    llm_api_key: str = "",
    llm_base_url: str = "",
    llm_model: str = "",
) -> GeminiRestBackend | OpenAICompatBackend:
    """Env-driven provider choice (ADR: config is data).

    LLM_BASE_URL non-blank -> OpenAI-compatible gateway; blank -> native Gemini
    SDK endpoint. Raises (not falls back) when unconfigured — callers decide
    whether to degrade to ScriptedBackend.
    """
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
    if base_url:
        return OpenAICompatBackend(api_key, model, registry, niche, base_url=base_url)
    return GeminiRestBackend(api_key, model, registry, niche)
