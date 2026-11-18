"""Real LLM backends for the native tool-calling loop (M2 close-out, PLAN §3.1).

Two providers behind the existing `LLMBackend` port (`agents.py`), so the
PlannerExecutor loop, checkpoint-first dispatch and token accounting are
provider-agnostic:

- `GeminiRestBackend`  — native Google AI Studio (Generative Language REST,
  `x-goog-api-key`). Function-calling via `functionDeclarations` sending the
  registry's Pydantic schemas through `parametersJsonSchema` (full JSON Schema
  support), assistant turns replayed verbatim across follow-up calls.

  Gemini 3.x REST traps handled here (each pinned by a unit test):
  1. `thoughtSignature` — the model attaches this opaque field to the first
     `functionCall` part. It is NOT representable on the normalized `ToolStep`,
     so it rides along in `ToolStep.provider_meta` while the raw assistant turn
     is replayed untouched in `contents` on turn 2+. Dropping it => HTTP 400
     `missing thought_signature`.
  2. `parametersJsonSchema` instead of `parameters` — the legacy field parses
     into an OpenAPI 3.0 protobuf subset and rejects `additionalProperties`,
     `$defs`/`$ref`, `anyOf` with HTTP 400. `parametersJsonSchema` accepts the
     full JSON Schema, so Pydantic v2 output is passed through unmodified.
  3. AQ. key 401s — if `x-goog-api-key` returns 401
     `ACCESS_TOKEN_TYPE_UNSUPPORTED` (AI Studio key quirk), the request is
     retried once with `?key=` as a query parameter. Also note for ops: if you
     restrict the key in Cloud Console, pick the **Generative Language API**
     (not "Gemini API", a known GCP dropdown trap).
- `OpenAICompatBackend` — any OpenAI-compatible `/chat/completions` endpoint
  (vLLM, OpenRouter, GLM, a Gemini-to-OpenAI gateway) via `LLM_BASE_URL`.
- `MetaModelBackend` (`meta.py`) — Meta Model API (Muse) at `api.meta.ai`:
  same wire shape plus cached-token parsing, USD cost estimation, a persistent
  spend ledger and a hard budget cap. Selected when the base URL host is
  `api.meta.ai` or `LLM_PROVIDER=meta`.

`build_backend()` picks per env: `LLM_BASE_URL` set -> OpenAI-compatible
(Meta-flavoured when it points at api.meta.ai); blank -> native Gemini.
No heavyweight SDK: httpx only, so CI stays hermetic (`ScriptedBackend`
covers offline tests).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx
from director_os.agent_runtime.agents import AgentRole, ToolStep
from director_os.agent_runtime.prompting import compose_agent_prompt
from director_os.agent_runtime.registry import ToolRegistry

if TYPE_CHECKING:
    from director_os.agent_runtime.meta import MetaModelBackend

GEMINI_DEFAULT_BASE = "https://generativelanguage.googleapis.com"
PROPOSE_TIMEOUT_S = 90.0
MAX_STEPS_PER_PROPOSAL = 12


class LLMBackendError(RuntimeError):
    pass


def _declarations(registry: ToolRegistry) -> list[dict[str, Any]]:
    """functionDeclarations for Gemini 3.x REST (trap #2).

    The legacy `parameters` field parses into an OpenAPI 3.0 protobuf subset
    and 400s on Pydantic v2 output (`additionalProperties`, `$defs`/`$ref`,
    `anyOf`). `parametersJsonSchema` supports the full JSON Schema, so the
    registry schema passes through untouched — no lossy stripping.
    """
    return [
        {
            "name": f["function"]["name"],
            "description": f["function"]["description"],
            "parametersJsonSchema": f["function"]["parameters"],
        }
        for f in registry.openai_schemas()
    ]


class GeminiRestBackend:
    """Native Google AI Studio function-calling backend (LLM_API_KEY).

    Multi-turn capable: every `propose()` call on the same instance replays the
    conversation so far (`self._history`). Assistant turns go back in verbatim
    so Gemini 3.x `thoughtSignature` fields survive byte-for-byte (trap #1);
    tool results are fed back via `record_results()` as functionResponse parts.
    """

    provider = "gemini"

    # bound on replayed conversation turns (user + model entries)
    HISTORY_MAX_ENTRIES = 24

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
        self._history: list[dict[str, Any]] = []

    def _post(self, client: httpx.Client, url: str, payload: dict[str, Any]) -> httpx.Response:
        """POST with x-goog-api-key, plus the trap #3 fallback.

        Some AI Studio AQ. keys reject the `x-goog-api-key` header with 401
        ACCESS_TOKEN_TYPE_UNSUPPORTED; the `?key=` query-param form resolves
        it. Retried once with the query param (header dropped).
        """
        try:
            r = client.post(url, headers={"x-goog-api-key": self.api_key}, json=payload)
        except httpx.HTTPError as e:
            raise LLMBackendError(f"gemini request failed: {e}") from e
        if r.status_code == 401 and "ACCESS_TOKEN_TYPE_UNSUPPORTED" in r.text:
            try:
                r = client.post(url, params={"key": self.api_key}, json=payload)
            except httpx.HTTPError as e:
                raise LLMBackendError(f"gemini request failed: {e}") from e
        return r

    def _remember(self, entry: dict[str, Any]) -> None:
        self._history.append(entry)
        if len(self._history) > self.HISTORY_MAX_ENTRIES:
            del self._history[: len(self._history) - self.HISTORY_MAX_ENTRIES]

    def reset_history(self) -> None:
        """Drop replayed turns (new conversation on the next propose)."""
        self._history.clear()

    def record_results(self, steps: list[ToolStep], results: list[str]) -> None:
        """Append tool results as functionResponse parts (Gemini shape).

        Pairs with the replayed assistant `functionCall` turn so the next
        `propose()` is a valid multi-step tool conversation. Optional port
        extension — callers duck-type (`hasattr(backend, "record_results")`),
        so PlannerExecutor stays provider-agnostic.
        """
        parts = [
            {
                "functionResponse": {
                    "name": step.tool,
                    "response": {"result": str(res)[:4000]},
                }
            }
            for step, res in zip(steps, results, strict=False)
        ]
        if parts:
            self._remember({"role": "user", "parts": parts})

    def propose(self, state_summary: str, role: AgentRole) -> list[ToolStep]:
        self.last_system_prompt = compose_agent_prompt(
            agent=f"{role}_agent",
            role=role,
            niche=self.niche,
            state_summary=state_summary,
            registry=self.registry,
        )
        user_turn = {
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
        payload: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": self.last_system_prompt}]},
            "contents": [*self._history, user_turn],
            "tools": [{"functionDeclarations": _declarations(self.registry)}],
            "generationConfig": {"temperature": 0.4, "maxOutputTokens": 2048},
        }
        url = f"{self.base_url}/v1beta/models/{self.model}:generateContent"
        try:
            with httpx.Client(transport=self._transport, timeout=PROPOSE_TIMEOUT_S) as client:
                r = self._post(client, url, payload)
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
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        # Trap #1: replay the assistant turn VERBATIM — thoughtSignature rides
        # on the first functionCall part and must not be re-serialized through
        # the normalized ToolStep. Normalized copies live on ToolStep.provider_meta
        # for callers that need them.
        if parts:
            self._remember({"role": "model", "parts": parts})
        steps: list[ToolStep] = []
        for part in parts:
            call = part.get("functionCall")
            if not call:
                continue
            name = call.get("name", "")
            if name in self.registry.names():
                meta: dict[str, Any] = {}
                if part.get("thoughtSignature"):
                    meta["thoughtSignature"] = part["thoughtSignature"]
                if call.get("id"):
                    meta["call_id"] = call["id"]
                steps.append(
                    ToolStep(
                        tool=name,
                        args=call.get("args") or {},
                        checkpoint_note=f"{self.provider}:{self.model} proposal",
                        provider_meta=meta or None,
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

    def _payload(self, state_summary: str, role: AgentRole) -> dict[str, Any]:
        self.last_system_prompt = compose_agent_prompt(
            agent=f"{role}_agent",
            role=role,
            niche=self.niche,
            state_summary=state_summary,
            registry=self.registry,
        )
        return {
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

    def _extra_payload(self) -> dict[str, Any]:
        """Provider-optional request fields (merged after the vanilla payload)."""
        return {}

    def _absorb_usage(self, usage: dict[str, Any]) -> None:
        """Extract token accounting from the response usage object."""
        self.last_usage = {
            "tokens_in": int(usage.get("prompt_tokens", 0)),
            "tokens_out": int(usage.get("completion_tokens", 0)),
        }

    def propose(self, state_summary: str, role: AgentRole) -> list[ToolStep]:
        payload = self._payload(state_summary, role)
        extras = self._extra_payload()
        payload.update(extras)
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            with httpx.Client(transport=self._transport, timeout=PROPOSE_TIMEOUT_S) as client:
                r = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                if r.status_code == 400 and extras:
                    # a provider rejected an optional extra (e.g. a reasoning-effort
                    # naming variant): retry once with the vanilla payload
                    vanilla = {k: v for k, v in payload.items() if k not in extras}
                    r = client.post(
                        f"{self.base_url}/chat/completions", headers=headers, json=vanilla
                    )
        except httpx.HTTPError as e:
            raise LLMBackendError(f"openai-compat request failed: {e}") from e
        if r.status_code != 200:
            raise LLMBackendError(f"openai-compat {r.status_code}: {r.text[:300]}")
        data = r.json()
        self._absorb_usage(data.get("usage", {}))
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
    llm_provider: str = "",
) -> GeminiRestBackend | OpenAICompatBackend:
    """Env-driven provider choice (ADR: config is data).

    LLM_BASE_URL non-blank -> OpenAI-compatible gateway (Meta-flavoured when
    the host is api.meta.ai or LLM_PROVIDER=meta); blank -> native Gemini SDK
    endpoint (or Meta when LLM_PROVIDER=meta). Raises (not falls back) when
    unconfigured — callers decide whether to degrade to ScriptedBackend.
    """
    from director_os.agent_runtime.meta import META_DEFAULT_BASE
    from director_os.core.config import get_settings

    s = get_settings()
    api_key = llm_api_key or s.llm_api_key
    base_url = (llm_base_url or s.llm_base_url).strip()
    model = llm_model or s.llm_model
    provider = (llm_provider or s.llm_provider).strip().lower()
    if not api_key:
        raise LLMBackendError(
            "LLM not configured: set LLM_API_KEY (and optionally LLM_BASE_URL/LLM_MODEL)"
        )
    if not model:
        raise LLMBackendError("LLM_MODEL is required when LLM_API_KEY is set")
    if base_url:
        if provider == "meta" or "api.meta.ai" in base_url:
            return _meta_backend(api_key, model, registry, niche, base_url)
        return OpenAICompatBackend(api_key, model, registry, niche, base_url=base_url)
    if provider == "meta":
        return _meta_backend(api_key, model, registry, niche, META_DEFAULT_BASE)
    return GeminiRestBackend(api_key, model, registry, niche)


def _meta_backend(
    api_key: str, model: str, registry: ToolRegistry, niche: str, base_url: str
) -> MetaModelBackend:
    """Assemble the Meta backend with its spend ledger + budget guard."""
    # lazy: meta.py imports this module (OpenAICompatBackend) at module scope
    from director_os.agent_runtime.meta import MetaModelBackend, SpendLedger
    from director_os.core.config import get_settings

    s = get_settings()
    if "contributor" in model and not s.llm_allow_training_on_content:
        raise LLMBackendError(
            f"{model} lets Meta train on the content you send. Bodycam footage carries "
            "PII — use the standard tier (e.g. muse-ember-1.0), or set "
            "LLM_ALLOW_TRAINING_ON_CONTENT=1 to accept explicitly."
        )
    ledger_path = s.llm_spend_ledger or str(s.output_dir / "llm_spend.jsonl")
    return MetaModelBackend(
        api_key,
        model,
        registry,
        niche,
        base_url=base_url,
        ledger=SpendLedger(ledger_path),
        budget_usd=s.llm_budget_usd,
    )
