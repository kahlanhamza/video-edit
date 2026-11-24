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
import os
import time
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
# Capacity routing: a demand spike (429/503 UNAVAILABLE) on the primary model
# downgrades to the next candidate FOR THE BACKEND'S LIFETIME — availability
# beats headliner. Chain order is evidence-backed (fleet probe): the 3.7/3.8
# flash tier burns its free-tier quota fast; the flash-lite tiers serve.
DEFAULT_FALLBACK_MODELS = (
    "gemini-3.7-flash,gemini-3.5-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite"
)
# Groq free-tier TPM ceiling: keep the composed prompt inside ~6k chars so a
# Director proposal never trips the 8k tokens-per-minute window (SKILL: wire facts).
GROQ_PROMPT_CHAR_BUDGET = 6000
# HTTP statuses that mean "this model/tier is out of capacity right now" —
# walk the chain instead of failing the stage.
CAPACITY_STATUSES = frozenset({429, 500, 503})
# Strings an openai-compat gateway returns when the MODEL NAME itself is dead
# (decommissioned / typo / not provisioned / rotated out of the catalog) — skip
# the candidate, not the stage. 404 on /chat/completions = same meaning.
_MODEL_NOT_FOUND_MARKERS = (
    "model_not_found",
    "does not exist",
    "decommissioned",
    "no longer supported",
    "not a valid model",
    "unknown model",
    "invalid model",
    "no endpoints found",
)
# Signatures a tier raises when it is UNAVAILABLE (cross-tier failover may
# proceed): capacity 429/500/503, dead key 401/403, tier exhaustion. A 400
# contract error carries no marker and re-raises — failover is for
# availability, never for masking bugs.
_CAPACITY_ERROR_MARKERS = (
    "429",
    "rate limit",
    "too many requests",
    "quota",
    "capacity exhausted",
    "503",
    "500",
    "401",
    "403",
    "unauthorized",
    "forbidden",
)


class LLMBackendError(RuntimeError):
    pass


def strip_code_fences(text: str) -> str:
    """Lenient JSON extraction: drop ```json fences, keep the outermost object."""
    import re

    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text.strip())
    first, last = text.find("{"), text.rfind("}")
    if first >= 0 and last > first:
        return text[first : last + 1]
    return text


def extract_json_payload(text: str, expect: str = "object") -> Any:
    """Notebook-grade lenient JSON extraction (the Phase A char-0 failure class).

    Reasoning models wrap the payload in <think>...</think>; proxies prepend
    prose or wrap it in ```json fences. Order of operations (the manual sandbox
    workflow's `extract_json`): strip think blocks, strip fences, then slice
    from the first `{`/`[` to the last `}`/`]` and json.loads. Raises ValueError
    with the body head quoted when nothing parseable remains.
    """
    import json
    import re

    cleaned = re.sub(r"<(?:think|thought)>[\s\S]*?</(?:think|thought)>", "", text or "")
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned.strip())
    open_ch, close_ch = ("{", "}") if expect == "object" else ("[", "]")
    a, b = cleaned.find(open_ch), cleaned.rfind(close_ch)
    if a == -1 or b == -1 or b <= a:
        head = " ".join(cleaned.split())[:160] or "(empty response)"
        raise ValueError(f"no JSON {expect} in response — got: {head!r}")
    payload = cleaned[a : b + 1]
    try:
        return json.loads(payload)
    except json.JSONDecodeError as first:
        # Last-chance repair for the LLM syntax slips the strict parse hates
        # (live E2E: `Expecting ',' delimiter` on EVERY Phase A retry — the
        # model stacked object keys without a comma). The surgery is
        # conservative and only accepted when json.loads validates it.
        repaired = repair_common_llm_json(payload)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            raise first from first


def repair_common_llm_json(text: str) -> str:
    """The two LLM JSON crimes json.loads cannot forgive: smart quotes and
    stacked values with the comma forgotten between them (objects AND arrays).
    Only patterns that are ALWAYS invalid in JSON are touched."""
    import re as _re

    smart = {"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"}
    fixed = text
    for bad, good in smart.items():
        fixed = fixed.replace(bad, good)
    fixed = _re.sub(r",\s*([}\]])", r"\1", fixed)
    # bare newline between value-end token and value-start token → comma
    fixed = _re.sub(r'(["\}\]\w])\s*\n(\s*["\{\[])', r"\1,\n\2", fixed)
    return fixed


def _modal_vlm_fn():
    """Seam for the deployed fleet proxy (lazy `modal` import keeps CI hermetic)."""
    import modal

    return modal.Function.from_name("directoros-vlm", "vlm_call")


def _env_model_list(raw: str) -> list[str]:
    return [m.strip() for m in raw.split(",") if m.strip()]


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
        fallback_models: list[str] | None = None,
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
        # Geo egress: some datacenter regions get 400 FAILED_PRECONDITION
        # ("User location is not supported") from the Generative Language API.
        # On the first geo-block we transparently switch to the Modal fleet's
        # `vlm_call` proxy (US/EU egress) and the decision sticks for the
        # backend's lifetime — same contract as perception.vision.
        self.modal_egress = True
        self._egress_active: str = "direct"
        self._fallbacks: list[str] = (
            [m for m in fallback_models if m != model]
            if fallback_models is not None
            else [
                m
                for m in _env_model_list(
                    os.getenv("GEMINI_FALLBACK_MODELS", DEFAULT_FALLBACK_MODELS)
                )
                if m != model
            ]
        )

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
            # the trim must never open the conversation with a bare model turn
            while self._history and self._history[0].get("role") != "user":
                del self._history[0]

    def _generate(self, payload: dict[str, Any], model_path: str = "") -> dict[str, Any]:
        """Send one generateContent with unified GEO + CAPACITY routing.

        Walk the model chain (primary + fallbacks); per model: try the direct
        endpoint first (geo-block switches egress to the fleet, STICKY), and
        on key-level capacity errors (429/500/503) move to the next model.
        Once the fleet is the active egress, every remaining model goes
        through it. The model switch is sticky — a quota window does not
        un-downgrade mid-conversation.
        """
        models = [self.model, *self._fallbacks]
        last_status, last_body = 0, ""
        for candidate in models:
            self.model = candidate
            url = f"{self.base_url}/v1beta/models/{candidate}:generateContent"
            if self._egress_active == "direct":
                try:
                    with httpx.Client(
                        transport=self._transport, timeout=PROPOSE_TIMEOUT_S
                    ) as client:
                        r = self._post(client, url, payload)
                except httpx.HTTPError as e:
                    raise LLMBackendError(f"gemini request failed: {e}") from e
                if r.status_code == 200:
                    return r.json()
                last_status, last_body = r.status_code, r.text
                if self.modal_egress and "location is not supported" in r.text:
                    self._egress_active = "modal"  # sticky; same model rides the fleet
                elif r.status_code not in (429, 500, 503):
                    raise LLMBackendError(f"gemini {r.status_code}: {r.text[:300]}")
                else:
                    continue  # capacity on this model — try the next candidate
            out = self._fleet_call(json.dumps(payload))
            if out["status"] == 200:
                return json.loads(out["body"])
            last_status, last_body = out["status"], out["body"]
            if last_status not in (429, 500, 503):
                raise LLMBackendError(f"gemini via modal egress {last_status}: {last_body[:300]}")
        raise LLMBackendError(
            f"gemini capacity exhausted across {len(models)} model(s) "
            f"via {self._egress_active} egress: {last_status} {last_body[:200]}"
        )

    def _fleet_call(self, body: str) -> dict[str, Any]:
        """One model through the fleet proxy; transient statuses back off x3."""
        import time as _time

        from director_os.core.config import get_settings

        settings = get_settings()
        if not (settings.modal_token_id and settings.modal_token_secret):
            raise LLMBackendError(
                "Gemini geo-blocked from this region and MODAL_TOKEN_ID/SECRET "
                "not configured for fleet egress"
            )
        fn = _modal_vlm_fn()
        status, out_body = 0, ""
        for attempt in range(3):
            try:
                out = fn.remote(model=self.model, payload_json=body)
            except Exception as e:  # noqa: BLE001 — fleet-side timeout/transport = retryable
                status, out_body = 503, f"fleet transport error: {e}"
            else:
                status, out_body = int(out.get("status", 0)), str(out.get("body", ""))
            if status not in (429, 500, 503):
                return {"status": status, "body": out_body}
            _time.sleep(4.0 * (attempt + 1))
        return {"status": status, "body": out_body}

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
        data = self._generate(payload)
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
        #
        # Protocol shape: the USER turn is remembered too. Without it, round
        # 2's contents START with the bare model(functionCall) turn — Gemini
        # 400s with "function call turn comes immediately after a user turn
        # or after a function response turn" (caught live by the Eustis
        # autodrive). History becomes user -> model -> user(fr) -> user -> ...
        self._remember(user_turn)
        if parts:
            self._remember({"role": "model", "parts": parts})
        steps: list[ToolStep] = []
        for part in parts:
            call = part.get("functionCall")
            if not call:
                continue
            name = call.get("name", "")
            # FAIL-CLOSED WITH FEEDBACK (Google "Secure AI Agents" law): a
            # hallucinated tool name is NOT silently dropped — it becomes a
            # step the registry rejects with the known-tool list, and the
            # executor feeds that error back through record_results so the
            # model learns the real toolbox in the next round.
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

    def complete_json(self, system: str, user: str, max_tokens: int = 4096) -> str:
        """Plain-text JSON completion (Phase A macro plan — no tools).

        Walks the tier's model fallbacks: a geo/capacity failure on one
        model downgrades to the next before the tier ever raises.
        """
        payload: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": max_tokens},
        }
        last_err = ""
        for candidate in [self.model, *self._fallbacks]:
            self.model = candidate
            try:
                data = self._generate(payload, model_path=candidate)
            except LLMBackendError as e:
                last_err = str(e)
                continue
            usage = data.get("usageMetadata", {})
            self.last_usage = {
                "tokens_in": int(usage.get("promptTokenCount", 0)),
                "tokens_out": int(usage.get("candidatesTokenCount", 0)),
            }
            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            # `thought` parts are reasoning summaries, NOT the completion —
            # joining them prepends prose to the JSON payload and Phase A's
            # strict parse died on every retry (live E2E evidence)
            return "".join(str(p.get("text", "")) for p in parts if not p.get("thought"))
        raise LLMBackendError(f"gemini complete_json exhausted fallbacks: {last_err[:200]}")


class OpenAICompatBackend:
    """OpenAI-compatible /chat/completions backend (LLM_BASE_URL route).

    Capacity routing: on 429 the backend HONORS the ``Retry-After`` header
    (capped by ``retry_after_max_s``) and retries the same model once before
    downgrading to the next candidate in ``LLM_FALLBACK_MODELS``; a dead model
    name (400 model_not_found family) skips straight to the next candidate.
    Only when every candidate is exhausted does the tier raise — the
    ``FailoverBackend`` then hands over to the next provider tier.

    ``prompt_char_budget`` (Groq: 6000) clamps the state summary so a proposal
    fits inside Groq's ~8k TPM window.

    Multi-turn (M5 story-map Director): every propose() replays the
    conversation so far (`self._history`) — assistant tool_calls verbatim
    (ids must survive), tool results via `record_results()` as
    ``role="tool"`` messages. `complete_json()` runs the same capacity
    ladder WITHOUT tools for the Phase A macro plan.
    """

    provider = "openai_compat"

    def __init__(
        self,
        api_key: str,
        model: str,
        registry: ToolRegistry,
        niche: str = "bodycam_documentary",
        base_url: str = "https://api.openai.com/v1",
        transport: httpx.BaseTransport | None = None,
        fallback_models: list[str] | None = None,
        prompt_char_budget: int = 0,
        retry_after_max_s: float = 20.0,
    ):
        self.api_key = api_key
        self._transport = transport
        self.model = model
        self.registry = registry
        self.niche = niche
        self.base_url = base_url.rstrip("/")
        self.last_usage: dict[str, int] | None = None
        self.last_system_prompt = ""
        self.prompt_char_budget = prompt_char_budget
        self.retry_after_max_s = retry_after_max_s
        self._fallbacks: list[str] = (
            [m for m in fallback_models if m != model]
            if fallback_models is not None
            else _env_model_list(os.getenv("LLM_FALLBACK_MODELS", ""))
        )
        self._history: list[dict[str, Any]] = []

    def reset_history(self) -> None:
        """Drop replayed turns (new conversation on the next propose)."""
        self._history.clear()

    def record_results(self, steps: list[ToolStep], results: list[str]) -> None:
        """Append tool results as ``role="tool"`` messages (OpenAI shape)."""
        for step, res in zip(steps, results, strict=False):
            call_id = (step.provider_meta or {}).get("call_id") or f"call_{len(self._history)}"
            self._history.append(
                {"role": "tool", "tool_call_id": call_id, "content": str(res)[:4000]}
            )

    def _sanitize_schemas(self) -> list[dict[str, Any]]:
        """OpenAI-compat tool schemas with DOT-FREE names.

        Wire fact (OpenRouter/Nvidia NIM, 2026-11-19): function names may only
        contain a-z A-Z 0-9 _ - — the namespaced `timeline.trim_clip` style is
        rejected with 400 before the model even runs. We mangle '.' -> '__' on
        the way out and un-mangle on the way in, so the registry keeps its
        canonical dotted names while every gateway accepts the request.
        """
        out: list[dict[str, Any]] = []
        for schema in self.registry.openai_schemas():
            fn = dict(schema.get("function", {}))
            if fn.get("name"):
                fn["name"] = str(fn["name"]).replace(".", "__")
            out.append({**schema, "function": fn})
        return out

    def _payload(self, state_summary: str, role: AgentRole) -> dict[str, Any]:
        if self.prompt_char_budget and len(state_summary) > self.prompt_char_budget:
            state_summary = (
                state_summary[: self.prompt_char_budget]
                + "\n…(state summary clamped to the provider TPM budget)"
            )
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
                # multi-turn: replay the conversation so far (assistant
                # tool_calls verbatim + role=tool results) between system and
                # the fresh user turn — the Phase B refinement loop depends
                # on the model seeing its own previous edits
                *self._history,
                {
                    "role": "user",
                    "content": "Timeline state:\n"
                    + state_summary
                    + "\n\nPropose the next tool calls (max "
                    f"{MAX_STEPS_PER_PROPOSAL}) via tool_calls.",
                },
            ],
            "tools": self._sanitize_schemas(),
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

    def _model_not_found(self, r: httpx.Response) -> bool:
        if r.status_code == 404:
            return True  # unknown model id on a well-formed request
        return r.status_code == 400 and any(m in r.text.lower() for m in _MODEL_NOT_FOUND_MARKERS)

    def _post_candidate(self, payload: dict[str, Any], headers: dict[str, str]) -> httpx.Response:
        try:
            with httpx.Client(transport=self._transport, timeout=PROPOSE_TIMEOUT_S) as client:
                r = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                if r.status_code == 400 and self._extra_payload():
                    # a provider rejected an optional extra (e.g. a reasoning-effort
                    # naming variant): retry once with the vanilla payload
                    r = client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=self._vanilla_payload(payload),
                    )
        except httpx.HTTPError as e:
            raise LLMBackendError(f"openai-compat request failed: {e}") from e
        return r

    def _vanilla_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        extras = self._extra_payload()
        return {k: v for k, v in payload.items() if k not in extras}

    def propose(self, state_summary: str, role: AgentRole) -> list[ToolStep]:
        """Walk [model, *fallbacks]; honor 429 Retry-After, skip dead models."""
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        candidates = [self.model, *self._fallbacks]
        last_status, last_body = 0, ""
        for candidate in candidates:
            self.model = candidate
            payload = self._payload(state_summary, role)
            payload.update(self._extra_payload())
            r = self._post_candidate(payload, headers)
            if r.status_code == 200:
                return self._absorb_proposal(r)
            last_status, last_body = r.status_code, r.text
            if r.status_code == 429:
                # honor the provider's window: sleep Retry-After (capped), retry
                # the same model once, then downgrade to the next candidate
                delay = self._retry_after_s(r)
                time.sleep(delay)
                r = self._post_candidate(payload, headers)
                if r.status_code == 200:
                    return self._absorb_proposal(r)
                last_status, last_body = r.status_code, r.text
                if r.status_code in CAPACITY_STATUSES:
                    continue
            elif self._model_not_found(r):
                continue  # dead/rotated model name — next candidate
            elif r.status_code in CAPACITY_STATUSES:
                continue  # transient provider outage — next candidate
            raise LLMBackendError(f"openai-compat {r.status_code}: {r.text[:300]}")
        raise LLMBackendError(
            f"openai-compat capacity exhausted across {len(candidates)} model(s): "
            f"{last_status} {last_body[:200]}"
        )

    def _retry_after_s(self, r: httpx.Response) -> float:
        raw = r.headers.get("retry-after", "1")
        try:
            return max(0.0, min(float(raw), self.retry_after_max_s))
        except ValueError:
            return 1.0

    def _absorb_proposal(self, r: httpx.Response) -> list[ToolStep]:
        data = r.json()
        self._absorb_usage(data.get("usage", {}))
        msg = data.get("choices", [{}])[0].get("message", {})
        # multi-turn: remember the assistant turn VERBATIM so tool_call ids
        # survive into the next replay (Groq/OpenRouter require id pairing
        # for the role="tool" responses)
        if msg.get("tool_calls"):
            self._history.append(msg)
        elif msg.get("content"):
            self._history.append({"role": "assistant", "content": str(msg["content"])[:2000]})
        steps: list[ToolStep] = []
        for call in msg.get("tool_calls") or []:
            fn = call.get("function", {})
            # un-mangle the wire name back to the canonical dotted form
            name = str(fn.get("name", "")).replace("__", ".")
            if not name:
                continue
            # fail-closed with feedback: unknown names reach the registry and
            # are rejected with the known-tool list (never silently dropped)
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            meta = {"call_id": call.get("id")} if call.get("id") else None
            steps.append(
                ToolStep(
                    tool=name,
                    args=args,
                    checkpoint_note=f"{self.provider}:{self.model} proposal",
                    provider_meta=meta,
                )
            )
            if len(steps) >= MAX_STEPS_PER_PROPOSAL:
                break
        return steps

    def complete_json(self, system: str, user: str, max_tokens: int = 4096) -> str:
        """Plain-text JSON completion (Phase A macro plan — no tools attached).

        Walks the same [model, *fallbacks] capacity ladder as propose(): a
        429 honors Retry-After once per candidate, dead model names skip to
        the next, and only full exhaustion raises to the tier above.
        """
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        candidates = [self.model, *self._fallbacks]
        last_status, last_body = 0, ""
        for candidate in candidates:
            self.model = candidate
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.3,
                "max_tokens": max_tokens,
            }
            r = self._post_candidate(payload, headers)
            if r.status_code == 200:
                text = self._extract_completion(r)
                if text.strip():
                    return text
                last_status, last_body = 200, "(empty content)"
                continue  # reasoning-only output — try the next candidate
            last_status, last_body = r.status_code, r.text
            if r.status_code == 429:
                time.sleep(self._retry_after_s(r))
                r = self._post_candidate(payload, headers)
                if r.status_code == 200:
                    text = self._extract_completion(r)
                    if text.strip():
                        return text
                    last_status, last_body = 200, "(empty content)"
                    continue
                last_status, last_body = r.status_code, r.text
                if r.status_code in CAPACITY_STATUSES:
                    continue
            elif self._model_not_found(r) or r.status_code in CAPACITY_STATUSES:
                continue
            raise LLMBackendError(f"openai-compat {r.status_code}: {r.text[:300]}")
        raise LLMBackendError(
            f"openai-compat capacity exhausted across {len(candidates)} model(s): "
            f"{last_status} {last_body[:200]}"
        )

    def _extract_completion(self, r: httpx.Response) -> str:
        """content first; reasoning-only models (nemotron/gpt-oss family) fall
        back to the reasoning field so a 200 never yields an empty string."""
        data = r.json()
        self._absorb_usage(data.get("usage", {}))
        msg = data.get("choices", [{}])[0].get("message", {})
        content = str(msg.get("content") or "").strip()
        if content:
            return content
        for key in ("reasoning", "reasoning_content"):
            alt = str(msg.get(key) or "").strip()
            if alt:
                return alt
        return ""


def build_backend(
    registry: ToolRegistry,
    niche: str = "bodycam_documentary",
    *,
    llm_api_key: str = "",
    llm_base_url: str = "",
    llm_model: str = "",
    llm_provider: str = "",
) -> Any:
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
        return OpenAICompatBackend(
            api_key,
            model,
            registry,
            niche,
            base_url=base_url,
            fallback_models=_env_model_list(s.llm_fallback_models) or None,
            prompt_char_budget=_groq_prompt_budget(base_url),
        )
    if provider == "meta":
        return _meta_backend(api_key, model, registry, niche, META_DEFAULT_BASE)
    return GeminiRestBackend(api_key, model, registry, niche)


def _groq_prompt_budget(base_url: str) -> int:
    """Clamp budget only on Groq (8k TPM free tier); override via LLM_PROMPT_CHAR_BUDGET."""
    override = os.getenv("LLM_PROMPT_CHAR_BUDGET", "")
    if override:
        return max(0, int(override))
    return GROQ_PROMPT_CHAR_BUDGET if "groq.com" in base_url else 0


class FailoverBackend:
    """Ordered provider tiers behind one LLMBackend port (stage-6 routing law).

    propose() walks the tiers: Groq (openai-compat, model chain inside the
    tier) -> OpenRouter (:free) -> native Gemini. A tier that raises a
    CAPACITY error hands over to the next tier; a contract error (400 shape,
    auth) re-raises immediately — failover is for capacity, not for bugs.
    """

    provider = "failover"

    def __init__(self, backends: list[Any]):
        if not backends:
            raise LLMBackendError("FailoverBackend requires at least one backend")
        self._backends = list(backends)
        self.active = self._backends[0]
        self.last_usage: dict[str, int] | None = None

    @property
    def tier_names(self) -> list[str]:
        return [f"{getattr(b, 'provider', '?')}:{getattr(b, 'model', '')}" for b in self._backends]

    @property
    def model(self) -> str:
        """Active tier label (provider:model) — logging/pipeline face."""
        return f"{getattr(self.active, 'provider', '?')}:{getattr(self.active, 'model', '')}"

    def complete_json(self, system: str, user: str, max_tokens: int = 4096) -> str:
        """Phase A macro plan across the tier chain (capacity-walking)."""
        failures: list[str] = []
        for backend in self._backends:
            complete = getattr(backend, "complete_json", None)
            if not callable(complete):
                continue
            try:
                text = complete(system, user, max_tokens=max_tokens)
            except LLMBackendError as e:
                msg = str(e).lower()
                if any(marker in msg for marker in _CAPACITY_ERROR_MARKERS):
                    failures.append(f"{getattr(backend, 'provider', '?')}: {e}")
                    continue
                raise
            self.active = backend
            self.last_usage = getattr(backend, "last_usage", None)
            return text
        raise LLMBackendError(
            "all LLM tiers exhausted for complete_json ("
            + " | ".join(failures)
            + " | tiers: "
            + ",".join(self.tier_names)
            + ")"
        )

    def reset_history(self) -> None:
        for backend in self._backends:
            reset = getattr(backend, "reset_history", None)
            if callable(reset):
                reset()

    def propose(self, state_summary: str, role: AgentRole) -> list[ToolStep]:
        failures: list[str] = []
        for backend in self._backends:
            try:
                steps = backend.propose(state_summary, role)
            except LLMBackendError as e:
                msg = str(e).lower()
                if any(marker in msg for marker in _CAPACITY_ERROR_MARKERS):
                    failures.append(f"{getattr(backend, 'provider', '?')}: {e}")
                    continue
                raise  # contract/auth bug — failover would only mask it
            self.active = backend
            self.last_usage = getattr(backend, "last_usage", None)
            return steps
        raise LLMBackendError(
            "all LLM tiers exhausted ("
            + " | ".join(failures)
            + " | tiers: "
            + ",".join(self.tier_names)
            + ")"
        )

    def record_results(self, steps: list[ToolStep], results: list[str]) -> None:
        record = getattr(self.active, "record_results", None)
        if callable(record):
            record(steps, results)


def build_backend_chain(
    registry: ToolRegistry,
    niche: str = "bodycam_documentary",
    **overrides: Any,
) -> Any:
    """Stage-6 routing chain from env (config is data):

    1. ``LLM_*``        — primary: Groq openai-compat (model chain via
       ``LLM_FALLBACK_MODELS``), Meta, or native Gemini when the URL is blank.
    2. ``OPENROUTER_*`` — secondary openai-compat tier (``OPENROUTER_MODEL``).
    3. native Gemini    — last resort on the AI Studio key (the VISION key
       hosts the shared ``AQ.`` credential), chain ``GEMINI_FALLBACK_MODELS``.

    Single configured tier -> that backend bare (zero overhead); several ->
    ``FailoverBackend``. Raises only when NOTHING is configured.
    """
    from director_os.core.config import get_settings

    s = get_settings()
    tiers: list[Any] = []
    if s.llm_api_key or overrides.get("llm_api_key"):
        tiers.append(build_backend(registry, niche, **overrides))
    if s.openrouter_api_key and s.openrouter_base_url:
        already = any(
            getattr(t, "base_url", "").rstrip("/") == s.openrouter_base_url.rstrip("/")
            and getattr(t, "model", "") == s.openrouter_model
            for t in tiers
        )
        if not already:
            tiers.append(
                OpenAICompatBackend(
                    s.openrouter_api_key,
                    s.openrouter_model or "nvidia/nemotron-3-ultra:free",
                    registry,
                    niche,
                    base_url=s.openrouter_base_url,
                    # the tier IS the fallback: its own chain rides
                    # OPENROUTER_FALLBACK_MODELS (catalog-rotation proof)
                    fallback_models=_env_model_list(s.openrouter_fallback_models) or None,
                )
            )
    # The last-resort Gemini tier only makes sense on a REAL AI Studio key —
    # a Groq `gsk_`/OpenRouter `sk-or-` key would just 401 there.
    gemini_key = s.vision_api_key or (
        s.llm_api_key if s.llm_api_key.startswith(("AQ.", "AIza")) else ""
    )
    if gemini_key and not any(isinstance(t, GeminiRestBackend) for t in tiers):
        model = s.vision_model or "gemini-3.5-flash-lite"
        tiers.append(
            GeminiRestBackend(
                gemini_key,
                model,
                registry,
                niche,
                fallback_models=_env_model_list(s.gemini_fallback_models) or None,
            )
        )
    if not tiers:
        raise LLMBackendError(
            "LLM not configured: set LLM_API_KEY (optionally OPENROUTER_API_KEY / "
            "VISION_API_KEY for the fallback tiers)"
        )
    return FailoverBackend(tiers) if len(tiers) > 1 else tiers[0]


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
