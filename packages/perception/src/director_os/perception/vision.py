"""Vision LLM adapter — frame understanding (PLAN §4.6, M4).

Normalizes multimodal scene understanding behind an injected client port, so
tests stay hermetic and provider choice (Z.ai GLM-4V, OpenAI, Modal-hosted
open weights) is a constructor argument, not a rewrite:

- `FrameQuery`: sampled frames + a question -> `SceneInsight`s (timestamped,
  structured findings the Director agent can act on)
- The real cloud adapter ships when `VISION_API_KEY` exists (input pending);
  the port + normalized contract + hermetic fake land now so every downstream
  consumer is already written against the final shape.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import httpx
    from director_os.perception.vision_meta import MetaVisionClient

from pydantic import BaseModel, Field


class VisionError(RuntimeError):
    pass


# Shared answer contract for every vision provider (Gemini + Meta Spark):
# the model MUST reply with a JSON array of SceneInsight-shaped objects.
INSIGHT_INSTRUCTION = (
    "You are the vision half of a documentary editor. "
    "Answer ONLY with a JSON array of objects: "
    '[{"timestamp_s": float, "kind": "observation|risk|'
    'redaction_candidate|event", "description": str, '
    '"confidence": float 0..1}]. Question: '
)

# Stage-4 capacity routing (config is data): the Gemini primary walks
# VISION_FALLBACK_MODELS; when the tier exhausts, the openai-compat
# OpenRouter tier takes over. Chain order: LIVE :free ids first (verified
# 2026-11-19 against /models + a tool-call probe), then the originally
# requested ids (skipped cheaply when the catalog no longer serves them).
# NEVER use the bogus "openrouter/free" id — it is not a real model.
DEFAULT_OPENROUTER_VISION_MODELS: tuple[str, ...] = (
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3.5-super-vl:free",
    "google/gemma-4-31b:free",
    "qwen/qwen3.8-27b:free",
)
OPENROUTER_VISION_MODELS: tuple[str, ...] = DEFAULT_OPENROUTER_VISION_MODELS
VISION_CAPACITY_STATUSES = frozenset({429, 500, 503})
# model ids the catalog no longer serves (rotation) — skip candidate, not stage
_VISION_MODEL_DEAD_MARKERS = (
    "no endpoints found",
    "not a valid model",
    "does not exist",
    "no longer supported",
    "model_not_found",
)
_CAPACITY_ERROR_MARKERS = (
    "429",
    "rate limit",
    "too many requests",
    "quota",
    "capacity",
    "503",
    "500",
    "401",
    "403",
    "unauthorized",
    "forbidden",
)


class FrameRef(BaseModel):
    """One sampled frame: media timestamp + the image itself."""

    timestamp_s: float = Field(ge=0.0)
    image_path: Path

    @property
    def exists(self) -> bool:
        return self.image_path.exists()


class SceneInsight(BaseModel):
    """One structured finding about a frame (Director-readable)."""

    timestamp_s: float
    kind: str = "observation"  # observation | risk | redaction_candidate | event
    description: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class VisionClient(Protocol):
    """Provider port: question + frames -> raw insights."""

    def analyze(self, question: str, frames: Sequence[FrameRef]) -> list[SceneInsight]: ...


class FakeVisionClient:
    """Deterministic hermetic client — pins the contract in tests."""

    def __init__(self, canned: list[SceneInsight] | None = None):
        self.canned = canned or []
        self.calls: list[tuple[str, int]] = []

    def analyze(self, question: str, frames: Sequence[FrameRef]) -> list[SceneInsight]:
        self.calls.append((question, len(frames)))
        return list(self.canned)


class VisionAdapter:
    """Application service: validates inputs, delegates to the provider port."""

    def __init__(self, client: VisionClient):
        self._client = client

    def understand(
        self, question: str, frames: Sequence[FrameRef], min_confidence: float = 0.0
    ) -> list[SceneInsight]:
        if not question.strip():
            raise VisionError("question must not be empty")
        missing = [f.image_path for f in frames if not f.exists]
        if missing:
            raise VisionError(f"frame(s) missing on disk: {missing[0]}")
        if not frames:
            raise VisionError("no frames provided")
        insights = self._client.analyze(question, frames)
        return [i for i in insights if i.confidence >= min_confidence]


class GeminiVisionClient:
    """Real multimodal provider: Google AI Studio generateContent (VISION_API_KEY).

    Frames ride as inline_data JPEG/PNG base64; the model answers with a JSON
    array of insights — malformed output degrades to one low-confidence
    observation instead of failing the pipeline.
    """

    provider = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.5-flash-lite",
        base_url: str = "https://generativelanguage.googleapis.com",
        transport: httpx.BaseTransport | None = None,
        modal_egress: bool = True,
        fallback_models: list[str] | None = None,
    ):
        if not api_key:
            raise VisionError("GeminiVisionClient requires VISION_API_KEY")
        self._transport = transport
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.last_usage: dict[str, int] | None = None
        # Some datacenter regions are geo-blocked by the Generative Language API
        # (400 FAILED_PRECONDITION "User location is not supported"). When that
        # happens and MODAL_TOKEN_* exist, the call retries through the deployed
        # Modal fleet's `vlm_call` (US/EU egress) and the decision sticks.
        self.modal_egress = modal_egress
        self._egress_active: str = "direct"
        self._fallbacks: list[str] = (
            [m for m in fallback_models if m != model]
            if fallback_models is not None
            else _env_model_list(os.getenv("VISION_FALLBACK_MODELS", "gemini-3.1-flash-lite"))
        )

    def _frames_body(self, question: str, frames: Sequence[FrameRef], prefix: str) -> dict:
        """Shared multimodal payload: question (+ optional instruction prefix)
        followed by every frame as stamped inline_data."""
        import base64

        parts: list[dict[str, Any]] = [{"text": prefix + question}]
        for f in frames:
            data = base64.b64encode(f.image_path.read_bytes()).decode()
            mime = "image/png" if f.image_path.suffix.lower() == ".png" else "image/jpeg"
            parts.append(
                {"text": f"frame at {f.timestamp_s}s:"},
            )
            parts.append({"inline_data": {"mime_type": mime, "data": data}})
        return {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048},
        }

    @staticmethod
    def _candidates_text(data: dict[str, Any]) -> str:
        """Candidate text parts joined; `thought` parts DROPPED — reasoning
        summaries riding inside the text broke every strict JSON consumer
        (live E2E: contact-sheet 503s, critic garbage rounds, Phase A parse)."""
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        return "".join(str(p.get("text", "")) for p in parts if not p.get("thought"))

    def raw_analysis(self, question: str, frames: Sequence[FrameRef]) -> str:
        """The model's VERBATIM reply (thought-free, fence-stripped).

        The ReAct critic asks for an ACTION OBJECT — `analyze`'s insight-array
        decode would shred it (object keys iterated as insights, descriptions
        joined without structure). The critic needs the model's own decision
        text, not a paraphrase channel."""
        import json as _json

        response: dict[str, Any] = self._generate(self._frames_body(question, frames, ""), _json)
        text = self._candidates_text(response).strip()
        if text.startswith("```"):
            text = text[3:].removeprefix("json").strip()
        if text.endswith("```"):
            text = text[:-3].strip()
        return text

    def analyze(self, question: str, frames: Sequence[FrameRef]) -> list[SceneInsight]:
        import json as _json

        response: dict[str, Any] = self._generate(
            self._frames_body(question, frames, INSIGHT_INSTRUCTION), _json
        )
        return self._parse(response, frames)

    def _generate(self, body: dict[str, Any], _json: Any) -> dict[str, Any]:
        """Walk [model, *fallbacks] with unified GEO + CAPACITY routing.

        Per candidate: direct endpoint first (geo-block switches egress to the
        fleet, STICKY); 429/500/503 move to the next candidate; anything else
        raises. Via the fleet, transient statuses back off x3 inside
        `_post_via_modal`, then the chain continues.
        """
        import httpx as _httpx

        models = [self.model, *self._fallbacks]
        last_status, last_body = 0, ""
        for candidate in models:
            self.model = candidate
            if self._egress_active == "direct":
                try:
                    with _httpx.Client(transport=self._transport, timeout=90.0) as client:
                        r = client.post(
                            f"{self.base_url}/v1beta/models/{candidate}:generateContent",
                            headers={"x-goog-api-key": self.api_key},
                            json=body,
                        )
                except _httpx.HTTPError as e:
                    raise VisionError(f"vision request failed: {e}") from e
                if r.status_code == 200:
                    return r.json()
                last_status, last_body = r.status_code, r.text
                if self.modal_egress and "location is not supported" in r.text:
                    self._egress_active = "modal"  # sticky; same candidate rides the fleet
                elif r.status_code in VISION_CAPACITY_STATUSES:
                    continue  # capacity on this model — try the next candidate
                else:
                    raise VisionError(f"vision provider {r.status_code}: {r.text[:200]}")
            status, out_body = self._post_via_modal(body, _json)
            if status == 200:
                return _json.loads(out_body)
            last_status, last_body = status, out_body
            if status not in VISION_CAPACITY_STATUSES:
                break  # hard error through the fleet — do not burn the chain
        raise VisionError(
            f"vision capacity exhausted across {len(models)} model(s) via "
            f"{self._egress_active} egress: {last_status} {last_body[:200]}"
        )

    def _post_via_modal(self, body: dict[str, Any], _json: Any) -> tuple[int, str]:
        """One candidate through the fleet proxy; transient statuses back off x3."""
        import time as _time

        from director_os.core.config import get_settings

        settings = get_settings()
        if not (settings.modal_token_id and settings.modal_token_secret):
            raise VisionError(
                "Gemini geo-blocked from this region and MODAL_TOKEN_ID/SECRET "
                "not configured for fleet egress"
            )
        import modal

        fn = modal.Function.from_name("directoros-vlm", "vlm_call")
        payload = _json.dumps(body)
        status, out_body = 0, ""
        for attempt in range(3):  # 503/429 demand spikes are transient — back off and retry
            out = fn.remote(model=self.model, payload_json=payload)
            status, out_body = int(out.get("status", 0)), str(out.get("body", ""))
            if status == 200:
                return status, out_body
            if status not in (429, 500, 503):
                break
            _time.sleep(4.0 * (attempt + 1))
        return status, out_body

    def _parse(self, data: dict[str, Any], frames: Sequence[FrameRef]) -> list[SceneInsight]:
        import json as _json

        usage = data.get("usageMetadata", {})
        self.last_usage = {
            "tokens_in": int(usage.get("promptTokenCount", 0)),
            "tokens_out": int(usage.get("candidatesTokenCount", 0)),
        }
        text = self._candidates_text(data).strip()
        try:
            if text.startswith("```"):
                text = text.strip("`").removeprefix("json").strip()
            raw = _json.loads(text)
            fallback_ts = frames[0].timestamp_s if frames else 0.0
            return [
                SceneInsight(
                    timestamp_s=float(i.get("timestamp_s", fallback_ts)),
                    kind=str(i.get("kind", "observation")),
                    description=str(i.get("description", ""))[:500],
                    confidence=float(i.get("confidence", 0.5)),
                )
                for i in raw
            ]
        except (ValueError, TypeError, KeyError):
            return [
                SceneInsight(
                    timestamp_s=frames[0].timestamp_s,
                    kind="observation",
                    description=text[:500] or "(empty vision response)",
                    confidence=0.3,
                )
            ]


def _env_model_list(raw: str) -> list[str]:
    return [m.strip() for m in raw.split(",") if m.strip()]


def parse_insights_text(text: str, frames: Sequence[FrameRef]) -> list[SceneInsight]:
    """Shared JSON-array decoder for every text-answering vision provider.

    Malformed output degrades to one low-confidence observation instead of
    failing the pipeline (same contract as the original Gemini parser).
    """
    import json as _json

    text = text.strip()
    try:
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        raw = _json.loads(text)
        fallback_ts = frames[0].timestamp_s if frames else 0.0
        return [
            SceneInsight(
                timestamp_s=float(i.get("timestamp_s", fallback_ts)),
                kind=str(i.get("kind", "observation")),
                description=str(i.get("description", ""))[:500],
                confidence=float(i.get("confidence", 0.5)),
            )
            for i in raw
        ]
    except (ValueError, TypeError, KeyError):
        return [
            SceneInsight(
                timestamp_s=frames[0].timestamp_s if frames else 0.0,
                kind="observation",
                description=text[:500] or "(empty vision response)",
                confidence=0.3,
            )
        ]


class OpenRouterVisionClient:
    """OpenAI-compat multimodal fallback tier (stage 4, OPENROUTER_*).

    Frames ride as ``image_url`` data URIs; the pinned model chain is
    ``google/gemma-4-31b:free`` -> ``qwen/qwen3.8-27b:free`` — never the bogus
    "openrouter/free" id. A 429/500/503 on one model walks to the next;
    exhaustion raises a capacity-marked VisionError so callers can react.
    """

    provider = "openrouter"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        models: list[str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        if not api_key:
            raise VisionError("OpenRouterVisionClient requires OPENROUTER_API_KEY")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        chain = list(
            models
            if models is not None
            else _env_model_list(os.getenv("OPENROUTER_VISION_MODELS", ""))
            or list(DEFAULT_OPENROUTER_VISION_MODELS)
        )
        self.model = chain[0]
        self._models = [m for m in chain if m]
        self._transport = transport
        self.last_usage: dict[str, int] | None = None

    def analyze(self, question: str, frames: Sequence[FrameRef]) -> list[SceneInsight]:
        import base64

        import httpx

        content: list[dict[str, Any]] = [{"type": "text", "text": INSIGHT_INSTRUCTION + question}]
        for f in frames:
            mime = "image/png" if f.image_path.suffix.lower() == ".png" else "image/jpeg"
            data = base64.b64encode(f.image_path.read_bytes()).decode()
            content.append({"type": "text", "text": f"frame at {f.timestamp_s}s:"})
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}
            )
        messages = [{"role": "user", "content": content}]
        last_status, last_body = 0, ""
        for candidate in self._models:
            self.model = candidate
            try:
                with httpx.Client(transport=self._transport, timeout=90.0) as client:
                    r = client.post(
                        f"{self.base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={"model": candidate, "messages": messages, "temperature": 0.2},
                    )
            except httpx.HTTPError as e:
                raise VisionError(f"openrouter vision request failed: {e}") from e
            last_status, last_body = r.status_code, r.text
            if r.status_code == 200:
                data = r.json()
                usage = data.get("usage", {})
                self.last_usage = {
                    "tokens_in": int(usage.get("prompt_tokens", 0)),
                    "tokens_out": int(usage.get("completion_tokens", 0)),
                }
                text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return parse_insights_text(text, frames)
            if r.status_code == 404 or (
                r.status_code == 400
                and any(m in r.text.lower() for m in _VISION_MODEL_DEAD_MARKERS)
            ):
                continue  # rotated-out model id — next candidate
            if r.status_code not in VISION_CAPACITY_STATUSES:
                break  # hard error — not a capacity problem
        raise VisionError(
            f"openrouter vision capacity exhausted across {len(self._models)} model(s): "
            f"{last_status} {last_body[:200]}"
        )


class FailoverVisionClient:
    """Ordered vision tiers behind one VisionClient port (stage-4 routing law).

    Primary Gemini (model chain inside the tier) -> OpenRouter (:free). A tier
    that raises a CAPACITY error hands over to the next; anything else
    re-raises — failover is for capacity, not for bugs.
    """

    provider = "failover"

    def __init__(self, clients: list[VisionClient]):
        if not clients:
            raise VisionError("FailoverVisionClient requires at least one client")
        self._clients = list(clients)
        self.active = self._clients[0]
        self.last_usage: dict[str, int] | None = None

    def analyze(self, question: str, frames: Sequence[FrameRef]) -> list[SceneInsight]:
        failures: list[str] = []
        for client in self._clients:
            try:
                insights = client.analyze(question, frames)
            except VisionError as e:
                msg = str(e).lower()
                if any(marker in msg for marker in _CAPACITY_ERROR_MARKERS):
                    failures.append(f"{getattr(client, 'provider', '?')}: {e}")
                    continue
                raise  # contract bug — failover would only mask it
            self.active = client
            self.last_usage = getattr(client, "last_usage", None)
            return insights
        raise VisionError(
            "all vision tiers exhausted ("
            + " | ".join(failures)
            + " | tiers: "
            + ",".join(
                f"{getattr(c, 'provider', '?')}:{getattr(c, 'model', '')}" for c in self._clients
            )
            + ")"
        )


def build_vision_client() -> VisionClient:
    """Provider selection law (M4 VLM).

    - ``VISION_PROVIDER=fake`` -> hermetic fake (CI / offline runs)
    - ``VISION_PROVIDER=gemini`` -> Gemini iff ``VISION_API_KEY`` else fake
    - ``VISION_PROVIDER=meta`` -> Meta Spark media understanding on the shared
      Model API key (``LLM_API_KEY``), spend-guarded against the same ledger
    - unset (auto): ``VISION_API_KEY`` wins; else a Meta key on an
      ``api.meta.ai`` base URL wins; else fake
    - When OPENROUTER creds exist, a ``:free`` fallback tier rides behind the
      Gemini primary via ``FailoverVisionClient`` (capacity-only failover).
    """
    from urllib.parse import urlparse

    from director_os.core.config import get_settings

    settings = get_settings()
    provider = (settings.vision_provider or "").strip().lower()
    if provider == "fake":
        return FakeVisionClient()
    if provider == "meta":
        return _meta_vision_client(settings)
    primary: VisionClient | None = None
    if provider == "gemini" or settings.vision_api_key:
        if settings.vision_api_key:
            primary = GeminiVisionClient(
                settings.vision_api_key,
                model=settings.vision_model or "gemini-3.5-flash-lite",
                fallback_models=_env_model_list(settings.vision_fallback_models) or None,
            )
    elif settings.llm_api_key and urlparse(settings.llm_base_url).hostname == "api.meta.ai":
        return _meta_vision_client(settings)
    if primary is None:
        return FakeVisionClient()
    fallback = _openrouter_vision_fallback(settings)
    if fallback is None:
        return primary
    return FailoverVisionClient([primary, fallback])


def _openrouter_vision_fallback(settings: Any) -> OpenRouterVisionClient | None:
    """OpenRouter :free tier when OPENROUTER creds exist (stage-4 fallback law)."""
    if not (settings.openrouter_api_key and settings.openrouter_base_url):
        return None
    return OpenRouterVisionClient(
        settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )


def _meta_vision_client(settings: Any) -> MetaVisionClient:
    """Meta vision client wired to the shared text-LLM spend ledger.

    One $20 credit pool, one budget cap: the ledger file (and therefore the
    pre-call budget check) is the SAME one ``MetaModelBackend`` writes.
    """
    from director_os.perception.vision_meta import JsonlSpendRecorder, MetaVisionClient

    ledger_path = settings.llm_spend_ledger or str(settings.output_dir / "llm_spend.jsonl")
    return MetaVisionClient(
        settings.llm_api_key,
        ledger=JsonlSpendRecorder(ledger_path),
        budget_usd=settings.llm_budget_usd,
        allow_contributor=settings.llm_allow_training_on_content,
    )
