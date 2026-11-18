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

    def analyze(self, question: str, frames: Sequence[FrameRef]) -> list[SceneInsight]:
        import base64
        import json as _json

        parts: list[dict[str, Any]] = [{"text": INSIGHT_INSTRUCTION + question}]
        for f in frames:
            data = base64.b64encode(f.image_path.read_bytes()).decode()
            mime = "image/png" if f.image_path.suffix.lower() == ".png" else "image/jpeg"
            parts.append(
                {"text": f"frame at {f.timestamp_s}s:"},
            )
            parts.append({"inline_data": {"mime_type": mime, "data": data}})
        body = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048},
        }
        response: dict[str, Any] = self._post(body, _json)
        return self._parse(response, frames)

    def _post(self, body: dict[str, Any], _json: Any) -> dict[str, Any]:
        import httpx

        if self._egress_active == "modal":
            return self._post_via_modal(body, _json)
        with httpx.Client(transport=self._transport, timeout=90.0) as client:
            r = client.post(
                f"{self.base_url}/v1beta/models/{self.model}:generateContent",
                headers={"x-goog-api-key": self.api_key},
                json=body,
            )
        if r.status_code != 200:
            if self.modal_egress and "location is not supported" in r.text:
                self._egress_active = "modal"
                return self._post_via_modal(body, _json)
            raise VisionError(f"vision provider {r.status_code}: {r.text[:200]}")
        return r.json()

    def _post_via_modal(self, body: dict[str, Any], _json: Any) -> dict[str, Any]:
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
        last_status, last_body = None, ""
        for attempt in range(3):  # 503/429 demand spikes are transient — back off and retry
            out = fn.remote(model=self.model, payload_json=payload)
            last_status, last_body = int(out.get("status", 0)), str(out.get("body", ""))
            if last_status == 200:
                return _json.loads(last_body)
            if last_status not in (429, 500, 503):
                break
            _time.sleep(4.0 * (attempt + 1))
        raise VisionError(f"vision provider via modal egress {last_status}: {last_body[:200]}")

    def _parse(self, data: dict[str, Any], frames: Sequence[FrameRef]) -> list[SceneInsight]:
        import json as _json

        usage = data.get("usageMetadata", {})
        self.last_usage = {
            "tokens_in": int(usage.get("promptTokenCount", 0)),
            "tokens_out": int(usage.get("candidatesTokenCount", 0)),
        }
        text = "".join(
            p.get("text", "")
            for p in data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        ).strip()
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


def build_vision_client() -> VisionClient:
    """Provider selection law (M4 VLM).

    - ``VISION_PROVIDER=fake`` -> hermetic fake (CI / offline runs)
    - ``VISION_PROVIDER=gemini`` -> Gemini iff ``VISION_API_KEY`` else fake
    - ``VISION_PROVIDER=meta`` -> Meta Spark media understanding on the shared
      Model API key (``LLM_API_KEY``), spend-guarded against the same ledger
    - unset (auto): ``VISION_API_KEY`` wins; else a Meta key on an
      ``api.meta.ai`` base URL wins; else fake
    """
    from urllib.parse import urlparse

    from director_os.core.config import get_settings

    settings = get_settings()
    provider = (settings.vision_provider or "").strip().lower()
    if provider == "fake":
        return FakeVisionClient()
    if provider == "gemini":
        return (
            GeminiVisionClient(settings.vision_api_key)
            if settings.vision_api_key
            else FakeVisionClient()
        )
    if provider == "meta":
        return _meta_vision_client(settings)
    if settings.vision_api_key:
        return GeminiVisionClient(settings.vision_api_key)
    if settings.llm_api_key and urlparse(settings.llm_base_url).hostname == "api.meta.ai":
        return _meta_vision_client(settings)
    return FakeVisionClient()


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
