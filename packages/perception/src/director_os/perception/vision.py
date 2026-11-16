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

from pydantic import BaseModel, Field


class VisionError(RuntimeError):
    pass


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
        model: str = "gemini-2.5-flash",
        base_url: str = "https://generativelanguage.googleapis.com",
        transport: httpx.BaseTransport | None = None,
    ):
        if not api_key:
            raise VisionError("GeminiVisionClient requires VISION_API_KEY")
        self._transport = transport
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.last_usage: dict[str, int] | None = None

    def analyze(self, question: str, frames: Sequence[FrameRef]) -> list[SceneInsight]:
        import base64
        import json as _json

        import httpx

        parts: list[dict[str, Any]] = [
            {
                "text": (
                    "You are the vision half of a documentary editor. "
                    "Answer ONLY with a JSON array of objects: "
                    '[{"timestamp_s": float, "kind": "observation|risk|'
                    'redaction_candidate|event", "description": str, '
                    '"confidence": float 0..1}]. Question: ' + question
                )
            }
        ]
        for f in frames:
            data = base64.b64encode(f.image_path.read_bytes()).decode()
            mime = "image/png" if f.image_path.suffix.lower() == ".png" else "image/jpeg"
            parts.append(
                {"text": f"frame at {f.timestamp_s}s:"},
            )
            parts.append({"inline_data": {"mime_type": mime, "data": data}})
        with httpx.Client(transport=self._transport, timeout=90.0) as client:
            r = client.post(
                f"{self.base_url}/v1beta/models/{self.model}:generateContent",
                headers={"x-goog-api-key": self.api_key},
                json={
                    "contents": [{"role": "user", "parts": parts}],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048},
                },
            )
        if r.status_code != 200:
            raise VisionError(f"vision provider {r.status_code}: {r.text[:200]}")
        data = r.json()
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
            return [
                SceneInsight(
                    timestamp_s=float(i.get("timestamp_s", f.timestamp_s)),
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
    """VISION_API_KEY set -> real Gemini client; blank -> hermetic fake."""
    from director_os.core.config import get_settings

    key = get_settings().vision_api_key
    if key:
        return GeminiVisionClient(key)
    return FakeVisionClient()
