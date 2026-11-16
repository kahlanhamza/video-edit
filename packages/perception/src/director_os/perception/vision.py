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
from typing import Protocol

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
