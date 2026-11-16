"""Transcript contract — normalized ASR output (PLAN §4.5, §5.1).

Every ASR provider (Gladia in M2, WhisperX fallback in M4) normalizes into
`Transcript`: word-level timings + speaker labels + optional language/sentiment.
Downstream stages (subtitle alignment, speech-trim cutting, Director planning)
only ever see this shape — provider quirks stop at the adapter.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class WordTiming(BaseModel):
    """One recognized word with millisecond-grade timing (source seconds)."""

    text: str = Field(min_length=1)
    start_s: float = Field(ge=0.0)
    end_s: float = Field(gt=0.0)
    speaker: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    def model_post_init(self, __context) -> None:
        if self.end_s <= self.start_s:
            raise ValueError(f"word timing inverted: {self.start_s} >= {self.end_s}")


class TranscriptSegment(BaseModel):
    """A diarized utterance: consecutive words from one speaker."""

    speaker: str
    start_s: float = Field(ge=0.0)
    end_s: float = Field(gt=0.0)
    text: str = Field(min_length=1)
    words: list[WordTiming] = Field(default_factory=list)


class Transcript(BaseModel):
    """The canonical perception-stage speech product for one asset."""

    asset_id: str
    language: str = "en"
    duration_s: float = Field(ge=0.0)
    words: list[WordTiming] = Field(default_factory=list)
    segments: list[TranscriptSegment] = Field(default_factory=list)
    provider: str = "unknown"

    @property
    def speakers(self) -> list[str]:
        return sorted({w.speaker for w in self.words if w.speaker})

    @property
    def speech_intervals(self) -> list[tuple[float, float]]:
        """Merged speech intervals — the base of PLAN guardrail 4 cutting."""
        if not self.words:
            return []
        intervals = sorted((w.start_s, w.end_s) for w in self.words)
        merged = [list(intervals[0])]
        for start, end in intervals[1:]:
            if start <= merged[-1][1] + 0.2:  # 200ms gap tolerance
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return [(a, b) for a, b in merged]

    def text_in_range(self, start_s: float, end_s: float) -> str:
        """All words overlapping [start, end) — for segment-local subtitles."""
        return " ".join(
            w.text for w in self.words if w.start_s < end_s and w.end_s > start_s
        ).strip()
