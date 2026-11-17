"""ComputeBackend port models + protocol — the contract every adapter honors."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field


class ComputeNotAvailable(RuntimeError):
    """An adapter cannot service a job kind — message says which .env flips it."""


class Span(BaseModel):
    """A time window of interest (music-flagged span, separation window)."""

    start_s: float = Field(ge=0.0)
    end_s: float = Field(gt=0.0)

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


class TrackHit(BaseModel):
    """One per-frame box of a tracked face (RAW source pixels).

    `score` is the detector confidence for real detections; interpolated
    frames carry the last real score and `interpolated=True`.
    """

    t_s: float = Field(ge=0.0)
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(gt=0)
    h: int = Field(gt=0)
    score: float = Field(ge=0.0, le=1.0, default=0.0)
    interpolated: bool = False


class FaceTrack(BaseModel):
    """One persistent identity across time — Person #1 stays Person #1.

    `hits` covers every sampled frame of the track's active span (stride
    gaps filled by linear interpolation), so the renderer and QA can walk
    per-frame boxes without re-tracking. `min_score` is the floor across
    real detections — QA uses it to prune shaky tracks.
    """

    track_id: str
    hits: list[TrackHit] = Field(default_factory=list)

    @property
    def t_start_s(self) -> float:
        return self.hits[0].t_s if self.hits else 0.0

    @property
    def t_end_s(self) -> float:
        return self.hits[-1].t_s if self.hits else 0.0

    @property
    def n_real(self) -> int:
        return sum(1 for h in self.hits if not h.interpolated)

    @property
    def min_score(self) -> float:
        return min((h.score for h in self.hits if not h.interpolated), default=0.0)


class StemsResult(BaseModel):
    """Demucs output for separated spans (paths are local artifacts)."""

    vocals_path: Path | None = None
    music_path: Path | None = None
    spans: list[Span] = Field(default_factory=list)
    backend: str


class AsrAlignResult(BaseModel):
    """WhisperX-style transcription with word-level alignment."""

    language: str = ""
    words: list[dict] = Field(default_factory=list)  # {word, start_s, end_s, speaker?}
    segments: list[dict] = Field(default_factory=list)
    backend: str


class ComputeBackend(Protocol):
    """Structural port — adapters duck-type this, callers never import classes."""

    name: str

    def detect_faces(
        self,
        video_path: Path | str,
        sample_every_s: float = 0.1,
        max_samples: int = 4000,
        score_threshold: float = 0.35,
        progress: Callable[[int, int], None] | None = None,
    ) -> list[FaceTrack]:
        """Detect + track faces -> persistent per-frame tracks (raw pixels)."""
        ...

    def separate_stems(
        self,
        audio_path: Path | str,
        spans: list[Span],
        out_dir: Path | str | None = None,
    ) -> StemsResult:
        """Separate vocals/music for flagged spans (on-demand Demucs)."""
        ...

    def transcribe_align(self, audio_path: Path | str, language: str = "") -> AsrAlignResult:
        """Transcribe + word-level align (WhisperX-class)."""
        ...

    def capabilities(self) -> dict[str, bool]:
        """Which job kinds this adapter can actually run right now."""
        ...
