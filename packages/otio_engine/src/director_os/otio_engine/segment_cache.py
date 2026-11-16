"""Segment cache — content-hashed render units (ADR-0003, PLAN §5.1 `segment_cache`).

The cache key is a stable SHA-256 over the FULL render specification: source range,
filter chains, blur regions, card *content* digests, codec + audio-law settings.
Changing one narrator word, one blur sigma or one card pixel changes exactly one key —
which is what makes "fix one word, re-render in seconds" true by construction.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Literal

from director_os.core.models import AudioContract, BlurRegion
from pydantic import BaseModel, Field

MIN_SEGMENT_BYTES = 50_000


class CardOverlaySpec(BaseModel):
    """A rendered card PNG placed on the canvas for a segment-local time window."""

    png_path: str
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    t_start_s: float = Field(ge=0.0)
    t_end_s: float = Field(gt=0.0)
    fade_s: float = Field(default=0.5, ge=0.0, le=5.0)


class AudioRenderSpec(AudioContract):
    """Audio law inherited + validated (48 kHz / stereo — ADR-0005)."""

    codec: str = "aac"
    bitrate_kbps: int = 192


class VideoRenderSpec(BaseModel):
    w: int = Field(default=1920, gt=0)
    h: int = Field(default=1080, gt=0)
    fps: int = Field(default=30, gt=0)
    codec: str = "libx264"
    crf: int = Field(default=20, ge=0, le=51)
    preset: str = "veryfast"
    pix_fmt: str = "yuv420p"


class SegmentRenderSpec(BaseModel):
    schema_version: int = 1
    kind: Literal["raw", "narration"] = "raw"
    asset_id: str = ""
    source_start_s: float = Field(default=0.0, ge=0.0)
    source_dur_s: float = Field(gt=0.0)
    video: VideoRenderSpec = Field(default_factory=VideoRenderSpec)
    audio: AudioRenderSpec = Field(default_factory=AudioRenderSpec)
    audio_filters: list[str] = Field(default_factory=list)
    video_filters: list[str] = Field(default_factory=list)
    blur_regions: list[BlurRegion] = Field(default_factory=list)
    zoom_factor: float = Field(default=1.0, ge=1.0, le=3.0)
    cards: list[CardOverlaySpec] = Field(default_factory=list)
    narration_text: str = ""
    codec_profile: str = ""


def segment_hash(spec: SegmentRenderSpec) -> str:
    payload = spec.model_dump(mode="json", exclude={"cards": {"__all__": {"png_path"}}})
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha256(("dos-seg-v1|" + canonical).encode()).hexdigest()
    # card file CONTENT invalidates the key (pixel-level card edits re-render)
    for card in sorted(spec.cards, key=lambda c: c.png_path):
        digest = _file_digest(card.png_path)
        h = hashlib.sha256((h + "|" + digest).encode()).hexdigest()
    return h[:40]


def _file_digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:24]


class SegmentCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        return self.root / f"{key}.mp4"

    def get(self, key: str) -> Path | None:
        p = self.path_for(key)
        return p if p.exists() and p.stat().st_size > MIN_SEGMENT_BYTES else None

    def put(self, tmp_path: str | Path, key: str) -> Path:
        dest = self.path_for(key)
        shutil.move(str(tmp_path), dest)
        return dest

    def count(self) -> int:
        return sum(1 for _ in self.root.glob("*.mp4"))
