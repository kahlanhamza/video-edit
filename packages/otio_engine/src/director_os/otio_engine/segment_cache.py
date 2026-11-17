"""Segment cache — content-hashed render units (ADR-0003, PLAN §5.1 `segment_cache`).

The cache key is a stable SHA-256 over the FULL render specification: source range,
filter chains, blur regions, zoom keyframes, freeze callouts, card *content* digests,
codec + audio-law settings. Changing one narrator word, one blur sigma, one zoom
keyframe or one card pixel changes exactly one key — which is what makes "fix one
word, re-render in seconds" true by construction.

Acceptance on `put()` is ffprobe-based, not byte-count-based: an all-black narration
segment is legitimately tiny (a few dozen KB), while a half-written encode from a
killed worker must never enter the cache. A chunk is valid iff it is a playable
MP4 whose duration matches the segment spec within tolerance.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Literal

from director_os.core.models import AudioContract, BlurRegion, ZoomKeyframe
from director_os.otio_engine.contract import FreezeCallout
from pydantic import BaseModel, Field

MIN_SEGMENT_BYTES = 2_048
DURATION_TOLERANCE = 0.25  # seconds; guards truncated tail frames


class SegmentCacheError(RuntimeError):
    pass


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
    schema_version: int = 2  # v2: blur-before-zoom ordering + keyframed zoompan + freeze
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
    zoom_keyframes: list[ZoomKeyframe] = Field(default_factory=list)
    freeze_callout: FreezeCallout | None = None
    spotlight_png_path: str | None = None
    cards: list[CardOverlaySpec] = Field(default_factory=list)
    narration_text: str = ""
    codec_profile: str = ""


def segment_hash(spec: SegmentRenderSpec) -> str:
    payload = spec.model_dump(
        mode="json",
        exclude={"cards": {"__all__": {"png_path"}}, "spotlight_png_path": True},
    )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha256(("dos-seg-v1|" + canonical).encode()).hexdigest()
    # card/overlay file CONTENT invalidates the key (pixel-level edits re-render)
    for card in sorted(spec.cards, key=lambda c: c.png_path):
        digest = _file_digest(card.png_path)
        h = hashlib.sha256((h + "|" + digest).encode()).hexdigest()
    if spec.spotlight_png_path:
        digest = _file_digest(spec.spotlight_png_path)
        h = hashlib.sha256((h + "|" + digest).encode()).hexdigest()
    return h[:40]


def _file_digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:24]


class SegmentCache:
    def __init__(self, root: str | Path):
        # Absolute by contract: concat lists and DAG paths must resolve from any
        # CWD — ffmpeg's concat demuxer resolves `file` lines relative to the
        # LIST file, not the process CWD, so relative roots broke final concat.
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        return self.root / f"{key}.mp4"

    def get(self, key: str) -> Path | None:
        p = self.path_for(key)
        return p if p.exists() and p.stat().st_size > MIN_SEGMENT_BYTES else None

    def put(self, tmp_path: str | Path, key: str, expect_dur_s: float | None = None) -> Path:
        """Admit a rendered chunk into the cache after ffprobe validation.

        Rejected chunks raise `SegmentCacheError` and are deleted — a killed
        worker's partial encode never becomes a cache entry.
        """
        tmp_path = Path(tmp_path)
        dest = self.path_for(key)
        try:
            _validate_chunk(tmp_path, expect_dur_s)
        except SegmentCacheError:
            tmp_path.unlink(missing_ok=True)
            raise
        shutil.move(str(tmp_path), dest)
        return dest

    def count(self) -> int:
        # only admitted chunks — a killed worker's .tmp.mp4 corpse is NOT an entry
        return sum(1 for p in self.root.glob("*.mp4") if not p.name.endswith(".tmp.mp4"))


def _validate_chunk(path: Path, expect_dur_s: float | None) -> None:
    from director_os.core.qa_probe import probe  # local import avoids probe<->cache cycle risk

    if not path.exists() or path.stat().st_size < MIN_SEGMENT_BYTES:
        raise SegmentCacheError(f"chunk too small / missing: {path}")
    try:
        info = probe(path)
    except Exception as e:  # garbage input -> ffprobe exits non-zero / malformed JSON
        raise SegmentCacheError(f"chunk is not a playable MP4: {path} ({e})") from e
    if not info.has_video:
        raise SegmentCacheError(f"chunk has no video stream: {path}")
    if expect_dur_s is not None and abs(info.duration_s - expect_dur_s) > DURATION_TOLERANCE:
        raise SegmentCacheError(
            f"chunk duration {info.duration_s:.2f}s != expected {expect_dur_s:.2f}s "
            f"(truncated encode?)"
        )
