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

from director_os.audio_engine.two_bus import SEGMENT_AAC_BITRATE_KBPS
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
    # 512k (codec reconstruction law): 192k AAC overshot the -2.5 dBFS 2-bus
    # master by up to +4.5 dB on decode — see two_bus.SEGMENT_AAC_BITRATE_KBPS.
    bitrate_kbps: int = SEGMENT_AAC_BITRATE_KBPS


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
    # --- 2-bus speech isolation (audio_mode=isolate_2bus): the segment's audio
    # rides this pre-mixed master WAV instead of the raw source stream. The
    # PATH is machine-local (excluded from the hash); the CONTENT digest is
    # part of the cache key, so a re-enhance re-renders exactly these segments.
    enhanced_audio_path: str | None = None
    enhanced_audio_digest: str | None = None
    # where the enhanced WAV slice starts. The 2-bus master rides the ASSET
    # clock; `source_start_s` rides the clock of the file actually sliced (a
    # per-clip baked redaction file starts at 0:00). None -> slice the WAV at
    # `source_start_s` (raw-source segments: both clocks are the asset clock).
    enhanced_audio_start_s: float | None = Field(default=None, ge=0.0)
    # --- per-clip baked redaction source (tracked blur baked BEFORE the DAG):
    # the segment cuts from a pre-blurred clip file; `source_digest` (derived
    # from the plan digest + clip range, not a file read) rides the cache key
    # so a re-swept plan re-renders exactly the affected clips. The source
    # FILE PATH itself lives on SegmentRender (never hashed).
    source_digest: str | None = None
    # --- narrator voiceover stem: pre-synthesized TTS clip (content-addressed
    # by script) mapped onto the narration segment. Path excluded from the
    # hash; the content digest rides the key so an edited script re-renders.
    narration_audio_path: str | None = None
    narration_audio_digest: str | None = None
    # --- multi-b-roll narration visuals (anti-frozen-b-roll law): 2-3 source
    # windows concatenated under ONE continuous TTS block; the list rides the
    # cache key so a re-planned cut re-renders exactly these segments.
    broll_windows: list[tuple[float, float]] | None = None
    # --- A1 audio trim: the VO window inside the TTS stem. Part of the cache
    # key (plain model fields) so a trim re-renders exactly this block.
    vo_in_s: float = Field(default=0.0, ge=0.0)
    vo_out_s: float | None = Field(default=None, ge=0.0)


def segment_hash(spec: SegmentRenderSpec) -> str:
    payload = spec.model_dump(
        mode="json",
        exclude={
            "cards": {"__all__": {"png_path"}},
            "spotlight_png_path": True,
            "enhanced_audio_path": True,  # machine-local path; content digest rides in the key
            "narration_audio_path": True,  # machine-local path; content digest rides in the key
        },
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
    if spec.enhanced_audio_path:
        digest = spec.enhanced_audio_digest or _file_digest(spec.enhanced_audio_path)
        h = hashlib.sha256(("enh|" + h + "|" + digest).encode()).hexdigest()
    if spec.source_digest:
        h = hashlib.sha256(("src|" + h + "|" + spec.source_digest).encode()).hexdigest()
    if spec.narration_audio_path:
        digest = spec.narration_audio_digest or _file_digest(spec.narration_audio_path)
        h = hashlib.sha256(("narr|" + h + "|" + digest).encode()).hexdigest()
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
