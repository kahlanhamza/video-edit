"""PLAN §5.2 — the OpenTimelineIO custom metadata contract.

Every `otio.schema.Clip` carries structured `metadata["director_os"]`:

- raw_bwc clips:  {"kind": "raw", "audio_mode": "loudnorm_48k" | "isolate_2bus" | "silence",
                   "trim_silence": true,
                   "zoom_keyframes": [...], "blur_regions": [...], "angle_id": "bwc_1",
                   "freeze_callout": {"freeze_at_s": ..., "hold_s": ..., "dim_alpha": ...,
                                      "spotlight_bbox": {...} | null} | null}
- narrator clips: {"kind": "narration", "tts_provider": "edge", "voice_id": "...",
                   "script_text": "...", "broll_asset_id": ..., "broll_range": [s, e],
                   "broll_audio_volume": 0.0, "jl_cut": {"j_s": 0.4}}
- overlay items:  {"kind": "card", "template": "case_outcome", "fields": {...},
                   "fade_s": 0.5}

Typed access goes through `attach()` / `parse_clip_meta()` — raw dict access elsewhere
is a contract violation waiting to happen.
"""

from __future__ import annotations

from typing import Any, Literal

from director_os.core.models import BBox, BlurRegion, ZoomKeyframe
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

METADATA_KEY = "director_os"
SCHEMA_VERSION = 1


class JLCut(BaseModel):
    """Audio lead/lag across a visual cut (PLAN §3.3C `apply_jl_cut`)."""

    j_s: float = Field(default=0.4, ge=0.0, le=10.0)
    l_s: float = Field(default=0.0, ge=0.0, le=10.0)


class FreezeCallout(BaseModel):
    """Freeze-frame evidence highlight (PLAN §3.3E `vision_fx.freeze_callout`).

    At `freeze_at_s` (segment-local seconds) the picture freezes for `hold_s`
    while audio keeps running — narration rides the freeze. `dim_alpha` darkens
    the background; `spotlight_bbox` (RAW-FOOTAGE pixels, same canvas as the
    blur regions) is punched out at full brightness with an accent ring. A
    `None` bbox dims the whole frame (e.g. under a full-screen document).
    """

    freeze_at_s: float = Field(ge=0.0)
    hold_s: float = Field(default=3.0, gt=0.0, le=15.0)
    dim_alpha: float = Field(default=0.45, ge=0.0, le=0.9)
    spotlight_bbox: BBox | None = None


class RawClipMeta(BaseModel):
    kind: Literal["raw"] = "raw"
    asset_id: str
    # isolate_2bus: broadcast speech-isolation mix (neural enhancer + gate);
    # the compiled master WAV must exist in `compile_timeline(enhanced_audio=)`
    audio_mode: Literal["loudnorm_48k", "isolate_2bus", "silence"] = "loudnorm_48k"
    trim_silence: bool = True
    zoom_keyframes: list[ZoomKeyframe] = Field(default_factory=list)
    blur_regions: list[BlurRegion] = Field(default_factory=list)
    freeze_callout: FreezeCallout | None = None
    angle_id: str | None = None
    act_name: str | None = None  # 6-act documentary marker (M5 Director session)


class NarratorClipMeta(BaseModel):
    kind: Literal["narration"] = "narration"
    tts_provider: str = "edge"
    voice_id: str = "en-US-BrianMultilingualNeural"
    script_text: str
    broll_asset_id: str | None = None
    broll_range: tuple[float, float] | None = None
    # 2-3 sequential muted b-roll windows under ONE spoken block — the TTS
    # audio stays continuous while the visuals cut window-to-window (anti
    # "frozen b-roll" law); the renderer cycles the list to fill the block.
    broll_windows: list[tuple[float, float]] | None = None
    broll_audio_volume: float = 0.0
    jl_cut: JLCut | None = None
    act_name: str | None = None  # 6-act documentary marker (M5 Director session)


class CardMeta(BaseModel):
    kind: Literal["card"] = "card"
    template: str
    fields: dict[str, Any] = Field(default_factory=dict)
    fade_s: float = Field(default=0.5, ge=0.0, le=5.0)


ClipMeta = RawClipMeta | NarratorClipMeta | CardMeta
_CLIP_META_ADAPTER = TypeAdapter(ClipMeta)


class MetadataContractError(RuntimeError):
    pass


def attach(otio_clip: Any, meta: ClipMeta) -> None:
    """Write a validated metadata block onto an OTIO clip."""
    otio_clip.metadata[METADATA_KEY] = meta.model_dump(mode="json")


def parse_clip_meta(otio_clip: Any) -> ClipMeta:
    """Read + validate the metadata block. Missing/invalid blocks are hard errors."""
    raw = getattr(otio_clip, "metadata", {}).get(METADATA_KEY)
    if raw is None:
        raise MetadataContractError(
            f"OTIO item '{getattr(otio_clip, 'name', '?')}' has no "
            f"'{METADATA_KEY}' metadata block (contract violation)"
        )
    try:
        return _CLIP_META_ADAPTER.validate_python(raw)
    except ValidationError as e:
        raise MetadataContractError(f"invalid {METADATA_KEY} metadata: {e}") from e


def meta_kind(otio_clip: Any) -> str:
    raw = getattr(otio_clip, "metadata", {}).get(METADATA_KEY, {})
    return str(raw.get("kind", "?"))
