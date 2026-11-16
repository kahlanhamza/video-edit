"""Timeline builder — typed specs in, OpenTimelineIO out (ADR-0002).

Track naming conventions used by the compiler:

- `V1_master`          the master video cut (raw + narrator clips, sequential)
- `V2_overlay_cards`   graphic card overlays (gaps + cards; absolute positions)
- `A1_narration`       narrator audio track (populated in M2 with TTS clips)

`GapSpec` items exist so overlay positions are absolute and OTIO-idiomatic.

OpenTimelineIO >= 0.18 consolidates Timeline/Track/Clip/Gap under
`opentimelineio.schema`; opentime stays in `opentimelineio.opentime`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import opentimelineio.adapters as ot_adapters
import opentimelineio.opentime as ot_time
import opentimelineio.schema as ot_schema
from director_os.core.models import AudioContract
from director_os.otio_engine.contract import (
    METADATA_KEY,
    SCHEMA_VERSION,
    CardMeta,
    NarratorClipMeta,
    RawClipMeta,
    attach,
)
from pydantic import BaseModel, Field

MASTER_TRACK = "V1_master"
OVERLAY_TRACK = "V2_overlay_cards"
NARRATION_TRACK = "A1_narration"


class GapSpec(BaseModel):
    """Empty space on a track (used on the overlay track for absolute placement)."""

    kind: Literal["gap"] = "gap"
    name: str = "gap"
    dur_s: float = Field(gt=0.0)


class ClipSpec(BaseModel):
    name: str
    source_start_s: float = Field(default=0.0, ge=0.0)
    dur_s: float = Field(gt=0.0)
    meta: RawClipMeta | NarratorClipMeta | CardMeta


TrackItem = ClipSpec | GapSpec


class TrackSpec(BaseModel):
    name: str
    kind: Literal["Video", "Audio"] = "Video"
    items: list[TrackItem] = Field(default_factory=list)


class TimelineSpec(BaseModel):
    name: str
    fps: int = Field(default=30, gt=0)
    width: int = Field(default=1920, gt=0)
    height: int = Field(default=1080, gt=0)
    audio: AudioContract = Field(default_factory=AudioContract)
    tracks: list[TrackSpec] = Field(default_factory=list)


def _seconds(value: float, rate: int) -> ot_time.RationalTime:
    """RationalTime representing `value` seconds at `rate` fps (value in rate units)."""
    return ot_time.RationalTime(value=float(value * rate), rate=rate)


def build_timeline(spec: TimelineSpec) -> ot_schema.Timeline:
    tl = ot_schema.Timeline(name=spec.name)
    tl.global_start_time = _seconds(0.0, spec.fps)
    tl.metadata[METADATA_KEY] = {
        "schema_version": SCHEMA_VERSION,
        "fps": spec.fps,
        "resolution": {"width": spec.width, "height": spec.height},
        "audio": spec.audio.model_dump(mode="json"),
    }
    for tspec in spec.tracks:
        track = ot_schema.Track(name=tspec.name, kind=tspec.kind)
        for item in tspec.items:
            if isinstance(item, GapSpec):
                track.append(
                    ot_schema.Gap(
                        source_range=ot_time.TimeRange(
                            start_time=_seconds(0.0, spec.fps),
                            duration=_seconds(item.dur_s, spec.fps),
                        )
                    )
                )
                continue
            clip = ot_schema.Clip(name=item.name)
            clip.source_range = ot_time.TimeRange(
                start_time=_seconds(item.source_start_s, spec.fps),
                duration=_seconds(item.dur_s, spec.fps),
            )
            attach(clip, item.meta)
            track.append(clip)
        tl.tracks.append(track)
    return tl


def save_otio(timeline: ot_schema.Timeline, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ot_adapters.write_to_string(timeline, "otio_json"))
    return path


def load_otio(path: str | Path) -> ot_schema.Timeline:
    return ot_adapters.read_from_file(str(path))


def timeline_metadata(timeline: ot_schema.Timeline) -> dict:
    return dict(timeline.metadata.get(METADATA_KEY, {}))


def track_by_name(timeline: ot_schema.Timeline, name: str):
    for t in timeline.tracks:
        if t.name == name:
            return t
    return None


def item_duration_s(item) -> float:
    sr = getattr(item, "source_range", None)
    return sr.duration.to_seconds() if sr else 0.0


def is_gap(item) -> bool:
    """True for OTIO Gap items (no metadata contract block, by design)."""
    return isinstance(item, ot_schema.Gap)


def track_duration_s(track) -> float:
    return sum(item_duration_s(it) for it in track)


def to_json_dict(timeline: ot_schema.Timeline) -> dict:
    return json.loads(ot_adapters.write_to_string(timeline, "otio_json"))
