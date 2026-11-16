"""OTIO contract + timeline builder round-trip (ADR-0002, PLAN §5.2)."""

from __future__ import annotations

import opentimelineio as otio  # type: ignore[import-untyped]
import pytest
from director_os.core.models import BBox, BlurRegion
from director_os.otio_engine.compiler import master_track
from director_os.otio_engine.contract import (
    CardMeta,
    MetadataContractError,
    NarratorClipMeta,
    RawClipMeta,
    parse_clip_meta,
)
from director_os.otio_engine.timeline import (
    ClipSpec,
    GapSpec,
    TimelineSpec,
    TrackSpec,
    build_timeline,
    load_otio,
    save_otio,
    timeline_metadata,
)


def spec() -> TimelineSpec:
    return TimelineSpec(
        name="case1_demo",
        fps=30,
        width=1280,
        height=720,
        tracks=[
            TrackSpec(
                name="V1_master",
                items=[
                    ClipSpec(
                        name="seg_raw",
                        source_start_s=0.0,
                        dur_s=6.0,
                        meta=RawClipMeta(
                            asset_id="bwc_1",
                            angle_id="bwc_1",
                            blur_regions=[BlurRegion(bbox=BBox(x=10, y=10, w=60, h=40), sigma=24)],
                        ),
                    ),
                    ClipSpec(
                        name="seg_narr",
                        source_start_s=0.0,
                        dur_s=4.0,
                        meta=NarratorClipMeta(script_text="The stop begins quietly."),
                    ),
                ],
            ),
            TrackSpec(
                name="V2_overlay_cards",
                items=[
                    GapSpec(dur_s=7.0),
                    ClipSpec(
                        name="card1",
                        source_start_s=0.0,
                        dur_s=3.0,
                        meta=CardMeta(
                            template="location_intro",
                            fields={"title": "JSO", "subtitle": "Jacksonville, FL"},
                        ),
                    ),
                ],
            ),
        ],
    )


def test_metadata_roundtrip_through_otio_json(tmp_path):
    tl = build_timeline(spec())
    meta = timeline_metadata(tl)
    assert meta["fps"] == 30
    assert meta["audio"]["sample_rate"] == 48000

    path = save_otio(tl, tmp_path / "case1.otio")
    tl2 = load_otio(path)
    master = master_track(tl2)
    items = list(master)
    assert len(items) == 2
    m1 = parse_clip_meta(items[0])
    assert isinstance(m1, RawClipMeta)
    assert m1.asset_id == "bwc_1"
    assert m1.blur_regions[0].bbox.w == 60
    m2 = parse_clip_meta(items[1])
    assert isinstance(m2, NarratorClipMeta)
    assert m2.script_text == "The stop begins quietly."


def test_otio_gaps_position_overlay_cards():
    tl = build_timeline(spec())
    overlay = next(t for t in tl.tracks if t.name == "V2_overlay_cards")
    kinds = [type(i).__name__ for i in overlay]
    assert kinds == ["Gap", "Clip"], "cards must be positioned via OTIO gaps"


def test_missing_metadata_is_a_hard_error():
    clip = otio.schema.Clip(name="bare")
    with pytest.raises(MetadataContractError):
        parse_clip_meta(clip)
