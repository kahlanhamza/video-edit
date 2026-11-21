"""Pro-polish mutation tools (set_zoom / set_freeze_callout / set_audio_mode).

Hermetic: a real OTIO timeline from build_timeline, no media, no network.
Contract read-back goes through RawClipMeta.model_validate so the tools can
never write metadata the compiler would reject.
"""

from __future__ import annotations

import pytest
from director_os.agent_runtime import (
    DirectorToolbox,
    ToolDispatchError,
    ToolOperationError,
    build_director_registry,
)
from director_os.otio_engine.contract import NarratorClipMeta, RawClipMeta
from director_os.otio_engine.timeline import (
    MASTER_TRACK,
    ClipSpec,
    GapSpec,
    TimelineSpec,
    TrackSpec,
    build_timeline,
)

PROJECT = "proj_polish_test"


@pytest.fixture()
def timeline():
    return build_timeline(
        TimelineSpec(
            name="polish_case",
            fps=30,
            width=1280,
            height=720,
            tracks=[
                TrackSpec(
                    name=MASTER_TRACK,
                    items=[
                        ClipSpec(
                            name="raw_1",
                            source_start_s=0.0,
                            dur_s=10.0,
                            meta=RawClipMeta(asset_id="bwc_1"),
                        ),
                        ClipSpec(
                            name="raw_2",
                            source_start_s=12.0,
                            dur_s=8.0,
                            meta=RawClipMeta(asset_id="bwc_1"),
                        ),
                        ClipSpec(
                            name="narr_1",
                            source_start_s=0.0,
                            dur_s=5.0,
                            meta=NarratorClipMeta(
                                script_text="Arrival at 18:11.",
                                broll_asset_id="bwc_1",
                            ),
                        ),
                    ],
                ),
                TrackSpec(
                    name="overlay",
                    items=[
                        GapSpec(name="gap", dur_s=18.0),
                    ],
                ),
            ],
        )
    )


@pytest.fixture()
def registry(timeline):
    return build_director_registry(DirectorToolbox(timeline, PROJECT, resolution=(1280, 720)))


def _raw_meta(timeline, index: int) -> RawClipMeta:
    item = list(timeline.tracks[0])[index]
    return RawClipMeta.model_validate(item.metadata["director_os"])


# ---------------------------------------------------------------- zoom
def test_zoom_curve_roundtrip(timeline, registry):
    r = registry.dispatch(
        "timeline.set_zoom",
        {"item_index": 0, "keyframes": [[2.0, 1.15], [0.0, 1.05]]},
    )
    assert r["ok"] is True
    kf = _raw_meta(timeline, 0).zoom_keyframes
    assert [(k.t_s, k.zoom_factor) for k in kf] == [(0.0, 1.05), (2.0, 1.15)]  # sorted
    assert "peak 1.15x" in r["detail"]


def test_zoom_clear_and_out_of_range(timeline, registry):
    registry.dispatch("timeline.set_zoom", {"item_index": 0, "keyframes": [[1.0, 1.2]]})
    r = registry.dispatch("timeline.set_zoom", {"item_index": 0, "keyframes": []})
    assert _raw_meta(timeline, 0).zoom_keyframes == []
    assert "peak 1.00x" in r["detail"]

    with pytest.raises(ToolOperationError, match="outside clip"):
        registry.dispatch("timeline.set_zoom", {"item_index": 0, "keyframes": [[11.0, 1.1]]})
    with pytest.raises(ToolOperationError, match="invalid keyframe"):
        registry.dispatch("timeline.set_zoom", {"item_index": 0, "keyframes": [[1.0, 0.8]]})
    # the failed attempts must not have mutated the contract
    assert _raw_meta(timeline, 0).zoom_keyframes == []


def test_zoom_on_narration_rejected(timeline, registry):
    with pytest.raises(ToolOperationError, match="not a raw clip"):
        registry.dispatch("timeline.set_zoom", {"item_index": 2, "keyframes": []})


# ---------------------------------------------------------------- freeze
def test_freeze_callout_roundtrip_and_clear(timeline, registry):
    r = registry.dispatch(
        "timeline.set_freeze_callout",
        {
            "item_index": 1,
            "at_local_s": 3.0,
            "hold_s": 4.0,
            "dim_alpha": 0.5,
            "spotlight_bbox": [100, 200, 300, 80],
        },
    )
    assert r["ok"] is True and "spotlight [100, 200, 300, 80]" in r["detail"]
    fc = _raw_meta(timeline, 1).freeze_callout
    assert fc is not None and fc.freeze_at_s == 3.0 and fc.hold_s == 4.0
    assert fc.spotlight_bbox is not None and fc.spotlight_bbox.w == 300

    r = registry.dispatch(
        "timeline.set_freeze_callout", {"item_index": 1, "at_local_s": 0.0, "hold_s": 0}
    )
    assert r["ok"] is True and "cleared" in r["detail"]
    assert _raw_meta(timeline, 1).freeze_callout is None


def test_freeze_validation(timeline, registry):
    with pytest.raises(ToolOperationError, match="outside clip"):
        registry.dispatch(
            "timeline.set_freeze_callout",
            {"item_index": 0, "at_local_s": 10.5, "hold_s": 2.0},
        )
    with pytest.raises(ToolOperationError, match="exceeds canvas"):
        registry.dispatch(
            "timeline.set_freeze_callout",
            {"item_index": 0, "at_local_s": 1.0, "spotlight_bbox": [1200, 0, 300, 80]},
        )


# ---------------------------------------------------------------- audio mode
def test_audio_mode_switch(timeline, registry):
    r = registry.dispatch("timeline.set_audio_mode", {"item_index": 0, "mode": "silence"})
    assert r["ok"] is True and "loudnorm_48k -> silence" in r["detail"]
    assert _raw_meta(timeline, 0).audio_mode == "silence"


def test_audio_mode_rejects_unknown(timeline, registry):
    with pytest.raises(ToolDispatchError):
        registry.dispatch("timeline.set_audio_mode", {"item_index": 0, "mode": "boost"})
    assert _raw_meta(timeline, 0).audio_mode == "loudnorm_48k"


def test_registry_now_has_15_tools(registry):
    names = set(registry.names())
    assert {
        "timeline.set_zoom",
        "timeline.set_freeze_callout",
        "timeline.set_audio_mode",
    } <= names
    assert len(names) == 15  # 12 legacy + 3 polish
