"""Integration (PLAN §8 M1 done-when, in miniature):

- OTIO timeline -> compile -> segment-cached render -> concat -c copy -> QA probe
- audio law holds on the final master (48 kHz stereo)
- re-render with zero changes = 100% cache hits
- changing one segment spec (blur sigma) re-renders exactly one segment
"""

from __future__ import annotations

from pathlib import Path

import pytest
from director_os.core.models import BBox, BlurRegion
from director_os.core.qa_probe import assert_audio_law, blackdetect_seconds, probe
from director_os.otio_engine.compiler import compile_timeline
from director_os.otio_engine.contract import CardMeta, RawClipMeta
from director_os.otio_engine.renderer import render_dag
from director_os.otio_engine.segment_cache import (
    AudioRenderSpec,
    SegmentCache,
    VideoRenderSpec,
)
from director_os.otio_engine.timeline import (
    ClipSpec,
    GapSpec,
    TimelineSpec,
    TrackSpec,
    build_timeline,
    save_otio,
)


def build_spec(sigma: int = 12):
    return TimelineSpec(
        name="fixture_case",
        fps=30,
        width=640,
        height=360,
        tracks=[
            TrackSpec(
                name="V1_master",
                items=[
                    ClipSpec(
                        name="seg1",
                        source_start_s=0.0,
                        dur_s=6.0,
                        meta=RawClipMeta(
                            asset_id="a1",
                            blur_regions=[
                                BlurRegion(bbox=BBox(x=10, y=10, w=60, h=40), sigma=sigma)
                            ],
                        ),
                    ),
                    ClipSpec(
                        name="seg2", source_start_s=6.0, dur_s=6.0, meta=RawClipMeta(asset_id="a1")
                    ),
                ],
            ),
            TrackSpec(
                name="V2_overlay_cards",
                items=[
                    GapSpec(dur_s=7.0),
                    ClipSpec(
                        name="card",
                        source_start_s=0.0,
                        dur_s=3.0,
                        meta=CardMeta(
                            template="location_intro",
                            fields={"title": "FIXTURE PD", "subtitle": "Synthetic case file"},
                        ),
                    ),
                ],
            ),
        ],
    )


def render_tree(tmp_path: Path, sigma: int = 12, src: Path | None = None):
    tl = build_timeline(build_spec(sigma))
    otio_path = save_otio(tl, tmp_path / "case.otio")
    cache = SegmentCache(tmp_path / "segments")
    dag = compile_timeline(
        tl,
        assets={"a1": str(src)},
        cache=cache,
        video=VideoRenderSpec(w=640, h=360, fps=30),
        audio=AudioRenderSpec(),
        out_dir=tmp_path / "render",
    )
    return otio_path, cache, dag


def test_full_pipeline_fixture_render(fixture_video: Path, tmp_path):
    otio_path, cache, dag = render_tree(tmp_path, src=fixture_video)
    assert otio_path.exists()

    report = render_dag(dag, cache, log=lambda *_: None)
    assert report.rendered == 2 and report.cached == 0

    final = probe(report.final_path)
    assert final.has_video and final.has_audio
    assert final.duration_s == pytest.approx(12.0, abs=0.9)
    assert final.sample_rate == 48000, "audio law: 48 kHz on the final master"
    assert final.channels == 2, "audio law: stereo on the final master"
    assert_audio_law(final)
    assert blackdetect_seconds(report.final_path) < 1.0


def test_unchanged_timeline_is_fully_cached(fixture_video: Path, tmp_path):
    _, cache, dag = render_tree(tmp_path, src=fixture_video)
    render_dag(dag, cache, log=lambda *_: None)

    # rebuild the SAME spec -> compile -> identical hashes
    _, _, dag2 = render_tree(tmp_path, src=fixture_video)
    assert [s.key for s in dag2.segments] == [s.key for s in dag.segments]
    report2 = render_dag(dag2, cache, log=lambda *_: None)
    assert report2.rendered == 0 and report2.cached == 2


def test_single_spec_change_renders_exactly_one_segment(fixture_video: Path, tmp_path):
    _, cache, dag = render_tree(tmp_path, src=fixture_video)
    render_dag(dag, cache, log=lambda *_: None)
    mtimes_before = {
        s.key: (cache.path_for(s.key).stat().st_mtime_ns if cache.get(s.key) else 0)
        for s in dag.segments
    }

    # one blur sigma tweak -> seg1 key changes, seg2 key stable
    _, _, dag2 = render_tree(tmp_path, sigma=16, src=fixture_video)
    changed = [
        s.key
        for s in dag2.segments
        if mtimes_before.get(s.key)
        != (cache.path_for(s.key).stat().st_mtime_ns if cache.get(s.key) else 0)
    ]
    assert len(dag2.segments) == 2
    report3 = render_dag(dag2, cache, log=lambda *_: None)
    assert report3.rendered == 1 and report3.cached == 1
    assert len(changed) <= 1


def test_exported_otio_reloads_for_nle_roundtrip(fixture_video: Path, tmp_path):
    otio_path, _, dag = render_tree(tmp_path, src=fixture_video)
    from director_os.otio_engine.timeline import load_otio

    tl = load_otio(otio_path)
    from director_os.otio_engine.timeline import MASTER_TRACK

    master = next(t for t in tl.tracks if t.name == MASTER_TRACK)
    assert len(list(master)) == 2
    assert sum(
        (i.source_range.duration.to_seconds() if i.source_range else 0) for i in master
    ) == pytest.approx(12.0)
