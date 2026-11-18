"""Stage 10 — Creator Ship Pack (mp4 + chapters + srt + SEO + thumbnails + otio)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from director_os.otio_engine.contract import CardMeta, NarratorClipMeta, RawClipMeta
from director_os.otio_engine.timeline import ClipSpec, TimelineSpec, TrackSpec, build_timeline
from director_os.worker.ship_pack import (
    _fmt_ts,
    build_seo,
    build_ship_pack,
    chapter_markers,
    first_last_frame_luma,
)


@pytest.fixture()
def deps(tmp_path: Path):
    class Settings:
        output_dir = tmp_path / "output"

    class Deps:
        settings = Settings()

    Deps.settings.output_dir.mkdir(parents=True, exist_ok=True)
    return Deps()


@pytest.fixture()
def timeline():
    spec = TimelineSpec(
        name="t",
        fps=30,
        width=320,
        height=240,
        tracks=[
            TrackSpec(
                name="V1_master",
                items=[
                    ClipSpec(
                        name="raw_1", source_start_s=0.0, dur_s=6.0, meta=RawClipMeta(asset_id="a1")
                    ),
                    ClipSpec(
                        name="vo_1",
                        source_start_s=0.0,
                        dur_s=2.0,
                        meta=NarratorClipMeta(script_text="The pursuit begins at the alley."),
                    ),
                ],
            ),
            TrackSpec(
                name="V2_overlay_cards",
                items=[
                    ClipSpec(
                        name="card_1",
                        source_start_s=1.0,
                        dur_s=3.0,
                        meta=CardMeta(template="location_intro", fields={"title": "CASE FILE"}),
                    )
                ],
            ),
        ],
    )
    return build_timeline(spec)


def test_fmt_ts():
    assert _fmt_ts(0) == "0:00"
    assert _fmt_ts(65) == "1:05"
    assert _fmt_ts(3700) == "1:01:40"


def test_chapter_markers_cards_and_narration(timeline):
    markers = chapter_markers(timeline)
    labels = [label for _, label in markers]
    assert any("CASE FILE" in label for label in labels)
    assert any("Narration:" in label for label in labels)
    assert markers[0][0] == 0.0, "YouTube chapters require the first entry at 0:00"


def test_first_last_frame_luma_on_real_video(fixture_video):
    first, last = first_last_frame_luma(fixture_video)
    assert 0.0 <= first <= 255.0
    assert 0.0 <= last <= 255.0
    # testsrc2 frames are colorful — never black
    assert first > 6.0 and last > 6.0


def test_build_seo_deterministic():
    seo = build_seo(
        project_title="Eustis PD",
        transcript={"words": [{"text": "unit"}, {"text": "twelve"}]},
        insights=[{"kind": "event", "description": "Officer draws TASER", "confidence": 0.9}],
        card_subtitle="AXON BODY 3 — 2022-07-04T18:11:55-04:00 — Eustis, FL",
        chapters=[(0.0, "Cold open"), (1.0, "CASE FILE")],
        duration_s=120.0,
    )
    assert "Eustis" in seo["title"]
    assert "Officer draws TASER" in seo["description"]
    assert seo["keywords"]
    assert seo["generator"] == "deterministic_v1"


def test_build_ship_pack_end_to_end(fixture_video, timeline, deps):
    out_dir = Path(deps.settings.output_dir) / "pipeline" / "p1"
    out_dir.mkdir(parents=True, exist_ok=True)
    srt = out_dir / "subtitles.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:02,000\nhello\n", encoding="utf-8")

    pack = build_ship_pack(
        "p1",
        deps,
        timeline,
        fixture_video,
        transcript={"words": [{"text": "copy", "start_s": 0.1, "end_s": 0.3}]},
        insights=[{"kind": "event", "description": "gunshot heard", "confidence": 0.8}],
        enhance_info=None,
        duration_s=8.0,
        project_title="Case File Alpha",
    )

    assert (pack.pack_dir / "final_1080p.mp4").exists()
    assert (pack.pack_dir / "chapters.txt").exists()
    assert "0:00" in (pack.pack_dir / "chapters.txt").read_text()
    assert (pack.pack_dir / "captions.srt").read_text().startswith("1\n")
    assert (pack.pack_dir / "seo.md").exists()
    assert "Case File Alpha" in (pack.pack_dir / "seo.md").read_text()
    assert (pack.pack_dir / "metadata.json").exists()
    manifest = json.loads((pack.pack_dir / "metadata.json").read_text())
    assert manifest["stage"] == "ship_pack"
    assert manifest["verification"]["clean_first_last_frames"] is True
    assert manifest["content_id_risk"]["music_flagged"] is False
    assert manifest["chapters"] and manifest["chapters"][0][0] == 0.0
    thumbs = list((pack.pack_dir / "thumbnails").glob("*.jpg"))
    assert thumbs, "thumbnail candidates must be extracted"


def test_ship_pack_carries_music_risk(fixture_video, timeline, deps):
    class EnhanceInfo:
        enhancer_input = "preclean"
        music_flagged_spans = [(10.0, 25.0)]

    pack = build_ship_pack(
        "p2", deps, timeline, fixture_video, enhance_info=EnhanceInfo(), duration_s=8.0
    )
    assert pack.content_id_risk["music_flagged"] is True
    assert pack.content_id_risk["demucs_used"] is False
    assert "review before upload" in pack.content_id_risk["upload_safe_note"]

    class EnhanceInfoDemucs(EnhanceInfo):
        enhancer_input = "demucs_vocals"

    pack2 = build_ship_pack(
        "p3", deps, timeline, fixture_video, enhance_info=EnhanceInfoDemucs(), duration_s=8.0
    )
    assert pack2.content_id_risk["demucs_used"] is True
    assert "Demucs" in pack2.content_id_risk["upload_safe_note"]


def test_ship_pack_idempotent(fixture_video, timeline, deps):
    args = dict(
        transcript={"words": []},
        insights=[],
        duration_s=8.0,
    )
    p1 = build_ship_pack("p4", deps, timeline, fixture_video, **args)
    sha1 = json.loads((p1.pack_dir / "metadata.json").read_text())["final_sha256"]
    p2 = build_ship_pack("p4", deps, timeline, fixture_video, **args)
    sha2 = json.loads((p2.pack_dir / "metadata.json").read_text())["final_sha256"]
    assert sha1 == sha2
