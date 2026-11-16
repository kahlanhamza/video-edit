"""Full-resolution 1080p demo — the Milestone-1 done-when, end to end.

Builds a documentary-shaped timeline (3 raw BWC segments with privacy blur,
a narration bridge on black, a location-intro card), compiles it, renders
with the parallel segment fleet, concats losslessly, and QA-probes the
master. Run via `make demo-1080p`.

The demo source is generated synthetically (720p, 45s, tone audio) so the
demo needs no network and no FOIA packet. Real footage drops in unchanged.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from director_os.core.config import get_settings
from director_os.core.models import BBox, BlurRegion
from director_os.core.qa_probe import assert_audio_law, probe
from director_os.otio_engine.compiler import compile_timeline
from director_os.otio_engine.contract import CardMeta, NarratorClipMeta, RawClipMeta
from director_os.otio_engine.renderer import render_dag
from director_os.otio_engine.segment_cache import AudioRenderSpec, SegmentCache, VideoRenderSpec
from director_os.otio_engine.timeline import (
    ClipSpec,
    GapSpec,
    TimelineSpec,
    TrackSpec,
    build_timeline,
    save_otio,
)

W, H, FPS = 1920, 1080, 30


def make_demo_source(out: Path, seconds: int = 45) -> Path:
    """Synthetic 'bodycam-like' source: moving patterns + ambient audio bed."""
    if out.exists() and out.stat().st_size > 0:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=1280x720:rate={FPS}:duration={seconds}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=220:duration={seconds},lowpass=f=600",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "26",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-shortest",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


def demo_timeline() -> TimelineSpec:
    return TimelineSpec(
        name="demo_case_1080p",
        fps=FPS,
        width=W,
        height=H,
        tracks=[
            TrackSpec(
                name="V1_master",
                items=[
                    ClipSpec(
                        name="open_bwc",
                        source_start_s=0.0,
                        dur_s=14.0,
                        meta=RawClipMeta(
                            asset_id="bwc_1",
                            audio_mode="loudnorm_48k",
                            blur_regions=[
                                BlurRegion(bbox=BBox(x=1600, y=40, w=260, h=70), sigma=18),
                                BlurRegion(bbox=BBox(x=60, y=940, w=420, h=100), sigma=24),
                            ],
                        ),
                    ),
                    ClipSpec(
                        name="narration_bridge",
                        source_start_s=0.0,
                        dur_s=6.0,
                        meta=NarratorClipMeta(
                            script_text=(
                                "Officer units arrive at the corner of 5th and Market "
                                "ninety seconds after the call came in."
                            ),
                        ),
                    ),
                    ClipSpec(
                        name="contact_bwc",
                        source_start_s=14.0,
                        dur_s=12.0,
                        meta=RawClipMeta(
                            asset_id="bwc_1",
                            audio_mode="loudnorm_48k",
                            blur_regions=[
                                BlurRegion(bbox=BBox(x=1600, y=40, w=260, h=70), sigma=18),
                            ],
                        ),
                    ),
                    ClipSpec(
                        name="close_bwc",
                        source_start_s=26.0,
                        dur_s=10.0,
                        meta=RawClipMeta(asset_id="bwc_1", audio_mode="loudnorm_48k"),
                    ),
                ],
            ),
            TrackSpec(
                name="V2_overlay_cards",
                items=[
                    GapSpec(dur_s=2.0),
                    ClipSpec(
                        name="card_location",
                        source_start_s=0.0,
                        dur_s=5.0,
                        meta=CardMeta(
                            template="location_intro",
                            fields={"title": "5TH & MARKET", "subtitle": "16:42:11 - BWC 1"},
                        ),
                    ),
                    GapSpec(dur_s=24.0),
                    ClipSpec(
                        name="card_case",
                        source_start_s=0.0,
                        dur_s=5.0,
                        meta=CardMeta(
                            template="location_intro",
                            fields={"title": "CASE 26-0417", "subtitle": "FOIA release"},
                        ),
                    ),
                ],
            ),
        ],
    )


def compile_once(out_dir: Path, src: Path, cache: SegmentCache):
    return compile_timeline(
        build_timeline(demo_timeline()),
        assets={"bwc_1": str(src)},
        cache=cache,
        video=VideoRenderSpec(w=W, h=H, fps=FPS, preset="veryfast", crf=20),
        audio=AudioRenderSpec(),
        out_dir=out_dir,
    )


def main() -> int:
    settings = get_settings()
    out_dir = settings.output_dir / "demo_1080p"
    src = make_demo_source(settings.output_dir / "ingest" / "demo_source_720p.mp4")

    print("[1/5] build OTIO timeline (1080p documentary shape)")
    tl = build_timeline(demo_timeline())
    otio_path = save_otio(tl, out_dir / "demo_1080p.otio")

    print("[2/5] compile -> content-hashed segment DAG")
    cache = SegmentCache(settings.segments_dir)
    dag = compile_once(out_dir, src, cache)
    print(f"      {len(dag.segments)} segments, {dag.total_dur_s:.1f}s master")

    print("[3/5] parallel segment render (fleet pool)")
    t0 = time.monotonic()
    report = render_dag(dag, cache)
    first_pass = time.monotonic() - t0
    print(
        f"      rendered={report.rendered} cached={report.cached} "
        f"in {first_pass:.1f}s -> {report.final_path}"
    )

    print("[4/5] zero-change re-render (cache proof)")
    dag2 = compile_once(out_dir, src, cache)
    t0 = time.monotonic()
    report2 = render_dag(dag2, cache)
    cached_pass = time.monotonic() - t0
    print(f"      rendered={report2.rendered} cached={report2.cached} in {cached_pass:.1f}s")

    print("[5/5] QA probe")
    info = probe(report.final_path)
    assert_audio_law(info)
    assert info.width == W and info.height == H, f"expected {W}x{H}, got {info.width}x{info.height}"

    print()
    print(f"FINAL  {report.final_path}")
    print(
        f"       {info.width}x{info.height} @ {info.fps:.0f}fps | "
        f"{info.duration_s:.1f}s | {info.sample_rate}Hz/{info.channels}ch | {info.video_codec}"
    )
    print(f"       timeline: {otio_path}")
    print(
        f"       cache proof: {report2.cached}/{len(dag2.segments)} hits, "
        f"{first_pass:.1f}s -> {cached_pass:.1f}s ({cached_pass / first_pass * 100:.0f}%)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
