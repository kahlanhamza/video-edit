"""Demo: keyframed punch-in zoom + freeze-callout on REAL bodycam footage.

Runs the v2 render path (geometry -> blur -> zoom -> freeze -> spotlight -> fps)
over a slice of the Eustis FL bodycam master produced by scripts/e2e_real_video.py.
Zoom ramps 1.0 -> 1.35 over the first 5 s; at t=10 s the picture freezes for 4 s
while audio keeps running and a dim + spotlight overlay highlights the subject.

Usage:
    uv run python scripts/demo_zoom_freeze.py --source <path-to-master.mp4>
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from director_os.core.models import BBox, ZoomKeyframe
from director_os.otio_engine.compiler import compile_timeline
from director_os.otio_engine.contract import FreezeCallout, RawClipMeta
from director_os.otio_engine.renderer import render_dag
from director_os.otio_engine.segment_cache import (
    AudioRenderSpec,
    SegmentCache,
    VideoRenderSpec,
)
from director_os.otio_engine.timeline import (
    MASTER_TRACK,
    ClipSpec,
    TimelineSpec,
    TrackSpec,
    build_timeline,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        default="/home/z/my-project/download/eustis_bodycam_master_720p.mp4",
        help="real-footage master to slice from",
    )
    ap.add_argument("--start", type=float, default=100.0)
    ap.add_argument("--dur", type=float, default=16.0)
    ap.add_argument("--out", default="./output/demo_zoom_freeze")
    args = ap.parse_args()

    src = Path(args.source)
    if not src.exists():
        print(f"source missing: {src}")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = SegmentCache(out_dir / "segments")

    tl = build_timeline(
        TimelineSpec(
            name="zoom-freeze demo — Eustis FL bodycam",
            fps=30,
            width=1280,
            height=720,
            tracks=[
                TrackSpec(
                    name=MASTER_TRACK,
                    kind="Video",
                    items=[
                        ClipSpec(
                            name="eustis_zoom_freeze",
                            source_start_s=args.start,
                            dur_s=args.dur,
                            meta=RawClipMeta(
                                asset_id="demo_master",
                                audio_mode="loudnorm_48k",
                                trim_silence=False,
                                zoom_keyframes=[ZoomKeyframe(t_s=5.0, zoom_factor=1.35)],
                                freeze_callout=FreezeCallout(
                                    freeze_at_s=10.0,
                                    hold_s=4.0,
                                    dim_alpha=0.45,
                                    spotlight_bbox=BBox(x=820, y=230, w=330, h=380),
                                ),
                            ),
                        )
                    ],
                )
            ],
        )
    )

    dag = compile_timeline(
        tl,
        {"demo_master": src},
        cache,
        video=VideoRenderSpec(w=1280, h=720, fps=30, crf=21),
        audio=AudioRenderSpec(),
        out_dir=out_dir,
    )
    report = render_dag(dag, cache, log=print, max_workers=1)
    print(f"[done] {report.final_path} ({report.total_dur_s:.1f}s)")

    final = out_dir / "final.mp4"
    for probe_t, name in [(3.0, "prezoom"), (7.0, "zoomed"), (12.0, "frozen_spotlit")]:
        png = out_dir / f"frame_{name}.png"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{probe_t:.2f}",
                "-i",
                str(final),
                "-frames:v",
                "1",
                str(png),
            ],
            check=True,
        )
        print(f"[frame] t={probe_t}s -> {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
