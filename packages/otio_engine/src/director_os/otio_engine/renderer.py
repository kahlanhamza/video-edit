"""Segment renderer + cached DAG execution (PLAN §7 Stage 9).

Every segment is rendered with the audio law flags (`-ar 48000 -ac 2` after any
`-af` chain) and identical codec params, so the final concat is a lossless
`-c copy`. Cache hits skip the encode entirely.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from director_os.core.ffmpeg_args import assert_audio_law, exact_slice_args
from director_os.core.qa_probe import has_audio_stream
from director_os.otio_engine.compiler import RenderDAG
from director_os.otio_engine.filters import compose_filter_complex, plain_video_chain
from director_os.otio_engine.segment_cache import SegmentCache, SegmentRenderSpec


class RenderError(RuntimeError):
    pass


@dataclass
class RenderReport:
    rendered: int
    cached: int
    final_path: Path
    total_dur_s: float


def build_segment_cmd(
    spec: SegmentRenderSpec,
    src_path: str | None,
    out_path: str | Path,
    src_has_audio: bool | None = None,
) -> list[str]:
    v = spec.video
    a = spec.audio
    dur = spec.source_dur_s
    cmd: list[str] = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]

    graph, uses_complex = compose_filter_complex(spec)
    n_card_inputs = len(spec.cards)

    if spec.kind == "narration" and src_path is None:
        # narration on black (TTS audio lands in M2; broll visuals per contract)
        cmd += ["-f", "lavfi", "-i", f"color=c=black:size={v.w}x{v.h}:rate={v.fps}"]
        src_has_audio = False
    elif src_path is None:
        raise RenderError("raw segment without a source path")
    else:
        if src_has_audio is None:
            src_has_audio = has_audio_stream(src_path)
        # guardrail 2: exact slicing, never -to
        cmd += exact_slice_args(spec.source_start_s, spec.source_start_s + dur)
        cmd += ["-i", str(src_path)]

    n_inputs = 1 + n_card_inputs
    for card in spec.cards:
        cmd += ["-loop", "1", "-i", card.png_path]

    # ---- audio routing: raw clips carry their own audio; everything else rides a
    #      silent 48 kHz stereo bed so the stream map never goes missing.
    silence_needed = (spec.kind == "narration") or not src_has_audio
    if silence_needed:
        audio_input_idx = n_inputs
        cmd += [
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r={a.sample_rate}:cl={'stereo' if a.channels == 2 else 'mono'}",
        ]
    else:
        audio_input_idx = 0

    # ---- maps
    cmd += ["-map", "[vout]" if uses_complex else "0:v"]
    cmd += ["-map", f"{audio_input_idx}:a"]

    # ---- video path
    if uses_complex:
        cmd += ["-filter_complex", graph]
    else:
        cmd += ["-vf", plain_video_chain(spec)]

    # ---- audio path (guardrail 1: -af precedes the audio-law flags)
    if spec.audio_filters and spec.kind == "raw":
        cmd += ["-af", ",".join(spec.audio_filters)]
    cmd += ["-ar", str(a.sample_rate), "-ac", str(a.channels)]
    cmd += [
        "-c:v",
        v.codec,
        "-preset",
        v.preset,
        "-crf",
        str(v.crf),
        "-pix_fmt",
        v.pix_fmt,
        "-r",
        str(v.fps),
    ]
    cmd += ["-c:a", a.codec, "-b:a", f"{a.bitrate_kbps}k"]
    cmd += ["-t", f"{dur:.6f}", str(out_path)]

    assert_audio_law(cmd)
    return cmd


def render_segment(spec: SegmentRenderSpec, src_path: str | None, out_path: str | Path) -> None:
    cmd = build_segment_cmd(spec, src_path, out_path)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RenderError(f"segment render failed ({out_path}):\n{r.stderr[-2000:]}")


def concat_segments(parts: list[Path], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lst = out_path.with_suffix(".concat.txt")
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    r = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(lst),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    lst.unlink(missing_ok=True)
    if r.returncode != 0:
        raise RenderError(f"concat failed:\n{r.stderr[-2000:]}")
    return out_path


def render_dag(
    dag: RenderDAG,
    cache: SegmentCache,
    log=print,
) -> RenderReport:
    rendered = cached = 0
    parts: list[Path] = []
    for seg in dag.segments:
        hit = cache.get(seg.key)
        if hit is not None:
            cached += 1
            log(f"  [cache] {seg.key[:12]} ({seg.dur_s:.1f}s)")
        else:
            tmp = cache.root / f"{seg.key}.tmp.mp4"
            render_segment(seg.spec, seg.src_path, tmp)
            cache.put(tmp, seg.key)
            rendered += 1
            log(f"  [render] {seg.key[:12]} ({seg.dur_s:.1f}s)")
        parts.append(cache.path_for(seg.key))
    concat_segments(parts, dag.final_path)
    return RenderReport(
        rendered=rendered,
        cached=cached,
        final_path=dag.final_path,
        total_dur_s=dag.total_dur_s,
    )
