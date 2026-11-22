#!/usr/bin/env python3
"""Final v6 render — caption-truth v6 + baked zoom, full 657.7 s master.

Identical chain to the accepted v5 render (same base master_v3_blurred.mp4,
same wm-blur + zoom + cards + audio mux, same encoder params). The ONLY
change: the ASS is subtitles_v6.ass — captions rebuilt on the master clock
(Gladia master ASR + LLM supervisor + committed caption_v4 laws).

Chain (engine composition order in otio_engine/filters.py):
  chunk 0   wm-blur (redaction BEFORE zoom) -> split -> zoom branch
            [0, 7.985 s) pre-scale 1472x828 lanczos + zoompan ease 1.0->1.15
            over frames 0..108 then hold -> passthrough branch [7.985, 80)
            -> concat -> ass (v6) -> cards
  chunks1-8 wm-blur -> ass (v6) -> cards
  concat 9 chunks -> mux audio_v3.m4a -> master_postfx_v6.mp4

Idempotent per phase with .done markers; safe against sandbox 10-min kills.
Usage: render_final_v6.py chunk N | concat | mux | all
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

WORK = Path("/home/z/my-project/pipeline_run/final_state")
SRC = Path("/home/z/my-project/pipeline_run/v3/master_v3_blurred.mp4")
AUDIO = Path("/home/z/my-project/pipeline_run/v3/audio_v3.m4a")
ASS = WORK / "subtitles_v6.ass"
BURNED = WORK / "master_v6_burned.mp4"
OUT = Path("/home/z/my-project/download/eustis_final_720p_v6.mp4")
LOG = WORK / "burn_v6.log"

BURN_CHUNK_S = 80.0
FPS = 30
W, H = 1280, 720
ZOOM_END_S = 7.985  # raw_1 duration in master timeline (239.55 frames @30)
ZOOM_ANCHOR_S = 3.6
ZOOM_MAX = 1.15


def log(msg: str) -> None:
    print(msg, flush=True)


def _sh(cmd: list[str], tag: str) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{tag} failed: {r.stderr[-600:]}")


def probe_dur(p: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
        capture_output=True,
        text=True,
    )
    return float(r.stdout.strip())


def to_sec(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def fmt(sec: float) -> str:
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def chunk_ass(ci: int, dur: float, t0: float) -> Path:
    """Flash-law chunk ASS (global -> local shift; drop fully-outside cues)."""
    kept, dropped = 0, 0
    out_lines, header = [], []
    for ln in ASS.read_text(encoding="utf-8").splitlines():
        if ln.startswith("Dialogue: "):
            parts = ln.split(",", 9)
            s, e = to_sec(parts[1]), to_sec(parts[2])
            if e <= t0 + 0.005 or s >= t0 + dur:
                dropped += 1
                continue
            ls, le = max(0.0, s - t0), min(dur + 0.4, e - t0)
            if le - ls < 0.05:
                dropped += 1
                continue
            out_lines.append(",".join([parts[0], fmt(ls), fmt(le)] + parts[3:]))
            kept += 1
        elif ln.strip():
            header.append(ln)
    path = WORK / f"captions_v6_chunk_{ci:02d}.ass"
    path.write_text("\n".join(header + out_lines) + "\n", encoding="utf-8")
    log(f"chunk ASS {ci}: kept {kept}, dropped {dropped} (flash law)")
    return path


def zoom_expr() -> str:
    f_anchor = int(round(ZOOM_ANCHOR_S * FPS))
    f_end = int(round(ZOOM_END_S * FPS))
    core = f"if(lte(on,{f_end}),1.1500,1.0000)"
    return f"if(lte(on,{f_anchor}),1.0000+(+0.1500)*(on-0)/{f_anchor},{core})"


def burn_chunk(ci: int) -> None:
    t0 = ci * BURN_CHUNK_S
    mark = WORK / f"burn_v6_{ci:02d}.done"
    part = WORK / f"burn_v6_{ci:02d}.mp4"
    if mark.exists() and part.exists():
        log(f"burn {ci} cached")
        return
    total = probe_dur(SRC)
    dur = min(BURN_CHUNK_S, total - t0)
    ass = chunk_ass(ci, dur, t0)
    esc = str(ass).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")

    wm = json.loads(Path("/home/z/my-project/pipeline_run/v3/wm_spans_v3.json").read_text())[
        "spans"
    ]
    # committed v4 law: median box across all wm spans
    import numpy as np

    arr = np.array([sp["box"] for sp in wm])
    bx, by, bw, bh = [int(v) for v in np.median(arr, axis=0)]
    bx, by = max(0, bx - 2), max(0, by - 2)
    bw = min(W - bx, bw + 4)
    bh = min(H - by, bh + 4)

    cards = json.loads(Path("/home/z/my-project/pipeline_run/v3/cards_meta_v3.json").read_text())[
        "card_times"
    ]

    fc = (
        f"[0:v]split=2[bgw][wmw];"
        f"[wmw]crop={bw}:{bh}:{bx}:{by},gblur=sigma=20[wmb];"
        f"[bgw][wmb]overlay={bx}:{by}:shortest=1[v0];"
    )
    if ci == 0:
        # engine law: redaction BEFORE zoom; cards/captions AFTER zoom
        pw, ph = int(W * ZOOM_MAX) // 2 * 2, int(H * ZOOM_MAX) // 2 * 2
        fc += (
            f"[v0]split=2[zp][pp];"
            f"[zp]trim=end={ZOOM_END_S:.3f},setpts=PTS-STARTPTS,fps={FPS},"
            f"scale={pw}:{ph}:flags=lanczos,"
            f"zoompan=z='{zoom_expr()}':x='iw/2-(iw/zoom/2)':"
            f"y='ih/2-(ih/zoom/2)':d=1:s={W}x{H}:fps={FPS},"
            f"setsar=1,format=yuv420p[z1];"
            f"[pp]trim=start={ZOOM_END_S:.3f},setpts=PTS-STARTPTS,"
            f"setsar=1,format=yuv420p[p1];"
            f"[z1][p1]concat=n=2:v=1:a=0[vm];"
        )
        cur = "[vm]"
    else:
        cur = "[v0]"
    fc += f"{cur}ass='{esc}'[subbed];"
    prev = "[subbed]"
    inputs = []
    for png, g_t0 in cards:
        lt0 = g_t0 - t0
        t_end = min(lt0 + 6.5, dur - 0.2)
        if t_end <= 0.5 or lt0 >= dur:
            continue
        idx = len(inputs) + 1
        inputs.append((Path("/home/z/my-project/pipeline_run/v3") / png, idx))
        fc += (
            f"[{idx}:v]format=rgba[c{idx}];"
            f"{prev}[c{idx}]overlay=0:0:eof_action=repeat:"
            f"enable='between(t,{max(0.0, lt0):.2f},{t_end:.2f})'[v{idx}];"
        )
        prev = f"[v{idx}]"
    fc += f"{prev}format=yuv420p[vout]"

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{t0:.3f}",
        "-t",
        f"{dur:.3f}",
        "-i",
        str(SRC),
    ]
    for png, _i in inputs:
        cmd += ["-i", str(png)]
    cmd += [
        "-filter_complex",
        fc,
        "-map",
        "[vout]",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "superfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-t",
        f"{dur:.3f}",
        str(part),
    ]
    _sh(cmd, f"burn v6 chunk {ci}")
    mark.write_text("ok")
    log(
        f"burn v6 chunk {ci + 1}/9 done ({dur:.1f}s, {len(inputs)} cards, "
        f"{'zoom+v6caps' if ci == 0 else 'v6caps'})"
    )


def concat() -> None:
    total = probe_dur(SRC)
    n = max(1, -(-int(total * 1000) // int(BURN_CHUNK_S * 1000)))
    parts = []
    for ci in range(n):
        part = WORK / f"burn_v6_{ci:02d}.mp4"
        if not ((WORK / f"burn_v6_{ci:02d}.done").exists() and part.exists()):
            raise SystemExit(f"burn chunk {ci} missing")
        parts.append(part)
    # v5 committed law: mp4 edit lists break concat-demuxer -c copy timestamp
    # offsets; go via mpegts (annexb, no edit lists) then wrap back — the
    # plain mp4 concat produces a collapsed-timeline file (verified: 19729
    # packets all pts<80s). Deterministic offsets via +genpts.
    ts_files = []
    for ci, part in enumerate(parts):
        ts = WORK / f"burn_v6_{ci:02d}.ts"
        if not ts.exists() or ts.stat().st_size == 0:
            _sh(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(part),
                    "-c",
                    "copy",
                    "-bsf:v",
                    "h264_mp4toannexb",
                    "-f",
                    "mpegts",
                    str(ts),
                ],
                f"ts wrap {ci}",
            )
        ts_files.append(ts)
    all_ts = WORK / "burn_v6_all.ts"
    with all_ts.open("wb") as out:
        for ts in ts_files:
            out.write(ts.read_bytes())
    _sh(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-fflags",
            "+genpts",
            "-i",
            str(all_ts),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(BURNED),
        ],
        "v6 concat wrap",
    )
    log(f"concat: {probe_dur(BURNED):.2f}s, {BURNED.stat().st_size / 1e6:.1f} MB")


def mux() -> None:
    if not BURNED.exists():
        raise SystemExit("concat first")
    _sh(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(BURNED),
            "-i",
            str(AUDIO),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-shortest",
            "-movflags",
            "+faststart",
            str(OUT),
        ],
        "v6 mux",
    )
    d_v, d_f = probe_dur(BURNED), probe_dur(OUT)
    if abs(d_f - d_v) > 1.0:
        raise SystemExit(f"duration drifted {abs(d_f - d_v):.2f}s — refusing")
    log(f"muxed: {OUT.name} {d_f:.2f}s, {OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    if phase == "chunk":
        burn_chunk(int(sys.argv[2]))
    elif phase == "concat":
        concat()
    elif phase == "mux":
        mux()
    elif phase == "all":
        total = probe_dur(SRC)
        n = max(1, -(-int(total * 1000) // int(BURN_CHUNK_S * 1000)))
        for ci in range(n):
            burn_chunk(ci)
        concat()
        mux()
    else:
        raise SystemExit(f"usage: {sys.argv[0]} chunk N | concat | mux | all")
