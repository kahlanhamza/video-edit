"""`ingest.build_proxies` — 480p web proxy + scrubsheet sprite + 64k ASR audio.

PLAN §3.3 namespace A. Proxies exist so the Director Studio can scrub footage
without streaming the raw master, and ASR gets a small mono audio track.
Pure FFmpeg — no external keys, deterministic outputs, safe to re-run.

Outputs land in `output/pipeline/<project>/assets/<asset_id>/`:
  proxy.mp4      H.264/AAC, height locked to 480, +faststart, 48 kHz stereo
  scrubsheet.jpg fixed-width thumbnail sprite (1 row, ~every 2 s)
  audio64k.m4a   AAC 64 kbps mono 48 kHz (ASR input)
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

from director_os.core.qa_probe import probe

PROXY_HEIGHT = 480
SCRUB_THUMB_H = 60
SCRUB_INTERVAL_S = 2.0
SCRUB_MAX_THUMBS = 40  # cap sprite width for long raws


class ProxyBuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProxyBundle:
    proxy_path: Path
    scrub_path: Path
    audio_path: Path


def _run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise ProxyBuildError(f"ffmpeg failed ({' '.join(cmd[:6])}...): {r.stderr[-800:]}")


def _even(n: int) -> int:
    return max(2, n - n % 2)  # libx264 yuv420p demands even dimensions


def proxy_video_args(height: int = PROXY_HEIGHT) -> list[str]:
    """Scale-to-height filter args for the 480p proxy (even width enforced)."""
    return [
        "-vf",
        f"scale=-2:{height}:flags=bicubic",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "28",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-movflags",
        "+faststart",
    ]


def build_proxies(source: str | Path, out_dir: str | Path) -> ProxyBundle:
    """Build proxy.mp4 + scrubsheet.jpg + audio64k.m4a next to each other."""
    src = Path(source)
    if not src.exists():
        raise FileNotFoundError(f"source missing: {src}")
    info = probe(src)
    if not info.has_video:
        raise ProxyBuildError(f"source has no video stream: {src}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    proxy = out / "proxy.mp4"
    scrub = out / "scrubsheet.jpg"
    audio = out / "audio64k.m4a"

    # output options live after all inputs (ffmpeg parses per-input options first)
    out_opts = [*proxy_video_args(), str(proxy)]
    if info.has_audio:
        _run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), *out_opts])
        _run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(src),
                "-vn",
                "-c:a",
                "aac",
                "-b:a",
                "64k",
                "-ar",
                "48000",
                "-ac",
                "1",
                str(audio),
            ]
        )
    else:
        # silent raw: still produce a playable proxy carrying a silent stereo track
        _run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(src),
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=48000:cl=stereo",
                "-shortest",
                *out_opts,
            ]
        )

    dur = max(info.duration_s, 0.1)
    n = min(SCRUB_MAX_THUMBS, max(1, math.ceil(dur / SCRUB_INTERVAL_S)))
    fps = n / dur  # select one frame every ~2 s
    thumb_w = _even(int(SCRUB_THUMB_H * (info.width or 16) / max(info.height or 9, 1)))
    _run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(proxy),
            "-vf",
            f"fps={fps:.6f},scale={thumb_w}:{SCRUB_THUMB_H}:flags=bicubic,tile=1x{n}",
            "-frames:v",
            "1",
            "-q:v",
            "5",
            str(scrub),
        ]
    )
    for p in (proxy, scrub, audio):
        if not p.exists() or p.stat().st_size == 0:
            raise ProxyBuildError(f"proxy artifact missing/empty: {p.name}")
    return ProxyBundle(proxy_path=proxy, scrub_path=scrub, audio_path=audio)
