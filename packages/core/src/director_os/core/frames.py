"""Frame-grab service — random-access single-frame JPEG extraction.

The frame-by-frame law (owner, 2026-11-23): the editor — human OR agent —
must be able to address ANY frame of the media cheaply. "Select things on
frame X and fix only that frame" requires frame X as an image, not a
re-render of the whole video. This module is the shared primitive behind:

- ``GET /assets/{id}/frame``   — the Studio scrubber / frame-level timeline
- the agent ``frame.grab`` tool — evidence-grounded, frame-scoped edits

Efficiency contract (live editing must not melt the machine):

- ONE ffmpeg decode per (frame, width): accurate input seek (``-ss`` before
  ``-i`` decodes forward from the previous keyframe to the exact timestamp),
  single frame out, JPEG bytes only — never a full passthrough
- mtime-LRU disk cache per asset (default 600 JPEGs): scrubbing back and
  forth is free; eviction runs in oldest-first tranches so a burst of grabs
  does not thrash
- never upscales: a width >= the source's natural width decodes at native
  size (no wasted pixels, no soft frames)
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
from pathlib import Path

DEFAULT_FRAME_WIDTH = 640
MIN_FRAME_WIDTH = 64
MAX_FRAME_WIDTH = 1920
DEFAULT_MAX_CACHE_FILES = 600
_EVICT_TRANCHE_FRACTION = 10  # evict 10% over the cap in one pass
_GRAB_TIMEOUT_S = 60.0


class FrameGrabError(RuntimeError):
    pass


def clamp_width(width: int | None) -> int | None:
    """Clamp a requested width into range; None stays None (native decode)."""
    if width is None:
        return None
    return max(MIN_FRAME_WIDTH, min(MAX_FRAME_WIDTH, int(width)))


def frame_cache_path(cache_dir: Path, t_s: float, width: int | None) -> Path:
    """Deterministic cache key for one (time, width) frame grab."""
    w = "native" if width is None else f"w{width}"
    return Path(cache_dir) / f"t{int(round(max(0.0, t_s) * 1000.0)):09d}{w}.jpg"


def grab_frame(
    src: Path | str,
    t_s: float,
    *,
    cache_dir: Path | str,
    width: int | None = DEFAULT_FRAME_WIDTH,
    max_files: int = DEFAULT_MAX_CACHE_FILES,
) -> Path:
    """Extract ONE frame at ``t_s`` seconds as a cached JPEG; returns its path.

    ``width=None`` decodes at native size; an explicit width is clamped and
    skipped entirely when it would upscale. Repeated grabs of the same
    (t, width) are served from the LRU cache with zero ffmpeg cost.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FrameGrabError("ffmpeg not available on PATH")
    src = Path(src)
    if not src.exists():
        raise FrameGrabError(f"media missing: {src}")

    t = max(0.0, float(t_s))
    w = clamp_width(width)
    cache_dir = Path(cache_dir)
    out = frame_cache_path(cache_dir, t, w)
    if out.exists():
        out.touch()  # LRU refresh — this frame was just wanted again
        return out

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part.jpg")
    cmd = [
        ffmpeg,
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{t:.3f}",
        "-i",
        str(src),
        "-frames:v",
        "1",
    ]
    if w is not None:
        cmd += ["-vf", f"scale={w}:-2"]
    cmd += ["-q:v", "3", str(tmp)]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=_GRAB_TIMEOUT_S)  # noqa: S603
    except subprocess.TimeoutExpired as e:
        raise FrameGrabError(
            f"frame grab at {t:.3f}s timed out after {_GRAB_TIMEOUT_S:.0f}s"
        ) from e
    if r.returncode != 0:
        detail = r.stderr[-300:]
        if not tmp.exists() or tmp.stat().st_size == 0:
            # seek past EOF: ffmpeg writes nothing and exits nonzero — a
            # clear error beats a silently black frame
            raise FrameGrabError(f"no frame at {t:.3f}s (past end of media?) [{detail[-120:]}]")
        raise FrameGrabError(f"frame grab at {t:.3f}s failed: {detail}")
    if not tmp.exists() or tmp.stat().st_size == 0:
        raise FrameGrabError(f"no frame at {t:.3f}s (past end of media?)")
    tmp.replace(out)
    _evict_lru(cache_dir, max_files)
    return out


def _evict_lru(cache_dir: Path, max_files: int) -> None:
    """Oldest-first eviction in tranches — scrub bursts never thrash."""
    try:
        files = sorted(
            (p for p in cache_dir.glob("*.jpg") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
        )
    except OSError:
        return  # cache health never breaks a grab
    excess = len(files) - max_files
    if excess <= 0:
        return
    tranche = excess + max(len(files) // _EVICT_TRANCHE_FRACTION, 1)
    for p in files[:tranche]:
        with contextlib.suppress(OSError):
            p.unlink()  # racing with another grab: the loser keeps the file
