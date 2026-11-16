"""Worker tasks — the CPU render fleet entrypoints (PLAN §2 layer 4)."""

from __future__ import annotations

import subprocess

from director_os.worker.celery_app import celery_app

ALLOWED_BINARIES = {"ffmpeg", "ffprobe"}


@celery_app.task(name="directoros.ping")
def ping() -> str:
    """Liveness probe for the fleet."""
    return "pong"


@celery_app.task(name="directoros.run_ffmpeg")
def run_ffmpeg(cmd: list[str]) -> dict:
    """Execute an ffmpeg/ffprobe command built by the OTIO compiler.

    Security: only whitelisted binaries; no shell interpolation.
    """
    if not cmd or cmd[0] not in ALLOWED_BINARIES:
        raise ValueError(f"worker only runs {sorted(ALLOWED_BINARIES)}, got: {cmd[:1]}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {r.stderr[-1000:]}")
    return {"rc": 0, "stderr_tail": r.stderr[-500:]}
