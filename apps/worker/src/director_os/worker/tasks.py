"""Worker tasks — the CPU render fleet entrypoints (PLAN §2 layer 4).

Render fan-out: segments are independent render units (ADR-0003), so the
orchestrator task splits a DAG into per-segment `render_segment_task` jobs
(one ffmpeg encode each) and a final `concat_task` that runs `concat -c copy`
over the ordered chunk keys. Any worker that dies mid-encode only loses its
own segment — everything already in the cache is never re-encoded.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

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


@celery_app.task(name="directoros.advance_job")
def advance_job_task(
    job_id: str,
    project_id: str,
    source: str,
    database_url: str,
    custom_headers: dict[str, str] | None = None,
) -> dict:
    """Resume/advance one durable pipeline job (stages persist their products).

    `custom_headers` is optional and positional-safe: tasks queued by an
    older gateway carry four args and still unpack. See
    `worker.pipeline` for the stage contract.
    """
    from director_os.worker.pipeline import advance_job

    return advance_job(job_id, project_id, source, database_url, custom_headers=custom_headers)


@celery_app.task(name="directoros.replan_job")
def replan_job_task(
    project_id: str,
    source: str,
    database_url: str,
    custom_headers: dict[str, str] | None = None,
) -> dict:
    """Re-run Stage 6+ after a CaseDossier refresh (new source attached).

    Ingest + ASR short-circuit on the persisted domain rows — the video is
    never re-downloaded and Gladia never re-runs; only planning, render,
    subtitles and QA re-execute against the refreshed dossier.
    """
    from director_os.worker.pipeline import advance_job

    return advance_job(
        None, project_id, source, database_url, force_replan=True, custom_headers=custom_headers
    )


# ---------------------------------------------------------------------------
# segment-parallel render fleet (M1 close-out: issue #9 parallel rendering)
# ---------------------------------------------------------------------------


@celery_app.task(name="directoros.render_segment")
def render_segment_task(spec_json: dict[str, Any], src_path: str | None, segment_key: str) -> dict:
    """Encode ONE segment into the shared segment cache.

    `spec_json` is a serialized `SegmentRenderSpec` (Pydantic, validated on
    arrival — a malformed payload never reaches ffmpeg). The key was computed
    by the compiler; workers re-verify it matches the spec so a stale
    orchestrator cannot poison the cache.
    """
    from director_os.otio_engine.renderer import render_one
    from director_os.otio_engine.segment_cache import SegmentCache, SegmentRenderSpec, segment_hash

    spec = SegmentRenderSpec.model_validate(spec_json)
    actual = segment_hash(spec)
    if actual != segment_key:
        raise ValueError(f"segment key mismatch: orchestrator={segment_key} recomputed={actual}")

    cache = SegmentCache(get_segments_dir())
    seg = _make_dag_node(segment_key, spec, src_path, cache)
    path = render_one(seg, cache)
    return {"key": segment_key, "chunk": str(path), "bytes": path.stat().st_size}


@celery_app.task(name="directoros.concat_segments")
def concat_task(ordered_keys: list[str], out_path: str) -> dict:
    """Lossless `concat -c copy` over cache chunks in DAG order."""
    from director_os.otio_engine.segment_cache import SegmentCache

    cache = SegmentCache(get_segments_dir())
    parts = [cache.path_for(k) for k in ordered_keys]
    missing = [str(p) for p in parts if not p.exists()]
    if missing:
        raise FileNotFoundError(f"segment chunks missing from cache: {missing[:3]}")
    final = concat_cached(parts, Path(out_path))
    return {"final": str(final), "segments": len(parts)}


def _make_dag_node(key: str, spec, src_path, cache):
    from director_os.otio_engine.compiler import SegmentRender

    return SegmentRender(
        key=key,
        spec=spec,
        src_path=src_path,
        out_path=cache.path_for(key),
        dur_s=spec.source_dur_s,
    )


def get_segments_dir() -> Path:
    from director_os.core.config import get_settings

    d = get_settings().segments_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def concat_cached(parts: list[Path], out_path: Path) -> Path:
    """FFmpeg concat demuxer with `-c copy` — identical to otio_engine.renderer."""
    from director_os.otio_engine.renderer import concat_segments

    return concat_segments(parts, out_path)
