"""SSE job progress — FastAPI Server-Sent Events stream (PLAN §8 M2).

`GET /projects/{project_id}/jobs/{job_id}/events` streams pipeline progress as
SSE frames, polling the durable `jobs` row:

    event: progress
    data: {"stage": "render", "progress_pct": 60.0, "error": null}

    event: done
    data: {"stage": "human_gate", "progress_pct": 100.0, ...}

Terminal = progress_pct >= 100 (pipeline parks at human_gate 100) OR an error
message present. A hard `max_seconds` cap guarantees the stream always closes
(so tests and proxies never hang); a closed DB row ends the stream.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator

from director_os.core.db import JobRow
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

TERMINAL_PROGRESS = 100.0
DEFAULT_POLL_S = 0.25
DEFAULT_MAX_S = 20.0


def _job_snapshot(s: Session, job_id: str) -> dict:
    row = s.execute(select(JobRow).where(JobRow.id == job_id)).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="job not found")
    return {
        "job_id": row.id,
        "project_id": row.project_id,
        "stage": row.current_stage,
        "progress_pct": round(float(row.progress_pct or 0.0), 1),
        "error": row.error_message,
    }


def _is_terminal(snapshot: dict) -> bool:
    return snapshot["progress_pct"] >= TERMINAL_PROGRESS or snapshot["error"] is not None


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"


async def job_progress_stream(
    job_id: str,
    session_factory,
    poll_s: float = DEFAULT_POLL_S,
    max_seconds: float = DEFAULT_MAX_S,
) -> AsyncIterator[str]:
    """Yield SSE frames until the job goes terminal or the cap expires."""
    loop = asyncio.get_running_loop()

    def read() -> dict:
        s = session_factory()
        try:
            return _job_snapshot(s, job_id)
        finally:
            s.close()

    # 404 must surface as an HTTP error, not inside the stream
    last = await loop.run_in_executor(None, read)
    yield _sse("progress", last)
    if _is_terminal(last):
        yield _sse("done", last)
        return

    deadline = time.monotonic() + max_seconds
    while time.monotonic() < deadline:
        await asyncio.sleep(poll_s)
        snap = await loop.run_in_executor(None, read)
        if snap != last:
            yield _sse("progress", snap)
            last = snap
        if _is_terminal(last):
            yield _sse("done", last)
            return

    yield _sse("done", {**last, "reason": "timeout"})


def require_job(s: Session, project_id: str, job_id: str) -> dict:
    """Validate that project + job exist; returns the first snapshot (404-safe)."""
    try:
        snapshot = _job_snapshot(s, job_id)
        if snapshot["project_id"] != project_id:
            raise HTTPException(status_code=404, detail="job not found for project")
        return snapshot
    finally:
        s.close()
