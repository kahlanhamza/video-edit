"""RunTracker — the live observability spine for one pipeline job (Studio mandate).

Every stage becomes a *step* with its own progress bar; every `deps.log` line
becomes a streamed console line; every timeline save pushes a revision event.
All of it lands in durable rows (`job_steps` / `job_logs` / `job_events`) so
the API can tail the stream over SSE with a monotonic cursor — crash-safe,
multi-worker-safe, and replayable from `snapshot()` after a page reload.

Event kinds: `job` (stage/overall pct), `step` (per-step lifecycle+pct),
`log` (console line), `timeline` (NLE revision bump), `error` (terminal).

DESIGN LAW: observability must never break the pipeline. Every write path is
(a) a pure-write statement — no read-then-write transaction upgrades, which
deadlock SQLite under concurrent readers — and (b) guarded: a tracker failure
degrades to a stdout line, never a pipeline exception.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import wraps
from typing import Any, TypeVar

import sqlalchemy as sa
from director_os.core.db import JobEventRow, JobLogRow, JobRow, JobStepRow

MAX_SNAPSHOT_LOGS = 400

F = TypeVar("F", bound=Callable[..., Any])


def _now() -> datetime:
    return datetime.now(UTC)


def _safe[F: Callable[..., Any]](fn: F) -> F:
    """A tracker failure degrades to a stdout line — never kills the pipeline."""

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 — observability is best-effort
            print(f"[tracker] degraded ({type(e).__name__}: {e})", flush=True)
            return None

    return wrapper  # type: ignore[return-value]


class StepHandle:
    """Handle a pipeline stage uses to report sub-progress while it runs."""

    def __init__(self, tracker: RunTracker, row_id: int, key: str):
        self._tracker = tracker
        self._row_id = row_id
        self.key = key
        self._finished = False  # skip() claims the terminal state; exit won't re-done

    def pct(self, value: float, detail: str = "") -> None:
        """Sub-progress 0-100 for THIS step (the console bar under its label)."""
        self._tracker._update_step_pct(self._row_id, value, detail)

    def skip(self, detail: str = "") -> None:
        if self._finished:
            return
        self._finished = True
        self._tracker._finish_step(self._row_id, "skipped", detail)


class RunTracker:
    """Created once per pipeline run; owns the observability writes for a job."""

    def __init__(self, sessions, job_id: str, project_id: str):
        self._sessions = sessions
        self._job_id = job_id
        self._project_id = project_id
        self._order = 0
        self._pct_cache: dict[int, tuple[float, str]] = {}
        self._started: dict[int, float] = {}  # row_id -> monotonic start
        self._step_meta: dict[int, tuple[str, str, int]] = {}  # row_id -> (key, label, order)

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _step_payload(
        row_id: int,
        key: str,
        label: str,
        status: str,
        pct: float,
        detail: str,
        order: int,
        duration_ms: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "seq_step": row_id,
            "key": key,
            "label": label,
            "status": status,
            "pct": pct,
            "detail": detail,
            "order": order,
        }
        if duration_ms is not None:
            payload["duration_ms"] = duration_ms
        return payload

    @_safe
    def _emit(self, kind: str, payload: dict[str, Any]) -> None:
        with self._sessions() as s:
            s.add(
                JobEventRow(
                    job_id=self._job_id, project_id=self._project_id, kind=kind, payload=payload
                )
            )
            s.commit()

    @_safe
    def _finish_step(self, row_id: int, status: str, detail: str) -> None:
        started = self._started.pop(row_id, None)
        duration_ms = int((time.monotonic() - started) * 1000) if started else 0
        with self._sessions() as s:
            # pure UPDATE (no read-then-write upgrade — SQLite BUSY_SNAPSHOT law)
            s.execute(
                sa.update(JobStepRow)
                .where(JobStepRow.id == row_id)
                .values(
                    status=status,
                    pct=100.0 if status == "done" else JobStepRow.pct,
                    detail=detail[:1000],
                    ended_at=_now(),
                    duration_ms=duration_ms,
                )
            )
            s.commit()
        key, label, order = self._step_meta.get(row_id, ("?", "?", 0))
        self._emit(
            "step",
            self._step_payload(
                row_id,
                key,
                label,
                status,
                100.0 if status == "done" else -1.0,
                detail,
                order,
                duration_ms,
            ),
        )

    @_safe
    def _update_step_pct(self, row_id: int, value: float, detail: str) -> None:
        value = max(0.0, min(100.0, float(value)))
        cached = self._pct_cache.get(row_id)
        if cached is not None and cached[0] == value and cached[1] == detail:
            return  # throttle identical broadcasts
        self._pct_cache[row_id] = (value, detail)
        values: dict[str, Any] = {"pct": value}
        if detail:
            values["detail"] = detail[:1000]
        with self._sessions() as s:
            s.execute(sa.update(JobStepRow).where(JobStepRow.id == row_id).values(**values))
            s.commit()
        key, label, order = self._step_meta.get(row_id, ("?", "?", 0))
        self._emit("step", self._step_payload(row_id, key, label, "running", value, detail, order))

    # ------------------------------------------------------------------ public API
    @_safe
    def stage(self, stage: str, pct: float, error: str | None = None) -> None:
        """Overall job stage + percentage (mirrors the jobs row; emits `job`)."""
        with self._sessions() as s:
            s.execute(
                sa.update(JobRow)
                .where(JobRow.id == self._job_id)
                .values(current_stage=stage, progress_pct=float(pct), error_message=error)
            )
            s.add(
                JobEventRow(
                    job_id=self._job_id,
                    project_id=self._project_id,
                    kind="job",
                    payload={"stage": stage, "progress_pct": round(float(pct), 1), "error": error},
                )
            )
            s.commit()

    @_safe
    def log(self, message: str, level: str = "info") -> None:
        """One console line. Auto-escalates failure-ish lines to warn."""
        if level == "info" and ("FAILED" in message or "failed (" in message):
            level = "warn"
        with self._sessions() as s:
            s.add(
                JobLogRow(
                    job_id=self._job_id,
                    project_id=self._project_id,
                    level=level,
                    message=message[:4000],
                )
            )
            s.commit()
        self._emit("log", {"level": level, "message": message[:4000]})

    @contextmanager
    def step(self, key: str, label: str, detail: str = "") -> Iterator[StepHandle]:
        """Span one pipeline stage: running -> done | failed | skipped.

        The span itself is failure-transparent: it records the step failure and
        re-raises, so pipeline behavior is byte-identical with/without tracking.
        """
        row_id = -1
        handle = StepHandle(self, row_id, key)
        try:
            self._order += 1
            with self._sessions() as s:
                row = JobStepRow(
                    job_id=self._job_id,
                    project_id=self._project_id,
                    key=key[:80],
                    label=label[:200],
                    status="running",
                    order=self._order,
                    detail=detail[:1000],
                )
                s.add(row)
                s.commit()
                row_id = row.id
            handle = StepHandle(self, row_id, key)
            self._started[row_id] = time.monotonic()
            self._step_meta[row_id] = (key[:80], label[:200], self._order)
        except Exception as e:  # noqa: BLE001 — degraded tracking must not stop the run
            print(f"[tracker] step '{label}' not tracked ({e})", flush=True)
        self._emit(
            "step",
            self._step_payload(
                row_id, key[:80], label[:200], "running", 0.0, detail[:1000], self._order
            ),
        )
        try:
            yield handle
        except Exception as e:
            handle._finished = True
            self._finish_step(row_id, "failed", f"{type(e).__name__}: {e}"[:1000])
            self.error(f"step '{label}' failed: {e}")
            raise
        else:
            if not handle._finished:
                self._finish_step(row_id, "done", "")

    @_safe
    def timeline(self, version: int, note: str = "") -> None:
        """A new NLE timeline revision exists — the Studio refetches and redraws."""
        self._emit("timeline", {"version": version, "note": note[:500]})

    @_safe
    def error(self, message: str) -> None:
        self.log(message, level="error")

    # ------------------------------------------------------------------ reads
    def snapshot(self) -> dict[str, Any]:
        """Initial render payload for the console (steps + logs + job state)."""
        with self._sessions() as s:
            job = s.get(JobRow, self._job_id)
            steps = s.execute(
                sa.select(JobStepRow)
                .where(JobStepRow.job_id == self._job_id)
                .order_by(JobStepRow.id)
            ).scalars()
            step_rows = [
                {
                    "id": r.id,
                    "key": r.key,
                    "label": r.label,
                    "status": r.status,
                    "pct": r.pct,
                    "detail": r.detail,
                    "order": r.order,
                    "duration_ms": r.duration_ms,
                }
                for r in steps
            ]
            logs = s.execute(
                sa.select(JobLogRow)
                .where(JobLogRow.job_id == self._job_id)
                .order_by(JobLogRow.id.desc())
                .limit(MAX_SNAPSHOT_LOGS)
            ).scalars()
            log_rows = [
                {"id": r.id, "level": r.level, "message": r.message} for r in reversed(list(logs))
            ]
            job_state = (
                {
                    "job_id": job.id,
                    "stage": job.current_stage,
                    "progress_pct": round(float(job.progress_pct or 0.0), 1),
                    "error": job.error_message,
                }
                if job
                else None
            )
        return {"job": job_state, "steps": step_rows, "logs": log_rows}

    def events_after(self, after_seq: int) -> list[dict[str, Any]]:
        """Ordered SSE deltas with monotonic cursor `after_seq` (exclusive)."""
        with self._sessions() as s:
            rows = s.execute(
                sa.select(JobEventRow)
                .where(JobEventRow.job_id == self._job_id, JobEventRow.id > after_seq)
                .order_by(JobEventRow.id)
            ).scalars()
            return [{"seq": r.id, "kind": r.kind, "payload": dict(r.payload or {})} for r in rows]
