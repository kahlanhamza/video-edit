"""Checkpoint / rollback — durable timeline state (PLAN §3.3 checkpoint-before-execute).

Every agent step that mutates the timeline first writes the FULL serialized OTIO
state to `timeline_versions` (PLAN §5.1). A failed or regretted step rolls back
to the prior checkpoint; a human can too (`rollback(project_id, version)`).

Tool invocations land in `agent_tool_logs` with wall-time and args, giving the
Critic agent (and the cost dashboard in M5) a complete audit trail.
"""

from __future__ import annotations

import json
import time
from typing import Any

import opentimelineio.adapters as ot_adapters
from director_os.core.db import AgentToolLogRow, TimelineVersionRow
from director_os.otio_engine.timeline import to_json_dict
from sqlalchemy import desc, select
from sqlalchemy.exc import IntegrityError

CHECKPOINT_WRITE_ATTEMPTS = 5


class CheckpointError(RuntimeError):
    pass


class CheckpointStore:
    """DB-backed timeline snapshots + agent tool audit log."""

    def __init__(self, session_factory):
        self._sessions = session_factory

    # ------------------------------------------------------------- snapshots
    def checkpoint(
        self,
        timeline,
        project_id: str,
        note: str,
        created_by: str = "director_agent",
    ) -> int:
        """Persist the timeline state; returns the new version_num.

        The version number is assigned by SELECT-max then INSERT — two
        concurrent dispatches (double-click, inline fallback racing a
        recovered broker) could both read the same max and race the write
        (CR-007). The (project_id, version_num) unique constraint is the
        backstop; on conflict we re-read the max and retry — a checkpoint is
        content-addressed by the timeline it stores, so the retry writes the
        SAME state under the winning number.
        """
        payload = to_json_dict(timeline)
        for attempt in range(CHECKPOINT_WRITE_ATTEMPTS):
            with self._sessions() as s:
                last = s.execute(
                    select(TimelineVersionRow.version_num)
                    .where(TimelineVersionRow.project_id == project_id)
                    .order_by(desc(TimelineVersionRow.version_num))
                    .limit(1)
                ).scalar_one_or_none()
                version = (last or 0) + 1
                s.add(
                    TimelineVersionRow(
                        project_id=project_id,
                        version_num=version,
                        otio_json=payload,
                        created_by=created_by,
                        checkpoint_note=note[:500],
                    )
                )
                try:
                    s.commit()
                    return version
                except IntegrityError:
                    # a concurrent writer took this version number first —
                    # roll this session back and contend again
                    s.rollback()
                    if attempt == CHECKPOINT_WRITE_ATTEMPTS - 1:
                        raise
                    time.sleep(0.02 * (attempt + 1))
        raise CheckpointError(
            f"checkpoint did not converge after {CHECKPOINT_WRITE_ATTEMPTS} attempts"
        )

    def list_versions(self, project_id: str) -> list[dict[str, Any]]:
        with self._sessions() as s:
            rows = s.execute(
                select(TimelineVersionRow)
                .where(TimelineVersionRow.project_id == project_id)
                .order_by(TimelineVersionRow.version_num)
            ).scalars()
            return [
                {
                    "version_num": r.version_num,
                    "created_by": r.created_by,
                    "checkpoint_note": r.checkpoint_note,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]

    def load_version(self, project_id: str, version_num: int):
        """Rebuild the OTIO timeline object for a stored version."""
        with self._sessions() as s:
            row = s.execute(
                select(TimelineVersionRow).where(
                    TimelineVersionRow.project_id == project_id,
                    TimelineVersionRow.version_num == version_num,
                )
            ).scalar_one_or_none()
            if row is None:
                raise CheckpointError(f"no version {version_num} for project {project_id}")
            return ot_adapters.read_from_string(json.dumps(row.otio_json), "otio_json")

    def rollback(self, project_id: str, to_version_num: int) -> tuple[Any, int]:
        """Restore a prior version as the NEWEST checkpoint (history is never erased)."""
        timeline = self.load_version(project_id, to_version_num)
        new_version = self.checkpoint(
            timeline,
            project_id,
            note=f"rollback to v{to_version_num}",
            created_by="director_agent",
        )
        return timeline, new_version

    # ------------------------------------------------------------- audit log
    def log_tool(
        self,
        agent_name: str,
        tool_name: str,
        args: dict[str, Any],
        result_summary: str = "",
        duration_ms: int = 0,
        project_id: str | None = None,
        ok: bool = True,
        error: str | None = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        token_cost_usd: float | None = None,
    ) -> None:
        summary = result_summary if ok else f"ERROR: {error}"
        with self._sessions() as s:
            s.add(
                AgentToolLogRow(
                    project_id=project_id,
                    agent_name=agent_name,
                    tool_name=tool_name,
                    args_json=args,
                    result_summary=summary[:1000],
                    duration_ms=duration_ms,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    token_cost_usd=token_cost_usd,
                )
            )
            s.commit()

    def recent_logs(self, project_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._sessions() as s:
            rows = s.execute(
                select(AgentToolLogRow)
                .where(AgentToolLogRow.project_id == project_id)
                .order_by(desc(AgentToolLogRow.id))
                .limit(limit)
            ).scalars()
            return [
                {
                    "agent": r.agent_name,
                    "tool": r.tool_name,
                    "args": r.args_json,
                    "summary": r.result_summary,
                    "duration_ms": r.duration_ms,
                    "tokens_in": r.tokens_in,
                    "tokens_out": r.tokens_out,
                    "token_cost_usd": r.token_cost_usd,
                }
                for r in rows
            ]
