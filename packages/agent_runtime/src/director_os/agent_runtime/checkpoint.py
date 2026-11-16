"""Checkpoint / rollback — durable timeline state (PLAN §3.3 checkpoint-before-execute).

Every agent step that mutates the timeline first writes the FULL serialized OTIO
state to `timeline_versions` (PLAN §5.1). A failed or regretted step rolls back
to the prior checkpoint; a human can too (`rollback(project_id, version)`).

Tool invocations land in `agent_tool_logs` with wall-time and args, giving the
Critic agent (and the cost dashboard in M5) a complete audit trail.
"""

from __future__ import annotations

import json
from typing import Any

import opentimelineio.adapters as ot_adapters
from director_os.core.db import AgentToolLogRow, TimelineVersionRow
from director_os.otio_engine.timeline import to_json_dict
from sqlalchemy import desc, select


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
        """Persist the timeline state; returns the new version_num."""
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
                    otio_json=to_json_dict(timeline),
                    created_by=created_by,
                    checkpoint_note=note[:500],
                )
            )
            s.commit()
            return version

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
                }
                for r in rows
            ]
