"""Session memory — progress files as the durable brain (PLAN §3.3D).

Research-backed pattern for long-running agents: each new session starts
with NO memory, so progress files, task lists and notes ARE the memory.
A sandbox window can die mid-session (owner's connection drops, 10-minute
tool ceiling); the next session must resume exactly where this one left off
instead of re-deriving state from scratch.

Three workspace-local tools, all atomic-writes, no media access:

  session.plan     the task ledger (what remains) — set / mark done / read
  session.notes    timestamped notes: facts learned, decisions, blockers
  session.status   one compact digest the NEXT session reads FIRST

The state lives in `session_state.json` inside the session work dir next to
the checkpoints (agent.db) and the caption workbench — everything a resume
needs in one folder. `session.status` is deliberately the cheapest possible
read: plan summary + last notes + checkpoint version count + the artifact
inventory, so a resumed session spends ONE tool call re-orienting.

Every input is a Pydantic model (the registry validates BEFORE the tool runs).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal

from director_os.agent_runtime.editing_tools import ToolOperationError
from director_os.agent_runtime.registry import ToolRegistry
from pydantic import BaseModel, Field

MAX_TASKS = 24
MAX_TASK_LEN = 400
MAX_NOTES = 40
MAX_NOTE_LEN = 800
NOTE_PREVIEW = 220


def _atomic_write(p: Path, content: str) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(p)


def _load_state(work_dir: Path) -> dict[str, Any]:
    p = work_dir / "session_state.json"
    if not p.exists():
        return {"plan": [], "notes": [], "sessions": []}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # a torn write (sandbox killed mid-write) must not poison the resume:
        # fall back to the .tmp sibling if it parses, else to empty
        tmp = p.with_suffix(".json.tmp")
        if tmp.exists():
            try:
                return json.loads(tmp.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        return {"plan": [], "notes": [], "sessions": []}
    d.setdefault("plan", [])
    d.setdefault("notes", [])
    d.setdefault("sessions", [])
    return d


# ---------------------------------------------------------------- inputs
class _PlanSet(BaseModel):
    action: Literal["set", "mark_done", "get"]
    tasks: list[str] | None = Field(default=None, max_length=MAX_TASKS, min_length=1)
    task_index: int | None = Field(default=None, ge=0)


class _Note(BaseModel):
    action: Literal["append", "read"] = "append"
    text: str = Field(default="", max_length=MAX_NOTE_LEN)


# ---------------------------------------------------------------- tools
def register_session_memory_tools(registry: ToolRegistry, work_dir: Path) -> None:
    """Register the durable session-memory surface for one work dir."""
    work_dir.mkdir(parents=True, exist_ok=True)

    def session_plan(
        action: str, tasks: list[str] | None, task_index: int | None
    ) -> dict[str, Any]:
        state = _load_state(work_dir)
        if action == "get":
            return {"plan": state["plan"]}
        if action == "set":
            if not tasks:
                raise ToolOperationError("set requires a non-empty tasks list")
            if len(tasks) > MAX_TASKS:
                raise ToolOperationError(f"plan capped at {MAX_TASKS} tasks (got {len(tasks)})")
            for t in tasks:
                if not t.strip():
                    raise ToolOperationError("empty task string in plan")
            state["plan"] = [
                {"task": t.strip()[:MAX_TASK_LEN], "done": False, "updated_at": time.time()}
                for t in tasks
            ]
        elif action == "mark_done":
            if task_index is None:
                raise ToolOperationError("mark_done requires task_index")
            if not 0 <= task_index < len(state["plan"]):
                raise ToolOperationError(
                    f"task_index {task_index} out of range (plan has {len(state['plan'])} tasks)"
                )
            state["plan"][task_index]["done"] = True
            state["plan"][task_index]["updated_at"] = time.time()
        _atomic_write(work_dir / "session_state.json", json.dumps(state, indent=1))
        done = sum(1 for t in state["plan"] if t["done"])
        return {
            "detail": f"plan {action}: {done}/{len(state['plan'])} done",
            "plan": state["plan"],
        }

    def session_notes(action: str, text: str) -> dict[str, Any]:
        state = _load_state(work_dir)
        if action == "append":
            if not text.strip():
                raise ToolOperationError("append requires non-empty text")
            state["notes"].append({"at": time.time(), "text": text.strip()[:MAX_NOTE_LEN]})
            state["notes"] = state["notes"][-MAX_NOTES:]
            _atomic_write(work_dir / "session_state.json", json.dumps(state, indent=1))
            return {"detail": f"note #{len(state['notes'])} recorded"}
        return {"notes": state["notes"][-10:]}

    def session_status() -> dict[str, Any]:
        state = _load_state(work_dir)
        plan = state["plan"]
        done = sum(1 for t in plan if t["done"])
        # one-line digest per pending task — the resume orientation read
        pending = [f"{i}: {t['task'][:NOTE_PREVIEW]}" for i, t in enumerate(plan) if not t["done"]]
        # artifact inventory of the work dir (what a resume needs present)
        artifacts = sorted(
            p.name
            for p in work_dir.iterdir()
            if p.is_file() and p.suffix in (".json", ".ass", ".otio", ".db")
        )
        versions = 0
        db = work_dir / "agent.db"
        if db.exists():
            try:
                import sqlite3

                with sqlite3.connect(db) as con:
                    versions = int(
                        con.execute("SELECT count(*) FROM timeline_versions").fetchone()[0]
                    )
            except Exception:  # noqa: BLE001 — db schema drift: report 0
                versions = 0
        return {
            "n_tasks_done": done,
            "n_tasks_total": len(plan),
            "pending_tasks": pending,
            "last_notes": state["notes"][-5:],
            "durable_versions": versions,
            "artifacts": artifacts,
        }

    registry.register(
        "session.plan",
        "The durable task ledger. action=set replaces the plan (start of a "
        "session); mark_done ticks a task off by index as you complete it; "
        "get reads it back. Survives process death — the next session "
        "resumes from exactly this list.",
        _PlanSet,
        lambda **kw: session_plan(**kw),
    )
    registry.register(
        "session.notes",
        "Append a timestamped note (facts learned from evidence, decisions "
        "made, blockers hit) or read the last notes. Notes ARE the memory "
        "between sessions — record anything the next session must not "
        "re-derive.",
        _Note,
        lambda **kw: session_notes(**kw),
    )
    registry.register(
        "session.status",
        "Resume digest: pending tasks, last notes, durable checkpoint "
        "version count, work-dir artifact inventory. Call this FIRST on a "
        "resumed session — one call replaces re-deriving the whole state.",
        type("_EmptyIn", (BaseModel,), {}),
        lambda: session_status(),
    )
