"""Hermetic tests for the session memory surface (session_memory).

The research-backed initializer/worker law: a session that dies mid-run must
leave progress files, task lists and notes as the memory — and a NEW session
must resume from them with ONE orientation read. No network, no media: a tmp
work dir is the whole world.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from director_os.agent_runtime.editing_tools import ToolOperationError
from director_os.agent_runtime.registry import ToolDispatchError, ToolRegistry
from director_os.agent_runtime.session_memory import register_session_memory_tools


@pytest.fixture()
def work(tmp_path: Path) -> Path:
    return tmp_path / "agent_run"


def _registry(work: Path) -> ToolRegistry:
    reg = ToolRegistry()
    register_session_memory_tools(reg, work)
    return reg


def test_plan_set_get_roundtrip(work: Path) -> None:
    reg = _registry(work)
    out = reg.dispatch(
        "session.plan", {"action": "set", "tasks": ["read evidence", "tag 609s window"]}
    )
    assert out["detail"].startswith("plan set")
    assert out["plan"][0]["done"] is False
    got = reg.dispatch("session.plan", {"action": "get"})
    assert [t["task"] for t in got["plan"]] == ["read evidence", "tag 609s window"]
    # the state file is real and on disk
    st = json.loads((work / "session_state.json").read_text())
    assert len(st["plan"]) == 2


def test_plan_mark_done_ticks_only_that_task(work: Path) -> None:
    reg = _registry(work)
    reg.dispatch("session.plan", {"action": "set", "tasks": ["a", "b", "c"]})
    out = reg.dispatch("session.plan", {"action": "mark_done", "task_index": 1})
    assert out["plan"][1]["done"] is True
    assert out["plan"][0]["done"] is False
    assert out["detail"] == "plan mark_done: 1/3 done"


def test_plan_validation_failures(work: Path) -> None:
    reg = _registry(work)
    with pytest.raises(ToolDispatchError):
        reg.dispatch("session.plan", {"action": "set", "tasks": []})
    with pytest.raises(ToolOperationError):
        reg.dispatch("session.plan", {"action": "set", "tasks": ["ok", "   "]})
    reg.dispatch("session.plan", {"action": "set", "tasks": ["only"]})
    with pytest.raises(ToolOperationError):
        reg.dispatch("session.plan", {"action": "mark_done", "task_index": 5})


def test_notes_append_read_and_cap(work: Path) -> None:
    reg = _registry(work)
    for i in range(45):
        reg.dispatch("session.notes", {"action": "append", "text": f"note {i}"})
    read = reg.dispatch("session.notes", {"action": "read", "text": ""})
    assert len(read["notes"]) == 10  # read returns the last 10
    st = json.loads((work / "session_state.json").read_text())
    assert len(st["notes"]) == 40  # cap keeps the most recent 40
    assert st["notes"][-1]["text"] == "note 44"
    with pytest.raises(ToolOperationError):
        reg.dispatch("session.notes", {"action": "append", "text": "   "})


def test_status_digest_orienting_a_resume(work: Path) -> None:
    reg = _registry(work)
    reg.dispatch(
        "session.plan",
        {"action": "set", "tasks": ["inspect 600-635s", "apply tags", "re-run QA"]},
    )
    reg.dispatch("session.plan", {"action": "mark_done", "task_index": 0})
    reg.dispatch(
        "session.notes",
        {"action": "append", "text": "VAD confirms speech 609-628; ASR loop confirmed"},
    )
    st = reg.dispatch("session.status", {})
    assert st["n_tasks_done"] == 1
    assert st["n_tasks_total"] == 3
    assert len(st["pending_tasks"]) == 2
    assert any("609-628" in n["text"] for n in st["last_notes"])
    assert "session_state.json" in st["artifacts"]


def test_memory_survives_process_death(work: Path) -> None:
    """The resume law: a FRESH registry over the same work dir sees everything."""
    reg1 = _registry(work)
    reg1.dispatch("session.plan", {"action": "set", "tasks": ["tag the loop window"]})
    reg1.dispatch("session.plan", {"action": "mark_done", "task_index": 0})
    reg1.dispatch("session.notes", {"action": "append", "text": "checkpointed v3 before re-burn"})
    # simulate a crash: no teardown, a brand-new registry/process over the dir
    reg2 = _registry(work)
    st = reg2.dispatch("session.status", {})
    assert st["n_tasks_done"] == 1 and st["n_tasks_total"] == 1
    assert st["last_notes"][0]["text"].startswith("checkpointed")


def test_torn_state_file_falls_back_to_tmp(work: Path) -> None:
    """A killed process can leave a torn write; resume must not poison."""
    reg1 = _registry(work)
    reg1.dispatch("session.plan", {"action": "set", "tasks": ["survivor task"]})
    good = (work / "session_state.json").read_text()
    (work / "session_state.json.tmp").write_text(good)
    (work / "session_state.json").write_text("{torn json")
    reg2 = _registry(work)
    got = reg2.dispatch("session.plan", {"action": "get"})
    assert [t["task"] for t in got["plan"]] == ["survivor task"]
