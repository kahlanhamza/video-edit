"""selection.track agent tool — evidence law, op creation, undoability."""

from __future__ import annotations

import shutil

import pytest
from director_os.agent_runtime.editing_tools import ToolOperationError
from director_os.agent_runtime.registry import ToolRegistry
from director_os.agent_runtime.selection_tools import register_selection_tools
from director_os.core.db import init_models, make_engine, make_session_factory
from director_os.core.ops import OpsStore, fold_ops


@pytest.fixture()
def work_dir(tmp_path, moving_box_video):
    """A work dir holding the synthetic clip as the working master."""
    wd = tmp_path / "work"
    wd.mkdir()
    shutil.copy(moving_box_video, wd / "master_postfx_v4.mp4")
    return wd


@pytest.fixture()
def env(tmp_path, work_dir):
    engine = make_engine(f"sqlite:///{tmp_path}/sel_tools.db")
    init_models(engine)
    store = OpsStore(make_session_factory(engine), "p1")
    reg = ToolRegistry()
    register_selection_tools(reg, store, work_dir, agent_name="test_agent")
    return reg, store


EVIDENCE = {"frame_path": "/tmp/grabbed.jpg", "frame_no": 31, "tool": "frame.grab"}


def test_one_selection_tool_registered(env):
    reg, _ = env
    assert "selection.track" in reg.names()


def test_selection_track_requires_evidence(env):
    reg, _ = env
    with pytest.raises(ToolOperationError, match="evidence"):
        reg.dispatch(
            "selection.track",
            {"anchor_t_s": 1.0, "x": 100, "y": 90, "w": 28, "h": 28},
        )


def test_selection_track_creates_a_tracked_blur_op(env):
    reg, store = env
    out = reg.dispatch(
        "selection.track",
        {
            "anchor_t_s": 1.0,  # frame 30
            "x": 100,
            "y": 90,
            "w": 28,
            "h": 28,
            "span_s": 4.0,
            "evidence": EVIDENCE,
        },
    )
    assert out["ok"] is True
    assert out["backend"] == "ncc-cpu"
    assert out["n_hits"] >= 60
    assert out["op_id"]
    rec = store.get(out["op_id"])
    assert rec is not None and rec.kind == "blur"
    assert rec.origin == "agent"
    assert rec.params["mode"] == "tracked"
    assert len(rec.params["tracked"]) == out["n_hits"]
    # evidence carried INTO the op (audit trail)
    assert rec.evidence["tool"] == "frame.grab"
    assert rec.evidence["anchor_frame"] == 30
    # the fold renders it as an effect
    fold = fold_ops(store.list_ops(include_revoked=True))
    assert fold.active_ops == 1
    assert fold.effects[0]["op_id"] == rec.id


def test_selection_track_is_undoable(env):
    reg, store = env
    out = reg.dispatch(
        "selection.track",
        {"anchor_t_s": 1.0, "x": 100, "y": 90, "w": 28, "h": 28, "evidence": EVIDENCE},
    )
    revoked = store.revoke(out["op_id"], note="agent changed its mind")
    assert revoked.revoked is True
    fold = fold_ops(store.list_ops(include_revoked=True))
    assert fold.active_ops == 0
    assert fold.revoked_ops == 1


def test_selection_track_refuses_lost_objects(env, flat_video, tmp_path):
    """The tool surfaces the honest refusal as a ToolOperationError."""
    wd = tmp_path / "work_flat"
    wd.mkdir()
    shutil.copy(flat_video, wd / "master_postfx_v4.mp4")
    engine = make_engine(f"sqlite:///{tmp_path}/sel_flat.db")
    init_models(engine)
    reg = ToolRegistry()
    register_selection_tools(reg, OpsStore(make_session_factory(engine), "p1"), wd)
    with pytest.raises(ToolOperationError, match="texture-free"):
        reg.dispatch(
            "selection.track",
            {"anchor_t_s": 0.2, "x": 150, "y": 100, "w": 24, "h": 24, "evidence": EVIDENCE},
        )


def test_selection_track_clamps_span_to_tape(env):
    reg, _ = env
    out = reg.dispatch(
        "selection.track",
        {
            "anchor_t_s": 3.5,  # near the 4 s end
            "x": 100,
            "y": 90,
            "w": 28,
            "h": 28,
            "span_s": 10.0,  # would run past the tape
            "evidence": EVIDENCE,
        },
    )
    assert out["ok"] is True  # clamped, not failed


def test_selection_track_explicit_detector_assist_without_tracks_falls_back(env):
    """detector-assist with no sweep data: the chooser falls back to NCC."""
    reg, _ = env
    out = reg.dispatch(
        "selection.track",
        {
            "anchor_t_s": 1.0,
            "x": 100,
            "y": 90,
            "w": 28,
            "h": 28,
            "backend": "detector-assist",
            "evidence": EVIDENCE,
        },
    )
    assert out["backend"] == "ncc-cpu"  # fallback, not a 4xx


def test_selection_track_sam2_request_is_loud(env):
    reg, _ = env
    with pytest.raises(ToolOperationError, match="sam2"):
        reg.dispatch(
            "selection.track",
            {
                "anchor_t_s": 1.0,
                "x": 100,
                "y": 90,
                "w": 28,
                "h": 28,
                "backend": "sam2",
                "evidence": EVIDENCE,
            },
        )
