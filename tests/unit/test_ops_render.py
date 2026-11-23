"""Tracked op paths ride the render plan (roadmap step 4).

- pure merge (`merge_op_params`): op params -> Pass-2 trajectories on the
  SAME smoothing/envelope path as detector tracks; out-of-range hits dropped;
  malformed params skipped; no ops -> plan unchanged (cache stays warm)
- pipeline helper (`_render_plan_with_ops`): active tracked blur ops merge
  into the payload plan; the digest carries the op fingerprint so a new/revoked
  op invalidates exactly the baked segments; revoked ops NEVER render
"""

from __future__ import annotations

import pytest
from director_os.core.db import TimelineOpRow, init_models, make_engine, make_session_factory
from director_os.vision_fx.trajectory import (
    PerFrameRedaction,
    RedactionTrajectory,
    TrackedBox,
    merge_op_params,
)
from director_os.worker.pipeline import PipelineDeps, _render_plan_with_ops

FPS = 30


def _plan(n_frames: int = 300) -> PerFrameRedaction:
    boxes = [
        TrackedBox(frame_idx=i, t_s=round(i / FPS, 3), x=10, y=10, w=40, h=40)
        for i in range(60, 200)
    ]
    return PerFrameRedaction(
        fps=FPS,
        n_frames=n_frames,
        frame_w=320,
        frame_h=240,
        trajectories=[RedactionTrajectory(track_id="person_1", boxes=boxes)],
    )


def _op_params(path: list[tuple[int, int, int, int]], backend: str = "ncc-cpu") -> dict:
    return {
        "mode": "tracked",
        "tracked_backend": backend,
        "tracked": [
            {"t_s": i / FPS, "x": x, "y": y, "w": w, "h": h, "score": 0.9, "interpolated": False}
            for i, (x, y, w, h) in enumerate(path)
        ],
    }


def test_merge_appends_op_trajectory_on_the_pass2_path():
    """Op boxes get the SAME One-Euro smoothing + padded envelope as sweeps."""
    plan = _plan()
    path = [(100, 100, 28, 28)] * 30  # static box, frames 0..29
    merged, changed = merge_op_params(plan, [_op_params(path)])
    assert changed is True
    ids = [t.track_id for t in merged.trajectories]
    assert "sel_ncc-cpu" in ids
    assert "person_1" in ids  # the sweep trajectory survives
    op_tr = next(t for t in merged.trajectories if t.track_id == "sel_ncc-cpu")
    # the render contract: boxes_at(frame) returns a padded box at that frame
    box = op_tr.boxes[0]
    assert box.frame_idx == 0
    # padding envelope grows the 28x28 prompt (scale law) — never smaller
    assert box.w >= 28 and box.h >= 28
    # boxes stay inside the frame
    assert box.x >= 0 and box.x + box.w <= 320
    assert merged.boxes_at(10) != []


def test_merge_drops_out_of_range_and_malformed():
    plan = _plan(n_frames=100)
    path = [(50 + 2 * i, 50, 28, 28) for i in range(60)]  # frames 0..59, in range
    params = _op_params(path)
    params["tracked"].append(
        {"t_s": 90 / FPS, "x": 10, "y": 10, "w": 28, "h": 28, "score": 0.9, "interpolated": False}
    )  # frame 90 < 100 — stays
    params["tracked"].append({"t_s": 120 / FPS, "x": 1, "y": 1, "w": 4, "h": 4})  # >= n_frames
    params["tracked"].append({"garbage": True})  # malformed — skipped
    merged, changed = merge_op_params(plan, [params])
    assert changed is True
    op_tr = next(t for t in merged.trajectories if t.track_id.startswith("sel_"))
    assert op_tr.boxes[-1].frame_idx < 100


def test_merge_ignores_non_tracked_and_empty_params():
    plan = _plan()
    merged, changed = merge_op_params(
        plan,
        [
            {"mode": "static", "x": 1, "y": 1, "w": 4, "h": 4},
            {},
            None,
            {"mode": "tracked", "tracked": []},
        ],
    )
    assert changed is False
    assert merged is plan  # identical object — the caller keeps the old digest


def test_merge_single_hit_refused():
    plan = _plan()
    merged, changed = merge_op_params(plan, [_op_params([(100, 100, 28, 28)])])
    assert changed is False


# --------------------------- pipeline helper ---------------------------


@pytest.fixture()
def sessions(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/ops_render.db")
    init_models(engine)
    return make_session_factory(engine)


@pytest.fixture()
def deps(sessions):
    from types import SimpleNamespace

    logs: list[str] = []
    d = PipelineDeps(sessions=sessions, settings=SimpleNamespace(), log=logs.append)
    d._logs = logs  # type: ignore[attr-defined]  # test handle
    return d


def _payload() -> dict:
    return {
        "per_frame": _plan().model_dump(mode="json"),
        "plan_digest": "abc12345",
    }


def test_render_plan_merges_active_tracked_ops(deps, sessions):
    with sessions() as s:
        s.add(
            TimelineOpRow(
                project_id="p1",
                kind="blur",
                params=_op_params([(100 + 2 * i, 60, 28, 28) for i in range(20)]),
                origin="agent",
            )
        )
        s.commit()
    out = _render_plan_with_ops(deps, "p1", _payload())
    assert out is not None
    plan, digest = out
    assert "abc12345+ops" in digest
    assert any(t.track_id.startswith("sel_") for t in plan.trajectories)


def test_render_plan_ignores_revoked_ops(deps, sessions):
    from datetime import UTC, datetime

    with sessions() as s:
        s.add(
            TimelineOpRow(
                project_id="p1",
                kind="blur",
                params=_op_params([(100, 60, 28, 28)] * 5),
                origin="human",
                revoked_at=datetime.now(UTC),
            )
        )
        s.commit()
    out = _render_plan_with_ops(deps, "p1", _payload())
    assert out is not None
    plan, digest = out
    assert digest == "abc12345"  # unchanged — cache stays warm
    assert not any(t.track_id.startswith("sel_") for t in plan.trajectories)


def test_render_plan_without_per_frame_returns_none(deps):
    assert _render_plan_with_ops(deps, "p1", {"regions": []}) is None


def test_render_plan_with_no_ops_keeps_digest(deps):
    out = _render_plan_with_ops(deps, "p1", _payload())
    assert out is not None
    plan, digest = out
    assert digest == "abc12345"
    assert plan.n_frames == 300


def test_render_plan_op_edit_changes_digest(deps, sessions):
    """The export-gate cache law: an op edit invalidates the baked segments."""
    with sessions() as s:
        s.add(
            TimelineOpRow(
                project_id="p1",
                kind="blur",
                params=_op_params([(100, 60, 28, 28)] * 10),
                origin="agent",
            )
        )
        s.commit()
    d1 = _render_plan_with_ops(deps, "p1", _payload())[1]
    with sessions() as s:
        rows = s.query(TimelineOpRow).all()
        rows[0].params = _op_params([(200, 80, 30, 30)] * 10)  # the agent moved the box
        s.commit()
    d2 = _render_plan_with_ops(deps, "p1", _payload())[1]
    assert d1 != d2


def test_render_plan_stamps_baked_ops_rev(deps, sessions, tmp_path):
    """The bake stamps the export-gate cursor INTO the payload — never client-trusted."""
    from director_os.core.db import PerceptionIndexRow

    with sessions() as s:
        s.add(
            TimelineOpRow(
                project_id="p1",
                kind="blur",
                params=_op_params([(100, 60, 28, 28)] * 10),
                origin="agent",
            )
        )
        s.add(PerceptionIndexRow(asset_id="a1", redaction_tracks_json={"regions": []}))
        s.commit()
    out = _render_plan_with_ops(deps, "p1", _payload(), asset_id="a1")
    assert out is not None
    with sessions() as s:
        prow = s.get(PerceptionIndexRow, "a1")
        assert prow.redaction_tracks_json["baked_ops_rev"] == 1
