"""Per-track redaction opt-out — keep the subject visible, plates always blurred."""

from __future__ import annotations

from pathlib import Path

import pytest
from director_os.core.db import (
    PerceptionIndexRow,
    ProjectRow,
    init_models,
    make_engine,
    make_session_factory,
)
from director_os.vision_fx.compute.base import FaceTrack, TrackHit
from director_os.vision_fx.compute.tracker import EmptySweepError, merge_redaction_tracks
from director_os.vision_fx.trajectory import repair_trajectories
from director_os.worker.pipeline import _resolve_keep_track_ids


class Deps:
    def __init__(self, db_url: str):
        engine = make_engine(db_url)
        init_models(engine)
        self.sessions = make_session_factory(engine)


@pytest.fixture()
def deps(tmp_path: Path):
    d = Deps(f"sqlite:///{tmp_path}/r.db")
    with d.sessions() as s:
        s.add(ProjectRow(title="t"))
        s.commit()
    return d


def _track(track_id: str, hits: int = 5, start: float = 0.0) -> FaceTrack:
    return FaceTrack(
        track_id=track_id,
        hits=[
            TrackHit(t_s=start + i * 0.1, x=10 + i, y=10, w=40, h=40, interpolated=False)
            for i in range(hits)
        ],
    )


def test_env_keep_list(monkeypatch, deps):
    monkeypatch.setenv("REDACT_KEEP_TRACKS", "person_1, person_2")
    monkeypatch.delenv("REDACT_TRACK_POLICY", raising=False)
    keep, policy = _resolve_keep_track_ids(deps, "a1")
    assert keep == {"person_1", "person_2"}
    assert policy == "all"


def test_plates_never_keepable(monkeypatch, deps):
    monkeypatch.setenv("REDACT_KEEP_TRACKS", "person_1,plate_0")
    keep, _ = _resolve_keep_track_ids(deps, "a1")
    assert "plate_0" not in keep
    assert "person_1" in keep


def test_persisted_studio_keep_list(monkeypatch, deps):
    monkeypatch.delenv("REDACT_KEEP_TRACKS", raising=False)
    with deps.sessions() as s:
        s.add(
            PerceptionIndexRow(
                asset_id="aX", redaction_tracks_json={"keep_track_ids": ["person_7"]}
            )
        )
        s.commit()
    keep, _ = _resolve_keep_track_ids(deps, "aX")
    assert keep == {"person_7"}


def test_kept_tracks_leave_the_plan(deps):
    """The zero-leak contract holds for the FILTERED track set (not the raw one)."""
    face_tracks = [_track("person_1", 6), _track("person_2", 3, start=1.0)]
    plates = [_track("plate_0", 2, start=2.0)]
    keep = {"person_2"}
    visible = [t for t in face_tracks if t.track_id not in keep]
    merged = merge_redaction_tracks(visible, plates)
    plan = repair_trajectories(merged, fps=30, frame_w=640, frame_h=480)
    from director_os.vision_fx.redact_track import assert_zero_leaks

    assert_zero_leaks(plan, merged, 30.0)  # consistent: plan covers exactly `merged`
    ids = {tr.track_id for tr in plan.trajectories}
    assert "person_1" in ids and "plate_0" in ids
    assert "person_2" not in ids  # kept visible by policy


def test_keeping_everything_is_legal_not_vacuous(deps):
    face_tracks = [_track("person_1", 4)]
    keep = {"person_1"}
    visible = [t for t in face_tracks if t.track_id not in keep]
    with pytest.raises(EmptySweepError):
        merge_redaction_tracks(visible, [])  # tracker still raises for a degenerate sweep
    # the payload builder catches this when a deliberate keep-list is active
    # (see _tracked_redaction_payload) — the merged list is simply empty
