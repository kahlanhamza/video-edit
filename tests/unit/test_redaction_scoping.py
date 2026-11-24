"""Post-plan redaction scoping — the operator mandate made testable.

Law: detection compute pays ONLY for footage the edit kept. A 43-min source
that keeps 8 min sweeps 8 min, and two queued jobs on the same asset share
one sweep (cache + lock) instead of decoding 77k raw frames twice.
"""

from __future__ import annotations

from typing import Any

import pytest
from director_os.otio_engine.contract import RawClipMeta
from director_os.otio_engine.timeline import (
    ClipSpec,
    GapSpec,
    TimelineSpec,
    TrackSpec,
    build_timeline,
)
from director_os.worker.pipeline import (
    _kept_source_ranges,
    _ranges_digest,
    stage_redaction,
)


def _master_spec(asset_id: str, spans: list[tuple[float, float]]) -> TimelineSpec:
    """Master track from (source_start_s, dur_s) raw clips of ONE asset."""
    return TimelineSpec(
        name="t",
        fps=30,
        width=640,
        height=360,
        tracks=[
            TrackSpec(
                name="V1_master",
                items=[
                    ClipSpec(
                        name=f"raw_{i}",
                        source_start_s=s,
                        dur_s=d,
                        meta=RawClipMeta(asset_id=asset_id),
                    )
                    for i, (s, d) in enumerate(spans)
                ],
            )
        ],
    )


def test_kept_ranges_exclude_gaps_and_foreign_assets() -> None:
    tl = build_timeline(
        TimelineSpec(
            name="t",
            fps=30,
            width=640,
            height=360,
            tracks=[
                TrackSpec(
                    name="V1_master",
                    items=[
                        ClipSpec(
                            name="r1",
                            source_start_s=0.0,
                            dur_s=6.0,
                            meta=RawClipMeta(asset_id="a1"),
                        ),
                        GapSpec(dur_s=7.0),
                        ClipSpec(
                            name="r2",
                            source_start_s=10.0,
                            dur_s=5.0,
                            meta=RawClipMeta(asset_id="a1"),
                        ),
                        ClipSpec(
                            name="other",
                            source_start_s=3.0,
                            dur_s=4.0,
                            meta=RawClipMeta(asset_id="a2"),
                        ),
                    ],
                )
            ],
        )
    )
    # gap excluded, foreign asset excluded, spans sorted
    assert _kept_source_ranges(tl, "a1") == [(0.0, 6.0), (10.0, 15.0)]


def test_kept_ranges_merge_hairline_cuts() -> None:
    tl = build_timeline(_master_spec("a1", [(0.0, 6.0), (6.5, 5.0)]))  # 0.5s gap -> merge
    assert _kept_source_ranges(tl, "a1") == [(0.0, 11.5)]


def test_ranges_digest_stable_and_backend_aware() -> None:
    assert _ranges_digest("yunet", [(1.0, 2.0)]) == _ranges_digest("yunet", [(1.001, 2.004)])
    assert _ranges_digest("yunet", [(1.0, 2.0)]) != _ranges_digest("tracked", [(1.0, 2.0)])
    assert _ranges_digest("yunet", None) != _ranges_digest("yunet", [(0.0, 1.0)])


def _deps(tmp_path: Any, monkeypatch: Any, asset_id: str = "a1deadbeef"):
    """Isolated deps: OUTPUT_DIR + DB both inside tmp_path (never the repo's
    real output dir — the sweep cache is content-addressed and shared)."""
    import director_os.core.credentials  # noqa: F401 — registers vault tables on Base
    from director_os.core.config import get_settings
    from director_os.core.db import MediaAssetRow, ProjectRow, init_models
    from director_os.worker.pipeline import make_deps
    from sqlalchemy import create_engine

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    get_settings.cache_clear()
    db = f"sqlite:///{tmp_path}/t.db"
    init_models(create_engine(db))
    deps = make_deps(db, redact_backend="yunet")
    with deps.sessions() as s:
        s.add(ProjectRow(id="p1", title="P1"))
        s.add(
            MediaAssetRow(
                id=asset_id,
                project_id="p1",
                duration_s=600.0,
                width=640,
                height=360,
                fps=30.0,
            )
        )
        s.commit()
    return deps


def test_no_kept_footage_never_sweeps(tmp_path: Any, monkeypatch: Any) -> None:
    deps = _deps(tmp_path, monkeypatch, asset_id="a_nokeep")
    monkeypatch.setattr(
        "director_os.perception.faces.detect_faces",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("sweep ran on empty plan")),
    )
    tl = build_timeline(TimelineSpec(name="t", tracks=[]))
    regions = stage_redaction("a_nokeep", tmp_path / "src.mp4", "p1", deps, timeline=tl)
    assert regions == []


def test_sweep_is_scoped_to_kept_ranges(tmp_path: Any, monkeypatch: Any) -> None:
    asset = "a_scoped"
    deps = _deps(tmp_path, monkeypatch, asset_id=asset)
    seen: dict[str, Any] = {}

    def fake_detect_faces(video, sample_every_s=1.0, ranges=None, **kw):
        seen["ranges"] = ranges
        from director_os.perception.faces import FaceDetection, FaceSweep, FaceSweepStats

        return FaceSweep(
            stats=FaceSweepStats(
                frames_sampled=2,
                frames_decoded_failed=0,
                detections=0,
                detector="fake",
                sample_every_s=1.0,
            ),
            detections=[
                FaceDetection.model_construct(t_s=12.0, bbox=None, score=0.9),
            ],
        )

    monkeypatch.setattr("director_os.perception.faces.detect_faces", fake_detect_faces)
    monkeypatch.setattr("director_os.perception.faces.cluster_to_regions", lambda *a, **k: [])
    tl = build_timeline(_master_spec(asset, [(10.0, 8.0), (40.0, 5.0)]))
    regions = stage_redaction(asset, tmp_path / "src.mp4", "p1", deps, timeline=tl)
    assert regions == []
    # THE LAW: the sweep saw only the two kept spans — never the whole 600s source
    assert seen["ranges"] == [(10.0, 18.0), (40.0, 45.0)]


def test_second_job_reuses_sweep_cache(tmp_path: Any, monkeypatch: Any) -> Any:
    asset = "a_cache"
    deps = _deps(tmp_path, monkeypatch, asset_id=asset)
    calls = {"n": 0}

    def fake_detect_faces(video, sample_every_s=1.0, ranges=None, **kw):
        calls["n"] += 1
        from director_os.perception.faces import FaceSweep, FaceSweepStats

        return FaceSweep(
            stats=FaceSweepStats(
                frames_sampled=1,
                frames_decoded_failed=0,
                detections=0,
                detector="fake",
                sample_every_s=1.0,
            ),
            detections=[],
        )

    monkeypatch.setattr("director_os.perception.faces.detect_faces", fake_detect_faces)
    monkeypatch.setattr("director_os.perception.faces.cluster_to_regions", lambda *a, **k: [])
    tl = build_timeline(_master_spec(asset, [(10.0, 8.0)]))
    first = stage_redaction(asset, tmp_path / "src.mp4", "p1", deps, timeline=tl)
    assert calls["n"] == 1

    # simulate a co-running / queued second job on the same asset+plan: the
    # persisted payload already carries this ranges digest -> no re-decode
    second = stage_redaction(asset, tmp_path / "src.mp4", "p1", deps, timeline=tl)
    assert second == first
    assert calls["n"] == 1, "second job must reuse the winner's sweep"


def test_replan_new_ranges_resweep(tmp_path: Any, monkeypatch: Any) -> Any:
    asset = "a_replan"
    deps = _deps(tmp_path, monkeypatch, asset_id=asset)
    calls = {"n": 0}

    def fake_detect_faces(video, sample_every_s=1.0, ranges=None, **kw):
        calls["n"] += 1
        from director_os.perception.faces import FaceSweep, FaceSweepStats

        return FaceSweep(
            stats=FaceSweepStats(
                frames_sampled=1,
                frames_decoded_failed=0,
                detections=0,
                detector="fake",
                sample_every_s=1.0,
            ),
            detections=[],
        )

    monkeypatch.setattr("director_os.perception.faces.detect_faces", fake_detect_faces)
    monkeypatch.setattr("director_os.perception.faces.cluster_to_regions", lambda *a, **k: [])
    tl1 = build_timeline(_master_spec(asset, [(10.0, 8.0)]))
    tl2 = build_timeline(_master_spec(asset, [(20.0, 8.0)]))  # moved cut
    stage_redaction(asset, tmp_path / "src.mp4", "p1", deps, timeline=tl1)
    stage_redaction(asset, tmp_path / "src.mp4", "p1", deps, timeline=tl2)
    assert calls["n"] == 2, "a moved cut invalidates the old ranges digest -> re-sweep"


def test_detect_faces_ranges_never_decode_cut_footage(monkeypatch: Any) -> None:
    """cv2-level proof: seeks jump straight to kept spans; reads stop at span end."""
    import pathlib

    import director_os.perception.faces as faces

    class FakeDetector:
        def __init__(self) -> None:
            self.detect_calls = 0

        @property
        def name(self) -> str:
            return "fake"

        def detect(self, frame: Any) -> list[Any]:
            self.detect_calls += 1
            return []

    class FakeCap:
        def __init__(self) -> None:
            self.seeks_ms: list[float] = []
            self.reads = 0

        def isOpened(self) -> bool:
            return True

        def get(self, prop: int) -> float:
            return 30.0  # fps

        def set(self, prop: int, value: float) -> bool:
            self.seeks_ms.append(value)
            return True

        def read(self) -> tuple[bool, Any]:
            self.reads += 1
            return True, object()  # "endless" stream — only span guards stop it

        def release(self) -> None:
            pass

    cap = FakeCap()
    det = FakeDetector()
    monkeypatch.setattr(faces.cv2, "VideoCapture", lambda *_: cap)
    monkeypatch.setattr(faces, "make_detector", lambda *a, **k: det)

    sweep = faces.detect_faces(
        pathlib.Path("fake.mp4"),
        sample_every_s=1.0,
        ranges=[(5.0, 6.0), (10.0, 11.0)],
    )
    # THE LAW: seeks jump straight to each kept span — cut footage never decoded
    assert cap.seeks_ms == [5000.0, 10000.0]
    # each 1s span at 30fps: 30 reads + 1 read past the end-guard per span
    assert cap.reads == 62
    # 1 sampled frame per span (stride = 30 at 1 fps)
    assert det.detect_calls == 2
    assert sweep.stats.frames_sampled == 2


def test_clip_plan_t_s_never_negative_for_fractional_start_s() -> None:
    """A clip starting at a fractional second (203.07s) with boxes exactly on
    its first frames: clip-local t_s is computed from the INTEGER frame offset
    — the old frame_idx/fps - start_s drifted negative on sub-frame float
    error, threw TrackedBox(ge=0) validation, and baked the clip RAW
    (full-tape run 3: two clips shipped unblurred that way)."""
    from director_os.vision_fx.trajectory import (
        PerFrameRedaction,
        RedactionTrajectory,
        TrackedBox,
    )
    from director_os.worker.pipeline import _clip_plan

    fps = 30
    start_s = 203.07
    f0 = int(round(start_s * fps))  # 6092 — 6092/30 - 203.07 = -0.0033
    boxes = []
    for i in range(3):
        fi = f0 + i * fps
        boxes.append(
            TrackedBox(frame_idx=fi - f0 + f0, t_s=round(fi / fps, 3), x=10, y=10, w=20, h=20)
        )
    plan = PerFrameRedaction(
        fps=fps,
        n_frames=f0 + 4 * fps,
        frame_w=1920,
        frame_h=1080,
        trajectories=[RedactionTrajectory(track_id="face_1", boxes=boxes)],
    )
    clip = _clip_plan(plan, start_s, 3.0)
    assert len(clip.trajectories) == 1
    got = clip.trajectories[0].boxes
    assert [b.frame_idx for b in got] == [0, 30, 60]
    assert all(b.t_s >= 0 for b in got), [b.t_s for b in got]
    assert got[0].t_s == pytest.approx(0.0, abs=1 / fps)
