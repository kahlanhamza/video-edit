"""The dailies review pass — the closed agentic loop (post-render).

The notebook's manual fix-up loop becomes code: parse a strict JSON verdict,
catch story-structure crimes deterministically (the notebook's hand-fixed
arrest-before-witness law), dispatch capped repairs through the SAME typed
DirectorToolbox registry, and never let a failing review kill a QA-passed
master.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from director_os.agent_runtime.review import (
    ReviewRepair,
    ReviewVerdict,
    apply_repairs,
    deterministic_review,
    parse_review_verdict,
    text_review_backend,
    vision_review_backend,
)
from director_os.otio_engine.contract import NarratorClipMeta, RawClipMeta
from director_os.otio_engine.timeline import (
    MASTER_TRACK,
    ClipSpec,
    GapSpec,
    TimelineSpec,
    TrackSpec,
    build_timeline,
)
from director_os.worker.review_pass import (
    build_evidence,
    build_review_backend,
    stage_review,
)

PROJECT = "proj_review_test"


def _verdict_json(**overrides: object) -> str:
    base = {
        "verdict": "needs_repairs",
        "issues": ["frame_2 is 40s of parked car dead air"],
        "repairs": [{"kind": "trim_raw", "item_index": 1, "new_start_s": 12.0, "new_dur_s": 3.0}],
    }
    base.update(overrides)
    return json.dumps(base)


def _timeline_source_windows() -> TimelineSpec:
    """Cold open (sanctioned flash-forward) + a body that walks BACKWARDS:
    the notebook's real crime — a late act placed before an earlier one."""
    return TimelineSpec(
        name="review_case",
        fps=30,
        width=1920,
        height=1080,
        tracks=[
            TrackSpec(
                name=MASTER_TRACK,
                items=[
                    ClipSpec(
                        name="raw_hook",
                        source_start_s=200.0,
                        dur_s=6.0,
                        meta=RawClipMeta(asset_id="a1"),
                    ),
                    ClipSpec(
                        name="raw_body",
                        source_start_s=10.0,
                        dur_s=8.0,
                        meta=RawClipMeta(asset_id="a1"),
                    ),
                    ClipSpec(
                        name="raw_out_of_order",
                        source_start_s=5.0,
                        dur_s=6.0,
                        meta=RawClipMeta(asset_id="a1"),
                    ),
                    ClipSpec(
                        name="narr",
                        source_start_s=0.0,
                        dur_s=4.0,
                        meta=NarratorClipMeta(script_text="An out of order body."),
                    ),
                ],
            ),
            TrackSpec(name="V2_overlay_cards", items=[GapSpec(dur_s=24.0)]),
        ],
    )


@pytest.fixture()
def timeline():
    return build_timeline(_timeline_source_windows())


# ------------------------------------------------------------------ parse


def test_parse_review_verdict_fenced_json():
    text = f"Here is my review:\n```json\n{_verdict_json()}\n```"
    verdict = parse_review_verdict(text)
    assert verdict is not None
    assert verdict.verdict == "needs_repairs"
    assert verdict.repairs[0].kind == "trim_raw"
    assert verdict.repairs[0].new_start_s == 12.0


def test_parse_review_verdict_garbage_returns_none():
    assert parse_review_verdict("the video looks fine, nothing to do") is None
    assert parse_review_verdict("") is None


def test_parse_review_verdict_unknown_kinds_dropped():
    text = _verdict_json(
        repairs=[
            {"kind": "retype_narration", "item_index": 0, "script": "lie"},
            {"kind": "add_blur", "item_index": 0, "bbox": [1, 2, 3, 4]},
            {"kind": "trim_raw", "item_index": 1, "new_start_s": 5.0, "new_dur_s": 2.0},
        ]
    )
    verdict = parse_review_verdict(text)
    assert verdict is not None
    assert [r.kind for r in verdict.repairs] == ["trim_raw"]
    assert any("dropped" in i for i in verdict.issues)


def test_parse_review_verdict_malformed_repair_fails_loud():
    assert (
        parse_review_verdict(_verdict_json(repairs=[{"kind": "trim_raw", "item_index": "one"}]))
        is None
    )


def test_parse_review_verdict_repairs_imply_needs_repairs():
    verdict = parse_review_verdict(_verdict_json(verdict="ok"))
    assert verdict is not None
    assert verdict.verdict == "needs_repairs"


# ---------- deterministic reviewer: the notebook's chronology law, as code


def test_deterministic_review_catches_backwards_body(timeline):
    issues = deterministic_review(timeline)
    chrono = [i for i in issues if i.startswith("chronology:")]
    assert chrono, "the backwards body must be caught"
    assert "5.0-" in chrono[0]  # the 5-9s source window placed after 10-18s


def test_deterministic_review_accepts_cold_open_excused_body():
    spec = _timeline_source_windows()
    spec.tracks[0].items = [
        spec.tracks[0].items[0],  # cold open at 200s — the hook exception
        ClipSpec(
            name="raw_body",
            source_start_s=10.0,
            dur_s=8.0,
            meta=RawClipMeta(asset_id="a1"),
        ),
        ClipSpec(
            name="raw_late",
            source_start_s=30.0,
            dur_s=6.0,
            meta=RawClipMeta(asset_id="a1"),
        ),
        spec.tracks[0].items[3],
    ]
    issues = deterministic_review(build_timeline(spec))
    assert not [i for i in issues if i.startswith("chronology:")]


def test_deterministic_review_flags_no_footage():
    empty = build_timeline(
        TimelineSpec(
            name="empty_case",
            fps=30,
            width=1920,
            height=1080,
            tracks=[
                TrackSpec(name=MASTER_TRACK, items=[GapSpec(dur_s=4.0)]),
            ],
        )
    )
    issues = deterministic_review(empty)
    assert any("no raw clips" in i for i in issues)


# --------------------------------------------------- repair application


def test_apply_repairs_empty_list_is_a_no_op(timeline):
    applied, refused = apply_repairs(timeline, PROJECT, [])
    assert applied == [] and refused == []


def test_apply_repairs_sequential_indices_and_mutations(timeline):
    applied, refused = apply_repairs(
        timeline,
        PROJECT,
        [
            ReviewRepair(
                kind="trim_raw", item_index=1, new_start_s=12.0, new_dur_s=3.0, reason="dead air"
            ),
            ReviewRepair(
                kind="remove_clip", item_index=3, reason="narration block contributes nothing"
            ),
        ],
    )
    assert refused == []
    assert len(applied) == 2
    from director_os.otio_engine.timeline import track_by_name

    items = list(track_by_name(timeline, MASTER_TRACK))
    assert len(items) == 3  # one removed
    trimmed = items[1]
    assert trimmed.source_range.start_time.to_seconds() == pytest.approx(12.0)
    assert trimmed.source_range.duration.to_seconds() == pytest.approx(3.0)


def test_apply_repairs_out_of_range_index_refused_loudly(timeline):
    before = [list(t) for t in timeline.tracks]
    applied, refused = apply_repairs(
        timeline,
        PROJECT,
        [_rr(kind="remove_clip", item_index=99, reason="ghost clip")],
    )
    assert applied == []
    assert refused and "99" in refused[0]
    assert [list(t) for t in timeline.tracks] == before  # never a silent partial apply


def test_apply_repairs_cap_enforced(timeline):
    repairs = [_rr(kind="remove_clip", item_index=0, reason=f"r{i}") for i in range(15)]
    applied, refused = apply_repairs(timeline, PROJECT, repairs)
    assert len(applied) == 4  # the master only ever had 4 clips
    assert sum(1 for r in refused if "cap" in r) == 3  # 15th..13th refused by the cap


def _rr(**kw) -> ReviewRepair:
    return ReviewRepair(**kw)


def test_move_clip_repair_restores_chronology(timeline):
    applied, refused = apply_repairs(
        timeline,
        PROJECT,
        [_rr(kind="move_clip", item_index=3, move_to=1, reason="footer clip belongs after body")],
    )
    assert refused == []
    assert len(applied) == 1
    from director_os.otio_engine.timeline import track_by_name

    items = list(track_by_name(timeline, MASTER_TRACK))
    assert items[1].name == "narr"


# --------------------------------------------------------------- backends


def test_vision_review_backend_refuses_the_fake():
    from director_os.perception.vision import FakeVisionClient

    assert vision_review_backend(FakeVisionClient(canned=[])) is None


def test_vision_review_backend_parses_json_insight():
    class _GeminiLike:
        provider = "gemini"

        def __init__(self) -> None:
            self.seen_frames = 0

        def analyze(self, question: str, frames):
            self.seen_frames = len(frames)
            from director_os.perception.vision import SceneInsight

            return [SceneInsight(timestamp_s=1.0, description=f"```json\n{_verdict_json()}\n```")]

    client = _GeminiLike()
    backend = vision_review_backend(client)
    assert backend is not None
    verdict = backend.review([(0.0, "f.jpg")], "evidence")
    assert verdict.verdict == "needs_repairs" and len(verdict.repairs) == 1
    assert client.seen_frames == 1


def test_text_review_backend_requires_complete_json():
    assert text_review_backend(None) is None
    assert text_review_backend(SimpleNamespace()) is None


def test_text_review_backend_routes_through_complete_json():
    class _Chain:
        provider = "failover"

        def complete_json(self, system: str, user: str, max_tokens: int = 4096) -> str:
            self.system = system
            self.user = user
            return _verdict_json()

    chain = _Chain()
    backend = text_review_backend(chain)
    assert backend is not None
    verdict = backend.review([], "EVIDENCE")
    assert "reviewer" in chain.system.lower()
    assert "EVIDENCE" in chain.user
    assert verdict.repairs[0].kind == "trim_raw"


# ------------------------------------------------------------ stage_review


def _deps(tmp_path, *, backend, review_pass: bool = True, log=None):
    return SimpleNamespace(
        settings=SimpleNamespace(output_dir=tmp_path),
        review_pass=review_pass,
        review_backend_factory=(lambda: backend) if backend is not None else None,
        director_backend_factory=None,
        log=log or (lambda _m: None),
    )


def _fake_backend(verdict: ReviewVerdict):
    class _Fake:
        def review(self, frames, evidence):  # noqa: ANN001
            self.frames = frames
            self.evidence = evidence
            return verdict

    return _Fake()


def test_stage_review_off_flag_is_honest(tmp_path):
    outcome = stage_review(
        tmp_path / "master.mp4",
        build_timeline(_timeline_source_windows()),
        [],
        PROJECT,
        _deps(tmp_path, backend=None, review_pass=False),
    )
    assert outcome.status == "off"


def test_stage_review_unavailable_no_backend(tmp_path):
    outcome = stage_review(
        tmp_path / "master.mp4",
        build_timeline(_timeline_source_windows()),
        [],
        PROJECT,
        _deps(tmp_path, backend=None),
    )
    assert outcome.status == "unavailable"
    assert outcome.summary  # honest, never a silent pass


def test_stage_review_applies_repairs(tmp_path, timeline, monkeypatch):
    import director_os.worker.review_pass as rp

    frame_png = tmp_path / "f0.jpg"
    frame_png.write_bytes(b"\xff\xd8fake")
    monkeypatch.setattr(
        rp, "sample_master_frames", lambda *a, **k: [(6.0, frame_png), (12.0, frame_png)]
    )
    backend = _fake_backend(
        ReviewVerdict.model_validate(
            {
                "verdict": "needs_repairs",
                "issues": ["dead air"],
                "repairs": [
                    {"kind": "trim_raw", "item_index": 1, "new_start_s": 12.0, "new_dur_s": 3.0},
                ],
            }
        )
    )
    logs: list[str] = []
    outcome = stage_review(
        tmp_path / "master.mp4",
        timeline,
        [],
        PROJECT,
        _deps(tmp_path, backend=backend, log=logs.append),
    )
    assert outcome.status == "applied"
    assert len(outcome.applied) == 1
    assert outcome.frames == [6.0, 12.0]
    assert any("[review]" in line for line in logs)
    from director_os.otio_engine.timeline import track_by_name

    items = list(track_by_name(timeline, MASTER_TRACK))
    assert items[1].source_range.start_time.to_seconds() == pytest.approx(12.0)


def test_stage_review_clean_round_touches_nothing(tmp_path, timeline, monkeypatch):
    import director_os.worker.review_pass as rp

    monkeypatch.setattr(rp, "sample_master_frames", lambda *a, **k: [(6.0, tmp_path / "f.jpg")])
    outcome = stage_review(
        tmp_path / "master.mp4",
        timeline,
        [],
        PROJECT,
        _deps(tmp_path, backend=_fake_backend(ReviewVerdict(verdict="ok", issues=[], repairs=[]))),
    )
    assert outcome.status == "clean"
    assert outcome.applied == []


def test_stage_review_failing_backend_never_kills(tmp_path, timeline, monkeypatch):
    import director_os.worker.review_pass as rp
    from director_os.agent_runtime.review import ReviewVerdictError

    class _Broken:
        def review(self, frames, evidence):  # noqa: ANN001
            raise ReviewVerdictError("all vision tiers exhausted")

    monkeypatch.setattr(rp, "sample_master_frames", lambda *a, **k: [(6.0, tmp_path / "f.jpg")])
    outcome = stage_review(
        tmp_path / "master.mp4",
        timeline,
        [],
        PROJECT,
        _deps(tmp_path, backend=_Broken()),
    )
    assert outcome.status == "unavailable"
    assert "tiers" in outcome.summary


def test_stage_review_carries_deterministic_issues_even_unavailable(tmp_path, timeline):
    logs: list[str] = []
    outcome = stage_review(
        tmp_path / "master.mp4",
        timeline,
        ["loudness: -3 dBFS"],
        PROJECT,
        _deps(tmp_path, backend=None, log=logs.append),
    )
    assert outcome.status == "unavailable"
    # the chronology crime of the fixture timeline is EVIDENCED even with no viewer
    assert any(i.startswith("chronology:") for i in outcome.issues)


def test_build_evidence_is_bounded_and_names_program(tmp_path):
    evidence = build_evidence(
        deterministic_issues=["chronology: x"],
        audit_failures=["loudness: hot"],
        timeline=build_timeline(_timeline_source_windows()),
        master_path=tmp_path / "m.mp4",
    )
    assert "MASTER: m.mp4" in evidence
    assert "raw_body" in evidence
    assert "STRUCTURAL FINDS" in evidence and "QA FAILURES" in evidence
    assert len(evidence) <= 4000
    assert "max 12 repairs" in evidence


def test_build_review_backend_accepts_fulfilled_port(tmp_path):
    backend = _fake_backend(ReviewVerdict())
    deps = _deps(tmp_path, backend=backend)
    assert build_review_backend(deps) is backend


def test_apply_repairs_duration_floor_refused_before_render(timeline):
    from director_os.otio_engine.timeline import track_by_name

    applied, refused = apply_repairs(
        timeline,
        PROJECT,
        [_rr(kind="remove_clip", item_index=1, reason="dead air")],
        min_final_s=20.0,
    )
    assert applied == []
    assert refused and "DURATION LAW" in refused[0]
    items = list(track_by_name(timeline, MASTER_TRACK))
    assert len(items) == 4  # the program is untouched - no re-render is burned


def test_apply_repairs_duration_floor_allows_safe_repairs(timeline):
    applied, refused = apply_repairs(
        timeline,
        PROJECT,
        [
            ReviewRepair(
                kind="trim_raw",
                item_index=1,
                new_start_s=10.0,
                new_dur_s=6.0,
                reason="tighten dead air",
            )
        ],
        min_final_s=20.0,
    )
    assert refused == [] and len(applied) == 1
