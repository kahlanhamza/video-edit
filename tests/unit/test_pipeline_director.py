"""Stage 6 planning — LLM 6-act path, deterministic fallback, force_replan.

Hermetic: the Director backend is injected (`deps.director_backend_factory`),
so the full stage_planning flow (scaffold -> Phase A -> Phase B expansion ->
retention check -> timeline version) runs against the real registry and DB
with zero network. The fallback path covers the no-LLM case.
"""

from __future__ import annotations

import dataclasses
import json

import pytest
from director_os.core.config import get_settings
from director_os.core.db import (
    MediaAssetRow,
    ProjectRow,
    TimelineVersionRow,
    init_models,
    make_engine,
    make_session_factory,
    new_id,
)
from director_os.otio_engine.segment_cache import VideoRenderSpec
from director_os.worker.pipeline import PipelineDeps, stage_planning

SOURCE_DUR = 2627.0


def _script_n(n: int, seed: str) -> str:
    """A narration script with EXACTLY n words (the word-count law fixture)."""
    words = seed.split()
    filler = "the record shows the camera keeps rolling and every detail lands on the tape".split()
    i = 0
    while len(words) < n:
        words.append(filler[i % len(filler)])
        i += 1
    return " ".join(words[:n])


def _plan_json() -> str:
    """Word-count-law plan (hook-first): 50-word hook <=8s in, 60-word mid
    blocks, 85-word outro, utterance-aligned raw windows — ~919s, ratio ~0.37.
    Matches the TRANSCRIPT below (40s utterances every 60s)."""
    acts = [
        {
            "name": "Cold Open",
            "blocks": [
                {"type": "raw", "source_start_s": 1662.0, "dur_s": 4.0},
                {
                    "type": "narration",
                    "script": _script_n(
                        50,
                        "July 4th 2022 Eustis Florida. A bodycam is rolling on a "
                        "police officer and what it catches over the next forty "
                        "minutes turns a routine holiday call into a criminal case "
                        "the report never saw coming. Every second of it is on tape.",
                    ),
                    "dur_s": 20.0,
                },
                {"type": "raw", "source_start_s": 1925.0, "dur_s": 45.0},
            ],
        }
    ]
    for i in range(1, 7):
        u = 60 + (i - 1) * 420
        blocks = [
            {
                "type": "narration",
                "script": _script_n(
                    60,
                    f"Act {i} commentary on what the camera just showed and what the "
                    "verified record says about it, plainly",
                ),
                "dur_s": 22.0,
            },
            {"type": "raw", "source_start_s": float(u), "dur_s": 45.0},
            {"type": "raw", "source_start_s": float(u + 180), "dur_s": 45.0},
            {
                "type": "narration",
                "script": _script_n(
                    60,
                    f"Act {i} closing analysis of the contradiction the tape just "
                    "proved, said calmly and on the record",
                ),
                "dur_s": 22.0,
            },
        ]
        if i == 6:
            blocks.append(
                {
                    "type": "narration",
                    "script": _script_n(
                        85,
                        "The record closes the loop: booking finishes, the tests come "
                        "back, and the case files state the verified outcome in plain "
                        "terms for everyone involved in this stop",
                    ),
                    "dur_s": 32.0,
                }
            )
        acts.append({"name": f"Act {i}", "blocks": blocks})
    return json.dumps({"title": "Eustis", "acts": acts, "cards": []})


class FakeDirectorBackend:
    provider = "fake"
    active_label = "fake:director"
    model = "fake:director"

    def __init__(self):
        self.complete_json_calls = 0

    def complete_json(self, system: str, user: str, max_tokens: int = 4096) -> str:
        self.complete_json_calls += 1
        return _plan_json()

    def propose(self, state_summary: str, role: str):
        return []

    def record_results(self, steps, results) -> None:
        pass

    def reset_history(self) -> None:
        pass


@pytest.fixture()
def deps(tmp_path, monkeypatch):
    for var in ("LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL", "OPENROUTER_API_KEY", "VISION_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    settings = dataclasses.replace(
        get_settings(),
        output_dir=tmp_path / "output",
        ingest_dir=tmp_path / "output" / "ingest",
        segments_dir=tmp_path / "output" / "segments",
    )
    get_settings.cache_clear()
    engine = make_engine(f"sqlite:///{tmp_path}/planning.db")
    init_models(engine)
    return PipelineDeps(
        sessions=make_session_factory(engine),
        settings=settings,
        video=VideoRenderSpec(w=640, h=360, fps=30, preset="ultrafast", crf=34),
        log=lambda m: None,
    )


@pytest.fixture()
def project_and_asset(deps):
    with deps.sessions() as s:
        pid = new_id()
        s.add(ProjectRow(id=pid, title="Eustis full tape"))
        s.commit()
        asset = MediaAssetRow(
            project_id=pid,
            source_url="/tmp/eustis.webm",
            role="raw_bwc",
            duration_s=SOURCE_DUR,
            fps=30.0,
            width=1920,
            height=1080,
        )
        s.add(asset)
        s.commit()
        aid = asset.id
    return pid, aid


TRANSCRIPT = {
    "words": [
        {
            "text": "w",
            "start_s": float(t),
            "end_s": float(t) + 0.4,
            "speaker": "Officer" if t % 120 < 60 else "Suspect",
        }
        for t in range(0, 2600, 60)
    ],
    "segments": [
        {
            "speaker": "Officer" if t % 120 < 60 else "Suspect",
            "start_s": float(t),
            "end_s": float(t) + 40.0,
            "text": f"utterance at {t}s about the case",
        }
        for t in range(0, 2600, 60)
    ],
}


def _versions(deps, pid):
    from sqlalchemy import select

    with deps.sessions() as s:
        return (
            s.execute(select(TimelineVersionRow).where(TimelineVersionRow.project_id == pid))
            .scalars()
            .all()
        )


def _final_row(deps, pid):
    rows = _versions(deps, pid)
    return max(rows, key=lambda r: r.version_num)


def test_stage_planning_llm_path_creates_director_version(deps, project_and_asset):
    pid, aid = project_and_asset
    fake = FakeDirectorBackend()
    deps.director_backend_factory = lambda: fake

    timeline, version = stage_planning(aid, TRANSCRIPT, pid, deps)
    assert fake.complete_json_calls >= 1
    # the PERSISTED plan version (checkpoints from the tool loop also exist)
    final = _final_row(deps, pid)
    assert final.created_by == "director_agent"
    assert "LLM 6-act doc" in final.checkpoint_note
    with deps.sessions() as s:
        project = s.get(ProjectRow, pid)
        assert project.active_otio_version_id == version

    from director_os.agent_runtime.plan_invariants import invariants_ok

    ok, failures = invariants_ok(timeline, SOURCE_DUR)
    assert ok, [f.detail for f in failures]


def test_stage_planning_falls_back_without_llm(deps, project_and_asset):
    pid, aid = project_and_asset
    timeline, version = stage_planning(aid, TRANSCRIPT, pid, deps)
    final = _final_row(deps, pid)
    assert final.created_by == "assembler_agent"
    assert "fallback" in final.checkpoint_note
    # no LLM path -> no tool-loop checkpoints, exactly one version row
    assert len(_versions(deps, pid)) == 1

    from director_os.agent_runtime.plan_invariants import invariants_ok

    ok, failures = invariants_ok(timeline, SOURCE_DUR)
    assert ok, [f.detail for f in failures]


def test_stage_planning_short_source_keeps_autocut(deps, project_and_asset):
    pid, aid = project_and_asset
    with deps.sessions() as s:
        asset = s.get(MediaAssetRow, aid)
        asset.duration_s = 12.0
        s.commit()
    _timeline, _version = stage_planning(aid, TRANSCRIPT, pid, deps)
    final = _final_row(deps, pid)
    assert final.created_by == "assembler_agent"
    assert "auto-cut" in final.checkpoint_note


def test_stage_planning_reuse_and_force_replan(deps, project_and_asset):
    pid, aid = project_and_asset
    deps.director_backend_factory = lambda: FakeDirectorBackend()
    _, v1 = stage_planning(aid, TRANSCRIPT, pid, deps)
    n_after_first = len(_versions(deps, pid))
    # second call without force: reuses (no new versions, no new LLM call)
    fake2 = FakeDirectorBackend()
    deps.director_backend_factory = lambda: fake2
    _, v2 = stage_planning(aid, TRANSCRIPT, pid, deps)
    assert v2 == v1
    assert fake2.complete_json_calls == 0
    assert len(_versions(deps, pid)) == n_after_first
    # force_replan: a fresh Director session runs and persists a new version
    _, v3 = stage_planning(aid, TRANSCRIPT, pid, deps, force_replan=True)
    assert v3 > v2
    assert len(_versions(deps, pid)) > n_after_first
    assert "LLM 6-act doc" in _final_row(deps, pid).checkpoint_note
