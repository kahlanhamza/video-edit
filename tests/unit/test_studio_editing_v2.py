"""Studio editing v2 — the acceptance battery for this round of fixes.

1. Timeline history endpoints: GET /timeline/versions + POST /timeline/activate
   (undo/redo is moving the active pointer; versions are never destroyed).
2. NLE snapshot carries b-roll windows + the narration streaming key; the
   /narration/{key}.mp3 endpoint serves the synthesized TTS stem.
3. audio_prep without a neural enhancer SKIPS loudly and the render completes
   (loudnorm downgrade) instead of killing the pipeline at 52%.
4. Phase A survives provider errors (LLMBackendError retried; garbage bodies
   surface their head in the error so the operator sees WHY).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


# ------------------------------------------------------------------ fixtures
@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    import director_os.api.main as api_main
    from director_os.core.config import get_settings
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/api.db")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    get_settings.cache_clear()
    api_main._engine = None  # force re-bind to the tmp database
    with TestClient(api_main.app) as client:
        yield client
    get_settings.cache_clear()


def _add_project(api_main) -> str:
    from director_os.core.db import ProjectRow

    api_main.session()  # binds the lazy engine + session factory
    with api_main._session_factory() as s:
        row = ProjectRow(title="Editing v2 case")
        s.add(row)
        s.commit()
        return row.id


def _add_version(api_main, project_id: str, note: str, otio: dict | None = None) -> int:
    from director_os.core.db import ProjectRow, TimelineVersionRow

    with api_main._session_factory() as s:
        last = (
            s.query(TimelineVersionRow)
            .filter(TimelineVersionRow.project_id == project_id)
            .order_by(TimelineVersionRow.version_num.desc())
            .first()
        )
        num = (last.version_num if last else 0) + 1
        s.add(
            TimelineVersionRow(
                project_id=project_id,
                version_num=num,
                otio_json=otio or {},
                created_by="assembler_agent",
                checkpoint_note=note,
            )
        )
        s.get(ProjectRow, project_id).active_otio_version_id = num
        s.commit()
        return num


def _narration_timeline(asset_id: str) -> dict:
    """One raw clip + one narration clip with multi-window b-roll."""
    from director_os.otio_engine.contract import NarratorClipMeta, RawClipMeta
    from director_os.otio_engine.timeline import (
        ClipSpec,
        TimelineSpec,
        TrackSpec,
        build_timeline,
        to_json_dict,
    )

    spec = TimelineSpec(
        name="t",
        fps=30,
        width=320,
        height=240,
        tracks=[
            TrackSpec(
                name="V1_master",
                items=[
                    ClipSpec(
                        name="raw_0",
                        source_start_s=0.0,
                        dur_s=10.0,
                        meta=RawClipMeta(asset_id=asset_id),
                    ),
                    ClipSpec(
                        name="narration_1",
                        source_start_s=100.0,
                        dur_s=15.0,
                        meta=NarratorClipMeta(
                            script_text="The officer steps out of the cruiser.",
                            broll_asset_id=asset_id,
                            broll_range=(100.0, 115.0),
                            broll_windows=[(100.0, 115.0), (200.0, 212.0), (40.0, 50.0)],
                            broll_audio_volume=0.0,
                        ),
                    ),
                ],
            ),
        ],
    )
    return to_json_dict(build_timeline(spec))


def _add_asset(api_main, project_id: str) -> str:
    from director_os.core.db import MediaAssetRow

    with api_main._session_factory() as s:
        row = MediaAssetRow(
            project_id=project_id,
            duration_s=2627.0,
            fps=30.0,
            width=1920,
            height=1080,
        )
        s.add(row)
        s.commit()
        return row.id


# ----------------------------------------------------- 1. versions + activate
def test_timeline_history_list_and_activate(api_client):
    import director_os.api.main as api_main

    pid = _add_project(api_main)
    _add_version(api_main, pid, "auto-cut v1")
    v2 = _add_version(api_main, pid, "manual trim")

    # the EXISTING GET /timeline carries the undo history
    r = api_client.get(f"/projects/{pid}/timeline")
    assert r.status_code == 200
    body = r.json()
    assert [v["version_num"] for v in body["versions"]] == [1, 2]
    assert body["active_version"] == v2

    # undo: activate v1 — nothing is destroyed, the pointer just moves
    r = api_client.post(f"/projects/{pid}/timeline/activate", json={"version_num": 1})
    assert r.status_code == 200
    assert r.json()["active_version"] == 1
    assert r.json()["previous_version"] == v2

    body = api_client.get(f"/projects/{pid}/timeline").json()
    assert body["active_version"] == 1

    # redo: v2 again, unknown versions 404
    assert (
        api_client.post(f"/projects/{pid}/timeline/activate", json={"version_num": 2}).status_code
        == 200
    )
    assert (
        api_client.post(f"/projects/{pid}/timeline/activate", json={"version_num": 99}).status_code
        == 404
    )
    assert api_client.get("/projects/nope/timeline").status_code == 404


# ------------------------------------ 2. NLE snapshot b-roll + narration audio
def test_nle_snapshot_carries_broll_and_streams_narration(api_client, tmp_path):
    import director_os.api.main as api_main
    from director_os.core.config import get_settings

    pid = _add_project(api_main)
    asset_id = _add_asset(api_main, pid)
    _add_version(api_main, pid, "narration cut", otio=_narration_timeline(asset_id))

    r = api_client.get(f"/projects/{pid}/timeline/nle")
    assert r.status_code == 200
    snap = r.json()
    narr = next(c for c in snap["clips"] if c["kind"] == "narration")
    assert narr["broll_windows"] == [[100.0, 115.0], [200.0, 212.0], [40.0, 50.0]]
    assert narr["broll_audio_volume"] == 0.0
    key = narr["narration_key"]
    assert key == hashlib.sha256(b"The officer steps out of the cruiser.").hexdigest()[:24]

    # 404 while the TTS stem is not synthesized yet
    assert api_client.get(f"/projects/{pid}/narration/{key}.mp3").status_code == 404

    # synthesize a fake stem + manifest -> the endpoint streams it
    narr_dir = Path(get_settings().output_dir) / "pipeline" / pid / "narration"
    narr_dir.mkdir(parents=True, exist_ok=True)
    mp3 = narr_dir / "vo.mp3"
    mp3.write_bytes(b"\xff\xfb" + b"0" * 128)
    (narr_dir / "clips.json").write_text(
        json.dumps(
            {
                "clips": [
                    {
                        "script_text": "The officer steps out of the cruiser.",
                        "duration_s": 4.2,
                        "clip": str(mp3),
                        "item": "narration_1",
                    }
                ]
            }
        )
    )
    r = api_client.get(f"/projects/{pid}/narration/{key}.mp3")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/mpeg")
    assert len(r.content) == 130

    # bad keys are rejected without touching the disk
    assert api_client.get(f"/projects/{pid}/narration/zzz!.mp3").status_code in (400, 404)


# --------------------------------- 3. audio_prep graceful-skip without enhancer
def test_audio_prep_skips_without_enhancer(fixture_video: Path, tmp_path):
    """No deep-filter on the machine -> the step is SKIPPED, the render runs
    loudnorm, and the job reaches human_gate (used to die at audio_prep)."""
    import dataclasses

    from director_os.audio_engine.enhance import EnhancerUnavailableError
    from director_os.core.config import get_settings
    from director_os.core.db import (
        JobStepRow,
        ProjectRow,
        init_models,
        make_engine,
        make_session_factory,
    )
    from director_os.otio_engine.segment_cache import VideoRenderSpec
    from director_os.worker.pipeline import PipelineDeps, run_pipeline

    settings = get_settings()
    settings = dataclasses.replace(
        settings,
        output_dir=tmp_path / "output",
        ingest_dir=tmp_path / "output" / "ingest",
        segments_dir=tmp_path / "output" / "segments",
    )
    engine = make_engine(f"sqlite:///{tmp_path}/p.db")
    init_models(engine)
    sessions = make_session_factory(engine)
    with sessions() as s:
        row = ProjectRow(title="no enhancer case")
        s.add(row)
        s.commit()
        pid = row.id

    deps = PipelineDeps(
        sessions=sessions,
        settings=settings,
        video=VideoRenderSpec(w=320, h=240, fps=30, preset="ultrafast", crf=34),
    )
    deps.audio_mode = "isolate_2bus"
    deps.speech_enhancer = None
    deps.transcribe = _stub_transcriber

    # THE hostile machine: no neural enhancer anywhere
    def no_enhancer(*_a, **_kw):
        raise EnhancerUnavailableError("deep-filter CLI not found (test machine)")

    import director_os.audio_engine.enhance as enhance_mod

    original = enhance_mod.resolve_enhancer
    enhance_mod.resolve_enhancer = no_enhancer
    try:
        result = run_pipeline(fixture_video, pid, deps)
    finally:
        enhance_mod.resolve_enhancer = original

    assert result.audit.ok, "the pipeline must COMPLETE without an enhancer"
    assert result.final_path is not None and result.final_path.exists()
    with sessions() as s:
        job = s.query(JobStepRow).filter(JobStepRow.key == "audio_prep").one_or_none()
        assert job is not None and job.status == "skipped", (
            "the audio_prep step must be honestly SKIPPED, not failed"
        )


def _stub_transcriber(path, asset_id):
    from director_os.perception.transcript import Transcript, WordTiming

    words = []
    t = 0.0
    for i in range(24):
        words.append(
            WordTiming(
                text=f"w{i}", start_s=round(t, 2), end_s=round(t + 0.4, 2), speaker="OFFICER_1"
            )
        )
        t += 0.5
    return Transcript(
        asset_id=asset_id,
        language="en",
        duration_s=12.0,
        words=words,
        provider="stub",
    )
