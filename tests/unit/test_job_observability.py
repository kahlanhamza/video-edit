"""Full-visibility Studio observability — RunTracker lifecycle, SSE live
stream, snapshot endpoint, and the NLE timeline snapshot feed."""

from __future__ import annotations

import pytest


# ------------------------------------------------------------------ RunTracker
@pytest.fixture()
def sessions(tmp_path):
    from director_os.core.db import (
        JobRow,
        ProjectRow,
        init_models,
        make_engine,
        make_session_factory,
    )

    engine = make_engine(f"sqlite:///{tmp_path}/tracker.db")
    init_models(engine)
    factory = make_session_factory(engine)
    with factory() as s:
        s.add(ProjectRow(id="p1", title="case"))
        s.add(JobRow(id="j1", project_id="p1", current_stage="ingest", progress_pct=0.0))
        s.commit()
    return factory


def test_step_lifecycle_done_and_duration(sessions):
    from director_os.worker.progress import RunTracker

    tr = RunTracker(sessions, "j1", "p1")
    with tr.step("ingest", "Ingest source") as st:
        st.pct(40, "probing")
    snap = tr.snapshot()
    assert len(snap["steps"]) == 1
    step = snap["steps"][0]
    assert step["status"] == "done"
    assert step["pct"] == 100.0
    assert step["key"] == "ingest"
    assert step["duration_ms"] >= 0


def test_step_failure_marks_failed_and_logs_error(sessions):
    from director_os.worker.progress import RunTracker

    tr = RunTracker(sessions, "j1", "p1")
    with pytest.raises(RuntimeError), tr.step("render", "Render master"):
        raise RuntimeError("ffmpeg exploded")
    step = tr.snapshot()["steps"][0]
    assert step["status"] == "failed"
    assert "ffmpeg exploded" in step["detail"]
    logs = tr.snapshot()["logs"]
    assert any(e["level"] == "error" and "Render master" in e["message"] for e in logs)


def test_skip_is_terminal_and_not_re_done(sessions):
    from director_os.worker.progress import RunTracker

    tr = RunTracker(sessions, "j1", "p1")
    with tr.step("narration", "Narration TTS") as st:
        st.skip("no narrator clips")
    step = tr.snapshot()["steps"][0]
    assert step["status"] == "skipped"


def test_stage_updates_job_row_and_emits_event(sessions):
    from director_os.core.db import JobRow
    from director_os.worker.progress import RunTracker

    tr = RunTracker(sessions, "j1", "p1")
    tr.stage("render", 60.0)
    tr.log("  [render] 12 segment(s)")
    tr.timeline(3, "story map v3")
    with sessions() as s:
        job = s.get(JobRow, "j1")
        assert job.current_stage == "render"
        assert job.progress_pct == 60.0
    kinds = [e["kind"] for e in tr.events_after(0)]
    assert kinds == ["job", "log", "timeline"]


def test_pct_throttle_skips_identical_broadcasts(sessions):
    from director_os.worker.progress import RunTracker

    tr = RunTracker(sessions, "j1", "p1")
    with tr.step("qa", "QA") as st:
        st.pct(50, "same")
        st.pct(50, "same")  # identical — no second event
        st.pct(51, "same")
    step_events = [e for e in tr.events_after(0) if e["kind"] == "step"]
    # running + pct 50 + pct 51 + done — the duplicate pct 50 was throttled away
    assert len(step_events) == 4
    pcts = [e["payload"]["pct"] for e in step_events]
    assert pcts == [0.0, 50.0, 51.0, 100.0]


def test_error_log_line_auto_escalates_to_warn(sessions):
    from director_os.worker.progress import RunTracker

    tr = RunTracker(sessions, "j1", "p1")
    tr.log("[render] per-clip tracked bake failed (x) — raw source fallback")
    logs = tr.snapshot()["logs"]
    assert logs[0]["level"] == "warn"


# ------------------------------------------------------------------ API: snapshot + live SSE + NLE
@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    import director_os.api.main as api_main
    from director_os.core.config import get_settings
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/api_live.db")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    get_settings.cache_clear()
    api_main._engine = None
    with TestClient(api_main.app) as client:
        yield client, tmp_path
    get_settings.cache_clear()


def _seed_job(api_client):
    import director_os.api.main as api_main
    from director_os.core.db import JobRow

    r = api_client.post("/projects", json={"title": "live case"})
    pid = r.json()["id"]
    s = api_main.session()
    job = JobRow(project_id=pid, current_stage="render", progress_pct=100.0)
    s.add(job)
    s.commit()
    return pid, job.id


def test_snapshot_endpoint_returns_steps_and_logs(api_client):
    from director_os.worker.progress import RunTracker

    client, _ = api_client
    pid, jid = _seed_job(client)
    # find the tracker via a manual one bound to the same job
    import director_os.api.main as api_main

    tr = RunTracker(api_main._session_factory, jid, pid)
    with tr.step("ingest", "Ingest source"):
        pass
    tr.log("hello console")
    r = client.get(f"/projects/{pid}/jobs/{jid}/snapshot")
    assert r.status_code == 200
    body = r.json()
    assert body["steps"][0]["key"] == "ingest"
    assert body["logs"][0]["message"] == "hello console"


def test_live_sse_streams_steps_logs_and_done(api_client):
    import director_os.api.main as api_main
    from director_os.worker.progress import RunTracker

    client, _ = api_client
    pid, jid = _seed_job(client)
    tr = RunTracker(api_main._session_factory, jid, pid)
    with tr.step("ingest", "Ingest source") as st:
        st.pct(30, "half")
    tr.stage("render", 90.0)
    tr.log("line one")
    with client.stream("GET", f"/projects/{pid}/jobs/{jid}/live", params={"max_seconds": 2.0}) as r:
        assert r.status_code == 200
        text = "".join(chunk for chunk in r.iter_text())
    assert "event: step" in text
    assert "event: log" in text
    assert "event: job" in text
    assert "event: done" in text  # terminal (pct 100 seeded)


def test_live_sse_replays_only_after_cursor(api_client):
    import director_os.api.main as api_main
    from director_os.worker.progress import RunTracker

    client, _ = api_client
    pid, jid = _seed_job(client)
    tr = RunTracker(api_main._session_factory, jid, pid)
    tr.log("early line")
    with client.stream("GET", f"/projects/{pid}/jobs/{jid}/live", params={"max_seconds": 1.0}) as r:
        "".join(chunk for chunk in r.iter_text())
    # reconnect from a fresh cursor: the feed replays everything after 0
    events = tr.events_after(0)
    assert events[0]["kind"] == "log"


def _seed_nle_project(api_client, tmp_path):
    """project + asset + active OTIO version + captions + redaction regions."""
    import director_os.api.main as api_main
    from director_os.core.db import (
        MediaAssetRow,
        PerceptionIndexRow,
        ProjectRow,
        TimelineVersionRow,
    )
    from director_os.otio_engine.contract import CardMeta, NarratorClipMeta, RawClipMeta
    from director_os.otio_engine.timeline import (
        ClipSpec,
        TimelineSpec,
        TrackSpec,
        build_timeline,
        to_json_dict,
    )

    client, _ = api_client
    r = client.post("/projects", json={"title": "NLE case"})
    pid = r.json()["id"]
    spec = TimelineSpec(
        name="doc",
        tracks=[
            TrackSpec(
                name="V1_master",
                items=[
                    ClipSpec(
                        name="raw_1",
                        source_start_s=0.0,
                        dur_s=4.0,
                        meta=RawClipMeta(asset_id="a1"),
                    ),
                    ClipSpec(
                        name="narr_1",
                        dur_s=2.0,
                        meta=NarratorClipMeta(script_text="case opens"),
                    ),
                ],
            ),
            TrackSpec(
                name="V2_overlay_cards",
                items=[
                    ClipSpec(
                        name="card_1",
                        dur_s=1.5,
                        meta=CardMeta(template="location_intro", fields={"title": "CASE FILE"}),
                    )
                ],
            ),
        ],
    )
    timeline = build_timeline(spec)
    s = api_main.session()
    asset = MediaAssetRow(
        project_id=pid,
        source_url="/tmp/src.mp4",
        role="raw_bwc",
        duration_s=10.0,
        fps=30.0,
        width=1920,
        height=1080,
    )
    s.add(asset)
    s.flush()
    s.add(
        PerceptionIndexRow(
            asset_id=asset.id,
            redaction_tracks_json={
                "regions": [
                    {
                        "id": "r1",
                        "x": 10,
                        "y": 20,
                        "w": 300,
                        "h": 80,
                        "sigma": 24,
                        "label": "badge",
                        "mode": "tracked",
                        "t_start_s": 1.0,
                        "t_end_s": 4.0,
                        "created_by": "human_edit",
                    }
                ]
            },
        )
    )
    s.add(
        TimelineVersionRow(
            project_id=pid,
            version_num=1,
            otio_json=to_json_dict(timeline),
            created_by="director_agent",
        )
    )
    project = s.get(ProjectRow, pid)
    project.active_otio_version_id = 1
    s.commit()
    # caption cue store
    sub_dir = tmp_path / "out" / "pipeline" / pid
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "subtitles.json").write_text(
        '[{"start_ms": 500, "end_ms": 2500, "text": "OFFICER: 10-4", "speaker": "officer"}]',
        encoding="utf-8",
    )
    return pid, asset.id


def test_timeline_nle_endpoint_full_shape(api_client):
    client, tmp_path = api_client
    pid, asset_id = _seed_nle_project(api_client, tmp_path)
    r = client.get(f"/projects/{pid}/timeline/nle")
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == 1
    assert body["asset"]["width"] == 1920
    assert body["duration_s"] >= 6.0
    video = [c for c in body["clips"] if c["track"] == "V1_master"]
    cards = [c for c in body["clips"] if c["track"] == "V2_overlay_cards"]
    assert len(video) == 2
    assert video[0]["kind"] == "raw"
    assert video[1]["kind"] == "narration"
    assert video[1]["script"] == "case opens"
    # master_index counts gaps too — first two master items are our clips
    assert video[0]["master_index"] == 0
    assert video[1]["master_index"] == 1
    assert video[0]["source_start_s"] == pytest.approx(0.0)
    assert len(cards) == 1 and cards[0]["template"] == "location_intro"
    # starts are timeline-local: raw_1 at 0, narration after it (4s), card at 0
    assert video[0]["start_s"] == pytest.approx(0.0)
    assert video[1]["start_s"] == pytest.approx(4.0)
    assert cards[0]["start_s"] == pytest.approx(0.0)
    assert body["captions"][0]["text"] == "OFFICER: 10-4"
    assert body["effects"][0]["id"] == "r1"
    assert body["effects"][0]["t_start_s"] == 1.0


def test_timeline_nle_404_on_unknown_project(api_client):
    client, _ = api_client
    r = client.get("/projects/nope/timeline/nle")
    assert r.status_code == 404


def test_delete_project_purges_observability_rows(api_client):
    import director_os.api.main as api_main
    from director_os.core.db import JobEventRow, JobLogRow, JobStepRow
    from director_os.worker.progress import RunTracker

    client, _ = api_client
    pid, jid = _seed_job(client)
    tr = RunTracker(api_main._session_factory, jid, pid)
    with tr.step("ingest", "Ingest"):
        pass
    tr.log("bye")
    r = client.delete(f"/projects/{pid}")
    assert r.status_code == 204
    s = api_main.session()
    try:
        assert s.query(JobStepRow).filter_by(project_id=pid).count() == 0
        assert s.query(JobLogRow).filter_by(project_id=pid).count() == 0
        assert s.query(JobEventRow).filter_by(project_id=pid).count() == 0
    finally:
        s.close()
