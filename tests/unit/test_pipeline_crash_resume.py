"""Durable staged pipeline — the M2 done-when, in miniature.

- a raw source + transcribe callable triggers all stages autonomously
- a simulated worker crash mid-render is survived: the re-run does NOT
  re-run ASR (call count stays 1) and completes the job
- qa.audit_technical gates the final master (48kHz stereo law included)
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from director_os.core.db import (
    JobRow,
    MediaAssetRow,
    PerceptionIndexRow,
    ProjectRow,
    TimelineVersionRow,
)
from director_os.core.models import JobStage, ProjectStatus
from director_os.core.qa_probe import audit_technical
from director_os.otio_engine.segment_cache import VideoRenderSpec
from director_os.perception.transcript import Transcript, TranscriptSegment, WordTiming
from director_os.worker.pipeline import PipelineDeps, PipelineError, run_pipeline
from sqlalchemy import select


@pytest.fixture()
def project(deps) -> str:
    with deps.sessions() as s:
        row = ProjectRow(title="Crash Resume Case", status=ProjectStatus.created.value)
        s.add(row)
        s.commit()
        return row.id


def fake_transcriber_factory(call_log: list[str], fixture: Path):
    """Deterministic stub ASR: words every ~0.5s, two speakers."""

    def transcribe(path: Path, asset_id: str) -> Transcript:
        call_log.append("asr")
        info_dur = 12.0
        words = []
        segments = []
        t = 0.0
        speaker_cycle = ["OFFICER_1", "CIVILIAN"]
        for i in range(int(info_dur / 0.5)):
            speaker = speaker_cycle[i % 2]
            w = WordTiming(
                text=f"w{i}", start_s=round(t, 2), end_s=round(t + 0.4, 2), speaker=speaker
            )
            words.append(w)
            t += 0.5
        for speaker in speaker_cycle:
            spk_words = [w for w in words if w.speaker == speaker]
            if spk_words:
                segments.append(
                    TranscriptSegment(
                        speaker=speaker,
                        start_s=spk_words[0].start_s,
                        end_s=spk_words[-1].end_s,
                        text=" ".join(w.text for w in spk_words),
                        words=spk_words,
                    )
                )
        return Transcript(
            asset_id=asset_id,
            language="en",
            duration_s=info_dur,
            words=words,
            segments=segments,
            provider="stub",
        )

    return transcribe


@pytest.fixture()
def deps(tmp_path: Path):
    from director_os.core.config import get_settings
    from director_os.core.db import init_models, make_engine, make_session_factory

    settings = get_settings()
    settings = dataclasses.replace(
        settings,
        output_dir=tmp_path / "output",
        ingest_dir=tmp_path / "output" / "ingest",
        segments_dir=tmp_path / "output" / "segments",
    )
    engine = make_engine(f"sqlite:///{tmp_path}/pipeline.db")
    init_models(engine)
    return PipelineDeps(
        sessions=make_session_factory(engine),
        settings=settings,
        video=VideoRenderSpec(w=320, h=240, fps=30, preset="ultrafast", crf=34),
    )


def test_pipeline_runs_all_stages_autonomously(fixture_video: Path, tmp_path, project, deps):
    calls: list[str] = []
    deps.transcribe = fake_transcriber_factory(calls, fixture_video)
    result = run_pipeline(fixture_video, project, deps)

    assert result.audit.ok
    assert result.render.rendered >= 1
    assert result.final_path.exists()

    with deps.sessions() as s:
        job = s.get(JobRow, result.job_id)
        assert job.current_stage == JobStage.human_gate.value
        assert job.progress_pct == 100.0
        assert job.error_message is None

        asset = s.execute(select(MediaAssetRow)).scalars().one()
        assert asset.duration_s == pytest.approx(12.0, abs=1.0)

        per = s.get(PerceptionIndexRow, asset.id)
        assert len(per.transcript_json["words"]) == 24
        assert per.transcript_json["provider"] == "stub"

        versions = s.execute(select(TimelineVersionRow)).scalars().all()
        assert len(versions) == 1
        assert versions[0].created_by == "assembler_agent"

        proj = s.get(ProjectRow, project)
        assert proj.status == ProjectStatus.awaiting_review.value
        assert proj.active_otio_version_id == result.otio_version

    calls.clear()
    # second full run: everything idempotent — no ASR, no new versions
    result2 = run_pipeline(fixture_video, project, deps, job_id=result.job_id)
    assert calls == [], "ASR must not re-run when the transcript is persisted"
    assert result2.audit.ok
    with deps.sessions() as s:
        assert len(s.execute(select(TimelineVersionRow)).scalars().all()) == 1


def test_worker_crash_mid_render_never_reruns_asr(fixture_video: Path, tmp_path, project, deps):
    calls: list[str] = []
    deps.transcribe = fake_transcriber_factory(calls, fixture_video)

    import director_os.worker.pipeline as pipeline_mod

    original = pipeline_mod.render_dag

    def crashing_render(*args, **kwargs):
        raise RuntimeError("worker killed mid-encode (simulated)")

    deps.log = lambda msg: None
    # patch INSIDE stage_render's module namespace
    pipeline_mod.render_dag = crashing_render
    try:
        with pytest.raises(RuntimeError, match="worker killed"):
            run_pipeline(fixture_video, project, deps)
    finally:
        pipeline_mod.render_dag = original

    assert calls == ["asr"], "ASR ran exactly once before the crash"
    with deps.sessions() as s:
        job = s.execute(select(JobRow)).scalars().one()
        assert job.current_stage == JobStage.render.value
        assert "worker killed" in (job.error_message or "")

    # --- recovery run: same deps, unpatched renderer
    result = run_pipeline(fixture_video, project, deps)
    assert calls == ["asr"], "crash resume must NOT re-run ASR"
    assert result.audit.ok
    with deps.sessions() as s:
        job = s.get(JobRow, result.job_id)
        assert job.current_stage == JobStage.human_gate.value
        assert job.error_message is None


def test_qa_failure_parks_job_in_qa_failed(fixture_video: Path, tmp_path, project, deps):
    """An out-of-law master (44.1kHz) must fail the audit gate loudly."""
    import subprocess

    calls: list[str] = []
    deps.transcribe = fake_transcriber_factory(calls, fixture_video)

    # render stages fine, but swap the final for a 44.1kHz file before QA
    import director_os.worker.pipeline as pipeline_mod

    original_render_dag = pipeline_mod.render_dag
    law_breaker = tmp_path / "law_breaker.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=30:duration=4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=4",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "34",
            "-c:a",
            "aac",
            "-ar",
            "44100",
            "-ac",
            "1",
            "-shortest",
            str(law_breaker),
        ],
        check=True,
        capture_output=True,
    )

    def swap_render(*args, **kwargs):
        report = original_render_dag(*args, **kwargs)
        import shutil

        shutil.copy(law_breaker, report.final_path)
        return report

    pipeline_mod.render_dag = swap_render
    try:
        with pytest.raises(PipelineError, match="QA failed"):
            run_pipeline(fixture_video, project, deps)
    finally:
        pipeline_mod.render_dag = original_render_dag

    audit = audit_technical(law_breaker)
    assert not audit.ok
    assert any(c.name == "audio_law_48k_stereo" and not c.ok for c in audit.checks)

    with deps.sessions() as s:
        proj = s.get(ProjectRow, project)
        assert proj.status == ProjectStatus.qa_failed.value
        job = s.execute(select(JobRow)).scalars().one()
        assert "QA failed" in (job.error_message or "")
        assert "audio law violated" in (job.error_message or "")


def test_audit_technical_passes_on_good_master(fixture_video: Path):
    audit = audit_technical(fixture_video, expected_dur_s=12.0, duration_tolerance=1.0)
    assert audit.ok, [c.detail for c in audit.failures]
    names = [c.name for c in audit.checks]
    assert set(names) == {
        "streams_present",
        "audio_law_48k_stereo",
        "clean_out_frames",
        "black_frame_budget",
        "duration_drift",
    }
