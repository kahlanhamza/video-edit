"""Pipeline URL ingest (UA guardrail) + subtitle stage wiring (M2 close-out)."""

from __future__ import annotations

import dataclasses
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from director_os.core.config import get_settings
from director_os.core.db import JobRow, ProjectRow, init_models, make_engine, make_session_factory
from director_os.core.models import JobStage, ProjectStatus
from director_os.core.qa_probe import audit_layout_and_policy
from director_os.otio_engine.segment_cache import VideoRenderSpec
from director_os.perception.transcript import Transcript, TranscriptSegment, WordTiming
from director_os.worker.pipeline import PipelineDeps, PipelineError, run_pipeline


def fake_transcriber_factory(call_log: list[str]):
    def transcribe(path, asset_id: str):
        call_log.append("asr")
        words = []
        segments = []
        t = 0.0
        speakers = ["OFFICER_1", "CIVILIAN"]
        for i in range(12):  # 12 words x 0.5s = 6s of speech
            speaker = speakers[i % 2]
            words.append(
                WordTiming(
                    text=f"word{i}", start_s=round(t, 2), end_s=round(t + 0.4, 2), speaker=speaker
                )
            )
            t += 0.5
        for speaker in speakers:
            spk = [w for w in words if w.speaker == speaker]
            segments.append(
                TranscriptSegment(
                    speaker=speaker,
                    start_s=spk[0].start_s,
                    end_s=spk[-1].end_s,
                    text=" ".join(w.text for w in spk),
                    words=spk,
                )
            )
        return Transcript(
            asset_id=asset_id,
            language="en",
            duration_s=6.0,
            words=words,
            segments=segments,
            provider="stub",
        )

    return transcribe


@pytest.fixture()
def pipeline_and_project(tmp_path: Path) -> tuple[PipelineDeps, str]:
    settings = get_settings()
    settings = dataclasses.replace(
        settings,
        output_dir=tmp_path / "output",
        ingest_dir=tmp_path / "output" / "ingest",
        segments_dir=tmp_path / "output" / "segments",
    )
    engine = make_engine(f"sqlite:///{tmp_path}/pipeline_url.db")
    init_models(engine)
    sessions = make_session_factory(engine)
    with sessions() as s:
        row = ProjectRow(title="URL case", status=ProjectStatus.created.value)
        s.add(row)
        s.commit()
        pid = row.id
    deps = PipelineDeps(
        sessions=sessions,
        settings=settings,
        video=VideoRenderSpec(w=320, h=240, fps=30, preset="ultrafast", crf=34),
    )
    return deps, pid


class _HTTPFileServer:
    """Serves one payload; records every request's User-Agent header."""

    def __init__(self, payload: bytes):
        self.payload = payload
        self.user_agents: list[str] = []

        outer = self

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                outer.user_agents.append(self.headers.get("User-Agent", ""))
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(len(outer.payload)))
                self.end_headers()
                self.wfile.write(outer.payload)

            def log_message(self, *a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/fixture_bodycam.mp4"

    def stop(self):
        self.server.shutdown()


def test_pipeline_accepts_url_with_ua_guardrail(fixture_video, pipeline_and_project):
    deps, pid = pipeline_and_project
    server = _HTTPFileServer(fixture_video.read_bytes())
    try:
        calls: list[str] = []
        deps.transcribe = fake_transcriber_factory(calls)
        result = run_pipeline(server.url, pid, deps)

        assert result.audit.ok
        assert calls == ["asr"]  # exactly one ASR pass, on the downloaded copy
        assert server.user_agents and "DirectorOS" in server.user_agents[0]
        downloaded = list(Path(deps.settings.ingest_dir).glob("*"))
        assert downloaded and downloaded[0].stat().st_size == len(server.payload)

        with deps.sessions() as s:
            job = s.get(JobRow, result.job_id)
            assert job.current_stage == JobStage.human_gate.value
            assert job.progress_pct == 100.0
    finally:
        server.stop()


def test_pipeline_writes_policy_clean_subtitles(fixture_video, pipeline_and_project):
    deps, pid = pipeline_and_project
    calls: list[str] = []
    deps.transcribe = fake_transcriber_factory(calls)
    result = run_pipeline(fixture_video, pid, deps)

    ass_path = result.otio_path.parent / "subtitles.ass"
    assert ass_path.exists()
    text = ass_path.read_text(encoding="utf-8")
    assert "[Events]" in text
    assert "Lane1_" in text  # two-lane law applied

    # the sidecar passes the layout/policy audit out of the box
    report = audit_layout_and_policy(ass_path=ass_path)
    assert report.ok, [c.detail for c in report.failures]


def test_pipeline_empty_transcript_yields_empty_ass(fixture_video, pipeline_and_project):
    deps, pid = pipeline_and_project

    def empty_transcriber(path, asset_id):
        return Transcript(
            asset_id=asset_id, language="en", duration_s=1.0, words=[], segments=[], provider="stub"
        )

    deps.transcribe = empty_transcriber
    result = run_pipeline(fixture_video, pid, deps)
    ass_path = result.otio_path.parent / "subtitles.ass"
    assert ass_path.exists() and ass_path.read_text(encoding="utf-8") == ""


def test_download_reports_byte_progress_and_ua(pipeline_and_project, fixture_video):
    """core.http.download streams progress_cb(received, total) per chunk."""
    from director_os.core.http import download

    deps, _pid = pipeline_and_project
    server = _HTTPFileServer(fixture_video.read_bytes())
    try:
        seen: list[tuple[int, int | None]] = []
        dest = deps.settings.ingest_dir / "progress_fixture.mp4"
        out = download(server.url, dest, progress_cb=lambda r, t: seen.append((r, t)))
        assert out.exists() and out.stat().st_size == len(server.payload)
        assert seen, "progress callback never fired"
        last_received, last_total = seen[-1]
        assert last_received == len(server.payload)
        assert last_total == len(server.payload)  # Content-Length known
        assert seen[0][0] <= last_received  # monotonic
    finally:
        server.stop()


def test_url_download_failure_records_job_error(pipeline_and_project):
    """A dead URL must mark the JOB failed — never leave a zombie 'running'.

    Regression for the shipped zombie-job bug: the download used to run
    BEFORE the tracked failure scope, so a failed fetch raised with the job
    row untouched (ingest 0%, error NULL, Run clicks said nothing).
    """
    deps, pid = pipeline_and_project
    calls: list[str] = []
    deps.transcribe = fake_transcriber_factory(calls)
    # port 1 is guaranteed closed — connect fails fast, no external traffic
    dead_url = "http://127.0.0.1:1/never_gets_here.mp4"

    with pytest.raises((PipelineError, Exception)):  # noqa: B017 — httpx wraps too
        run_pipeline(dead_url, pid, deps)

    with deps.sessions() as s:
        from director_os.core.db import JobRow as JRow

        job = s.query(JRow).filter(JRow.project_id == pid).one_or_none()
        assert job is not None
        assert job.error_message, "the job row must record WHERE the run died"
        assert job.progress_pct < 100.0
    assert calls == []  # ASR never ran — nothing was spent
