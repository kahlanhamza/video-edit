"""ASR input law — never upload the raw source to Gladia.

CRIT-2026-11-23 (the owner's "critical bug" report): perception uploaded the
RAW 469 MB bodycam video to Gladia, whose edge answered HTTP 408 / dropped
the connection mid-body — the pipeline could never pass "Transcribe audio
(ASR)". Ingest already builds `audio64k.m4a` (AAC 64 kbps mono, ~20 MB for a
43-min source); ASR rides it. Locked here:

- an asset with a persisted audio bundle uploads THAT file
- a legacy asset without one gets a bundle extracted (and persists the path)
- extraction failure degrades to the raw source — loudly, never silently
"""

from __future__ import annotations

from pathlib import Path

import pytest
from director_os.core.db import MediaAssetRow
from director_os.worker.pipeline import _asr_input_path


@pytest.fixture()
def asset_project(tmp_path: Path):
    """A project + a registered asset (no audio bundle) + its deps."""
    import dataclasses

    from director_os.core.config import get_settings
    from director_os.core.db import ProjectRow, init_models, make_engine, make_session_factory
    from director_os.core.models import ProjectStatus
    from director_os.worker.pipeline import PipelineDeps

    settings = get_settings()
    settings = dataclasses.replace(settings, output_dir=tmp_path / "output")
    engine = make_engine(f"sqlite:///{tmp_path}/asr_input.db")
    init_models(engine)
    sessions = make_session_factory(engine)
    with sessions() as s:
        p = ProjectRow(title="ASR input case", status=ProjectStatus.ingested.value)
        s.add(p)
        s.flush()
        asset = MediaAssetRow(
            project_id=p.id,
            source_url=str(tmp_path / "raw_bodycam.webm"),
            role="raw_bwc",
            duration_s=2600.0,
            fps=30.0,
            width=1280,
            height=720,
        )
        s.add(asset)
        s.commit()
        pid, asset_id = p.id, asset.id

    deps = PipelineDeps(sessions=sessions, settings=settings)
    return deps, sessions, pid, asset_id, tmp_path


def test_bundle_on_the_row_wins_over_raw_source(asset_project):
    """r2_audio_key pointing at a real file -> THAT file is the ASR input."""
    deps, sessions, pid, asset_id, tmp_path = asset_project
    bundle = tmp_path / "output" / "bundle" / "audio64k.m4a"
    bundle.parent.mkdir(parents=True)
    bundle.write_bytes(b"fake-aac-bytes")
    with sessions() as s:
        row = s.get(MediaAssetRow, asset_id)
        row.r2_audio_key = str(bundle)
        s.commit()

    raw = tmp_path / "raw_bodycam.webm"
    raw.write_bytes(b"469-mb-of-video")
    out = _asr_input_path(asset_id, raw, pid, deps)
    assert out == bundle


def test_missing_bundle_is_extracted_and_persisted(asset_project, monkeypatch):
    """Legacy asset (no bundle): extraction runs, the path lands on the row."""
    deps, sessions, pid, asset_id, tmp_path = asset_project
    raw = tmp_path / "raw_bodycam.webm"
    raw.write_bytes(b"legacy-video")
    captured: list[list[str]] = []

    def fake_ffmpeg(cmd, what):
        captured.append(cmd)
        # the extractor names its output; materialize it like the real ffmpeg
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"extracted-aac")

    monkeypatch.setattr("director_os.worker.pipeline._run_ffmpeg", fake_ffmpeg)
    out = _asr_input_path(asset_id, raw, pid, deps)
    assert out.name == "audio64k.m4a"
    assert out.exists()
    assert captured, "extraction must actually run for a legacy asset"
    with sessions() as s:
        row = s.get(MediaAssetRow, asset_id)
        assert row.r2_audio_key == str(out)


def test_second_run_skips_extraction(asset_project, monkeypatch):
    """The persisted bundle short-circuits — a retry never re-encodes."""
    deps, sessions, pid, asset_id, tmp_path = asset_project
    raw = tmp_path / "raw_bodycam.webm"
    raw.write_bytes(b"legacy-video")
    bundle = tmp_path / "output" / "pipeline" / pid / "assets" / asset_id / "audio64k.m4a"
    bundle.parent.mkdir(parents=True)
    bundle.write_bytes(b"already-extracted")
    with sessions() as s:
        s.get(MediaAssetRow, asset_id).r2_audio_key = str(bundle)
        s.commit()
    monkeypatch.setattr(
        "director_os.worker.pipeline._run_ffmpeg",
        lambda cmd, what: (_ for _ in ()).throw(AssertionError("must not re-extract")),
    )
    out = _asr_input_path(asset_id, raw, pid, deps)
    assert out == bundle


def test_extraction_failure_falls_back_to_raw_loudly(asset_project, monkeypatch):
    """No bundle + dead extraction -> the raw source rides, with a console line."""
    deps, sessions, pid, asset_id, tmp_path = asset_project
    raw = tmp_path / "raw_bodycam.webm"
    raw.write_bytes(b"no-audio-stream")
    logs: list[str] = []

    def dead_ffmpeg(cmd, what):
        raise RuntimeError("no audio stream")

    monkeypatch.setattr("director_os.worker.pipeline._run_ffmpeg", dead_ffmpeg)
    monkeypatch.setattr(deps, "log", logs.append)
    out = _asr_input_path(asset_id, raw, pid, deps)
    assert out == raw
    assert any("extraction failed" in line for line in logs)
