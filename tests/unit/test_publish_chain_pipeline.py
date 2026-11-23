"""Publish-chain regressions (critique CR-001 / CR-002) — full pipeline, real
render at 320x240, same harness family as the crash-resume test.

CR-001: the ship pack MUST be built from the POST-FX deliverable when the
post-FX pass runs — the pack is what the human uploads to YouTube; baking it
from the pre-fx master silently shipped a cut with no captions/blur/enhanced
audio.

CR-002: the export gate cursor (baked_ops_rev + baked_timeline_version) must
be stamped by EVERY master render — not only the tracked-redaction path.
Without the default-mode stamp the Export button claims "dirty" forever and
publish-draft would 409 a perfectly baked program.

stage_post_fx and stage_qa are stubbed here: their internals have their own
tests; THIS test pins the wiring between them and the ship pack.
"""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

import pytest
from director_os.core.db import (
    MediaAssetRow,
    PerceptionIndexRow,
    ProjectRow,
    TimelineOpRow,
)
from director_os.core.models import ProjectStatus
from director_os.core.qa_probe import AuditCheck, AuditReport
from director_os.otio_engine.segment_cache import VideoRenderSpec
from director_os.worker import pipeline as pipeline_mod
from director_os.worker.pipeline import PipelineDeps, run_pipeline
from sqlalchemy import func, select


def fake_transcriber_factory(call_log: list[str]):
    """Deterministic stub ASR (same shape as the crash-resume harness): words
    every ~0.5s, two speakers, 12s."""
    from director_os.perception.transcript import (
        Transcript,
        TranscriptSegment,
        WordTiming,
    )

    def transcribe(path: Path, asset_id: str) -> Transcript:
        call_log.append("asr")
        info_dur = 12.0
        words: list[WordTiming] = []
        segments: list[TranscriptSegment] = []
        t = 0.0
        speaker_cycle = ["OFFICER_1", "CIVILIAN"]
        for i in range(int(info_dur / 0.5)):
            speaker = speaker_cycle[i % 2]
            words.append(
                WordTiming(
                    text=f"w{i}",
                    start_s=round(t, 2),
                    end_s=round(t + 0.4, 2),
                    speaker=speaker,
                )
            )
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
def project(deps) -> str:
    with deps.sessions() as s:
        row = ProjectRow(title="Publish Chain Case", status=ProjectStatus.created.value)
        s.add(row)
        s.commit()
        return row.id


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


def _stub_qa(final_path: Path) -> AuditReport:
    return AuditReport(
        path=str(final_path),
        ok=True,
        checks=[AuditCheck(name="wiring", ok=True, detail="stubbed for the wiring test")],
    )


def test_gate_stamped_without_tracked_redaction(fixture_video: Path, project, deps, monkeypatch):
    """CR-002: default REDACT_MODE (no tracked backend, no per_frame payload)
    still stamps the gate cursor — the Export button goes clean after a bake."""
    monkeypatch.setattr(pipeline_mod, "stage_qa", lambda final_path, *a, **kw: _stub_qa(final_path))
    calls: list[str] = []
    deps.transcribe = fake_transcriber_factory(calls)
    result = run_pipeline(fixture_video, project, deps)
    assert result.audit.ok

    with deps.sessions() as s:
        asset = s.execute(select(MediaAssetRow)).scalars().one()
        per = s.get(PerceptionIndexRow, asset.id)
        assert per is not None
        stamp = per.redaction_tracks_json or {}
        assert "baked_ops_rev" in stamp, "default-mode render never stamped the gate"
        assert "baked_timeline_version" in stamp

        max_rev = int(
            s.execute(
                select(func.max(TimelineOpRow.rev)).where(TimelineOpRow.project_id == project)
            ).scalar_one()
            or 0
        )
        assert stamp["baked_ops_rev"] == max_rev

        prow = s.get(ProjectRow, project)
        assert stamp["baked_timeline_version"] == prow.active_otio_version_id

    # and the exact gate state the API serves is CLEAN for this stamp
    from director_os.api.main import _compute_gate_state

    with deps.sessions() as s:
        gate = _compute_gate_state(s, project)
    assert gate["dirty"] is False
    assert gate["exported_once"] is True


def test_ship_pack_built_from_post_fx_master(fixture_video: Path, project, deps, monkeypatch):
    """CR-001: with deps.post_fx=True the pack's final_1080p.mp4 must be a
    copy of the post-fx deliverable, not the pre-fx master."""
    from director_os.worker.post_fx import PostFxReport

    calls: list[str] = []
    deps.transcribe = fake_transcriber_factory(calls)
    deps.post_fx = True

    seen: dict = {}

    def fake_stage_post_fx(final_path, timeline, transcript, pid, d, **kw):
        seen["pre_fx_master"] = Path(final_path)
        # a DISTINCT artifact: same container bytes + a marker tail so the
        # pack-copy assertion can tell post-fx from pre-fx by content
        postfx = final_path.with_name("master_postfx.mp4")
        shutil.copy2(final_path, postfx)
        with postfx.open("ab") as f:
            f.write(b"DIRECTOROS-POSTFX-MARKER")
        return PostFxReport(ok=True, final_path=postfx)

    def fake_stage_qa(final_path, *a, **kw):
        seen["qa_path"] = Path(final_path)
        return _stub_qa(final_path)

    monkeypatch.setattr("director_os.worker.post_fx.stage_post_fx", fake_stage_post_fx)
    monkeypatch.setattr(pipeline_mod, "stage_qa", fake_stage_qa)

    result = run_pipeline(fixture_video, project, deps)
    assert result.audit.ok

    # QA audited the POST-fx file, and the pipeline result carries it too
    assert seen["qa_path"] == seen["pre_fx_master"].with_name("master_postfx.mp4")
    assert result.final_path == seen["qa_path"]

    # the pack master is the POST-fx artifact (marker included) — the exact
    # bug CR-001 pinned: it used to be a copy of the pre-fx master
    pack_master = deps.settings.output_dir / "pipeline" / project / "ship_pack" / "final_1080p.mp4"
    assert pack_master.exists(), "ship pack missing — it is the folder the human uploads"
    assert pack_master.read_bytes().endswith(b"DIRECTOROS-POSTFX-MARKER")
