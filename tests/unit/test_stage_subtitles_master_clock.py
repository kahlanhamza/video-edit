"""stage_subtitles persists the MASTER-CLOCK cue store (v6 caption truth law).

The store the Studio grid / program monitor / ship pack read must be timed on
the EDITED timeline — narration cues from the TTS sidecars, raw cues remapped
through kept windows, zero raw captions over b-roll — and identical to what
post-fx burns on the deliverable (one shared builder).
"""

from __future__ import annotations

import json

from director_os.otio_engine.contract import NarratorClipMeta, RawClipMeta
from director_os.otio_engine.timeline import (
    ClipSpec,
    GapSpec,
    TimelineSpec,
    TrackSpec,
    build_timeline,
)
from director_os.subtitles.store import load_cues
from director_os.worker.pipeline import PipelineDeps, stage_subtitles


class _Settings:
    def __init__(self, output_dir):
        self.output_dir = output_dir


def _deps(tmp_path, log=None):
    return PipelineDeps(
        sessions=None,
        settings=_Settings(tmp_path),
        log=log or (lambda _msg: None),
    )


def _timeline():
    spec = TimelineSpec(
        name="store",
        fps=30,
        width=640,
        height=360,
        tracks=[
            TrackSpec(
                name="V1_master",
                items=[
                    ClipSpec(
                        name="raw1",
                        source_start_s=100.0,
                        dur_s=10.0,
                        meta=RawClipMeta(asset_id="a"),
                    ),
                    ClipSpec(
                        name="narr1",
                        source_start_s=500.0,
                        dur_s=8.0,
                        meta=NarratorClipMeta(script_text="the narrator explains the case here"),
                    ),
                ],
            ),
            TrackSpec(name="V2_overlay_cards", items=[GapSpec(dur_s=18.0)]),
        ],
    )
    return build_timeline(spec)


def test_stage_subtitles_store_is_master_clock(tmp_path):
    base = tmp_path / "pipeline" / "proj1"  # the stage's store root
    transcript = {
        "words": [
            {"text": "stop", "start_s": 102.0, "end_s": 102.5, "speaker": "Officer"},
            {"text": "the", "start_s": 102.6, "end_s": 102.8, "speaker": "Officer"},
            {"text": "vehicle", "start_s": 103.0, "end_s": 103.6, "speaker": "Officer"},
            # b-roll window speech — must never caption
            {"text": "background", "start_s": 502.0, "end_s": 502.6, "speaker": "Suspect"},
        ]
    }
    narration_dir = base / "narration"
    narration_dir.mkdir(parents=True)
    (narration_dir / "clips.json").write_text(
        json.dumps(
            {
                "clips": [
                    {
                        "script_text": "the narrator explains the case here",
                        "sidecar": "",
                        "duration_s": 6.0,
                    }
                ]
            }
        )
    )
    logs: list[str] = []
    out = stage_subtitles(_timeline(), transcript, "proj1", _deps(tmp_path, logs.append))
    assert out.name == "subtitles.ass"

    beats = load_cues(base)
    assert beats
    texts = " ".join(b.text for b in beats)
    assert "stop the vehicle" in texts
    assert "background" not in texts  # narrator-on-broll isolation
    narr = [b for b in beats if b.speaker == "Narrator"]
    assert narr and narr[0].start_ms >= 10_000  # master clock, after the raw clip
    officer = [b for b in beats if b.speaker == "Officer"]
    assert officer and officer[0].start_ms < 5_000  # remapped 102s -> 2s
    assert any("master-clock cue" in line for line in logs)

    # the burned ASS is styled per the committed law and matches the store
    ass = out.read_text()
    assert "PlayResX: 1280" in ass and "DejaVu Sans" in ass
