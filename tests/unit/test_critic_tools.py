"""Hermetic tests for the deterministic critic surface (critic.qa_self_audit).

The critic is a TOOL the editor agent must answer to before claiming done
(EditDuet's Editor/Critic pairing, single-agent form). Honest-absence law:
a check that never ran reports 'not established', never 'passed'. No media,
no network: build_timeline fixtures + a tmp work dir.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from director_os.agent_runtime.critic import register_critic_tools
from director_os.agent_runtime.editing_tools import DirectorToolbox
from director_os.agent_runtime.inspection_tools import REG_TIMELINE
from director_os.agent_runtime.registry import ToolRegistry
from director_os.otio_engine.contract import NarratorClipMeta, RawClipMeta
from director_os.otio_engine.timeline import (
    MASTER_TRACK,
    ClipSpec,
    TimelineSpec,
    TrackSpec,
    build_timeline,
    track_by_name,
)


@pytest.fixture()
def timeline():
    return build_timeline(
        TimelineSpec(
            name="critic_case",
            fps=30,
            width=1280,
            height=720,
            tracks=[
                TrackSpec(
                    name=MASTER_TRACK,
                    items=[
                        ClipSpec(
                            name="raw_1",
                            source_start_s=0.0,
                            dur_s=10.0,
                            meta=RawClipMeta(asset_id="a1"),
                        ),
                        ClipSpec(
                            name="raw_2",
                            source_start_s=10.0,
                            dur_s=8.0,
                            meta=RawClipMeta(asset_id="a1"),
                        ),
                        ClipSpec(
                            name="narr",
                            source_start_s=0.0,
                            dur_s=5.0,
                            meta=NarratorClipMeta(script_text="A narration line."),
                        ),
                    ],
                ),
            ],
        )
    )


@pytest.fixture()
def registry(timeline, tmp_path: Path):
    REG_TIMELINE.clear()
    REG_TIMELINE.append(timeline)
    reg = ToolRegistry()
    register_critic_tools(reg, tmp_path)
    yield reg


def test_green_timeline_audits_clean(registry, tmp_path: Path) -> None:
    out = registry.dispatch("qa.self_audit", {"source_duration_s": 30.0})
    assert out["green"] is True
    assert "GREEN" in out["verdict"]
    names = {c["name"] for c in out["checks"]}
    assert {
        "no_master_gaps",
        "no_zero_length_items",
        "narration_scripts_present",
        "zoom_within_subtle_law",
        "plan_invariants",
    } <= names


def test_invariants_absent_are_not_established_not_passed(registry) -> None:
    out = registry.dispatch("qa.self_audit", {})
    inv = next(c for c in out["checks"] if c["name"] == "plan_invariants")
    assert inv.get("not_established") is True
    assert "not established" in inv["detail"]
    # honest-absence does NOT turn the whole report green by luck: the other
    # structural checks still run
    assert any(c["name"] == "no_master_gaps" and c["ok"] for c in out["checks"])


def test_gap_in_master_is_red(registry, timeline) -> None:
    # remove_clip ripples (closes the hole), so build the gap like the
    # compiler does: an explicit Gap item on the master track
    import opentimelineio.opentime as ot_time
    import opentimelineio.schema as ot_schema

    track = track_by_name(timeline, MASTER_TRACK)
    track.insert(
        1,
        ot_schema.Gap(
            source_range=ot_time.TimeRange(
                start_time=ot_time.RationalTime(0, 30), duration=ot_time.RationalTime(8 * 30, 30)
            )
        ),
    )
    out = registry.dispatch("qa.self_audit", {})
    gap_check = next(c for c in out["checks"] if c["name"] == "no_master_gaps")
    assert gap_check["ok"] is False
    assert out["green"] is False
    assert "RED" in out["verdict"]
    assert "no_master_gaps" in out["verdict"]


def test_empty_narration_script_is_red(registry, timeline) -> None:
    # the toolbox rejects set_narration_text("") — the critic guards the
    # paths that bypass it (imports, crash-resume, foreign writers)
    track = track_by_name(timeline, MASTER_TRACK)
    track[2].metadata["director_os"]["script_text"] = ""
    out = registry.dispatch("qa.self_audit", {})
    chk = next(c for c in out["checks"] if c["name"] == "narration_scripts_present")
    assert chk["ok"] is False


def test_zoom_over_subtle_law_is_red(registry, timeline) -> None:
    box = DirectorToolbox(timeline, "critic_case")
    box.set_zoom_keyframes(0, [[0.0, 1.6]])  # over the 1.25 subtle law
    out = registry.dispatch("qa.self_audit", {})
    chk = next(c for c in out["checks"] if c["name"] == "zoom_within_subtle_law")
    assert chk["ok"] is False
    assert "1.6" in " ".join(chk["detail"])


def test_caption_work_qa_reads_session_verdict(registry, timeline, tmp_path: Path) -> None:
    # absent -> not established
    out = registry.dispatch("qa.self_audit", {})
    chk = next(c for c in out["checks"] if c["name"] == "caption_work_qa")
    assert chk.get("not_established") is True
    # a red session QA verdict propagates
    (tmp_path / "qa_caption_work.json").write_text(
        json.dumps(
            {
                "green": False,
                "checks": [
                    {
                        "name": "speech_gap_coverage",
                        "ok": False,
                        "detail": ["[600.0-605.0]s speech without any cue"],
                    }
                ],
            }
        )
    )
    out2 = registry.dispatch("qa.self_audit", {})
    chk2 = next(c for c in out2["checks"] if c["name"] == "caption_work_qa")
    assert chk2["ok"] is False
    assert out2["green"] is False


def test_full_workbench_wiring_green_roundtrip(tmp_path: Path, timeline) -> None:
    """The v3 session wiring: workshop + critic over one work dir."""
    REG_TIMELINE.clear()
    REG_TIMELINE.append(timeline)
    reg = ToolRegistry()
    register_critic_tools(reg, tmp_path)
    box = DirectorToolbox(timeline, "critic_case")
    from director_os.agent_runtime.editing_tools import build_director_registry

    build_director_registry(box)  # mutations land in the same shared object
    reg.dispatch("qa.self_audit", {"source_duration_s": 23.0})
    # the audit itself wrote nothing destructive; report exists only in-memory
    assert reg.names() == sorted(reg.names())
