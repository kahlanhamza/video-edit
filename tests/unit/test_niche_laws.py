"""Niche laws — a narration/commentary editor for ANY escalated-incident
footage (owner directive 2026-11-24: this is the only generality we sell).

The product edits narration/commentary footage across niches: bodycam
documentary (the default), first amendment audit, retail theft, trespassing,
and anything escalated enough to narrate over — NOT general-purpose video
editing (no weddings, no vlogs, no gaming). Every editing surface must stay
narration-centric; these tests pin the invariants so a general-purpose drift
fails CI instead of shipping.
"""

from __future__ import annotations

import opentimelineio.opentime as ot_time
import opentimelineio.schema as ot_schema
import pytest
from director_os.agent_runtime.editing_tools import DirectorToolbox, ToolOperationError
from director_os.otio_engine.contract import (
    VO_MIN_WINDOW_S,
    NarratorClipMeta,
    RawClipMeta,
    attach,
)


def _timeline_with_raw_and_narration() -> tuple[ot_schema.Timeline, int, int]:
    tl = ot_schema.Timeline()
    master = ot_schema.Track(name="V1_master")
    tl.tracks.append(master)
    raw = ot_schema.Clip(name="raw_0")
    raw.source_range = ot_time.TimeRange(
        start_time=ot_time.RationalTime(0, 30), duration=ot_time.RationalTime(20 * 30, 30)
    )
    attach(raw, RawClipMeta(asset_id="src_1", audio_mode="loudnorm_48k"))
    narr = ot_schema.Clip(name="narration_1")
    narr.source_range = ot_time.TimeRange(
        start_time=ot_time.RationalTime(0, 30), duration=ot_time.RationalTime(9 * 30, 30)
    )
    attach(narr, NarratorClipMeta(script_text="Officers attempted the traffic stop twice."))
    master.extend([raw, narr])
    return tl, 0, 1  # item indices


def test_audio_trim_is_vo_only_raw_keeps_its_source_audio():
    """The niche law: the A1 track is the VOICEOVER track. Raw commentary
    footage keeps its own audio — trimming a raw clip means trimming its
    SOURCE window (timeline.trim_clip), never an audio window."""
    tl, raw_idx, narr_idx = _timeline_with_raw_and_narration()
    tb = DirectorToolbox(tl, "niche")
    with pytest.raises(ToolOperationError, match="not a narration clip"):
        tb.trim_audio(raw_idx, vo_in_s=1.0, vo_out_s=4.0)
    # and the narration block still trims fine — the tool exists FOR it
    r = tb.trim_audio(narr_idx, vo_in_s=0.4, vo_out_s=5.0)
    assert r["ok"]


def test_vo_minimum_window_is_one_intelligible_phrase():
    """At +5% speed one word is ~0.3s — a block under 0.5s cannot carry
    narration, so the contract refuses the trim outright."""
    assert VO_MIN_WINDOW_S == 0.5
    with pytest.raises(Exception, match="cannot carry narration"):
        NarratorClipMeta(script_text="x", vo_in_s=0.0, vo_out_s=0.3)


def test_narration_blocks_default_to_muted_broll():
    """Narration REPLACES source audio (the narrator carries the story);
    ambience returns only via an explicit broll_audio_volume request."""
    meta = NarratorClipMeta(script_text="x")
    assert meta.broll_audio_volume == 0.0


def test_niche_default_is_bodycam_documentary_everywhere():
    """Project create + agent plan + chat all default to the narration
    niche — a general-purpose default would silently change the product."""
    import inspect

    import director_os.agent_runtime.llm as llm
    from director_os.api.main import ProjectCreate

    assert ProjectCreate.model_fields["niche_skill"].default == "bodycam_documentary"
    for fn_name in ("plan_steps",):
        fn = getattr(llm, fn_name, None)
        if fn is None:
            continue
        sig = inspect.signature(fn)
        assert sig.parameters["niche"].default == "bodycam_documentary", fn_name


def test_playbook_teaches_the_a1_trim_tool():
    """Dynamic skill injection: the Director Agent must KNOW the VO trim
    exists and when to reach for it (dead air / tails, never mid-word)."""
    from director_os.skills.loader import load_skill

    text = load_skill("bodycam_documentary")
    assert "timeline.trim_audio" in text
    assert "b-roll window" in text, "the VO trim must be taught as stem-only"
    assert "0.5s floor" in text
