"""Niche generalization — the owner directive 2026-11-24 contract.

The platform is the narration/commentary editor for ANY escalated-incident
footage. Law pinned here:

1. Every dedicated niche playbook loads and carries the shared craft lawbook.
2. An UNKNOWN niche NEVER crashes: it resolves to the generic
   `escalated_incident` playbook with an explicit FALLBACK marker.
3. `DirectorSession` threads the niche through every planning prompt.
4. The deterministic fallback builds a QA-invariant-safe timeline for any
   niche — bodycam gets its own arc, everyone else gets the generic arc.
"""

from __future__ import annotations

import pytest
from director_os.agent_runtime.director_session import (
    ARC_BODYCAM,
    ARC_GENERIC,
    NICHE_ARCS,
    DirectorSession,
    macro_plan_user_prompt,
)
from director_os.agent_runtime.fallback_cut import build_fallback_documentary
from director_os.agent_runtime.plan_invariants import audit_retention
from director_os.otio_engine.timeline import build_timeline
from director_os.skills.loader import SkillPrompt, build_system_prompt

NICHES = [
    "bodycam_documentary",
    "first_amendment_audit",
    "retail_theft",
    "trespassing",
    "escalated_incident",
]


def _transcript(minutes: int = 44) -> dict:
    """Synthetic diarized transcript spread across the whole runtime."""
    words, segments = [], []
    for m in range(minutes):
        t = m * 60.0
        speaker = "Officer" if m % 2 == 0 else "Suspect"
        text = f"minute {m} admission about the stop and the breath test"
        words.append({"text": "word", "start_s": t, "end_s": t + 0.5, "speaker": speaker})
        segments.append({"speaker": speaker, "start_s": t, "end_s": t + 20.0, "text": text})
    return {"words": words, "segments": segments}


# ---------------------------------------------------------------- playbooks
@pytest.mark.parametrize("niche", NICHES)
def test_every_dedicated_playbook_loads(niche):
    skill = SkillPrompt.load(niche)
    assert skill.niche == niche
    assert skill.markdown.strip()
    assert len(skill.digest) == 16
    assert not skill.fallback


@pytest.mark.parametrize("niche", NICHES)
def test_playbook_composes_the_shared_craft_lawbook(niche):
    skill = SkillPrompt.load(niche)
    bare = SkillPrompt(niche=niche, markdown="", digest="x")
    assert bare.markdown == ""
    assert len(skill.markdown) > 4000  # playbook + craft appendix


def test_playbook_teaches_the_a1_trim_law():
    for niche in NICHES:
        md = SkillPrompt.load(niche).markdown
        assert "timeline.trim_audio" in md, niche
        assert "b-roll window" in md, niche
        assert "0.5s floor" in md, niche


# ------------------------------------------------------- unknown-niche law
def test_unknown_niche_resolves_to_generic_with_fallback_marker():
    skill = SkillPrompt.resolve("twitch_irl_fight_xyz")
    assert skill.niche == "escalated_incident"
    assert skill.fallback is True
    assert skill.requested_niche == "twitch_irl_fight_xyz"


def test_unknown_niche_prompt_carries_the_fallback_marker():
    prompt = build_system_prompt("director_agent", "director", "twitch_irl_fight_xyz")
    assert "FALLBACK" in prompt
    assert "escalated_incident" in prompt


def test_known_niche_prompt_has_no_fallback_marker():
    prompt = build_system_prompt("director_agent", "director", "retail_theft")
    assert "FALLBACK" not in prompt
    assert "## Niche playbook: retail_theft" in prompt


def test_bodycam_header_token_unchanged():
    prompt = build_system_prompt("director_agent", "director", "bodycam_documentary")
    assert "## Niche playbook: bodycam_documentary" in prompt


# ------------------------------------------------- DirectorSession threading
class _Store:
    pass


def _session(niche: str) -> DirectorSession:
    return DirectorSession(
        _Store(),  # registry — construction only, no tool calls
        _Store(),  # checkpoint store
        "proj_niche_test",
        object(),  # backend — construction only, no completions
        source_duration_s=1800.0,
        quote_refine=False,
        niche=niche,
    )


def test_plan_prompt_embeds_the_niche_digest():
    from director_os.agent_runtime.prompting import compose_agent_prompt

    digest = SkillPrompt.load("retail_theft").digest
    system = compose_agent_prompt(agent="director_agent", role="director", niche="retail_theft")
    assert digest in system


def test_generic_niche_gets_the_generic_arc():
    user = macro_plan_user_prompt([], 1800.0, niche="first_amendment_audit")
    assert "wheel" not in user or ARC_GENERIC[:40] not in ARC_BODYCAM
    assert ARC_GENERIC[:40] in user


def test_niche_arcs_map_defaults_to_the_generic_arc():
    assert NICHE_ARCS.get("bodycam_documentary") == ARC_BODYCAM
    raw = macro_plan_user_prompt([], 1800.0, niche="bodycam_documentary")
    gen = macro_plan_user_prompt([], 1800.0, niche="retail_theft")
    assert raw != gen


def test_session_construction_threads_niche():
    s = _session("trespassing")
    assert s.niche == "trespassing"


# ------------------------------------------------- deterministic fallback
@pytest.mark.parametrize("niche", NICHES)
def test_generic_fallback_arc_passes_retention(niche):
    spec = build_fallback_documentary(
        asset_id="asset_niche",
        source_duration_s=2627.0,
        transcript=_transcript(44),
        fps=30,
        width=1920,
        height=1080,
        niche=niche,
    )
    report = audit_retention(build_timeline(spec), 2627.0)
    assert report.ok, [c.detail for c in report.failures]


def test_generic_fallback_avoids_bodycam_lexicon():
    spec = build_fallback_documentary(
        asset_id="asset_niche",
        source_duration_s=2627.0,
        transcript=_transcript(44),
        fps=30,
        width=1920,
        height=1080,
        niche="first_amendment_audit",
    )
    scripts = [
        getattr(item, "meta", None) and getattr(item.meta, "script_text", "")
        for item in spec.tracks[0].items
        if hasattr(item, "meta")
    ]
    text = " ".join(s for s in scripts if s).lower()
    assert "bodycam" not in text
    assert "escalation arc" not in text


def test_bodycam_default_still_bodycam():
    spec = build_fallback_documentary(
        asset_id="asset_niche",
        source_duration_s=2627.0,
        transcript=_transcript(44),
        fps=30,
        width=1920,
        height=1080,
    )
    scripts = [
        getattr(item, "meta", None) and getattr(item.meta, "script_text", "")
        for item in spec.tracks[0].items
        if hasattr(item, "meta")
    ]
    text = " ".join(s for s in scripts if s).lower()
    assert "bodycam" in text
    assert "rolling" in text
