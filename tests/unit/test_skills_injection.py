"""SKILL.md dynamic injection — niche playbooks into agent system prompts."""

from __future__ import annotations

import hashlib

import pytest
from director_os.agent_runtime.prompting import compose_agent_prompt
from director_os.agent_runtime.registry import ToolRegistry
from director_os.skills.loader import SkillPrompt, build_system_prompt


def test_load_skill_bodycam_playbook():
    from director_os.skills import load_skill

    md = load_skill("bodycam_documentary")
    assert len(md) > 200  # a real playbook, not a stub
    assert "#" in md  # markdown structure present


def test_unknown_niche_raises():
    from director_os.skills import load_skill

    with pytest.raises(FileNotFoundError):
        load_skill("no_such_niche")


def test_skill_prompt_digest_is_stable_sha256_prefix():
    a = SkillPrompt.load("bodycam_documentary")
    b = SkillPrompt.load("bodycam_documentary")
    expected = hashlib.sha256(a.markdown.encode("utf-8")).hexdigest()[:16]
    assert a.digest == expected
    assert a.digest == b.digest  # deterministic — auditable from tool logs


def test_build_system_prompt_sections_in_order():
    prompt = build_system_prompt(
        agent="director_agent",
        role="director",
        niche="bodycam_documentary",
        state_summary="project=p1 master_items=['Clip']",
        tool_names=["timeline.insert_clip", "timeline.set_blur"],
    )
    assert prompt.startswith(
        "You are director_agent, the director agent of a DirectorOS documentary pipeline."
    )
    assert "## Niche playbook: bodycam_documentary" in prompt
    assert SkillPrompt.load("bodycam_documentary").digest in prompt
    assert "- timeline.insert_clip" in prompt
    assert "- timeline.set_blur" in prompt
    assert "## Current timeline state" in prompt
    # stable injection order: header -> playbook -> toolbox -> state
    assert prompt.index("## Niche playbook") < prompt.index("## Available tools")
    assert prompt.index("## Available tools") < prompt.index("## Current timeline state")


def test_compose_agent_prompt_pulls_registry_names():
    reg = ToolRegistry()

    @reg.tool(name="timeline.trim_clip", description="trim a clip")
    def trim_clip(clip_id: str, start_s: float) -> str:
        return clip_id

    prompt = compose_agent_prompt(
        "director_agent", "director", "bodycam_documentary", state_summary="empty", registry=reg
    )
    assert "- timeline.trim_clip" in prompt
    assert "digest" in prompt


def test_empty_toolbox_and_state_are_omitted():
    prompt = build_system_prompt("assembler_agent", "assembler", "bodycam_documentary")
    assert "## Available tools" not in prompt
    assert "## Current timeline state" not in prompt
