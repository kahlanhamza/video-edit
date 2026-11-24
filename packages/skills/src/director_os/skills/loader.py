"""SKILL.md-style system-prompt injection (PLAN §3.2, M2).

A niche playbook (`skills/<niche>.md`) is loaded at runtime and composed into
every agent system prompt, together with the agent's role and the current
timeline state summary. The prompt carries the playbook's sha256 digest so
runs are auditable (the exact instructions that drove the cut are recoverable
from `agent_tool_logs`).

Niche-general law (owner directive 2026-11-24): the platform is the
narration/commentary editor for ANY escalated-incident footage. Dedicated
playbooks live per niche; AN UNKNOWN niche string can never crash a run —
it resolves to the generic `escalated_incident` playbook with an explicit
fallback marker in the prompt. The shared craft lawbook
(`skills/_common_craft.md`) is appended to EVERY playbook before digesting.
"""

from __future__ import annotations

import contextlib
import hashlib
from dataclasses import dataclass

from director_os.skills import load_skill

GENERIC_NICHE = "escalated_incident"
CRAFT_APPENDIX = "_common_craft"

PROMPT_HEADER = "You are {agent}, the {role} agent of a DirectorOS documentary pipeline."


@dataclass(frozen=True)
class SkillPrompt:
    """A loaded playbook plus its audit digest."""

    niche: str
    markdown: str
    digest: str
    requested_niche: str = ""
    fallback: bool = False

    @classmethod
    def load(cls, niche: str) -> SkillPrompt:
        md = _skill_markdown(niche)
        return cls(niche=niche, markdown=md, digest=cls._digest(md), requested_niche=niche)

    @classmethod
    def resolve(cls, niche: str) -> SkillPrompt:
        """Playbook for `niche`; an unknown niche resolves to the GENERIC
        playbook with `fallback=True` — auditable, never a crash."""
        try:
            loaded = cls.load(niche)
        except FileNotFoundError:
            loaded = cls.load(GENERIC_NICHE)
            return cls(
                niche=loaded.niche,
                markdown=loaded.markdown,
                digest=loaded.digest,
                requested_niche=niche,
                fallback=True,
            )
        return loaded

    @staticmethod
    def _digest(md: str) -> str:
        return hashlib.sha256(md.encode("utf-8")).hexdigest()[:16]


def _skill_markdown(niche: str) -> str:
    """Playbook + the shared craft appendix (digest covers the composition)."""
    md = load_skill(niche)
    with contextlib.suppress(FileNotFoundError):
        md = f"{md.strip()}\n\n{_compose_craft()}"
    return md


def _compose_craft() -> str:
    craft = load_skill(CRAFT_APPENDIX)
    return craft.strip()


def build_system_prompt(
    agent: str,
    role: str,
    niche: str,
    state_summary: str = "",
    tool_names: list[str] | None = None,
) -> str:
    """Compose the full system prompt: role header + playbook + toolbox + state.

    Injection order is stable so tests and LLM logs stay comparable:
    header, playbook, toolbox, state.
    """
    skill = SkillPrompt.resolve(niche)
    label = skill.niche
    if skill.fallback:
        label = f"{skill.niche} (FALLBACK from unknown niche {skill.requested_niche!r})"
    parts = [
        PROMPT_HEADER.format(agent=agent, role=role),
        f"## Niche playbook: {label} (digest {skill.digest})",
        skill.markdown.strip(),
    ]
    if tool_names:
        parts.append(
            "## Available tools (call exactly these names)\n"
            + "\n".join(f"- {t}" for t in tool_names)
        )
    if state_summary:
        parts.append(f"## Current timeline state\n{state_summary.strip()}")
    return "\n\n".join(p for p in parts if p)
