"""SKILL.md-style system-prompt injection (PLAN §3.2, M2).

A niche playbook (`skills/<niche>.md`) is loaded at runtime and composed into
every agent system prompt, together with the agent's role and the current
timeline state summary. The prompt carries the playbook's sha256 digest so
runs are auditable (the exact instructions that drove the cut are recoverable
from `agent_tool_logs`).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from director_os.skills import load_skill

PROMPT_HEADER = "You are {agent}, the {role} agent of a DirectorOS documentary pipeline."


@dataclass(frozen=True)
class SkillPrompt:
    """A loaded playbook plus its audit digest."""

    niche: str
    markdown: str
    digest: str

    @classmethod
    def load(cls, niche: str) -> SkillPrompt:
        md = load_skill(niche)
        digest = hashlib.sha256(md.encode("utf-8")).hexdigest()[:16]
        return cls(niche=niche, markdown=md, digest=digest)


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
    skill = SkillPrompt.load(niche)
    parts = [
        PROMPT_HEADER.format(agent=agent, role=role),
        f"## Niche playbook: {skill.niche} (digest {skill.digest})",
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
