"""Agent system-prompt composition — one call for LLM backends (M2).

Composes the injected niche playbook (SKILL.md-style), the agent role header,
the typed toolbox (from a ToolRegistry) and the current timeline state into a
single system prompt string. Production LLM backends (Pydantic AI, M2 close-out)
pass this straight through as the system instruction; ScriptedBackend ignores
it, keeping tests hermetic.
"""

from __future__ import annotations

from director_os.skills.loader import build_system_prompt


def compose_agent_prompt(
    agent: str,
    role: str,
    niche: str,
    state_summary: str = "",
    registry=None,  # ToolRegistry | None (avoid import cycle at runtime)
) -> str:
    tool_names = registry.names() if registry is not None else None
    return build_system_prompt(
        agent=agent, role=role, niche=niche, state_summary=state_summary, tool_names=tool_names
    )
