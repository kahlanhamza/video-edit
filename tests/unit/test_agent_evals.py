"""CI binding for the agent evals harness (scripts/agent_evals.py).

The harness is the SAME fixed scenario set scoring every agent version
(docs/research/agent-research-dossier.md, claim 3: "run evaluations of your
tools"). Importing it here means the scenarios cannot rot in a side script:
CI runs them on every push, RED on any law violation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agent_evals.py"
_spec = importlib.util.spec_from_file_location("agent_evals", _SCRIPT)
assert _spec is not None and _spec.loader is not None, f"evals harness missing at {_SCRIPT}"
agent_evals = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agent_evals)


@pytest.mark.parametrize(
    ("name", "fn"),
    agent_evals.SCENARIOS,
    ids=[s[0].split()[0] for s in agent_evals.SCENARIOS],
)
def test_agent_eval_scenario(name: str, fn, tmp_path: Path) -> None:
    ok, detail = fn(tmp_path)
    assert ok, f"{name}: {detail}"
