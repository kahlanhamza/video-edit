#!/usr/bin/env python3
"""Director session v5 — THE OPERATOR IS THE LLM.

Gemini is rate-limited (owner verdict: "it is unreliable"), the z-ai bridge is
a downgrade, so the session's LLM role is played by the operator assistant
directly: each round's tool steps are authored from the operator reading the
PREVIOUS round's measured tool results. The protocol stays identical to
v3/v4 — JSON steps, typed registry, checkpoint-before-execute, fail-closed
dispatch, durable session memory, deterministic critic floor — only the
brain is swapped. Every proposal cites the evidence it saw.

Usage:
  python agent_session_me.py round <N>       # execute round N decisions
  python agent_session_me.py report          # write run_report + copy artifacts

Decisions: pipeline_run/agent_run_me/decisions_<N>.json
  {"steps": [{"tool": "...", "args": {...}, "why": "one line"}]}
Results:   pipeline_run/agent_run_me/round_state_<N>.json
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for _p in sorted(REPO.glob("packages/*/src")):
    sys.path.insert(0, str(_p))  # noqa: E402 — sandbox bootstrap
sys.path.insert(0, str(REPO / "apps" / "worker" / "src"))  # noqa: E402

import opentimelineio.adapters as ot_adapters  # noqa: E402
from director_os.agent_runtime.agents import PlannerExecutor, ToolStep  # noqa: E402
from director_os.agent_runtime.checkpoint import CheckpointStore  # noqa: E402
from director_os.agent_runtime.critic import register_critic_tools  # noqa: E402
from director_os.agent_runtime.editing_tools import (  # noqa: E402
    DirectorToolbox,
    build_director_registry,
)
from director_os.agent_runtime.inspection_tools import (  # noqa: E402
    REG_TIMELINE,
    register_inspection_tools,
)
from director_os.agent_runtime.registry import ToolRegistry  # noqa: E402
from director_os.agent_runtime.session_memory import register_session_memory_tools  # noqa: E402
from director_os.agent_runtime.workshop_tools import register_workshop_tools  # noqa: E402
from director_os.core.db import Base, make_session_factory  # noqa: E402

RUN = Path(os.environ.get("AGENT_RUN_DIR", "/home/z/my-project/pipeline_run/agent_run_me"))
OUT = Path(os.environ.get("AGENT_OUT_DIR", "/home/z/my-project/download/eustis_agent_session_me"))
PIPE = REPO / "output" / "pipeline" / "5ad8f67d-3922-40ad-a354-8a1e8a6344ae"
PROJECT_ID = "5ad8f67d-3922-40ad-a354-8a1e8a6344ae"
OWNER_WINDOWS = [[607.7, 628.9]]  # the owner-flagged murmuring window, fresh-cut coords


class MeBackend:
    """LLMBackend port where the proposals are the operator's own decisions.

    Each round reads decisions_<N>.json (authored AFTER reading round_state_<N-1>)
    — the same JSON-steps protocol the Gemini/bridge backends speak, with the
    brain replaced by direct operator reasoning over measured evidence.
    """

    provider = "operator-llm"

    def __init__(self, round_no: int):
        self.round = round_no
        self.last_usage = {"tokens_in": 0, "tokens_out": 0}  # human brain: no tokens
        self.llm_log: list[dict] = [{
            "model": "operator (in-session reasoning)",
            "role": "director",
            "prompt_chars": 0,
            "response_chars": 0,
            "latency_s": 0.0,
        }]

    def record_results(self, steps: list[ToolStep], results: list[str]) -> None:
        pass  # results are persisted by the runner

    def reset_history(self) -> None:
        pass

    def propose(self, state_summary: str, role: str) -> list[ToolStep]:
        dfile = RUN / f"decisions_{self.round}.json"
        if not dfile.exists():
            return []
        data = json.loads(dfile.read_text())
        steps = []
        for st in (data.get("steps") or [])[:8]:
            if isinstance(st, dict) and st.get("tool"):
                steps.append(ToolStep(
                    tool=str(st["tool"]),
                    args=st.get("args") or {},
                    checkpoint_note=str(st.get("why", ""))[:300],
                ))
        return steps


def build_registry() -> ToolRegistry:
    work_tl = RUN / "working.otio"
    if not work_tl.exists():
        src = PIPE / "timeline.otio"
        RUN.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, work_tl)
        print(f"working copy created from shipped timeline ({src.stat().st_size} bytes)")
    timeline = ot_adapters.read_from_file(str(work_tl))
    REG_TIMELINE.clear()
    REG_TIMELINE.append(timeline)

    toolbox = DirectorToolbox(timeline, project_id="eustis_agent_me", resolution=(1280, 720))
    reg = build_director_registry(toolbox)
    register_inspection_tools(reg, RUN)
    register_critic_tools(reg, RUN)
    register_session_memory_tools(reg, RUN / "session")
    from sqlalchemy import create_engine

    engine = create_engine(f"sqlite:///{RUN / 'agent.db'}", future=True)
    Base.metadata.create_all(engine)
    store = CheckpointStore(make_session_factory(engine))
    register_workshop_tools(
        reg,
        work_dir=RUN,
        source_ass=PIPE / "subtitles.ass",
        evidence_path=RUN / "caption_evidence.json",
        vad_cache_path=RUN / "vad_spans_cache.json",
        store=store,
        project_id="eustis_agent_me",
    )
    return reg


def run_round(round_no: int) -> int:
    RUN.mkdir(parents=True, exist_ok=True)
    reg = build_registry()
    planner = PlannerExecutor(
        reg, _store(), project_id="eustis_agent_me", agent="operator_director"
    )
    dfile = RUN / f"decisions_{round_no}.json"
    if not dfile.exists():
        print(f"decisions_{round_no}.json missing — author it first")
        return 2
    timeline = REG_TIMELINE[0]
    t0 = time.monotonic()
    run = planner.run(MeBackend(round_no), timeline)
    timeline = REG_TIMELINE[0]  # undo.rollback rebinds the slot
    (RUN / "working.otio").write_text(ot_adapters.write_to_string(timeline) or "")

    results = []
    for s in run.steps:
        results.append({"tool": s.tool, "ok": s.ok, "detail": s.detail[:600]})
    state = {
        "round": round_no,
        "ok": run.ok,
        "rolled_back_to": run.rolled_back_to,
        "final_version": run.final_version,
        "steps": results,
        "wall_s": round(time.monotonic() - t0, 2),
    }
    (RUN / f"round_state_{round_no}.json").write_text(json.dumps(state, indent=1))
    print(f"round {round_no}: {[(s.tool, s.ok) for s in run.steps]} "
          f"ok={run.ok} rollback={run.rolled_back_to} ({state['wall_s']}s)")
    return 0 if run.ok else 1


def _store() -> CheckpointStore:
    from sqlalchemy import create_engine

    engine = create_engine(f"sqlite:///{RUN / 'agent.db'}", future=True)
    Base.metadata.create_all(engine)
    return CheckpointStore(make_session_factory(engine))


def report() -> int:
    reg = build_registry()
    audit = reg.dispatch("qa.self_audit", {})
    versions = _store().list_versions("eustis_agent_me")
    rounds = sorted(RUN.glob("round_state_*.json"))
    steps_ok, steps_total = 0, 0
    for r in rounds:
        st = json.loads(r.read_text())
        steps_total += len(st["steps"])
        steps_ok += sum(1 for s in st["steps"] if s["ok"])
    mem = {}
    mem_path = RUN / "session" / "session_state.json"
    if mem_path.exists():
        mem = json.loads(mem_path.read_text())
    report = {
        "provider": "operator-llm (the assistant IS the LLM this session; zero external calls)",
        "rounds": len(rounds),
        "steps_ok": steps_ok,
        "steps_total": steps_total,
        "critic_verdict": audit["verdict"],
        "critic_green": audit["green"],
        "critic_checks": audit["checks"],
        "durable_versions": len(versions),
        "owner_windows": OWNER_WINDOWS,
        "plan": mem.get("plan"),
        "notes": [n["text"] for n in (mem.get("notes") or [])][-6:],
        "wall_note": "decisions authored from measured evidence between rounds",
    }
    (RUN / "run_report.json").write_text(json.dumps(report, indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    for f in ("run_report.json", "working.otio", "captions_work.ass",
              "qa_caption_work.json", "caption_edits_applied.json",
              "timeline_saved.otio", "caption_evidence.json", "vad_spans_cache.json"):
        src = RUN / f
        if src.exists():
            shutil.copyfile(src, OUT / f)
    sess = RUN / "session" / "session_state.json"
    if sess.exists():
        shutil.copyfile(sess, OUT / "session_state.json")
    print(f"report: rounds={report['rounds']} steps={steps_ok}/{steps_total} "
          f"critic={report['critic_verdict']} versions={report['durable_versions']}")
    return 0 if report["critic_green"] else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "round":
        sys.exit(run_round(int(sys.argv[2])))
    elif cmd == "report":
        sys.exit(report())
    else:
        print(__doc__)
        sys.exit(2)
