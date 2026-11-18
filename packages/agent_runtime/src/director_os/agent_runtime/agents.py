"""Planner-executor agent loop — native tool execution with durable checkpoints.

M2 runtime (PLAN §8): a Director agent proposes tool calls; every mutating step
is checkpointed BEFORE execution, dispatched through the typed registry, and
logged. A step that fails validation or semantics is auto-rolled-back to the
pre-step checkpoint, so a bad plan can never corrupt the durable state.

The `LLMBackend` port keeps this hermetic: `ScriptedBackend` replays fixed plans
in tests; a Pydantic AI / OpenAI-compatible backend plugs in behind the same
protocol once API keys exist (production wiring in M2 close-out).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from director_os.agent_runtime.checkpoint import CheckpointStore
from director_os.agent_runtime.registry import ToolRegistry
from pydantic import BaseModel, Field

AgentRole = Literal["perceiver", "director", "assembler", "critic"]


class ToolStep(BaseModel):
    """One proposed tool call in an agent plan.

    `provider_meta` carries provider-opaque fields (e.g. Gemini 3.x
    `thoughtSignature` / functionCall `id`) that cannot survive normalization
    on the generic fields but may be required to replay the conversation.
    """

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    checkpoint_note: str = ""
    provider_meta: dict[str, Any] | None = None


class LLMBackend(Protocol):
    """Proposes the next batch of tool steps from a state summary."""

    def propose(self, state_summary: str, role: AgentRole) -> list[ToolStep]: ...


class ScriptedBackend:
    """Deterministic backend: replays a fixed plan (tests + dry runs)."""

    def __init__(self, steps: list[ToolStep]):
        self._steps = list(steps)
        self.propose_calls = 0

    def propose(self, state_summary: str, role: AgentRole) -> list[ToolStep]:
        self.propose_calls += 1
        return list(self._steps)


@dataclass
class StepResult:
    tool: str
    ok: bool
    detail: str
    version_before: int | None = None


@dataclass
class AgentRun:
    agent: str
    role: AgentRole
    ok: bool
    steps: list[StepResult] = field(default_factory=list)
    final_version: int | None = None
    rolled_back_to: int | None = None
    timeline: Any = None  # always the authoritative post-run timeline object

    @property
    def steps_ok(self) -> int:
        return sum(1 for s in self.steps if s.ok)


class PlannerExecutor:
    """Runs a plan against the director toolbox with checkpoint-before-execute."""

    def __init__(
        self,
        registry: ToolRegistry,
        store: CheckpointStore,
        project_id: str,
        agent: str = "director_agent",
        role: AgentRole = "director",
    ):
        self.registry = registry
        self.store = store
        self.project_id = project_id
        self.agent = agent
        self.role = role

    def execute_plan(self, steps: list[ToolStep], timeline) -> AgentRun:
        """Execute steps against `timeline`; returns the run WITH the final timeline.

        On a failed step: log, roll the durable state back to the pre-step
        checkpoint, stop the plan, and return the restored timeline. History
        versions are never erased — rollback writes a new version on top.
        """
        run = AgentRun(agent=self.agent, role=self.role, ok=True, timeline=timeline)
        version = self.store.checkpoint(
            timeline, self.project_id, note=f"{self.agent}: plan start ({len(steps)} steps)"
        )
        run.final_version = version

        for step in steps:
            version_before = self.store.checkpoint(
                timeline,
                self.project_id,
                note=f"{self.agent}: before {step.tool} — {step.checkpoint_note}",
            )
            started = time.monotonic()
            try:
                result = self.registry.dispatch(step.tool, step.args)
            except Exception as e:  # validation OR semantic failure -> durable rollback
                detail = f"{type(e).__name__}: {e}"
                self.store.log_tool(
                    self.agent,
                    step.tool,
                    step.args,
                    duration_ms=int((time.monotonic() - started) * 1000),
                    project_id=self.project_id,
                    ok=False,
                    error=detail,
                )
                restored, new_version = self.store.rollback(self.project_id, version_before)
                run.ok = False
                run.steps.append(
                    StepResult(
                        tool=step.tool, ok=False, detail=detail, version_before=version_before
                    )
                )
                run.rolled_back_to = new_version
                run.final_version = new_version
                run.timeline = restored
                return run

            detail = str(result.get("detail", result)) if isinstance(result, dict) else str(result)
            self.store.log_tool(
                self.agent,
                step.tool,
                step.args,
                result_summary=detail,
                duration_ms=int((time.monotonic() - started) * 1000),
                project_id=self.project_id,
            )
            run.steps.append(
                StepResult(tool=step.tool, ok=True, detail=detail, version_before=version_before)
            )
            run.final_version = version_before

        # terminal checkpoint so the post-plan state is durable even if the
        # process dies before the next agent step
        run.final_version = self.store.checkpoint(
            run.timeline, self.project_id, note=f"{self.agent}: plan complete"
        )
        return run

    def run(self, backend: LLMBackend, timeline) -> AgentRun:
        """Full loop: propose from current state, execute, return the run report.

        If the backend exposes `last_usage` ({tokens_in, tokens_out}), the
        proposal itself is logged as a `llm.propose` row — the M5 cost
        dashboard reads token spend from `agent_tool_logs`.
        """
        started = time.monotonic()
        steps = backend.propose(self._state_summary(timeline), self.role)
        usage = getattr(backend, "last_usage", None)
        self.store.log_tool(
            self.agent,
            "llm.propose",
            {"role": self.role},
            result_summary=f"{len(steps)} step(s) proposed",
            duration_ms=int((time.monotonic() - started) * 1000),
            project_id=self.project_id,
            ok=True,
            tokens_in=usage.get("tokens_in") if isinstance(usage, dict) else None,
            tokens_out=usage.get("tokens_out") if isinstance(usage, dict) else None,
            token_cost_usd=getattr(backend, "last_cost_usd", None),
        )
        run = self.execute_plan(steps, timeline)
        # Multi-turn providers replay tool results into the next proposal
        # (Gemini 3.x pairs functionResponse with the replayed functionCall
        # turn + its thoughtSignature). Duck-typed: ScriptedBackend needs none.
        record = getattr(backend, "record_results", None)
        if callable(record):
            record(steps, [sr.detail for sr in run.steps])
        return run

    def _state_summary(self, timeline) -> str:
        from director_os.otio_engine.compiler import master_track

        try:
            track = master_track(timeline)
            items = [type(i).__name__ for i in track]
        except Exception:
            items = []
        return f"project={self.project_id} master_items={items}"
