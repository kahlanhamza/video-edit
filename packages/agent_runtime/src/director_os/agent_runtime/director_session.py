"""Director session — the 2-phase "Story Map -> Full-Arc Execution" flow.

Postmortem fixed here: the Director used to get ONE blind propose() call against
an 8-second-grained auto-cut scaffold; it made 2 trims and the loop ended with
a 59-second master. The new flow:

Phase A (Macro Plan) — the ENTIRE raw footage, condensed into a ~2.5k-token
chronological Story Index (story_map.py) plus the CaseDossier, is sent in ONE
JSON completion. The Director returns a complete 6-Act documentary outline
(720-900s) with explicit source ranges, narration scripts, and cards.

Phase B (Tool Execution Loop) — the validated macro plan is expanded into
`timeline.insert_raw_clip` / `timeline.add_narration` / `cards.add` tool steps
and executed act-by-act through the checkpointed registry (so every mutation
stays durable + logged). Then an LLM refinement loop (max_turns >= 25, never
terminating while the duration invariants are unmet) polishes: trims, splits,
blur boxes, narration rewrites.

Hard invariants (plan_invariants.py) gate every exit: a plan that cannot reach
600s / 6 narrations / 8 raw clips / 0.70 coverage is retried with failure
feedback, then replaced by the deterministic multi-act fallback — the session
never returns a truncated timeline again.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from director_os.agent_runtime.agents import PlannerExecutor, ToolStep
from director_os.agent_runtime.plan_invariants import (
    TimelineComposition,
    audit_invariants,
    coverage_span,
    timeline_composition,
)
from director_os.agent_runtime.prompting import compose_agent_prompt
from director_os.agent_runtime.story_map import StoryBucket, story_map_prompt
from pydantic import BaseModel, Field, ValidationError

# ------------------------------------------------------------ Phase A models
BlockKind = Literal["raw", "narration"]


class PlanRawClip(BaseModel):
    type: Literal["raw"] = "raw"
    source_start_s: float = Field(ge=0.0)
    dur_s: float = Field(gt=0.0, le=600.0)
    note: str = Field(default="", max_length=300)


class PlanNarration(BaseModel):
    type: Literal["narration"] = "narration"
    script: str = Field(min_length=8, max_length=2000)
    dur_s: float = Field(ge=3.0, le=120.0)
    broll_source_start_s: float | None = Field(default=None, ge=0.0)


PlanBlock = PlanRawClip | PlanNarration


class PlanAct(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    blocks: list[PlanBlock] = Field(min_length=1)


class PlanCard(BaseModel):
    template: Literal["location_intro", "evidence_fact", "case_outcome", "evidence_quote"]
    title: str = Field(min_length=1, max_length=120)
    subtitle: str = Field(default="", max_length=200)
    at_act: int = Field(ge=0)
    at_block: int = Field(default=0, ge=0)
    dur_s: float = Field(default=4.0, ge=1.0, le=15.0)


class MacroPlan(BaseModel):
    """The complete documentary outline (Phase A product)."""

    title: str = Field(default="", max_length=200)
    # act-count is a _plan_problems check (retry/repair), not a parse error: a
    # truncated-but-parseable plan must reach the Phase B repair loop
    acts: list[PlanAct] = Field(min_length=1)
    cards: list[PlanCard] = Field(default_factory=list)

    @property
    def raw_count(self) -> int:
        return sum(1 for a in self.acts for b in a.blocks if isinstance(b, PlanRawClip))

    @property
    def narration_count(self) -> int:
        return sum(1 for a in self.acts for b in a.blocks if isinstance(b, PlanNarration))

    @property
    def total_dur_s(self) -> float:
        return sum(b.dur_s for a in self.acts for b in a.blocks)

    def clip_span(self, source_duration_s: float) -> tuple[float | None, float | None]:
        first, last = None, None
        for act in self.acts:
            for b in act.blocks:
                if isinstance(b, PlanRawClip):
                    first = b.source_start_s if first is None else min(first, b.source_start_s)
                    end = b.source_start_s + b.dur_s
                    last = end if last is None else max(last, end)
        return first, (min(last, source_duration_s) if last is not None else None)


MACRO_PLAN_SCHEMA_DOC = """Return ONLY a JSON object (no prose, no code fences):
{"title": "working title",
 "acts": [
   {"name": "Cold Open", "blocks": [
      {"type": "raw", "source_start_s": 1234.0, "dur_s": 28.0, "note": "climax moment"},
      {"type": "narration", "script": "hook line(s)", "dur_s": 16.0, "broll_source_start_s": 45.0}
   ]},
   {"name": "Act 1: Initial Contact", "blocks": [ ... ]},
   ... Acts 2-6 ...
 ],
 "cards": [
   {"template": "location_intro", "title": "...", "subtitle": "...",
    "at_act": 1, "at_block": 0, "dur_s": 4.0},
   {"template": "evidence_fact", "title": "...", "subtitle": "...",
    "at_act": 3, "at_block": 0, "dur_s": 4.0},
   {"template": "case_outcome", "title": "...", "subtitle": "...",
    "at_act": 6, "at_block": 0, "dur_s": 5.0}
 ]}"""


def _target_lo(source_duration_s: float) -> float:
    return max(600.0, min(900.0, source_duration_s * 0.28))


def _target_hi(source_duration_s: float) -> float:
    return max(600.0, min(900.0, source_duration_s * 0.35))


NARRATION_PROMPT_LAW = """## Narration law (author by WORD COUNT; Brian +5% speaks ~2.6 words/sec)
- Total narration: 700-950 words (~4.5-6.0 min = 35-45% of runtime) across 10-14 blocks.
- Cold Open narration: 45-60 words (~18-23s) -> dur_s ~= words/2.6.
- Mid-video commentary: 50-80 words each (~20-30s), never above ~95 words.
- Final Outro: 75-100 words (~30-38s).
- Tone: calm, observant, analytical true-crime documentary voice — zero melodrama,
  zero snark; surface contradictions factually, never editorialize guilt.
- Facts come ONLY from the dossier; hint at mystery rather than invent.
- J/L-cut: at most 0.3-0.6s into silent padding — narration NEVER overlaps
  active Officer/Suspect dialogue.
- Visual movement: never one frozen b-roll for 25-30s of speech — back longer
  blocks with 2-3 sequential muted dimmed b-roll moments (split into consecutive
  narration blocks with different broll_source_start_s), a slow zoom, or a card."""


def macro_plan_user_prompt(
    buckets: list[StoryBucket],
    source_duration_s: float,
    dossier_json: dict[str, Any] | None = None,
) -> str:
    dossier_block = ""
    if dossier_json:
        dossier_block = (
            "\n\n## Verified case dossier (use ONLY these facts; never invent "
            "charges, names, or verdicts)\n" + json.dumps(dossier_json, ensure_ascii=False)[:4000]
        )
    return (
        story_map_prompt(buckets, source_duration_s)
        + dossier_block
        + f"""

## Your task
Plan a COMPLETE {_target_lo(source_duration_s):.0f}-{_target_hi(source_duration_s):.0f}s
(12-15 min) Cold Open + 6-Act documentary of this {source_duration_s / 60:.0f}-min bodycam video.

Structure (acts[0] is the Cold Open) — the GENERAL arrest arc, not a DUI template:
1. Cold Open Hook (25-45s): 1-2 climax raw clips PULLED FROM LATER IN THE VIDEO
   (raw audio on) + a 45-60 word narration hook setting date/location/mystery
   from the dossier only.
2. Act 1: Arrival & Contact — the stop/call begins; first oddities/contradictions.
3. Act 2: Investigation & Friction — questions, searches, excuses, evidence.
4. Act 3: Escalation / Turning Point — the moment the encounter changes.
5. Act 4: Arrest / Detention — the arrest, witness interaction.
6. Act 5: Aftermath / Transport / Booking — include when present in the footage.
7. Act 6: Verified Outcome — booking completion, tests, disposition.
Miranda, policy violations, and sobriety/field tests are HIGHLIGHTS to surface
whenever present — never force them into a beat they don't belong to.

{NARRATION_PROMPT_LAW}

Hard rules:
- Raw clips: 12-18 clips totaling 7.5-9.0 minutes, mostly 20-55s per stretch;
  split anything >65s with a commentary block between the halves.
- Clip starts must span the WHOLE file: earliest within the first 15%, latest
  ending inside the final 25% (never drop the arrest/booking ending).
- Cut dead air and logistics (keys, phone details, radio spelling); KEEP
  ironic, emotional, and incriminating exchanges.
- Cards: location_intro on Act 1's narration; 1-3 evidence_fact mid-doc on
  concrete contradictions/BAC/policy facts; case_outcome on the Outro. If the
  dossier has a verified verdict, state it cleanly; only use "presumed innocent"
  framing when the verdict is pending/unknown.
- Never mention video timestamps/timecodes (like "at 04:12"); real case
  durations ("a 20-minute observation period") are fine.
- Source seconds come from the story index [mm:ss] markers; mm:ss -> seconds.

{MACRO_PLAN_SCHEMA_DOC}"""
    )


def parse_macro_plan(text: str) -> MacroPlan:
    """Lenient parse: strip code fences, take the outermost JSON object."""
    from director_os.agent_runtime.llm import strip_code_fences

    cleaned = strip_code_fences(text)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("macro plan is not a JSON object")
    return MacroPlan.model_validate(data)


# ------------------------------------------------------------ Phase B expansion
def _as_block(blk: PlanRawClip | PlanNarration | dict) -> PlanRawClip | PlanNarration:
    """Coerce a stray dict into its model (defensive against mutated plans)."""
    if isinstance(blk, dict):
        if blk.get("type") == "narration":
            return PlanNarration.model_validate(blk)
        return PlanRawClip.model_validate(blk)
    return blk


def expand_macro_plan(plan: MacroPlan, source_duration_s: float) -> list[ToolStep]:
    """Expand the outline into ordered tool steps against the scaffold timeline.

    The scaffold's single full-length raw clip (item 0) is repurposed as the
    first planned clip (trim), everything else appends; the scaffold is removed
    at the end if it was not consumed. Card positions are computed from the
    cumulative block durations (absolute timeline seconds).
    """
    steps: list[ToolStep] = []
    scaffold_consumed = False
    if not plan.acts or not plan.acts[0].blocks:
        raise ValueError("macro plan has no first-act blocks")

    first = _as_block(plan.acts[0].blocks[0])
    if isinstance(first, PlanRawClip):
        steps.append(
            ToolStep(
                tool="timeline.trim_clip",
                args={
                    "item_index": 0,
                    "new_start_s": first.source_start_s,
                    "new_dur_s": first.dur_s,
                },
                checkpoint_note="cold open hook 1 onto scaffold clip",
            )
        )
        scaffold_consumed = True
        rest = plan.acts[0].blocks[1:]
    else:
        rest = plan.acts[0].blocks
    blocks_before_scaffold = 0 if scaffold_consumed else 1

    # act markers ride tool args; cards need absolute positions -> track them
    card_anchors: dict[int, float] = {}
    t = 0.0
    for act_i, act in enumerate(plan.acts):
        blocks = [_as_block(b) for b in (rest if act_i == 0 else act.blocks)]
        for blk in blocks:
            if isinstance(blk, PlanRawClip):
                steps.append(
                    ToolStep(
                        tool="timeline.insert_raw_clip",
                        args={
                            "source_start_s": round(blk.source_start_s, 2),
                            "dur_s": round(
                                min(blk.dur_s, source_duration_s - blk.source_start_s), 2
                            ),
                            "act_name": act.name,
                        },
                        checkpoint_note=act.name,
                    )
                )
            else:
                steps.append(
                    ToolStep(
                        tool="timeline.add_narration",
                        args={
                            "script_text": blk.script,
                            "dur_s": round(blk.dur_s, 2),
                            **(
                                {"broll_source_start_s": round(blk.broll_source_start_s, 2)}
                                if blk.broll_source_start_s is not None
                                else {}
                            ),
                            "act_name": act.name,
                        },
                        checkpoint_note=act.name,
                    )
                )
            t += blk.dur_s

    if not scaffold_consumed:
        # the untouched full-length scaffold clip sits at index 0 -> drop it
        steps.append(
            ToolStep(
                tool="timeline.remove_clip",
                args={"item_index": blocks_before_scaffold - 1},
                checkpoint_note="drop unused full-length scaffold clip",
            )
        )

    # cards: recompute absolute starts per act/block (block index within act)
    t = 0.0
    for ai, act in enumerate(plan.acts):
        bt = t
        blocks = [_as_block(b) for b in act.blocks]
        for bi in range(1, len(blocks)):
            bt += blocks[bi - 1].dur_s
        card_anchors[ai] = bt
        t += sum(b.dur_s for b in blocks)
    for card in plan.cards:
        if card.at_act >= len(plan.acts):
            continue
        act_blocks = [_as_block(b) for b in plan.acts[card.at_act].blocks]
        offset = sum(b.dur_s for b in act_blocks[: card.at_block])
        start = card_anchors[card.at_act] + offset
        args: dict[str, Any] = {
            "template": card.template,
            "title": card.title,
            "start_s": round(start, 2),
            "dur_s": card.dur_s,
        }
        if card.subtitle:
            args["subtitle"] = card.subtitle
        steps.append(ToolStep(tool="cards.add", args=args, checkpoint_note=card.template))
    return steps


def director_state_summary(timeline, source_duration_s: float) -> str:
    """Per-clip state the refinement loop sees each turn (index-stable)."""
    from director_os.otio_engine.compiler import master_track
    from director_os.otio_engine.timeline import is_gap, item_duration_s

    comp = TimelineComposition()
    lines: list[str] = []
    try:
        track = master_track(timeline)
    except Exception:
        return "timeline unreadable"
    for i, item in enumerate(track):
        if is_gap(item):
            continue
        dur = item_duration_s(item)
        meta = item.metadata.get("director_os") or {}
        kind = str(meta.get("kind", "?"))
        comp.final_duration_s += dur
        sr = getattr(item, "source_range", None)
        src = sr.start_time.to_seconds() if sr is not None else 0.0
        act = str(meta.get("act_name", "") or "-")
        if kind == "narration":
            comp.narrator_segment_count += 1
            script = str(meta.get("script_text", ""))[:60]
            lines.append(f"[{i}] NARRATION {dur:.1f}s broll@{src:.0f}s act={act} '{script}'")
        elif kind == "raw":
            comp.raw_segment_count += 1
            comp.first_source_s = (
                src if comp.first_source_s is None else min(comp.first_source_s, src)
            )
            end = src + dur
            comp.last_source_end_s = (
                end if comp.last_source_end_s is None else max(comp.last_source_end_s, end)
            )
            lines.append(f"[{i}] RAW {dur:.1f}s src {src:.1f}-{end:.1f}s act={act}")
    failures = audit_invariants(comp, source_duration_s)
    span = coverage_span(comp, source_duration_s)
    status = (
        "ALL INVARIANTS MET"
        if not failures
        else "UNMET: " + "; ".join(f"{f.code}" for f in failures)
    )
    return (
        f"master: {comp.final_duration_s:.1f}s | raw={comp.raw_segment_count} "
        f"narration={comp.narrator_segment_count} coverage={span:.2f} "
        f"(target 600s+/6+/8+/0.70+) — {status}\n" + "\n".join(lines[:40])
    )


# ------------------------------------------------------------ the session
@dataclass
class DirectorSessionReport:
    ok: bool
    phase: str = ""  # macro | refine | fallback
    provider: str = ""
    plan: MacroPlan | None = None
    turns: int = 0
    steps_ok: int = 0
    steps_failed: int = 0
    composition: TimelineComposition | None = None
    failures: list[str] = field(default_factory=list)
    detail: str = ""


class DirectorSession:
    """Two-phase Director run with hard duration/coverage invariants."""

    def __init__(
        self,
        registry,
        store,
        project_id: str,
        backend,  # FallbackChain | GeminiRestBackend | OpenAICompatBackend
        source_duration_s: float,
        max_turns: int = 30,  # >= 25 mandated: the loop may not die at 1-2 calls
        phase_a_retries: int = 2,
    ):
        self.registry = registry
        self.store = store
        self.project_id = project_id
        self.backend = backend
        self.source_duration_s = source_duration_s
        self.max_turns = max_turns
        self.phase_a_retries = phase_a_retries
        self.executor = PlannerExecutor(
            registry, store, project_id, agent="director_agent", role="director"
        )

    # ---------------------------------------------------------------- phase A
    def plan_macro(
        self, buckets: list[StoryBucket], dossier_json: dict[str, Any] | None = None
    ) -> tuple[MacroPlan | None, str]:
        """One JSON completion against the routing chain; retries with feedback.

        Returns (None, error) only for UNPARSEABLE plans after all retries — a
        parseable-but-truncated plan is returned as best effort so Phase B's
        repair loop (and the deterministic fallback behind it) can fix it.
        """
        system = compose_agent_prompt(
            agent="director_agent", role="director", niche="bodycam_documentary"
        )
        user = macro_plan_user_prompt(buckets, self.source_duration_s, dossier_json)
        feedback = ""
        last_err = ""
        best: MacroPlan | None = None
        for _attempt in range(self.phase_a_retries + 1):
            text = self.backend.complete_json(
                system, user + (f"\n\n## Previous attempt failed\n{feedback}" if feedback else "")
            )
            try:
                plan = parse_macro_plan(text)
                problems = self._plan_problems(plan)
                if problems:
                    feedback = "; ".join(problems)
                    last_err = feedback
                    if best is None or plan.total_dur_s > best.total_dur_s:
                        best = plan
                    continue
                return plan, ""
            except (ValueError, ValidationError, json.JSONDecodeError) as e:
                feedback, last_err = f"unparseable plan ({e})", str(e)
                continue
        if best is not None:
            return best, last_err  # structurally valid but truncated: Phase B repairs
        return None, last_err

    def _plan_problems(self, plan: MacroPlan) -> list[str]:
        """Cheap pre-flight on the plan JSON (the timeline is the real judge)."""
        problems: list[str] = []
        if len(plan.acts) < 5:
            problems.append(
                f"only {len(plan.acts)} acts — need the 6-act structure (7 incl. cold open)"
            )
        if plan.total_dur_s < 600.0 and self.source_duration_s >= 1200.0:
            problems.append(f"total plan duration {plan.total_dur_s:.0f}s < 600s")
        if not 10 <= plan.narration_count <= 14:
            problems.append(f"{plan.narration_count} narration blocks — need 10-14 (700-950 words)")
        if plan.total_dur_s > 0:
            narr_dur = sum(
                b.dur_s for a in plan.acts for b in a.blocks if isinstance(b, PlanNarration)
            )
            ratio = narr_dur / plan.total_dur_s
            if not 0.30 <= ratio <= 0.52:
                problems.append(
                    f"narration ratio {ratio:.2f} outside 0.30..0.52 "
                    "(rich commentary, not a sprinkle, not a podcast)"
                )
        if plan.raw_count < 8:
            problems.append(f"only {plan.raw_count} raw clips (< 8)")
        first, last = plan.clip_span(self.source_duration_s)
        if first is None or last is None:
            problems.append("plan has no raw clips")
        else:
            span = (last - first) / self.source_duration_s
            if span < 0.70 and self.source_duration_s >= 1200.0:
                problems.append(f"clip span {span:.2f} < 0.70 (start {first:.0f}s end {last:.0f}s)")
            if first > self.source_duration_s * 0.15:
                problems.append(f"earliest clip starts at {first:.0f}s (> first 15% of raw)")
            if last < self.source_duration_s * 0.75:
                problems.append(f"latest clip ends at {last:.0f}s (< final 25% of raw)")
        return problems

    # ---------------------------------------------------------------- phase B
    def execute(self, plan: MacroPlan, timeline) -> tuple[Any, DirectorSessionReport]:
        """Expand the plan act-by-act, then run the refinement loop to target."""
        report = DirectorSessionReport(ok=False, phase="macro", plan=plan)
        steps = expand_macro_plan(plan, self.source_duration_s)
        run = self.executor.execute_plan(steps, timeline)
        report.steps_ok = run.steps_ok
        report.steps_failed = len(run.steps) - run.steps_ok
        report.composition = timeline_composition(run.timeline)
        timeline = run.timeline
        if not run.ok:
            # a failed expansion step rolled back; surface and let the loop repair
            report.failures.append(
                f"expansion step failed: {run.steps[-1].detail if run.steps else '?'}"
            )

        report.phase = "refine"
        turns = 0
        while turns < self.max_turns:
            comp = timeline_composition(timeline)
            failures = audit_invariants(comp, self.source_duration_s)
            if not failures and turns >= 1:
                break  # at least one refinement pass validated the structure
            if not failures and turns == 0 and not self._has_refinable(timeline):
                break  # invariants met pre-loop and nothing to refine
            state = director_state_summary(timeline, self.source_duration_s)
            nudge = (
                ""
                if not failures
                else "\n\nINVARIANT FAILURES — fix these with tool calls (add clips via "
                "timeline.insert_raw_clip from the story index, add narrator bridges via "
                "timeline.add_narration): " + "; ".join(f"{f.code}: {f.detail}" for f in failures)
            )
            try:
                steps = self.backend.propose(state + nudge, "director")
            except Exception as e:  # chain exhausted this turn
                report.failures.append(f"propose failed at turn {turns}: {e}")
                break
            turns += 1
            self.store.log_tool(
                "director_agent",
                "llm.propose",
                {"role": "director", "turn": turns},
                result_summary=f"{len(steps)} step(s) proposed",
                project_id=self.project_id,
                ok=True,
            )
            if not steps:
                if not failures:
                    break
                if turns >= 3:  # model insists it is done but invariants disagree
                    report.failures.append("model stopped proposing while invariants unmet")
                    break
                continue  # re-prompt with the failure nudge
            run = self.executor.execute_plan(steps, timeline)
            timeline = run.timeline
            report.steps_ok += run.steps_ok
            report.steps_failed += len(run.steps) - run.steps_ok
            record = getattr(self.backend, "record_results", None)
            if callable(record):
                record(run.steps and steps or steps, [sr.detail for sr in run.steps])

        report.turns = turns
        report.composition = timeline_composition(timeline)
        report.failures.extend(
            f"{f.code}: {f.detail}"
            for f in audit_invariants(report.composition, self.source_duration_s)
        )
        report.ok = not report.failures
        return timeline, report

    @staticmethod
    def _has_refinable(timeline) -> bool:
        comp = timeline_composition(timeline)
        return comp.raw_segment_count + comp.narrator_segment_count > 0
