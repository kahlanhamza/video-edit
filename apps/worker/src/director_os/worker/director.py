"""Stage 6.5 — Director autodrive: the LLM controls the edit, start to finish.

The deterministic v1 auto-cut (stage_planning) produces a uniform <=8s
segmentation of the raw window — correct, boring by construction. This stage
hands that timeline to the Director agent (native function-calling loop over
`DirectorToolbox`) together with a perception brief, and lets the model re-cut
it: drop dead air, tighten ins/outs, reorder for the hook, write narration,
place cards, extend privacy blurs.

Two invariants:

- EVERY tool step lands as a persisted timeline version through
  `CheckpointStore` (checkpoint-first, auto-rollback on a bad step), so a
  crash mid-autodrive resumes at the last good cut and every edit is
  auditable in `agent_tool_logs` — exactly the durability contract the
  interactive Studio API path already uses.
- Soft-fail by design: no key, provider outage, or a model that fails every
  round degrades to the deterministic v1 cut (`report["skipped"]`), never a
  broken pipeline. The LLM is a director, not a dependency.

Editorial contract comes from the injected niche skill (`compose_agent_prompt`
-> `bodycam_documentary`): hook inside 12s with raw audio, cut dead air with
speech/audio-event/motion intervals, narration rides the exact semantic moment
it describes, no timestamps in speech, bystander privacy, presumed innocence.
"""

from __future__ import annotations

import json
import os
from typing import Any

from director_os.agent_runtime.agents import (
    AgentRole,
    PlannerExecutor,
)
from director_os.agent_runtime.checkpoint import CheckpointStore
from director_os.agent_runtime.editing_tools import (
    DirectorToolbox,
    build_director_registry,
)
from director_os.agent_runtime.llm import LLMBackendError, build_backend
from director_os.core.db import PerceptionIndexRow, ProjectRow
from director_os.otio_engine.contract import parse_clip_meta
from director_os.otio_engine.timeline import (
    MASTER_TRACK,
    OVERLAY_TRACK,
    is_gap,
    item_duration_s,
    track_by_name,
    track_duration_s,
)

DEFAULT_MAX_ROUNDS = 4


# ----------------------------------------------------------------- perception brief
def _speech_beats(words: list[dict], max_chars: int = 150) -> list[str]:
    """Group role-tagged words into utterance beats the Director can cut on.

    A new beat starts on a speaker change or a >0.9s gap. Roles come from the
    proofread pass (Officer/Suspect/Narrator); raw ASR speaker labels pass
    through when proofreading never ran.
    """
    beats: list[str] = []
    cur_role: str | None = None
    cur_start = 0.0
    cur_end = 0.0
    cur_text: list[str] = []
    for w in words:
        role = str(w.get("role") or w.get("speaker") or "?")
        try:
            start = float(w.get("start_s") or 0.0)
            end = float(w.get("end_s") or 0.0)
        except (TypeError, ValueError):
            continue
        text = str(w.get("text") or "").strip()
        if not text:
            continue
        new_beat = cur_role is None or role != cur_role or (start - cur_end) > 0.9
        if new_beat and cur_text:
            joined = " ".join(cur_text)
            beats.append(f'{cur_start:.1f}-{cur_end:.1f}s {cur_role}: "{joined[:max_chars]}"')
            cur_text = []
        if new_beat:
            cur_role, cur_start = role, start
        cur_end = max(cur_end, end)
        cur_text.append(text)
    if cur_text:
        joined = " ".join(cur_text)
        beats.append(f'{cur_start:.1f}-{cur_end:.1f}s {cur_role}: "{joined[:max_chars]}"')
    return beats


def build_director_brief(
    sessions: Any,
    asset_id: str,
    transcript: dict,
    *,
    max_beats: int = 42,
    max_insights: int = 32,
    max_events: int = 28,
) -> str:
    """One compact, Director-readable digest of every perception product.

    Pure formatting over persisted products (transcript + perception_index);
    never talks to the network and never re-derives a paid pass.
    """
    lines: list[str] = []

    dur = float(transcript.get("duration_s") or 0.0)
    if dur > 0:
        lines.append(f"raw window: {dur:.1f}s of source footage")

    proof = transcript.get("proofread") or {}
    words = proof.get("words") or transcript.get("words") or []
    beats = _speech_beats(words)
    if beats:
        lines.append(f"speech beats ({len(beats)}; source seconds — the cutting grid):")
        shown = beats[:max_beats]
        lines += [f"  {b}" for b in shown]
        if len(beats) > max_beats:
            lines.append(f"  ... {len(beats) - max_beats} more beats elided")
    else:
        lines.append("NO SPEECH DETECTED — plan visuals-driven pacing; narration carries the story")

    with sessions() as s:
        row = s.get(PerceptionIndexRow, asset_id)
    shots: dict = (row.shots_json or {}) if row else {}
    red: dict = (row.redaction_tracks_json or {}) if row else {}

    vt = shots.get("visual_timeline") or {}
    insights = vt.get("insights") or []
    if insights:
        lines.append(f"VLM scene insights ({vt.get('model', '?')}; {len(insights)} findings):")
        for i in insights[:max_insights]:
            try:
                lines.append(
                    f"  t={float(i.get('timestamp_s', 0)):.1f}s [{i.get('kind', 'observation')}] "
                    f"{i.get('description', '')} (conf {float(i.get('confidence', 0)):.2f})"
                )
            except (TypeError, ValueError):
                continue
        if len(insights) > max_insights:
            lines.append(f"  ... {len(insights) - max_insights} more findings elided")

    sweep = shots.get("audio_events") or {}
    events = sweep.get("events") or []
    if events:
        lines.append(f"audio events ({sweep.get('backend', '?')}):")
        for e in events[:max_events]:
            lines.append(
                f"  {float(e.get('start_s', 0)):.1f}-{float(e.get('end_s', 0)):.1f}s "
                f"{e.get('kind', '?')} (peak {float(e.get('peak_score', 0)):.2f})"
            )
        if len(events) > max_events:
            lines.append(f"  ... {len(events) - max_events} more events elided")
        music = [e for e in events if "music" in str(e.get("kind", "")).lower()]
        if music:
            lines.append(
                f"COPYRIGHT GUARDRAIL: bystander music in {len(music)} span(s) — prefer "
                "narration over those windows; the music guard ducks them later"
            )

    fq = shots.get("frame_quality") or {}
    if fq.get("usable_ratio") is not None:
        spans = fq.get("unusable_spans") or []
        lines.append(
            f"frame quality: {float(fq['usable_ratio']):.0%} usable"
            + (f"; unusable: {spans[:6]}" if spans else "")
        )

    ocr = red.get("timestamp_ocr") or {}
    if ocr.get("iso"):
        lines.append(
            f"AXON overlay OCR: {ocr.get('device', 'BODYCAM')} {ocr['iso']}"
            + (f" location={ocr['location_text']}" if ocr.get("location_text") else "")
        )

    regions = red.get("regions") or []
    if regions:
        lines.append(
            f"privacy: {len(regions)} redaction region(s) already planned (Stage 2) — "
            "add blurs ONLY for threats outside those"
        )

    fod = shots.get("fod_qa") or {}
    for run in (fod.get("runs") or [])[:3]:
        for i in run.get("insights") or []:
            lines.append(
                f"tier-2 QA t={float(run.get('t_s', 0)):.1f}s [{i.get('kind', 'observation')}] "
                f"{i.get('description', '')}"
            )

    return "\n".join(lines)


# ----------------------------------------------------------------- state rendering
def _clip_line(index: int, item: Any) -> str:
    try:
        meta = parse_clip_meta(item)
    except Exception:  # noqa: BLE001 — a bare item still gets a row
        return f"  [{index}] UNKNOWN dur={item_duration_s(item):.2f}s"
    dur = item_duration_s(item)
    start = item.source_range.start_time.to_seconds() if item.source_range else 0.0
    if meta.kind == "raw":
        return (
            f"  [{index}] RAW src={start:.2f}s dur={dur:.2f}s "
            f"blur={len(meta.blur_regions)} zoom={len(meta.zoom_keyframes)} "
            f"audio={meta.audio_mode}"
        )
    if meta.kind == "narration":
        return (
            f"  [{index}] NARRATION src={start:.2f}s dur={dur:.2f}s "
            f"broll_vol={meta.broll_audio_volume:.2f} script={meta.script_text[:72]!r}"
        )
    return f"  [{index}] CARD {meta.template} dur={dur:.2f}s fields={json.dumps(meta.fields)[:90]}"


def master_state_lines(timeline) -> str:
    """Index-accurate rendering of the tracks the Director tools mutate.

    The indexes match `_master_clip`'s semantics (every item, gaps raise), so
    a function call the model emits is addressable exactly as listed.
    """
    lines: list[str] = []
    master = track_by_name(timeline, MASTER_TRACK)
    if master is not None:
        lines.append(f"master track ({track_duration_s(master):.1f}s total) — item indexes:")
        for i, item in enumerate(master):
            if is_gap(item):
                lines.append(f"  [{i}] GAP")
            else:
                lines.append(_clip_line(i, item))
    overlay = track_by_name(timeline, OVERLAY_TRACK)
    if overlay is not None:
        rows: list[str] = []
        t = 0.0
        for item in overlay:
            dur = item_duration_s(item)
            if not is_gap(item):
                try:
                    meta = parse_clip_meta(item)
                except Exception:  # noqa: BLE001
                    meta = None
                if meta is not None and meta.kind == "card":
                    rows.append(
                        f"  t={t:.1f}s CARD {meta.template} dur={dur:.1f}s "
                        f"fields={json.dumps(meta.fields)[:70]}"
                    )
                else:
                    rows.append(f"  t={t:.1f}s CARD dur={dur:.1f}s")
            t += dur
        if rows:
            lines.append("overlay track (cards over the master):")
            lines += rows
    return "\n".join(lines)


DIRECTOR_GOAL = """EDIT MISSION — you are the Director; the v1 cut above is a uniform \
<=8s segmentation of the raw window (correct, boring by construction). Re-cut it into a \
tight documentary:
1. HOOK: the most dramatic raw-audio moment must land within the first 12s of the FINAL \
cut — move_clip/trim_clip the opener to start inside the action.
2. KEEP/DROP: remove_clip dead air (use speech beats + audio events + VLM insights; keep \
40-60% of the raw window). Never drop a beat that carries the story.
3. TIGHTEN: trim_clip kept clips to the exact action (+/-0.6s pad); split_clip the slow \
middle of a long clip and remove_clip the slack; no kept clip over 8s.
4. NARRATE: narration.add on 3-6 kept clips whose raw audio is NOT key dialogue — veteran \
true-crime voice, present tense, contractions, <=2 sentences, never state guilt, never \
speak timestamps/seconds. Refine wording with narration.set_text.
5. CARDS: cards.add location_intro at t=0 with agency|location from the AXON OCR line when \
available; evidence_fact/evidence_quote only for facts visible in the brief; never let a \
card overlap the hook.
6. PRIVACY: redaction.add_blur only for NEW threats outside the Stage-2 plan (canvas px).
Execution rules: item_index refers to the master items EXACTLY as listed; steps run in \
order (<=12 per proposal) and later indexes shift after remove/move; one bad step rolls \
the whole proposal back — prefer fewer safe edits per round. When the cut is final, \
respond with NO function calls."""


class _BriefedDirector(PlannerExecutor):
    """PlannerExecutor whose state summary is the full Director console.

    Keeps the checkpoint-first execution semantics untouched; only the
    perception the model sees changes.
    """

    def __init__(
        self,
        registry: Any,
        store: CheckpointStore,
        project_id: str,
        brief: str,
        agent: str = "director_agent",
        role: AgentRole = "director",
    ):
        super().__init__(registry, store, project_id, agent=agent, role=role)
        self._brief = brief

    def _state_summary(self, timeline) -> str:  # noqa: D102 — see base class
        return (
            f"project={self.project_id}\n"
            + master_state_lines(timeline)
            + "\n\nPERCEPTION BRIEF\n"
            + self._brief
            + "\n\n"
            + DIRECTOR_GOAL
        )


# ----------------------------------------------------------------- the stage
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def stage_director(
    timeline: Any,
    asset_id: str,
    transcript: dict,
    project_id: str,
    deps: Any,
) -> tuple[Any, int, dict]:
    """LLM Director autodrive over the v1 cut. Returns (timeline, version, report).

    Rounds: propose -> checkpoint-first execute -> feed tool results back into
    the conversation (Gemini multi-turn) -> repeat until the model answers
    with no function calls, a round fails, or `director_max_rounds` is hit.
    The active-version pointer only moves when the model actually edited.
    """
    with deps.sessions() as s:
        project = s.get(ProjectRow, project_id)
        base_version = int(project.active_otio_version_id or 0) if project else 0

    if not getattr(deps, "llm_direct", False):
        return timeline, base_version, {"skipped": "llm_direct disabled"}

    store = CheckpointStore(deps.sessions)
    toolbox = DirectorToolbox(timeline, project_id, resolution=(deps.video.w, deps.video.h))
    registry = build_director_registry(toolbox)
    try:
        backend = build_backend(registry)
    except LLMBackendError as e:
        deps.log(f"  [director] LLM not available ({e}) — deterministic v1 cut rides on")
        return timeline, base_version, {"skipped": str(e)}

    brief = build_director_brief(deps.sessions, asset_id, transcript)
    executor = _BriefedDirector(registry, store, project_id, brief)
    max_rounds = deps.director_max_rounds or _env_int("DIRECTOR_MAX_ROUNDS", DEFAULT_MAX_ROUNDS)

    report: dict[str, Any] = {
        "provider": str(getattr(backend, "provider", "unknown")),
        "model": str(getattr(backend, "model", "")),
        "rounds": 0,
        "steps_ok": 0,
        "steps_failed": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "final_version": None,
        "skipped": None,
    }
    for rnd in range(1, max_rounds + 1):
        report["rounds"] = rnd
        run = executor.run(backend, timeline)
        report["steps_ok"] += sum(1 for st in run.steps if st.ok)
        report["steps_failed"] += sum(1 for st in run.steps if not st.ok)
        usage = getattr(backend, "last_usage", None)
        if isinstance(usage, dict):
            report["tokens_in"] += int(usage.get("tokens_in") or 0)
            report["tokens_out"] += int(usage.get("tokens_out") or 0)
        timeline = run.timeline  # mutated in place, or restored on rollback
        if run.ok and run.steps and run.final_version:
            report["final_version"] = run.final_version
        if not run.steps:
            deps.log(f"  [director] round {rnd}: model proposes no further edits — cut locked")
            break
        if not run.ok:
            deps.log(
                f"  [director] round {rnd}: step failed -> rolled back to "
                f"v{run.rolled_back_to}; next round sees the failure"
            )

    final_version = report["final_version"] or base_version
    if final_version != base_version:
        with deps.sessions() as s:
            project = s.get(ProjectRow, project_id)
            if project is not None:
                project.active_otio_version_id = final_version
                s.commit()
        deps.log(f"  [director] active timeline -> v{final_version}")
    report["final_version"] = final_version
    deps.log(
        f"  [director] {report['provider']}/{report['model']}: {report['rounds']} round(s), "
        f"{report['steps_ok']} edit(s) ok, {report['steps_failed']} failed, "
        f"{report['tokens_in']}+{report['tokens_out']} tok"
    )
    return timeline, final_version, report
