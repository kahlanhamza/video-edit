"""The dailies review pass — the master is LOOKED at, not trusted.

The missing half of the agentic loop (notebook lessons, 2026-11-24): the
operator's manual fix-up loop on the real harvests — render, sample frames
across the master, WATCH them, fix wrong facts / missed beats / misordered
acts by hand, re-render — is now a bounded agent pass the pipeline runs
after QA:

  1. DETERMINISTIC reviewer (always): structural evidence a model cannot
     fake — kept-source chronology across the program body (the notebook's
     hand-fixed arrest-before-witness law) + the kept-span program map.
  2. MODEL critic (when an eye is configured): a ReAct loop over the
     rendered master — the critic may REQUEST exact frames anywhere on the
     timeline ("inspect" rounds, on demand, bounded) before it issues its
     ONE strict JSON verdict: issues + bounded repairs. The editor applies
     them; the re-render returns with NEW frames; the critic judges again.
     researched basis: EditDuet (SIGGRAPH 2025) editor/critic duet, Agentic
     Video Intelligence (arXiv 2511.14446) retrieve-perceive-review,
     VideoAgent (ECCV 2024) iterative VLM tool rounds.
  3. REPAIRS go through the SAME typed DirectorToolbox registry the
     Director plans with (no private edit path), each applied or refused
     loudly, capped (MAX_REPAIRS), each seeing the timeline AFTER the
     previous repair (a sequential edit list, like the op log).

v1 repair surface is deliberately small and visual-free of coordinate law
traps: ``trim_raw`` (timeline.trim_clip), ``remove_clip``
(timeline.remove_clip), ``move_clip`` (timeline.move_clip). Narration
rewording is REFUSED in v1 (re-TTS is a pipeline decision, not a review
one) — the refusal is loud, never silent. Blur/zoom spatial repairs are
refused: coordinates law forbids eyeball mappings from a sampled frame.

No backend configured -> the deterministic evidence is still produced and
the outcome is ``unavailable`` — honest, never a silent pass. The pipeline
re-renders only when repairs were applied (segment cache: misses only).
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, ValidationError

MAX_REPAIRS = 12  # a review round is a polish pass, not a re-plan
CHRONO_TOLERANCE_S = 1.0  # hairline overlaps are not chronology crimes
REVIEW_FRAME_COUNT = 12
MAX_INSPECT_ROUNDS = 4  # ReAct ceiling: the critic may LOOK before it judges
INSPECT_BATCH_CAP = 8  # frames per inspect round (tokens law)

REVIEW_SYSTEM = """You are the dailies reviewer of an autonomous documentary editor.
You are shown frames sampled evenly across the RENDERED MASTER plus structural
evidence. Judge the WATCHING experience, not the plan: dead air, frozen or
ugly holds, burned-in artifacts, missing visual payoff, misordered story
beats, captions or graphics problems visible in frame.

You work in rounds. Before you judge, you may LOOK CLOSER — request exact
frames from anywhere on the master (act boundaries, suspect dead air, the
scene behind a suspicious anchor):
{"action": "inspect", "times": [<master seconds, max 8>], "why": "<one line>"}
The frames arrive as new attachments; your conversation grows each round.
When you have seen enough, produce the verdict:

Repairs (indices are 0-based positions on the master track, each repair
sees the timeline AFTER the previous one — order them accordingly):
- {"kind": "trim_raw", "item_index": N, "new_start_s": A, "new_dur_s": D}
    moves a RAW clip's source window to [A, A+D] (source seconds of the
    original footage). Use to cut dead air / skip to the visual payoff.
- {"kind": "remove_clip", "item_index": N}
    deletes a master clip entirely (black, frozen, unsafe, or dead footage).
- {"kind": "move_clip", "item_index": N, "move_to": M}
    reorders a clip to master position M (fix story order).

You may NOT rewrite narration (out of scope), add blur/zoom/freezes
(coordinates law), or add clips. Reply with ONLY a JSON object — either the
inspect action above, or:
{"verdict": "ok"|"needs_repairs",
 "issues": ["<one-line issue, frame_# referenced>", ...],
 "repairs": [<repair objects, max 12>]}
No prose outside the JSON."""


# ------------------------------------------------------------------ models
class ReviewRepair(BaseModel):
    """One bounded repair the reviewer proposed; validated before dispatch."""

    kind: Literal["trim_raw", "remove_clip", "move_clip"]
    item_index: int = Field(ge=0)
    new_start_s: float | None = Field(default=None, ge=0.0)
    new_dur_s: float | None = Field(default=None, gt=0.0)
    move_to: int | None = Field(default=None, ge=0)
    reason: str = Field(default="", max_length=500)


class ReviewVerdict(BaseModel):
    verdict: Literal["ok", "needs_repairs"] = "ok"
    issues: list[str] = Field(default_factory=list)
    repairs: list[ReviewRepair] = Field(default_factory=list)


class ReviewVerdictError(RuntimeError):
    pass


class ReviewBackend(Protocol):
    """Port: frames + structural evidence -> a strict JSON verdict."""

    def review(self, frames: list[tuple[float, Any]], evidence: str) -> ReviewVerdict:
        """`frames` is (timestamp_s, image_path) pairs; raises ReviewVerdictError
        on provider failure. Implementations must stay hermetic-injectable."""
        ...


def parse_review_verdict(text: str) -> ReviewVerdict | None:
    """First balanced JSON object in the text -> ReviewVerdict; None on miss.

    Repairs fields the model had no business proposing are dropped before
    validation only for `reason`/extra keys; unknown repair KINDS fail the
    whole parse loudly (a half-applied repair list the caller can act on is
    worse than a refused round).
    """
    from director_os.agent_runtime.chat_v2 import _extract_json

    data = _extract_json(text or "")
    if data is None:
        return None
    raw_repairs = data.get("repairs") or []
    kept = [
        {
            "kind": r.get("kind"),
            "item_index": r.get("item_index"),
            "new_start_s": r.get("new_start_s"),
            "new_dur_s": r.get("new_dur_s"),
            "move_to": r.get("move_to"),
            "reason": str(r.get("reason", ""))[:500],
        }
        for r in raw_repairs
        if isinstance(r, dict) and r.get("kind") in {"trim_raw", "remove_clip", "move_clip"}
    ]
    dropped = len(raw_repairs) - len(kept)
    verdict_raw = str(data.get("verdict", "")).strip().lower()
    verdict: Literal["ok", "needs_repairs"] = (
        "needs_repairs" if verdict_raw == "needs_repairs" or kept else "ok"
    )
    issues = [str(i)[:400] for i in (data.get("issues") or []) if str(i).strip()]
    try:
        out = ReviewVerdict(verdict=verdict, issues=issues, repairs=kept)  # type: ignore[arg-type]
    except ValidationError:
        return None
    if dropped:
        out.issues.append(
            f"review parse: {dropped} proposed repair(s) with out-of-scope kind(s) dropped"
        )
    return out


# --------------------------------------------------- deterministic reviewer
def _master_raw_windows(timeline) -> list[tuple[int, str, float, float, str | None]]:
    """(master_index, name, src_start_s, src_end_s, act) for every kept RAW clip."""
    from director_os.otio_engine.timeline import MASTER_TRACK, is_gap, track_by_name

    windows: list[tuple[int, str, float, float, str | None]] = []
    track = track_by_name(timeline, MASTER_TRACK)
    if track is None:
        return windows
    for idx, item in enumerate(track):
        if is_gap(item):
            continue
        meta = (getattr(item, "metadata", {}) or {}).get("director_os") or {}
        if meta.get("kind") != "raw":
            continue
        sr = getattr(item, "source_range", None)
        if sr is None:
            continue
        start = sr.start_time.to_seconds()
        end = start + sr.duration.to_seconds()
        windows.append(
            (idx, str(getattr(item, "name", "") or f"clip_{idx}"), start, end, meta.get("act_name"))
        )
    return windows


def deterministic_review(timeline) -> list[str]:
    """Structural evidence the model cannot fake. NEVER repairs by itself.

    Chronology law (the notebook's hand-fixed law): the program BODY must
    walk the source forward — only the cold open (the first kept clip, the
    hook/sting exception) may come from late in the source. Runtime cost is
    O(kept clips); swaps and duplicated-beat mistakes the Phase A planner
    made are caught here as concrete issues with clip indices the model can
    repair with move_clip/remove_clip.
    """
    issues: list[str] = []
    windows = _master_raw_windows(timeline)
    if not windows:
        issues.append("no raw clips on the master track — the program has no footage")
        return issues
    # The cold open is a sanctioned flash-forward; the CHRONOLOGICAL BODY
    # starts at the second kept clip and must walk the source forward among
    # ITSELF (body-relative — the cold open's far position never pollutes it).
    body = windows[1:]
    furthest: float | None = None
    for idx, name, s0, s1, act in body:
        if furthest is not None and s0 < furthest - CHRONO_TOLERANCE_S:
            issues.append(
                f"chronology: clip {idx} ({name}, act={act}) pulls source "
                f"{s0:.1f}-{s1:.1f}s BEFORE a body clip already past {furthest:.1f}s "
                "— story walks backwards"
            )
        furthest = s1 if furthest is None else max(furthest, s1)
    total_kept = sum(e - s for _, _, s, e, _ in windows)
    issues.append(
        f"program structure: {len(windows)} raw clips, kept footage {total_kept:.1f}s, "
        f"deepest body reach {windows[-1][3]:.1f}s"
    )
    return issues


# ------------------------------------------------------- repair application
def _repair_args(r: ReviewRepair) -> tuple[str, dict[str, Any]] | None:
    if r.kind == "trim_raw":
        if r.new_start_s is None or r.new_dur_s is None:
            return None
        return "timeline.trim_clip", {
            "item_index": r.item_index,
            "new_start_s": r.new_start_s,
            "new_dur_s": r.new_dur_s,
        }
    if r.kind == "remove_clip":
        return "timeline.remove_clip", {"item_index": r.item_index}
    if r.kind == "move_clip":
        if r.move_to is None:
            return None
        return "timeline.move_clip", {"from_index": r.item_index, "to_index": r.move_to}
    return None


def apply_repairs(
    timeline,
    project_id: str,
    repairs: list[ReviewRepair],
    *,
    resolution: tuple[int, int] = (1920, 1080),
    log: Any = None,
) -> tuple[list[str], list[str]]:
    """Dispatch capped repairs through the SAME DirectorToolbox registry.

    Returns (applied_details, refused_details) — every refusal names WHY.
    A failing registry never raises out of here (the pipeline must not die
    on a polish pass); unknown-tool/contract errors mark the whole round
    refused and stop (fail-closed).
    """
    from director_os.agent_runtime.editing_tools import (
        DirectorToolbox,
        build_director_registry,
    )
    from director_os.agent_runtime.registry import ToolRegistry

    def _log(msg: str) -> None:
        if log is not None:
            log(msg)

    applied: list[str] = []
    refused: list[str] = []
    if not repairs:
        return applied, refused
    registry = ToolRegistry()
    try:
        registry = build_director_registry(DirectorToolbox(timeline, project_id, resolution))
    except Exception as e:  # noqa: BLE001 — a toolbox the registry cannot wrap is refused wholesale
        return [], [f"repair registry unavailable, all {len(repairs)} refused ({e})"]

    for r in repairs[:MAX_REPAIRS]:
        bound = _repair_args(r)
        if bound is None:
            refused.append(f"{r.kind} #{r.item_index}: missing required window/position args")
            continue
        tool, args = bound
        try:
            result = registry.dispatch(tool, args)
            detail = str((result or {}).get("detail", "")) if isinstance(result, dict) else ""
            applied.append(f"{tool} #{r.item_index}: {r.reason or detail}")
        except Exception as e:  # noqa: BLE001 — loud refusal per repair
            refused.append(f"{tool} #{r.item_index}: {e}")
    for r in repairs[MAX_REPAIRS:]:
        refused.append(
            f"{r.kind} #{r.item_index}: beyond the {MAX_REPAIRS}-repair review cap — refused"
        )
    _log(f"  [review] repairs applied={len(applied)} refused={len(refused)}")
    return applied, refused


# -------------------------------------------------------------- VLM adapter
def vision_review_backend(client: Any) -> ReviewBackend | None:
    """Wrap a VisionClient into a ReviewBackend — None when it is a FAKE.

    `build_vision_client` degrades to FakeVisionClient when no key exists;
    a fake's verdict would be a silent lie about real footage (the
    never-a-silent-swap law), so the fake reads as REVIEW UNAVAILABLE.
    """
    if getattr(client, "provider", "fake") == "fake":
        return None

    class _V:
        provider = getattr(client, "provider", "vision")

        def review(self, frames: list[tuple[float, Any]], evidence: str) -> ReviewVerdict:
            from director_os.perception.vision import (
                FrameRef,
                VisionError,
            )

            refs = [FrameRef(timestamp_s=t, image_path=p) for t, p in frames]
            question = f"{REVIEW_SYSTEM}\n\n## Structural evidence\n{evidence}"
            try:
                insights = client.analyze(question, refs)
            except VisionError as e:
                raise ReviewVerdictError(str(e)) from e
            texts = [i.description for i in insights]
            verdict = parse_review_verdict("\n".join(texts))
            if verdict is not None:
                return verdict
            # honest fallback: the insights ARE the issues, repairs stay empty
            if texts:
                return ReviewVerdict(verdict="ok", issues=texts[:12], repairs=[])
            raise ReviewVerdictError("vision review returned nothing usable")

    return _V()


def text_review_backend(backend: Any) -> ReviewBackend | None:
    """Wrap a complete_json-capable LLM backend (the routing chain) as a
    ReviewBackend — the no-frames evidence channel: the model still reads
    the program map, QA failures and chronology issues and may propose
    structure repairs (move/trim/remove), it simply cannot SEE frames."""
    if backend is None or not callable(getattr(backend, "complete_json", None)):
        return None

    class _T:
        provider = getattr(backend, "provider", "llm")

        def review(self, frames: list[tuple[float, Any]], evidence: str) -> ReviewVerdict:
            from director_os.agent_runtime.llm import LLMBackendError

            try:
                text = backend.complete_json(REVIEW_SYSTEM, evidence, max_tokens=2048)
            except LLMBackendError as e:
                raise ReviewVerdictError(str(e)) from e
            verdict = parse_review_verdict(text)
            if verdict is None:
                raise ReviewVerdictError(f"unparseable review JSON: {str(text)[:160]!r}")
            return verdict

    return _T()


# ------------------------------------------------- the critic loop (ReAct)
class CriticTurnError(RuntimeError):
    pass


def parse_critic_round(text: str) -> tuple[str, dict[str, Any]]:
    """Classify one critic response: ("inspect", {...}) | ("verdict", {...}) |
    ("garbage", {})."""
    from director_os.agent_runtime.chat_v2 import _extract_json

    data = _extract_json(text or "")
    if isinstance(data, dict) and data.get("action") == "inspect":
        return "inspect", data
    if isinstance(data, dict):
        return "verdict", data
    return "garbage", {}


class CriticTurnAdapter(Protocol):
    """Port: one agentic conversation turn. Vision adapters attach frames;
    text adapters iterate evidence-only. `provider` feeds the audit log."""

    provider: str

    def ask(self, prompt: str, frames: list[tuple[float, Any]]) -> str: ...


def vision_critic_adapter(client: Any) -> CriticTurnAdapter | None:
    """A VisionClient as the critic's eyes. A FAKE client (no key) reads as
    None — a fake's sight would be a silent lie about real footage."""
    if getattr(client, "provider", "fake") == "fake":
        return None

    class _V:
        provider = getattr(client, "provider", "vision")

        def ask(self, prompt: str, frames: list[tuple[float, Any]]) -> str:
            from director_os.perception.vision import FrameRef, VisionError

            refs = [FrameRef(timestamp_s=t, image_path=p) for t, p in frames]
            # raw_analysis first: the critic needs the model's ACTION JSON
            # verbatim — analyze()'s insight-array decode would shred it
            # (live E2E: two garbage rounds, failout before any verdict)
            raw = getattr(client, "raw_analysis", None)
            if callable(raw):
                try:
                    return str(raw(prompt, refs))
                except VisionError as e:
                    raise CriticTurnError(str(e)) from e
                except Exception as e:  # noqa: BLE001 — structural errors surface LOUD too
                    raise CriticTurnError(f"vision adapter failed: {e}") from e
            try:
                insights = client.analyze(prompt, refs)
                return "\n".join(i.description for i in insights)
            except VisionError as e:
                raise CriticTurnError(str(e)) from e

    return _V()


def text_critic_adapter(backend: Any) -> CriticTurnAdapter | None:
    """A complete_json backend iterating on the evidence text — inspection
    rounds return the DETERMINISTIC read of what it asked for (no pixels)."""
    if backend is None or not callable(getattr(backend, "complete_json", None)):
        return None

    class _T:
        provider = getattr(backend, "provider", "llm")

        def __init__(self) -> None:
            self._backend = backend

        def ask(self, prompt: str, frames: list[tuple[float, Any]]) -> str:
            from director_os.agent_runtime.llm import LLMBackendError

            try:
                return self._backend.complete_json(REVIEW_SYSTEM, prompt, max_tokens=2048)
            except LLMBackendError as e:
                raise CriticTurnError(str(e)) from e

    return _T()


def run_critic(
    adapter: CriticTurnAdapter,
    *,
    base_prompt: str,
    initial_frames: list[tuple[float, Any]],
    grab_frames: Any,
    duration_s: float,
    log: Any = None,
) -> tuple[ReviewVerdict, list[dict[str, Any]]]:
    """The ReAct critic: LOOK (inspect) -> REASON -> (repeat) -> VERDICT.

    `grab_frames([t, ...]) -> [(t, path), ...]` is the worker's on-demand
    eye (worker-local cache dirs); `initial_frames` is the even sampling.
    Garbage rounds and the inspect ceiling fail LOUD (CriticTurnError) —
    the caller degrades to `unavailable`, never to a silent guess.
    """

    def _log(msg: str) -> None:
        if log is not None:
            log(msg)

    frames = list(initial_frames)
    prompt = base_prompt
    trace: list[dict[str, Any]] = []
    garbage_streak = 0
    garbage_detail = ""
    for rnd in range(1, MAX_INSPECT_ROUNDS + 2):
        try:
            text = adapter.ask(prompt, frames)
        except CriticTurnError:
            raise
        action, data = parse_critic_round(text)
        if action == "verdict":
            verdict = parse_review_verdict(text)
            if verdict is None:
                garbage_streak += 1
                garbage_detail = f"unparseable verdict: {str(text)[:160]!r}"
                trace.append({"round": rnd, "action": "garbage", "detail": garbage_detail})
                _log(f"  [review] critic round {rnd} unusable ({garbage_detail})")
            else:
                trace.append({"round": rnd, "action": "verdict", "detail": verdict.verdict})
                _log(f"  [review] critic verdict round {rnd}: {verdict.verdict}")
                return verdict, trace
        elif action == "inspect" and rnd <= MAX_INSPECT_ROUNDS:
            times_raw = data.get("times") or []
            times = sorted(
                {
                    round(min(max(float(t), 0.0), max(duration_s, 0.0)), 2)
                    for t in times_raw
                    if isinstance(t, (int, float))
                }
            )[:INSPECT_BATCH_CAP]
            if not times:
                garbage_streak += 1
                garbage_detail = "empty inspect times"
                trace.append({"round": rnd, "action": "garbage", "detail": garbage_detail})
                _log(f"  [review] critic round {rnd} unusable ({garbage_detail})")
            else:
                new = grab_frames(times)
                have = {round(t, 2) for t, _ in frames}
                frames += [(t, p) for t, p in new if round(t, 2) not in have]
                why = str(data.get("why", ""))[:200]
                manifest = ", ".join(f"{t:.2f}s" for t, _ in new) or "none"
                prompt = (
                    f"{prompt}\n\n[Round {rnd}] Frames NOW ATTACHED at: {manifest}."
                    + (
                        f' Request reason: "{why}".'
                        if why
                        else " Respond with your next action or verdict."
                    )
                    + f"\nTotal inspected so far: {len(frames)} frame(s)."
                )
                garbage_streak = 0
                trace.append(
                    {
                        "round": rnd,
                        "action": "inspect",
                        "detail": f"{len(new)} frame(s): {manifest} — {why}",
                    }
                )
                _log(f"  [review] critic inspected {len(new)} frame(s): {manifest}")
        else:
            garbage_streak += 1
            detail = (
                "inspect beyond the ceiling"
                if action == "inspect"
                else f"no JSON object in reply: {str(text)[:160]!r}"
            )
            garbage_detail = detail
            trace.append({"round": rnd, "action": "garbage", "detail": detail})
            _log(f"  [review] critic round {rnd} unusable ({detail})")
        if garbage_streak >= 2:
            raise CriticTurnError(
                "critic produced 2 consecutive unusable rounds — refusing to guess"
                + (f": {garbage_detail}" if garbage_detail else "")
            )
    raise CriticTurnError(f"critic never reached a verdict in {MAX_INSPECT_ROUNDS + 1} rounds")
