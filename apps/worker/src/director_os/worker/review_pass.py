"""Master review stage — post-render dailies review (the closed agent loop).

Sampling law: frames come from the SHIPPED artifact (`final_for_qa`), the
file the human would publish — with captions, blur and augmentation burned
in. Reviewing the pre-fx master is reviewing a different program.

Failure contract: the review pass NEVER kills the pipeline. A missing
viewer (`unavailable`), a failing VLM or unparseable verdict is logged loud
and the run continues to the human gate with the review's deterministic
evidence attached. Only APPLIED repairs change the program — and the caller
re-renders (cache: misses only) + re-audits.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal

from director_os.agent_runtime.review import (
    REVIEW_FRAME_COUNT,
    REVIEW_SYSTEM,
    CriticTurnAdapter,
    CriticTurnError,
    ReviewBackend,
    apply_repairs,
    deterministic_review,
    run_critic,
    text_critic_adapter,
    text_review_backend,
    vision_critic_adapter,
    vision_review_backend,
)
from pydantic import BaseModel, Field


class ReviewOutcome(BaseModel):
    status: Literal["clean", "applied", "unavailable", "off"] = "clean"
    issues: list[str] = Field(default_factory=list)
    applied: list[str] = Field(default_factory=list)
    refused: list[str] = Field(default_factory=list)
    summary: str = ""
    frames: list[float] = Field(default_factory=list)  # sampled master timestamps (s)
    inspect_rounds: int = 0  # ReAct rounds the critic spent LOOKING
    inspected: list[float] = Field(default_factory=list)  # every frame shown


def sample_master_frames(
    master_path: Path, *, cache_dir: Path, count: int = REVIEW_FRAME_COUNT, width: int = 640
) -> list[tuple[float, Path]]:
    """Evenly spaced frames across 6%..94% of the master (never head/tail black)."""
    from director_os.core.frames import grab_frame
    from director_os.core.qa_probe import probe

    try:
        dur = float(probe(str(master_path)).duration_s)
    except Exception as e:  # noqa: BLE001 — probe failure means no frames to sample
        raise RuntimeError(f"master duration probe failed: {e}") from e
    if dur <= 0:
        raise RuntimeError("master duration is zero — nothing to sample")
    count = max(2, count)
    lo, hi = 0.06, 0.94
    times = [round((lo + (hi - lo) * i / (count - 1)) * dur, 2) for i in range(count)]
    return [(t, grab_frame(master_path, t, cache_dir=cache_dir, width=width)) for t in times]


def build_review_backend(deps: Any) -> ReviewBackend | None:
    """Injection first, then vision (it SEES the frames), then the text chain
    (the no-frames lane: program map + QA + chronology evidence only). The
    one-shot ReviewBackend port remains the injection fallback for callers
    that hand us a pre-built backend."""
    factory = getattr(deps, "review_backend_factory", None)
    if factory is not None:
        backend = factory()
        wrapped = text_review_backend(backend)
        if wrapped is not None:
            return wrapped
        if callable(getattr(backend, "review", None)):
            return backend  # already satisfies the ReviewBackend port
        return None
    try:
        from director_os.perception.vision import build_vision_client

        vlm = vision_review_backend(build_vision_client())
        if vlm is not None:
            return vlm
    except Exception:  # noqa: BLE001 — vision misavailability is not a review crash
        pass
    chain = None
    chain_factory = getattr(deps, "director_backend_factory", None)
    if chain_factory is not None:
        try:
            chain = chain_factory()
        except Exception:  # noqa: BLE001 — an LLM outage degrades to unavailable
            chain = None
    else:
        try:
            from director_os.agent_runtime.llm import build_backend_chain
            from director_os.agent_runtime.registry import ToolRegistry

            chain = build_backend_chain(ToolRegistry())
        except Exception:  # noqa: BLE001 — nothing configured = unavailable, loudly
            chain = None
    return text_review_backend(chain)


def build_critic_adapter(deps: Any) -> CriticTurnAdapter | None:
    """The critic's EYE for the ReAct loop: injected `ask` first, then the
    vision client, then the text chain (inspection rounds return evidence
    text, never pixels). None = no eye configured at all."""
    factory = getattr(deps, "review_backend_factory", None)
    if factory is not None:
        backend = factory()
        if callable(getattr(backend, "ask", None)):
            return backend
        return None
    try:
        from director_os.perception.vision import build_vision_client

        eye = vision_critic_adapter(build_vision_client())
        if eye is not None:
            return eye
    except Exception:  # noqa: BLE001 — vision misavailability is not a crash
        pass
    chain_factory = getattr(deps, "director_backend_factory", None)
    chain = None
    if chain_factory is not None:
        try:
            chain = chain_factory()
        except Exception:  # noqa: BLE001 — an LLM outage degrades to unavailable
            chain = None
    else:
        try:
            from director_os.agent_runtime.llm import build_backend_chain
            from director_os.agent_runtime.registry import ToolRegistry

            chain = build_backend_chain(ToolRegistry())
        except Exception:  # noqa: BLE001 — nothing configured = unavailable
            chain = None
    return text_critic_adapter(chain)


def build_evidence(
    *,
    deterministic_issues: list[str],
    audit_failures: list[str],
    timeline,
    master_path: Path,
) -> str:
    """One bounded text pack: program map + watch-experience context."""
    from director_os.otio_engine.timeline import MASTER_TRACK, is_gap, track_by_name

    lines: list[str] = [f"MASTER: {master_path.name}"]
    track = track_by_name(timeline, MASTER_TRACK)
    if track is not None:
        for idx, item in enumerate(track):
            if is_gap(item):
                lines.append(f"  [{idx}] gap")
                continue
            meta = (getattr(item, "metadata", {}) or {}).get("director_os") or {}
            kind = meta.get("kind", "?")
            name = getattr(item, "name", "") or f"clip_{idx}"
            sr = getattr(item, "source_range", None)
            window = ""
            if sr is not None:
                s0 = sr.start_time.to_seconds()
                window = f" src {s0:.1f}s+{sr.duration.to_seconds():.1f}s"
            extra = ""
            if kind == "narration":
                extra = f" script: {str(meta.get('script_text', ''))[:110]!r}"
            elif kind == "card":
                extra = f" card: {name}"
            lines.append(f"  [{idx}] {kind} {name}{window}{extra}")

    if deterministic_issues:
        lines.append("STRUCTURAL FINDS: " + " | ".join(deterministic_issues))
    if audit_failures:
        lines.append("QA FAILURES: " + " | ".join(audit_failures))
    lines.append(
        f"NOTE: max {REVIEW_FRAME_COUNT} repairs per round; repairs run in listed order; "
        "each sees the timeline after the previous one."
    )
    return "\n".join(lines)[:4000]


def stage_review(
    final_for_qa: Path,
    timeline,
    audit_failures: list[str],
    project_id: str,
    deps: Any,
) -> ReviewOutcome:
    """The agentic review: LOOK (on demand) -> judge -> repair (maybe).

    The critic works in ReAct rounds — it may request exact frames anywhere
    on the master before judging; the editor applies its repairs through the
    typed registry; the re-render comes back with NEW frames and the critic
    judges again (the pipeline bounds the duet at 2 renders). Never raises.
    """
    started = time.monotonic()
    outcome = ReviewOutcome()
    if getattr(deps, "review_pass", True) is False:
        outcome.status = "off"
        return outcome

    det_issues = deterministic_review(timeline)
    outcome.issues.extend(det_issues)

    _log = getattr(deps, "log", lambda _m: None)
    try:
        frames = sample_master_frames(
            final_for_qa, cache_dir=Path(deps.settings.output_dir) / "review_frames" / project_id
        )
    except Exception as e:  # noqa: BLE001 — frame/capability failure: report, don't kill
        outcome.status = "unavailable"
        outcome.summary = f"frame sampling failed ({e})"
        _log(f"  [review] {outcome.summary}")
        return outcome
    outcome.frames = [t for t, _ in frames]
    outcome.inspected = list(outcome.frames)

    evidence = build_evidence(
        deterministic_issues=det_issues,
        audit_failures=list(audit_failures)[:8],
        timeline=timeline,
        master_path=final_for_qa,
    )

    on_demand: list[float] = []

    def _grab(times: list[float]) -> list[tuple[float, Path]]:
        from director_os.core.frames import grab_frame

        on_demand.extend(times)
        return [
            (t, grab_frame(final_for_qa, t, cache_dir=frames_cache_dir, width=640)) for t in times
        ]

    frames_cache_dir = Path(deps.settings.output_dir) / "review_frames" / project_id
    verdict = None
    eye = build_critic_adapter(deps)
    if eye is not None:
        from director_os.core.qa_probe import probe

        try:
            dur = float(probe(str(final_for_qa)).duration_s)
        except Exception:  # noqa: BLE001 — probe failure degrades honestly
            dur = 0.0
        try:
            if dur <= 0:
                raise CriticTurnError("master duration is zero — the critic cannot look")
            manifest = ", ".join(f"{t:.2f}s" for t, _ in frames)
            base_prompt = (
                f"{REVIEW_SYSTEM}\n\n## Structural evidence\n{evidence}\n\n"
                f"## Round 0\n{len(frames)} frame(s) ATTACHED at: {manifest}. "
                "Inspect more if needed, then give the verdict."
            )
            verdict, trace = run_critic(
                eye,
                base_prompt=base_prompt,
                initial_frames=frames,
                grab_frames=_grab,
                duration_s=dur,
                log=getattr(deps, "log", None),
            )
            outcome.inspect_rounds = sum(1 for t in trace if t.get("action") == "inspect")
            outcome.inspected.extend(on_demand)
        except Exception as e:  # noqa: BLE001 — a critic outage never kills a QA-passed master
            _log(f"  [review] critic loop failed ({e})")
            verdict = None
            outcome.issues.append(f"critic loop failed: {str(e)[:200]}")
    else:
        backend = build_review_backend(deps)
        if backend is None:
            outcome.status = "unavailable"
            outcome.summary = (
                "no review backend configured (set VISION_API_KEY) — deterministic "
                f"evidence: {len(det_issues)} check(s) recorded"
            )
            _log(f"  [review] {outcome.summary}")
            return outcome
        try:
            verdict = backend.review(frames, evidence)
        except Exception as e:  # noqa: BLE001 — a VLM outage never kills a QA-passed master
            outcome.status = "unavailable"
            outcome.summary = f"review backend failed ({e})"
            _log(f"  [review] {outcome.summary}")
            return outcome

    if verdict is None:
        outcome.status = "unavailable"
        outcome.summary = "critic produced no verdict — deterministic evidence recorded"
        _log(f"  [review] {outcome.summary}")
        return outcome

    outcome.issues.extend(verdict.issues[:12])
    if verdict.verdict == "ok" or not verdict.repairs:
        outcome.summary = f"clean — {len(verdict.issues)} issue note(s), no repairs proposed"
        getattr(deps, "log", lambda _m: None)(f"  [review] {outcome.summary}")
        return outcome

    applied, refused = apply_repairs(
        timeline, project_id, verdict.repairs, log=getattr(deps, "log", None)
    )
    outcome.applied = applied
    outcome.refused = refused
    if applied:
        outcome.status = "applied"
        outcome.summary = (
            f"{len(applied)} repair(s) applied, {len(refused)} refused — re-render follows"
        )
    else:
        outcome.status = "clean"
        outcome.summary = f"all {len(refused)} proposed repair(s) refused — program unchanged"
    getattr(deps, "log", lambda _m: None)(
        f"  [review] {outcome.summary} ({time.monotonic() - started:.1f}s)"
    )
    return outcome
