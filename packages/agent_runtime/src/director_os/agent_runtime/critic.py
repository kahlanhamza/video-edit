"""Agent CRITIC surface — deterministic self-audit (PLAN §3.3E).

EditDuet (arXiv:2509.10761) pairs an Editor agent with a Critic agent; the
critic's verdict grounds the revision loop. director-os keeps the editing
loop SINGLE-agent (Anthropic's multi-agent economics — ~15x tokens — do not
pay off for sequential cuts) but borrows the second role as a TOOL: the
agent must run `qa.self_audit` and answer to it before claiming done.

Honest-absence semantics (adapted from samuelgursky/davinci-resolve-mcp):
a check that never ran is reported as `not established`, NEVER as a pass.
"A tool result the final response describes as successful but with an
unreported error is not a clean pass."

What it audits (all deterministic, no LLM):
  timeline structure    gaps / zero-length items / empty narration scripts
  plan invariants       duration floor, narration ratio, coverage (needs
                        source_duration_s input — else not established)
  caption workbench     the qa_caption_work.json verdict of the session's
                        own apply_tag_edits (absent -> not established)
  shipped master QA     qa_v4.json gate count green/total (absent -> skip)

The verdict is compact (token budget): one line per failed check, counts
only for green families.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import opentimelineio.schema as ot_schema
from director_os.agent_runtime.editing_tools import ToolOperationError
from director_os.agent_runtime.inspection_tools import REG_TIMELINE
from director_os.agent_runtime.registry import ToolRegistry
from director_os.otio_engine.timeline import MASTER_TRACK, track_by_name
from pydantic import BaseModel, Field

ZOOM_LAW_MAX = 1.25  # subtle-polish law (skill v0.4)
MIN_CLIP_S = 0.05  # below this a "clip" is a construction bug


class _SourceDur(BaseModel):
    source_duration_s: float | None = Field(default=None, gt=0.0)


def register_critic_tools(registry: ToolRegistry, work_dir: Path) -> None:
    """Register the deterministic critic pass for one session work dir."""

    def qa_self_audit(source_duration_s: float | None) -> dict[str, Any]:
        if not REG_TIMELINE:
            raise ToolOperationError("no working timeline bound")
        tl = REG_TIMELINE[0]
        track = track_by_name(tl, MASTER_TRACK)
        if track is None:
            raise ToolOperationError(f"timeline has no '{MASTER_TRACK}' track")
        checks: list[dict[str, Any]] = []
        items = list(track)

        # 1. structural: master gaps, zero-length items, empty narration scripts
        gaps: list[str] = []
        zero: list[str] = []
        empty_script: list[str] = []
        t = 0.0
        prev_end: float | None = None
        for i, item in enumerate(items):
            dur = item.duration().to_seconds() if item.duration() else 0.0
            if isinstance(item, ot_schema.Gap):
                if dur > 0.25:
                    gaps.append(f"item {i}: gap {dur:.2f}s at {t:.1f}s")
            elif dur < MIN_CLIP_S:
                zero.append(f"item {i}: {dur * 1000:.0f}ms at {t:.1f}s")
            else:
                raw = getattr(item, "metadata", {}).get("director_os", {})
                if raw.get("kind") == "narration" and not str(raw.get("script_text", "")).strip():
                    empty_script.append(f"item {i} at {t:.1f}s")
            if prev_end is not None and not isinstance(item, ot_schema.Gap):
                expected = prev_end
                if t > expected + 0.25:
                    gaps.append(f"item {i}: {t - expected:.2f}s unexplained advance")
            prev_end = t + dur
            t += dur

        def _chk(name: str, bad: list[str], fix: str) -> dict[str, Any]:
            return {
                "name": name,
                "ok": not bad,
                "detail": bad[:4] if bad else fix,
            }

        checks.append(_chk("no_master_gaps", gaps, "master track contiguous"))
        checks.append(_chk("no_zero_length_items", zero, "all items >= 50ms"))
        checks.append(_chk("narration_scripts_present", empty_script, "all narration has text"))

        # 2. zoom law: every raw clip's zoom curve stays in [1.0, 1.25]
        # (stored contract form is ZoomKeyframe dicts; wire form is pairs).
        # NOTE: OTIO wraps metadata vectors in its own AnyVector/AnyDictionary
        # C++ types — type-sniffing isinstance(kf, dict) is False there, so
        # duck-type on .get() instead.
        bad_zoom: list[str] = []
        for i, item in enumerate(items):
            raw = (
                getattr(item, "metadata", {}).get("director_os", {})
                if hasattr(item, "metadata")
                else {}
            )
            for kf in raw.get("zoom_keyframes") or []:
                if hasattr(kf, "get"):
                    z = float(kf.get("zoom_factor", 1.0))
                elif isinstance(kf, (list, tuple)) and len(kf) >= 2:
                    z = float(kf[1])
                else:
                    continue
                if z > ZOOM_LAW_MAX:
                    bad_zoom.append(f"item {i}: zoom {z} > {ZOOM_LAW_MAX}")
        checks.append(
            _chk("zoom_within_subtle_law", bad_zoom, f"all zooms <= {ZOOM_LAW_MAX} (subtle polish)")
        )

        # 3. plan invariants — only when the source length is supplied
        if source_duration_s is not None:
            from director_os.agent_runtime.plan_invariants import (
                audit_invariants,
                timeline_composition,
            )

            comp = timeline_composition(tl)
            failures = audit_invariants(comp, source_duration_s)
            checks.append(
                {
                    "name": "plan_invariants",
                    "ok": not failures,
                    "detail": [str(f) for f in failures[:4]]
                    or f"composition: {comp.final_duration_s:.1f}s, "
                    f"{comp.narrator_segment_count} narration, "
                    f"{comp.raw_segment_count} raw",
                }
            )
        else:
            checks.append(
                {
                    "name": "plan_invariants",
                    "ok": True,
                    "not_established": True,
                    "detail": "not established — pass source_duration_s to enforce",
                }
            )

        # 4. caption workbench verdict (the session's own earlier QA run)
        qcw = work_dir / "qa_caption_work.json"
        if qcw.exists():
            rep = json.loads(qcw.read_text(encoding="utf-8"))
            failed = [c["name"] for c in rep.get("checks", []) if not c.get("ok")]
            checks.append(
                {
                    "name": "caption_work_qa",
                    "ok": bool(rep.get("green")),
                    "detail": failed or f"green: {rep.get('checks')}",
                }
            )
        else:
            checks.append(
                {
                    "name": "caption_work_qa",
                    "ok": True,
                    "not_established": True,
                    "detail": "not established — no qa.run_caption_checks has run",
                }
            )

        # 5. shipped master QA (read-only glance at the pipeline's own gates)
        q4 = work_dir / "qa_v4.json"
        if q4.exists():
            gates = json.loads(q4.read_text(encoding="utf-8"))
            n_bad = sum(1 for c in gates if not c.get("ok"))
            checks.append(
                {
                    "name": "shipped_master_qa_v4",
                    "ok": n_bad == 0,
                    "detail": f"{len(gates) - n_bad}/{len(gates)} gates green",
                }
            )

        hard_fail = [c["name"] for c in checks if not c.get("ok") and not c.get("not_established")]
        return {
            "green": not hard_fail,
            "verdict": (
                "GREEN — safe to save and finish"
                if not hard_fail
                else f"RED — fix {hard_fail} before timeline.save"
            ),
            "checks": checks,
        }

    registry.register(
        "qa.self_audit",
        "The deterministic critic pass over the CURRENT working timeline: "
        "structural integrity (gaps/zero-length/empty narration), zoom "
        "subtle-law, plan invariants (when you pass source_duration_s), "
        "your own caption-workbench QA verdict, and the shipped master's "
        "qa_v4 gates. Honest-absence: a check that never ran says 'not "
        "established', never 'passed'. Run BEFORE timeline.save and after "
        "any structural edit; fix RED names before claiming done.",
        _SourceDur,
        lambda **kw: qa_self_audit(**kw),
    )
