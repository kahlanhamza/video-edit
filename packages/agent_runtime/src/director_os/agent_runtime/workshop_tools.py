"""Agent ACTION surface — caption workbench, self-QA, undo and save (PLAN §3.3C).

`editing_tools.py` mutates the TIMELINE; `inspection_tools.py` reads the
EVIDENCE. A working pro editor also needs to ACT on the deliverable and
recover from regret — this module is the third leg of the toolbox:

  captions.init_work_copy     copy the shipped .ass into the session workspace
  captions.apply_tag_edits    rewrite windows as gray-italic tag cues (owner law)
  captions.list_work_cues     read back the edited cues (verify your own edit)
  qa.run_caption_checks       the 4 caption-truth gates on the WORK copy
  undo.list_checkpoints       version history of the working timeline
  undo.rollback               restore a durable version (plan-terminal action)
  timeline.save               export the working timeline (.otio) for compile

Workbench law: the shipped pipeline artifacts are NEVER mutated in place —
edits land in the session work dir, the agent verifies its own edit with the
QA tool, and only the owner's re-burn stage touches the master.

`undo.rollback` rebinds `REG_TIMELINE[0]` to the restored object; a session
must adopt it at the next round boundary (treat the step as plan-final), so
the executor's terminal checkpoint never resurrects pre-rollback state.

Every input is a Pydantic model (the registry validates BEFORE the tool runs).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Literal

import opentimelineio.adapters as ot_adapters
import opentimelineio.schema as ot_schema
from director_os.agent_runtime.editing_tools import ToolOperationError
from director_os.agent_runtime.inspection_tools import (
    REG_TIMELINE,
    _fmt_ass_to_sec,
    _parse_cues,
    _Window,
)
from director_os.agent_runtime.registry import ToolRegistry
from director_os.otio_engine.timeline import MASTER_TRACK, track_by_name
from pydantic import BaseModel, Field, model_validator

#: gray-italic inline override — the owner caption-truth law for unclear voice
TAG_OVERRIDE = r"{\i1\c&H9C9C9C&\2c&H9C9C9C&}"
TAG_WORDS = ("unaudible", "mumbling", "whispering", "cross-talk", "panicking")
MAX_EDITS_PER_CALL = 8
GAP_MIN_SPAN_S = 1.0  # a VAD speech span this long MUST carry a cue
LOOP_REPEAT_N = 3  # same text tiled back-to-back 3+ = hallucination-loop
MIN_REASON_LEN = 4


def _ass_ts(s: float) -> str:
    s = max(0.0, s)
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h}:{m:02d}:{sec:05.2f}"


def _plain_text(text: str) -> str:
    return re.sub(r"\{[^}]*\}", "", text).strip()


def _is_tag(text: str) -> bool:
    return _plain_text(text).startswith("*")


def _atomic_write(p: Path, content: str) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(p)


# ---------------------------------------------------------------- inputs
class _Empty(BaseModel):
    pass


class _TagEdit(BaseModel):
    start_s: float = Field(ge=0.0)
    end_s: float = Field(gt=0.0)
    tag: Literal["unaudible", "mumbling", "whispering", "cross-talk", "panicking"]
    reason: str = Field(min_length=MIN_REASON_LEN, max_length=400)

    @model_validator(mode="after")
    def _ordered(self) -> _TagEdit:
        if self.end_s <= self.start_s:
            raise ValueError("end_s must be > start_s")
        return self


class _TagEdits(BaseModel):
    edits: list[_TagEdit] = Field(min_length=1, max_length=MAX_EDITS_PER_CALL)

    @model_validator(mode="after")
    def _no_overlap(self) -> _TagEdits:
        ordered = sorted(self.edits, key=lambda e: e.start_s)
        for a, b in zip(ordered, ordered[1:], strict=False):
            if b.start_s < a.end_s:
                raise ValueError(
                    f"edit windows overlap: [{a.start_s}, {a.end_s}) vs [{b.start_s}, {b.end_s})"
                )
        return self


class _OwnerWindow(BaseModel):
    owner_windows: list[list[float]] | None = Field(default=None, max_length=8)


class _Rollback(BaseModel):
    to_version_num: int = Field(ge=1)


def register_workshop_tools(
    registry: ToolRegistry,
    work_dir: Path,
    source_ass: Path,
    evidence_path: Path,
    vad_cache_path: Path,
    store: Any,
    project_id: str,
) -> None:
    """Register the caption workbench + undo + save tools for one session.

    `store` is duck-typed to the CheckpointStore surface (checkpoint ignored
    here; list_versions/rollback used by the undo tools).
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    work_ass = work_dir / "captions_work.ass"

    # ------------------------------------------------------------ caption workbench
    def init_work_copy() -> dict[str, Any]:
        if not source_ass.exists():
            raise ToolOperationError(f"source captions missing: {source_ass}")
        work_ass.write_text(source_ass.read_text(encoding="utf-8"), encoding="utf-8")
        cues = _parse_cues(work_ass)
        tags = sum(1 for c in cues if _is_tag(c["text"]))
        return {
            "detail": f"work copy ready: {len(cues)} cues ({tags} tag cues) -> {work_ass.name}",
        }

    def apply_tag_edits(edits: list[dict[str, Any]]) -> dict[str, Any]:
        if not work_ass.exists():
            raise ToolOperationError("no work copy yet — call captions.init_work_copy first")
        lines = work_ass.read_text(encoding="utf-8").splitlines()
        applied: list[dict[str, Any]] = []
        for edit in edits:
            s, e, tag = float(edit["start_s"]), float(edit["end_s"]), str(edit["tag"])
            reason = str(edit["reason"])
            out: list[str] = []
            replaced_style: str | None = None
            n_replaced = 0
            for ln in lines:
                if not ln.startswith("Dialogue: "):
                    out.append(ln)
                    continue
                p = ln.split(",", 9)
                cs, ce = _fmt_ass_to_sec(p[1]), _fmt_ass_to_sec(p[2])
                if ce > s and cs < e:  # overlaps the edit window
                    n_replaced += 1
                    replaced_style = replaced_style or p[3]
                    continue
                out.append(ln)
            if n_replaced:
                style = replaced_style or "Officer"
                tag_line = (
                    f"Dialogue: 0,{_ass_ts(s)},{_ass_ts(e)},{style},,0,0,46,,{TAG_OVERRIDE}*{tag}*"
                )
                out.insert(_insert_pos(out, s), tag_line)
            else:
                # gap-fill law: speech without a cue still gets the gray tag
                tag_line = (
                    f"Dialogue: 0,{_ass_ts(s)},{_ass_ts(e)},Officer,,0,0,46,,{TAG_OVERRIDE}*{tag}*"
                )
                out.insert(_insert_pos(out, s), tag_line)
            _atomic_write(work_ass, "\n".join(out) + "\n")
            lines = out
            applied.append(
                {
                    "window": [round(s, 3), round(e, 3)],
                    "tag": tag,
                    "reason": reason.strip(),
                    "cues_replaced": n_replaced,
                }
            )
        audit = work_dir / "caption_edits_applied.json"
        log = json.loads(audit.read_text()) if audit.exists() else []
        log.extend([{**a, "applied_at": time.time()} for a in applied])
        _atomic_write(audit, json.dumps(log, indent=1))
        return {
            "detail": "; ".join(
                f"[{a['window'][0]:.1f}-{a['window'][1]:.1f}]s -> *{a['tag']}* "
                f"({a['cues_replaced']} cue(s) replaced)"
                for a in applied
            ),
        }

    def _insert_pos(lines: list[str], s: float) -> int:
        """Insert index that keeps the Dialogue block time-ordered."""
        pos, seen = 0, 0
        for ln in lines:
            if not ln.startswith("Dialogue: "):
                continue
            seen += 1
            if _fmt_ass_to_sec(ln.split(",", 9)[1]) <= s:
                pos = seen
        return pos

    def list_work_cues(start_s: float, end_s: float) -> dict[str, Any]:
        end_s = min(end_s, start_s + 120.0)
        if not work_ass.exists():
            raise ToolOperationError("no work copy yet — call captions.init_work_copy first")
        cues = _parse_cues(work_ass)
        hits = [c for c in cues if c["e"] > start_s and c["s"] < end_s]
        return {
            "window": [start_s, end_s],
            "cues": hits,
            "n_cues": len(hits),
            "n_tag_cues": sum(1 for c in hits if _is_tag(c["text"])),
        }

    # ------------------------------------------------------------ self-QA
    def run_caption_checks(owner_windows: list[list[float]] | None = None) -> dict[str, Any]:
        if not work_ass.exists():
            raise ToolOperationError("no work copy yet — call captions.init_work_copy first")
        if not evidence_path.exists():
            raise ToolOperationError(f"caption evidence missing: {evidence_path}")
        cues = _parse_cues(work_ass)
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        checks: list[dict[str, Any]] = []

        # 1. color-vs-role: plain-text cues FULLY INSIDE one role event must
        # match it. Straddling cues are skipped: caption_evidence segments are
        # diarization spans (some 16 s long, some zero-length boundary
        # artifacts), and boundary-straddling Officer/Suspect cues legitimately
        # overlap neighboring spans — judging them produced 25 false
        # violations in agent session v2 (the shipped pixel-level qa_v4 audit
        # uses lonely-cue sampling for the same reason).
        violations: list[str] = []
        for ev in evidence:
            role = str(ev.get("role", "")).title()
            if role not in ("Officer", "Suspect", "Narrator") or not ev.get("has_words"):
                continue
            if float(ev["e"]) - float(ev["s"]) < 0.3:  # boundary artifact
                continue
            for c in cues:
                if _is_tag(c["text"]):
                    continue
                inside = c["s"] >= ev["s"] - 0.05 and c["e"] <= ev["e"] + 0.05
                if inside and c["style"] != role:
                    violations.append(
                        f"[{c['s']:.1f}-{c['e']:.1f}]s style={c['style']} "
                        f"inside {role} speech {ev['s']:.1f}-{ev['e']:.1f}s "
                        f"({(c['text'] or '...')[:40]!r})"
                    )
        checks.append(
            {
                "name": "caption_color_vs_speaker",
                "ok": not violations,
                "n_violations": len(violations),
                "detail": violations[:5],
            }
        )

        # 2. owner windows carry ONLY gray tag cues
        win_viol: list[str] = []
        for w0, w1 in [tuple(w) for w in (owner_windows or [])]:
            for c in cues:
                if c["e"] > w0 and c["s"] < w1 and not _is_tag(c["text"]):
                    win_viol.append(
                        f"[{c['s']:.1f}-{c['e']:.1f}]s plain text in owner "
                        f"window [{w0:.1f}-{w1:.1f}]s: {c['text'][:40]!r}"
                    )
        checks.append(
            {
                "name": "owner_windows_tag_only",
                "ok": not win_viol,
                "n_windows": len(owner_windows or []),
                "detail": win_viol[:5],
            }
        )

        # 3. no hallucination loop: the same plain text tiled BACK-TO-BACK
        # 3+ times (each cue starting within 2 s of the previous one ending).
        # Legitimate short interjections ("Okay.") repeat across a runtime —
        # agent session v2 flagged those falsely; the ASR-loop signature is
        # TILING, not repetition.
        plain = sorted(
            [c for c in cues if not _is_tag(c["text"]) and c["text"]],
            key=lambda c: c["s"],
        )
        loops: list[str] = []
        run_text: str | None = None
        run_last_end = -1e9
        run_n = 0
        sentinel: dict[str, Any] | None = None
        for cur in [*plain, sentinel]:  # sentinel flushes the final run
            # inline condition (not a precomputed bool) so mypy keeps the
            # None-narrowing for the branch body
            if cur is not None and cur["text"] == run_text and cur["s"] - run_last_end <= 2.0:
                run_n += 1
                run_last_end = cur["e"]
                continue
            if run_n >= LOOP_REPEAT_N:
                loops.append(f"{(run_text or '')[:40]!r} x{run_n} tiled back-to-back")
            run_text = cur["text"] if cur else None
            run_n = 1
            run_last_end = cur["e"] if cur else -1e9
        checks.append(
            {
                "name": "no_loop_text",
                "ok": not loops,
                "detail": loops[:5],
            }
        )

        # 4. gap coverage: VAD speech spans >= 1.0s must carry SOME cue
        if not vad_cache_path.exists():
            raise ToolOperationError(
                f"VAD cache missing ({vad_cache_path}) — call "
                "audio.speech_spans first so the cache exists"
            )
        spans = [(float(a), float(b)) for a, b in json.loads(vad_cache_path.read_text())["spans"]]
        gaps: list[str] = []
        for a, b in spans:
            if b - a < GAP_MIN_SPAN_S:
                continue
            if not any(c["e"] > a and c["s"] < b for c in cues):
                gaps.append(f"[{a:.1f}-{b:.1f}]s speech without any cue")
        checks.append(
            {
                "name": "speech_gap_coverage",
                "ok": not gaps,
                "n_spans": len(spans),
                "detail": gaps[:5],
            }
        )

        # 5. a tag over CLEAN speech is a violation too (skill v0.5 law):
        # tagging evidence-backed clean dialogue mutes real words. A tag cue
        # fully inside a clean event (has_words, speech_ratio >= 0.5, rms >
        # -35 dB, repeat_n < 3) means the agent over-tagged.
        clean_viol: list[str] = []
        for c in cues:
            if not _is_tag(c["text"]):
                continue
            for ev in evidence:
                if float(ev["e"]) - float(ev["s"]) < 0.3:
                    continue
                clean = (
                    bool(ev.get("has_words"))
                    and float(ev.get("speech_ratio", 0.0)) >= 0.5
                    and float(ev.get("rms_db", -99.0)) > -35.0
                    and int(ev.get("repeat_n", 0)) < 3
                )
                if clean and c["s"] >= ev["s"] - 0.05 and c["e"] <= ev["e"] + 0.05:
                    clean_viol.append(
                        f"[{c['s']:.1f}-{c['e']:.1f}]s tag over CLEAN speech "
                        f"{ev['s']:.1f}-{ev['e']:.1f}s (ratio {ev.get('speech_ratio')})"
                    )
        checks.append(
            {
                "name": "no_tags_on_clean_speech",
                "ok": not clean_viol,
                "detail": clean_viol[:5],
            }
        )

        report = {
            "target": str(work_ass),
            "checks": checks,
            "green": all(c["ok"] for c in checks),
        }
        _atomic_write(work_dir / "qa_caption_work.json", json.dumps(report, indent=1))
        return report

    # ------------------------------------------------------------ undo
    def list_checkpoints() -> dict[str, Any]:
        versions = store.list_versions(project_id)
        return {
            "n_versions": len(versions),
            "versions": list(reversed(versions))[:50],
        }

    def rollback(to_version_num: int) -> dict[str, Any]:
        if not REG_TIMELINE:
            raise ToolOperationError("no working timeline bound")
        restored, new_version = store.rollback(project_id, to_version_num)
        REG_TIMELINE[0] = restored
        return {
            "detail": (
                f"rolled back to v{to_version_num} (durable as v{new_version}); "
                "the session adopts the restored timeline next round — treat "
                "this step as plan-final"
            ),
            "new_version": new_version,
        }

    # ------------------------------------------------------------ save
    def save_timeline() -> dict[str, Any]:
        if not REG_TIMELINE:
            raise ToolOperationError("no working timeline bound")
        tl = REG_TIMELINE[0]
        doc = ot_adapters.write_to_string(tl)
        if not doc:
            raise ToolOperationError("timeline serialization returned nothing")
        out = work_dir / "timeline_saved.otio"
        _atomic_write(out, doc)
        track = track_by_name(tl, MASTER_TRACK)
        n_items = len(list(track)) if track is not None else 0
        total = sum(
            (i.source_range.duration.to_seconds() if i.source_range else 0.0)
            for i in (track or [])
            if isinstance(i, ot_schema.Clip)
        )
        return {
            "detail": f"timeline saved: {n_items} master items, {total:.1f}s -> {out}",
            "path": str(out),
        }

    registry.register(
        "captions.init_work_copy",
        "Copy the shipped captions into the session workspace as "
        "captions_work.ass. REQUIRED before any tag edit or QA check — the "
        "shipped pipeline artifacts are never edited in place.",
        _Empty,
        lambda: init_work_copy(),
    )
    registry.register(
        "captions.apply_tag_edits",
        "Apply gray-italic tag edits (up to 8 windows): every cue overlapping "
        "a window is replaced by ONE merged tag cue (owner caption-truth law); "
        "a window with no cue gets a gap-fill tag. Non-overlapping windows "
        "only; write an audit JSON.",
        _TagEdits,
        lambda **kw: apply_tag_edits(**kw),
    )
    registry.register(
        "captions.list_work_cues",
        "Read back the EDITED work copy cues in [start_s, end_s] (style, "
        "text, tag flag) — verify your own edit before running QA. Window "
        "clamped to 120 s.",
        _Window,
        lambda **kw: list_work_cues(**kw),
    )
    registry.register(
        "qa.run_caption_checks",
        "The 5 caption-truth gates on the WORK copy: color-vs-speaker, "
        "owner-windows-tag-only (pass the owner windows), no back-to-back "
        "loop text, VAD gap coverage, and no-tags-on-clean-speech (tagging "
        "evidence-backed clean dialogue is a violation too). Writes "
        "qa_caption_work.json; green means the edit is ready for the "
        "owner's re-burn.",
        _OwnerWindow,
        lambda **kw: run_caption_checks(**kw),
    )
    registry.register(
        "undo.list_checkpoints",
        "List the durable timeline versions (newest first, capped 50) — "
        "every mutating step is checkpointed, so any state is restorable.",
        _Empty,
        lambda: list_checkpoints(),
    )
    registry.register(
        "undo.rollback",
        "Restore a timeline version as the newest durable state and REBIND "
        "the working timeline. PLAN-TERMINAL: the session adopts the restored "
        "object at the round boundary; do not plan mutations after it.",
        _Rollback,
        lambda **kw: rollback(**kw),
    )
    registry.register(
        "timeline.save",
        "Export the working timeline to timeline_saved.otio in the session "
        "workspace — the deliverable the owner's compile stage consumes. "
        "Call after your final mutation round.",
        _Empty,
        lambda: save_timeline(),
    )
