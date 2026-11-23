"""Agent READ surface — measured-evidence inspection tools (PLAN §3.3B).

The director MUTATION toolbox (`editing_tools.py`) can cut, narrate and blur,
but until now the agent could not SEE the evidence it is supposed to act on —
the owner's caption-truth laws are evidence laws, so they are unenforceable
without a read surface. This module registers the read-only tools the AI
editor agent needs against the measured post-FX artifacts of a project:

  project.describe              artifact inventory + durations (ffprobe guarded)
  qa.read_report                the QA gate reports (qa_v4 11 checks + legacy)
  transcript.read_window        dialogue events (role, time, text, audio evidence)
  captions.list_cues            burned ASS cues in a window (style + tag law)
  audio.speech_spans            VAD speech spans (cached compute, resumable)
  captions.propose_tag_window   owner-law proposals -> proposals_v1.json
  timeline.list_items           master-track inventory of the working timeline
  frame.grab                    ONE frame of the master at t_s as JPEG (LRU-cached)
  agent.finish                  terminal session summary marker

Every input is a Pydantic model (the registry validates BEFORE the tool runs).
All tools are read-only on media: the only writes are the proposals JSON and
the finish marker. `REG_TIMELINE` holds the working timeline so tools follow
the authoritative object even after a checkpoint rollback (the session
re-binds `REG_TIMELINE[0]` after every round).
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Literal

import opentimelineio.schema as ot_schema
from director_os.agent_runtime.editing_tools import ToolOperationError
from director_os.agent_runtime.registry import ToolRegistry
from director_os.otio_engine.timeline import MASTER_TRACK, track_by_name
from pydantic import BaseModel, Field, model_validator

#: holds the CURRENT working timeline (the session re-binds after rollbacks)
REG_TIMELINE: list[Any] = []

TAG_WORDS = ("unaudible", "mumbling", "whispering", "cross-talk", "panicking")
_WINDOW_MAX_S = 120.0  # context-protection clamp for window queries


def _work(work_dir: Path, name: str) -> Path:
    p = work_dir / name
    if not p.exists():
        raise ToolOperationError(f"artifact missing: {p}")
    return p


def _load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _fmt_ass_to_sec(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _parse_cues(ass_path: Path) -> list[dict[str, Any]]:
    out = []
    for ln in ass_path.read_text(encoding="utf-8").splitlines():
        if not ln.startswith("Dialogue: "):
            continue
        p = ln.split(",", 9)
        plain = re.sub(r"\{[^}]*\}", "", p[9]).strip()
        out.append(
            {
                "s": round(_fmt_ass_to_sec(p[1]), 3),
                "e": round(_fmt_ass_to_sec(p[2]), 3),
                "style": p[3],
                "text": plain,
            }
        )
    return out


def _probe_dur(path: Path) -> float | None:
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(path),
            ],
            timeout=30,
        )
        return round(float(out.decode().strip()), 2)
    except Exception:  # noqa: BLE001 — missing binary/file: report None
        return None


def _probe_fps(path: Path) -> float | None:
    """Stream frame rate as a float (parses ffprobe's `30000/1001` form)."""
    try:
        out = (
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=r_frame_rate",
                    "-of",
                    "csv=p=0",
                    str(path),
                ],
                timeout=30,
            )
            .decode()
            .strip()
        )
        if "/" in out:
            num, den = out.split("/", 1)
            return round(float(num) / float(den), 3)
        return round(float(out), 3)
    except Exception:  # noqa: BLE001 — missing binary/file: report None
        return None


def _load_or_compute_spans(work_dir: Path) -> list[tuple[float, float]]:
    """VAD speech spans, cached to vad_spans_cache.json (resumable)."""
    cache = work_dir / "vad_spans_cache.json"
    if cache.exists():
        d = _load_json(cache)
        return [(float(a), float(b)) for a, b in d["spans"]]
    try:
        import sys

        repo_scripts = Path(__file__).resolve().parents[5] / "scripts"
        sys.path.insert(0, str(repo_scripts))
        import vad_evidence
    except Exception as e:  # noqa: BLE001 — no engine available: hard fail
        raise ToolOperationError(
            f"cannot compute speech spans (no cache, no vad engine): {e}"
        ) from e
    pcm = vad_evidence.load_pcm(work_dir / "audio_src_16k.wav")
    flags, _rms, frame = vad_evidence.vad_flags(pcm)
    spans = vad_evidence.speech_spans(flags, frame, min_s=0.45, join_gap=0.30)
    cache.write_text(
        json.dumps(
            {
                "frame_s": frame / vad_evidence.SR,
                "sr": vad_evidence.SR,
                "spans": [[round(a, 3), round(b, 3)] for a, b in spans],
            }
        )
    )
    return spans


# ---------------------------------------------------------------- inputs
class _Empty(BaseModel):
    pass


class _FrameGrab(BaseModel):
    """Frame-addressable evidence: ONE frame at a timestamp, as an image."""

    t_s: float = Field(ge=0.0, le=24 * 3600.0)
    width: int = Field(default=640, ge=64, le=1920)


class _Window(BaseModel):
    start_s: float = Field(ge=0.0)
    end_s: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _ordered(self) -> _Window:
        if self.end_s <= self.start_s:
            raise ValueError("end_s must be > start_s")
        return self


class _TagWindow(BaseModel):
    start_s: float = Field(ge=0.0)
    end_s: float = Field(gt=0.0)
    tag: Literal["unaudible", "mumbling", "whispering", "cross-talk", "panicking"]
    reason: str = Field(min_length=4, max_length=400)


class _Finish(BaseModel):
    summary: str = Field(min_length=4, max_length=4000)


class _Index(BaseModel):
    item_index: int = Field(ge=0)


# ---------------------------------------------------------------- tools
def register_inspection_tools(registry: ToolRegistry, work_dir: Path) -> None:
    """Register the read-only evidence surface for one project's artifacts."""

    def project_describe() -> dict[str, Any]:
        inv: dict[str, Any] = {"work_dir": str(work_dir)}
        master = work_dir / "master_postfx_v4.mp4"
        inv["master_postfx_v4.mp4"] = {
            "exists": master.exists(),
            "size_mb": round(master.stat().st_size / 1e6, 1) if master.exists() else None,
            "duration_s": _probe_dur(master) if master.exists() else None,
        }
        for name in (
            "subtitles_v4.ass",
            "captions_v4.srt",
            "qa_v4.json",
            "caption_evidence.json",
            "decisions_v4.json",
            "caption_v4_stats.json",
        ):
            p = work_dir / name
            entry: dict[str, Any] = {"exists": p.exists()}
            if p.exists():
                entry["bytes"] = p.stat().st_size
            if name == "subtitles_v4.ass" and p.exists():
                cues = _parse_cues(p)
                entry["cues"] = len(cues)
                entry["tag_cues"] = sum(1 for c in cues if "*" in c["text"])
            inv[name] = entry
        st = work_dir / "caption_v4_stats.json"
        if st.exists():
            inv["caption_stats"] = _load_json(st)
        return inv

    def qa_read_report() -> dict[str, Any]:
        out: dict[str, Any] = {}
        q4 = work_dir / "qa_v4.json"
        if q4.exists():
            checks = _load_json(q4)
            out["qa_v4"] = {
                "checks": checks,
                "n": len(checks),
                "n_failed": sum(1 for c in checks if not c.get("ok")),
                "green": all(c.get("ok") for c in checks),
            }
        legacy = work_dir / "qa_report_final.json"
        if legacy.exists():
            out["qa_report_final"] = _load_json(legacy)
        if not out:
            raise ToolOperationError("no QA report found in work dir")
        return out

    def transcript_read_window(start_s: float, end_s: float) -> dict[str, Any]:
        end_s = min(end_s, start_s + _WINDOW_MAX_S)
        evs = _load_json(_work(work_dir, "caption_evidence.json"))
        hits = [e for e in evs if e["e"] > start_s and e["s"] < end_s]
        return {"window": [start_s, end_s], "events": hits, "n_events": len(hits)}

    def captions_list_cues(start_s: float, end_s: float) -> dict[str, Any]:
        end_s = min(end_s, start_s + _WINDOW_MAX_S)
        cues = _parse_cues(_work(work_dir, "subtitles_v4.ass"))
        hits = [c for c in cues if c["e"] > start_s and c["s"] < end_s]
        cov = sum(min(c["e"], end_s) - max(c["s"], start_s) for c in hits)
        return {
            "window": [start_s, end_s],
            "cues": hits,
            "n_cues": len(hits),
            "coverage_s": round(cov, 2),
        }

    def audio_speech_spans(start_s: float, end_s: float) -> dict[str, Any]:
        end_s = min(end_s, start_s + _WINDOW_MAX_S)
        spans = _load_or_compute_spans(work_dir)
        hits = [[round(a, 3), round(b, 3)] for a, b in spans if b > start_s and a < end_s]
        total = sum(min(b, end_s) - max(a, start_s) for a, b in hits)
        return {
            "window": [start_s, end_s],
            "spans": hits,
            "n_spans": len(hits),
            "speech_s": round(total, 2),
        }

    def propose_tag_window(start_s: float, end_s: float, tag: str, reason: str) -> dict[str, Any]:
        if tag not in TAG_WORDS:
            raise ToolOperationError(f"tag must be one of {TAG_WORDS}")
        p = work_dir / "proposals_v1.json"
        proposals = _load_json(p) if p.exists() else []
        proposals.append(
            {
                "start_s": round(start_s, 3),
                "end_s": round(end_s, 3),
                "tag": tag,
                "reason": reason.strip(),
                "proposed_at": time.time(),
            }
        )
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(proposals, indent=1))
        tmp.replace(p)
        return {
            "detail": f"proposal #{len(proposals)} filed: {tag} on "
            f"[{start_s:.1f}, {end_s:.1f}]s — {reason.strip()[:80]}"
        }

    def timeline_list_items() -> dict[str, Any]:
        items = _master_inventory()
        return {
            "n_items": len(items),
            "items": items,
            "total_s": items[-1]["start_s"] + items[-1]["dur_s"] if items else 0.0,
        }

    def timeline_get_item(item_index: int) -> dict[str, Any]:
        items = _master_inventory()
        if not 0 <= item_index < len(items):
            raise ToolOperationError(
                f"item_index {item_index} out of range (master has {len(items)} items)"
            )
        return items[item_index]

    def frame_grab(t_s: float, width: int) -> dict[str, Any]:
        """Frame-addressable evidence — the owner's frame-by-frame law.

        Extracts ONE frame of the working master at `t_s` as a JPEG (LRU
        cache: repeated grabs of the same frame are free). Use it to LOOK at
        a specific frame before acting on it — an edit proposal that names a
        frame must be grounded in that frame, not guessed from a time range.
        """
        master = work_dir / "master_postfx_v4.mp4"
        if not master.exists():
            # fall back to any assembled master in the work dir
            candidates = sorted(work_dir.glob("master*.mp4"))
            if not candidates:
                raise ToolOperationError(
                    f"no master video in work dir to grab frames from ({work_dir})"
                )
            master = candidates[0]
        from director_os.core.frames import FrameGrabError, grab_frame

        try:
            frame = grab_frame(master, t_s, cache_dir=work_dir / "frames", width=width)
        except FrameGrabError as e:
            raise ToolOperationError(str(e)) from e
        fps = _probe_fps(master) or 30.0
        return {
            "frame_path": str(frame),
            "t_s": round(t_s, 3),
            "frame_no": int(t_s * fps) + 1,  # 1-indexed like an NLE timecode
            "fps": fps,
            "width": width,
            "bytes": frame.stat().st_size,
            "note": "frame extracted; open frame_path to inspect pixels before editing",
        }

    def _master_inventory() -> list[dict[str, Any]]:
        if not REG_TIMELINE:
            raise ToolOperationError("no working timeline bound")
        tl = REG_TIMELINE[0]
        track = track_by_name(tl, MASTER_TRACK)
        if track is None:
            raise ToolOperationError(f"timeline has no '{MASTER_TRACK}' track")
        items = []
        t = 0.0
        for i, item in enumerate(track):
            dur = item.duration().to_seconds() if item.duration() else 0.0
            kind = "gap" if isinstance(item, ot_schema.Gap) else "?"
            text = None
            if not isinstance(item, ot_schema.Gap):
                raw = (
                    getattr(item, "metadata", {}).get("director_os", {})
                    if hasattr(item, "metadata")
                    else {}
                )
                kind = str(raw.get("kind", "?"))
                if kind == "narration":
                    text = str(raw.get("script_text", ""))[:120]
                src_in = src_out = None
                sr = getattr(item, "source_range", None)
                if sr is not None:
                    src_in = round(sr.start_time.to_seconds(), 3)
                    src_out = round(sr.end_time_exclusive().to_seconds(), 3)
                items.append(
                    {
                        "index": i,
                        "kind": kind,
                        "start_s": round(t, 3),
                        "dur_s": round(dur, 3),
                        "source_in_s": src_in,
                        "source_out_s": src_out,
                        "script": text,
                        "meta": raw,
                    }
                )
            else:
                items.append(
                    {"index": i, "kind": "gap", "start_s": round(t, 3), "dur_s": round(dur, 3)}
                )
            t += dur
        return items

    def agent_finish(summary: str) -> dict[str, Any]:
        marker = work_dir / "agent_finish.json"
        marker.write_text(
            json.dumps({"summary": summary.strip(), "finished_at": time.time()}, indent=1)
        )
        # the FULL summary rides the tool result — the audit log is the only
        # record some sessions leave (140 chars cut a real session's report)
        return {"detail": f"session complete — {summary.strip()[:1000]}"}

    registry.register(
        "project.describe",
        "Inventory the post-FX artifacts (master video, captions, QA, stats) "
        "with durations and cue counts. Call this FIRST to understand state.",
        _Empty,
        lambda: project_describe(),
    )
    registry.register(
        "qa.read_report",
        "Read the QA gate reports (qa_v4: duration, flash, blur, watermark, "
        "caption color-vs-speaker, ghost text, gap coverage).",
        _Empty,
        lambda: qa_read_report(),
    )
    registry.register(
        "transcript.read_window",
        "Dialogue events in [start_s, end_s] with role, text and MEASURED "
        "audio evidence (speech_ratio, rms_db, repeat_n). Window clamped to "
        "120 s. The primary tool for caption-truth judgments.",
        _Window,
        lambda **kw: transcript_read_window(**kw),
    )
    registry.register(
        "captions.list_cues",
        "Burned caption cues in [start_s, end_s]: style (Officer/Suspect/"
        "Narrator), text, gray *tag* cues included, plus cue coverage seconds.",
        _Window,
        lambda **kw: captions_list_cues(**kw),
    )
    registry.register(
        "audio.speech_spans",
        "VAD speech spans in [start_s, end_s] (cached compute, resumable). "
        "Cross-check against captions.list_cues: speech without a cue is an "
        "owner-law violation.",
        _Window,
        lambda **kw: audio_speech_spans(**kw),
    )
    registry.register(
        "captions.propose_tag_window",
        "File a caption-truth proposal: gray-italic tag (unaudible/mumbling/"
        "whispering/cross-talk/panicking) over [start_s, end_s] with an "
        "evidence-grounded reason. Proposals go to the owner for review.",
        _TagWindow,
        lambda **kw: propose_tag_window(**kw),
    )
    registry.register(
        "timeline.list_items",
        "List master-track items of the working timeline (index, kind raw/"
        "narration/gap, start, duration, source range, script text).",
        _Empty,
        lambda: timeline_list_items(),
    )
    registry.register(
        "timeline.get_item",
        "Full detail of ONE master-track item by index (source range, "
        "narration script, blur regions, audio mode) — read before mutating.",
        _Index,
        lambda **kw: timeline_get_item(**kw),
    )
    registry.register(
        "frame.grab",
        "Extract ONE frame of the working master at timestamp t_s as a JPEG "
        "(LRU-cached, cheap). Use it to LOOK at the exact frame an edit "
        "targets — frame-grounded decisions, not blind time-range guesses. "
        "Returns frame_path + frame_no (1-indexed NLE timecode).",
        _FrameGrab,
        lambda **kw: frame_grab(**kw),
    )
    registry.register(
        "agent.finish",
        "Terminal marker: write the session summary and stop the loop.",
        _Finish,
        lambda **kw: agent_finish(**kw),
    )
