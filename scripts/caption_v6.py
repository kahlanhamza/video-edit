#!/usr/bin/env python3
"""CAPTION v6 — rebuild caption truth on the MASTER clock (v5 regression fix).

The v5 burn pasted SOURCE-clock evidence onto the 657.7 s master. v6 restarts
from the shipped master audio itself:

  1. Gladia ASR + diarization on pipeline_run/v3/audio_v3.m4a  (asr_master_v6)
  2. VAD evidence per event on the same audio                   (evidence_v6)
  3. LLM supervisor pass (the operator, decisions_v6.json): roles, tags,
     word-locked corrections, word-run splits of mixed diarization
  4. THIS builder applies the committed caption_v4 build laws:
     - word-lock: corrected text re-aligns onto ORIGINAL word timestamps
       (difflib; unchanged words keep exact times) — sync cannot drift
     - rule-locked tags render gray-italic, never a speaker color
     - tag cues: VAD-extent trimmed, <=4 s chunks, 0.2 s gaps
     - cue law: <=5 words / <=27 chars / break on >1 s word gap
     - two-lane placement (lane 0 MarginV 46, lane 1 MarginV 96)
     - VAD gap-fill: uncovered speech >=0.45 s with voice evidence gets
       *unaudible* cues in free slots
     - accepted v3/v4 typography: DejaVu Sans 40 @ PlayRes 1280x720,
       Officer white / Suspect gold / Narrator cyan, karaoke \\k beats,
       \\fad(40,30)

Outputs: final_state/subtitles_v6.ass, final_state/captions_v6.srt,
         final_state/caption_v6_stats.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/z/my-project/scripts")

import numpy as np  # noqa: E402
import vad_evidence  # noqa: E402
from postfx_v3 import (  # noqa: E402
    CAPTION_FONTSIZE,
    CAPTION_MARGIN_V,
    FONT,
    _align_words,
    log,
)

WORK = Path("/home/z/my-project/pipeline_run/final_state")
V3 = Path("/home/z/my-project/pipeline_run/v3")
WAV = V3 / "audio_master_16k.wav"
ASS_V6 = WORK / "subtitles_v6.ass"
SRT_V6 = WORK / "captions_v6.srt"
STATS = WORK / "caption_v6_stats.json"

SR = vad_evidence.SR
FRAME = SR * vad_evidence.FRAME_MS // 1000

TAG_GRAY = "&H9C9C9C&"
RUN_MAX_S = 4.0
GAP_TAG_MAX_S = 3.5
GAP_MIN_S = 0.45
COVERAGE_PAD = 0.30

# committed v4 lane margins (events override the style MarginV)
LANE0_MV = 46
LANE1_MV = 96


def _fmt_ass(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _fmt_srt(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int(round((sec - int(sec)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _vad_extent(
    flags: np.ndarray, frame: int, s: float, e: float, cap: float
) -> tuple[float, float]:
    a = int(s * SR / frame)
    b = max(int(e * SR / frame), a + 1)
    seg = np.flatnonzero(flags[a:b])
    if len(seg):
        vs_t = (a + int(seg[0])) * frame / SR
        ve_t = (a + int(seg[-1]) + 1) * frame / SR
        s2 = max(s, min(vs_t, max(s, e - 0.5)))
        e2 = min(s2 + cap, max(ve_t, s2 + 0.5))
        return s2, max(e2, s2 + 0.5)
    return s, min(max(e, s + 0.5), s + 0.8)


def _tag_chunks(s: float, e: float, max_s: float) -> list[tuple[float, float]]:
    out = []
    cur = s
    while cur < e - 0.05:
        end = min(cur + max_s, e)
        out.append((cur, end))
        cur = end + 0.2
    return out or [(s, min(e, s + 0.5))]


def main() -> None:
    evs = json.loads((WORK / "caption_evidence_v6.json").read_text())
    tr = json.loads((WORK / "asr_master_v6.json").read_text())
    dec = json.loads((WORK / "decisions_v6.json").read_text())
    role_map = dec["role_map"]
    events_dec = dec["events"]

    pcm = vad_evidence.load_pcm(WAV)
    flags, _rms, frame = vad_evidence.vad_flags(pcm)

    consumed: set[int] = set()  # word indices already claimed (no double render)

    def words_in(a: float, b: float) -> list[dict]:
        # exclusive membership: a word renders exactly once (first claim wins);
        # diarization overlaps can never duplicate a word across events
        out = []
        for i, w in enumerate(tr["words"]):
            if i in consumed:
                continue
            if a - 0.05 <= w["s"] < b:
                out.append(w)
                consumed.add(i)
        return out

    # ------------------------------------------------ events -> runs
    tag_cues: list[dict] = []  # {s, e, style, tag}
    word_runs: list[dict] = []  # {s, e, style, words}
    n_dropped_words = 0

    for ev in evs:
        d = events_dec.get(str(ev["id"]))
        role_default = role_map.get(str(ev["speaker"]), "Officer")
        if d is None:
            word_runs.append(
                {
                    "s": ev["s"],
                    "e": ev["e"],
                    "style": role_default,
                    "words": words_in(ev["s"], ev["e"]),
                }
            )
            continue
        act = d.get("action")
        if act == "drop":
            n_dropped_words += ev.get("n_words", 0)
            continue
        if act == "tag":
            for cs, ce in _tag_chunks(ev["s"], ev["e"], RUN_MAX_S):
                if d.get("locked", "").startswith("owner"):
                    # owner-window law (committed v4): the tag covers the
                    # flagged span CONTINUOUSLY — the owner heard murmuring
                    # even where the cheap VAD hears silence. No VAD trim.
                    s2, e2 = cs, ce
                else:
                    s2, e2 = _vad_extent(flags, frame, cs, ce, RUN_MAX_S)
                tag_cues.append(
                    {"s": s2, "e": e2, "style": "Officer", "tag": d.get("tag", "unaudible")}
                )
            continue
        if act == "flip":
            word_runs.append(
                {
                    "s": ev["s"],
                    "e": ev["e"],
                    "style": d["role"],
                    "words": words_in(ev["s"], ev["e"]),
                }
            )
            continue
        if act == "correct":
            ws = words_in(ev["s"], ev["e"])
            fixed = _align_words(d["text"], ws)
            word_runs.append(
                {"s": ev["s"], "e": ev["e"], "style": role_default, "words": fixed or ws}
            )
            continue
        if act == "split":
            for run in d["runs"]:
                a, b, what = run[0], run[1], run[2]
                if what.startswith("TAG:"):
                    for cs, ce in _tag_chunks(a, b, RUN_MAX_S):
                        if ":owner" in what or d.get("locked", "").startswith("owner"):
                            # owner window: continuous coverage, no VAD trim
                            s2, e2 = cs, ce
                        else:
                            s2, e2 = _vad_extent(flags, frame, cs, ce, RUN_MAX_S)
                        tag_cues.append(
                            {"s": s2, "e": e2, "style": "Officer", "tag": what[4:].split(":")[0]}
                        )
                    continue
                if what == "DROP":
                    n_dropped_words += len(words_in(a, b))
                    continue
                ws = words_in(a, b)
                if not ws:
                    continue
                corr = (d.get("corrections") or {}).get(f"{a:.2f}-{b:.2f}")
                if corr:
                    ws = _align_words(corr, ws) or ws
                word_runs.append(
                    {"s": ws[0]["s"], "e": max(w["e"] for w in ws), "style": what, "words": ws}
                )
            continue
        word_runs.append(
            {
                "s": ev["s"],
                "e": ev["e"],
                "style": role_default,
                "words": words_in(ev["s"], ev["e"]),
            }
        )

    # merge adjacent same-tag tag_cues (<=1.0 s apart) then re-chunk
    tag_cues.sort(key=lambda t: t["s"])
    merged: list[dict] = []
    for t in tag_cues:
        if merged and t["tag"] == merged[-1]["tag"] and t["s"] - merged[-1]["e"] < 1.0:
            merged[-1]["e"] = max(merged[-1]["e"], t["e"])
        else:
            merged.append(dict(t))
    tag_cues = []
    for m in merged:
        for cs, ce in _tag_chunks(m["s"], m["e"], RUN_MAX_S):
            tag_cues.append({"s": cs, "e": ce, "style": "Officer", "tag": m["tag"]})

    # ------------------------------------------------ chunk word runs
    raw_chunks = []
    for r in word_runs:
        style = "Narrator" if r["style"].lower().startswith("narr") else r["style"].capitalize()
        cur = []
        for w in sorted(r["words"], key=lambda x: x["s"]):
            if cur and w["s"] - cur[-1]["e"] > 1.0:
                raw_chunks.append((style, cur))
                cur = []
            cur.append(w)
            if (
                len(cur) >= 5
                or sum(len(x["w"]) + 1 for x in cur) >= 27
                or (cur[-1]["w"][-1] in ",.?!" and len(cur) >= 3)
            ):
                raw_chunks.append((style, cur))
                cur = []
        if cur:
            raw_chunks.append((style, cur))

    # ------------------------------------------------ two-lane placement
    cand = [
        {
            "s": t["s"],
            "e": t["e"],
            "style": t["style"],
            "mv": LANE0_MV,
            "c": None,
            "tag": f"*{t['tag']}*",
        }
        for t in tag_cues
    ]
    cand += [
        {
            "s": c[0]["s"],
            "e": max(c[-1]["e"] + 0.06, c[0]["s"] + 0.30),
            "style": st,
            "mv": LANE0_MV,
            "c": c,
            "tag": None,
        }
        for st, c in raw_chunks
    ]
    cand.sort(key=lambda x: x["s"])

    placed, last_idx = [], [None, None]
    for it in cand:
        s, e, st = it["s"], it["e"], it["style"]
        l0 = placed[last_idx[0]] if last_idx[0] is not None else None
        l1 = placed[last_idx[1]] if last_idx[1] is not None else None
        if l0 and s < l0["e"] - 0.05 and l0["style"] != st:
            lane, mv = 1, LANE1_MV
            if l1 and s < l1["e"] + 0.02:
                if s - l1["s"] >= 0.25:
                    l1["e"] = s - 0.02
                else:
                    s = l1["s"] + 0.25
                    e = max(e, s + 0.25)
        else:
            lane, mv = 0, LANE0_MV
            if l0 and s < l0["e"] + 0.02:
                if s - l0["s"] >= 0.25:
                    l0["e"] = s - 0.02
                else:
                    s = l0["s"] + 0.25
                    e = max(e, s + 0.25)
        it2 = dict(it)
        it2.update({"s": s, "e": e, "mv": mv})
        placed.append(it2)
        last_idx[lane] = len(placed) - 1

    # ------------------------------------------------ VAD gap-fill
    spans = vad_evidence.speech_spans(flags, frame, min_s=GAP_MIN_S, join_gap=0.30)
    gap_fills = []
    for a, b in spans:
        if (
            b < 2.0
            or vad_evidence.voice_band_ratio(pcm, a, b) < 0.45
            or vad_evidence.segment_rms_db(pcm, a, b) < -42
        ):
            continue
        ivals = sorted(
            (it["s"] - COVERAGE_PAD, it["e"] + COVERAGE_PAD)
            for it in placed
            if it["e"] + COVERAGE_PAD > a and it["s"] - COVERAGE_PAD < b
        )
        cur, free = a, []
        for ps, pe in ivals:
            if pe <= cur:
                continue
            if ps > cur:
                free.append((cur, min(ps, b)))
            cur = max(cur, pe)
            if cur >= b:
                break
        if cur < b:
            free.append((cur, b))
        for fa, fb in free:
            if fb - fa < GAP_MIN_S:
                continue
            for cs, ce in _tag_chunks(fa, fb, GAP_TAG_MAX_S):
                gap_fills.append({"s": cs, "e": ce})
    inserted = 0
    for g in gap_fills:
        if any(g["s"] < it["e"] - 0.05 and g["e"] > it["s"] + 0.05 for it in placed):
            continue
        placed.append(
            {
                "s": g["s"],
                "e": g["e"],
                "style": "Officer",
                "mv": LANE0_MV,
                "c": None,
                "tag": "*unaudible*",
            }
        )
        inserted += 1
    placed.sort(key=lambda x: x["s"])
    log(f"gap-fill: {len(spans)} VAD speech spans -> {inserted} *unaudible* cues")

    # flash guard: a visible cue shorter than 0.22 s is a subliminal flash
    # (placement truncation of overlapping tag/word cues) — drop it
    n_flash = 0
    kept = []
    for it in placed:
        if it["e"] - it["s"] < 0.22:
            n_flash += 1
            continue
        kept.append(it)
    placed = kept
    if n_flash:
        log(f"flash guard: dropped {n_flash} cues shorter than 0.22 s")

    # ------------------------------------------------ emit ASS + SRT
    ass_header = f"""[Script Info]
; Caption v6 — master-clock truth (Gladia master ASR + LLM supervisor pass,
; committed caption_v4 laws: word-lock, rule-locked tags, VAD gap-fill,
; accepted v3 typography). v5 source-clock regression fixed.
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Officer,{FONT},{CAPTION_FONTSIZE},&H00FFFFFF,&H00CCC0B8,&H00000000,&H90000000,-1,0,0,0,100,100,0.4,0,1,2.9,1.6,2,80,80,{CAPTION_MARGIN_V},1
Style: Suspect,{FONT},{CAPTION_FONTSIZE},&H004DD8FF,&H004D9BB3,&H00000000,&H90000000,-1,0,0,0,100,100,0.4,0,1,2.9,1.6,2,80,80,{CAPTION_MARGIN_V},1
Style: Narrator,{FONT},{CAPTION_FONTSIZE},&H00FFE470,&H00B8A37C,&H00000000,&H90000000,-1,0,0,0,100,100,0.4,0,1,2.9,1.6,2,80,80,{CAPTION_MARGIN_V},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ass_lines, srt_lines = [], []
    for i, it in enumerate(placed, 1):
        s, e, st, mv = it["s"], it["e"], it["style"], it["mv"]
        if it["tag"]:
            body = f"{{\\i1\\c{TAG_GRAY}\\2c{TAG_GRAY}\\i0}}{it['tag']}"
            plain = it["tag"]
        else:
            k_parts = []
            c = it["c"]
            for wi, w in enumerate(c):
                ws = max(s, w["s"])
                nxt = min(e, c[wi + 1]["s"]) if wi + 1 < len(c) else e
                dur_cs = max(6, int(round((nxt - ws) * 100)))
                k_parts.append(f"{{\\k{dur_cs}}}{w['w']}")
            body = " ".join(k_parts)
            plain = " ".join(w["w"] for w in c)
        ass_lines.append(
            f"Dialogue: 0,{_fmt_ass(s)},{_fmt_ass(e)},{st},,0,0,{mv},,{{\\fad(40,30)}}{body}"
        )
        srt_lines.append(f"{i}\n{_fmt_srt(s)} --> {_fmt_srt(e)}\n{plain}\n")

    ASS_V6.write_text(ass_header + "\n".join(ass_lines) + "\n", encoding="utf-8")
    SRT_V6.write_text("\n".join(srt_lines), encoding="utf-8")

    styles = {}
    for it in placed:
        styles[it["style"]] = styles.get(it["style"], 0) + 1
    n_tags = sum(1 for it in placed if it["tag"])
    STATS.write_text(
        json.dumps(
            {
                "cues": len(placed),
                "tags": n_tags,
                "gap_fills": inserted,
                "dropped_words": n_dropped_words,
                "styles": styles,
                "clock": "master (Gladia on audio_v3.m4a)",
            },
            indent=1,
        )
    )
    log(
        f"captions v6: {len(placed)} cues -> {styles} ({n_tags} gray tags, "
        f"{inserted} gap-fills, {n_dropped_words} fragment words dropped)"
    )


if __name__ == "__main__":
    main()
