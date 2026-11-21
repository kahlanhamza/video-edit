#!/usr/bin/env python3
"""CAPTION v4 — evidence-driven caption truth (owner findings 2026-11-21).

Owner findings -> root cause (MEASURED) -> fix -> gate:
  A. "the officer talks but the caption is in yellow" (color lies).
     MEASURED: the 609.4-628.9 s block carries the same text repeated 3x
     ("people want to stop, stop. You know, ...") at VAD speech_ratio
     0.00-0.43 and rms -28..-40 dB — an ASR loop on barely-audible speech.
     The words are fabricated, so ANY speaker color there is a lie.
     FIX: neutral gray-italic tag cues for every unclear-voice span (no
     speaker color when the voice is not clear), owner-confirmed window
     override, plus an LLM role-flow audit that may flip roles ONLY on
     clearly-audible lines (speech_ratio >= 0.5), capped at 15%, every flip
     logged with evidence in decisions_v4.json.
  B. hallucinated captions survive ("people want to stop" x4, 7.8 s for 4
     words). FIX: rule-locked verdicts from measured audio: speech_ratio
     < 0.15 (fabrication on silence) or repeat_n >= 3 at speech_ratio < 0.5
     (ASR loop) FORCES a tag; the LLM only picks the tag word
     (unaudible/mumbling/whispering/cross-talk/panicking) and may correct
     text only where the voice evidence allows.
  C. "if a frame has no caption but has audio you must handle this".
     MEASURED: v3 leaves 24 VAD speech spans (22.7 s) caption-free, e.g.
     342.7-348.7 s — 6 s of REAL speech (sr 0.99) inside ev155's stretched
     "I had" (13.9 s for 2 words).
     FIX: VAD gap-fill — every uncovered speech span >= 0.45 s with voice
     evidence (voice_band >= 0.45, rms >= -42) gets *unaudible* gray-italic
     cue(s), chunked <= 3.5 s, placed only in free caption slots.
  D. word-lock law kept (sync cannot drift): corrections re-align onto the
     ORIGINAL word timestamps (difflib; unchanged words keep exact measured
     times). Cue chunking, two-lane placement, typography = the accepted v3
     look; tag cues add \\2c so the karaoke sweep is color-flat on long tags.

All v3 artifacts stay untouched (rollback safe): v4 writes subtitles_v4.ass,
captions_v4.srt, burn_v4_*, master_v4_*, master_postfx_v4.mp4, qa_v4.json.

Phases (each resumable, one process per call — sandbox kills long runs):
  decide | captions | burn N | burnconcat | mux | ship
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/z/my-project/scripts")
sys.path.insert(0, "/home/z/my-project/scripts")

import vad_evidence
from postfx_v3 import (
    BURN_CHUNK_S,
    CAPTION_FONTSIZE,
    CAPTION_MARGIN_V,
    CASE_FACTS,
    FONT,
    SHIP_PACK,
    V3,
    WORK,
    _align_words,
    _llm_json_array,
    _sh,
    log,
    probe_dur,
)
from vad_evidence import SR

V4 = V3  # artifacts live beside v3 (names carry the _v4 suffix)
DECISIONS = V4 / "decisions_v4.json"
ASS_V4 = V4 / "subtitles_v4.ass"
SRT_V4 = V4 / "captions_v4.srt"

# Owner-confirmed windows (reviewed by the owner on the accepted cut): spans
# where the voice is NOT audible enough for captions, and the ASR words are
# fabricated. Any-case pipeline: extend this table per case after review.
OWNER_WINDOWS = [
    {
        "start": 609.40,
        "end": 628.90,
        "tag": "unaudible",
        "note": "officer murmurs, not audible; ASR loop 'people want to stop'",
    },
]

TAG_WORDS = ("unaudible", "mumbling", "whispering", "cross-talk", "panicking")
TAG_GRAY = "&H9C9C9C&"  # BGR of the agreed dim gray, same as v3
RUN_MAX_S = 4.0  # merged tag run chunk length
GAP_TAG_MAX_S = 3.5  # gap-fill tag chunk length
GAP_MIN_S = 0.45  # minimum uncovered speech span worth a cue
COVERAGE_PAD = 0.30  # clearance required around existing cues
FLIP_CAP = 0.15  # max share of dialogue events to re-color


# ----------------------------------------------------------------- evidence
def _load_evidence() -> list[dict]:
    evs = json.loads((V4 / "caption_evidence.json").read_text())
    state = json.loads((WORK / "asm_state.json").read_text())
    words_by_id = {}
    for i, ev in enumerate(state["events"]):
        words_by_id[i] = ev.get("words") or []
    for ev in evs:
        ev["words"] = words_by_id.get(ev["id"], [])
    return evs


def _in_owner_window(ev: dict) -> dict | None:
    for w in OWNER_WINDOWS:
        if ev["s"] < w["end"] and ev["e"] > w["start"]:
            return w
    return None


# -------------------------------------------------------------------- decide
def _rule_constraint(ev: dict) -> str | None:
    """Measured-audio verdicts that outrank the LLM. None = LLM decides."""
    if _in_owner_window(ev):
        return "must_tag:owner_window"
    if ev["speech_ratio"] < 0.15:
        return "must_tag:silence"  # words on (near-)silence = fabricated
    if ev["repeat_n"] >= 3 and ev["speech_ratio"] < 0.5:
        return "must_tag:loop"  # ASR repetition loop
    return None


def phase_decide() -> None:
    evs = _load_evidence()
    dialogue = [e for e in evs if e.get("has_words") and not e["role"].lower().startswith("narr")]

    # v3 proofread text is reused as the LLM's starting point (it already
    # fixed names/units); here we judge FIT + truthfulness, not spelling.
    v3_fix = {}
    for f in json.loads((V4 / "proofread_v3.json").read_text()):
        v3_fix[f["id"]] = (f.get("clean_text") or "").strip()

    decisions: dict[int, dict] = {}
    if DECISIONS.exists():
        for d in json.loads(DECISIONS.read_text()):
            decisions[d["id"]] = d
    todo = [
        e
        for e in dialogue
        if e["id"] not in decisions or decisions[e["id"]].get("batch_done") is not True
    ]
    log(f"decide: {len(dialogue)} dialogue events, {len(todo)} to judge")

    # narrator events pass through untouched (our own script audio)
    for e in evs:
        if e.get("has_words") and e["role"].lower().startswith("narr") and e["id"] not in decisions:
            decisions[e["id"]] = {
                "id": e["id"],
                "action": "keep",
                "locked": "narrator",
                "batch_done": True,
            }
    dlg_by_id = {e["id"]: e for e in dialogue}
    order = [e["id"] for e in dialogue]

    B = 24
    nb = math.ceil(len(todo) / B)
    for bi, i in enumerate(range(0, len(todo), B), 1):
        batch = todo[i : i + B]
        lines = []
        for _k, e in enumerate(batch):
            prev_d = next_d = {"role": "-", "text": "-"}
            for j in range(order.index(e["id"]) - 1, -1, -1):
                if order[j] in dlg_by_id:
                    prev_d = {
                        "role": dlg_by_id[order[j]]["role"],
                        "text": dlg_by_id[order[j]]["text"][:70],
                    }
                    break
            for j in range(order.index(e["id"]) + 1, len(order)):
                if order[j] in dlg_by_id:
                    next_d = {
                        "role": dlg_by_id[order[j]]["role"],
                        "text": dlg_by_id[order[j]]["text"][:70],
                    }
                    break
            lines.append(
                json.dumps(
                    {
                        "id": e["id"],
                        "role": e["role"],
                        "t": round(e["s"], 1),
                        "text": (v3_fix.get(e["id"]) or e["text"]),
                        "prev": prev_d,
                        "next": next_d,
                        "evidence": {
                            "speech_ratio": e["speech_ratio"],
                            "rms_db": e["rms_db"],
                            "repeat_n": e["repeat_n"],
                            "stretch_s_per_word": e["stretch"],
                        },
                        "constraint": _rule_constraint(e) or "free",
                    },
                    ensure_ascii=False,
                )
            )
        prompt = f"""You are the caption supervisor of a police bodycam documentary.
Case: {json.dumps(CASE_FACTS, ensure_ascii=False)}

For EVERY input line judge what the caption must show, using the MEASURED audio evidence:
- evidence.speech_ratio = fraction of the line's time window where a voice-activity
  detector hears voiced speech (0..1). Below 0.15 there is effectively NO voiced
  speech: the ASR words are fabricated on noise -> constraint is "must_tag".
- evidence.repeat_n = how many times this exact text occurs within +-45 s.
  repeat_n >= 3 with speech_ratio < 0.5 is an ASR loop hallucination -> "must_tag".
- evidence.rms_db = loudness; stretch_s_per_word >> 0.6 means the ASR stretched
  a few words over a long window (ghost words).

Return ONLY a valid JSON array, one object per input id, no prose, no fences:
[{{"id": <int>, "action": "keep"|"correct"|"tag", "clean_text": "<text>",
   "tag": null or "unaudible"|"mumbling"|"whispering"|"cross-talk"|"panicking",
   "role_ok": true|false, "flip_to": null or "Officer"|"Suspect",
   "flip_reason": "<only when flip_to set, max 12 words>"}}]

Rules:
- constraint "must_tag:...": you MUST return action "tag" and pick the best tag
  word. Tags render as italic gray text with NO speaker color, so the audience
  is never told a wrong speaker. "unaudible" = voice present but unclear;
  "mumbling" = trailing off; "whispering"; "cross-talk" = two people overlap;
  "panicking" = clear distress. For must_tag you may NOT return text.
- constraint "free" AND speech_ratio >= 0.35: fix obvious ASR mishearings
  (names, agencies, ten-codes, units like "0.391 g/210L", "SFST", "EMS") using
  the case facts and the prev/next context. Keep the speaker's words and
  dialect; do NOT paraphrase, do NOT add or remove meaning; keep a similar
  length. If the line is pure garble a viewer could not make sense of, return
  action "tag" and choose the best tag word instead of inventing text.
- constraint "free" AND speech_ratio < 0.35: prefer "tag" unless the words are
  simple and plausible (yes/no/short answers) AND the evidence does not
  contradict them.
- Mask harsh profanity for YouTube monetization ("f***", "s***").
- role_ok: does the assigned role fit what the words say? "Officer" = the
  arrest team, dispatch, EMS (questions, orders, radio codes, Miranda,
  echo-confirmations of the suspect's answers). "Suspect" = Ofc. Oscar
  Mayorga, the impaired officer being arrested (excuses, begging, refusing).
  If the role is wrong for a CLEARLY audible line (speech_ratio >= 0.5), set
  flip_to and a short flip_reason. Never flip faint/garbled lines.
- Never include timestamps. Never invent content.
Lines:
{chr(10).join(lines)}"""
        arr = _llm_json_array(prompt)
        by_id = {int(x["id"]): x for x in arr if isinstance(x, dict) and "id" in x}
        for e in batch:
            fx = by_id.get(e["id"], {})
            constraint = _rule_constraint(e)
            action = (fx.get("action") or "").strip().lower()
            clean = (fx.get("clean_text") or "").strip()
            tag = (fx.get("tag") or "").strip().lower()
            if tag not in TAG_WORDS:
                tag = None
            # enforcement: measured rules outrank the LLM
            if constraint:
                d = {
                    "id": e["id"],
                    "action": "tag",
                    "tag": tag or "unaudible",
                    "locked": constraint,
                    "role": e["role"],
                    "batch_done": True,
                }
            else:
                if action not in ("keep", "correct", "tag"):
                    action = "keep"
                if action == "correct" and not clean:
                    action = "tag"
                if action == "tag" and not tag:
                    tag = "unaudible"
                d = {
                    "id": e["id"],
                    "action": action,
                    "clean_text": clean if action == "correct" else "",
                    "tag": tag if action == "tag" else None,
                    "locked": "llm",
                    "role": e["role"],
                    "batch_done": True,
                }
            # role audit: flips only on clearly audible speech, with a reason
            flip_to = (fx.get("flip_to") or "").strip().capitalize()
            if (
                not constraint
                and flip_to in ("Officer", "Suspect")
                and flip_to != e["role"]
                and e["speech_ratio"] >= 0.5
                and (fx.get("flip_reason") or "").strip()
            ):
                d["role"] = flip_to
                d["flip_reason"] = fx["flip_reason"].strip()[:120]
            decisions[e["id"]] = d
        DECISIONS.write_text(json.dumps([decisions[k] for k in sorted(decisions)], indent=1))
        log(f"  decide batch {bi}/{nb} done ({len(decisions)} total)")

    # global flip guard: too many flips = systematic error -> drop all flips
    flips = [d for d in decisions.values() if d.get("flip_reason")]
    if len(flips) > FLIP_CAP * len(dialogue):
        log(f"FLIP GUARD: {len(flips)} flips > {FLIP_CAP:.0%} cap — dropping all")
        for d in flips:
            orig = next(e["role"] for e in dialogue if e["id"] == d["id"])
            d["role"] = orig
            d["dropped_flip"] = d.pop("flip_reason", "")
        DECISIONS.write_text(json.dumps([decisions[k] for k in sorted(decisions)], indent=1))
    n_act = {"keep": 0, "correct": 0, "tag": 0}
    for d in decisions.values():
        if d.get("batch_done") and d["locked"] != "narrator":
            n_act[d["action"]] = n_act.get(d["action"], 0) + 1
    log(
        f"decide: keep={n_act['keep']} correct={n_act['correct']} "
        f"tag={n_act['tag']} flips={len([d for d in decisions.values() if d.get('flip_reason')])}"
    )


# ------------------------------------------------------------------ captions
def _vad() -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    pcm = vad_evidence.load_pcm(V4 / "audio_src_16k.wav")
    flags, rms, frame = vad_evidence.vad_flags(pcm)
    return pcm, flags, frame, rms


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
    """Trim [s, e] to the VAD-positive extent, clamped to [0.5, cap] s.

    The tag starts at the first voiced frame (so it sits on the actual
    mumble, not on silence) and never runs longer than cap seconds."""
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
    """Split [s, e] into <=max_s chunks separated by 0.2 s."""
    out = []
    cur = s
    while cur < e - 0.05:
        end = min(cur + max_s, e)
        out.append((cur, end))
        cur = end + 0.2
    return out or [(s, min(e, s + 0.5))]


def phase_captions() -> None:
    evs = _load_evidence()
    decisions = {d["id"]: d for d in json.loads(DECISIONS.read_text())}
    pcm, flags, frame, rms = _vad()
    # v3-accepted text (no-op corrections must fall back to keep, not tag)
    v3_fix = {
        f["id"]: (f.get("clean_text") or "").strip()
        for f in json.loads((V4 / "proofread_v3.json").read_text())
    }

    corrected_events = []
    for ev in evs:
        if not ev.get("has_words"):
            continue
        d = decisions.get(ev["id"], {"action": "keep", "role": ev["role"]})
        role = d.get("role") or ev["role"]
        # build-time truth guards (measured audio outranks any LLM verdict):
        #  - clearly audible speech (sr >= 0.8) must keep a real caption
        #    EXCEPT inside owner-confirmed windows (owner review outranks VAD)
        #  - words may only be CORRECTED when speech is actually present
        #    (sr >= 0.35) — never invented over faint audio
        orig_text = " ".join(w["w"] for w in ev["words"]).strip()
        accepted_text = (v3_fix.get(ev["id"]) or orig_text).strip()
        if d["action"] == "correct" and d.get("clean_text", "").strip() in (
            accepted_text,
            orig_text,
        ):
            d = {**d, "action": "keep"}  # no-op correction
        elif d["action"] == "tag" and ev["speech_ratio"] >= 0.8 and not _in_owner_window(ev):
            log(
                f"  guard: ev{ev['id']} tag downgraded to keep "
                f"(sr={ev['speech_ratio']:.2f} clearly audible)"
            )
            d = {**d, "action": "keep", "downgraded": "clear-speech-tag"}
        elif d["action"] == "correct" and ev["speech_ratio"] < 0.35:
            log(
                f"  guard: ev{ev['id']} correct downgraded to tag "
                f"(sr={ev['speech_ratio']:.2f} too faint to invent words)"
            )
            d = {
                **d,
                "action": "tag",
                "tag": d.get("tag") or "unaudible",
                "downgraded": "faint-speech-correct",
            }
        if d["action"] == "tag":
            win = _in_owner_window(ev)
            if win:
                # owner review outranks the VAD inside confirmed windows:
                # the tag covers the flagged span CONTINUOUSLY (the owner
                # heard murmuring even where the cheap VAD hears silence)
                s2 = max(ev["s"], win["start"])
                e2 = min(ev["e"], win["end"])
                if e2 - s2 < 0.5:
                    e2 = min(max(e2, s2 + 0.5), win["end"])
            else:
                s2, e2 = _vad_extent(flags, frame, ev["s"], ev["e"], RUN_MAX_S)
            corrected_events.append(
                {
                    "role": role,
                    "tag": win["tag"] if win else (d.get("tag") or "unaudible"),
                    "s": s2,
                    "e": e2,
                    "id": ev["id"],
                }
            )
        elif d["action"] == "correct" and d.get("clean_text"):
            new_words = _align_words(d["clean_text"], ev["words"])
            corrected_events.append(
                {"role": role, "words": new_words or ev["words"], "tag": None, "id": ev["id"]}
            )
        else:
            # keep — but keep the ACCEPTED v3 proofread text, not raw ASR:
            # the accepted cut shipped the proofread wording, so v4 must
            # inherit it verbatim (word-locked onto the same timestamps)
            accepted = v3_fix.get(ev["id"]) or orig_text
            accepted_words = _align_words(accepted, ev["words"]) or ev["words"]
            corrected_events.append(
                {"role": role, "words": accepted_words, "tag": None, "id": ev["id"]}
            )

    # ---- merge adjacent same-word tag events into runs, then chunk <=4 s
    tag_runs = []
    for ce in corrected_events:
        if ce.get("tag"):
            if tag_runs and ce["tag"] == tag_runs[-1]["tag"] and ce["s"] - tag_runs[-1]["e"] < 1.0:
                tag_runs[-1]["e"] = max(tag_runs[-1]["e"], ce["e"])
                tag_runs[-1]["ids"].append(ce["id"])
            else:
                tag_runs.append(
                    {
                        "tag": ce["tag"],
                        "s": ce["s"],
                        "e": ce["e"],
                        "role": ce["role"],
                        "ids": [ce["id"]],
                    }
                )
    tag_cues = []
    for run in tag_runs:
        for cs, ce in _tag_chunks(run["s"], run["e"], RUN_MAX_S):
            tag_cues.append(
                {"style": run["role"].capitalize(), "tag": run["tag"], "s": cs, "e": ce}
            )

    # ---- chunk non-tag events into cues (v3 law: <=5 words / <=27 chars)
    raw_chunks = []
    for ce in corrected_events:
        if ce.get("tag"):
            continue
        style = "Narrator" if ce["role"].lower().startswith("narr") else ce["role"].capitalize()
        cur = []
        for w in ce["words"]:
            if cur and w["s"] - cur[-1]["e"] > 1.0:
                raw_chunks.append((style, cur))
                cur = []
            cur.append(w)
            if (
                len(cur) >= 5
                or sum(len(x["w"]) + 1 for x in cur) >= 27
                or (w["w"][-1] in ",.?!" and len(cur) >= 3)
            ):
                raw_chunks.append((style, cur))
                cur = []
        if cur:
            raw_chunks.append((style, cur))

    # ---- two-lane overlap placement (v3 law) over event cues + tag cues
    cand = [(tc["s"], tc["e"], tc["style"], {"w": f"*{tc['tag']}*"}, None) for tc in tag_cues]
    cand += [
        (c[0]["s"], max(c[-1]["e"] + 0.06, c[0]["s"] + 0.30), style, None, c)
        for style, c in raw_chunks
    ]
    cand.sort(key=lambda x: x[0])

    placed, last_idx = [], [None, None]
    for s, e, style, tagw, c in cand:
        l0 = placed[last_idx[0]] if last_idx[0] is not None else None
        l1 = placed[last_idx[1]] if last_idx[1] is not None else None
        if l0 and s < l0["e"] - 0.05 and l0["style"] != style:
            lane, mv = 1, 96
            if l1 and s < l1["e"] + 0.02:
                if s - l1["s"] >= 0.25:
                    l1["e"] = s - 0.02
                else:
                    s = l1["s"] + 0.25
                    e = max(e, s + 0.25)
        else:
            lane, mv = 0, 46
            if l0 and s < l0["e"] + 0.02:
                if s - l0["s"] >= 0.25:
                    l0["e"] = s - 0.02
                else:
                    s = l0["s"] + 0.25
                    e = max(e, s + 0.25)
        placed.append(
            {"s": s, "e": e, "style": style, "mv": mv, "c": c, "tag": tagw["w"] if tagw else None}
        )
        last_idx[lane] = len(placed) - 1

    # ---- VAD gap-fill: speech with NO caption coverage -> *unaudible* cues
    gap_fills = []
    spans = vad_evidence.speech_spans(flags, frame, min_s=GAP_MIN_S, join_gap=0.30)
    for a, b in spans:
        if (
            b < 2.0
            or vad_evidence.voice_band_ratio(pcm, a, b) < 0.45
            or vad_evidence.segment_rms_db(pcm, a, b) < -42
        ):
            continue
        # subtract every placed cue (+clearance) from the speech span
        ivals = sorted(
            (it["s"] - COVERAGE_PAD, it["e"] + COVERAGE_PAD)
            for it in placed
            if it["e"] + COVERAGE_PAD > a and it["s"] - COVERAGE_PAD < b
        )
        cur = a
        free: list[tuple[float, float]] = []
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
    # insert only where no placed cue overlaps (lane-0 free slots)
    inserted = 0
    for g in gap_fills:
        if any(g["s"] < it["e"] - 0.05 and g["e"] > it["s"] + 0.05 for it in placed):
            continue
        placed.append(
            {
                "s": g["s"],
                "e": g["e"],
                "style": "Officer",
                "mv": 46,
                "c": None,
                "tag": "*unaudible*",
            }
        )
        inserted += 1
    placed.sort(key=lambda x: x["s"])
    log(f"gap-fill: {len(spans)} VAD speech spans -> {inserted} *unaudible* cues")

    # ---- emit ASS + SRT (v3 typography and colors; tags gray-flat)
    ass_header = f"""[Script Info]
; Post-FX v4 captions — evidence-gated, word-locked, hallucination-free
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
            plain = it["tag"]  # keep the asterisks: *unaudible* is the agreed tag notation
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
    ASS_V4.write_text(ass_header + "\n".join(ass_lines) + "\n", encoding="utf-8")
    SRT_V4.write_text("\n".join(srt_lines), encoding="utf-8")

    styles = {}
    for it in placed:
        styles[it["style"]] = styles.get(it["style"], 0) + 1
    n_tags = sum(1 for it in placed if it["tag"])
    log(f"captions v4: {len(placed)} cues -> {styles} ({n_tags} gray tags)")
    with open(V4 / "caption_v4_stats.json", "w") as fh:
        json.dump(
            {"cues": len(placed), "tags": n_tags, "gap_fills": inserted, "tag_runs": len(tag_runs)},
            fh,
            indent=1,
        )


# --------------------------------------------------------------------- burn
def _chunk_ass_v4(ci: int) -> Path:
    """v4 chunk ASS with the v3 flash law (drop fully-outside cues)."""
    text = ASS_V4.read_text(encoding="utf-8")
    t0 = ci * BURN_CHUNK_S
    dur = BURN_CHUNK_S
    head = text.split("[Events]")[0]
    out_lines = [
        head + "[Events]\nFormat: Layer, Start, End, Style, Name, "
        "MarginL, MarginR, MarginV, Effect, Text\n"
    ]

    def to_sec(ts: str) -> float:
        h, m, s = ts.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    def fmt(sec: float) -> str:
        sec = max(0.0, sec)
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = sec % 60
        return f"{h}:{m:02d}:{s:05.2f}"

    n_kept = n_drop = 0
    for ln in text.splitlines():
        if not ln.startswith("Dialogue: "):
            continue
        parts = ln.split(",", 9)
        s_sec, e_sec = to_sec(parts[1]), to_sec(parts[2])
        if e_sec <= t0 + 0.005 or s_sec >= t0 + dur:
            n_drop += 1
            continue
        local_s = max(0.0, s_sec - t0)
        local_e = min(dur + 0.4, e_sec - t0)
        if local_e - local_s < 0.05:
            n_drop += 1
            continue
        out_lines.append(",".join([parts[0], fmt(local_s), fmt(local_e)] + parts[3:]))
        n_kept += 1
    path = V4 / f"captions_v4_chunk_{ci:02d}.ass"
    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    log(f"chunk ASS {ci}: kept {n_kept}, dropped {n_drop} (flash law)")
    return path


def phase_burn(ci: int) -> None:
    """Burn v4 captions + the ACCEPTED cards + watermark on chunk ci."""
    src = V4 / "master_v3_blurred.mp4"
    info_d = probe_dur(src)
    n_chunks = max(1, -(-int(info_d * 1000) // int(BURN_CHUNK_S * 1000)))
    if not 0 <= ci < n_chunks:
        raise SystemExit(f"burn chunk {ci} out of range 0..{n_chunks - 1}")
    part = V4 / f"burn_v4_{ci:02d}.mp4"
    mark = V4 / f"burn_v4_{ci:02d}.done"
    if mark.exists() and part.exists():
        log(f"burn {ci} cached")
        return
    t0 = ci * BURN_CHUNK_S
    dur = min(BURN_CHUNK_S, info_d - t0)
    ass = _chunk_ass_v4(ci)
    esc = str(ass).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")

    wm = json.loads((V4 / "wm_spans_v3.json").read_text())["spans"]
    blur_fc = ""
    cur_v = "[0:v]"
    if wm:
        arr = np.array([sp["box"] for sp in wm])
        bx, by, bw, bh = [int(v) for v in np.median(arr, axis=0)]
        bx, by = max(0, bx - 2), max(0, by - 2)
        bw = min(1280 - bx, bw + 4)
        bh = min(720 - by, bh + 4)
        blur_fc += (
            f"{cur_v}split=2[bgw][wmw];"
            f"[wmw]crop={bw}:{bh}:{bx}:{by},gblur=sigma=20[wmb];"
            f"[bgw][wmb]overlay={bx}:{by}:shortest=1[v0];"
        )
        cur_v = "[v0]"

    cards_meta = json.loads((V4 / "cards_meta_v3.json").read_text())
    card_times = cards_meta["card_times"]
    fc = blur_fc + f"{cur_v}ass='{esc}'[subbed];"
    prev = "[subbed]"
    inputs = []
    for png, g_t0 in card_times:
        lt0 = g_t0 - t0
        t_end = min(lt0 + 6.5, dur - 0.2)
        if t_end <= 0.5 or lt0 >= dur:
            continue
        idx = len(inputs) + 1
        inputs.append((V4 / png, idx))
        fc += (
            f"[{idx}:v]format=rgba[c{idx}];"
            f"{prev}[c{idx}]overlay=0:0:eof_action=repeat:"
            f"enable='between(t,{max(0.0, lt0):.2f},{t_end:.2f})'[v{idx}];"
        )
        prev = f"[v{idx}]"
    fc += f"{prev}format=yuv420p[vout]"

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{t0:.3f}",
        "-t",
        f"{dur:.3f}",
        "-i",
        str(src),
    ]
    for png, _i in inputs:
        cmd += ["-i", str(png)]
    cmd += [
        "-filter_complex",
        fc,
        "-map",
        "[vout]",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "superfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-t",
        f"{dur:.3f}",
        str(part),
    ]
    _sh(cmd, f"burn v4 chunk {ci}")
    mark.write_text("ok")
    log(f"burn v4 chunk {ci + 1}/{n_chunks} done ({dur:.1f}s, {len(inputs)} cards)")


def phase_burnconcat() -> None:
    info_d = probe_dur(V4 / "master_v3_blurred.mp4")
    n_chunks = max(1, -(-int(info_d * 1000) // int(BURN_CHUNK_S * 1000)))
    parts = []
    for ci in range(n_chunks):
        part = V4 / f"burn_v4_{ci:02d}.mp4"
        if not ((V4 / f"burn_v4_{ci:02d}.done").exists() and part.exists()):
            raise SystemExit(f"burn chunk {ci} missing")
        parts.append(part)
    lst = V4 / "burn_v4_chunks.txt"
    lst.write_text("".join(f"file '{p.name}'\n" for p in parts))
    _sh(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(lst),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(V4 / "master_v4_burned.mp4"),
        ],
        "burn v4 concat",
    )
    log(
        f"master_v4_burned: {probe_dur(V4 / 'master_v4_burned.mp4'):.2f}s, "
        f"{(V4 / 'master_v4_burned.mp4').stat().st_size / 1e6:.1f} MB"
    )


def phase_mux() -> None:
    burned = V4 / "master_v4_burned.mp4"
    if not burned.exists():
        raise SystemExit("burnconcat first")
    final = V4 / "master_postfx_v4.mp4"
    # the accepted audio (with the patched outro narration) is REUSED untouched
    _sh(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(burned),
            "-i",
            str(V4 / "audio_v3.m4a"),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            str(final),
        ],
        "final v4 mux",
    )
    d_v, d_f = probe_dur(burned), probe_dur(final)
    if abs(d_f - d_v) > 1.0:
        raise SystemExit(f"duration drifted {abs(d_f - d_v):.2f}s — refusing")
    log(f"muxed: master_postfx_v4.mp4 {d_f:.2f}s, {final.stat().st_size / 1e6:.1f} MB")


# --------------------------------------------------------------------- ship
def phase_ship() -> None:
    import hashlib
    import shutil

    final = V4 / "master_postfx_v4.mp4"
    if not final.exists():
        raise SystemExit("mux first")
    shutil.copyfile(final, SHIP_PACK / "final_720p.mp4")
    shutil.copyfile(SRT_V4, SHIP_PACK / "captions.srt")
    shutil.copyfile(
        ASS_V4, Path("/home/z/my-project/download/eustis_acceptance_postfx_captions.ass")
    )
    shutil.copyfile(SRT_V4, Path("/home/z/my-project/download/eustis_acceptance_captions.srt"))
    meta_p = SHIP_PACK / "metadata.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        meta["files"]["final_720p.mp4"] = str(SHIP_PACK / "final_720p.mp4")
        meta["final_sha256"] = hashlib.sha256(final.read_bytes()).hexdigest()
        meta["verification"]["duration_s"] = round(probe_dur(final), 1)
        meta["post_fx_v4"] = {
            "caption_truth": "VAD-evidence gates: speech_ratio<0.15 or ASR-loop "
            "(repeat>=3 at sr<0.5) or owner-confirmed windows "
            "-> gray-italic *unaudible*-family tags; LLM corrects "
            "text only where voice evidence allows; corrections "
            "re-aligned onto original word timestamps",
            "speaker_colors": "role-flow audit on clearly audible lines only "
            "(sr>=0.5), flips capped 15%, every flip logged; "
            "unclear voice = neutral gray, never a speaker color",
            "gap_fill": "VAD speech with no caption coverage -> *unaudible* "
            "gray-italic cues (owner law: audio without caption is "
            "handled)",
        }
        meta_p.write_text(json.dumps(meta, indent=2))
    log(
        f"shipped: {SHIP_PACK / 'final_720p.mp4'} "
        f"({(SHIP_PACK / 'final_720p.mp4').stat().st_size / 1e6:.1f} MB)"
    )


PHASES = {
    "decide": phase_decide,
    "captions": phase_captions,
    "burn": lambda: phase_burn(int(sys.argv[2])),
    "burnconcat": phase_burnconcat,
    "mux": phase_mux,
    "ship": phase_ship,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in PHASES:
        raise SystemExit(f"usage: caption_v4.py {'|'.join(PHASES)} [arg]")
    PHASES[sys.argv[1]]()
