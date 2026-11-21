#!/usr/bin/env python3
"""Per-event caption evidence on the master clock (postfx v4 input).

For every asm_state event with words:
  - speech_ratio: VAD-positive fraction of the event span
  - rms_db:      mean loudness of the span
  - voice_band:  250-3400 Hz energy fraction (voice vs music/wind)
  - repeat_n:    how many times this normalized text occurs within +-45 s
  - stretch_s_per_word: span seconds per word (ASR-loop signature: >> 0.6)
Written to V3/caption_evidence.json.
"""
import json
import re
from pathlib import Path

import numpy as np

from vad_evidence import (V3, SR, FRAME_MS, load_pcm, vad_flags, speech_ratio,
                          segment_rms_db, voice_band_ratio)

FRAME = SR * FRAME_MS // 1000


def norm_txt(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def main() -> None:
    pcm = load_pcm(V3 / "audio_src_16k.wav")
    flags, rms, _ = vad_flags(pcm)
    state = json.loads((WORK := Path("/home/z/my-project/pipeline_run"))
                       .joinpath("asm_state.json").read_text())
    events = state["events"]

    evs = []
    for i, ev in enumerate(events):
        ws = ev.get("words") or []
        if not ws:
            evs.append({"id": i, "role": ev.get("role", "?"), "has_words": False})
            continue
        s, e = ws[0]["s"], ws[-1]["e"]
        txt = " ".join(w["w"] for w in ws)
        evs.append({
            "id": i, "role": ev.get("role", "?"), "has_words": True,
            "s": round(s, 2), "e": round(e, 2), "n_words": len(ws),
            "text": txt,
            "speech_ratio": round(speech_ratio(flags, FRAME, s, e), 3),
            "rms_db": round(segment_rms_db(pcm, s, e), 1),
            "voice_band": round(voice_band_ratio(pcm, s, e), 3),
            "stretch": round((e - s) / max(1, len(ws)), 2),
        })

    # repetition count within +-45 s window (normalized text, >2 tokens)
    for ev in evs:
        if not ev.get("has_words"):
            ev["repeat_n"] = 0
            continue
        nt = norm_txt(ev["text"])
        if len(nt) < 5:
            ev["repeat_n"] = 1
            continue
        ev["repeat_n"] = sum(
            1 for o in evs
            if o.get("has_words") and o["id"] != ev["id"]
            and norm_txt(o["text"]) == nt
            and abs(o["s"] - ev["s"]) <= 45)

    (V3 / "caption_evidence.json").write_text(json.dumps(evs, indent=1))
    n = [e for e in evs if e.get("has_words")]
    print(f"evidence for {len(n)} dialogue events -> {V3 / 'caption_evidence.json'}")

    # sanity: loop window events
    for ev in evs:
        if ev.get("has_words") and 605 <= ev["s"] <= 630:
            print(f"  ev{ev['id']:3d} {ev['role']:8s} {ev['s']:7.2f}-{ev['e']:7.2f} "
                  f"sr={ev['speech_ratio']:.2f} rms={ev['rms_db']:.0f} rep={ev['repeat_n']} "
                  f"st={ev['stretch']:.2f} | {ev['text'][:44]}")
    # distribution of speech_ratio for all dialogue events
    sr_all = np.array([e["speech_ratio"] for e in n])
    print(f"speech_ratio: min {sr_all.min():.2f} p10 {np.percentile(sr_all, 10):.2f} "
          f"median {np.median(sr_all):.2f} p90 {np.percentile(sr_all, 90):.2f}")
    print("events with sr<0.15:", int((sr_all < 0.15).sum()),
          "| sr<0.30:", int((sr_all < 0.30).sum()))


if __name__ == "__main__":
    main()
