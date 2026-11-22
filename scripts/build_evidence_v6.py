#!/usr/bin/env python3
"""Caption evidence v6 — measured audio truth per ASR event, MASTER clock.

Applies the committed caption_evidence.py law (postfx v4) to the fresh master
transcript: for every diarized segment
  speech_ratio / rms_db / voice_band / repeat_n / stretch
against the master 16k wav. Output: final_state/caption_evidence_v6.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/home/z/my-project/scripts")
import vad_evidence  # noqa: E402

WORK = Path("/home/z/my-project/pipeline_run/final_state")
SR = vad_evidence.SR
FRAME = SR * vad_evidence.FRAME_MS // 1000


def norm_txt(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def main() -> None:
    tr = json.loads((WORK / "asr_master_v6.json").read_text())
    pcm = vad_evidence.load_pcm(Path("/home/z/my-project/pipeline_run/v3/audio_master_16k.wav"))
    flags, _rms, _f = vad_evidence.vad_flags(pcm)

    evs = []
    for i, seg in enumerate(tr["segments"]):
        ws = [w for w in tr["words"] if seg["s"] - 0.01 <= w["s"] < seg["e"] + 0.01]
        if not ws:
            ws = [
                w
                for w in tr["words"]
                if w["speaker"] == seg["speaker"] and seg["s"] <= w["s"] < seg["e"]
            ]
        s, e = ws[0]["s"], ws[-1]["e"]
        txt = " ".join(w["w"] for w in ws)
        evs.append(
            {
                "id": i,
                "speaker": seg["speaker"],
                "has_words": True,
                "s": round(s, 2),
                "e": round(e, 2),
                "n_words": len(ws),
                "text": txt,
                "conf": round(sum(w["conf"] for w in ws) / len(ws), 3),
                "speech_ratio": round(vad_evidence.speech_ratio(flags, FRAME, s, e), 3),
                "rms_db": round(vad_evidence.segment_rms_db(pcm, s, e), 1),
                "voice_band": round(vad_evidence.voice_band_ratio(pcm, s, e), 3),
                "stretch": round((e - s) / max(1, len(ws)), 2),
            }
        )

    for ev in evs:
        nt = norm_txt(ev["text"])
        if len(nt) < 5:
            ev["repeat_n"] = 1
            continue
        ev["repeat_n"] = sum(
            1
            for o in evs
            if abs((o["s"] + o["e"]) / 2 - (ev["s"] + ev["e"]) / 2) <= 45
            and norm_txt(o["text"]) == nt
        )

    (WORK / "caption_evidence_v6.json").write_text(json.dumps(evs, ensure_ascii=False, indent=1))

    spk = {}
    for ev in evs:
        spk[ev["speaker"]] = spk.get(ev["speaker"], 0) + 1
    print(f"evidence v6: {len(evs)} events, speakers: {spk}")
    print(f"word span: {evs[0]['s']:.1f}s .. {max(e['e'] for e in evs):.1f}s (master 657.7s)")


if __name__ == "__main__":
    main()
