#!/usr/bin/env python3
"""Captions: notebook CELL 9 chunking + two-lane placement, at 1080p.

Outputs (master clock):
  work_v2/captions_data.json  - chunks for the HyperFrames HTML caption layer
  work_v2/final_subs.ass      - karaoke ASS (1080p, Roboto)
  work_v2/captions.srt        - plain SRT sidecar
  work_v2/chapters.txt        - act chapters
"""

import json
import re

WORK = "/home/z/my-project/work_v2"
BAD_RE = re.compile(
    r"\b(fuck\w*|motherfuck\w*|shit\w*|bitch\w*|pussy\w*|dick\w*|asshole\w*|goddamn\w*|damn\w*)\b",
    re.I,
)


def fmt_ass(s):
    return f"{int(s // 3600)}:{int((s % 3600) // 60):02d}:{s % 60:05.2f}"


def fmt_srt(s):
    return f"{int(s // 3600):02d}:{int((s % 3600) // 60):02d}:{int(s % 60):02d},{int(round((s - int(s)) * 1000)):03d}"


HDR = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Officer,Roboto,54,&H00FFFFFF,&H00CCC0B8,&H00000000,&H90000000,-1,0,0,0,100,100,0.6,0,1,4.2,2.4,2,120,120,69,1
Style: Suspect,Roboto,54,&H004DD8FF,&H004D9BB3,&H00000000,&H90000000,-1,0,0,0,100,100,0.6,0,1,4.2,2.4,2,120,120,69,1
Style: Witness,Roboto,54,&H009EFF7C,&H0054A66A,&H00000000,&H90000000,-1,0,0,0,100,100,0.6,0,1,4.2,2.4,2,120,120,69,1
Style: Narrator,Roboto,54,&H00FFE470,&H00B8A37C,&H00000000,&H90000000,-1,0,0,0,100,100,0.6,0,1,4.2,2.4,2,120,120,69,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def main():
    ts = json.load(open(f"{WORK}/subs_timeline.json"))
    events, bounds, total_dur = ts["events"], ts["bounds"], ts["total_dur"]

    # profanity: mask caption text (audio already muted in pack stage)
    bleeps = []
    for ev in events:
        for w in ev["words"]:
            if BAD_RE.search(w["w"]):
                bleeps.append([round(max(0.0, w["s"] - 0.04), 3), round(w["e"] + 0.04, 3)])
                w["w"] = BAD_RE.sub(lambda m: m.group(0)[0] + "***", w["w"])
    print(f"caption-masked profanity words: {len(bleeps)}")

    # notebook chunking: <=5 words, <=27 chars, break after punctuation (>=3 words), gap > 1.0s
    chunks = []
    for uid, ev in enumerate(events):
        st, tw = ev["role"], ev["words"]
        cur = []
        for w in tw:
            if cur and w["s"] - cur[-1]["e"] > 1.0:
                chunks.append((uid, st, cur))
                cur = []
            cur.append(w)
            ch = sum(len(x["w"]) + 1 for x in cur)
            if len(cur) >= 5 or ch >= 27 or (w["w"][-1] in ",.?!" and len(cur) >= 3):
                chunks.append((uid, st, cur))
                cur = []
        if cur:
            chunks.append((uid, st, cur))

    # notebook two-lane placement
    chunks.sort(key=lambda x: x[2][0]["s"])
    placed, last = [], [None, None]
    for _uid, st, c in chunks:
        s, e = c[0]["s"], max(c[-1]["e"] + 0.06, c[0]["s"] + 0.30)
        l0 = placed[last[0]] if last[0] is not None else None
        l1 = placed[last[1]] if last[1] is not None else None
        if l0 and s < l0["e"] - 0.05 and l0["st"] != st:
            lane, mv = 1, 144
            if l1 and s < l1["e"] + 0.02:
                if s - l1["s"] >= 0.25:
                    l1["e"] = s - 0.02
                else:
                    s = l1["s"] + 0.25
                    e = max(e, s + 0.25)
        else:
            lane, mv = 0, 69
            if l0 and s < l0["e"] + 0.02:
                if s - l0["s"] >= 0.25:
                    l0["e"] = s - 0.02
                else:
                    s = l0["s"] + 0.25
                    e = max(e, s + 0.25)
        placed.append(
            {"s": round(s, 3), "e": round(e, 3), "st": st, "mv": mv, "lane": lane, "c": c}
        )
        last[lane] = len(placed) - 1

    # ASS with karaoke \k
    ass_l, srt_l = [], []
    for i, it in enumerate(placed, 1):
        s, e, st, mv, c = it["s"], it["e"], it["st"], it["mv"], it["c"]  # noqa: E741 - notebook lane vars
        kp = [
            f"{{\\k{max(6, int(round(((min(e, c[wi + 1]['s']) if wi + 1 < len(c) else e) - max(s, w['s'])) * 100)))}}}{w['w']}"
            for wi, w in enumerate(c)
        ]
        ass_l.append(
            f"Dialogue: 0,{fmt_ass(s)},{fmt_ass(e)},{st},,0,0,{mv},,{{\\fad(40,30)}}" + " ".join(kp)
        )
        srt_l.append(f"{i}\n{fmt_srt(s)} --> {fmt_srt(e)}\n{' '.join(w['w'] for w in c)}\n")
    with open(f"{WORK}/final_subs.ass", "w", encoding="utf-8") as f:
        f.write(HDR + "\n".join(ass_l) + "\n")
    with open(f"{WORK}/captions.srt", "w", encoding="utf-8") as f:
        f.write("\n".join(srt_l))

    # chapters
    seen, ch = set(), []
    for b in bounds:
        if b["act"] not in seen:
            seen.add(b["act"])
            ch.append(f"{int(b['start'] // 60):02d}:{int(b['start'] % 60):02d} {b['act']}")
    with open(f"{WORK}/chapters.txt", "w") as f:
        f.write("\n".join(ch) + "\n")

    # HTML caption data: per-chunk absolute start/duration, words with relative times
    caps = []
    for it in placed:
        words = [{"w": w["w"], "rs": round(max(0.0, w["s"] - it["s"]), 3)} for w in it["c"]]
        caps.append(
            {
                "s": it["s"],
                "e": it["e"],
                "dur": round(it["e"] - it["s"], 3),
                "style": it["st"],
                "lane": it["lane"],
                "words": words,
            }
        )
    json.dump(
        {"captions": caps, "total_dur": total_dur, "bleeps": bleeps},
        open(f"{WORK}/captions_data.json", "w"),
        ensure_ascii=False,
    )
    print(
        f"chunks placed: {len(caps)} (lanes: L0={sum(1 for c in caps if c['lane'] == 0)}, L1={sum(1 for c in caps if c['lane'] == 1)})"
    )
    print(f"ass/srt/chapters/captions_data written; total {total_dur:.2f}s")
    used = {c["style"] for c in caps}
    print("styles used:", sorted(used))


if __name__ == "__main__":
    main()
