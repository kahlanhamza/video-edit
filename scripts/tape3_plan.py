#!/usr/bin/env python3
"""Tape 3 (Florida grocery / St. Johns County) edit-plan computation.

Takes the notebook's verbatim edit plan (work done by the owner on Kaggle,
uploaded 2026-11-28), applies the notebook's silence-trim, reports raw share,
and extends raw blocks with transcript-dense material until the trimmed raw
share clears the 0.32 floor. Emits parts_plan.json for the part cutter.
"""
import json, re, subprocess, sys

WORK = "/home/z/my-project/work_v2"
SRC = "/home/z/my-project/footage/florida_grocery_source.webm"

# Notebook CELL 6 PLAN, verbatim (owner's freshest approved edit plan).
PLAN = [
    {"kind": "raw", "act": "Cold Open", "s": 8.5,  "e": 16.9},
    {"kind": "nar", "act": "Cold Open", "b": [25.0, 49.0], "text": "A grocery store in St. Johns County, Florida. Deputies are called for a man causing a disturbance inside. Before they even reach the aisle, he already knows how this ends — and he's asking for it. What he did to get here involves one half-gallon of milk and a scene shoppers won't forget."},
    {"kind": "raw", "act": "Act 1", "s": 39.8, "e": 46.7},
    {"kind": "raw", "act": "Act 1", "s": 48.6, "e": 53.7},
    {"kind": "nar", "act": "Act 1", "b": [60.0, 84.0], "text": "It takes less than thirty seconds inside the store. The man refuses to calm down, refuses to leave, and keeps closing the distance. Deputies take him to the floor between the shelves — no weapons, no punches, just weight and patience. He isn't fighting them anymore. He just won't stop moving."},
    {"kind": "raw", "act": "Act 2", "s": 88.5,  "e": 97.4},
    {"kind": "raw", "act": "Act 2", "s": 100.1, "e": 115.2},
    {"kind": "raw", "act": "Act 2", "s": 117.3, "e": 122.5},
    {"kind": "nar", "act": "Act 2", "b": [148.0, 172.0], "text": "Then the story gets stranger. Cuffed on the floor, he claims someone stuck a needle in his back — nobody has a needle. Then comes a confession nobody asked for: acid, Molly, cocaine, what he calls a whole brick. What he's describing is excited delirium, and it changes every decision that follows."},
    {"kind": "raw", "act": "Act 3", "s": 126.7, "e": 143.9},
    {"kind": "raw", "act": "Act 3", "s": 178.7, "e": 184.6},
    {"kind": "raw", "act": "Act 3", "s": 202.3, "e": 217.1},
    {"kind": "nar", "act": "Act 3", "b": [238.5, 264.0], "text": "For the next several minutes he begs for fire rescue, says his leg is being crushed, then flips into grandeur — demanding a sergeant, a commander, declaring they're the Supreme Court. Deputies live by one rule here: a man this volatile doesn't stand up and walk away. The only question is where he goes next."},
    {"kind": "raw", "act": "Act 4", "s": 292.0, "e": 308.5},
    {"kind": "raw", "act": "Act 4", "s": 317.9, "e": 336.8},
    {"kind": "nar", "act": "Act 4", "b": [337.0, 356.0], "text": "Outside, deputies try the calm approach — deep breaths, air conditioning, the back of a patrol car. He compares himself to Ric Flair and answers to nobody. The deputies have seen this before; one of them was here last night. And what they found inside the store explains everything about this call."},
    {"kind": "raw", "act": "Act 5", "s": 357.3, "e": 377.3},
    {"kind": "raw", "act": "Act 5", "s": 408.6, "e": 430.9},
    {"kind": "nar", "act": "Act 5", "b": [497.6, 524.0], "text": "Here's what happened before the cameras rolled: the man was inside yelling, a jug of milk in his hand, pouring it over himself and throwing it around the store. That's disorderly conduct on its own. The destroyed milk makes it criminal mischief — but only if the store manager agrees to pursue charges. So they go ask him."},
    {"kind": "raw", "act": "Act 5", "s": 458.9, "e": 474.2},
    {"kind": "raw", "act": "Act 5", "s": 477.6, "e": 489.0},
    {"kind": "raw", "act": "Act 6", "s": 565.2, "e": 581.0},
    {"kind": "raw", "act": "Act 6", "s": 597.3, "e": 624.2},
    {"kind": "nar", "act": "Act 6", "b": [648.0, 672.0], "text": "That history changes the plan. Not jail first — a Baker Act: Florida's involuntary psychiatric hold, for someone in crisis who may be a danger to himself. The hospital comes before booking and charges. Get him cleared, get the statement, get the video — the jail cell will still be there after."},
    {"kind": "raw", "act": "Act 6", "s": 632.0, "e": 647.5},
    {"kind": "raw", "act": "Outro", "s": 672.7,  "e": 694.0},
    {"kind": "raw", "act": "Outro", "s": 748.9,  "e": 766.6},
    {"kind": "raw", "act": "Outro", "s": 869.5,  "e": 893.0},
    {"kind": "nar", "act": "Outro", "b": [806.6, 832.0], "text": "Booking turns surreal — he can't remember his own birthday. When the spitting starts, deputies cover his head with his own shirt. At the hospital he kicks a doctor and has to be lifted onto the stretcher by hand. One half-gallon of milk, six deputies, an ambulance, and a trip to jail — all traced back to one jug of milk."}
]

def trim_sil(s_t, e_t, clean_lines):
    """Notebook CELL 6 trim_sil, verbatim."""
    sp = [[max(s_t, cl["start"] - 0.35), min(e_t, cl["end"] + 0.45)] for cl in clean_lines if cl["end"] >= s_t and cl["start"] <= e_t]
    if not sp: return [(s_t, e_t)]
    sp.sort(); m = [sp[0]]
    for a, b in sp[1:]:
        if a - m[-1][1] <= 1.2: m[-1][1] = max(m[-1][1], b)
        else: m.append([a, b])
    return [(round(a, 2), round(b, 2)) for a, b in m if b - a >= 0.7]

def speech_secs(a, b, clean_lines):
    tot = 0.0
    for cl in clean_lines:
        s, e = max(a, cl["start"]), min(b, cl["end"])
        if e > s: tot += e - s
    return tot

def main():
    clean = json.load(open(f"{WORK}/clean_transcript.json"))
    src_dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", SRC]).decode().strip())
    print(f"source duration: {src_dur:.3f}s")

    raw_floor = 0.32 * src_dur
    print(f"raw-share floor 0.32 -> trimmed raw must be >= {raw_floor:.1f}s")

    # trim the owner's plan, verbatim rule
    trimmed = []
    for x in PLAN:
        if x["kind"] == "raw":
            for rs, re_ in trim_sil(x["s"], x["e"], clean):
                trimmed.append({"kind": "raw", "act": x["act"], "s": rs, "e": re_})
        else:
            trimmed.append(x)

    def stats(pl):
        raw = sum(x["e"] - x["s"] for x in pl if x["kind"] == "raw")
        nar_w = sum(len(x["text"].split()) for x in pl if x["kind"] == "nar")
        return raw, nar_w

    raw, nw = stats(trimmed)
    print(f"notebook plan trimmed: raw={raw:.1f}s ({raw/src_dur:.3f} share), nar words={nw}")

    # --- extension pass: find transcript-dense gaps to add as new raw blocks ---
    # candidate windows: contiguous speech-dense regions not covered by existing raw
    covered = [(x["s"], x["e"]) for x in trimmed if x["kind"] == "raw"]
    def cov(a, b):
        return any(s <= a and b <= e for s, e in covered)
    # density per 5s bucket over the whole source
    buckets = []
    for a in range(0, int(src_dur) - 4, 5):
        d = speech_secs(a, a + 5, clean)
        c = cov(a, a + 5)
        buckets.append((a, a + 5, d, c))
    # build candidate runs of uncovered buckets with density >= 3.0s speech / 5s
    runs, cur = [], None
    for a, b, d, c in buckets:
        if not c and d >= 3.0:
            if cur and a - cur[1] <= 5: cur = (cur[0], b, cur[2] + d)
            else:
                if cur: runs.append(cur)
                cur = (a, b, d)
        else:
            if cur: runs.append(cur)
            cur = None
    if cur: runs.append(cur)
    runs = [r for r in runs if r[1] - r[0] >= 12.0]
    runs.sort(key=lambda r: -r[2])
    print("\ntop uncovered speech-dense runs:")
    for r in runs[:14]:
        print(f"  {r[0]:7.1f}-{r[1]:7.1f}  ({r[1]-r[0]:5.1f}s, speech {r[2]:5.1f}s)")

    # greedy add until trimmed raw share target 0.34 (buffer above 0.32)
    # preference: narrative-fit runs (Act 5 manager scene first), cap length near the need
    pref = [(530.0, 550.0, "Act 5"), (760.0, 800.0, "Outro"), (825.0, 875.0, "Outro"), (15.0, 35.0, "Cold Open")]
    target = 0.34 * src_dur
    added = []
    need = target - raw
    print(f"\nneed +{need:.1f}s trimmed raw to reach share 0.34")
    if need > 0:
        raw_mids = [((x["s"] + x["e"]) / 2, x["act"]) for x in trimmed if x["kind"] == "raw"]
        def act_for(a, b):
            mid = (a + b) / 2
            return min(raw_mids, key=lambda t: abs(t[0] - mid))[1]
        for pa, pb, pact in pref:
            if need <= 0: break
            cap = min(pb - pa, need + 8.0)
            a, b = pa, pa + cap
            a, b = round(a * 2) / 2, round(b * 2) / 2
            if cov(a, b): continue
            t = trim_sil(a, b, clean)
            got = sum(e - s for s, e in t)
            if got < 8.0: continue
            act = pact
            for rs, re_ in t:
                added.append({"kind": "raw", "act": act, "s": rs, "e": re_})
            print(f"  added {act} raw {a:.1f}-{b:.1f} (trimmed {got:.1f}s)")
            need -= got
    # keep the owner's PLAN order as backbone; insert each added block right after
    # the last existing block of the same act whose source end <= block start
    merged = list(trimmed)
    for blk in added:
        idx = 0
        for i, x in enumerate(merged):
            if x["act"] == blk["act"] and (x["e"] if x["kind"] == "raw" else 1e9) <= blk["s"]:
                idx = i + 1
        merged.insert(idx, blk)
    raw2, nw2 = stats(merged)
    print(f"final plan: raw={raw2:.1f}s (share {raw2/src_dur:.3f}), nar words={nw2}, blocks={len(merged)}")
    for a in dict.fromkeys(x["act"] for x in merged):
        n = sum(1 for x in merged if x["act"] == a)
        secs = sum(x["e"] - x["s"] for x in merged if x["act"] == a and x["kind"] == "raw")
        print(f"  {a}: {n} blocks, raw {secs:.1f}s")

    out = {"source": SRC, "source_duration_s": src_dur, "plan": merged,
           "added_blocks": added,
           "laws": {"raw_share": round(raw2 / src_dur, 4), "raw_floor": 0.32,
                    "raw_secs": round(raw2, 2), "nar_words": nw2}}
    json.dump(out, open(f"{WORK}/parts_plan.json", "w"), indent=1)
    print(f"\nwrote {WORK}/parts_plan.json")

if __name__ == "__main__":
    main()
