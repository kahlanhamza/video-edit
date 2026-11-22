#!/usr/bin/env python3
"""QA gate for the v6 caption rebuild — the 11 committed v4 gates adapted,
plus the two v6-specific laws the owner demanded.

What changed vs qa_v4.py:
  - FINAL/SRT/ASS point at the v6 artifacts; the comparison base is the
    ACCEPTED master_v3_blurred.mp4 (the pre-blur base was cleaned up; v6
    builds on the same accepted blur, so the blur gates assert the burn
    did not degrade it).
  - gates 3/4 (face leak / pass-2 effectiveness) are reimplemented as
    "blur preserved": haar-cascade face boxes at the known 199-203 s leak
    window + sampled instants must not get sharper in the final, and the
    watermark box must stay collapsed.
  - VAD runs on the MASTER 16 kHz wav (master clock) — the v5 bug class
    (source-clock evidence) is structurally impossible here.
  - NEW gate 12: narrator-on-b-roll — no Officer/Suspect cue may overlap a
    Narrator cue (b-roll carries narrator captions, never speaker captions).
  - NEW gate 13: font legibility — measured caption glyph height >= 26 px
    at 720p (the v5 regression rendered ~13 px effective).

Checks:
  1  duration identical to the accepted base
  2  no paragraph flash at any burn boundary
  3  blur preserved at the 199-203 s leak window (face boxes)
  4  blur preserved on sampled instants (no sharpening anywhere)
  5  watermark exact-pixel box blur on caption-free instants
  6  captions present + audio present
  7  no hallucinated loop text (v4 ghosts + v6 ASR ghosts)
  8  owner window (609.4-628.9) carries ONLY gray tag cues
  9  caption color audit (Officer=white, Suspect=gold, Narrator=cyan, tag=gray)
 10  no VAD speech span >= 1.0 s without caption coverage (master clock)
 11  caption presence: pixels during cue, clean after (no stuck text)
 12  narrator-on-b-roll: no speaker cue overlaps a narrator cue
 13  font legibility: caption glyph height >= 26 px at 720p
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/z/my-project/scripts")

import cv2  # noqa: E402
from postfx_v3 import (  # noqa: E402
    BURN_CHUNK_S,
    V3,
    _grab,
    _lap,
    _text_mass,
    log,
    probe_dur,
)
from vad_evidence import (  # noqa: E402
    load_pcm,
    segment_rms_db,
    speech_spans,
    vad_flags,
    voice_band_ratio,
)

WORK = Path("/home/z/my-project/pipeline_run/final_state")
FINAL = Path("/home/z/my-project/download/eustis_final_720p_v6.mp4")
BASE = V3 / "master_v3_blurred.mp4"  # the accepted blurred master
ASS = WORK / "subtitles_v6.ass"
SRT = WORK / "captions_v6.srt"
WAV16K = V3 / "audio_master_16k.wav"
QA = WORK / "qa_v6.json"

OWNER_WINDOWS = [(609.40, 628.90)]
CARD_PAD = 7.0


def _cue_iv() -> list[tuple[float, float, str, str]]:
    def to_sec(ts):
        h, m, s = ts.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    out = []
    for ln in ASS.read_text(encoding="utf-8").splitlines():
        if not ln.startswith("Dialogue: "):
            continue
        p = ln.split(",", 9)
        plain = re.sub(r"\{[^}]*\}", "", p[9]).strip()
        out.append((to_sec(p[1]), to_sec(p[2]), p[3], plain))
    return out


def _glyph_colors(frame) -> dict:
    h, w = frame.shape[:2]
    roi = frame[h - 170 : h - 8, 40 : w - 40]
    b, g, r = roi[:, :, 0].astype(int), roi[:, :, 1].astype(int), roi[:, :, 2].astype(int)
    dark = ((b < 70) & (g < 70) & (r < 70)).astype(np.uint8)
    dark_near = cv2.dilate(dark, np.ones((5, 5), np.uint8)) > 0
    white = (r > 215) & (g > 215) & (b > 215)
    yellow = (r > 190) & (g > 150) & (b < 150) & ((r - b) > 60)
    sky = (b > 190) & (g > 150) & (r < 170) & ((b - r) > 60)
    grayish = (np.abs(r - g) < 26) & (np.abs(g - b) < 26) & (r > 115) & (r < 200)
    counts = {}
    for name, m in (("white", white), ("yellow", yellow), ("sky", sky), ("gray", grayish)):
        counts[name] = int((m & dark_near).sum())
    neutral = (white | grayish) & dark_near
    if neutral.any():
        vals = np.concatenate([r[neutral], g[neutral], b[neutral]])
        counts["bright_p98"] = int(np.percentile(vals, 98))
        counts["gray_p80"] = int(np.percentile(vals, 80))
    else:
        counts["bright_p98"] = 0
        counts["gray_p80"] = 0
    return counts


def _caption_added_colors(t: float) -> dict | None:
    ff_, fb_ = _grab(FINAL, t), _grab(BASE, t)
    if ff_ is None or fb_ is None:
        return None
    cf, cb = _glyph_colors(ff_), _glyph_colors(fb_)
    return {k: max(0, int(cf[k]) - int(cb.get(k, 0))) for k in cf}


def _caption_added_frame(t: float):
    """Mask of pixels the burn ADDED at t (glyph pixels only)."""
    ff_, fb_ = _grab(FINAL, t), _grab(BASE, t)
    if ff_ is None or fb_ is None:
        return None
    d = np.abs(ff_.astype(int) - fb_.astype(int)).sum(axis=2)
    return d > 60


def _face_boxes(frame):
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes = cascade.detectMultiScale(gray, 1.15, 4, minSize=(46, 46))
    return [tuple(int(v) for v in b) for b in boxes]


def main() -> None:
    if not FINAL.exists():
        raise SystemExit("mux first")
    checks: list[dict] = []

    def check(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        QA.write_text(json.dumps(checks, indent=1))
        log(f"  [{'OK' if ok else 'FAIL'}] {name}: {detail}")

    info_d = probe_dur(FINAL)
    cues = _cue_iv()

    # ---- 1 duration
    check(
        "duration",
        abs(info_d - probe_dur(BASE)) <= 1.0,
        f"final {info_d:.2f}s vs accepted base {probe_dur(BASE):.2f}s",
    )

    # ---- 2 flash regression at burn boundaries
    n_chunks = max(1, -(-int(info_d * 1000) // int(BURN_CHUNK_S * 1000)))
    bounds = [ci * BURN_CHUNK_S for ci in range(1, n_chunks)]
    spikes = []
    for bnd in bounds:
        worst = 0.0
        for t in np.arange(bnd - 0.15, bnd + 0.45, 1 / 30):
            ff_, fb_ = _grab(FINAL, t), _grab(BASE, t)
            if ff_ is None or fb_ is None:
                continue
            worst = max(worst, _text_mass(ff_, "center") / max(1.0, _text_mass(fb_, "center")))
        if worst > 4.0:
            spikes.append((round(bnd, 1), round(worst, 1)))
    check(
        "no_paragraph_flash",
        not spikes,
        f"boundaries {bounds} worst ratio {max((s[1] for s in spikes), default=0.0):.1f}x (<4.0)",
    )

    # ---- 3 blur preserved at the measured 199-203 s leak window
    # The accepted master_v3_blurred already passed the v4 leak gate; v6 must
    # not sharpen it. Faces are detected on the FINAL (if a face is legible
    # enough to detect, its region must not be sharper than the accepted base).
    ok3, notes = True, []
    for t in (199.5, 200.5, 201.5, 202.5):
        fb, ff = _grab(BASE, t), _grab(FINAL, t)
        if fb is None or ff is None:
            continue
        boxes = _face_boxes(fb) + _face_boxes(ff)
        if not boxes:
            notes.append(f"t={t}: no face box (fully blurred — ok)")
            continue
        for x, y, w, h in boxes:
            ix, iy = x + w // 4, y + h // 4
            iw, ih = max(4, w // 2), max(4, h // 2)
            v0 = max(_lap(fb[iy : iy + ih, ix : ix + iw]), 1e-6)
            v1 = max(_lap(ff[iy : iy + ih, ix : ix + iw]), 1e-6)
            ok3 = ok3 and (v1 < 40.0 or v0 / v1 >= 1.8)
            notes.append(f"t={t} box({x},{y},{w},{h}): v1={v1:.0f} ratio={v0 / v1:.1f}")
    check("blur_preserved_199_203s", ok3 and len(notes) == 4, "; ".join(notes) or "frames missing")

    # ---- 4 blur preserved on sampled instants (no sharpening anywhere)
    rng = np.random.default_rng(6)
    cards_meta4 = json.loads((V3 / "cards_meta_v3.json").read_text())
    card_iv4 = [(t0 - 0.5, t0 + 7.0) for _png, t0 in cards_meta4["card_times"]]
    samples = [round(float(x), 2) for x in rng.uniform(20.0, info_d - 20.0, 30)]
    samples = [t for t in samples if not any(a <= t <= b for a, b in card_iv4) and t >= 9.0]
    samples = samples[:18]
    worst_sharp = 0.0
    for t in samples:
        fb, ff = _grab(BASE, t), _grab(FINAL, t)
        if fb is None or ff is None:
            continue
        # caption band excluded (glyphs legitimately add edges); wm box too
        h, w = ff.shape[:2]
        core = slice(60, h - 190), slice(60, w - 60)
        v0 = max(_lap(fb[core]), 1e-6)
        v1 = _lap(ff[core])
        worst_sharp = max(worst_sharp, v1 / v0)
    check(
        "blur_preserved_no_sharpening",
        worst_sharp < 1.25,
        f"worst final/base sharpness x{worst_sharp:.2f} over {len(samples)} "
        f"sampled instants (<1.25; card windows and the 0-9s zoom cut "
        f"excluded)",
    )

    # ---- 5 watermark exact pixels on caption-free instants
    wm = json.loads((V3 / "wm_spans_v3.json").read_text())["spans"]
    arr = np.array([sp["box"] for sp in wm])
    bx, by, bw, bh = [int(v) for v in np.median(arr, axis=0)]
    bx, by, bw, bh = max(0, bx - 2), max(0, by - 2), bw + 4, bh + 4
    cue_iv = sorted((a, b) for a, b, _st, _tx in cues)

    def _free(t):
        return all(not (a - 0.2 <= t <= b + 0.2) for a, b in cue_iv)

    cand = [round(float(x), 2) for x in np.arange(5.0, max(6.0, info_d - 3.0), 0.5)]
    free = [t for t in cand if _free(t)]
    step = max(1, len(free) // 24)
    free = free[::step]
    rs, abs_vars = [], []
    for t in free:
        fb, ff = _grab(BASE, t), _grab(FINAL, t)
        if fb is None or ff is None:
            continue
        v0 = _lap(fb[by : by + bh, bx : bx + bw])
        v1 = _lap(ff[by : by + bh, bx : bx + bw])
        abs_vars.append(v1)
        if v0 >= 60:
            rs.append(v0 / max(v1, 1e-6))
    worst_absv = max(abs_vars, default=0.0)
    wm_ok = (not rs) or (min(rs) >= 1.6 and worst_absv < 70.0)
    # note: v6 re-applies the same gblur(20) over the already-blurred wm box,
    # so ratios sit near 1.0 when the base box is already collapsed; the gate
    # is that the box NEVER becomes legible (absolute variance stays low).
    if not rs:
        wm_ok = worst_absv < 70.0
    check(
        "watermark_exact_pixels",
        wm_ok,
        f"box ({bx},{by},{bw},{bh}): {len(abs_vars)} caption-free samples, "
        f"worst final var {worst_absv:.0f} (<70), min base/final x"
        f"{min(rs, default=0.0):.1f}",
    )

    # ---- 6 captions + audio present
    srt = SRT.read_text()
    n_cues = srt.count(" --> ")
    n_tags = len(re.findall(r"\*(?:unaudible|mumbling|whispering|cross-talk|panicking)\*", srt))
    a_dur = float(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=duration",
                "-of",
                "csv=p=0",
                str(FINAL),
            ]
        )
        .decode()
        .strip()
    )
    check(
        "captions_and_audio",
        n_cues >= 200 and abs(a_dur - info_d) <= 0.1 and n_tags >= 10,
        f"{n_cues} cues ({n_tags} gray tags), audio {a_dur:.2f}s",
    )

    # ---- 7 no hallucinated text (v4 owner ghosts + v6 ASR ghosts)
    hay = (ASS.read_text(encoding="utf-8") + srt).lower()
    ghosts = [g for g in ("people want to stop", "want to stop", "salvi", "42 to 14") if g in hay]
    check(
        "no_hallucinated_loop_text",
        not ghosts,
        f"ghost phrases absent: {not ghosts} (hits: {ghosts or 'none'})",
    )

    # ---- 8 owner window carries ONLY gray tag cues
    lo, hi = OWNER_WINDOWS[0]
    inwin = [
        (a, b, st, tx) for a, b, st, tx in cues if b > lo and a < hi and not (a < lo and b < lo)
    ]
    bad = [c for c in inwin if "*" not in c[3]]
    tagcov = sum(min(b, hi) - max(a, lo) for a, b, _st, tx in inwin if "*" in tx)
    win_dur = hi - lo
    check(
        "loop_window_gray_tags_only",
        not bad and tagcov >= 0.75 * win_dur,
        f"{len(inwin)} cues in {lo}-{hi}s, all tags: {not bad}, tag "
        f"coverage {tagcov:.1f}s of {win_dur:.0f}s (>=75%)",
    )

    # ---- 9 caption color audit (lonely cues; skip card windows)
    cards_meta = json.loads((V3 / "cards_meta_v3.json").read_text())
    card_iv = [(t0, t0 + 6.5) for _png, t0 in cards_meta["card_times"]]
    expect = {"Officer": "white", "Suspect": "yellow", "Narrator": "sky", "Tag": "gray"}
    lonely = []
    for i, (a, b, st, tx) in enumerate(cues):
        if any((a < o[1] + 0.12 and b > o[0] - 0.12) for j, o in enumerate(cues) if j != i):
            continue
        if any(a < ce and b > cs for cs, ce in card_iv):
            continue
        lonely.append((a, b, st, tx))
    step = max(1, len(lonely) // 36)
    sample = lonely[::step][:36]
    mism, tested = [], 0
    for a, b, st, tx in sample:
        kind = "Tag" if "*" in tx else st
        want = expect[kind]
        t = max(a + 0.10, b - 0.15)
        c = _caption_added_colors(t)
        if c is None:
            continue
        tested += 1
        tot = sum(
            v for k, v in c.items() if not k.startswith("bright") and not k.startswith("gray_p")
        )
        if tot < 150:
            continue
        if want == "white":
            okk = c["white"] >= 1.2 * max(c["yellow"], c["sky"]) and (
                c["bright_p98"] > 230 or c["white"] > c["gray"]
            )
        elif want == "yellow":
            okk = c["yellow"] >= 1.5 * max(c["white"], c["sky"])
        elif want == "sky":
            okk = c["sky"] >= 1.5 * max(c["white"], c["yellow"])
        else:
            okk = c["gray"] >= 1.2 * c["yellow"] and c["gray"] >= c["white"] and c["gray_p80"] < 200
        if not okk:
            mism.append(f"{kind}@{a:.1f}s want={want} got={c}")
    check(
        "caption_color_matches_speaker",
        tested >= 10 and not mism,
        f"{tested} lonely cues sampled, {len(mism)} color mismatches"
        + (f" e.g. {mism[:3]}" if mism else ""),
    )

    # ---- 10 gap coverage on the MASTER clock
    pcm = load_pcm(WAV16K)
    flags, _rms, frame = vad_flags(pcm)
    spans = speech_spans(flags, frame, min_s=0.45, join_gap=0.30)
    cue_iv2 = [(a, b) for a, b, _st, _tx in cues]
    residual = []
    for a, b in spans:
        if b < 2.0 or voice_band_ratio(pcm, a, b) < 0.45 or segment_rms_db(pcm, a, b) < -42:
            continue
        cur = a
        while cur < b:
            cov = any(c[0] - 0.30 <= cur <= c[1] + 0.30 for c in cue_iv2)
            if not cov:
                t2 = cur
                while t2 < b and not any(c[0] - 0.30 <= t2 <= c[1] + 0.30 for c in cue_iv2):
                    t2 += 0.05
                if t2 - cur >= 1.0:
                    residual.append((round(cur, 2), round(t2, 2)))
                cur = t2 + 0.05
            else:
                cur += 0.05
    check(
        "no_speech_without_caption",
        not residual,
        f"{len(spans)} VAD spans checked on the master clock; "
        f"uncovered >=1.0s: {residual[:4] or 'none'}",
    )

    # ---- 11 caption presence: pixels during cue, clean after
    stuck, skipped = [], 0
    cue_starts = sorted(a for a, _b, _st, _tx in cues)
    for a, b, _st, tx in sample[:24]:
        nxt = min((s for s in cue_starts if s >= b - 0.01), default=None)
        if not (nxt is not None and nxt - b >= 0.30):
            skipped += 1
            continue
        if "*" in tx:
            skipped += 1
            continue
        after_t = b + min(0.40, (nxt - b) / 2)
        fr_in = _caption_added_colors(max(a + 0.10, b - 0.15))
        fr_out = _caption_added_colors(after_t)
        if fr_in is None or fr_out is None:
            continue
        m_in = sum(
            v for k, v in fr_in.items() if not k.startswith("bright") and not k.startswith("gray_p")
        )
        m_out = sum(
            v
            for k, v in fr_out.items()
            if not k.startswith("bright") and not k.startswith("gray_p")
        )
        if m_in < 100:
            stuck.append(f"missing@{a:.1f}s (mass {m_in:.0f})")
        elif m_out > max(60.0, 0.5 * m_in):
            stuck.append(f"stuck@{b:.1f}s (in {m_in:.0f} out {m_out:.0f})")
    check(
        "caption_presence_no_stuck",
        not stuck,
        f"{24 - skipped} cues presence-checked ({skipped} skipped: no "
        f"isolation gap), problems: {stuck[:3] or 'none'}",
    )

    # ---- 12 narrator-on-b-roll: no speaker cue overlaps a narrator cue
    narr = [(a, b) for a, b, st, _tx in cues if st == "Narrator"]
    speak = [(a, b, st) for a, b, st, _tx in cues if st in ("Officer", "Suspect")]
    viol = []
    for na, nb in narr:
        for sa, sb, st in speak:
            ov = min(nb, sb) - max(na, sa)
            if ov > 0.15:
                viol.append(f"{st}[{sa:.1f}-{sb:.1f}] overlaps Narrator[{na:.1f}-{nb:.1f}]")
    check(
        "narrator_on_broll",
        not viol,
        f"{len(narr)} narrator cues; speaker overlaps: {viol[:3] or 'none'}",
    )

    # ---- 13 font legibility: glyph height >= 26 px at 720p (caption band only)
    heights = []
    for a, b, _st, tx in sample[:16]:
        if "*" in tx:
            continue
        mask = _caption_added_frame(max(a + 0.10, b - 0.15))
        if mask is None:
            continue
        band = mask[560:720, :]  # caption band: zoom/card diffs cannot leak in
        ys, _xs = np.nonzero(band)
        if len(ys) >= 60:
            heights.append(int(ys.max() - ys.min()))
    min_h = min(heights) if heights else 0
    check(
        "font_legibility",
        len(heights) >= 6 and min_h >= 26,
        f"{len(heights)} measured cue glyph heights (caption band), min "
        f"{min_h}px (>=26px at 720p; v5 regression was ~13px)",
    )

    failed = [c for c in checks if not c["ok"]]
    log(f"QA v6: {len(checks) - len(failed)}/{len(checks)} green")
    if failed:
        raise SystemExit(f"QA FAILED: {[c['name'] for c in failed]}")


if __name__ == "__main__":
    main()
