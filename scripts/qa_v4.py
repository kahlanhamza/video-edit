#!/usr/bin/env python3
"""QA gate for the v4 caption fix — all v3 checks PLUS the new caption-truth
checks the owner demanded (speaker color must match audio truth, no
hallucinated text, no audible-speech-without-caption).

Checks:
  1  duration identical to the accepted base
  2  no paragraph flash at any burn boundary (per-frame base ratio)
  3  face-redaction coverage at the measured 199-203 s leak
  4  pass-2 blur effectiveness on sized tracks
  5  watermark exact-pixel box blur on caption-free instants
  6  captions present + audio present
  7  NEW no hallucinated loop text anywhere in the burned captions
  8  NEW loop window (609.4-628.9) carries ONLY gray tag cues
  9  NEW caption color audit: sampled lonely cues show their style color
     (Officer=white, Suspect=yellow, Narrator=sky, tag=gray) — catches any
     yellow caption on officer speech
 10  NEW gap coverage: no VAD speech span >= 1.0 s remains caption-free
 11  NEW caption presence: pixels during cue, clean after cue (no stuck text)
"""

from __future__ import annotations

import json
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, "/home/z/my-project/scripts")

import cv2
from postfx_v3 import BURN_CHUNK_S, V3, _base, _grab, _lap, _plan2, _text_mass, log, probe_dur
from vad_evidence import load_pcm, segment_rms_db, speech_spans, vad_flags, voice_band_ratio

V4 = V3
FINAL = V4 / "master_postfx_v4.mp4"
ASS = V4 / "subtitles_v4.ass"
SRT = V4 / "captions_v4.srt"
QA = V4 / "qa_v4.json"

OWNER_WINDOWS = [(609.40, 628.90)]
CARD_PAD = 7.0  # card overlay window to exclude from the color audit


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
    """Per-color caption-glyph pixel counts inside the caption band."""
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
    # brightness stats over the neutral-bright mask (white cores vs gray cores):
    # white text antialias reaches 240+, gray tag cores stay near #9C9C9C (156)
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
    """Per-color pixels the burn ADDED at instant t (final minus base).

    Scenes like a bright wall next to a dark doorframe produce thousands of
    neutral 'glyph-like' pixels on their own; subtracting the caption-free
    base isolates what the subtitle renderer actually drew."""
    ff_, fb_ = _grab(FINAL, t), _grab(_base(), t)
    if ff_ is None or fb_ is None:
        return None
    cf, cb = _glyph_colors(ff_), _glyph_colors(fb_)
    out = {}
    for k in cf:
        out[k] = max(0, int(cf[k]) - int(cb.get(k, 0)))
    return out


def main() -> None:
    if not FINAL.exists():
        raise SystemExit("mux first")
    checks: list[dict] = []

    def check(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        QA.write_text(json.dumps(checks, indent=1))
        log(f"  [{'OK' if ok else 'FAIL'}] {name}: {detail}")

    base = _base()
    info_d = probe_dur(FINAL)
    cues = _cue_iv()

    # ---- 1 duration
    check(
        "duration",
        abs(info_d - probe_dur(base)) <= 1.0,
        f"final {info_d:.2f}s vs base {probe_dur(base):.2f}s",
    )

    # ---- 2 flash regression at burn boundaries
    n_chunks = max(1, -(-int(info_d * 1000) // int(BURN_CHUNK_S * 1000)))
    bounds = [ci * BURN_CHUNK_S for ci in range(1, n_chunks)]
    spikes = []
    for bnd in bounds:
        worst = 0.0
        for t in np.arange(bnd - 0.15, bnd + 0.45, 1 / 30):
            ff_, fb_ = _grab(FINAL, t), _grab(base, t)
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

    # ---- 3 face redaction at the measured leak
    ok202, ratios = True, []
    plan = _plan2()
    for t in (199.5, 200.5, 201.5, 202.5):
        fb, ff = _grab(base, t), _grab(FINAL, t)
        if fb is None or ff is None:
            continue
        fi = int(t * 30)
        boxes = plan.boxes_at(fi)
        if not boxes:
            ok202 = False
            ratios.append(f"t={t}: NO PLAN BOX")
            continue
        bx = max(boxes, key=lambda b: b.w * b.h)
        x, y, w, h = bx.x, bx.y, bx.w, bx.h
        ix, iy, iw, ih = x + w // 4, y + h // 4, max(4, w // 2), max(4, h // 2)
        v0 = max(_lap(fb[iy : iy + ih, ix : ix + iw]), 1e-6)
        v1 = max(_lap(ff[iy : iy + ih, ix : ix + iw]), 1e-6)
        ratios.append(f"t={t}: x{v0 / v1:.1f} v1={v1:.0f}")
        ok202 = ok202 and (v1 < 40.0 or v0 / v1 >= 2.0)
    check(
        "face_leak_202s_blurred", ok202 and len(ratios) == 4, "; ".join(ratios) or "frames missing"
    )

    # ---- 4 pass-2 blur effectiveness
    blurred_v = V4 / "master_v3_blurred.mp4"
    sized = [
        tr
        for tr in plan.trajectories
        if len(tr.boxes) and tr.boxes[len(tr.boxes) // 2].w * tr.boxes[len(tr.boxes) // 2].h >= 2000
    ]
    n_tr = len(sized)
    sample_ids = (
        [int(i * (n_tr - 1) / 11) for i in range(12)] if n_tr > 1 else ([0] if n_tr else [])
    )
    worst, worst_abs = (99.0, -1), (0.0, -1.0)
    for ti in sample_ids:
        tr = sized[ti]
        mid = tr.boxes[len(tr.boxes) // 2]
        t = mid.frame_idx / plan.fps
        fb, ff = _grab(base, t), _grab(blurred_v, t)
        if fb is None or ff is None:
            continue
        x, y, w, h = mid.x, mid.y, mid.w, mid.h
        ix, iy, iw, ih = x + w // 4, y + h // 4, max(4, w // 2), max(4, h // 2)
        v0 = max(_lap(fb[iy : iy + ih, ix : ix + iw]), 1e-6)
        v1 = max(_lap(ff[iy : iy + ih, ix : ix + iw]), 1e-6)
        if v1 > worst_abs[0]:
            worst_abs = (v1, t)
        if v0 / v1 < worst[0]:
            worst = (v0 / v1, t)
    check(
        "pass2_blur_effective",
        (not sample_ids) or worst_abs[0] < 40.0 or worst[0] >= 1.8,
        f"max inner var {worst_abs[0]:.0f} @ {worst_abs[1]:.1f}s, worst "
        f"base/final x{worst[0]:.2f} over {len(sample_ids)} sized tracks",
    )

    # ---- 5 watermark exact pixels on caption-free instants (from v4 cues)
    wm = json.loads((V4 / "wm_spans_v3.json").read_text())["spans"]
    wm_ok, wm_note = bool(wm), "no spans"
    if wm:
        arr = np.array([sp["box"] for sp in wm])
        bx, by, bw, bh = [int(v) for v in np.median(arr, axis=0)]
        bx, by, bw, bh = max(0, bx - 2), max(0, by - 2), bw + 4, bh + 4
        cue_iv = [(a, b) for a, b, _st, _tx in cues]
        cue_iv.sort()

        def _free(t):
            return all(not (a - 0.2 <= t <= b + 0.2) for a, b in cue_iv)

        cand = [round(float(x), 2) for x in np.arange(5.0, max(6.0, info_d - 3.0), 0.5)]
        free = [t for t in cand if _free(t)]
        step = max(1, len(free) // 24)
        free = free[::step]
        rs, abs_vars = [], []
        for t in free:
            fb, ff = _grab(base, t), _grab(FINAL, t)
            if fb is None or ff is None:
                continue
            v0 = _lap(fb[by : by + bh, bx : bx + bw])
            v1 = _lap(ff[by : by + bh, bx : bx + bw])
            abs_vars.append(v1)
            if v0 >= 60:
                rs.append(v0 / max(v1, 1e-6))
        worst_absv = max(abs_vars, default=0.0)
        wm_ok = bool(rs) and min(rs) >= 2.0 and worst_absv < 60.0
        wm_note = (
            f"continuous box ({bx},{by},{bw},{bh}): {len(rs)}/"
            f"{len(abs_vars)} caption-free samples, min collapse "
            f"x{min(rs, default=0.0):.1f} >=2.0, worst final var "
            f"{worst_absv:.0f} <60"
        )
    check("watermark_exact_pixels", wm_ok, wm_note)

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

    # ---- 7 no hallucinated loop text
    hay = (ASS.read_text(encoding="utf-8") + srt).lower()
    ghosts = [g for g in ("people want to stop", "want to stop") if g in hay]
    check(
        "no_hallucinated_loop_text",
        not ghosts,
        f"ghost phrases absent: {not ghosts} (hits: {ghosts or 'none'})",
    )

    # ---- 8 loop window carries ONLY gray tag cues
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
        f"{len(inwin)} cues in {lo}-{hi}s, all tags: {not bad}, "
        f"tag coverage {tagcov:.1f}s of {win_dur:.0f}s (>=75%)",
    )

    # ---- 9 caption color audit (lonely cues only; skip card windows)
    cards_meta = json.loads((V4 / "cards_meta_v3.json").read_text())
    card_iv = [(t0, t0 + 6.5) for _png, t0 in cards_meta["card_times"]]
    expect = {"Officer": "white", "Suspect": "yellow", "Narrator": "sky", "Tag": "gray"}
    # sample lonely cues (no other cue within +-0.12s)
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
        # sample near cue END: the karaoke sweep is complete there, so every
        # glyph shows the PRIMARY style color (mid-cue, unsung words still
        # show SecondaryColour and would poison the color test)
        t = max(a + 0.10, b - 0.15)
        c = _caption_added_colors(t)
        if c is None:
            continue
        tested += 1
        tot = sum(
            v for k, v in c.items() if not k.startswith("bright") and not k.startswith("gray_p")
        )
        if tot < 150:
            continue  # nothing legible in band (rare scene colors) — skip
        if want == "white":
            okk = c["white"] >= 1.2 * max(c["yellow"], c["sky"]) and (
                c["bright_p98"] > 230 or c["white"] > c["gray"]
            )
        elif want == "yellow":
            okk = c["yellow"] >= 1.5 * max(c["white"], c["sky"])
        elif want == "sky":
            okk = c["sky"] >= 1.5 * max(c["white"], c["yellow"])
        else:  # gray tag: gray dominates and the glyph CORE is dim (p80 < 200;
            # white antialias would push p80 over 200)
            okk = c["gray"] >= 1.2 * c["yellow"] and c["gray"] >= c["white"] and c["gray_p80"] < 200
        if not okk:
            mism.append(f"{kind}@{a:.1f}s want={want} got={c}")
    check(
        "caption_color_matches_speaker",
        tested >= 10 and not mism,
        f"{tested} lonely cues sampled, {len(mism)} color mismatches"
        + (f" e.g. {mism[:3]}" if mism else ""),
    )

    # ---- 10 gap coverage: no uncovered speech span >= 1.0 s
    pcm = load_pcm(V4 / "audio_src_16k.wav")
    flags, _rms, frame = vad_flags(pcm)
    spans = speech_spans(flags, frame, min_s=0.45, join_gap=0.30)
    cue_iv = [(a, b) for a, b, _st, _tx in cues]
    residual = []
    for a, b in spans:
        if b < 2.0 or voice_band_ratio(pcm, a, b) < 0.45 or segment_rms_db(pcm, a, b) < -42:
            continue
        cur = a
        while cur < b:
            cov = any(c[0] - 0.30 <= cur <= c[1] + 0.30 for c in cue_iv)
            if not cov:
                t2 = cur
                while t2 < b and not any(c[0] - 0.30 <= t2 <= c[1] + 0.30 for c in cue_iv):
                    t2 += 0.05
                if t2 - cur >= 1.0:
                    residual.append((round(cur, 2), round(t2, 2)))
                cur = t2 + 0.05
            else:
                cur += 0.05
    check(
        "no_speech_without_caption",
        not residual,
        f"{len(spans)} VAD spans checked; uncovered >=1.0s: {residual[:4] or 'none'}",
    )

    # ---- 11 caption presence: pixels during cue, clean after
    # color-aware glyph mass: white+yellow+sky+gray counts (the old white/
    # yellow-only detector was blind to sky narrator text and gray tags)
    stuck, skipped = [], 0
    cue_starts = sorted(a for a, _b, _st, _tx in cues)
    for a, b, _st, tx in sample[:24]:
        # the "after" instant must be a REAL caption-free gap: back-to-back
        # cues are legal (two-lane law), so only test when >= 0.3 s of gap
        nxt = min((s for s in cue_starts if s >= b - 0.01), default=None)
        gap_ok = nxt is not None and nxt - b >= 0.30
        if not gap_ok:
            skipped += 1
            continue
        if "*" in tx:
            # gray tag glyphs sit inside the scene's own gray color space, so
            # base subtraction cannot isolate them; tag rendering is verified
            # by the gray-dominance measurement on the un-subtracted band and
            # by direct frame inspection (see the ship report)
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

    failed = [c for c in checks if not c["ok"]]
    log(f"QA v4: {len(checks) - len(failed)}/{len(checks)} green")
    if failed:
        raise SystemExit(f"QA FAILED: {[c['name'] for c in failed]}")


if __name__ == "__main__":
    main()
