#!/usr/bin/env python3
"""POST-FX v3 — the seven owner findings from the 2026-11-20 review, fixed.

Owner findings (verbatim issues -> fix -> gate):
  1. face blur sometimes not applied even when the face is clear / rotated
     faces get jumpy.  ROOT CAUSE (measured): pass-1 swept at 0.5 s cadence on
     the WEAK packaged YuNet fallback (yolov8n-face.onnx was never fetched)
     and the entry gap is real: at t=202.5 the bearded officer's face is
     sharp from ~199 s while the plan only starts 202.83 s.
     FIX: pass-2 dense sweep on the assembled base — YOLOv8n-face @960px,
     conf 0.15, 0.25 s cadence, horizontal-flip augmentation (profiles),
     min_hits=2, lead 8 frames — rendered as a SECOND blur pass over the
     already-blurred base (double blur where pass-1 already covers, new blur
     where it leaked).  POLICY: redaction stays (public-release bodycam is
     safest monetized with identities redacted); flip REDACT_FACES=0 to ship
     faces visible on future cases.
  2. BIG PARAGRAPH of stacked captions flashes for ~2 frames.  ROOT CAUSE
     (measured, reproduced frame-exact): pipe_09b's make_shifted_ass clamps
     EVERY cue before the 330 s split boundary to t=0.00-0.05 instead of
     dropping it — all ~250 earlier cues render stacked at the chunk-B start.
     FIX: chunk ASS builder DROPS cues fully before the chunk start; only a
     genuinely-crossing cue is clamped.  QA gate re-decodes every burn-boundary
     window and fails on caption-mass spikes.
  3. case cards overflow (text runs outside the card).  FIX: content-fit card
     renderer — every line measured with PIL, wrapped by pixel width, box
     grows to the content; title/subtitle shrink until they fit.
  4. wrong Gladia words are never corrected.  FIX: LLM proofread pass (the
     notebook cell-5 technique) over every caption utterance with case facts;
     corrections are re-aligned to the ORIGINAL word timestamps (difflib:
     unchanged words keep exact times, changed spans interpolate inside the
     utterance span) so sync cannot drift.
  5. unintelligible captions shown as weird wrong text.  FIX: the proofreader
     returns "" for pure garble and the builder renders "*inaudible*" /
     "*mumbling*" / "*cross-talk*" (italic, dimmed) on the original timing.
  6. narration voice reads unit tokens literally ("0.391g/210L" -> "gee slash
     two ten el").  FIX: TTS text-normalization law (tts_norm.py: units,
     slashes, symbols -> spoken words) + surgical re-voice of the one Outro
     narration span that carries the token, fitted to the original span with
     edge-tts word-boundary timings so captions stay word-locked.
  7. narration/editing/typography craft.  FIX: typography upgraded (bigger
     captions, cleaner card layout, per-speaker colors kept), YouTube narration
     techniques encoded in pipe_03 prompt docs + tts_norm for the voice.

Iron rules kept: post-fx touches ONLY the assembled deliverable chain
(blur-chunk base -> pass-2 -> burn), never the 43-minute source; everything
derived from probes/timeline (no hardcoded video length).

Phases (each resumable, one process per call — the sandbox kills long runs):
  copy-src | sweep2-seg N | sweep2-done | plan2 | chunk2 N | concat2 |
  wm | captions | cards | burn N | burnconcat | voice | mux | qa | ship
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/z/my-project/director-os")
for _p in sorted(ROOT.glob("packages/*/src")):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

WORK = Path("/home/z/my-project/pipeline_run")
V3 = WORK / "v3"
OLD_BLUR = WORK / "blur_run"
SHIP_PACK = Path("/home/z/my-project/download/eustis_acceptance_ship_pack")
SHIPPED_FINAL = SHIP_PACK / "final_720p.mp4"

FPS = 30
BLUR_CHUNK_FRAMES = 3600  # 120 s per pass-2 chunk
SWEEP_SAMPLE_S = 0.25  # pass-2 detection cadence
SWEEP_CONF = 0.15
SWEEP_IMGSZ = 960
SWEEP_SEG_S = 120.0  # sweep segment length (resumable)
MIN_HITS = 2  # real detections required to keep a track
LEAD_FRAMES = 8  # 0.27 s lead-in/out per window
BRIDGE_STRIDE_FRAMES = 8  # round(30 * 0.25)

BURN_CHUNK_S = 80.0  # flash-fix burn chunks (sized for the sandbox process window)
REDACT_FACES = os.environ.get("REDACT_FACES", "1") == "1"

CAPTION_FONTSIZE = 40
CAPTION_MARGIN_V = 42
FONT = "DejaVu Sans"

# Case facts for the caption proofreader (any-case pipeline: replace per case)
CASE_FACTS = {
    "case": "Eustis, Florida — July 4, 2022, on-duty DUI arrest",
    "people": [
        "Officer Oscar Mayorga — the impaired officer being arrested",
        "Sgt. M. Neal — supervising officer",
        "Ofc. J. Wilson — assisting officer",
    ],
    "facts": [
        "smell of alcohol, open Modelo tall-boy cans in the patrol truck",
        "claims Benadryl / allergy medication, then admits three tall boys of Modelo",
        "refuses field sobriety tests and the breathalyzer",
        "suppressed without pay, criminally charged; BAC later confirmed 0.391 g/210L",
        "police radio codes: 10-8, 10-15, EMS, en route, SFST",
    ],
}


def log(msg: str) -> None:
    print(msg, flush=True)


def _sh(cmd: list[str], what: str) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{what} failed: {r.stderr[-500:]}")


def probe_dur(path: Path) -> float:
    return float(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ]
        )
        .decode()
        .strip()
    )


def probe_vstream(path: Path):
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames",
            "-of",
            "json",
            str(path),
        ]
    ).decode()
    st = json.loads(out)["streams"][0]
    num, den = st["avg_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else FPS
    return int(st["width"]), int(st["height"]), fps, int(st.get("nb_frames") or 0)


# ------------------------------------------------------------------ copy-src
def phase_copy_src() -> None:
    """Concat the surviving pass-1 blur chunks -> video base; extract audio."""
    V3.mkdir(parents=True, exist_ok=True)
    base = V3 / "base_v3.mp4"
    if base.exists() and (V3 / "copy_src.done").exists():
        log("copy-src cached")
        return
    chunks = sorted(OLD_BLUR.glob("blur_chunk_*.mp4"))
    if not chunks:
        raise SystemExit("no pass-1 blur chunks found")
    lst = V3 / "base_chunks.txt"
    lst.write_text("".join(f"file '{c}'\n" for c in chunks))
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
            "-an",
            str(base),
        ],
        "base concat",
    )
    # audio: the shipped final's audio track is the accepted 2-bus mix — reuse
    # it bit-exactly (AAC stream copy; no re-encode, zero audio risk)
    aud = V3 / "audio_src.m4a"
    if not aud.exists():
        _sh(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(SHIPPED_FINAL),
                "-map",
                "0:a:0",
                "-c:a",
                "copy",
                str(aud),
            ],
            "audio extract",
        )
    w, h, fps, nfr = probe_vstream(base)
    (V3 / "copy_src.done").write_text(f"{w}x{h}@{fps:.2f} n={nfr}")
    log(
        f"base_v3: {w}x{h} {fps:.2f}fps {probe_dur(base):.2f}s "
        f"({len(chunks)} chunks) + audio {probe_dur(aud):.2f}s"
    )


def _base() -> Path:
    p = V3 / "base_v3.mp4"
    if not p.exists():
        raise SystemExit("base missing — run copy-src")
    return p


# ------------------------------------------------------------------- sweep2
def _yolo_detector():
    from director_os.vision_fx.compute.yolo_onnx import YoloOnnxFaceDetector

    model = ROOT / "data" / "models" / "yolov8n-face-lindevs.onnx"
    if not model.exists():
        raise SystemExit("yolov8n-face model missing — run scripts/fetch_yolov8n_face.py")
    return YoloOnnxFaceDetector(model, conf_threshold=SWEEP_CONF, input_size=SWEEP_IMGSZ)


def _decode_segment(video: Path, t0: float, t1: float):
    """ffmpeg rawvideo pipe -> (frame_idx, bgr) for [t0, t1)."""

    w, h, _fps, _n = probe_vstream(video)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{t0:.3f}",
        "-t",
        f"{t1 - t0:.3f}",
        "-i",
        str(video),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    f0 = int(round(t0 * FPS))
    idx = f0
    while True:
        buf = proc.stdout.read(w * h * 3)
        if len(buf) < w * h * 3:
            break
        yield idx, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        idx += 1
    proc.stdout.close()
    proc.wait()


def phase_sweep2_seg(ci: int) -> None:
    """Dense pass-2 face sweep, segment ci (resumable; flip-augmented)."""
    import cv2

    out = V3 / f"sweep2_seg_{ci:02d}.json"
    if out.exists():
        log(f"sweep2 seg {ci} cached")
        return
    video = _base()
    dur = probe_dur(video)
    t0 = ci * SWEEP_SEG_S
    t1 = min(dur, t0 + SWEEP_SEG_S)
    if t0 >= dur:
        raise SystemExit(f"segment {ci} beyond video end ({dur:.1f}s)")
    log(
        f"sweep2 seg {ci}: decoding [{t0:.0f}s -> {t1:.0f}s] (expect "
        f"{int((t1 - t0) / SWEEP_SAMPLE_S)} samples)"
    )
    det = _yolo_detector()
    frame_stride = max(1, int(round(FPS * SWEEP_SAMPLE_S)))
    prev_gray = prev_pts = None
    dets, ts, motions = [], [], []
    import time as _time

    _t0 = _time.time()
    n_samples = 0
    for idx, frame in _decode_segment(video, t0, t1):
        if idx % frame_stride:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None and prev_pts is not None and len(prev_pts) >= 10:
            nxt, st, _e = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
            good = st.ravel() == 1
            if good.sum() >= 8:
                med = np.median((nxt[good] - prev_pts[good]).reshape(-1, 2), axis=0)
                motions.append((float(med[0]), float(med[1])))
            else:
                motions.append((0.0, 0.0))
        else:
            motions.append((0.0, 0.0))
        pts = cv2.goodFeaturesToTrack(
            gray, maxCorners=200, qualityLevel=0.01, minDistance=8, blockSize=7
        )
        prev_gray, prev_pts = gray, pts
        hits = det.detect(frame)
        flipped = cv2.flip(frame, 1)
        W = frame.shape[1]
        for x, y, w, h, s in det.detect(flipped):
            # un-mirror the box
            hits.append((W - x - w, y, w, h, s * 0.95))
        dets.append([(int(x), int(y), int(w), int(h), float(s)) for x, y, w, h, s in hits])
        ts.append(round(idx / FPS, 3))
        n_samples += 1
        if n_samples % 80 == 0:
            log(
                f"  seg {ci}: {n_samples} samples @ {idx / FPS:.0f}s "
                f"({_time.time() - _t0:.0f}s elapsed)"
            )
    out.write_text(json.dumps({"dets": dets, "ts": ts, "motions": motions}))
    n = sum(len(d) for d in dets)
    log(f"sweep2 seg {ci}: {len(dets)} samples, {n} raw hits")


def phase_sweep2_done() -> None:
    """Track + interpolate across all segments -> face_tracks2.json."""
    from director_os.vision_fx.compute.tracker import track_and_interpolate

    out = V3 / "face_tracks2.json"
    if out.exists():
        log("sweep2-done cached")
        return
    segs = sorted(V3.glob("sweep2_seg_*.json"))
    if not segs:
        raise SystemExit("no sweep segments — run sweep2-seg first")
    dets, ts, motions = [], [], []
    for s in segs:
        d = json.loads(s.read_text())
        dets.extend(d["dets"])
        ts.extend(d["ts"])
        motions.extend(d["motions"])
    order = np.argsort(ts)
    dets = [dets[i] for i in order]
    ts = [ts[i] for i in order]
    motions = [motions[i] for i in order]
    # remove duplicate timestamps at segment overlaps (defensive)
    seen, dd, tt, mm = set(), [], [], []
    for d, t, m in zip(dets, ts, motions, strict=True):
        if t in seen:
            continue
        seen.add(t)
        dd.append(d)
        tt.append(t)
        mm.append(tuple(m))
    tracks = track_and_interpolate(
        dd,
        tt,
        gmc_vectors=mm,
        track_buffer=60,
        min_hits=MIN_HITS,
    )
    keep = []
    for t in tracks:
        real = [h for h in t.hits if not h.interpolated]
        # drop tracks whose boxes are microscopic (detector noise)
        if real and max(h.w * h.h for h in real) < 120:
            continue
        keep.append(t)
    out.write_text(json.dumps([t.model_dump(mode="json") for t in keep]))
    spans = sorted(
        (h["hits"][0]["t_s"], h["hits"][-1]["t_s"])
        for h in (t.model_dump(mode="json") for t in keep)
        if h["hits"]
    )
    log(
        f"sweep2: {len(keep)} tracks (min_hits={MIN_HITS}); "
        f"first spans {[(round(a, 1), round(b, 1)) for a, b in spans[:6]]}"
    )


# --------------------------------------------------------------------- plan2
def phase_plan2() -> None:
    """repair_trajectories with the DENSE sweep stride declared."""
    from director_os.vision_fx.compute.base import FaceTrack
    from director_os.vision_fx.trajectory import repair_trajectories

    video = _base()
    w, h, fps, nfr = probe_vstream(video)
    if nfr <= 0:
        nfr = int(round(probe_dur(video) * fps))
    tracks = [
        FaceTrack.model_validate(t) for t in json.loads((V3 / "face_tracks2.json").read_text())
    ]
    plan = repair_trajectories(
        tracks,
        fps=fps,
        frame_w=w,
        frame_h=h,
        n_frames=nfr,
        sample_stride_frames=BRIDGE_STRIDE_FRAMES,
        lead_frames=LEAD_FRAMES,
        min_window_frames=2,
    )
    if not REDACT_FACES:
        # policy flag (owner question 2026-11-21: "does public material really
        # need to hide the face?"). Public-release bodycam ships faces visible;
        # the watermark band (burn phase) is unaffected. Nothing downstream
        # redacts faces when the flag is 0.
        plan = plan.model_copy(update={"trajectories": []})
        (V3 / "plan2.json").write_text(plan.model_dump_json())
        log("REDACT_FACES=0 — face trajectories dropped (watermark-only policy)")
        return
    (V3 / "plan2.json").write_text(plan.model_dump_json())
    # density law: any 1..bridge-budget hole is a flicker
    budget = max(25, int(round(2.5 * BRIDGE_STRIDE_FRAMES)))
    problems = []
    for tr in plan.trajectories:
        idx = [b.frame_idx for b in tr.boxes]
        for a, b in zip(idx, idx[1:], strict=False):
            if 1 < b - a <= budget:
                problems.append(f"{tr.track_id}: {b - a}-frame hole at {a}")
                break
    if problems:
        raise SystemExit(f"PASS-2 PLAN NOT DENSE: {problems[:5]}")
    from director_os.vision_fx.redact_track import audit_coverage

    leaks = audit_coverage(plan, tracks, plan.fps)
    if leaks:
        raise SystemExit(f"PASS-2 REDACTION LEAK: {len(leaks)} raw detections uncovered")
    total_boxes = sum(len(t.boxes) for t in plan.trajectories)
    log(
        f"plan2: {len(plan.trajectories)} trajectories, {total_boxes} dense boxes, "
        f"dense + zero-leak, {nfr} frames"
    )


def _plan2():
    from director_os.vision_fx.trajectory import PerFrameRedaction

    return PerFrameRedaction.model_validate_json((V3 / "plan2.json").read_text())


# --------------------------------------------------------------------- chunk2
def phase_chunk2(ci: int) -> None:
    from director_os.vision_fx.redact_track import render_tracked_redaction

    plan = _plan2()
    n_chunks = max(1, -(-plan.n_frames // BLUR_CHUNK_FRAMES))
    if not 0 <= ci < n_chunks:
        raise SystemExit(f"chunk2 {ci} out of range 0..{n_chunks - 1}")
    part = V3 / f"p2_chunk_{ci:02d}.mp4"
    mark = V3 / f"p2_chunk_{ci:02d}.done"
    if mark.exists() and part.exists():
        log(f"chunk2 {ci} cached")
        return
    f0, f1 = ci * BLUR_CHUNK_FRAMES, min((ci + 1) * BLUR_CHUNK_FRAMES, plan.n_frames)
    render_tracked_redaction(
        _base(),
        plan,
        part,
        mode="gauss",
        sigma=24,
        crf=18,
        preset="veryfast",
        frame_start=f0,
        frame_end=f1,
    )
    mark.write_text(str(f1 - f0))
    log(f"chunk2 {ci + 1}/{n_chunks} rendered ({f1 - f0} frames)")


def phase_concat2() -> None:
    plan = _plan2()
    n_chunks = max(1, -(-plan.n_frames // BLUR_CHUNK_FRAMES))
    parts = []
    for ci in range(n_chunks):
        part = V3 / f"p2_chunk_{ci:02d}.mp4"
        if not ((V3 / f"p2_chunk_{ci:02d}.done").exists() and part.exists()):
            raise SystemExit(f"chunk2 {ci} missing")
        parts.append(part)
    out = V3 / "master_v3_blurred.mp4"
    lst = V3 / "p2_chunks.txt"
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
            str(out),
        ],
        "pass-2 concat",
    )
    log(f"master_v3_blurred: {probe_dur(out):.2f}s, {out.stat().st_size / 1e6:.1f} MB")


# ----------------------------------------------------------------------- wm
def phase_wm() -> None:
    """Multi-position watermark presence sweep (fully local, template matching).

    The v1 chain blurred ONE static box all video long — scenery got blurred
    when the mark was absent, and the mark was missed at its other positions
    ((710,656)/(710,571)/(609,540) measured). This sweep finds WHERE and WHEN
    the mark is present at 1 s resolution with multi-scale template matching,
    clusters hits into spans, and emits per-span constant boxes (zero
    within-span flicker). Generic: template is cut from the video itself.
    """
    import cv2

    out = V3 / "wm_spans_v3.json"
    if out.exists():
        log("wm cached")
        return
    video = _base()
    dur = probe_dur(video)
    # template: crop the mark at its known canonical box on a frame where the
    # v1 QA measured it sharp (43s region, watermark band)
    cap = cv2.VideoCapture(str(video))
    canon = [719, 658, 552, 58]
    tmpl = None
    for t in (43.2, 100.0, 200.0, 330.0):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        if ok:
            x, y, w, h = canon
            crop = f[y : y + h, x : x + w]
            g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if float(g.std()) > 18:  # text present (not already blurred)
                tmpl = g
                break
    if tmpl is None:
        cap.release()
        out.write_text(json.dumps({"spans": [], "note": "template not found"}))
        log("wm: no sharp template frame found — nothing to blur?")
        return

    scales = [0.8, 0.9, 1.0, 1.1, 1.25]
    hits: list[tuple[float, list[int]]] = []
    times = list(np.arange(0.5, dur, 1.0))
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        best = (0.0, None)
        for sc in scales:
            tw, th = max(20, int(canon[2] * sc)), max(10, int(canon[3] * sc))
            if tw >= gray.shape[1] or th >= gray.shape[0]:
                continue
            t2 = cv2.resize(tmpl, (tw, th))
            # search the bottom 45% of the frame (every measured position)
            y0 = int(gray.shape[0] * 0.55)
            region = gray[y0:, :]
            if t2.shape[0] >= region.shape[0] or t2.shape[1] >= region.shape[1]:
                continue
            res = cv2.matchTemplate(region, t2, cv2.TM_CCOEFF_NORMED)
            _mn, mx, _mnloc, mxloc = cv2.minMaxLoc(res)
            if mx > best[0]:
                best = (mx, [mxloc[0], mxloc[1] + y0, tw, th])
        if best[0] >= 0.55 and best[1] is not None:
            hits.append((float(t), best[1]))
    cap.release()
    log(f"wm sweep: {len(hits)}/{len(times)} sampled frames show the mark")

    # cluster into spans (gap tolerance 4 s), per-span median box; the sweep
    # walks at 1 s steps so a cluster's first-seen time is reconstructed from
    # its hit count (last_seen - (n-1)*1s), then padded 1 s / 1.5 s
    spans: list[dict] = []
    for t, box in hits:
        if spans and t - spans[-1]["last"] <= 4.0:
            spans[-1]["boxes"].append(box)
            spans[-1]["last"] = t
        else:
            spans.append({"last": t, "boxes": [box]})
    result = []
    for sp in spans:
        arr = np.array(sp["boxes"])
        med = [int(v) for v in np.median(arr, axis=0)]
        a_t = max(0.0, sp["last"] - 1.0 * (len(sp["boxes"]) - 1) - 1.0)
        b_t = min(dur, sp["last"] + 1.5)
        result.append({"t_start_s": round(a_t, 2), "t_end_s": round(b_t, 2), "box": med})
    out.write_text(json.dumps({"spans": result}, indent=1))
    for sp in result:
        log(f"   [{sp['t_start_s']:7.1f} -> {sp['t_end_s']:7.1f}] box={sp['box']}")


# ----------------------------------------------------------------- captions
_KARAOKE = re.compile(r"\{\\k(\d+)\}")


def _norm_tok(w: str) -> str:
    return re.sub(r"[^a-z0-9]", "", w.lower())


def _align_words(new_text: str, orig_words: list[dict]) -> list[dict]:
    """difflib alignment of corrected text onto ORIGINAL word timestamps.

    Unchanged words keep their exact measured times (sync law); a replaced
    span redistributes the span's time across its new tokens by token length.
    """
    import difflib

    tokens = new_text.split()
    if not tokens:
        return []
    if not orig_words:
        total = max(0.35, sum(w["e"] - w["s"] for w in orig_words) or 1.0)
        weights = [max(2, len(t)) for t in tokens]
        sw, cur, res = sum(weights), orig_words[0]["s"] if orig_words else 0.0, []
        for t, wt in zip(tokens, weights, strict=False):
            d = total * (wt / sw)
            res.append({"w": t, "s": cur, "e": cur + d})
            cur += d
        return res
    sm = difflib.SequenceMatcher(
        None,
        [_norm_tok(w["w"]) for w in orig_words],
        [_norm_tok(t) for t in tokens],
        autojunk=False,
    )
    timed: list[dict | None] = [None] * len(tokens)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(j2 - j1):
                ow = orig_words[i1 + k]
                timed[j1 + k] = {
                    "w": tokens[j1 + k],
                    "s": ow["s"],
                    "e": max(ow["e"], ow["s"] + 0.06),
                }
        elif tag == "replace" and i2 > i1:
            t0 = orig_words[i1]["s"]
            t1 = max(orig_words[i2 - 1]["e"], t0 + 0.14)
            sub = tokens[j1:j2]
            weights = [max(2, len(x)) for x in sub]
            sw, cur = sum(weights), t0
            for k, (tok, wt) in enumerate(zip(sub, weights, strict=False)):
                d = (t1 - t0) * (wt / sw)
                timed[j1 + k] = {"w": tok, "s": cur, "e": cur + d}
                cur += d
        elif tag == "insert" and i2 > i1:
            t0 = orig_words[i1]["s"]
            sub = tokens[j1:j2]
            weights = [max(2, len(x)) for x in sub]
            sw, cur = sum(weights), t0
            step = min(0.10, 0.6 / max(1, len(sub)))
            for k, tok in enumerate(sub):
                timed[j1 + k] = {"w": tok, "s": cur, "e": cur + step}
                cur += step
        elif tag == "delete" and j2 > j1:
            t0 = orig_words[i1]["s"] if i2 > i1 else orig_words[-1]["e"]
            step = min(0.10, 0.6 / max(1, j2 - j1))
            cur = t0
            for k in range(j1, j2):
                timed[k] = {"w": tokens[k], "s": cur, "e": cur + step}
                cur += step
    # fill any residual Nones inside the utterance span
    span_s = orig_words[0]["s"]
    span_e = max(w["e"] for w in orig_words)
    j = 0
    while j < len(timed):
        if timed[j] is None:
            je = j
            while je < len(timed) and timed[je] is None:
                je += 1
            pe = timed[j - 1]["e"] if j > 0 else span_s
            ns = timed[je]["s"] if je < len(timed) else span_e
            if ns <= pe + 0.06:
                pe = max(span_s, ns - 0.18 * (je - j))
            step = max(0.12 * (je - j), ns - pe) / (je - j)
            for k in range(j, je):
                timed[k] = {"w": tokens[k], "s": pe + (k - j) * step, "e": pe + (k - j + 1) * step}
        j += 1
    return [t for t in timed if t is not None]


def _llm_json_array(prompt: str, expect_json: bool = True) -> list | dict:
    sys.path.insert(0, "/home/z/my-project/scripts")
    from llm_bridge import llm_chat

    for attempt in range(3):
        try:
            txt = llm_chat(prompt, timeout=240)
            m = re.search(r"\[.*\]|\{.*\}", txt, re.S)
            if m:
                return json.loads(m.group(0))
            if not expect_json:
                return txt
        except Exception as e:  # noqa: BLE001
            log(f"  llm attempt {attempt + 1} failed: {e}")
    raise RuntimeError("LLM proofread failed after retries")


def phase_captions() -> None:
    """Proofread + rebuild karaoke captions from the word-locked events.

    Source of truth: asm_state.json events (role + word timings on the master
    clock). The LLM pass returns clean TEXT per utterance; words are then
    re-aligned onto the original timestamps (_align_words), so the caption can
    never drift out of sync even when every word is replaced.
    """
    state = json.loads((WORK / "asm_state.json").read_text())
    events = state["events"]
    corr_path = V3 / "proofread_v3.json"

    utts = []
    for i, ev in enumerate(events):
        words = ev.get("words") or []
        if not words:
            continue
        text = " ".join(w["w"] for w in words).strip()
        if not text:
            continue
        utts.append(
            {
                "id": i,
                "role": ev.get("role", "?"),
                "t": round(words[0]["s"], 2),
                "n_words": len(words),
                "text": text,
            }
        )

    # resume-aware: corr_path holds progress after EVERY batch (a failed batch
    # never re-bills the earlier ones); a partial file is continued, not trusted
    fixes: list[dict] = []
    if corr_path.exists():
        fixes = json.loads(corr_path.read_text())
        log(f"proofread progress: {len(fixes)}/{len(utts)} utterances cached")
    done_ids = {f["id"] for f in fixes}
    todo = [u for u in utts if u["id"] not in done_ids]
    if todo:
        log(f"proofreading {len(todo)}/{len(utts)} utterances via LLM (batches of 26)...")
        B = 26
        nb = math.ceil(len(utts) / B)
        for bi, i in enumerate(range(0, len(todo), B), 1):
            batch = todo[i : i + B]
            lines = "\n".join(
                json.dumps(
                    {"id": u["id"], "role": u["role"], "t": u["t"], "text": u["text"]},
                    ensure_ascii=False,
                )
                for u in batch
            )
            prompt = f"""Proofread raw ASR captions for a police bodycam documentary.
Case facts: {json.dumps(CASE_FACTS, ensure_ascii=False)}

Return ONLY a valid JSON array matching every input id (no prose, no fences):
[{{"id": <int>, "clean_text": "<corrected text>", "tag": null or "inaudible"|"mumbling"|"cross-talk"}}]

Rules:
- Fix obvious ASR mishearings from context (names, agencies, police codes, units like "0.391 g/210L" = breath-alcohol reading, "ten codes", "SFST", "EMS").
- Keep the speaker's words and dialect; do NOT paraphrase, do NOT add or remove meaning. Keep roughly the same length.
- Mask harsh profanity for YouTube monetization ("f***", "s***").
- If a line is pure garble, hallucinated, or unintelligible nonsense that a viewer could not make sense of: set clean_text to "" and set tag to "inaudible" (or "mumbling" if they trail off, "cross-talk" if two people overlap).
- Keep punctuation light. Never include timestamps.
Lines:
{lines}"""
            arr = _llm_json_array(prompt)
            by_id = {int(x["id"]): x for x in arr if isinstance(x, dict) and "id" in x}
            n_edits = 0
            for u in batch:
                fx = by_id.get(u["id"], {})
                clean = (fx.get("clean_text") or "").strip()
                if clean and clean != u["text"]:
                    n_edits += 1
                fixes.append({"id": u["id"], "clean_text": clean, "tag": fx.get("tag")})
            corr_path.write_text(json.dumps(fixes))
            log(f"  batch {bi}/{nb} done ({n_edits} edits, {len(fixes)}/{len(utts)} total)")
        log(f"proofread: {len(fixes)} utterances")
    fix_map = {f["id"]: f for f in fixes}

    # ---- apply corrections onto the word timeline
    changed = inaud = 0
    corrected_events = []
    for u in utts:
        ev = events[u["id"]]
        fx = fix_map.get(u["id"], {})
        tag = fx.get("tag")
        clean = fx.get("clean_text") or ""
        words = ev["words"]
        if tag or not clean:
            corrected_events.append(
                {"role": ev.get("role", "?"), "words": words, "tag": tag or "inaudible"}
            )
            inaud += 1
            continue
        if clean == u["text"]:
            corrected_events.append({"role": ev.get("role", "?"), "words": words, "tag": None})
            continue
        new_words = _align_words(clean, words)
        if new_words:
            changed += 1
            corrected_events.append({"role": ev.get("role", "?"), "words": new_words, "tag": None})
        else:
            corrected_events.append({"role": ev.get("role", "?"), "words": words, "tag": None})
    log(f"corrections applied: {changed} utterances rewritten, {inaud} tagged")

    # ---- chunking into caption cues (<=5 words / <=27 chars, same law as v1)
    raw_chunks = []
    for uid, ev in enumerate(corrected_events):
        style = "Narrator" if ev["role"].lower().startswith("narr") else ev["role"].capitalize()
        tag = ev.get("tag")
        if tag:
            s = ev["words"][0]["s"]
            e = max(ev["words"][-1]["e"], s + 0.5)
            raw_chunks.append((uid, style, [{"w": f"*{tag}*", "s": s, "e": e}]))
            continue
        cur = []
        for w in ev["words"]:
            if cur and w["s"] - cur[-1]["e"] > 1.0:
                raw_chunks.append((uid, style, cur))
                cur = []
            cur.append(w)
            if (
                len(cur) >= 5
                or sum(len(x["w"]) + 1 for x in cur) >= 27
                or (w["w"][-1] in ",.?!" and len(cur) >= 3)
            ):
                raw_chunks.append((uid, style, cur))
                cur = []
        if cur:
            raw_chunks.append((uid, style, cur))
    raw_chunks.sort(key=lambda x: x[2][0]["s"])

    # ---- two-lane overlap placement (v1 law, kept)
    placed, last_idx = [], [None, None]
    for _uid, style, c in raw_chunks:
        s = c[0]["s"]
        e = max(c[-1]["e"] + 0.06, s + 0.30)
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
        placed.append({"s": s, "e": e, "style": style, "mv": mv, "c": c})
        last_idx[lane] = len(placed) - 1

    # ---- styles (typography pass: +4px, tighter outline, same palette)
    ass_header = f"""[Script Info]
; Post-FX v3 captions — proofread, word-locked, flash-free chunking
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

    def fmt_ass(sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = sec % 60
        return f"{h}:{m:02d}:{s:05.2f}"

    def fmt_srt(sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        ms = int(round((sec - int(sec)) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    ass_lines, srt_lines = [], []
    for i, it in enumerate(placed, 1):
        s, e, st, mv, c = it["s"], it["e"], it["style"], it["mv"], it["c"]
        k_parts = []
        for wi, w in enumerate(c):
            ws = max(s, w["s"])
            nxt = min(e, c[wi + 1]["s"]) if wi + 1 < len(c) else e
            dur_cs = max(6, int(round((nxt - ws) * 100)))
            k_parts.append(f"{{\\k{dur_cs}}}{w['w']}")
        body = " ".join(k_parts)
        if c[0]["w"].startswith("*") and c[0]["w"].endswith("*"):
            body = "{\\i1\\c&H9C9C9C&}" + body + "{\\i0\\c&HFFFFFF&}"
        ass_lines.append(
            f"Dialogue: 0,{fmt_ass(s)},{fmt_ass(e)},{st},,0,0,{mv},,{{\\fad(40,30)}}{body}"
        )
        plain = " ".join(w["w"] for w in c)
        srt_lines.append(f"{i}\n{fmt_srt(s)} --> {fmt_srt(e)}\n{plain}\n")

    (V3 / "subtitles_v3.ass").write_text(ass_header + "\n".join(ass_lines) + "\n", encoding="utf-8")
    (V3 / "captions_v3.srt").write_text("\n".join(srt_lines), encoding="utf-8")
    styles = {}
    for it in placed:
        styles[it["style"]] = styles.get(it["style"], 0) + 1
    log(
        f"captions: {len(placed)} cues -> {styles} "
        f"({sum(1 for i in placed if i['c'][0]['w'].startswith('*'))} inaudible)"
    )


# --------------------------------------------------------------------- cards
def _fit_lines(text: str, font, max_w: int, draw) -> list[str]:
    """Pixel-measured greedy wrap: no line ever exceeds max_w."""
    words = text.split()
    lines, cur = [], ""
    for w in words:
        cand = (cur + " " + w).strip()
        if draw.textlength(cand, font=font) <= max_w or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def phase_cards() -> None:
    """Content-fit case cards (owner finding 3: text must never overflow).

    Every string is measured with PIL; fonts shrink until the content fits and
    the box grows to the content — the old fixed 76/132px box with a single
    58-char wrap is gone. Same visual family (accent bar, dark panel, header).
    """
    from PIL import Image, ImageDraw, ImageFont

    plan = json.loads((WORK / "auto_plan.json").read_text())
    ts_data = json.loads((WORK / "timeline.json").read_text())
    part_bounds, total_dur = ts_data["part_bounds"], ts_data["total_dur"]

    bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    reg_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ACCENTS = {
        "location_intro": (220, 45, 45),
        "evidence_fact": (56, 189, 248),
        "case_outcome": (245, 185, 50),
    }
    W, H = 1280, 720
    MAX_TEXT_W = 470  # max text width inside the card
    MAX_BOX_W = 620
    PAD_Y, GAP = 14, 6
    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))

    card_meta = []
    for ci, c in enumerate(plan.get("cards", [])):
        accent = ACCENTS.get(c["template"], (148, 163, 184))
        head = c["template"].replace("_", " ").upper()
        title = c.get("title", "")
        subtitle = c.get("subtitle", "")
        f_head = ImageFont.truetype(bold_path, 13)

        title_px, f_title, t_lines = 24, None, [title]
        for px in (24, 22, 20, 18, 16):
            f_try = ImageFont.truetype(bold_path, px)
            lines = _fit_lines(title, f_try, MAX_TEXT_W, probe)
            if len(lines) <= 2:
                title_px, f_title, t_lines = px, f_try, lines
                break

        s_lines, f_sub, sub_px = [], None, 16
        if subtitle:
            for px in (16, 15, 14, 13):
                f_try = ImageFont.truetype(reg_path, px)
                lines = _fit_lines(subtitle, f_try, MAX_TEXT_W, probe)
                if len(lines) <= 3:
                    f_sub, s_lines, sub_px = f_try, lines, px
                    break

        content_w = max(
            [probe.textlength(head, font=f_head)]
            + [probe.textlength(ln, font=f_title) for ln in t_lines]
            + [probe.textlength(ln, font=f_sub) for ln in s_lines]
        )
        bw = int(min(MAX_BOX_W, max(322, content_w + 78)))
        text_x = 58
        box_h = int(
            PAD_Y
            + 18
            + 4
            + len(t_lines) * int(title_px * 1.3)
            + ((GAP + len(s_lines) * int(sub_px * 1.35)) if s_lines else 0)
            + PAD_Y
        )

        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle(
            [38, 34, 38 + bw, 34 + box_h],
            radius=10,
            fill=(14, 17, 24, 225),
            outline=(255, 255, 255, 45),
            width=1,
        )
        d.rounded_rectangle([38, 34, 45, 34 + box_h], radius=4, fill=accent + (255,))
        y = 34 + PAD_Y
        d.text((text_x, y), head, font=f_head, fill=accent + (255,))
        y += 18 + 4
        for ln in t_lines:
            d.text((text_x, y), ln, font=f_title, fill=(255, 255, 255, 255))
            y += int(title_px * 1.3)
        for ln in s_lines:
            if ln == s_lines[0]:
                y += GAP
            d.text((text_x, y), ln, font=f_sub, fill=(203, 213, 225, 255))
            y += int(sub_px * 1.35)
        if y - 34 > box_h:  # defensive: box must contain everything drawn
            box_h = y - 34 + 2
            img2 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            d = ImageDraw.Draw(img2)
            d.rounded_rectangle(
                [38, 34, 38 + bw, 34 + box_h],
                radius=10,
                fill=(14, 17, 24, 225),
                outline=(255, 255, 255, 45),
                width=1,
            )
            d.rounded_rectangle([38, 34, 45, 34 + box_h], radius=4, fill=accent + (255,))
            y = 34 + PAD_Y
            d.text((text_x, y), head, font=f_head, fill=accent + (255,))
            y += 18 + 4
            for ln in t_lines:
                d.text((text_x, y), ln, font=f_title, fill=(255, 255, 255, 255))
                y += int(title_px * 1.3)
            for ln in s_lines:
                if ln == s_lines[0]:
                    y += GAP
                d.text((text_x, y), ln, font=f_sub, fill=(203, 213, 225, 255))
                y += int(sub_px * 1.35)
            img = img2
        p = V3 / f"card_v3_{ci}.png"
        img.save(p)
        card_meta.append({"png": p.name, "template": c["template"], "box": [38, 34, bw, box_h]})
        log(
            f"card {ci} [{c['template']}]: {bw}x{box_h}px content-fit "
            f"(title {title_px}px/{len(t_lines)}L, sub {sub_px}px/{len(s_lines)}L)"
        )

    def act_first_nar_time(act: str):
        for b in part_bounds:
            if b["act"] == act and b["kind"] == "narration":
                return b["start"]
        return None

    card_times = []
    for m, c in zip(card_meta, plan.get("cards", []), strict=False):
        t = act_first_nar_time(c.get("near_act", ""))
        if t is None:
            t = max(0.0, total_dur - 32.0)
        card_times.append([m["png"], round(float(t), 2)])
    with open(V3 / "cards_meta_v3.json", "w") as fh:
        json.dump({"card_times": card_times, "total_dur": total_dur}, fh)
    log(f"cards: {len(card_times)} content-fit card(s)")


# ---------------------------------------------------------------- burn N
def _chunk_ass_v3(ci: int, cues_path: Path) -> Path:
    """Chunk-shifted ASS that DROPS cues fully before the chunk start.

    THE FLASH-BUG FIX (owner finding 2): pipe_09b clamped every pre-boundary
    cue to 0.00-0.05 so ~250 cues stacked into one paragraph for 2 frames at
    the 330 s boundary. Law here: a cue is kept only if it genuinely overlaps
    the chunk window [t0, t1); only a crossing cue gets its start clamped.
    """
    text = cues_path.read_text(encoding="utf-8")
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
        # Dialogue: <layer>,<start>,<end>,<style>,<name>,<ml>,<mr>,<mv>,<fx>,<text>
        s_sec, e_sec = to_sec(parts[1]), to_sec(parts[2])
        if e_sec <= t0 + 0.005:  # fully before this chunk -> DROP
            n_drop += 1
            continue
        if s_sec >= t0 + dur:  # fully after -> DROP
            n_drop += 1
            continue
        local_s = max(0.0, s_sec - t0)
        local_e = min(dur + 0.4, e_sec - t0)
        if local_e - local_s < 0.05:
            n_drop += 1
            continue
        out_lines.append(",".join([parts[0], fmt(local_s), fmt(local_e)] + parts[3:]))
        n_kept += 1
    path = V3 / f"captions_chunk_{ci:02d}.ass"
    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    log(f"chunk ASS {ci}: kept {n_kept}, dropped {n_drop} (flash law)")
    return path


def phase_burn(ci: int) -> None:
    """Burn captions + content-fit cards + per-span watermark on chunk ci.

    Video-only output; the accepted audio track is muxed once at the end
    (phase mux) so AAC framing never straddles a chunk boundary.
    """
    src = V3 / "master_v3_blurred.mp4"
    info_d = probe_dur(src)
    n_chunks = max(1, -(-int(info_d * 1000) // int(BURN_CHUNK_S * 1000)))
    if not 0 <= ci < n_chunks:
        raise SystemExit(f"burn chunk {ci} out of range 0..{n_chunks - 1}")
    part = V3 / f"burn_v3_{ci:02d}.mp4"
    mark = V3 / f"burn_v3_{ci:02d}.done"
    if mark.exists() and part.exists():
        log(f"burn {ci} cached")
        return
    t0 = ci * BURN_CHUNK_S
    dur = min(BURN_CHUNK_S, info_d - t0)
    ass = _chunk_ass_v3(ci, V3 / "subtitles_v3.ass")
    esc = str(ass).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")

    # watermark chain: the mark is baked at a FIXED position for the whole
    # video — the span sweep proved presence at ~91% of samples and the
    # "gaps" are template-match misses over busy backgrounds (visually
    # verified at 205s/225s/390s), NOT absences. So: ONE exact-pixel box
    # blurred CONTINUOUSLY — rectangular crop+gblur+overlay (the owner's
    # no-rounded-default law), zero enable-window flicker, and far less
    # scenery damage than the old 46%-width band.
    wm = json.loads((V3 / "wm_spans_v3.json").read_text())["spans"]
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
        wm_note = f"continuous box ({bx},{by},{bw},{bh})"
    else:
        wm_note = "no watermark spans found"

    cards_meta = json.loads((V3 / "cards_meta_v3.json").read_text())
    card_times = cards_meta["card_times"]
    fc = blur_fc + f"{cur_v}ass='{esc}'[subbed];"
    prev = "[subbed]"
    inputs = []
    for png, g_t0 in card_times:
        lt0 = g_t0 - t0
        t_end = min(lt0 + 6.5, dur - 0.2)
        if t_end <= 0.5 or lt0 >= dur:
            continue
        idx = len(inputs) + 1  # ffmpeg input index = position among INCLUDED cards
        inputs.append((V3 / png, idx))
        # single-frame input + eof_action=repeat + enable window: the card is
        # decoded ONCE and repeated (the old -loop 1 + fade chain re-decoded
        # 4 PNGs at 30 fps through rgba fades — the burn ran at <4 fps and
        # blew the process window). Static lower-third pop-on/off is the
        # documentary standard; alpha fades buy nothing here.
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
    subprocess.run(cmd, check=True)
    mark.write_text("ok")
    log(f"burn chunk {ci + 1}/{n_chunks} done ({dur:.1f}s, {wm_note}, {len(inputs)} cards)")


def phase_burnconcat() -> None:
    info_d = probe_dur(V3 / "master_v3_blurred.mp4")
    n_chunks = max(1, -(-int(info_d * 1000) // int(BURN_CHUNK_S * 1000)))
    parts = []
    for ci in range(n_chunks):
        part = V3 / f"burn_v3_{ci:02d}.mp4"
        if not ((V3 / f"burn_v3_{ci:02d}.done").exists() and part.exists()):
            raise SystemExit(f"burn chunk {ci} missing")
        parts.append(part)
    lst = V3 / "burn_v3_chunks.txt"
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
            str(V3 / "master_v3_burned.mp4"),
        ],
        "burn concat",
    )
    log(
        f"master_v3_burned: {probe_dur(V3 / 'master_v3_burned.mp4'):.2f}s, "
        f"{(V3 / 'master_v3_burned.mp4').stat().st_size / 1e6:.1f} MB"
    )


# ------------------------------------------------------------------- voice
VOICE_SPAN = (628.92, 657.65)  # Outro narration span on the master clock


def _tts_normalize(text: str) -> str:
    """TTS text-normalization law (owner finding 6).

    Units, ratios and symbols must be SPOKEN, not spelled: '0.391g/210L' ->
    '0.391 grams per 210 liters'. Kept in a standalone function so the
    narration planner can reuse it for any future video.
    """
    import tts_norm

    return tts_norm.normalize(text)


def phase_voice() -> None:
    """Surgical re-voice of the Outro narration span (edge-tts + word timing).

    The token '0.391g/210L' is read 'gee slash two ten el' by the TTS voice.
    The span is pure narration over muted b-roll, so replacing its audio with
    a re-voiced, span-fitted clip cannot touch the rest of the mix. edge-tts
    WordBoundary events give the word timings -> captions stay word-locked.

    Graceful degradation: any synth/normalizer failure logs and SKIPS (the
    mux then passes the accepted audio through unpatched) — a voice patch may
    never strand the whole render.
    """
    wav = V3 / "narration_outro_fixed.wav"
    words_json = V3 / "narration_outro_words.json"
    if wav.exists() and words_json.exists():
        log("voice cached")
        return
    t = json.loads((WORK / "timeline.json").read_text())
    span = None
    for b in t["part_bounds"]:
        if b["kind"] == "narration" and abs(b["start"] - VOICE_SPAN[0]) < 1.0:
            span = (b["start"], b["end"])
    if span is None:
        log(f"outro narration span not found in timeline ({VOICE_SPAN}) — voice skipped")
        return
    events = json.loads((WORK / "asm_state.json").read_text())["events"]
    ev = None
    for e in events:
        ws = e.get("words") or []
        if ws and abs(ws[0]["s"] - span[0]) < 0.5 and "0.391" in " ".join(w["w"] for w in ws):
            ev = e
    if ev is None:
        log("outro narration event with 0.391 token not found — voice skipped")
        return
    raw_text = " ".join(w["w"] for w in ev["words"])
    try:
        import asyncio

        import edge_tts

        speak = _tts_normalize(raw_text)
        if not speak or not speak.strip():
            raise RuntimeError("normalizer returned empty text")
        log(f"re-voicing span {span[0]:.2f}-{span[1]:.2f}s ({span[1] - span[0]:.1f}s)")
        log(f"normalized: ...{speak[-180:]}")

        async def synth():
            voices = [
                ("en-US-BrianNeural-Fastest", "+0%"),
                ("en-US-BrianNeural-Fastest", "+4%"),
                ("en-US-BrianNeural-Fastest", "+8%"),
            ]
            for voice, rate in voices:
                out = V3 / f"tts_try_{rate.strip('+')}.mp3"
                words = []
                comm = edge_tts.Communicate(speak, voice, rate=rate)
                with open(out, "wb") as f:
                    async for chunk in comm.stream():
                        if chunk["type"] == "audio":
                            f.write(chunk["data"])
                        elif chunk["type"] == "WordBoundary":
                            words.append(
                                {
                                    "w": chunk["text"],
                                    "s": chunk["offset"] / 1e7,
                                    "e": (chunk["offset"] + chunk["duration"]) / 1e7,
                                }
                            )
                d = probe_dur(out)
                log(f"  {voice} {rate}: {d:.2f}s for {span[1] - span[0]:.1f}s span")
                if d <= (span[1] - span[0]) + 0.6:
                    return out, words, d
            return out, words, d

        out, words, d = asyncio.run(synth())
        span_dur = span[1] - span[0]
        tempo = min(1.15, max(0.85, d / span_dur))
        _sh(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(out),
                "-af",
                f"atempo={tempo:.4f},apad",
                "-t",
                f"{span_dur:.3f}",
                "-ar",
                "48000",
                "-ac",
                "1",
                str(wav),
            ],
            "tts fit",
        )
        for w in words:
            w["s"] = round(span[0] + w["s"] / tempo, 3)
            w["e"] = round(span[0] + w["e"] / tempo, 3)
        words_json.write_text(
            json.dumps({"span": list(span), "tempo": tempo, "words": words}, indent=1)
        )
        log(f"voice: fitted at tempo x{tempo:.3f}, {len(words)} word boundaries")
    except Exception as e:  # noqa: BLE001
        (V3 / "voice_skipped.json").write_text(json.dumps({"reason": str(e)[:400]}, indent=1))
        log(f"voice patch SKIPPED ({str(e)[:200]}) — audio passes through unpatched")


def phase_mux() -> None:
    """Patch the Outro span into the accepted audio track + mux the final."""
    burned = V3 / "master_v3_burned.mp4"
    if not burned.exists():
        raise SystemExit("burnconcat first")
    patched = V3 / "audio_v3.m4a"
    if patched.exists():
        log("mux audio cached")
    else:
        span = None
        pj = V3 / "narration_outro_words.json"
        if pj.exists():
            span = tuple(json.loads(pj.read_text())["span"])
        if span and (V3 / "narration_outro_fixed.wav").exists():
            wav = V3 / "narration_outro_fixed.wav"
            a0, a1 = span
            # crossfade 40 ms on the left edge to avoid any seam pop
            _sh(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(V3 / "audio_src.m4a"),
                    "-i",
                    str(wav),
                    "-filter_complex",
                    f"[0:a]atrim=0:{a0:.3f},asetpts=PTS-STARTPTS[base0];"
                    f"[0:a]atrim={a1:.3f},asetpts=PTS-STARTPTS[tail];"
                    "[1:a]aformat=sample_rates=48000:channel_layouts=stereo,afade=t=in:st=0:d=0.04[nar];"
                    "[base0][nar]acrossfade=d=0.04:c1=tri:c2=tri[head];"
                    "[head][tail]concat=n=2:v=0:a=1[out]",
                    "-map",
                    "[out]",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-ar",
                    "48000",
                    "-ac",
                    "2",
                    str(patched),
                ],
                "audio patch",
            )
            log(f"audio: outro narration replaced at {a0:.2f}-{a1:.2f}s")
        else:
            import shutil

            shutil.copyfile(V3 / "audio_src.m4a", patched)
            log("audio: passed through unpatched (no voice span)")
    final = V3 / "master_postfx_v3.mp4"
    _sh(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(burned),
            "-i",
            str(patched),
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
        "final mux",
    )
    d_v = probe_dur(burned)
    d_f = probe_dur(final)
    if abs(d_f - d_v) > 1.0:
        raise SystemExit(f"duration drifted {abs(d_f - d_v):.2f}s — refusing to ship")
    log(f"muxed: master_postfx_v3.mp4 {d_f:.2f}s, {final.stat().st_size / 1e6:.1f} MB")


# ---------------------------------------------------------------------- qa
def _grab(path: Path, t: float):
    import cv2

    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
    ok, f = cap.read()
    cap.release()
    return f if ok else None


def _text_mass(frame, band="center") -> float:
    """Caption-pixel mass in a band; a stacked paragraph spikes this."""
    import cv2

    h, w = frame.shape[:2]
    if band == "center":
        roi = frame[int(h * 0.18) : int(h * 0.72), int(w * 0.12) : int(w * 0.88)]
    else:  # caption band
        roi = frame[h - 170 : h - 8, 40 : w - 40]
    b, g, r = roi[:, :, 0].astype(int), roi[:, :, 1].astype(int), roi[:, :, 2].astype(int)
    glyph = (((b > 200) & (g > 200) & (r > 200)) | ((r > 200) & (g > 120) & (b < 160))).astype(
        np.uint8
    )
    dark = ((b < 60) & (g < 60) & (r < 60)).astype(np.uint8)
    dark_near = cv2.dilate(dark, np.ones((5, 5), np.uint8))
    return float(((glyph > 0) & (dark_near > 0)).sum())


def _lap(img) -> float:
    import cv2

    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def phase_qa() -> None:

    final = V3 / "master_postfx_v3.mp4"
    base = _base()
    if not final.exists():
        raise SystemExit("mux first")
    checks: list[dict] = []
    qa_p = V3 / "qa_v3.json"

    def check(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        qa_p.write_text(json.dumps(checks, indent=1))
        log(f"  [{'OK' if ok else 'FAIL'}] {name}: {detail}")

    info_d = probe_dur(final)
    # ---- 1. duration law
    check(
        "duration",
        abs(info_d - probe_dur(base)) <= 1.0,
        f"final {info_d:.2f}s vs base {probe_dur(base):.2f}s",
    )

    # ---- 2. FLASH REGRESSION: no caption-mass spike at any burn boundary
    n_chunks = max(1, -(-int(info_d * 1000) // int(BURN_CHUNK_S * 1000)))
    bounds = [ci * BURN_CHUNK_S for ci in range(1, n_chunks)]
    spikes = []
    for b in bounds:
        base_mass = np.median(
            [
                _text_mass(f, "center")
                for t in np.arange(b - 2.0, b - 0.6, 0.2)
                if (f := _grab(final, t)) is not None
            ]
        )
        worst = 0.0
        for t in np.arange(b - 0.15, b + 0.45, 1 / 30):
            f = _grab(final, t)
            if f is None:
                continue
            m = _text_mass(f, "center")
            worst = max(worst, m / max(1.0, base_mass))
        if worst > 4.0:
            spikes.append((round(b, 1), round(worst, 1)))
    check(
        "no_paragraph_flash",
        not spikes,
        f"boundaries {bounds} worst mass ratio "
        f"{max((s[1] for s in spikes), default=0.0):.1f}x (<4.0 required)",
    )

    # ---- 3. face redaction coverage at the measured leak (t~199-203)
    ok202 = True
    ratios = []
    for t in (199.5, 200.5, 201.5, 202.5):
        fb, ff = _grab(base, t), _grab(final, t)
        if fb is None or ff is None:
            continue
        # the officer's face region (plan2 box area) — inner 50%
        plan = _plan2()
        fi = int(t * FPS)
        boxes = plan.boxes_at(fi)
        if not boxes:
            ok202 = False
            ratios.append(f"t={t}: NO PLAN BOX (leak)")
            continue
        bx = max(boxes, key=lambda b: b.w * b.h)
        x, y, w, h = bx.x, bx.y, bx.w, bx.h
        ix, iy, iw, ih = x + w // 4, y + h // 4, max(4, w // 2), max(4, h // 2)
        v0 = max(_lap(fb[iy : iy + ih, ix : ix + iw]), 1e-6)
        v1 = max(_lap(ff[iy : iy + ih, ix : ix + iw]), 1e-6)
        ratios.append(f"t={t}: x{v0 / v1:.1f} v1={v1:.0f}")
        # the law is the FINAL is blurred. A sharp base (the 199-202s leak)
        # shows as a collapse ratio; an already-blurred base shows as v1 < 40.
        ok202 = ok202 and (v1 < 40.0 or v0 / v1 >= 2.0)
    check(
        "face_leak_202s_blurred",
        (ok202 and len(ratios) == 4) if REDACT_FACES else True,
        "; ".join(ratios)
        or ("REDACT_FACES=0 — policy off" if not REDACT_FACES else "frames missing"),
    )

    # ---- 4. pass-2 blur sample check over the plan (spread sample, policy-aware)
    plan = _plan2()
    if not REDACT_FACES:
        check("pass2_blur_effective", True, "REDACT_FACES=0 — policy off")
    else:
        worst = (99.0, -1)
        worst_abs = (0.0, -1.0)
        n_tr = len(plan.trajectories)
        sample_ids = (
            [int(i * (n_tr - 1) / 11) for i in range(12)] if n_tr > 1 else ([0] if n_tr else [])
        )
        for ti in sample_ids:
            tr = plan.trajectories[ti]
            mid = tr.boxes[len(tr.boxes) // 2]
            t = mid.frame_idx / plan.fps
            fb, ff = _grab(base, t), _grab(final, t)
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
        # pass-2 windows overlap pass-1 coverage (base often already blurred),
        # so a base/final ratio of 1.0 there is EXPECTED, not a failure. The
        # law: the final region must be blurred — absolute Laplacian collapse
        # (<40) or a real collapse ratio vs a sharp base (>=1.8).
        check(
            "pass2_blur_effective",
            (not sample_ids) or worst_abs[0] < 40.0 or worst[0] >= 1.8,
            f"max final inner var {worst_abs[0]:.0f} @ {worst_abs[1]:.1f}s, "
            f"worst base/final x{worst[0]:.2f} over {len(sample_ids)} tracks",
        )

    # ---- 5. watermark: the mark is baked at a fixed position for the whole
    #         video (span sweep + visual gap verification at 205/225/390s)
    #         -> burned as ONE continuous exact box. Prove it is blurred
    #         EVERYWHERE, gaps included.
    wm = json.loads((V3 / "wm_spans_v3.json").read_text())["spans"]
    wm_notes, wm_ok = [], bool(wm)
    if wm:
        arr = np.array([sp["box"] for sp in wm])
        bx, by, bw, bh = [int(v) for v in np.median(arr, axis=0)]
        bx, by, bw, bh = max(0, bx - 2), max(0, by - 2), bw + 4, bh + 4
        times = list(np.arange(15.0, max(16.0, info_d - 3.0), 30.0)) + [
            203.5,
            205.0,
            225.0,
            390.0,
            400.0,
        ]
        rs, abs_vars = [], []
        for t in sorted({round(float(x), 2) for x in times}):
            fb, ff = _grab(base, t), _grab(final, t)
            if fb is None or ff is None:
                continue
            v0 = _lap(fb[by : by + bh, bx : bx + bw])
            v1 = _lap(ff[by : by + bh, bx : bx + bw])
            abs_vars.append(v1)
            if v0 >= 60:
                rs.append(v0 / max(v1, 1e-6))
        worst_abs = max(abs_vars, default=0.0)
        wm_ok = bool(rs) and min(rs) >= 2.0 and worst_abs < 60.0
        wm_notes.append(
            f"continuous exact box ({bx},{by},{bw},{bh}): "
            f"{len(rs)}/{len(abs_vars)} sharp-base samples, "
            f"min collapse x{min(rs, default=0.0):.1f} (>=2.0), "
            f"worst final var {worst_abs:.0f} (<60)"
        )
    check("watermark_exact_pixels", wm_ok, "; ".join(wm_notes) or "no spans")

    # ---- 6. captions present + inaudible tags + audio present
    srt = (V3 / "captions_v3.srt").read_text()
    n_cues = srt.count(" --> ")
    n_inaud = len(re.findall(r"\*(?:inaudible|mumbling|cross-talk)\*", srt))
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
                str(final),
            ]
        )
        .decode()
        .strip()
    )
    check(
        "captions_and_audio",
        n_cues >= 200 and abs(a_dur - info_d) <= 0.1,
        f"{n_cues} cues ({n_inaud} inaudible-tagged), audio {a_dur:.2f}s",
    )

    failed = [c for c in checks if not c["ok"]]
    log(f"QA: {len(checks) - len(failed)}/{len(checks)} green")
    if failed:
        raise SystemExit(f"QA FAILED: {[c['name'] for c in failed]}")


# -------------------------------------------------------------------- ship
def phase_ship() -> None:
    import shutil

    final = V3 / "master_postfx_v3.mp4"
    if not final.exists():
        raise SystemExit("mux first")
    shutil.copyfile(final, SHIP_PACK / "final_720p.mp4")
    shutil.copyfile(V3 / "captions_v3.srt", SHIP_PACK / "captions.srt")
    shutil.copyfile(
        V3 / "subtitles_v3.ass",
        Path("/home/z/my-project/download/eustis_acceptance_postfx_captions.ass"),
    )
    shutil.copyfile(
        V3 / "captions_v3.srt", Path("/home/z/my-project/download/eustis_acceptance_captions.srt")
    )
    meta_p = SHIP_PACK / "metadata.json"
    if meta_p.exists():
        import hashlib

        meta = json.loads(meta_p.read_text())
        meta["files"]["final_720p.mp4"] = str(SHIP_PACK / "final_720p.mp4")
        meta["final_sha256"] = hashlib.sha256(final.read_bytes()).hexdigest()
        meta["verification"]["duration_s"] = round(probe_dur(final), 1)
        meta["post_fx_v3"] = {
            "face_redaction": "pass-2 YOLOv8n-face dense sweep (0.25s, conf 0.15, "
            "flip-augmented) over the assembled base; entry-gap "
            "leak at ~200s closed; REDACT_FACES=1 policy flag",
            "flash_fix": "chunk ASS drops pre-boundary cues (was: clamp-to-0.05 "
            "stacked ~250 cues into a 2-frame paragraph at 330s)",
            "cards": "content-fit: PIL-measured wrap + font shrink + box grows to content",
            "captions": "LLM proofread (case-facts, profanity mask, garble -> "
            "*inaudible*/*mumbling*/*cross-talk*) re-aligned onto "
            "original word timestamps; typography +4px",
            "watermark": "mark baked at one fixed position (span sweep + "
            "visual gap verification) -> ONE continuous exact-"
            "pixel box blur, no scenery band, no flicker",
            "voice": "Outro re-voiced with TTS unit normalization "
            "(0.391g/210L -> '0.391 grams per 210 liters'), "
            "span-fitted, word timings locked",
        }
        meta_p.write_text(json.dumps(meta, indent=2))
    log(
        f"shipped: {SHIP_PACK / 'final_720p.mp4'} "
        f"({(SHIP_PACK / 'final_720p.mp4').stat().st_size / 1e6:.1f} MB)"
    )


PHASES = {
    "copy-src": phase_copy_src,
    "sweep2-seg": lambda: phase_sweep2_seg(int(sys.argv[2])),
    "sweep2-done": phase_sweep2_done,
    "plan2": phase_plan2,
    "chunk2": lambda: phase_chunk2(int(sys.argv[2])),
    "concat2": phase_concat2,
    "wm": phase_wm,
    "captions": phase_captions,
    "cards": phase_cards,
    "burn": lambda: phase_burn(int(sys.argv[2])),
    "burnconcat": phase_burnconcat,
    "voice": phase_voice,
    "mux": phase_mux,
    "qa": phase_qa,
    "ship": phase_ship,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in PHASES:
        raise SystemExit(f"usage: postfx_v3.py {'|'.join(PHASES)} [arg]")
    PHASES[sys.argv[1]]()
