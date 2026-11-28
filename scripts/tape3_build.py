#!/usr/bin/env python3
"""Tape 3 (St. Johns County grocery) part builder.

Stages (each cached, rerun-safe):
  cut      - raw parts per plan (1080p30, notebook AF chain, pcm audio in mkv)
  nar      - edge-tts narration + faster-whisper word alignment + B-roll parts
  audio    - DeepFilterNet3 denoise + notebook EQ mix + profanity mute/tone
  blur     - yuNet face sweep + tracked blur burn-in
  timeline - master-clock events/bounds/total (probe-accurate)

Notebook deviations (documented): no hflip (keeps AXON bar top-right so the
owner's channel watermark can own the top-left), 1920x1080 instead of 720p.
Everything else follows upload/notebook89ea144f9a (2).ipynb verbatim.
"""

import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

WORK = "/home/z/my-project/work_v2"
PARTS = f"{WORK}/parts_m"
SRC = "/home/z/my-project/footage/florida_grocery_source.webm"
DF = "/home/z/my-project/bin/deep-filter"
YUNET = f"{WORK}/face_detection_yunet_2023mar.onnx"

BAD_RE = re.compile(
    r"\b(fuck\w*|motherfuck\w*|shit\w*|bitch\w*|pussy\w*|dick\w*|asshole\w*|goddamn\w*|damn\w*)\b",
    re.I,
)
BLEEP_RE = re.compile(
    r"\b(fuck\w*|motherfuck\w*|shit\w*|bitch\w*|pussy\w*|dick\w*|asshole\w*|cunt\w*|bastard\w*)\b",
    re.I,
)

VF_BASE = "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=30,format=yuv420p"
NAR_LOOK = ",hue=s=0,eq=contrast=1.08:brightness=-0.02,noise=alls=7:allf=t+u,vignette=PI/4.5"
AF = "highpass=f=85:poles=2,loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000,aformat=channel_layouts=stereo"


def ff(*args, retries=2):
    args = [str(a) for a in args]
    for attempt in range(1, retries + 1):
        r = subprocess.run(args, capture_output=True, text=True)
        if r.returncode == 0:
            return r
        print(
            f"!!! ffmpeg exited {r.returncode} (attempt {attempt}/{retries})\n$ {' '.join(args[:12])} ...\n{r.stderr[-1200:]}"
        )
    raise RuntimeError(f"ffmpeg failed: exit {r.returncode}: {r.stderr[-400:]}")


def probe_dur(p):
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
                p,
            ]
        )
        .decode()
        .strip()
    )


def part_ok(p, min_dur=0.2):
    try:
        return os.path.exists(p) and probe_dur(p) > min_dur
    except Exception:
        return False


def load():
    plan = json.load(open(f"{WORK}/parts_plan.json"))["plan"]
    utts = json.load(open(f"{WORK}/raw_transcript.json"))["utterances"]
    clean = json.load(open(f"{WORK}/clean_transcript.json"))
    return plan, utts, clean


def norm(w):
    return re.sub(r"[^a-z0-9]", "", w.lower())


def align_words(txt, ow, s0, s1):
    """Notebook CELL 6 align_words, verbatim."""
    toks = txt.split()
    if not toks:
        return []
    if not ow:
        tot, wts = max(0.35, s1 - s0), [max(2, len(t)) for t in toks]
        sw, c, res = sum(wts), s0, []
        for t, w in zip(toks, wts, strict=False):
            d = tot * (w / sw)
            res.append({"w": t, "s": c, "e": c + d})
            c += d
        return res
    import difflib

    sm = difflib.SequenceMatcher(
        None, [norm(w["w"]) for w in ow], [norm(t) for t in toks], autojunk=False
    )
    timed = [None] * len(toks)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(j2 - j1):
                timed[j1 + k] = {
                    "w": toks[j1 + k],
                    "s": ow[i1 + k]["s"],
                    "e": max(ow[i1 + k]["e"], ow[i1 + k]["s"] + 0.06),
                }
        elif tag == "replace":
            t0, t1 = ow[i1]["s"], max(ow[i2 - 1]["e"], ow[i1]["s"] + 0.14)
            wts = [max(2, len(x)) for x in toks[j1:j2]]
            sw, cur = sum(wts), t0
            for k, (tok, wt) in enumerate(zip(toks[j1:j2], wts, strict=False)):
                d = (t1 - t0) * (wt / sw)
                timed[j1 + k] = {"w": tok, "s": cur, "e": cur + d}
                cur += d
    j = 0
    while j < len(timed):
        if timed[j] is None:
            je = j
            while je < len(timed) and timed[je] is None:
                je += 1
            pe = timed[j - 1]["e"] if j > 0 else ow[0]["s"]
            ns = timed[je]["s"] if je < len(timed) else ow[-1]["e"]
            if ns <= pe + 0.06:
                pe = max(0.0, ns - 0.18 * (je - j))
            step = max(0.12 * (je - j), ns - pe) / (je - j)
            for k in range(j, je):
                timed[k] = {"w": toks[k], "s": pe + (k - j) * step, "e": pe + (k - j + 1) * step}
            j = je
        else:
            j += 1
    return timed


def seek_args(s, dur):
    pre = max(0.0, s - 10.0)
    return ["-ss", f"{pre:.3f}", "-i", SRC, "-ss", f"{s - pre:.3f}", "-t", f"{dur:.3f}"]


def stage_cut(plan):
    os.makedirs(PARTS, exist_ok=True)
    plan_hash = hashlib.sha1(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:10]
    hashfile = f"{PARTS}/cut_hash.txt"
    resume = os.path.exists(hashfile) and open(hashfile).read().strip() == plan_hash
    if not resume:
        for f in glob.glob(f"{PARTS}/part_*.mkv"):
            os.remove(f)
    open(hashfile, "w").write(plan_hash)
    for idx, item in enumerate(plan, 1):
        if item["kind"] != "raw":
            continue
        out = f"{PARTS}/part_{idx:03d}.mkv"
        if resume and part_ok(out):
            continue
        s_t, dur = item["s"], item["e"] - item["s"]
        print(f"[cut {idx:03d}] raw {s_t:.2f}+{dur:.2f}s")
        ff(
            "ffmpeg",
            "-y",
            *seek_args(s_t, dur),
            "-filter_complex",
            f"[0:v]{VF_BASE}[v];[0:a]{AF}[a]",
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-c:a",
            "pcm_s16le",
            out,
        )
        if not part_ok(out, dur - 1.0):
            raise RuntimeError(f"part {idx} invalid after encode")


def stage_nar(plan):
    os.makedirs(PARTS, exist_ok=True)
    import soundfile as sf
    from faster_whisper import WhisperModel

    wm = WhisperModel("base.en", device="cpu", compute_type="int8")
    for idx, item in enumerate(plan, 1):
        if item["kind"] != "nar":
            continue
        mp3 = f"{PARTS}/n_{idx:03d}.mp3"
        tw = f"{PARTS}/t_{idx:03d}.wav"
        w16 = f"{PARTS}/w_{idx:03d}.wav"
        if not os.path.exists(mp3):
            print(f"[nar {idx:03d}] edge-tts {len(item['text'].split())}w")
            ff(
                "python3",
                "-m",
                "edge_tts",
                "--voice",
                "en-US-BrianMultilingualNeural",
                "--rate=+5%",
                "--text",
                item["text"],
                "--write-media",
                mp3,
            )
        if not os.path.exists(tw):
            ff(
                "ffmpeg",
                "-y",
                "-i",
                mp3,
                "-af",
                "silenceremove=start_periods=1:start_threshold=-45dB:stop_periods=-1:stop_duration=0.20:stop_threshold=-45dB,volume=-1.5dB,alimiter=limit=0.841,aresample=48000,aformat=channel_layouts=stereo",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-c:a",
                "pcm_s16le",
                tw,
            )
        if not os.path.exists(w16):
            ff("ffmpeg", "-y", "-i", tw, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", w16)
        if not part_ok(f"{PARTS}/part_{idx:03d}.mkv"):
            arr, _ = sf.read(w16, dtype="float32")
            tts_ow = [
                {"w": w.word.strip(), "s": w.start, "e": w.end}
                for ws in wm.transcribe(arr, word_timestamps=True)[0]
                for w in (ws.words or [])
                if w.word.strip()
            ]
            nd = probe_dur(tw)
            bs, be = item["b"]
            bt = f"/tmp/bt_{idx:03d}.mkv"
            ff(
                "ffmpeg",
                "-y",
                *seek_args(bs, max(4.0, be - bs)),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "18",
                bt,
            )
            ff(
                "ffmpeg",
                "-y",
                "-stream_loop",
                "-1",
                "-i",
                bt,
                "-i",
                tw,
                "-t",
                f"{nd + 0.35:.3f}",
                "-filter_complex",
                f"[0:v]{VF_BASE}{NAR_LOOK}[v];[1:a]aresample=48000,aformat=channel_layouts=stereo,apad=pad_dur=0.35[a]",
                "-map",
                "[v]",
                "-map",
                "[a]",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-c:a",
                "pcm_s16le",
                f"{PARTS}/part_{idx:03d}.mkv.tmp.mkv",
            )
            os.replace(f"{PARTS}/part_{idx:03d}.mkv.tmp.mkv", f"{PARTS}/part_{idx:03d}.mkv")
            json.dump(tts_ow, open(f"{PARTS}/tts_ow_{idx:03d}.json", "w"))
            print(f"[nar {idx:03d}] broll part built, tts {nd:.2f}s")


def stage_audio(plan):
    """DeepFilterNet3 + notebook EQ mix + profanity mute/tone on RAW parts only."""
    for idx, item in enumerate(plan, 1):
        if item["kind"] != "raw":
            continue
        src = f"{PARTS}/part_{idx:03d}.mkv"
        out = f"{PARTS}/clean_{idx:03d}.mkv"
        if part_ok(out):
            continue
        dur = item["e"] - item["s"]
        shutil.rmtree("/tmp/df_o", ignore_errors=True)
        os.makedirs("/tmp/df_o", exist_ok=True)
        ff(
            "ffmpeg",
            "-y",
            "-i",
            src,
            "-vn",
            "-af",
            "highpass=f=90:poles=2,aresample=48000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            "/tmp/df_in.wav",
        )
        r = subprocess.run(
            [DF, "-D", "-o", "/tmp/df_o", "/tmp/df_in.wav"], capture_output=True, text=True
        )
        outs = glob.glob("/tmp/df_o/*.wav")
        if r.returncode != 0 or not outs:
            raise RuntimeError(f"deep-filter failed on part {idx}: {r.stderr[-300:]}")
        # profanity spans in source time -> part-relative
        clean = json.load(open(f"{WORK}/clean_transcript.json"))
        utts = json.load(open(f"{WORK}/raw_transcript.json"))["utterances"]
        s_t, e_t = item["s"], item["e"]
        spans = []
        for cl in clean:
            if cl["end"] < s_t or cl["start"] > e_t:
                continue
            for w in utts[cl["id"]].get("words", []):
                word = w.get("word", "").strip()
                if word and BLEEP_RE.search(word):
                    a, b = max(0.0, w["start"] - s_t - 0.04), min(dur, w["end"] - s_t + 0.04)
                    if b > a:
                        spans.append((round(a, 2), round(b, 2)))
        print(f"[audio {idx:03d}] df ok, bleeps={len(spans)}")
        fc = (
            "[0:a]equalizer=f=300:width_type=o:width=1.2:g=-2.5,"
            "equalizer=f=2800:width_type=o:width=1.2:g=2.5,"
            "equalizer=f=4500:width_type=o:width=1.2:g=2.0,treble=g=2.0:f=6800,"
            "acompressor=threshold=-18dB:ratio=3:attack=10:release=120[v];"
            "[1:a]highpass=f=150,lowpass=f=4500,volume=0.08[amb];"
            "[v][amb]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.841,"
            "aresample=48000,aformat=channel_layouts=stereo[vo]"
        )
        if spans:
            mute = "+".join(f"between(t,{a},{b})" for a, b in spans)
            fc += (
                f";[vo]volume='if({mute},0,1)':eval=frame[vm];"
                f"sine=frequency=1000:sample_rate=48000:duration={dur:.3f}[t0];"
                f"[t0]volume='if({mute},0.25,0)':eval=frame[tn];[vm][tn]amix=inputs=2:duration=first:normalize=0[a]"
            )
        else:
            fc += ";[vo]anull[a]"
        ff(
            "ffmpeg",
            "-y",
            "-i",
            outs[0],
            "-i",
            "/tmp/df_in.wav",
            "-filter_complex",
            fc,
            "-map",
            "[a]",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            "/tmp/df_mix.wav",
        )
        ff(
            "ffmpeg",
            "-y",
            "-i",
            src,
            "-i",
            "/tmp/df_mix.wav",
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "copy",
            "-c:a",
            "pcm_s16le",
            out,
        )


def ensure_yunet():
    if os.path.exists(YUNET) and os.path.getsize(YUNET) > 100000:
        return
    import urllib.request

    url = "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    print("downloading yuNet model...")
    urllib.request.urlretrieve(url, YUNET)
    print("yunet:", os.path.getsize(YUNET), "bytes")


def stage_blur(plan):
    """yuNet face sweep per frame + tracked blur burn-in (per-clip moving boxblur)."""
    import cv2

    ensure_yunet()
    for idx, item in enumerate(plan, 1):
        if item["kind"] != "raw":
            continue
        src = f"{PARTS}/clean_{idx:03d}.mkv"
        out = f"{PARTS}/final_{idx:03d}.mkv"
        tracks_json = f"{PARTS}/tracks_{idx:03d}.json"
        if part_ok(out):
            continue
        cap = cv2.VideoCapture(src)  # noqa: SIM115 - released in finally paths below
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        det = cv2.FaceDetectorYN_create(YUNET, "", (W, H), 0.9)
        boxes = []
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            det.setInputSize((W, H))
            _, faces = det.detect(frame)
            fb = []
            if faces is not None:
                for f in faces:
                    x, y, w, h = f[0], f[1], f[2], f[3]
                    if w < 36 or h < 36:  # ignore tiny/false positives
                        continue
                    fb.append([int(x), int(y), int(w), int(h)])
            boxes.append(fb)
            n += 1
            if n % 300 == 0:
                print(f"[blur {idx:03d}] {n} frames swept")
        cap.release()
        if not boxes:
            shutil.copy(src, out)
            continue
        # temporal smoothing: keep boxes with >=3 consecutive hits; interpolate gaps <= 15 frames
        tracks = []  # each: {frames: {i: box}}
        for i, fb in enumerate(boxes):
            for b in fb:
                matched = None
                for t in tracks:
                    li = max(t["frames"])
                    lb = t["frames"][li]
                    if i - li <= 15:
                        cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
                        lx, ly = lb[0] + lb[2] / 2, lb[1] + lb[3] / 2
                        if abs(cx - lx) < max(80, lb[2]) and abs(cy - ly) < max(80, lb[3]):
                            matched = t
                            break
                if matched is None:
                    tracks.append({"frames": {i: b}})
                else:
                    matched["frames"][i] = b
        tracks = [t for t in tracks if len(t["frames"]) >= 3]
        print(f"[blur {idx:03d}] {n} frames, {len(tracks)} tracked face(s)")
        json.dump(
            {
                "fps": fps,
                "w": W,
                "h": H,
                "n_frames": n,
                "tracks": {
                    k: {str(i): v for i, v in t["frames"].items()} for k, t in enumerate(tracks)
                },
            },
            open(tracks_json, "w"),
        )
        # burn tracked blur via pipe encode (blur follows each face per frame)
        import subprocess as sp

        enc = sp.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{W}x{H}",
                "-r",
                f"{fps:.3f}",
                "-i",
                "-",
                "-i",
                src,
                "-map",
                "0:v",
                "-map",
                "1:a?",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "copy",
                out,
            ],
            stdin=sp.PIPE,
        )
        cap = cv2.VideoCapture(src)  # noqa: SIM115 - released in finally paths below
        pad = 0.35  # blur box padding factor
        for i in range(n):
            ok, frame = cap.read()
            if not ok:
                break
            for t in tracks:
                fr = t["frames"]
                if i in fr:
                    b = fr[i]
                else:
                    lo = max(k for k in fr if k < i) if any(k < i for k in fr) else None
                    hi = min(k for k in fr if k > i) if any(k > i for k in fr) else None
                    if lo is None and hi is None:
                        continue
                    if lo is None:
                        b = fr[hi]
                    elif hi is None:
                        b = fr[lo]
                    else:
                        a0, a1 = fr[lo], fr[hi]
                        f = (i - lo) / (hi - lo)
                        b = [int(a0[c] + f * (a1[c] - a0[c])) for c in range(4)]
                    if lo is not None and hi is not None and (i - lo > 15 or hi - i > 15):
                        continue  # deep mid-gaps stay unblurred
                x, y, w, h = b
                pw, ph = int(w * (1 + pad * 2)), int(h * (1 + pad * 2))
                px, py = max(0, int(x - w * pad)), max(0, int(y - h * pad))
                px2, py2 = min(W, px + pw), min(H, py + ph)
                if px2 - px < 8 or py2 - py < 8:
                    continue
                roi = frame[py:py2, px:px2]
                frame[py:py2, px:px2] = cv2.blur(roi, (51, 51))
            enc.stdin.write(frame.tobytes())
        enc.stdin.close()
        enc.wait()
        cap.release()
        if not part_ok(out, (item["e"] - item["s"]) - 1.0):
            raise RuntimeError(f"blur part {idx} invalid")


def stage_pack(plan):
    """delogo third-party watermark + pack browser-ready mp4 assets for HyperFrames."""
    OUT = f"{WORK}/assets_hf"
    os.makedirs(OUT, exist_ok=True)
    DL = "delogo=x=4:y=976:w=536:h=100:show=0"
    for idx, item in enumerate(plan, 1):
        out = f"{OUT}/part_{idx:03d}.mp4"
        expect = None
        if item["kind"] == "raw":
            expect = item["e"] - item["s"]
        if part_ok(out) and (expect is None or probe_dur(out) >= expect - 0.6):
            continue
        if os.path.exists(out) and expect is not None and probe_dur(out) < expect - 0.6:
            print(
                f"[pack {idx:03d}] re-encoding truncated asset ({probe_dur(out):.2f}s < {expect:.2f}s)"
            )
        for cand in (
            f"{PARTS}/final_{idx:03d}.mkv",
            f"{PARTS}/clean_{idx:03d}.mkv",
            f"{PARTS}/part_{idx:03d}.mkv",
        ):
            if os.path.exists(cand):
                break
        else:
            raise RuntimeError(f"no source part for index {idx}")
        print(f"[pack {idx:03d}] <- {os.path.basename(cand)}")
        ff(
            "ffmpeg",
            "-y",
            "-i",
            cand,
            "-vf",
            f"{DL},format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            out,
        )


def stage_timeline(plan):
    events, bounds = [], []
    cursor = 0.0
    clean = json.load(open(f"{WORK}/clean_transcript.json"))
    utts = json.load(open(f"{WORK}/raw_transcript.json"))["utterances"]

    def asset(idx):
        return f"{WORK}/assets_hf/part_{idx:03d}.mp4"

    for idx, item in enumerate(plan, 1):
        d = probe_dur(asset(idx))
        if item["kind"] == "raw":
            s_t, e_t = item["s"], item["e"]
            dur = e_t - s_t
            for cl in clean:
                if cl["end"] >= s_t and cl["start"] <= e_t:
                    ow = [
                        {"w": w["word"].strip(), "s": w["start"], "e": w["end"]}
                        for w in utts[cl["id"]].get("words", [])
                        if w.get("word", "").strip()
                    ]
                    sh = []
                    for w in align_words(cl["text"], ow, cl["start"], cl["end"]):
                        ws, we = max(0.0, w["s"] - s_t), min(dur, w["e"] - s_t)
                        if not (s_t <= (w["s"] + w["e"]) / 2.0 <= e_t) or we <= ws + 0.04:
                            continue
                        if ws >= d:  # beyond the actual (possibly truncated) part -> drop
                            continue
                        sh.append(
                            {
                                "w": w["w"],
                                "s": round(ws + cursor, 3),
                                "e": round(min(we, d) + cursor, 3),
                            }
                        )
                    if sh:
                        events.append({"role": cl["role"], "words": sh})
        else:
            ow_cache = f"{WORK}/nar_ow_{idx:03d}.json"
            d_asset = d
            nd = max(0.4, d_asset - 0.35)
            if os.path.exists(ow_cache):
                tts_ow = json.load(open(ow_cache))
            else:
                import soundfile as sf
                from faster_whisper import WhisperModel

                ff(
                    "ffmpeg",
                    "-y",
                    "-i",
                    f"{WORK}/assets_hf/part_{idx:03d}.mp4",
                    "-vn",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    "/tmp/nar_a.wav",
                )
                arr, _ = sf.read("/tmp/nar_a.wav", dtype="float32")
                wm = WhisperModel("base.en", device="cpu", compute_type="int8")
                tts_ow = [
                    {"w": w.word.strip(), "s": w.start, "e": w.end}
                    for ws in wm.transcribe(arr[: int(nd * 16000)], word_timestamps=True)[0]
                    for w in (ws.words or [])
                    if w.word.strip()
                ]
                json.dump(tts_ow, open(ow_cache, "w"))
            events.append(
                {
                    "role": "Narrator",
                    "words": [
                        {
                            "w": w["w"],
                            "s": round(min(w["s"], nd) + cursor, 3),
                            "e": round(min(w["e"], nd) + cursor, 3),
                        }
                        for w in align_words(item["text"], tts_ow, 0.0, nd)
                    ],
                }
            )
        bounds.append(
            {
                "act": item["act"],
                "kind": item["kind"],
                "start": round(cursor, 3),
                "end": round(cursor + d, 3),
            }
        )
        cursor += d
    total = probe_dur_master(plan)
    out = {"events": events, "bounds": bounds, "total_dur": total}
    json.dump(out, open(f"{WORK}/subs_timeline.json", "w"), ensure_ascii=False)
    raw_s = sum(b["end"] - b["start"] for b in bounds if b["kind"] == "raw")
    print(f"timeline: total={total:.2f}s raw_in_master={raw_s:.1f}s events={len(events)}")
    src_dur = json.load(open(f"{WORK}/parts_plan.json"))["source_duration_s"]
    print(f"raw share in master: {raw_s / src_dur:.3f}")


def probe_dur_master(plan):
    t = 0.0
    for idx in range(1, len(plan) + 1):
        p = f"{WORK}/assets_hf/part_{idx:03d}.mp4"
        t += probe_dur(p)
    return round(t, 3)


if __name__ == "__main__":
    stage = sys.argv[1]
    plan, utts, clean = load()
    if stage == "cut":
        stage_cut(plan)
    elif stage == "nar":
        stage_nar(plan)
    elif stage == "audio":
        stage_audio(plan)
    elif stage == "blur":
        stage_blur(plan)
    elif stage == "pack":
        stage_pack(plan)
    elif stage == "timeline":
        stage_timeline(plan)
    else:
        raise SystemExit(f"unknown stage {stage}")
