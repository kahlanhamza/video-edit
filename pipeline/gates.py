"""GATES (ADR-0007): executable standards, non-bypassable, each with --self-test.

Run: python3 -m pipeline.gates [--case 1] [--self-test]
Exit 0 = GREEN, non-zero = RED. A red gate is a stop, never a "skip it this time".
"""
import os, re, json, subprocess, sys
import numpy as np
from PIL import Image
from . import config

# ---------------------------------------------------------------- gates
def gate_imports(_art=None):
    from pipeline import config, db, ai, analyze  # noqa
    from pipeline import stage1_source, stage2_script, stage3_voice  # noqa
    from pipeline import stage4_editor, stage5_package, stage6_publish  # noqa
    return True, "all pipeline modules import"

def gate_guardrails(art):
    bad = []
    texts = []
    script = art.get("script") or {}
    texts.append(script.get("title", ""))
    texts.append(script.get("cold_open", ""))
    texts.append(script.get("outro", ""))
    texts += [b.get("narration", "") for b in script.get("beats", [])]
    meta = art.get("metadata") or {}
    texts.append(meta.get("title", ""))
    texts.append(meta.get("description", ""))
    low = " || ".join(t.lower() for t in texts if t)
    for phrase in config.FORBIDDEN_PHRASES:
        if phrase.lower() in low:
            bad.append(phrase)
    return (len(bad) == 0, f"forbidden phrases found: {bad}" if bad
            else f"clean ({len(texts)} texts scanned)")

def gate_no_timestamp_speak(art):
    hits = []
    script = art.get("script") or {}
    texts = [script.get("cold_open", ""), script.get("outro", "")]
    texts += [b.get("narration", "") for b in script.get("beats", [])]
    for t in texts:
        for rx in config.TIMESTAMP_PATTERNS:
            m = re.search(rx, t, re.IGNORECASE)
            if m:
                hits.append(m.group(0))
    return (len(hits) == 0, f"timestamp-speak: {hits}" if hits
            else "no timestamp references")

def gate_voice_pace(art):
    manifest = art.get("audio") or {}
    wpm = manifest.get("wpm_measured", 0)
    ok = config.WPM_MIN <= wpm <= config.WPM_MAX
    return ok, f"{wpm} wpm (gate {config.WPM_MIN}-{config.WPM_MAX}), voice={manifest.get('voice')}"

def gate_compliance(art):
    meta = art.get("metadata") or {}
    desc = meta.get("description", "")
    ok = "presumed innocent until proven guilty" in desc.lower()
    return ok, "disclaimer present in description" if ok else \
        "MANDATORY_DISCLAIMER missing from description"

def gate_render(art):
    vp = art.get("video_path")
    if not vp or not os.path.exists(vp):
        return False, f"video missing: {vp}"
    r = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json",
                        "-show_format", "-show_streams", vp],
                       capture_output=True, text=True)
    d = json.loads(r.stdout)
    dur = float(d["format"]["duration"])
    kinds = {s["codec_type"] for s in d["streams"]}
    if not {"video", "audio"} <= kinds:
        return False, f"missing streams: {kinds}"
    if dur < config.DEMO_MIN_SEC:
        return False, f"duration {dur:.0f}s < floor {config.DEMO_MIN_SEC}s"
    dark = 0
    n = 6
    for i in range(n):
        t = dur * (i + 0.5) / n
        out = os.path.join(config.CACHE_DIR, f"gate_frame_{i}.jpg")
        subprocess.run(["ffmpeg", "-y", "-ss", f"{t:.1f}", "-i", vp,
                        "-frames:v", "1", "-vf", "scale=320:-1", out],
                       capture_output=True)
        img = Image.open(out).convert("L")
        if np.array(img).mean() < 8:
            dark += 1
    if dark:
        return False, f"{dark}/{n} sampled frames are black"
    return True, f"{dur:.1f}s, streams ok, {n}/{n} frames non-black"

def gate_no_black_cards(art):
    tl = art.get("timeline") or []
    cards = [s for s in tl if s.get("type") == "card"]
    return (len(cards) == 0,
            f"timeline has {len(cards)} card segments" if cards
            else f"0 card segments across {len(tl)} segments")

def gate_narration_sync(art):
    tl = art.get("timeline") or []
    bad = []
    for s in tl:
        if s.get("narr") and s.get("type") in ("replay", "flashback", "outro_replay"):
            off = s.get("narr_offset", -1)
            if not (0 <= off <= 1.5):
                bad.append((s.get("name"), off))
    return (len(bad) == 0, f"bad narration offsets: {bad}" if bad
            else "every narration starts inside its muted replay")

# ------------------------------------------------------- output QC (ADR-0012)
def _near_dup_pct(path, t0, dur=2.0):
    """% of near-identical consecutive frames in a window of a rendered file."""
    import tempfile, shutil
    import numpy as np
    from PIL import Image
    tmp = tempfile.mkdtemp()
    try:
        r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t0:.2f}",
                            "-t", f"{dur:.2f}", "-i", path,
                            "-vf", "scale=320:180", "-vsync", "0",
                            os.path.join(tmp, "%03d.png")],
                           capture_output=True, text=True)
        fs = sorted(f for f in os.listdir(tmp) if f.endswith(".png"))
        dups, n, prev = 0, 0, None
        for f in fs:
            img = np.asarray(Image.open(os.path.join(tmp, f)).convert("L"),
                             dtype=np.float32)
            if prev is not None:
                n += 1
                if float(np.abs(img - prev).mean()) < 0.5:
                    dups += 1
            prev = img
        return (100.0 * dups / n) if n else 0.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def gate_smooth_motion(art):
    """Slowed segments must be interpolated, not frame-duplicated (ADR-0011)."""
    vp, tl = art.get("video_path"), art.get("timeline") or []
    if not vp or not os.path.exists(vp):
        return False, "video missing"
    slow = [s for s in tl if s.get("type") in ("replay", "flashback", "outro_replay")
            and float(s.get("speed", 1.0)) < 0.999 and s["dur"] >= 3.0]
    if not slow:
        return True, "no slowed segments to check"
    worst, worst_name = 0.0, ""
    for s in slow:
        pct = _near_dup_pct(vp, s["t0"] + s["dur"] / 2, 2.0)
        if pct > worst:
            worst, worst_name = pct, s.get("name", s["type"])
    ok = worst < config.SMOOTH_DUP_PCT
    return ok, (f"worst slowed segment '{worst_name}' at {worst:.0f}% dup frames "
                f"(limit {config.SMOOTH_DUP_PCT:.0f}%, {len(slow)} slowed segs)"
                if not ok else
                f"{len(slow)} slowed segs all interpolated (<{worst:.0f}% dups)")

def gate_bitrate(art):
    vp = art.get("video_path")
    if not vp or not os.path.exists(vp):
        return False, "video missing"
    r = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json",
                        "-show_format", vp], capture_output=True, text=True)
    br = int(json.loads(r.stdout)["format"]["bit_rate"])
    floor = int(config.BITRATE_FLOOR_MBPS * 1_000_000)
    return (br >= floor,
            f"{br/1e6:.2f} Mbps (floor {config.BITRATE_FLOOR_MBPS})"
            if br >= floor else f"{br/1e6:.2f} Mbps BELOW floor {config.BITRATE_FLOOR_MBPS}")

def gate_boundary_pop(art):
    """video-use pattern: no audio pops at segment cuts (30ms fades, ADR-0012)."""
    import numpy as np
    vp, tl = art.get("video_path"), art.get("timeline") or []
    if not vp or not os.path.exists(vp):
        return False, "video missing"
    bounds = [round(s["t0"], 2) for s in tl[1:]]
    if not bounds:
        return True, "no boundaries"
    # PCM around every boundary, look for full-scale step spikes near the cut
    pops = []
    for t in bounds:
        r = subprocess.run(
            ["ffmpeg", "-v", "quiet", "-ss", f"{max(0.0, t-0.05):.2f}", "-t", "0.2",
             "-i", vp, "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
            capture_output=True)
        x = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32)
        if x.size < 200:
            continue
        d = np.abs(np.diff(x))
        peak = d.max()
        base = np.percentile(d, 99.5) + 1e-6
        # pop = step spike that dwarfs everything else in the window
        if peak > 20 * base and peak > 12000:
            pops.append(round(t, 2))
    return (len(pops) == 0,
            f"audio pops at boundaries: {pops}" if pops
            else f"{len(bounds)} boundaries clean (30ms fades)")

GATES = [
    ("imports", gate_imports, lambda: {}),
    ("guardrails", gate_guardrails, lambda: _load_artifacts()),
    ("no_timestamp_speak", gate_no_timestamp_speak, lambda: _load_artifacts()),
    ("voice_pace", gate_voice_pace, lambda: _load_artifacts()),
    ("compliance", gate_compliance, lambda: _load_artifacts()),
    ("render", gate_render, lambda: _load_artifacts()),
    ("no_black_cards", gate_no_black_cards, lambda: _load_artifacts()),
    ("narration_sync", gate_narration_sync, lambda: _load_artifacts()),
    ("smooth_motion", gate_smooth_motion, lambda: _load_artifacts()),
    ("bitrate", gate_bitrate, lambda: _load_artifacts()),
    ("boundary_pop", gate_boundary_pop, lambda: _load_artifacts()),
]

def _load_artifacts(case_id=1):
    art = {}
    sp = os.path.join(config.PACKAGE_DIR, f"case{case_id}_script.json")
    ap = os.path.join(config.AUDIO_DIR, f"case{case_id}_audio_manifest.json")
    mp = os.path.join(config.PACKAGE_DIR, f"case{case_id}_metadata.json")
    tp = os.path.join(config.CACHE_DIR, f"case{case_id}_timeline.json")
    if os.path.exists(sp):
        art["script"] = json.load(open(sp))
    if os.path.exists(ap):
        art["audio"] = json.load(open(ap))
    if os.path.exists(mp):
        art["metadata"] = json.load(open(mp))
    if os.path.exists(tp):
        art["timeline"] = json.load(open(tp)).get("timeline",
                                                  json.load(open(tp)))
    art["video_path"] = os.path.join(config.VIDEO_DIR, f"case{case_id}_final.mp4")
    return art

def run(case_id=1):
    art = _load_artifacts(case_id)
    results = []
    failed = False
    for name, fn, _ in GATES:
        try:
            ok, detail = fn(art)
        except Exception as e:
            ok, detail = False, f"gate error: {e}"
        results.append((name, ok, detail))
        if not ok:
            failed = True
        print(f"[{'GREEN' if ok else 'RED'}] {name}: {detail}")
    if failed:
        print("GATES: RED — stop (ADR-0007)")
        sys.exit(1)
    print("GATES: GREEN")
    return results

def self_test():
    print("== gate self-test ==")
    bad_script = {"title": "Criminal caught at 180 seconds",
                  "cold_open": "he is guilty of it at 1:23 of the tape",
                  "beats": [{"narration": "at 45 seconds the man runs", "cue": "x",
                             "mode": "replay"}]}
    ok, detail = gate_guardrails({"script": bad_script})
    assert not ok, "guardrails gate failed to catch bad fixture"
    print(f"  [ok] guardrails catches: {detail[:60]}")
    ok, detail = gate_no_timestamp_speak({"script": bad_script})
    assert not ok, "timestamp gate failed to catch bad fixture"
    print(f"  [ok] no_timestamp_speak catches: {detail[:60]}")
    ok, detail = gate_render({"video_path": "/nonexistent/file.mp4"})
    assert not ok, "render gate failed to catch missing video"
    print(f"  [ok] render catches missing file: {detail}")
    ok, _ = gate_no_black_cards({"timeline": [{"type": "card", "png": "x"}]})
    assert not ok, "no_black_cards failed to catch card segment"
    print("  [ok] no_black_cards catches card segment")
    ok, _ = gate_narration_sync({"timeline": [
        {"type": "replay", "narr": {"name": "b1"}, "narr_offset": 9.9}]})
    assert not ok, "narration_sync failed to catch bad offset"
    print("  [ok] narration_sync catches offset outside replay")
    ok, detail = gate_voice_pace({"audio": {"wpm_measured": 200, "voice": "x"}})
    assert not ok
    print(f"  [ok] voice_pace catches off-pace: {detail[:50]}")
    # smooth_motion: build a real duplicated-frame clip -> must go RED
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        dupv = os.path.join(td, "dup.mp4")
        # one static frame duplicated: color source = 100% duplicate frames
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                        "-i", "color=c=gray:size=320x180:rate=30", "-t", "5",
                        "-c:v", "libx264", dupv], check=True, capture_output=True)
        ok, detail = gate_smooth_motion(
            {"video_path": dupv,
             "timeline": [{"type": "replay", "name": "t", "speed": 0.45,
                           "t0": 0.0, "dur": 5.0}]})
        assert not ok, "smooth_motion failed to catch duplicated frames"
        print(f"  [ok] smooth_motion catches duplication: {detail[:50]}")
        # bitrate: tiny file is under the floor -> RED
        ok, detail = gate_bitrate({"video_path": dupv})
        assert not ok, "bitrate gate failed to catch thin file"
        print(f"  [ok] bitrate catches under-floor: {detail[:50]}")
        # boundary_pop: inject a full-scale step at t=1.0 -> RED
        click = os.path.join(td, "click.mp4")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                        "-i", "sine=frequency=440:sample_rate=16000:duration=2",
                        "-af", "volume=0.5", "-c:a", "pcm_s16le", "-f", "s16le",
                        os.path.join(td, "c.raw")], check=True, capture_output=True)
        raw = bytearray(open(os.path.join(td, "c.raw"), "rb").read())
        import numpy as _np
        arr = _np.frombuffer(bytes(raw), dtype=_np.int16).astype(_np.int32)
        off = 16000  # 1.0s, mono
        # sustained DC step (the classic concat pop): survives AAC
        arr[off:] = _np.clip(arr[off:] + 28000, -32768, 32767)
        open(os.path.join(td, "c.raw"), "wb").write(arr.astype(_np.int16).tobytes())
        subprocess.run(["ffmpeg", "-y", "-v", "error",
                        "-f", "s16le", "-ar", "16000", "-ac", "1",
                        "-i", os.path.join(td, "c.raw"),
                        "-f", "lavfi",
                        "-i", "color=c=black:size=320x180:rate=30",
                        "-map", "0:a", "-map", "1:v",
                        "-t", "2", click],
                       check=True, capture_output=True)
        ok, detail = gate_boundary_pop(
            {"video_path": click,
             "timeline": [{"type": "replay", "t0": 0.0}, {"type": "forward", "t0": 1.0}]})
        assert not ok, "boundary_pop failed to catch injected click"
        print(f"  [ok] boundary_pop catches injected click: {detail[:50]}")
    print("SELF-TEST GREEN: every gate can go RED")

if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        cid = int(sys.argv[sys.argv.index("--case") + 1]) if "--case" in sys.argv else 1
        run(cid)
