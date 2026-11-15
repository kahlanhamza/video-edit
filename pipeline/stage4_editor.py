"""Stage 4 EDITOR v3 — freeze-and-explain, smooth (ADR-0004 + ADR-0011/0012).

  HOOK      raw dramatic moment plays untouched WITH audio (catchable opener)
  EXPLAIN   narration over a MUTED replay of the exact semantic moment the
            beat's cue matched — interpolated when slowed (never frame-dup),
            real-time when the window is too still to slow, speed ladder
            0.75->0.30 instead of long freeze pads (freeze capped 1.2s)
  RESUME    between narrations footage plays forward 1.0x WITH original audio
            from where the replay reached, silent stretches capped at 12s
  FLASHBACK desaturated replay when the beat references the past (incl.
            automatic rewinds to an already-passed anchor)
  OUTRO     replay of the most dramatic *moving* moment + end overlay
  NO full-screen text cards anywhere (gated).

Quality craft adopted from video-use (ADR-0012): 30ms audio fades at every
segment cut (no pops), interpolation for all slowed content, motion-floor
rule. Canvas 720p (ADR-0011), CRF 18.
"""
import os, json, math, subprocess
from . import config, interp

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
TIMELINE_V = 10          # bump to invalidate stale cached timelines

def probe_dur(path):
    r = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json",
                        "-show_format", path], capture_output=True, text=True)
    return float(json.loads(r.stdout)["format"]["duration"])

# ------------------------------------------------------------- motion arrays
def _motion_arrays(clips):
    """Per-second motion energy per clip, cached (analyze.motion_energy)."""
    out = {}
    for c in clips:
        p = os.path.join(config.CACHE_DIR, f"motion_{c['key']}.json")
        if os.path.exists(p):
            out[c["key"]] = json.load(open(p))
        else:
            from . import analyze
            arr = analyze.motion_energy(c["path"], c["dur"])
            json.dump(arr, open(p, "w"))
            out[c["key"]] = arr
    return out

def _wmot(arr, start, end):
    a, b = int(max(0, start)), int(max(int(start) + 1, min(len(arr), end)))
    seg = arr[a:b]
    return sum(seg) / len(seg) if seg else 0.0

def _stillness(arr, start, end):
    """Median of 3s-slice motion means across the window (raw 0-255 scale).
    Robust to brief motion at the anchor: catches windows that CONTAIN long
    still stretches even if their average looks lively."""
    a, b = int(max(0, start)), int(min(len(arr), max(int(start) + 1, end)))
    vals = []
    i = a
    while i < b:
        sl = arr[i:min(b, i + 3)]
        if sl:
            vals.append(sum(sl) / len(sl))
        i += 3
    if not vals:
        return 0.0
    vals.sort()
    return vals[len(vals) // 4]   # 25th percentile: 75% of slices must move

# ------------------------------------------------------------- anchor matching
_STOP = set("the a an of and or in on at to with for during from by his her their "
            "person officer man woman one two no is are was near beside into "
            "view scene camera footage moment others stops stopped car".split())

def _tokens(s):
    return {w for w in s.lower().replace(",", " ").split() if w not in _STOP}

def match_anchor(cue, anchors, used_idx):
    cue_t = _tokens(cue)
    best, best_hits = None, -1
    for i, a in enumerate(anchors):
        if i in used_idx:
            continue
        hits = len(cue_t & _tokens(a.get("label", "")))
        if hits > best_hits:
            best, best_hits = i, hits
    if best is None:
        for i in range(len(anchors)):
            if i not in used_idx:
                return i, 0
        return 0, 0
    return best, best_hits

# ------------------------------------------------------------- replay planner
def plan_replay(kind, name, narr, clip, ci, anchors, anchor, cursor, motion,
                speed_cap, grade, punch=True, extra=None):
    """Choose start/speed/window for one muted narration segment.

    Rules (ADR-0011): window motion below MOTION_FLOOR -> real-time muted
    (never slow a still scene); speed ladder absorbs narration length;
    frame-clone freeze capped at MAX_FREEZE; narration tail that cannot fit
    spills into the next segment's muted head instead of freezing.
    """
    D = narr["dur"]
    needed = D + 0.8
    start = max(0.0, min(anchor["t"] - 1.0, clip["dur"] - 2))
    avail = max(1.0, clip["dur"] - start)
    speed = max(min(speed_cap, avail / needed), config.MIN_SPEED)
    if speed >= 0.999:
        speed = 1.0
    # never slow-motion a still scene: median slice motion across the window
    win_end = min(clip["dur"], start + needed * speed)  # window actually played
    if speed < 1.0 and _stillness(motion, start, win_end) < config.MOTION_FLOOR:
        speed = 1.0
    play_dur = avail / speed
    pad = max(0.0, needed - play_dur)
    spill = max(0.0, pad - config.MAX_FREEZE)
    out_dur = needed if spill <= 0 else play_dur + config.MAX_FREEZE
    seg = {"type": kind, "clip_i": ci, "name": name, "narr": narr,
           "narr_offset": config.NARR_OFFSET, "speed": speed,
           "in": start, "out": min(start + needed * speed, clip["dur"]),
           "out_dur": round(out_dur, 3), "punch": punch, "grade": grade,
           "motion": round(_stillness(motion, start, min(clip["dur"], start + out_dur * speed)), 3)}
    if spill > 0:
        seg["narr_spill"] = round(spill, 2)
    seg.update(extra or {})
    if cursor is not None and start < cursor - 2.0 and kind == "replay":
        seg["grade"] = config.FLASHBACK_GRADE
        seg["type"] = "flashback"
        seg["note"] = (seg.get("note", "") + " [rewind]").strip()
    return seg

def seg_dur(s):
    return s["out_dur"] if s["type"] in ("replay", "flashback", "outro_replay") \
        else s["out"] - s["in"]

# ------------------------------------------------------------- timeline build
def build_timeline(script, audio_manifest, anchors_by_clip, clips, motion):
    narr = {b["name"]: b for b in audio_manifest["beats"]}
    beats = script.get("beats", [])
    def act_for(i):
        if i < 4:
            return 0
        if i < 9:
            return 1
        return 2
    used = {k: set() for k in anchors_by_clip}

    timeline = []
    pos = 0.0

    def add(seg):
        nonlocal pos
        seg["t0"] = round(pos, 3)
        seg["dur"] = round(seg_dur(seg), 3)
        pos += seg["dur"]
        timeline.append(seg)

    # ---- HOOK: most dramatic anchor of act 1
    a1 = anchors_by_clip[clips[0]["key"]]
    hook_i = max(range(len(a1)), key=lambda i: a1[i]["score"])
    hook = a1[hook_i]
    h_in = max(0.0, hook["t"] - 1.2)
    h_out = min(clips[0]["dur"], h_in + config.HOOK_SEC)
    add({"type": "hook", "clip_i": 0, "in": h_in, "out": h_out,
         "lowerthird": f"{clips[0]['agency']} | {clips[0]['location']}",
         "note": f"hook anchor: {hook['label']}"})
    cursor = [0.0] * len(clips)
    cursor[0] = h_in

    # ---- cold open over the hook moment (rewind + deep slow-mo, interpolated)
    co = narr.get("cold_open")
    if co:
        seg = plan_replay("replay", "cold_open", co, clips[0], 0, a1, hook, cursor[0],
                          motion[clips[0]["key"]], config.DEEP_SPEED,
                          config.FREEZE_GRADE,
                          extra={"note": f"cold open over hook: {hook['label']}"})
        add(seg)
        cursor[0] = seg["in"] + seg["dur"] * seg["speed"]

    # ---- acts
    act_beats = {}
    for i, b in enumerate(beats):
        act_beats.setdefault(act_for(i), []).append((i, b))
    chapter_marks = []

    for act in range(3):
        clip = clips[act]
        anchors = anchors_by_clip[clip["key"]]
        mot = motion[clip["key"]]
        if act > 0:
            first_t = anchors[0]["t"] if anchors else 0
            cursor[act] = max(0.0, first_t - 55.0)
        add({"type": "forward", "clip_i": act, "in": cursor[act],
             "out": min(cursor[act] + 10.0, clip["dur"]),
             "note": "act entry"})
        cursor[act] = timeline[-1]["out"]
        chapter_marks.append({"clip": act + 1, "t_video": round(timeline[-1]["t0"], 1),
                              "t_source": round(cursor[act], 1)})

        act_list = act_beats.get(act, [])
        for bi, (i, b) in enumerate(act_list):
            u = narr.get(f"beat_{i+1}")
            if not u:
                continue
            if bi == 0:
                timeline[-1]["lowerthird"] = f"{clip['agency']} | {clip['location']}"
            # flashback requested by the script: pick a moving past window
            if b.get("mode") == "flashback":
                fb = [a for a in anchors if a["t"] < cursor[act] - 5
                      and _stillness(mot, a["t"] - 1, a["t"] + 6) >= config.MOTION_FLOOR]
                if fb:
                    f = fb[-1]
                    seg = plan_replay("flashback", u["name"], u, clip, act, anchors, f,
                                      cursor[act], mot, config.FLASHBACK_SPEED,
                                      config.FLASHBACK_GRADE,
                                      extra={"note": f"flashback: {f['label']}"})
                    add(seg)
                    cursor[act] = seg["in"] + seg["dur"] * seg["speed"]
                    continue
            idx, hits = match_anchor(b.get("cue", ""), anchors, used[clip["key"]])
            used[clip["key"]].add(idx)
            a = anchors[idx]
            seg = plan_replay("replay", u["name"], u, clip, act, anchors, a,
                              cursor[act], mot, config.REPLAY_SPEED,
                              config.FREEZE_GRADE,
                              extra={"note": f"cue '{b.get('cue','')}' -> {a['label']} (hits={hits})"})
            add(seg)
            cursor[act] = seg["in"] + seg["dur"] * seg["speed"]
            # forward gap: viewer resumes from where the replay stopped (<=12s)
            if clip["dur"] - cursor[act] >= 4.0:
                g = min(12.0 if bi == len(act_list) - 1 else 10.0,
                        clip["dur"] - cursor[act])
                fwd = {"type": "forward", "clip_i": act, "in": cursor[act],
                       "out": min(cursor[act] + g, clip["dur"]), "note": "resume"}
                if seg.get("narr_spill"):
                    fwd["mute_head"] = seg["narr_spill"]
                    fwd["note"] = f"resume (muted {seg['narr_spill']}s: narration tail)"
                add(fwd)
                cursor[act] = timeline[-1]["out"]

    # ---- outro over the most dramatic MOVING moment of act 3 + end overlay
    oc = narr.get("outro")
    a3 = anchors_by_clip[clips[2]["key"]]
    mot3 = motion[clips[2]["key"]]
    fin = max(a3, key=lambda a: a["score"] * (0.5 + 0.5 * a.get("motion", 0.5)))
    seg = plan_replay("outro_replay", "outro", oc, clips[2], 2, a3, fin,
                      cursor[2], mot3, config.REPLAY_SPEED, config.FREEZE_GRADE,
                      extra={"note": f"outro over: {fin['label']}"})
    # end overlay deserves a clean tail: +2.5s beyond narration, no VO
    seg["out_dur"] = round(seg["out_dur"] + 2.5, 3)
    seg["out"] = min(seg["in"] + seg["out_dur"] * seg["speed"], clips[2]["dur"])
    add(seg)
    return timeline, chapter_marks

# ------------------------------------------------------------- rendering
def _has_audio(path):
    r = subprocess.run(["ffprobe", "-v", "quiet", "-select_streams", "a",
                        "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                        path], capture_output=True, text=True)
    return "audio" in r.stdout

def _codec_name(path):
    r = subprocess.run(["ffprobe", "-v", "quiet", "-select_streams", "v:0",
                        "-show_entries", "stream=codec_name", "-of", "csv=p=0",
                        path], capture_output=True, text=True)
    return (r.stdout or "").strip()

def _base_geom(punch=False):
    w, h = config.RES_W, config.RES_H
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos,"
          f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1")
    if punch:
        vw, vh = int(w * config.PUNCH), int(h * config.PUNCH)
        vf += f",scale={vw}:{vh}:flags=lanczos,crop={w}:{h},setsar=1"
    return vf

def _drawtext(tf_path, x, y, size, t0, t1):
    return (f"drawtext=fontfile={FONT_BOLD}:textfile={tf_path}:fontsize={size}:"
            f"fontcolor=white:x={x}:y={y}:box=1:boxcolor=black@0.55:"
            f"boxborderw=12:enable='between(t,{t0},{t1})'")

def _afade_out(st):
    return f"afade=t=out:st={st:.3f}:d={config.AUDIO_FADE}"

def render_segment(seg, out_path, clips):
    dur_out = seg["dur"]
    clip_path = clips[seg["clip_i"]]["path"]
    dec = ["-c:v", "libdav1d"] if _codec_name(clip_path) == "av1" else []
    fade = config.AUDIO_FADE
    label_vf = ""
    if seg.get("lowerthird"):
        tf = out_path + ".lt.txt"
        with open(tf, "w") as f:
            f.write(seg["lowerthird"])
        label_vf = "," + _drawtext(tf, 36, config.RES_H - 100, 28, 0.6, 5.0)
    af = [f"afade=t=in:st=0:d={fade}", _afade_out(max(0.0, dur_out - fade))]
    if seg["type"] in ("forward", "hook"):
        mute_head = seg.get("mute_head")
        vf = _base_geom() + f",fps={config.FPS}" + label_vf
        cmd = ["ffmpeg", "-y", "-ss", f"{seg['in']:.3f}"] + dec + \
              ["-i", clip_path, "-t", f"{dur_out:.3f}"]
        if _has_audio(clip_path):
            vol = "" if seg["type"] == "hook" else f"volume={config.FORWARD_BED_DB}dB,"
            if mute_head:
                vol += f"volume=enable='lt(t,{mute_head:.2f})':volume=0,"
            cmd += ["-af", vol + "aresample=async=1:first_pts=0," + ",".join(af)]
        else:
            cmd += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
                    "-af", ",".join(af)]
        cmd += ["-vf", vf, "-r", str(config.FPS)]
        if not _has_audio(clip_path):
            cmd += ["-shortest"]
        cmd += ["-c:v", "libx264", "-preset", config.X264_PRESET,
                "-crf", str(config.CRF), "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "192k",
                out_path]
        subprocess.run(cmd, check=True, capture_output=True)
    else:
        speed = seg.get("speed", config.REPLAY_SPEED)
        raw_len = seg["out"] - seg["in"]
        vf = _base_geom(seg.get("punch"))
        if seg.get("grade"):
            vf += "," + seg["grade"]
        vf += "," + interp.slow_chain(speed) + label_vf
        pad = max(0.0, dur_out - raw_len / speed)
        if pad > 0.05:
            pad = min(pad, config.MAX_FREEZE)
            vf += f",tpad=stop_mode=clone:stop_duration={pad:.2f}"
        if seg["type"] == "flashback":
            tf = out_path + ".fb.txt"
            with open(tf, "w") as f:
                f.write("REWIND")
            vf += "," + _drawtext(tf, 40, 60, 26, 0.4, 3.0)
        if seg["type"] == "outro_replay":
            tf1 = out_path + ".end1.txt"
            tf2 = out_path + ".end2.txt"
            with open(tf1, "w") as f:
                f.write(config.END_OVERLAY_LINE1)
            with open(tf2, "w") as f:
                f.write(config.END_OVERLAY_LINE2)
            vf += ("," + _drawtext(tf1, "(w-text_w)/2", config.RES_H // 2 - 60,
                                   46, 0.8, dur_out)
                   + "," + _drawtext(tf2, "(w-text_w)/2", config.RES_H // 2 + 20,
                                     32, 1.2, dur_out))
        cmd = ["ffmpeg", "-y", "-ss", f"{seg['in']:.3f}"] + dec + \
              ["-i", clip_path, "-t", f"{raw_len:.3f}",
               "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
               "-map", "0:v", "-map", "1:a",
               "-af", ",".join(af),
               "-vf", vf, "-t", f"{dur_out:.3f}", "-r", str(config.FPS),
               "-c:v", "libx264", "-preset", config.X264_PRESET,
               "-crf", str(config.CRF), "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "192k",
               out_path]
        subprocess.run(cmd, check=True, capture_output=True)

def render_all(timeline, clips):
    seg_files = []
    for n, seg in enumerate(timeline):
        out = os.path.join(config.CACHE_DIR, f"seg_{n:03d}.mp4")
        if not (os.path.exists(out) and os.path.getsize(out) > 50_000):
            render_segment(seg, out, clips)
            print(f"  seg {n:03d} [{seg['type']}] {seg['dur']:.1f}s "
                  f"speed={seg.get('speed','1.0')} {seg.get('note','')[:40]}")
        else:
            print(f"  seg {n:03d} cached [{seg['type']}]")
        seg_files.append(out)
    return seg_files

def concat(seg_files, out_path):
    lst = out_path + ".list.txt"
    with open(lst, "w") as f:
        for p in seg_files:
            f.write(f"file '{p}'\n")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                    "-c", "copy", out_path], check=True, capture_output=True)
    os.remove(lst)

def narration_track(timeline, total, out_path):
    inputs, chains, labels = [], [], []
    for seg in timeline:
        u = seg.get("narr")
        if not u:
            continue
        start_ms = int((seg["t0"] + seg.get("narr_offset", 0.3)) * 1000)
        inputs += ["-i", u["path"]]
        idx = len(labels)
        chains.append(f"[{idx}:a]aformat=sample_rates=48000:channel_layouts=stereo,"
                      f"adelay={start_ms}|{start_ms}[s{idx}]")
        labels.append(f"[s{idx}]")
    graph = ";".join(chains) + ";" + "".join(labels) + \
        f"amix=inputs={len(labels)}:duration=longest:normalize=0,apad[out]"
    subprocess.run(["ffmpeg", "-y"] + inputs + ["-filter_complex", graph,
                    "-map", "[out]", "-t", f"{total:.3f}", out_path],
                   check=True, capture_output=True)

def final_mux(concat_path, narr_path, out_path, total):
    subprocess.run(["ffmpeg", "-y", "-i", concat_path, "-i", narr_path,
                    "-filter_complex",
                    "[0:a][1:a]amix=inputs=2:duration=first:normalize=0[mix]",
                    "-map", "0:v", "-map", "[mix]", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k", "-t", f"{total:.3f}",
                    out_path], check=True, capture_output=True)

# ------------------------------------------------------------- entry
def run(case_id, script, audio_manifest):
    tl_path = os.path.join(config.CACHE_DIR, f"case{case_id}_timeline.json")
    clips = []
    for item in config.USED_POOL:
        p = os.path.join(config.FOOTAGE_DIR, item["key"] + ".webm")
        clips.append({**item, "path": p, "dur": probe_dur(p)})
    if os.path.exists(tl_path):
        tl_all = json.load(open(tl_path))
        if tl_all.get("v") != TIMELINE_V:
            tl_all = None
    else:
        tl_all = None
    if tl_all:
        timeline = tl_all["timeline"]
        print("  timeline cached")
    else:
        motion = _motion_arrays(clips)
        anchors_by_clip = {}
        for item in clips:
            ap = os.path.join(config.CACHE_DIR, f"anchors_{item['key']}.json")
            anchors_by_clip[item["key"]] = json.load(open(ap))["anchors"]
        timeline, chapter_marks = build_timeline(script, audio_manifest,
                                                 anchors_by_clip, clips, motion)
        json.dump({"v": TIMELINE_V, "timeline": timeline,
                   "chapters": chapter_marks},
                  open(tl_path, "w"), indent=2, ensure_ascii=False)
        tl_all = {"timeline": timeline, "chapters": chapter_marks}
        for s in timeline:
            print(f"  [{s['type']:>13}] t0={s['t0']:>6.1f} dur={s['dur']:>5.1f}"
                  f" speed={s.get('speed','-')}"
                  f"  {s.get('note','')[:50]}")
    total = sum(s["dur"] for s in timeline)
    print(f"  timeline: {len(timeline)} segments, {total:.0f}s ({total/60:.1f} min)")
    seg_files = render_all(timeline, clips)
    concat_path = os.path.join(config.CACHE_DIR, f"case{case_id}_concat.mp4")
    if not os.path.exists(concat_path):
        concat(seg_files, concat_path)
    narr_path = os.path.join(config.CACHE_DIR, f"case{case_id}_narration.m4a")
    if not os.path.exists(narr_path):
        narration_track(timeline, total, narr_path)
    final = os.path.join(config.VIDEO_DIR, f"case{case_id}_final.mp4")
    final_mux(concat_path, narr_path, final, total)
    dur = probe_dur(final)
    print(f"  FINAL: {final} ({dur/60:.1f} min, {os.path.getsize(final)//1048576}MB)")
    return final, dur, tl_all["timeline"], tl_all["chapters"]
