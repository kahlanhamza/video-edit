"""Stage 4 EDITOR v2 — freeze-and-explain (ADR-0004, AC-4).

  HOOK      raw dramatic moment plays untouched WITH audio (catchable opener)
  EXPLAIN   narration over a MUTED slow-motion replay of the exact semantic
            moment the beat's cue matched (anchor-matched, zoom + grade);
            `tpad` freeze extends the replay when the narrator needs more time
  RESUME    between narrations footage plays forward 1.0x WITH original audio
            from the point the replay reached — the viewer "keeps watching"
  FLASHBACK desaturated slow-mo when the beat references the past
  OUTRO     slow replay of the final moment + on-frame end overlay
  NO full-screen text cards anywhere (gated).
"""
import os, json, subprocess
from . import config

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

def probe_dur(path):
    r = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json",
                        "-show_format", path], capture_output=True, text=True)
    return float(json.loads(r.stdout)["format"]["duration"])

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

# ------------------------------------------------------------- timeline build
def build_timeline(script, audio_manifest, anchors_by_clip, clips):
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

    def seg_dur(s):
        return s["out_dur"] if s["type"] in ("replay", "flashback", "outro_replay") \
            else s["out"] - s["in"]

    def add(seg):
        nonlocal pos
        seg["t0"] = round(pos, 3)
        seg["dur"] = round(seg_dur(seg), 3)
        pos += seg["dur"]
        timeline.append(seg)

    # pacing: gaps sized to land runtime near target
    narr_total = sum(b["dur"] for b in audio_manifest["beats"])
    target = script.get("_target_sec", 960)
    gap_budget = max(120.0, target - narr_total * 1.45 - config.HOOK_SEC - 40)
    n_gaps = max(1, len(beats))
    gap_base = min(22.0, max(6.0, gap_budget / n_gaps))

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

    # ---- cold open over the hook moment (rewind + muted slow-mo)
    co = narr.get("cold_open")
    if co:
        raw_len = co["dur"] * config.REPLAY_SPEED
        win_out = min(h_in + raw_len, clips[0]["dur"])
        add({"type": "replay", "clip_i": 0, "name": "cold_open", "narr": co,
             "narr_offset": config.NARR_OFFSET, "speed": config.REPLAY_SPEED,
             "in": h_in, "out": win_out, "out_dur": co["dur"] + 0.8,
             "punch": True, "grade": config.FREEZE_GRADE,
             "note": f"cold open over hook: {hook['label']}"})
        cursor[0] = win_out

    # ---- acts
    act_beats = {}
    for i, b in enumerate(beats):
        act_beats.setdefault(act_for(i), []).append((i, b))
    chapter_marks = []

    for act in range(3):
        clip = clips[act]
        anchors = anchors_by_clip[clip["key"]]
        if act > 0:
            first_t = anchors[0]["t"] if anchors else 0
            cursor[act] = max(0.0, first_t - 55.0)
        entry_gap = min(gap_base + 4.0, 24.0)
        add({"type": "forward", "clip_i": act, "in": cursor[act],
             "out": min(cursor[act] + entry_gap, clip["dur"]),
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
            D = u["dur"]
            did_flashback = False
            if b.get("mode") == "flashback":
                fb_candidates = [a for a in anchors if a["t"] < cursor[act] - 5]
                if fb_candidates:
                    fb = fb_candidates[-1]
                    raw_len = min(D * config.FLASHBACK_SPEED,
                                  max(2.0, clip["dur"] - fb["t"]))
                    add({"type": "flashback", "clip_i": act, "name": u["name"],
                         "narr": u, "narr_offset": config.NARR_OFFSET,
                         "speed": config.FLASHBACK_SPEED,
                         "in": fb["t"], "out": min(fb["t"] + raw_len, clip["dur"]),
                         "out_dur": D + 0.8, "punch": True,
                         "grade": config.FLASHBACK_GRADE,
                         "note": f"flashback: {fb['label']}"})
                    did_flashback = True
            if not did_flashback:
                idx, hits = match_anchor(b.get("cue", ""), anchors, used[clip["key"]])
                used[clip["key"]].add(idx)
                a = anchors[idx]
                replay_start = max(0.0, min(a["t"] - 1.0, clip["dur"] - 2))
                if replay_start <= cursor[act] + 1.0:
                    replay_start = min(cursor[act] + 1.0, clip["dur"] - 2)
                raw_len = D * config.REPLAY_SPEED
                win_out = min(replay_start + raw_len, clip["dur"])
                add({"type": "replay", "clip_i": act, "name": u["name"],
                     "narr": u, "narr_offset": config.NARR_OFFSET,
                     "speed": config.REPLAY_SPEED,
                     "in": replay_start, "out": win_out, "out_dur": D + 0.8,
                     "punch": True, "grade": config.FREEZE_GRADE,
                     "note": f"cue '{b.get('cue','')}' -> {a['label']} (hits={hits})"})
                cursor[act] = win_out
            # forward gap: viewer resumes from where the replay stopped
            g = min(gap_base * (1.35 if bi == len(act_list) - 1 else 1.0),
                    clip["dur"] - cursor[act])
            if g >= 4.0:
                add({"type": "forward", "clip_i": act, "in": cursor[act],
                     "out": cursor[act] + g})
                cursor[act] = timeline[-1]["out"]

    # ---- outro over final dramatic moment of act 3 + end overlay
    oc = narr.get("outro")
    a3 = anchors_by_clip[clips[2]["key"]]
    fin = max(a3, key=lambda a: a["score"])
    raw_len = ((oc["dur"] if oc else 20.0) * config.REPLAY_SPEED)
    add({"type": "outro_replay", "clip_i": 2,
         "in": max(0.0, fin["t"] - 1.0),
         "out": min(max(0.0, fin["t"] - 1.0) + raw_len, clips[2]["dur"]),
         "out_dur": (oc["dur"] + 2.0) if oc else 20.0,
         "narr": oc, "narr_offset": config.NARR_OFFSET,
         "speed": config.REPLAY_SPEED, "punch": True,
         "grade": config.FREEZE_GRADE,
         "note": f"outro over: {fin['label']}"})
    return timeline, chapter_marks

# ------------------------------------------------------------- rendering
def _has_audio(path):
    r = subprocess.run(["ffprobe", "-v", "quiet", "-select_streams", "a",
                        "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                        path], capture_output=True, text=True)
    return "audio" in r.stdout

def _decode_args(path):
    r = subprocess.run(["ffprobe", "-v", "quiet", "-select_streams", "v:0",
                        "-show_entries", "stream=codec_name", "-of", "csv=p=0",
                        path], capture_output=True, text=True)
    return ["-c:v", "libdav1d"] if (r.stdout or "").strip() == "av1" else []

def _base_geom(punch=False):
    vf = ("scale=1920:1080:force_original_aspect_ratio=decrease,"
          "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1")
    if punch:
        vf += ",scale=2054:1156,crop=1920:1080,setsar=1"
    return vf

def _drawtext(tf_path, x, y, size, t0, t1):
    return (f"drawtext=fontfile={FONT_BOLD}:textfile={tf_path}:fontsize={size}:"
            f"fontcolor=white:x={x}:y={y}:box=1:boxcolor=black@0.55:"
            f"boxborderw=12:enable='between(t,{t0},{t1})'")

def _codec(out_path):
    return ["-c:v", "libx264", "-preset", config.X264_PRESET,
            "-crf", str(config.CRF), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "192k",
            out_path]

def render_segment(seg, out_path, clips):
    dur_out = seg["dur"]
    clip_path = clips[seg["clip_i"]]["path"]
    dec = _decode_args(clip_path)
    label_vf = ""
    if seg.get("lowerthird"):
        tf = out_path + ".lt.txt"
        with open(tf, "w") as f:
            f.write(seg["lowerthird"])
        label_vf = "," + _drawtext(tf, 36, 980, 34, 0.6, 5.0)
    if seg["type"] in ("forward", "hook"):
        vf = _base_geom() + ",fps=30" + label_vf
        cmd = ["ffmpeg", "-y", "-ss", f"{seg['in']:.3f}"] + dec + \
              ["-i", clip_path, "-t", f"{dur_out:.3f}"]
        if _has_audio(clip_path):
            vol = "" if seg["type"] == "hook" else f"volume={config.FORWARD_BED_DB}dB,"
            cmd += ["-af", f"{vol}aresample=async=1:first_pts=0"]
        else:
            cmd += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
        cmd += ["-vf", vf, "-r", "30"]
        if not _has_audio(clip_path):
            cmd += ["-shortest"]
        cmd += _codec(out_path)
        subprocess.run(cmd, check=True, capture_output=True)
    else:
        speed = seg.get("speed", config.REPLAY_SPEED)
        raw_len = seg["out"] - seg["in"]
        vf = _base_geom(seg.get("punch"))
        if seg.get("grade"):
            vf += "," + seg["grade"]
        vf += f",setpts=PTS/{speed:.2f},fps=30"
        pad = max(0.0, dur_out - raw_len / speed)
        if pad > 0.05:
            vf += f",tpad=stop_mode=clone:stop_duration={pad:.2f}"
        if seg["type"] == "flashback":
            tf = out_path + ".fb.txt"
            with open(tf, "w") as f:
                f.write("FLASHBACK")
            vf += "," + _drawtext(tf, 40, 60, 30, 0.4, 3.0)
        if seg["type"] == "outro_replay":
            tf1 = out_path + ".end1.txt"
            tf2 = out_path + ".end2.txt"
            with open(tf1, "w") as f:
                f.write(config.END_OVERLAY_LINE1)
            with open(tf2, "w") as f:
                f.write(config.END_OVERLAY_LINE2)
            vf += ("," + _drawtext(tf1, "(w-text_w)/2", 470, 58, 0.8, dur_out)
                   + "," + _drawtext(tf2, "(w-text_w)/2", 560, 40, 1.2, dur_out))
        vf += label_vf
        cmd = ["ffmpeg", "-y", "-ss", f"{seg['in']:.3f}"] + dec + \
              ["-i", clip_path, "-t", f"{raw_len:.3f}",
               "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
               "-map", "0:v", "-map", "1:a",
               "-vf", vf, "-t", f"{dur_out:.3f}", "-r", "30"] + _codec(out_path)
        subprocess.run(cmd, check=True, capture_output=True)

def render_all(timeline, clips):
    seg_files = []
    for n, seg in enumerate(timeline):
        out = os.path.join(config.CACHE_DIR, f"seg_{n:03d}.mp4")
        if not (os.path.exists(out) and os.path.getsize(out) > 50_000):
            render_segment(seg, out, clips)
            print(f"  seg {n:03d} [{seg['type']}] {seg['dur']:.1f}s")
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
        timeline = tl_all["timeline"]
        print("  timeline cached")
    else:
        anchors_by_clip = {}
        for item in clips:
            ap = os.path.join(config.CACHE_DIR, f"anchors_{item['key']}.json")
            anchors_by_clip[item["key"]] = json.load(open(ap))["anchors"]
        timeline, chapter_marks = build_timeline(script, audio_manifest,
                                                 anchors_by_clip, clips)
        json.dump({"timeline": timeline, "chapters": chapter_marks},
                  open(tl_path, "w"), indent=2, ensure_ascii=False)
        tl_all = {"timeline": timeline, "chapters": chapter_marks}
        for s in timeline:
            print(f"  [{s['type']:>13}] t0={s['t0']:>6.1f} dur={s['dur']:>5.1f}"
                  f"  {s.get('note','')[:58]}")
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
