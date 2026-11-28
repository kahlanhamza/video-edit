#!/usr/bin/env python3
"""Full decode validation of packed assets; prints problem list."""
import subprocess, json, sys

A = "/home/z/my-project/work_v2/tape3_hf/assets"
bad = []
for i in range(1, 33):
    p = f"{A}/part_{i:03d}.mp4"
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-f", "null", "-"],
                       capture_output=True, text=True)
    errs = [l for l in r.stderr.splitlines() if any(k in l.lower() for k in ("error", "corrupt", "invalid"))]
    r2 = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_entries",
                         "stream=nb_read_frames", "-of", "default=noprint_wrappers=1:nokey=1",
                         "-select_streams", "v", p], capture_output=True, text=True)
    nf = r2.stdout.strip()
    r3 = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                         "-of", "default=noprint_wrappers=1:nokey=1", p], capture_output=True, text=True)
    dur = float(r3.stdout.strip() or 0)
    exp = round(dur * 30)
    flag = bool(errs) or (not nf) or abs(int(nf or 0) - exp) > 3
    print(i, "OK" if not flag else f"BAD dur={dur:.2f} frames={nf} exp={exp} errs={errs[:1]}")
    if flag: bad.append(i)
print("PROBLEM:", bad)
