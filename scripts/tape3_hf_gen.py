#!/usr/bin/env python3
"""Generate the HyperFrames project for tape 3 (root + scenes structure)."""

import json
import os

WORK = "/home/z/my-project/work_v2"
PROJ = f"{WORK}/tape3_hf"

plan = json.load(open(f"{WORK}/parts_plan.json"))["plan"]
tl = json.load(open(f"{WORK}/subs_timeline.json"))
caps = json.load(open(f"{WORK}/captions_data.json"))
bounds = tl["bounds"]
total = tl["total_dur"]

STYLE_COLOR = {
    "Officer": "#FFFFFF",
    "Suspect": "#FFD84D",
    "Witness": "#7CFF9E",
    "Narrator": "#70E4FF",
}
LANE_BOTTOM = {0: 69, 1: 144}


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# ---------- index.html ----------
videos = []
for idx, b in enumerate(bounds, 1):
    d = round(b["end"] - b["start"], 3)
    videos.append(
        f'    <video id="p{idx:03d}" class="clip" data-start="{b["start"]}" data-duration="{d}" '
        f'data-track-index="1" playsinline data-has-audio="true" preload="auto" '
        f'src="assets/part_{idx:03d}.mp4" style="position:absolute;inset:0;width:100%;height:100%;'
        f'object-fit:contain;background:#000;z-index:1"></video>'
    )

index_html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=1920, height=1080" />
    <script src="assets/gsap.min.js"></script>
    <style>
      :root {{ --ground: #000; }}
      * {{ margin: 0; padding: 0; box-sizing: border-box; }}
      html, body {{
        margin: 0; width: 1920px; height: 1080px; overflow: hidden;
        background: var(--ground);
      }}
      #root {{ width: 100%; height: 100%; background: #000; }}
    </style>
  </head>
  <body>
    <div
      id="root"
      data-composition-id="main"
      data-start="0"
      data-duration="{total}"
      data-width="1920"
      data-height="1080"
    >
{chr(10).join(videos)}
      <div
        id="wm"
        data-composition-id="wm"
        data-composition-src="compositions/wm.html"
        data-start="0"
        data-duration="{total}"
        data-track-index="2"
        data-width="1920"
        data-height="1080"
        style="position:absolute;inset:0;width:100%;height:100%;z-index:2"
      ></div>
      <div
        id="caps"
        data-composition-id="caps"
        data-composition-src="compositions/caps.html"
        data-start="0"
        data-duration="{total}"
        data-track-index="3"
        data-width="1920"
        data-height="1080"
        style="position:absolute;inset:0;width:100%;height:100%;z-index:3"
      ></div>
    </div>
    <script>
      window.__timelines["main"] = gsap.timeline({{ paused: true }});
    </script>
  </body>
</html>
"""
with open(f"{PROJ}/index.html", "w") as f:
    f.write(index_html)

# ---------- compositions/wm.html ----------
wm_html = """<template>
  <style>
    #root { position: absolute; inset: 0; pointer-events: none; }
    #wm-img { width: 100%; height: 100%; object-fit: contain; display: block; }
  </style>
  <div id="root" data-composition-id="wm" data-width="1920" data-height="1080">
    <img id="wm-img" src="assets/wm.png" alt="Lens and Law" />
  </div>
  <script>
    const tl = gsap.timeline({ paused: true });
    window.__timelines["wm"] = tl;
  </script>
</template>
"""
os.makedirs(f"{PROJ}/compositions", exist_ok=True)
with open(f"{PROJ}/compositions/wm.html", "w") as f:
    f.write(wm_html)

# ---------- compositions/caps.html ----------
groups_html = []
caps_js = []
for gi, c in enumerate(caps["captions"]):
    s = round(max(0.10, c["s"]), 3)
    e = min(c["e"], total - 0.02)
    d = round(e - s, 3)
    if d <= 0.12:
        continue
    color = STYLE_COLOR[c["style"]]
    bottom = LANE_BOTTOM[c["lane"]]
    words_html = []
    words_js = []
    for wi, w in enumerate(c["words"]):
        wid = f"cap{gi:03d}w{wi:02d}"
        words_html.append(f'<span id="{wid}" class="w">{esc(w["w"])}</span>')
        words_js.append(f'["{wid}",{w["rs"]}]')
    inner = f'<span class="capbox" style="color:{color}">' + " ".join(words_html) + "</span>"
    groups_html.append(
        f'      <div id="cap{gi:03d}" class="cap clip" data-start="{s}" data-duration="{d}" '
        f'style="bottom:{bottom}px">{inner}</div>'
    )
    caps_js.append(f'{{id:"cap{gi:03d}",s:{s},d:{d},words:[{",".join(words_js)}]}}')

caps_html = f"""<template>
  <style>
    #root {{ position: absolute; inset: 0; pointer-events: none; overflow: visible; }}
    .cap {{
      position: absolute; left: 0; right: 0;
      display: flex; justify-content: center; align-items: flex-end;
      overflow: visible;
    }}
    .capbox {{
      display: inline-block; max-width: 1560px;
      font-family: Roboto, sans-serif; font-weight: 700; font-size: 54px;
      line-height: 1.22; text-align: center; letter-spacing: 0.2px;
      background: rgba(0, 0, 0, 0.36); border-radius: 10px;
      padding: 10px 26px 12px;
      -webkit-text-stroke: 4px #000;
      paint-order: stroke fill;
      text-shadow: 0 3px 4px rgba(0, 0, 0, 0.55);
      overflow: visible;
    }}
    .capbox .w {{ opacity: 0.5; }}
  </style>
  <div id="root" data-composition-id="caps" data-width="1920" data-height="1080">
{chr(10).join(groups_html)}
  </div>
  <script>
    const CAPS = [{",".join(caps_js)}];
    const tl = gsap.timeline({{ paused: true }});
    CAPS.forEach(function (c) {{
      tl.fromTo("#" + c.id, {{ opacity: 0 }}, {{ opacity: 1, duration: 0.12, ease: "power1.out" }}, c.s);
      c.words.forEach(function (wd) {{
        tl.to("#" + c.id + " #" + wd[0], {{ opacity: 1, duration: 0.08, ease: "none" }}, c.s + wd[1]);
      }});
      tl.to("#" + c.id, {{ opacity: 0, duration: 0.12, ease: "power1.in" }}, Math.max(c.s, c.s + c.d - 0.12));
    }});
    window.__timelines["caps"] = tl;
  </script>
</template>
"""
with open(f"{PROJ}/compositions/caps.html", "w") as f:
    f.write(caps_html)
print(
    f"index.html: {len(videos)} videos; wm.html; caps.html: {len(groups_html)} groups, total {total}s"
)
