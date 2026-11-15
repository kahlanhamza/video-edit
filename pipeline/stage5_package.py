"""Stage 5 PACKAGING: fit-checked thumbnail + metadata (chapters from timeline)."""
import os, json
from PIL import Image, ImageDraw, ImageFont
from . import config, stage4_editor

def make_thumbnail(script, out_png):
    W, H = 1280, 720
    img = Image.new("RGB", (W, H), (10, 10, 12))
    d = ImageDraw.Draw(img)
    f_mid = ImageFont.truetype(stage4_editor.FONT_BOLD, 56)
    f_badge = ImageFont.truetype(stage4_editor.FONT_BOLD, 34)
    d.rectangle([0, 0, 26, H], fill=(200, 30, 30))
    d.text((80, 60), "BODYCAM", font=f_badge, fill=(200, 30, 30))
    d.text((80, 110), "PUBLIC RECORD", font=f_badge, fill=(120, 120, 120))
    title = script.get("title", "INSTANT REGRET").upper()
    words, lines, cur = title.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f_mid) > 1080:
            lines.append(cur); cur = w
        else:
            cur = t
    lines.append(cur)
    y = 240
    for ln in lines[:3]:
        d.text((80, y), ln, font=f_mid, fill=(240, 240, 240)); y += 84
    hook = "HE DIDN'T PLAN FOR THIS"
    size = 92
    while size > 30 and d.textlength(hook, font=ImageFont.truetype(
            stage4_editor.FONT_BOLD, size)) > 1120:
        size -= 6
    f_big = ImageFont.truetype(stage4_editor.FONT_BOLD, size)
    bw = d.textlength(hook, font=f_big)
    d.text(((W - bw) / 2, H - 140), hook, font=f_big, fill=(200, 30, 30))
    img.save(out_png)

def _fmt_ts(sec):
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

def build_metadata(case_id, script, chapters, timeline, video_path, thumb_path):
    ci = script.get("_pool", [])
    desc = [script.get("title", ""), "", script.get("cold_open", ""), "", "CHAPTERS"]
    for c in chapters:
        label = {1: "ACT 1", 2: "ACT 2", 3: "ACT 3"}.get(c["clip"], "ACT")
        desc.append(f"{_fmt_ts(c['t_video'])} {label}")
    desc.append("")
    desc.append("SOURCE / ATTRIBUTION")
    for i, c in enumerate(ci, 1):
        desc.append(f"Act {i}: {c['agency']} - {c['location']} ({c['date']}). "
                    f"Public record / public domain. Raw footage, no narration.")
    desc.append("")
    desc.append("DISCLAIMER")
    desc.append(config.MANDATORY_DISCLAIMER)
    desc.append("")
    desc.append(f"{config.CHANNEL_NAME} {config.CHANNEL_HANDLE} - "
                "documentary commentary on public-record bodycam footage.")
    metadata = {
        "title": script.get("title", "")[:100],
        "description": "\n".join(desc),
        "tags": config.DEFAULT_TAGS + ["public record", "police pursuit"],
        "category_id": config.YOUTUBE_CATEGORY_ID,
        "made_for_kids": False,
        "chapters": [{"ts": _fmt_ts(c["t_video"]), "label": c["clip"],
                      "t_video": c["t_video"]} for c in chapters],
        "language": config.LANGUAGE,
    }
    out = os.path.join(config.PACKAGE_DIR, f"case{case_id}_metadata.json")
    json.dump(metadata, open(out, "w"), indent=2, ensure_ascii=False)
    return metadata, out

def run(case_id, script, chapters, timeline, video_path):
    thumb = os.path.join(config.THUMB_DIR, f"case{case_id}_thumb.png")
    make_thumbnail(script, thumb)
    metadata, meta_path = build_metadata(case_id, script, chapters, timeline,
                                         video_path, thumb)
    print(f"  thumbnail: {thumb}")
    print(f"  metadata: {meta_path} ({len(metadata['tags'])} tags, "
          f"{len(metadata['chapters'])} chapters)")
    return thumb, metadata
