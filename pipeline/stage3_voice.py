"""Stage 3 VOICE v2: one consistent voice, measured natural pace, per-unit resume."""
import os, json, wave
from . import config, ai

def _wav_dur(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / w.getframerate()

def run(case_id, script):
    manifest_path = os.path.join(config.AUDIO_DIR, f"case{case_id}_audio_manifest.json")
    if os.path.exists(manifest_path):
        print("  audio cached")
        return json.load(open(manifest_path))
    manifest = {"voice": config.TTS_VOICE, "speed": config.TTS_SPEED, "beats": []}
    units = [("cold_open", script.get("cold_open", ""))]
    units += [(f"beat_{i+1}", b["narration"]) for i, b in enumerate(script.get("beats", []))]
    if script.get("outro"):
        units.append(("outro", script["outro"]))
    words_total, dur_total = 0, 0.0
    for name, text in units:
        if not text or not text.strip():
            continue
        out = os.path.join(config.AUDIO_DIR, f"case{case_id}_{name}.wav")
        if os.path.exists(out) and os.path.getsize(out) > 4000:
            dur = _wav_dur(out)
        else:
            ai.tts(text, out, voice=config.TTS_VOICE, speed=config.TTS_SPEED)
            dur = _wav_dur(out)
        w = len(text.split())
        words_total += w
        dur_total += dur
        manifest["beats"].append({"name": name, "path": out, "dur": dur,
                                  "words": w, "text": text})
        print(f"  [{name}] {dur:.1f}s ({w}w)")
    manifest["wpm_measured"] = round(words_total / dur_total * 60, 1) if dur_total else 0
    json.dump(manifest, open(manifest_path, "w"), indent=2, ensure_ascii=False)
    print(f"  narration total {dur_total:.0f}s | measured {manifest['wpm_measured']} wpm")
    return manifest
