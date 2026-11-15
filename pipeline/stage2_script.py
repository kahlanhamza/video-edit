"""Stage 2 SCRIPTWRITER v2: conversational narration, NO timestamp-speak,
per-beat mode + semantic cue matched to VLM anchor labels (ADR-0005).
"""
import os, json, re
from . import config, ai

SYSTEM = """You are the head writer of a documentary bodycam channel ("REGRET
ON CAMERA"). You write like a veteran true-crime podcaster talking to one
viewer: conversational, contractions, present tense, short sentences, dry wit,
zero melodrama. You never editorialize about guilt or innocence, never
dehumanize anyone, and stick to what the footage shows. Tension comes from
facts and the sequence of decisions, not from adjectives. You NEVER mention
timestamps, timecodes, durations or "seconds"/"minutes" of any kind. You write
for adults; nothing gratuitous."""

TIMESTAMP_RE = [re.compile(p, re.IGNORECASE) for p in config.TIMESTAMP_PATTERNS]

def _sanitize(text):
    low = text.lower()
    for phrase in config.FORBIDDEN_PHRASES:
        if phrase in low:
            rep = config.SAFE_REPLACEMENTS.get(phrase, "the individual")
            text = re.sub(re.escape(phrase), rep, text, flags=re.IGNORECASE)
            low = text.lower()
    return text

def _strip_timestamps(text):
    for rx in TIMESTAMP_RE:
        text = rx.sub("moments later", text)
    return re.sub(r"\s{2,}", " ", text)

def _anchor_menu(anchors_by_clip):
    lines = []
    for ci, item in enumerate(config.USED_POOL, 1):
        key = item["key"]
        lines.append(f"CLIP {ci} = {item['agency']} ({item['location']}): "
                     f"{item['summary']} The irony: {item['regret']}")
        for a in anchors_by_clip.get(key, []):
            lines.append(f"  - moment @ {a['t']}s [{a.get('intensity','tense')}]: "
                         f"{a.get('label','?')}")
    return "\n".join(lines)

def build_beats(target_sec, anchors_by_clip):
    # ~50% narration coverage at measured 140 wpm -> target_sec * 1.15 words
    n_words = int(target_sec * 1.15)
    prompt = f"""Write the narration script for a {target_sec//60}-minute raw-bodycam commentary video.

STRUCTURE (JSON only):
1. "title": video title, max 85 chars, curiosity-driven but factual
2. "cold_open": 2-3 punchy sentences narrated OVER the single most dramatic
   moment (a replay). Name what we're about to see, tease the irony.
3. "beats": 12-16 beats in act order: beats 1-4 over CLIP 1, beats 5-9 over
   CLIP 2, beats 10-14 over CLIP 3 (adjust to what the moments allow).
   Each beat:
   - "narration": 45-65 words, conversational documentary voice
   - "mode": "replay" (default; slow-mo over the moment you describe) or
     "flashback" (referencing something the viewer already saw)
   - "cue": <=10 words matching ONE listed moment (e.g. "chief fails
     sobriety test", "driver refuses to exit", "handcuffs go on")
   - "chapter": act title for the first beat of each act, else ""
4. "outro": 3-4 sentences: recap the pattern + remind that everyone shown is
   presumed innocent until proven guilty.
5. "chapters": array of {{"clip": 1|2|3, "title": "..."}} (2-4 entries)

HARD RULES:
- Total narration {n_words} words (+/-10%). Count before answering.
- NEVER reference timestamps, timecodes, seconds or minutes anywhere. Not
  "at 180 seconds", not "for the next five minutes", not "at the 1:23 mark".
- NEVER say "clip 1/2/3" in narration - the viewer just sees footage.
- NEVER use: {', '.join(config.FORBIDDEN_PHRASES)}. Say "the driver", "the
  chief", "the man", "the individual". Charges are allegations only.
- Each beat's cue MUST copy one of the listed moment labels (you may shorten).
  Do not invent moments that are not listed.

AVAILABLE MOMENTS (from frame analysis - these are REAL, use only these):
{_anchor_menu(anchors_by_clip)}"""
    return ai.chat_json(prompt, SYSTEM)

def apply_guardrails(script):
    script["title"] = _strip_timestamps(_sanitize(script.get("title", "")))
    for k in ("cold_open", "outro"):
        if script.get(k):
            script[k] = _strip_timestamps(_sanitize(script[k]))
    for b in script.get("beats", []):
        b["narration"] = _strip_timestamps(_sanitize(b.get("narration", "")))
        b["cue"] = _sanitize(b.get("cue", "") or "")
        b["mode"] = b.get("mode", "replay")
        b["chapter"] = _sanitize(b.get("chapter", "") or "")
    return script

def run(case_id, anchors_by_clip, target_sec):
    out = os.path.join(config.PACKAGE_DIR, f"case{case_id}_script.json")
    if os.path.exists(out):
        print("  script cached")
        return json.load(open(out))
    script = apply_guardrails(build_beats(target_sec, anchors_by_clip))
    script["_target_sec"] = target_sec
    script["_pool"] = [{k: ci[k] for k in ("key", "agency", "location", "date",
                                           "summary", "regret", "role")}
                       for ci in config.USED_POOL]
    json.dump(script, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"  script saved ({len(script.get('beats', []))} beats)")
    return script
