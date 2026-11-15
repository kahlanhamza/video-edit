"""AI bridge over the z-ai CLI: GLM chat + vision + TTS. 429-aware."""
import os, re, json, subprocess, shutil, time
from . import config

def chat(prompt, system=None, retries=3, thinking=False):
    cmd = ["z-ai", "chat", "-p", prompt]
    if system:
        cmd += ["-s", system]
    if thinking:
        cmd.append("-t")
    out_path = os.path.join(config.CACHE_DIR, "ai_response.json")
    last_err = None
    for attempt in range(retries):
        try:
            subprocess.run(cmd + ["-o", out_path], check=True, capture_output=True,
                           timeout=300)
            with open(out_path) as f:
                data = json.load(f)
            content = data["choices"][0]["message"]["content"]
            if content and content.strip():
                return content
            last_err = "empty content"
        except Exception as e:
            last_err = str(e)
        time.sleep(15 * (attempt + 1))
    raise RuntimeError(f"z-ai chat failed after {retries} tries: {last_err}")

def vision(prompt, images, retries=3):
    cmd = ["z-ai", "vision", "-p", prompt]
    for im in images:
        cmd += ["-i", im]
    out_path = os.path.join(config.CACHE_DIR, "vision_resp.json")
    last_err = None
    for attempt in range(retries):
        try:
            subprocess.run(cmd + ["-o", out_path], check=True, capture_output=True,
                           timeout=180)
            with open(out_path) as f:
                data = json.load(f)
            content = data["choices"][0]["message"]["content"]
            if content and content.strip():
                return content
            last_err = "empty"
        except Exception as e:
            last_err = str(e)
        time.sleep(15 * (attempt + 1))
    raise RuntimeError(f"z-ai vision failed: {last_err}")

def extract_json(text):
    text = re.sub(r"```(?:json)?", "", text)
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    frag = text[start:i + 1]
                    try:
                        return json.loads(frag)
                    except json.JSONDecodeError:
                        try:
                            return json.loads(frag.replace(",}", "}").replace(",]", "]"))
                        except json.JSONDecodeError:
                            break
    raise ValueError("no valid JSON found in: " + text[:200])

def chat_json(prompt, system=None, retries=3):
    return extract_json(chat(prompt, system=system, retries=retries))

# --------------------------------------------------------------- TTS
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
MAX_TTS_CHARS = 900

def _split_chunks(text, limit=MAX_TTS_CHARS):
    sentences = _SENT_SPLIT.split(text.strip())
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) + 1 <= limit:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return chunks

def tts(text, out_path, voice=None, speed=None):
    """Synthesize narration WAV; chunked + ffmpeg-concatenated. One voice, one speed.
    Rate-limit aware: backoff on 429 (observed in production)."""
    voice = voice or config.TTS_VOICE
    speed = speed or config.TTS_SPEED
    chunks = _split_chunks(text)
    parts = []
    for n, chunk in enumerate(chunks):
        part = f"{out_path}.part{n}.wav"
        ok = False
        for attempt in range(6):
            try:
                subprocess.run(["z-ai", "tts", "-i", chunk, "-o", part,
                                "-v", voice, "-s", str(speed), "-f", "wav"],
                               check=True, capture_output=True, timeout=180)
                if os.path.getsize(part) > 4000:
                    ok = True
                    break
            except Exception:
                pass
            time.sleep(20 * (attempt + 1))  # 429 backoff
        if not ok:
            raise RuntimeError(f"TTS chunk {n} failed after backoff")
        parts.append(part)
        time.sleep(2.0)
    if len(parts) == 1:
        shutil.move(parts[0], out_path)
    else:
        lst = out_path + ".list.txt"
        with open(lst, "w") as f:
            for p in parts:
                f.write(f"file '{p}'\n")
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                        "-ar", "24000", "-ac", "1", out_path],
                       check=True, capture_output=True)
        os.remove(lst)
        for p in parts:
            os.remove(p)
    return out_path
