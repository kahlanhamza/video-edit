"""SEMANTIC ANCHORS (ADR-0005): find and label the 'moments' in raw footage.

Motion energy (OpenCV frame-diff) + audio RMS (loudness spikes) -> candidate
timestamps; each anchor is labeled by GLM vision so the scriptwriter's beat
cues can be matched to real moments, not bare timestamps.
"""
import os, json, subprocess, wave, tempfile
import numpy as np
from . import config, ai

def probe_dur(path):
    r = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json",
                        "-show_format", path], capture_output=True, text=True)
    return float(json.loads(r.stdout)["format"]["duration"])

def motion_energy(path, clip_dur):
    """Per-second motion score via OpenCV frame differencing (stride-sampled)."""
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(fps // 3))          # ~3 samples/sec
    bins = np.zeros(int(clip_dur) + 2)
    counts = np.zeros(int(clip_dur) + 2)
    prev = None
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % stride == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            small = cv2.resize(frame, (160, 90))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev is not None:
                d = float(np.abs(gray - prev).mean())
                t = idx / fps
                b = min(int(t), len(bins) - 1)
                bins[b] += d
                counts[b] += 1
            prev = gray
        idx += 1
    cap.release()
    counts[counts == 0] = 1
    return (bins / counts).tolist()

def audio_rms(path, clip_dur):
    """Per-second loudness via PCM extraction + numpy RMS."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    subprocess.run(["ffmpeg", "-y", "-i", path, "-vn", "-ac", "1", "-ar", "16000",
                    tmp_path], check=True, capture_output=True)
    with wave.open(tmp_path, "rb") as w:
        n = w.getnframes()
        data = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32)
    os.remove(tmp_path)
    rate = 16000
    per = rate
    nbins = int(clip_dur) + 2
    rms = np.zeros(nbins)
    for b in range(nbins):
        seg = data[b * per:(b + 1) * per]
        if len(seg):
            rms[b] = float(np.sqrt(np.mean(seg ** 2)) / 32768.0)
    return rms.tolist()

def _normalize(xs):
    xs = np.array(xs, dtype=np.float32)
    if xs.max() > 0:
        xs = (xs - xs.min()) / (xs.max() - xs.min() + 1e-6)
    return xs

def find_anchors(path, topk=12, spacing=10.0, dur=None):
    dur = dur or probe_dur(path)
    motion = _normalize(motion_energy(path, dur))
    audio = _normalize(audio_rms(path, dur))
    score = 0.55 * motion + 0.45 * audio
    cand = []
    for t in range(2, int(dur) - 3):
        win = score[max(0, t - 3):t + 4]
        if score[t] >= win.max() and score[t] > 0.35:
            cand.append((float(t), float(score[t])))
    anchors, last_t = [], -999
    for t, s in sorted(cand, key=lambda x: -x[1]):
        if all(abs(t - a["t"]) >= spacing for a in anchors):
            anchors.append({"t": t, "score": round(s, 3),
                            "motion": round(float(motion[int(t)]), 3),
                            "audio": round(float(audio[int(t)]), 3)})
        if len(anchors) >= topk:
            break
    anchors.sort(key=lambda a: a["t"])
    return anchors, dur

def label_anchor(path, t, dur):
    ts = max(0.5, min(t - 0.8, dur - 1.0))
    f = os.path.join(config.CACHE_DIR,
                     f"anchor_{os.path.basename(path)[:24]}_{int(t)}.jpg")
    if not os.path.exists(f):
        subprocess.run(["ffmpeg", "-y", "-ss", f"{ts:.1f}", "-i", path,
                        "-frames:v", "1", "-vf", "scale=960:-1", "-q:v", "3", f],
                       capture_output=True)
    ans = ai.vision(
        "This is a frame from raw police bodycam/dashcam footage. Reply in JSON only: "
        '{"label": "<=10 words: who is doing what (e.g. \'officer walks toward driver window\')", '
        '"intensity": "calm|tense|dramatic", "people": "one|two|several"} '
        "Describe only what is visible. No speculation.", [f])
    try:
        j = json.loads(ans[ans.find("{"):ans.rfind("}") + 1])
    except Exception:
        j = {"label": ans[:80], "intensity": "tense", "people": "several"}
    return j

def analyze_clip(key, path, topk=12):
    out = os.path.join(config.CACHE_DIR, f"anchors_{key}.json")
    if os.path.exists(out):
        print(f"  [cached] {key}")
        return json.load(open(out))
    dur = probe_dur(path)
    anchors, dur = find_anchors(path, topk=topk, dur=dur)
    print(f"  {key}: {len(anchors)} anchors from {dur:.0f}s")
    for a in anchors:
        lab = label_anchor(path, a["t"], dur)
        a["label"] = lab.get("label", "")
        a["intensity"] = lab.get("intensity", "tense")
        print(f"    t={a['t']:>5.0f}s score={a['score']:.2f} [{a['intensity']}] {a['label']}")
    json.dump({"key": key, "dur": dur, "anchors": anchors}, open(out, "w"),
              indent=2, ensure_ascii=False)
    return json.load(open(out))

def run():
    results = []
    for item in config.USED_POOL:
        path = os.path.join(config.FOOTAGE_DIR, item["key"] + ".webm")
        results.append(analyze_clip(item["key"], path,
                                    topk=10 if item["role"] != "act3" else 8))
    return results

if __name__ == "__main__":
    run()
