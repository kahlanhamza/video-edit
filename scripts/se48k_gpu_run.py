"""MossFormer2_SE_48K real GPU run — the Bus-1 neural law proven on a free T4.

Instruction-E evidence run: the local sandbox has no torch, so the strongest
open speech extractor (ClearerVoice MossFormer2_SE_48K, Apache-2.0, native
48 kHz) is validated where it is meant to run — KaggleBatchGpuBackend's
batch lane. The kernel:

  1. pip-installs `clearvoice` (repo-clone fallback per the upstream README)
  2. downloads the raw noisy bodycam WAV from the kernel-inputs release
  3. runs MossFormer2_SE_48K (CUDA when available) -> enhanced 48 kHz WAV
  4. runs the SAME Silero VAD + SNR math as audio_engine.vad/snr (mirrored
     inline — identical chunking, threshold, merge gap, median floor) on
     RAW and CLEAN over the IDENTICAL speech spans, then evaluates the
     production gates: clean_snr >= 14 dB AND (delta_noise <= -8 dB OR
     snr improvement >= 6 dB), zero clipping
  5. writes enhanced_48k_mossformer2.wav + snr_metrics.json to /kaggle/working

Phases (idempotent + resumable — the sandbox kills processes between calls):
  prepare   pick the densest-speech 60 s window (repo VAD + best_speech_window)
            and slice it to 48 kHz mono WAV
  upload    attach the WAV to the `kernel-inputs` release (replace-if-exists)
  push      push the kernel (server-side run starts)
  poll      print kernel status (exit 3 while still running — re-invoke me)
  harvest   download enhanced WAV + metrics, cross-check SNR host-side with
            the REAL audio_engine.snr gate code, export the 15 s A/B pair,
            write evidence JSON into download/

Usage: uv run scripts/se48k_gpu_run.py <phase>
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "kahlanhamza/video-edit"
RELEASE_TAG = "kernel-inputs"
ASSET_NAME = "se48k_raw_60s.wav"
VIDEO = Path("/home/z/my-project/download/eustis_bodycam_master_720p.mp4")
WORK_DIR = ROOT / "output" / "se48k_run"
RAW_WAV = WORK_DIR / ASSET_NAME
WINDOW_REF = WORK_DIR / "window.json"
HARVEST_DIR = WORK_DIR / "harvest"
EVIDENCE = Path("/home/z/my-project/download/m4_evidence_se48k_gpu.json")
AB_DIR = Path("/home/z/my-project/download/ab_se48k")
SLUG = "directoros-se48k-eustis-a01"
# Kaggle derives the real kernel slug from the kernel TITLE for new kernels —
# push persists the actual ref here so poll/harvest survive process restarts.
REF_FILE = WORK_DIR / "kernel_ref.txt"
TERMINAL = {"complete", "error", "cancelAcknowledged", "cancelRequested"}
WINDOW_S = 60.0

# The kernel mirrors audio_engine.vad (Silero v5 ONNX, 512-sample chunks @16k,
# threshold 0.5, 0.12 s merge gap) and audio_engine.snr (100 ms frames, median
# floor, same gate thresholds). Kept token-replaced instead of str.format so
# the dict literals do not need brace escaping.
SE48K_KERNEL_TEMPLATE = """# DirectorOS SE48K kernel — MossFormer2_SE_48K on free T4
import json, subprocess, sys, urllib.request

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "clearvoice", "onnxruntime"],
               check=True)
import numpy as np
import torch
import onnxruntime as ort

RAW_URL = "@@RAW_URL@@"
VAD_URL = "@@VAD_URL@@"
RAW = "/kaggle/working/raw_48k.wav"
CLEAN = "/kaggle/working/enhanced_48k_mossformer2.wav"
VAD_ONNX = "/kaggle/working/silero_vad.onnx"

urllib.request.urlretrieve(RAW_URL, RAW)
urllib.request.urlretrieve(VAD_URL, VAD_ONNX)

try:  # PyPI wheel per upstream README ("pip install clearvoice")
    from clearvoice import ClearVoice
except ImportError:  # repo fallback
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/modelscope/ClearerVoice-Studio.git",
         "/kaggle/working/ClearerVoice-Studio"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "-r", "/kaggle/working/ClearerVoice-Studio/requirements.txt"],
        check=True,
    )
    sys.path.insert(0, "/kaggle/working/ClearerVoice-Studio")
    from clearvoice import ClearVoice

use_cuda = bool(torch.cuda.is_available())
try:
    voicer = ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"],
                        use_cuda=use_cuda, gpu_id=0)
except TypeError:
    voicer = ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])
output = voicer(input_path=RAW, online_write=False)
voicer.write(output, output_path=CLEAN)
print("enhanced written:", use_cuda and "cuda" or "cpu")


# ---- PCM reading (wave stdlib — mirrors leveling.decode_pcm_mono math) ----
import wave

def read_pcm_mono(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    assert sw == 2, f"expected 16-bit WAV, sampwidth={sw}"
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1).astype(np.float32)
    return x, sr


raw_pcm, sr_raw = read_pcm_mono(RAW)
clean_pcm, sr_clean = read_pcm_mono(CLEAN)
print("decoded raw", raw_pcm.size, sr_raw, "| clean", clean_pcm.size, sr_clean)


# ---- 16 kHz for Silero VAD (torchaudio -> librosa fallback) ----
def resample_to_16k(x, sr):
    if sr == 16000:
        return x.astype(np.float32)
    try:
        import torchaudio

        t = torch.from_numpy(x.astype(np.float32))[None]
        return torchaudio.functional.resample(t, sr, 16000)[0].numpy().astype(np.float32)
    except Exception:
        import librosa

        return librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=16000)


# ---- Silero VAD v5 ONNX — identical semantics to audio_engine.vad ----
# (official streaming protocol: 64-sample rolling context + 512-sample chunk)
opts = ort.SessionOptions()
opts.intra_op_num_threads = 1
opts.inter_op_num_threads = 1
sess = ort.InferenceSession(VAD_ONNX, sess_options=opts)
CHUNK = 512
CONTEXT = 64
VAD_SR = 16000


def merge_spans(spans, min_gap=0.12):
    spans = sorted((s, e) for s, e in spans if e > s)
    if not spans:
        return []
    out = [list(spans[0])]
    for s, e in spans[1:]:
        if s - out[-1][1] < min_gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def speech_regions(x16, threshold=0.5):
    state = np.zeros((2, 1, 128), np.float32)
    context = np.zeros((1, CONTEXT), np.float32)
    spans = []
    open_s = None
    total = (x16.size // CHUNK) * CHUNK
    for i in range(0, total, CHUNK):
        window = np.concatenate([context, x16[i:i + CHUNK][None, :]], axis=1)
        prob, state = sess.run(
            None,
            {"input": window, "state": state, "sr": np.array(VAD_SR, np.int64)},
        )
        context = window[:, -CONTEXT:]
        p = float(prob[0][0])
        t0, t1 = i / VAD_SR, (i + CHUNK) / VAD_SR
        if p >= threshold and open_s is None:
            open_s = t0
        elif p < threshold and open_s is not None:
            spans.append((open_s, t1))
            open_s = None
    if open_s is not None:
        spans.append((open_s, total / VAD_SR))
    return merge_spans(spans)


# ---- SNR math — identical definitions to audio_engine.snr ----
def dbfs_rms(x):
    if x.size == 0:
        return -120.0
    r = float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))
    return max(-120.0, 20.0 * np.log10(r + 1e-12))


def speech_rms_db(x, sr, spans, pad=0.10):
    m = merge_spans([(max(0.0, s - pad), e + pad) for s, e in spans], 0.05)
    parts = []
    for s, e in m:
        a, b = int(s * sr), min(int(e * sr), x.size)
        if b > a:
            parts.append(x[a:b])
    return dbfs_rms(np.concatenate(parts)) if parts else -120.0


def noise_floor_db(x, sr, spans, frame_s=0.10):
    fn = max(1, int(frame_s * sr))
    nf = x.size // fn
    if nf == 0:
        return dbfs_rms(x)
    frames = x[: nf * fn].reshape(nf, fn).astype(np.float64)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    t0 = np.arange(nf) * frame_s
    t1 = t0 + frame_s
    loud = np.zeros(nf, bool)
    for s, e in merge_spans(spans, 0.05):
        loud |= (t1 > s) & (t0 < e)
    quiet = rms[~loud]
    if quiet.size == 0:
        return dbfs_rms(x)
    return max(-120.0, 20.0 * np.log10(float(np.median(quiet)) + 1e-12))


raw16 = resample_to_16k(raw_pcm, sr_raw)
spans = speech_regions(raw16)  # gate truth from RAW — SAME spans for both sides
print("speech spans:", len(spans), "speech_s:",
      round(sum(e - s for s, e in spans), 2))

raw_speech = speech_rms_db(raw_pcm, sr_raw, spans)
clean_speech = speech_rms_db(clean_pcm, sr_clean, spans)
raw_floor = noise_floor_db(raw_pcm, sr_raw, spans)
clean_floor = noise_floor_db(clean_pcm, sr_clean, spans)
raw_snr = raw_speech - raw_floor
clean_snr = clean_speech - clean_floor
delta_noise = clean_floor - raw_floor
improvement = clean_snr - raw_snr
peak_clean_dbfs = max(-120.0, 20.0 * np.log10(float(np.max(np.abs(clean_pcm))) + 1e-12))

gate_clean = clean_snr >= 14.0
gate_reduction = (delta_noise <= -8.0) or (improvement >= 6.0)
gate_clip = peak_clean_dbfs <= -1.0

metrics = {
    "engine": "mossformer2_se_48k",
    "device": "cuda" if use_cuda else "cpu",
    "window_s": [float(min(spans[0])) if spans else 0.0, float(raw_pcm.size / sr_raw)],
    "speech_spans_s": [[round(s, 3), round(e, 3)] for s, e in spans],
    "speech_seconds": round(sum(e - s for s, e in spans), 2),
    "duration_raw_s": round(raw_pcm.size / sr_raw, 3),
    "duration_clean_s": round(clean_pcm.size / sr_clean, 3),
    "raw_speech_rms_db": round(raw_speech, 2),
    "clean_speech_rms_db": round(clean_speech, 2),
    "raw_noise_floor_db": round(raw_floor, 2),
    "clean_noise_floor_db": round(clean_floor, 2),
    "raw_snr_db": round(raw_snr, 2),
    "clean_snr_db": round(clean_snr, 2),
    "delta_noise_db": round(delta_noise, 2),
    "improvement_db": round(improvement, 2),
    "clean_peak_dbfs": round(peak_clean_dbfs, 2),
    "gates": {
        "clean_snr_ge_14": bool(gate_clean),
        "noise_reduction": bool(gate_reduction),
        "zero_clipping": bool(gate_clip),
    },
    "all_gates_pass": bool(gate_clean and gate_reduction and gate_clip),
}
json.dump(metrics, open("/kaggle/working/snr_metrics.json", "w"), indent=2)
print(json.dumps(metrics, indent=2))
print("DIRECTOROS_RESULT=" + json.dumps({
    "engine": "mossformer2_se_48k",
    "device": metrics["device"],
    "raw_snr_db": metrics["raw_snr_db"],
    "clean_snr_db": metrics["clean_snr_db"],
    "delta_noise_db": metrics["delta_noise_db"],
    "improvement_db": metrics["improvement_db"],
    "all_gates_pass": metrics["all_gates_pass"],
}))
"""


def _load_env() -> None:
    for line in (ROOT / ".env").read_text().splitlines():
        name, sep, val = line.partition("=")
        if sep and name.strip() and not name.strip().startswith("#"):
            os.environ.setdefault(name.strip(), val.strip())


def _gh() -> tuple[str, str]:
    token = os.environ.get("GH_TOKEN") or _read_gh_token()
    return token, f"https://api.github.com/repos/{REPO}"


def _read_gh_token() -> str:
    url = subprocess.run(
        ["git", "remote", "get-url", "origin"], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()
    return url.removeprefix("https://").split("@")[0]


def _client():
    from director_os.core.kaggle import KaggleClient

    return KaggleClient(username=os.environ["KAGGLE_USERNAME"], key=os.environ["KAGGLE_KEY"])


def _kernel_slug() -> str:
    return REF_FILE.read_text().strip() if REF_FILE.exists() else SLUG


# ----------------------------------------------------------------- prepare
def phase_prepare() -> None:
    """Densest-speech 60 s window (repo VAD + ab_export logic) -> 48k mono WAV."""

    from director_os.audio_engine.ab_export import best_speech_window
    from director_os.audio_engine.leveling import decode_pcm_mono
    from director_os.audio_engine.vad import SileroVad

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    pcm16 = decode_pcm_mono(str(VIDEO), 16_000)
    vad = SileroVad()
    spans = vad.speech_regions(pcm16)
    total_s = pcm16.size / 16_000
    st, end = best_speech_window(spans, total_s, window_s=WINDOW_S)
    speech_s = sum(e - s for s, e in spans if e > st and s < end)
    print(
        f"full 244s audio: {len(spans)} VAD spans, speech-densest window "
        f"{st:.1f}s..{end:.1f}s (speech {speech_s:.1f}s)"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{st:.3f}",
            "-t",
            f"{end - st:.3f}",
            "-i",
            str(VIDEO),
            "-vn",
            "-map",
            "0:a:0",
            "-ar",
            "48000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(RAW_WAV),
        ],
        check=True,
    )
    WINDOW_REF.write_text(json.dumps({"start_s": round(st, 3), "end_s": round(end, 3)}))
    print(f"raw slice: {RAW_WAV} size={RAW_WAV.stat().st_size / 1e6:.1f}MB")


# ------------------------------------------------------------------ upload
def phase_upload() -> None:
    import httpx

    token, api = _gh()
    with httpx.Client(headers={"Authorization": f"token {token}"}, timeout=180.0) as c:
        r = c.get(f"{api}/releases/tags/{RELEASE_TAG}")
        if r.status_code == 404:
            r = c.post(
                f"{api}/releases",
                json={
                    "tag_name": RELEASE_TAG,
                    "name": "Kernel input media (ephemeral)",
                    "body": (
                        "Media referenced by DirectorOS cloud kernels "
                        "(kernel-reachable HTTPS contract). Safe to prune; "
                        "the runner re-uploads."
                    ),
                },
            )
            r.raise_for_status()
        rel = r.json()
        rel_id = rel["id"]
        for a in rel.get("assets", []):
            if a["name"] == ASSET_NAME:
                c.delete(f"{api}/releases/assets/{a['id']}").raise_for_status()
        data = RAW_WAV.read_bytes()
        up = httpx.post(
            f"https://uploads.github.com/repos/{REPO}/releases/{rel_id}/assets",
            params={"name": ASSET_NAME},
            headers={"Authorization": f"token {token}", "Content-Type": "audio/wav"},
            content=data,
            timeout=300.0,
        )
        up.raise_for_status()
        asset = up.json()
        print("uploaded:", asset["browser_download_url"], f"{asset['size'] / 1e6:.1f}MB")


# -------------------------------------------------------------------- push
def phase_push() -> None:
    _load_env()
    raw_url = f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}/{ASSET_NAME}"
    # the VAD weights ride in the repo (same bundling law as audio_engine) —
    # the kernel fetches them from the raw main branch (public repo).
    vad_url = (
        "https://raw.githubusercontent.com/kahlanhamza/video-edit/main/"
        "packages/audio_engine/src/director_os/audio_engine/models/silero_vad.onnx"
    )
    script = SE48K_KERNEL_TEMPLATE.replace("@@RAW_URL@@", raw_url).replace("@@VAD_URL@@", vad_url)
    client = _client()
    push_slug = _kernel_slug()
    pushed = client.push_kernel(push_slug, "DirectorOS MossFormer2 SE (Eustis evidence)", script)
    ref = str(pushed.get("ref", "")).removeprefix("/code/") or SLUG
    REF_FILE.parent.mkdir(parents=True, exist_ok=True)
    REF_FILE.write_text(ref)
    print("pushed:", pushed.get("ref"), "| url:", pushed.get("url"))


# -------------------------------------------------------------------- poll
def phase_poll(budget_s: float = 480.0, interval_s: float = 20.0) -> None:
    _load_env()
    client = _client()
    slug = _kernel_slug()
    deadline = time.monotonic() + budget_s
    status = "unknown"
    while time.monotonic() < deadline:
        status = client.kernel_status(slug)
        print(f"[{time.strftime('%H:%M:%S')}] status={status}", flush=True)
        if status in TERMINAL:
            break
        time.sleep(interval_s)
    if status not in TERMINAL:
        print("still running — re-invoke: uv run scripts/se48k_gpu_run.py poll")
        sys.exit(3)
    if status == "error":
        out = client.kernel_output(slug)
        tail = "".join(str(rec.get("data", "")) for rec in out["log"])[-2000:]
        print("KERNEL FAILED:\n", tail)
        sys.exit(4)
    print("complete — harvest: uv run scripts/se48k_gpu_run.py harvest")


# -------------------------------------------------------------------- log
def phase_log(max_lines: int = 120) -> None:
    _load_env()
    client = _client()
    slug = _kernel_slug()
    out = client.kernel_output(slug)
    text = "".join(str(rec.get("data", "")) for rec in out["log"])
    lines = text.splitlines()
    print(f"kernel={slug} log_lines={len(lines)} files={out['files']}")
    for line in lines[-max_lines:]:
        print(line[:200])


# ----------------------------------------------------------------- harvest
def phase_harvest() -> None:
    _load_env()
    from director_os.audio_engine.ab_export import export_ab_pair
    from director_os.audio_engine.snr import measure_snr, snr_gate

    client = _client()
    slug = _kernel_slug()
    out = client.kernel_output(slug)
    names = [e["name"] for e in out["files"]]
    for need in ("enhanced_48k_mossformer2.wav", "snr_metrics.json"):
        if need not in names:
            raise SystemExit(f"kernel output missing {need}; got {names}")
    clean = client.download_output_file(slug, "enhanced_48k_mossformer2.wav", HARVEST_DIR / "audio")
    kmetrics = json.loads(
        client.download_output_file(slug, "snr_metrics.json", HARVEST_DIR / "metrics").read_text()
    )
    win = json.loads(WINDOW_REF.read_text())
    raw_ab = export_ab_pair(
        RAW_WAV,
        Path(clean),
        AB_DIR,
        speech_intervals=[tuple(s) for s in kmetrics["speech_spans_s"]],
        duration_s=kmetrics["duration_clean_s"],
    )

    # host-side cross-check with the REAL production gate code: the kernel's
    # spans are window-relative — shift into this slice's local timeline (the
    # slice IS the window, so they apply as-is) and re-measure both files.
    spans = [tuple(s) for s in kmetrics["speech_spans_s"]]
    raw_m = measure_snr(str(RAW_WAV), spans)
    clean_m = measure_snr(str(clean), spans)
    ok, detail = snr_gate(
        raw_m["snr_db"], clean_m["snr_db"], clean_m["noise_floor_db"] - raw_m["noise_floor_db"]
    )

    evidence = {
        "kernel": slug,
        "backend": "kaggle-gpu (free T4)",
        "model": "ClearerVoice MossFormer2_SE_48K (Apache-2.0, native 48 kHz)",
        "device": kmetrics["device"],
        "window": win,
        "audio_url": f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}/{ASSET_NAME}",
        "enhanced_path": str(clean),
        "ab_raw_15s": str(raw_ab.raw_path),
        "ab_clean_15s": str(raw_ab.clean_path),
        "kernel_metrics": kmetrics,
        "host_crosscheck": {
            "raw": {k: round(v, 2) for k, v in raw_m.items()},
            "clean": {k: round(v, 2) for k, v in clean_m.items()},
            "gate_ok": ok,
            "gate_detail": detail,
        },
        "kernel_log_tail": "".join(str(rec.get("data", "")) for rec in out["log"])[-400:],
    }
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2))
    if not ok:
        raise SystemExit("PRODUCTION SNR GATE FAILED — see gate_detail above")
    print("OK — production SNR gates passed, evidence written:", EVIDENCE)


PHASES = {
    "prepare": phase_prepare,
    "upload": phase_upload,
    "push": phase_push,
    "poll": phase_poll,
    "log": phase_log,
    "harvest": phase_harvest,
}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in PHASES:
        raise SystemExit(f"usage: se48k_gpu_run.py {'|'.join(PHASES)}")
    PHASES[sys.argv[1]]()
