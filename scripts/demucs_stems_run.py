"""Demucs stems real run — M4 evidence (issue #12 "strip background music").

Exercises the REAL cloud lane end to end: KaggleBatchGpuBackend's kernel
template pushed through core.kaggle.KaggleClient against a free Tesla T4,
with the audio hosted as a public GitHub release asset (the documented
kernel-reachable URL contract; R2 presigned is the production path).

Phases (each idempotent + resumable — sandbox kills processes between calls):
  prepare   extract 44.1 kHz stereo mp3 from the master video
  upload    attach audio to the `kernel-inputs` release (replace-if-exists)
  push      push the Demucs two-stems kernel (server-side run starts)
  poll      print kernel status (exit 3 while still running — re-invoke me)
  harvest   download vocals/no_vocals, verify durations, build StemsResult,
            write evidence JSON into download/

Usage: uv run scripts/demucs_stems_run.py <phase>
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
ASSET_NAME = "eustis_audio_44k.mp3"
VIDEO = Path("/home/z/my-project/download/eustis_bodycam_master_720p.mp4")
AUDIO_OUT = ROOT / "output" / "stems_run" / ASSET_NAME
HARVEST_DIR = ROOT / "output" / "stems_run"
EVIDENCE = Path("/home/z/my-project/download/m4_evidence_demucs_stems.json")
SLUG = "directoros-stems-eustis-a01"
# Kaggle derives the real kernel slug from the kernel TITLE for new kernels —
# push persists the actual ref here so poll/harvest survive process restarts.
REF_FILE = ROOT / "output" / "stems_run" / "kernel_ref.txt"
TERMINAL = {"complete", "error", "cancelAcknowledged", "cancelRequested"}


def _load_env() -> None:
    for line in (ROOT / ".env").read_text().splitlines():
        name, sep, val = line.partition("=")
        if sep and name.strip() and not name.strip().startswith("#"):
            os.environ.setdefault(name.strip(), val.strip())


def _gh() -> tuple[str, str]:
    token = os.environ["GH_TOKEN"] if "GH_TOKEN" in os.environ else _read_gh_token()
    return token, f"https://api.github.com/repos/{REPO}"


def _read_gh_token() -> str:
    url = subprocess.run(
        ["git", "remote", "get-url", "origin"], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()
    return url.removeprefix("https://").split("@")[0]


# ----------------------------------------------------------------- prepare
def phase_prepare() -> None:
    AUDIO_OUT.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(VIDEO),
        "-vn",
        "-ar",
        "44100",
        "-ac",
        "2",
        "-b:a",
        "192k",
        str(AUDIO_OUT),
    ]
    subprocess.run(cmd, check=True)
    dur = _ffprobe_duration(AUDIO_OUT)
    print(f"audio: {AUDIO_OUT} size={AUDIO_OUT.stat().st_size / 1e6:.1f}MB dur={dur:.1f}s")


def _ffprobe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return float(out)


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
        data = AUDIO_OUT.read_bytes()
        up = httpx.post(
            f"https://uploads.github.com/repos/{REPO}/releases/{rel_id}/assets",
            params={"name": ASSET_NAME},
            headers={"Authorization": f"token {token}", "Content-Type": "audio/mpeg"},
            content=data,
            timeout=300.0,
        )
        up.raise_for_status()
        asset = up.json()
        print("uploaded:", asset["browser_download_url"], f"{asset['size'] / 1e6:.1f}MB")


# -------------------------------------------------------------------- push
def phase_push() -> None:
    _load_env()
    from director_os.core.kaggle import KaggleClient
    from director_os.vision_fx.compute.kaggle_batch import DEMUCS_KERNEL_TEMPLATE

    audio_url = f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}/{ASSET_NAME}"
    script = DEMUCS_KERNEL_TEMPLATE.format(audio_url=audio_url, model_name="htdemucs")
    client = KaggleClient(username=os.environ["KAGGLE_USERNAME"], key=os.environ["KAGGLE_KEY"])
    # Re-runs target the EXISTING kernel (Kaggle auto-slugged it from the
    # title) as a new version; first run uses SLUG and persists the real ref.
    push_slug = REF_FILE.read_text().strip() if REF_FILE.exists() else SLUG
    pushed = client.push_kernel(push_slug, "DirectorOS Stems (Eustis evidence run)", script)
    ref = str(pushed.get("ref", "")).removeprefix("/code/") or SLUG
    REF_FILE.parent.mkdir(parents=True, exist_ok=True)
    REF_FILE.write_text(ref)
    print("pushed:", pushed.get("ref"), "| url:", pushed.get("url"))


# -------------------------------------------------------------------- poll
def phase_poll(budget_s: float = 480.0, interval_s: float = 20.0) -> None:
    _load_env()
    from director_os.core.kaggle import KaggleClient

    client = KaggleClient(username=os.environ["KAGGLE_USERNAME"], key=os.environ["KAGGLE_KEY"])
    slug = REF_FILE.read_text().strip() if REF_FILE.exists() else SLUG
    deadline = time.monotonic() + budget_s
    status = "unknown"
    while time.monotonic() < deadline:
        status = client.kernel_status(slug)
        print(f"[{time.strftime('%H:%M:%S')}] status={status}", flush=True)
        if status in TERMINAL:
            break
        time.sleep(interval_s)
    if status not in TERMINAL:
        print("still running — re-invoke: uv run scripts/demucs_stems_run.py poll")
        sys.exit(3)
    if status == "error":
        out = client.kernel_output(slug)
        tail = "".join(str(rec.get("data", "")) for rec in out["log"])[-1500:]
        print("KERNEL FAILED:\n", tail)
        sys.exit(4)
    print("complete — harvest: uv run scripts/demucs_stems_run.py harvest")


# --------------------------------------------------------------------- log
def phase_log(max_lines: int = 120) -> None:
    """Print the kernel log tail — debugging failed runs."""
    _load_env()
    from director_os.core.kaggle import KaggleClient

    client = KaggleClient(username=os.environ["KAGGLE_USERNAME"], key=os.environ["KAGGLE_KEY"])
    slug = REF_FILE.read_text().strip() if REF_FILE.exists() else SLUG
    out = client.kernel_output(slug)
    text = "".join(str(rec.get("data", "")) for rec in out["log"])
    lines = text.splitlines()
    print(f"kernel={slug} log_lines={len(lines)} files={out['files']}")
    for line in lines[-max_lines:]:
        print(line[:200])


# ----------------------------------------------------------------- harvest
def phase_harvest() -> None:
    _load_env()
    from director_os.core.kaggle import KaggleClient
    from director_os.vision_fx.compute.base import StemsResult

    client = KaggleClient(username=os.environ["KAGGLE_USERNAME"], key=os.environ["KAGGLE_KEY"])
    slug = REF_FILE.read_text().strip() if REF_FILE.exists() else SLUG
    out = client.kernel_output(slug)
    files = [e["name"] for e in out["files"] if e["name"].endswith(("vocals.wav", "no_vocals.wav"))]
    if not files:
        raise SystemExit(f"no stem files in kernel output: {out['files']}")
    local: dict[str, Path] = {}
    for fname in files:
        dest = client.download_output_file(slug, fname, HARVEST_DIR / "stems")
        local[Path(fname).name] = dest
    vocals = local["vocals.wav"]
    music = local["no_vocals.wav"]
    dur_v, dur_m = _ffprobe_duration(vocals), _ffprobe_duration(music)
    vol_v = _volumedetect(vocals)
    vol_m = _volumedetect(music)
    result = StemsResult(vocals_path=vocals, music_path=music, spans=[], backend="kaggle-gpu")
    evidence = {
        "kernel": slug,
        "backend": result.backend,
        "model": "htdemucs (two_stems=vocals)",
        "audio_url": f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}/{ASSET_NAME}",
        "vocals_path": str(vocals),
        "music_path": str(music),
        "vocals_duration_s": round(dur_v, 2),
        "music_duration_s": round(dur_m, 2),
        "vocals_mean_volume_db": vol_v.get("mean_volume"),
        "vocals_max_volume_db": vol_v.get("max_volume"),
        "music_mean_volume_db": vol_m.get("mean_volume"),
        "music_max_volume_db": vol_m.get("max_volume"),
        "kernel_log_tail": "".join(str(rec.get("data", "")) for rec in out["log"])[-400:],
    }
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2))
    if abs(dur_v - dur_m) > 0.5:
        raise SystemExit("stem durations diverge — corrupt output")
    print("OK — evidence written:", EVIDENCE)


def _volumedetect(path: Path) -> dict[str, str]:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    stats: dict[str, str] = {}
    for line in proc.stderr.splitlines():
        if "mean_volume:" in line:
            stats["mean_volume"] = line.rsplit(":", 1)[1].strip().split(" ")[0]
        elif "max_volume:" in line:
            stats["max_volume"] = line.rsplit(":", 1)[1].strip().split(" ")[0]
    return stats


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
        raise SystemExit(f"usage: demucs_stems_run.py {'|'.join(PHASES)}")
    PHASES[sys.argv[1]]()
