#!/usr/bin/env python3
"""Gladia ASR on the SHIPPED MASTER audio (v6 caption rebuild).

The v5 regression burned SOURCE-clock captions onto the 657.7 s master. v6
restarts caption truth on the MASTER clock itself: this driver transcribes
pipeline_run/v3/audio_v3.m4a (the exact shipped audio) with diarization, so
every word timestamp is directly burnable — no clock mapping can drift.

Chunk-safe like asr_full.py: the Gladia job id is persisted immediately; a
killed run resumes polling. Outputs:
  final_state/asr_master_v6_raw.json   full provider response (utterances etc.)
  final_state/asr_master_v6.json       {words:[{w,s,e,speaker,conf}], segments:[...]}
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/z/my-project/director-os")
for p in (
    REPO / "packages" / "core" / "src",
    REPO / "packages" / "perception" / "src",
):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from director_os.perception.gladia import GladiaClient  # noqa: E402

AUDIO = Path("/home/z/my-project/pipeline_run/v3/audio_v3.m4a")
WAV = Path("/home/z/my-project/pipeline_run/v3/audio_master_16k.wav")
WORK = Path("/home/z/my-project/pipeline_run/final_state")
STATE = WORK / "asr_master_v6_state.json"
RAW_OUT = WORK / "asr_master_v6_raw.json"
OUT = WORK / "asr_master_v6.json"


def load_key() -> str:
    for line in (REPO / ".env").read_text().splitlines():
        if line.startswith("GLADIA_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("GLADIA_API_KEY missing")


def ensure_wav() -> None:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(WAV)],
        capture_output=True,
        text=True,
    )
    try:
        ok = abs(float(r.stdout.strip()) - 657.7) < 3.0
    except ValueError:
        ok = False
    if ok and WAV.stat().st_size > 5_000_000:
        print(f"[asr6] wav cached ({WAV.stat().st_size / 1e6:.1f} MB)")
        return
    print("[asr6] extracting 16k mono wav from the shipped master audio...")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(AUDIO),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(WAV),
        ],
        check=True,
    )
    print(f"[asr6] wav ready ({WAV.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    if OUT.exists():
        print("[asr6] transcript already saved — nothing to do")
        return 0
    ensure_wav()
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    client = GladiaClient(api_key=load_key(), contact_email="dev@director-os.local")

    job_id = state.get("job_id")
    if not job_id:
        print("[asr6] uploading master wav to Gladia...")
        url = client._upload(WAV)
        job_id = client._create_job(url)
        STATE.write_text(json.dumps({"job_id": job_id, "created_at": time.time()}))
        print(f"[asr6] job created: {job_id} (persisted)")
    else:
        print(f"[asr6] resuming poll of job {job_id}")

    deadline = time.monotonic() + 540.0
    while True:
        data = client._http.get(
            f"{client._base}/v2/transcription/{job_id}", headers=client._headers()
        ).json()
        status = data.get("status")
        print(f"[asr6] status={status}", flush=True)
        if status == "done":
            RAW_OUT.write_text(json.dumps(data))
            tr = client._normalize(data, "master-v6")
            words = [
                {
                    "w": w.text,
                    "s": round(w.start_s, 3),
                    "e": round(w.end_s, 3),
                    "speaker": w.speaker,
                    "conf": round(w.confidence, 3),
                }
                for w in tr.words
            ]
            segments = [
                {
                    "speaker": s.speaker,
                    "s": round(s.start_s, 3),
                    "e": round(s.end_s, 3),
                    "text": s.text,
                }
                for s in tr.segments
            ]
            OUT.write_text(
                json.dumps(
                    {"words": words, "segments": segments, "n_words": len(words)},
                    ensure_ascii=False,
                    indent=1,
                )
            )
            STATE.unlink(missing_ok=True)
            print(f"[asr6] DONE: {len(words)} words, {len(segments)} segments -> {OUT}")
            return 0
        if status in ("error", "failed"):
            print(f"[asr6] JOB FAILED: {str(data)[:400]}")
            STATE.unlink(missing_ok=True)
            return 1
        if time.monotonic() > deadline:
            print("[asr6] chunk deadline — job id persisted, rerun to resume")
            return 2
        time.sleep(client._poll_interval_s)


if __name__ == "__main__":
    sys.exit(main())
