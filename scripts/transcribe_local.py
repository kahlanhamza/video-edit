"""Local ASR for the acceptance run — faster-whisper -> Gladia-shaped JSON.

The sandbox lost the GLADIA_API_KEY with the recycle, so the 43-min tape is
transcribed locally (CPU, int8, word timestamps). Output matches the
perception Transcript shape the whole pipeline consumes:
  {"words": [{"text", "start_s", "end_s", "speaker"}],
   "segments": [{"speaker", "start_s", "end_s", "text"}], "language": "en"}

Chunk-resumable: segments stream into a sidecar after each 60s audio chunk, so
a killed run re-enters and continues from the last completed minute.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def extract_audio(src: Path, dst: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(src),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(dst),
        ],
        check=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.getenv("WHISPER_MODEL", "small"))
    args = ap.parse_args()

    src = Path(args.source)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wav = out.with_suffix(".16k.wav")

    if not wav.exists() or wav.stat().st_size == 0:
        print(f"[asr] extracting 16k mono audio -> {wav.name}", flush=True)
        extract_audio(src, wav)

    from faster_whisper import WhisperModel

    print(f"[asr] loading whisper '{args.model}' (int8 CPU)...", flush=True)
    t0 = time.time()
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    segments_out: list[dict] = []
    words_out: list[dict] = []
    if out.exists():
        try:
            prev = json.loads(out.read_text())
            segments_out = list(prev.get("segments") or [])
            words_out = list(prev.get("words") or [])
            if segments_out:
                print(f"[asr] resuming with {len(segments_out)} persisted segments", flush=True)
        except (ValueError, OSError):
            pass

    done_to = max((float(s["end_s"]) for s in segments_out), default=0.0)
    duration = float(
        json.loads(
            subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "json",
                    str(wav),
                ],
                capture_output=True,
                text=True,
            ).stdout
        )["format"]["duration"]
    )
    print(f"[asr] {duration / 60:.1f} min of audio, resuming from {done_to:.0f}s", flush=True)

    CHUNK = 300.0
    cursor = done_to
    while cursor < duration - 1.0:
        end = min(cursor + CHUNK, duration)
        t_chunk = time.time()
        segs, _info = model.transcribe(
            str(wav),
            language="en",
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 700},
            clip_timestamps=f"{cursor},{end}",
        )
        n_seg = 0
        for seg in segs:
            text = (seg.text or "").strip()
            if not text:
                continue
            segments_out.append(
                {
                    "speaker": "Unknown",
                    "start_s": round(float(seg.start), 3),
                    "end_s": round(float(seg.end), 3),
                    "text": text,
                }
            )
            for w in seg.words or []:
                token = (w.word or "").strip()
                if not token:
                    continue
                words_out.append(
                    {
                        "text": token,
                        "start_s": round(float(w.start), 3),
                        "end_s": round(float(w.end), 3),
                        "speaker": "Unknown",
                    }
                )
            n_seg += 1
        cursor = end
        tmp = out.with_suffix(".tmp.json")
        tmp.write_text(
            json.dumps({"language": "en", "segments": segments_out, "words": words_out}, indent=1)
        )
        tmp.replace(out)
        print(
            f"[asr] {end / 60:5.1f} min done (+{time.time() - t_chunk:.0f}s, "
            f"{len(segments_out)} segments)",
            flush=True,
        )

    out.write_text(
        json.dumps({"language": "en", "segments": segments_out, "words": words_out}, indent=1)
    )
    print(
        f"[asr] COMPLETE: {len(segments_out)} segments / {len(words_out)} words "
        f"in {time.time() - t0:.0f}s -> {out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
