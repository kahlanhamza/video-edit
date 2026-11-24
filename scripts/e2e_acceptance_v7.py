"""REAL end-to-end acceptance of the caption law (d6b939e) in a keyless sandbox.

REAL:   edge-tts synthesis over the network via the repo's own
        synthesize_with_timestamps (TRUE word boundaries), ffmpeg ingest +
        proxy, deterministic fallback plan, narration TTS + word sidecars,
        segment render, YuNet face sweep + tracked blur, the master-clock
        caption builder, the ASS burn over the blurred master, QA audit.
SEAM:   `deps.transcribe` — the pipeline's official injection point (no Gladia
        key in this sandbox). The injected words are the fixture's TRUE spoken
        words with TRUE edge-tts word-boundary timestamps, so every downstream
        timing math (source->master remap, cue windows) exercises real data.

Run:  uv run python scripts/e2e_acceptance_v7.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "perception" / "src"))
sys.path.insert(0, str(REPO / "packages" / "audio_engine" / "src"))

from director_os.audio_engine.tts import EdgeTtsBackend  # noqa: E402
from director_os.core.db import ProjectRow, new_id  # noqa: E402
from director_os.otio_engine.segment_cache import VideoRenderSpec  # noqa: E402
from director_os.perception.transcript import Transcript, TranscriptSegment, WordTiming  # noqa: E402
from director_os.worker.pipeline import make_deps, run_pipeline  # noqa: E402

VOICE = "en-US-BrianMultilingualNeural"
OUT = Path("output/e2e_v7")
# 480s = DOCUMENTARY_MIN_SOURCE_S: at/above it the doc planner engages narration
FIXTURE_DUR = 480

# (start_s, speaker, line) — police radio chatter spread across 480 s
LINES = [
    (6.0, "Officer", "Unit one to dispatch, we have a code four in progress at Fifth and Main."),
    (60.0, "Officer", "Dispatch copies, show unit one en route, ETA two minutes out."),
    (140.0, "Officer", "Step out of the vehicle and keep your hands where I can see them."),
    (225.0, "Suspect", "I got off work an hour ago officer, there is nothing in the car."),
    (
        300.0,
        "Officer",
        "You are being detained for a field sobriety test, do not reach for anything.",
    ),
    (355.0, "Suspect", "This is not right, I want my lawyer on the phone right now."),
    (410.0, "Officer", "Stay right there, do not move, the K nine unit is on the way."),
    (445.0, "Suspect", "I did nothing wrong, call my lawyer, call my lawyer right now."),
]


def synth_all() -> list[dict]:
    """Real edge-tts with TRUE word boundaries -> line mp3s + word stamps."""
    out: list[dict] = []
    for start, speaker, text in LINES:
        dest = OUT / f"line_{int(start)}.mp3"
        audio, stamps = EdgeTtsBackend().synthesize_with_timestamps(text, VOICE)
        dest.write_bytes(audio)
        words = [
            {
                "text": w.word,
                "start_s": round(start + w.start_s, 3),
                "end_s": round(start + w.end_s, 3),
                "speaker": speaker,
            }
            for w in stamps
        ]
        out.append({"start": start, "speaker": speaker, "text": text, "words": words})
        print(
            f"  [tts] {start:>5.1f}s {speaker:<7} "
            f"{len(words)} words -> {dest.name} ({len(audio)} B)"
        )
    (OUT / "true_word_stamps.json").write_text(json.dumps(out, indent=1))
    return out


def build_fixture(lines: list[dict], dest: Path) -> None:
    """FIXTURE_DUR s: testsrc2 video + TTS lines delayed to their start times.

    Three bounded ffmpeg steps (the sandbox watchdog kills one giant lavfi
    graph): video-only render, adelayed audio bed, then a stream-copy mux.
    """
    video = OUT / "video_only.mp4"
    bed = OUT / "audio_bed.m4a"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=640x360:rate=30:duration={FIXTURE_DUR}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "30",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
        capture_output=True,
    )
    inputs: list[str] = ["-f", "lavfi", "-i", f"anullsrc=r=48000:cl=stereo:d={FIXTURE_DUR}"]
    filter_parts: list[str] = []
    for i, line in enumerate(lines):
        p = OUT / f"line_{int(line['start'])}.mp3"
        inputs += ["-i", str(p)]
        filter_parts.append(f"[{i + 1}:a]adelay={int(line['start'] * 1000)}:all=1[a{i}]")
    mix = "".join(f"[a{i}]" for i in range(len(lines)))
    filt = ";".join(
        [*filter_parts, f"[0:a]{mix}amix=inputs={len(lines) + 1}:duration=first:normalize=0[out]"]
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            *inputs,
            "-filter_complex",
            filt,
            "-map",
            "[out]",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-t",
            str(FIXTURE_DUR),
            str(bed),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-i",
            str(bed),
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-t",
            str(FIXTURE_DUR),
            str(dest),
        ],
        check=True,
        capture_output=True,
    )
    print(f"  [fixture] {dest} ({dest.stat().st_size} bytes)")


def transcribe_seam(media: Path, asset_id: str) -> Transcript:
    """The official pipeline seam. Returns the fixture's TRUE spoken words
    (edge-tts word boundaries captured at synthesis time) — offline stand-in
    for Gladia, not invented data."""
    lines = json.loads((OUT / "true_word_stamps.json").read_text())
    words = [
        WordTiming(text=w["text"], start_s=w["start_s"], end_s=w["end_s"], speaker=w["speaker"])
        for line in lines
        for w in line["words"]
    ]
    segments = [
        TranscriptSegment(
            speaker=line["speaker"],
            start_s=line["words"][0]["start_s"],
            end_s=line["words"][-1]["end_s"],
            text=line["text"],
            words=[
                WordTiming(
                    text=w["text"], start_s=w["start_s"], end_s=w["end_s"], speaker=line["speaker"]
                )
                for w in line["words"]
            ],
        )
        for line in lines
        if line["words"]
    ]
    return Transcript(
        asset_id=asset_id,
        language="en",
        duration_s=FIXTURE_DUR,
        words=words,
        segments=segments,
        provider="fixture-word-boundaries",
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[e2e-v7] synthesizing fixture lines (real edge-tts, true word stamps)...")
    lines = synth_all()
    src = OUT / "fixture_speech_v7.mp4"
    print("[e2e-v7] building 480s spoken fixture...")
    build_fixture(lines, src)

    log_path = OUT / "pipeline_log.txt"

    def log(msg: str) -> None:
        print(msg)
        with log_path.open("a") as fh:
            fh.write(str(msg) + "\n")

    log_path.write_text("")
    # the sandbox env may carry a foreign DATABASE_URL — keep this run local
    import director_os.core.credentials  # noqa: F401 — register SettingsMetaRow
    from director_os.core.db import init_models, make_engine

    db_url = f"sqlite:///{(OUT / 'directoros.db').resolve()}"
    init_models(make_engine(db_url))
    deps = make_deps(
        db_url,
        transcribe=transcribe_seam,
        video=VideoRenderSpec(w=640, h=360, fps=30),
        log=log,
    )
    with deps.sessions() as s:
        project = ProjectRow(id=new_id(), title="E2E v7 caption-law acceptance")
        s.add(project)
        s.commit()
        pid = project.id
    print(f"[e2e-v7] project: {pid}")

    result = run_pipeline(src, pid, deps)

    print("\n================ E2E V7 RESULT ================")
    print(f"project:   {result.project_id}")
    print(f"otio:      {result.otio_path}")
    print(f"final:     {result.final_path} ({result.final_path.stat().st_size} bytes)")
    print(f"audit ok:  {result.audit.ok}")
    for c in result.audit.failures:
        print(f"  FAIL {c.name}: {c.detail}")
    print(f"log:       {log_path}")
    (OUT / "last_result.json").write_text(
        json.dumps(
            {
                "project_id": result.project_id,
                "otio_path": str(result.otio_path),
                "final_path": str(result.final_path),
                "audit_ok": result.audit.ok,
            },
            indent=1,
        )
    )
    print("\nnext: uv run python scripts/verify_e2e_v7.py")
    sys.stdout.flush()
    sys.stderr.flush()
    # hard exit: the sqlite tracker keeps a non-daemon thread alive after
    # main() — without this the process hangs after a fully successful run
    os._exit(0 if result.audit.ok else 1)


if __name__ == "__main__":
    sys.exit(main())
