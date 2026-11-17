"""Full e2e pipeline smoke — REAL artifacts, REAL Gladia ASR, no mocks.

Builds a spoken bodycam-like fixture (edge-tts radio chatter + testsrc2 video),
then drives run_pipeline through every stage: ingest (480p proxy + scrubsheet),
perception (live Gladia v2), planning (auto-cut v1), render (segment cache +
blur + card + loudnorm), subtitles (json/ass/srt), QA audit.

Run from repo root with .env exported:
    set -a; . ./.env; set +a; uv run python scripts/e2e_smoke.py

The resulting project row lives in the same sqlite DB the API reads, so the
Director Studio web UI can open it right after this script finishes.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "perception" / "src"))

from director_os.core.config import get_settings  # noqa: E402
from director_os.core.db import ProjectRow, new_id  # noqa: E402
from director_os.otio_engine.segment_cache import VideoRenderSpec  # noqa: E402
from director_os.perception.gladia import GladiaClient  # noqa: E402
from director_os.worker.pipeline import make_deps, run_pipeline  # noqa: E402

VOICE = "en-US-BrianMultilingualNeural"

LINES = [
    (0.5, "Unit one to dispatch, we have a code four in progress at Fifth and Main."),
    (6.0, "Dispatch copies, show unit one en route, ETA two minutes."),
]

OUT = Path("output/e2e")


def synth_line(text: str, dest: Path) -> None:
    import edge_tts

    async def run() -> None:
        await edge_tts.Communicate(text, VOICE).save(str(dest))

    asyncio.run(run())


def build_fixture(dest: Path) -> None:
    """12s clip: testsrc2 video + two TTS radio utterances, 48k stereo AAC."""
    OUT.mkdir(parents=True, exist_ok=True)
    parts = []
    for i, (_, text) in enumerate(LINES):
        p = OUT / f"line{i}.mp3"
        synth_line(text, p)
        parts.append(p)
        print(f"  [tts] line{i}: {p.stat().st_size} bytes")

    # input 0 = video, input 1 = silent 12s bed, inputs 2.. = TTS lines
    delays = [f"adelay={int(delay * 1000)}:all=1" for delay, _ in LINES]
    inputs: list[str] = ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo:d=12"]
    for p in parts:
        inputs += ["-i", str(p)]
    filter_parts = [f"[{i + 2}:a]{d}[a{i}]" for i, d in enumerate(delays)]
    mix = "".join(f"[a{i}]" for i in range(len(parts)))
    filter = ";".join(
        [*filter_parts, f"[1:a]{mix}amix=inputs={len(parts) + 1}:duration=first:normalize=0[out]"]
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=320x240:rate=30:duration=12",
        *inputs,
        "-filter_complex",
        filter,
        "-map",
        "0:v",
        "-map",
        "[out]",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "30",
        "-c:a",
        "aac",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-t",
        "12",
        "-shortest",
        str(dest),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"  [fixture] {dest} ({dest.stat().st_size} bytes)")


def main() -> int:
    settings = get_settings()
    src = OUT / "fixture_speech.mp4"
    print("[e2e] building spoken fixture...")
    build_fixture(src)

    db_url = settings.database_url
    print(f"[e2e] db: {db_url}")

    deps = make_deps(
        db_url,
        transcribe=GladiaClient(
            api_key=settings.gladia_api_key, contact_email=settings.contact_email
        ).transcribe,
        video=VideoRenderSpec(w=640, h=360, fps=30),
        log=lambda m: print(m),
    )

    with deps.sessions() as s:
        project = ProjectRow(id=new_id(), title="E2E smoke — spoken fixture")
        s.add(project)
        s.commit()
        pid = project.id
    print(f"[e2e] project: {pid}")

    result = run_pipeline(src, pid, deps)

    print("\n================ E2E RESULT ================")
    print(f"project:   {result.project_id}")
    print(f"asset:     {result.asset_id}")
    print(f"job:       {result.job_id}")
    print(f"otio ver:  {result.otio_version}")
    print(f"final:     {result.final_path} ({result.final_path.stat().st_size} bytes)")
    print(f"render:    {result.render.rendered} rendered / {result.render.cached} cached")
    print(f"audit ok:  {result.audit.ok}")
    for c in result.audit.checks:
        print(f"  - {c.name}: {c.detail}")
    if result.audit.failures:
        for c in result.audit.failures:
            print(f"  FAIL {c.name}: {c.detail}")

    pipe_dir = settings.output_dir / "pipeline" / pid
    subs = pipe_dir / "subtitles"
    if subs.exists():
        for f in sorted(subs.iterdir()):
            print(f"subtitle artifact: {f.name} ({f.stat().st_size} bytes)")
        srt = (subs / "subtitles.srt").read_text()
        print("--- srt head ---")
        print("\n".join(srt.splitlines()[:8]))

    proxy = pipe_dir / "assets" / result.asset_id
    if proxy.exists():
        for f in sorted(proxy.iterdir()):
            print(f"proxy artifact: {f.name} ({f.stat().st_size} bytes)")

    print(f"\nweb UI: http://localhost:3000/projects/{pid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
