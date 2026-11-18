"""Full e2e pipeline on REAL footage — no synthetic fixtures, no mocks.

Takes a real source video (default: the Eustis FL bodycam file from Wikimedia
Commons), cuts a working segment, runs the M4 vision stack (redaction sweep +
optional frame-quality/VLM-index/timestamp-OCR enrichment), then drives
run_pipeline through every stage on real pixels and real speech:

  ingest (probe + 480p proxy + scrubsheet)
  -> redaction sweep (yuNet CPU | Modal GPU YOLO+ByteTrack; idempotent)
  -> perception (LIVE Gladia v2 ASR with diarization)
  -> vision enrichment (--vision: frame quality + contact-sheet indexing + AXON OCR)
  -> planning (auto-cut v1; OCR populates the intro card when present)
  -> render (segment cache + tracked/static blur + location card + loudnorm)
  -> subtitles (json/ass/srt from the real transcript)
  -> QA audit
  -> burn-in pass (speaker-colored .ass subtitles into the master)

Run from repo root with .env exported:
    set -a; . ./.env; set +a; uv run python scripts/e2e_real_video.py

Options: --start/--dur (segment window, default 60s + 240s), --full (skip
segmenting, use the whole source), --render WxH (default 1280x720),
--redaction-backend yunet|modal|none, --vision (VLM enrichment when keys exist),
--llm-direct (the Gemini Director agent re-cuts the v1 timeline before render).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "perception" / "src"))

from director_os.core.config import get_settings  # noqa: E402
from director_os.core.db import MediaAssetRow, PerceptionIndexRow, ProjectRow, new_id  # noqa: E402
from director_os.core.qa_probe import probe  # noqa: E402
from director_os.otio_engine.segment_cache import VideoRenderSpec  # noqa: E402
from director_os.perception.gladia import GladiaClient  # noqa: E402
from director_os.perception.transcript import Transcript  # noqa: E402
from director_os.worker.pipeline import (  # noqa: E402
    make_deps,
    run_pipeline,
    stage_ingest,
    stage_redaction,
)

DEFAULT_SOURCE = REPO / "data" / "media" / "bodycam_eustis.webm"
DOWNLOAD_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/6/67/"
    "Bodycam-_Drunk_Cop_Arrested_While_On_Duty._Eustis%2C_FL._July_04%2C_2022..webm"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default=str(DEFAULT_SOURCE))
    p.add_argument("--start", type=float, default=60.0, help="segment window start (s)")
    p.add_argument("--dur", type=float, default=240.0, help="segment duration (s)")
    p.add_argument("--full", action="store_true", help="use the whole source, no segment cut")
    p.add_argument("--render", default="1280x720", help="render geometry WxH")
    p.add_argument("--max-face-regions", type=int, default=6)
    p.add_argument(
        "--redaction-backend",
        choices=("yunet", "modal", "none"),
        default="yunet",
        help="yuNet CPU sweep (offline) | Modal GPU YOLO-face+plate+ByteTrack | skip",
    )
    p.add_argument(
        "--vision",
        action="store_true",
        help="vision enrichment: frame-quality scoring + VLM contact-sheet "
        "indexing + AXON timestamp OCR (VLM parts need VISION_API_KEY)",
    )
    p.add_argument(
        "--llm-direct",
        action="store_true",
        help="Director autodrive: after the v1 auto-cut, the Gemini Director agent "
        "re-cuts the timeline (keep/drop/tighten/narration/cards) via checkpointed "
        "tool calls; soft-fails to the v1 cut when the LLM is unavailable",
    )
    p.add_argument("--skip-asr", action="store_true", help="reuse persisted transcript if any")
    p.add_argument(
        "--stub-asr",
        action="store_true",
        help="keyless smoke: empty transcript, real pixels/sweep/render (no Gladia)",
    )
    p.add_argument(
        "--project-id",
        default=None,
        help="resume an existing project (sandbox-safe: re-invoke until QA passes)",
    )
    return p.parse_args()


def ensure_source(src: Path) -> Path:
    if src.exists():
        return src
    is_local_missing = src == DEFAULT_SOURCE and not DEFAULT_SOURCE.exists()
    if not str(src).startswith(("http://", "https://")) and is_local_missing:
        print(f"[source] {src} missing — downloading from Wikimedia (448 MB, respects 429 backoff)")
        REPO.joinpath("data/media").mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "curl",
                "-sL",
                "--retry",
                "5",
                "--retry-all-errors",
                "-A",
                "DirectorOS-Pipeline/1.0 (E2E; dev@director-os.local)",
                "-o",
                str(DEFAULT_SOURCE),
                DOWNLOAD_URL,
            ],
            check=True,
        )
    return src


def cut_segment(src: Path, start: float, dur: float, dest: Path) -> Path:
    """Stream-copy cut (keyframe-snapped): fast, keeps real pixels untouched."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.2f}",
        "-i",
        str(src),
        "-t",
        f"{dur:.2f}",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        str(dest),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return dest


def make_transcribe(settings) -> object:
    """Gladia callable that hands the API a clean 16k mono WAV (small upload)."""
    gladia = GladiaClient(api_key=settings.gladia_api_key, contact_email=settings.contact_email)

    def transcribe(path: Path, asset_id: str) -> Transcript:
        wav = Path(path).with_suffix(".asr.wav")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                str(wav),
            ],
            check=True,
            capture_output=True,
        )
        size_mb = wav.stat().st_size / 1e6
        print(f"  [asr] uploading {size_mb:.1f} MB 16k mono wav to Gladia...")
        return gladia.transcribe(wav, asset_id)

    return transcribe


def _stub_transcribe(path: Path, asset_id: str) -> Transcript:
    """Keyless smoke: empty transcript — everything else stays real."""
    return Transcript(asset_id=asset_id, duration_s=probe(path).duration_s)


def burn_subtitles(master: Path, ass_path: Path, dest: Path) -> Path:
    """Post-pass: speaker-colored .ass into the master (re-encode, 720p fast)."""
    escaped = str(ass_path).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    vf = f"ass='{escaped}'"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(master),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "copy",
            str(dest),
        ],
        check=True,
        capture_output=True,
    )
    return dest


def main() -> int:
    args = parse_args()
    settings = get_settings()

    src = ensure_source(Path(args.source))
    info = probe(src)
    print(
        f"[source] {src.name}: {info.duration_s:.1f}s {info.width}x{info.height}@{info.fps:.2f} "
        f"({src.stat().st_size / 1e6:.0f} MB)"
    )

    if args.full:
        segment = src
    else:
        segment = REPO / "data" / "media" / f"segment_{int(args.start)}s_{int(args.dur)}s.webm"
        print(f"[segment] cutting {args.dur:.0f}s from t={args.start:.0f}s (stream copy)")
        cut_segment(src, args.start, args.dur, segment)
        seg_info = probe(segment)
        print(f"[segment] {seg_info.duration_s:.1f}s actual (keyframe snap)")

    w, h = (int(x) for x in args.render.lower().split("x"))
    deps = make_deps(
        settings.database_url,
        transcribe=(
            None
            if (args.skip_asr or args.stub_asr)
            else (
                _stub_transcribe if getattr(args, "stub_asr", False) else make_transcribe(settings)
            )
        ),
        video=VideoRenderSpec(w=w, h=h, fps=30),
        redact_backend=None if args.redaction_backend == "none" else args.redaction_backend,
        vision_enrich=args.vision,
        llm_direct=args.llm_direct,
        log=lambda m: print(m),
    )

    if args.project_id:
        pid = args.project_id
        with deps.sessions() as s:
            if s.get(ProjectRow, pid) is None:
                raise SystemExit(f"project {pid} does not exist")
        print(f"[project] resuming {pid}")
    else:
        with deps.sessions() as s:
            project = ProjectRow(id=new_id(), title="E2E real footage — Eustis FL bodycam")
            s.add(project)
            s.commit()
            pid = project.id
        print(f"[project] {pid}")

    # stage 1 up front so the redaction sweep lands before the render stage
    asset_id = stage_ingest(segment, pid, deps)
    if args.redaction_backend != "none":
        stage_redaction(asset_id, segment, pid, deps, max_regions=args.max_face_regions)
    with deps.sessions() as s:
        asset = s.get(MediaAssetRow, asset_id)
        assert asset is not None
        prow = s.get(PerceptionIndexRow, asset_id)
        red = (prow.redaction_tracks_json or {}) if prow else {}
        n_regions = len(red.get("regions", []))
        shots = (prow.shots_json or {}) if prow else {}
        print(f"[asset] {asset_id} {asset.width}x{asset.height} {asset.duration_s:.1f}s")
        print(f"[redaction] backend={red.get('source', 'none')} regions={n_regions}")
        if args.vision:
            fq = shots.get("frame_quality") or {}
            ocr = red.get("timestamp_ocr") or {}
            vt = shots.get("visual_timeline") or {}
            print(
                f"[vision] quality={fq.get('usable_ratio', 'n/a')} "
                f"timeline_insights={len(vt.get('insights', []))} "
                f"ocr={ocr.get('iso', 'not found')}"
            )

    result = run_pipeline(segment, pid, deps)

    print("\n================ REAL-FOOTAGE E2E RESULT ================")
    print(f"project:   {result.project_id}")
    print(f"asset:     {result.asset_id}")
    print(f"otio ver:  {result.otio_version}")

    # the Director's paper trail: every timeline version + who authored it
    from director_os.agent_runtime.checkpoint import CheckpointStore

    store = CheckpointStore(deps.sessions)
    versions = store.list_versions(pid)
    dir_versions = [v for v in versions if "director" in str(v.get("created_by", ""))]
    if dir_versions:
        first, last = dir_versions[0], dir_versions[-1]
        print(
            f"director versions: {len(dir_versions)} "
            f"(v{first['version_num']}..v{last['version_num']})"
        )
        for v in dir_versions[-4:]:
            print(f"  v{v['version_num']}: {v.get('checkpoint_note', '')[:90]}")
    logs = store.recent_logs(pid, limit=40)
    if logs:
        edit_prefixes = ("timeline.", "narration.", "cards.", "redaction.")
        edits = [row for row in logs if str(row.get("tool", "")).startswith(edit_prefixes)]
        if edits:
            print(f"director tool calls ({len(edits)}):")
            for row in edits:
                mark = "ok" if row.get("ok", True) else "FAIL"
                print(f"  [{mark}] {row['tool']} {json.dumps(row.get('args', {}))[:110]}")

    print(f"final:     {result.final_path} ({result.final_path.stat().st_size / 1e6:.1f} MB)")
    print(f"render:    {result.render.rendered} rendered / {result.render.cached} cached")
    print(f"redaction regions applied: {n_regions}")
    print(f"audit ok:  {result.audit.ok}")
    for c in result.audit.checks:
        print(f"  - {c.name}: {c.detail}")

    pipe_dir = settings.output_dir / "pipeline" / pid
    ass = pipe_dir / "subtitles.ass"
    srt = pipe_dir / "subtitles.srt"
    if srt.exists():
        head = "\n".join(srt.read_text().splitlines()[:12])
        print("--- real transcript (srt head) ---")
        print(head)

    deliverables = REPO.parent / "download"  # /home/z/my-project/download
    deliverables.mkdir(parents=True, exist_ok=True)
    final_copy = deliverables / "eustis_bodycam_master_720p.mp4"
    final_copy.write_bytes(result.final_path.read_bytes())
    if ass.exists():
        subbed = deliverables / "eustis_bodycam_master_720p_subbed.mp4"
        print(f"[burn-in] speaker-colored subtitles -> {subbed.name}")
        burn_subtitles(final_copy, ass, subbed)
        print(f"[burn-in] {subbed.name} ({subbed.stat().st_size / 1e6:.1f} MB)")
    for artifact, name in (
        (srt, "eustis_bodycam_subtitles.srt"),
        (ass, "eustis_bodycam_subtitles.ass"),
    ):
        if artifact.exists():
            (deliverables / name).write_bytes(artifact.read_bytes())

    print(f"\nweb UI: http://localhost:3000/projects/{pid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
