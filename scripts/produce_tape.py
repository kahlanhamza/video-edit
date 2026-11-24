"""Produce a channel-ready LENS & LAW tape from a verified raw source.

The production path (pick-by-pick, one tape per run):

  ingest -> redaction sweep (yuNet CPU) -> perception (live Gladia ASR)
  -> planning (Gemini Director when a key exists, deterministic fallback
     otherwise; the content gate + foreign-narration exclusion run INSIDE
     the planner) -> render -> v6 captions -> caption-guard report
  -> subtitle burn-in + watermark burn-in (one re-encode)
  -> deliverable copy + production report JSON in download/

Deliverable naming: <slug>_master_<H>p_final.mp4 plus subtitles and the
production report (case-gate verdict, every foreign-narration window that
was replaced, caption-guard samples, QA audit lines).

Run from repo root with keys exported:
    GLADIA_API_KEY=... LLM_API_KEY=... uv run python scripts/produce_tape.py \
        --source footage/newlondon_audit.mp4 --title "..." --dossier-json docs/cases/newlondon.json
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
from director_os.core.db import PerceptionIndexRow, ProjectRow, new_id  # noqa: E402
from director_os.core.qa_probe import probe  # noqa: E402
from director_os.otio_engine.segment_cache import VideoRenderSpec  # noqa: E402
from director_os.perception.transcript import Transcript  # noqa: E402
from director_os.vision_fx.caption_guard import scan_burned_captions  # noqa: E402
from director_os.worker.pipeline import (  # noqa: E402
    make_deps,
    run_pipeline,
    stage_ingest,
    stage_redaction,
)

DOWNLOAD = REPO.parent / "download"
BRAND_PACK = REPO / "assets" / "brand"
BADGE = BRAND_PACK / "lens-and-law-corner-badge-1200x400.png"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--slug", required=True, help="deliverable filename stem")
    p.add_argument(
        "--corner",
        default="bottom_right",
        choices=("top_left", "top_right", "bottom_left", "bottom_right"),
    )
    p.add_argument("--render", default="1920x1080", help="master geometry WxH")
    p.add_argument(
        "--dossier-json", default=None, help="verified case-facts JSON (CaseDossier shape)"
    )
    p.add_argument("--case-facts", default="", help="one-line case brief for the content gate")
    p.add_argument("--max-face-regions", type=int, default=8)
    p.add_argument("--skip-asr", action="store_true", help="reuse persisted transcript")
    p.add_argument(
        "--no-llm", action="store_true", help="skip the Gemini Director, deterministic only"
    )
    p.add_argument("--project-id", default=None)
    p.add_argument(
        "--finalize-only",
        action="store_true",
        help="resume a finished pipeline: burn captions + badge on the "
        "persisted master and deliver (no ingest/ASR/render)",
    )
    return p.parse_args()


def deps_sessions_from(settings):
    """A session factory for finalize-only reads (no full deps build)."""
    from contextlib import contextmanager

    from director_os.core.db import init_models, make_engine
    from sqlalchemy.orm import Session

    engine = make_engine(settings.database_url)
    init_models(engine)

    @contextmanager
    def _sessions():
        with Session(engine) as s:
            yield s

    return _sessions()


def make_transcribe(settings):
    """Gladia callable on a clean 16k mono WAV (small upload)."""
    from director_os.perception.gladia import GladiaClient

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
        print(f"  [asr] uploading {wav.stat().st_size / 1e6:.1f} MB 16k mono wav to Gladia...")
        return gladia.transcribe(wav, asset_id)

    return transcribe


def finalize(master: Path, ass: Path | None, out: Path, corner: str) -> Path:
    """One re-encode: v6 captions burn-in + the LENS & LAW corner badge.

    The badge pack ships true RGBA overlays — the native overlay filter
    composites them range-correct (the rgba-space roundtrip darkened the
    base frame ~9% and encoded at half the speed)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = BRAND_PACK / f"lens-and-law-overlay-1920x1080-{corner.replace('_', '-')}.png"
    fc = "[0:v][1:v]overlay=0:0,format=yuv420p[v]"
    if ass is not None and ass.exists():
        escaped = str(ass).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
        fc = f"[0:v]ass='{escaped}'[vb];[vb][1:v]overlay=0:0,format=yuv420p[v]"
    argv = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(master),
        "-i",
        str(frame),
        "-filter_complex",
        fc,
        "-map",
        "[v]",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "copy",
        str(out),
    ]
    subprocess.run(argv, check=True, capture_output=True)
    return out


def main() -> int:
    args = parse_args()
    settings = get_settings()
    src = Path(args.source)
    assert src.exists(), f"source missing: {src}"
    info = probe(src)
    print(f"[source] {src.name}: {info.duration_s:.1f}s {info.width}x{info.height}@{info.fps:.2f}")

    # --- content gate, evaluated BEFORE anything expensive (fail fast)
    from director_os.agent_runtime.content_policy import evaluate_case

    verdict = evaluate_case(args.case_facts)
    print(f"[gate] allowed={verdict.allowed} mode={verdict.mode} reason={verdict.reason}")
    if not verdict.allowed:
        return 2

    # --- caption-guard pre-check: burned-in captions on the SOURCE tape
    guard = scan_burned_captions(src, info.duration_s)
    print(f"[caption-guard] median={guard['median_score']} flagged={guard['flagged']}")
    if guard["flagged"]:
        print(
            "[caption-guard] WARNING: source carries burned-in captions — re-source or shift lanes"
        )

    w, h = (int(x) for x in args.render.lower().split("x"))
    pid = args.project_id
    if args.finalize_only:
        # the pipeline already finished: burn captions + badge on the persisted
        # master, audit the FINAL file, deliver (everything else is cached)
        pipe_dir = settings.output_dir / "pipeline" / str(pid)
        master = pipe_dir / "final.mp4"
        assert master.exists(), f"persisted master missing: {master}"
        ass = pipe_dir / "subtitles.ass"
        final_out = DOWNLOAD / f"{args.slug}_master_{h}p_final.mp4"
        if not (final_out.exists() and final_out.stat().st_size > 0):
            finalize(master, ass, final_out, args.corner)
        else:
            print(f"[finalize] {final_out.name} already burned — skipping the re-encode")
        size_mb = final_out.stat().st_size / 1e6
        print(f"[finalize] captions + LENS & LAW badge -> {final_out.name} ({size_mb:.1f} MB)")
        for artifact, name in (
            (pipe_dir / "subtitles.srt", f"{args.slug}_captions.srt"),
            (ass, f"{args.slug}_captions.ass"),
        ):
            if artifact.exists():
                (DOWNLOAD / name).write_bytes(artifact.read_bytes())
        from director_os.core.qa_probe import audit_technical

        audit = audit_technical(final_out, expected_dur_s=None)
        print(f"[qa] audit ok={audit.ok}")
        for c in audit.checks:
            print(f"  - {c.name}: {c.detail}")
        with deps_sessions_from(settings) as s:
            prow = s.get(PerceptionIndexRow, pid)
        transcript = (prow.transcript_json or {}) if prow else {}
        from director_os.agent_runtime.foreign_media import foreign_narration_ranges

        foreign_report = [
            {"start_s": round(a, 2), "end_s": round(b, 2), "reason": label}
            for a, b, label in foreign_narration_ranges(transcript)
        ]
        final_info = probe(final_out)
        report = {
            "tape": args.title,
            "slug": args.slug,
            "source": {
                "file": src.name,
                "duration_s": round(info.duration_s, 2),
                "geometry": f"{info.width}x{info.height}",
            },
            "master": {
                "file": final_out.name,
                "duration_s": round(final_info.duration_s, 2),
                "geometry": f"{final_info.width}x{final_info.height}",
            },
            "content_gate": {
                "allowed": verdict.allowed,
                "mode": verdict.mode,
                "categories": sorted(verdict.categories),
                "reason": verdict.reason,
            },
            "foreign_narration_replaced": foreign_report,
            "caption_guard": guard,
            "qa_audit": {
                "ok": audit.ok,
                "checks": [{"name": c.name, "detail": c.detail} for c in audit.checks],
            },
            "project_id": pid,
        }
        report_path = DOWNLOAD / f"{args.slug}_production_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"[report] {report_path}")
        return 0

    deps = make_deps(
        settings.database_url,
        transcribe=None if args.skip_asr else make_transcribe(settings),
        video=VideoRenderSpec(w=w, h=h, fps=30),
        redact_backend="yunet",
        vision_enrich=False,
        llm_direct=not args.no_llm,
        log=lambda m: print(m),
    )

    if args.project_id:
        pid = args.project_id
        with deps.sessions() as s:
            assert s.get(ProjectRow, pid) is not None, f"project {pid} does not exist"
        print(f"[project] resuming {pid}")
    else:
        with deps.sessions() as s:
            project = ProjectRow(id=new_id(), title=args.title)
            s.add(project)
            s.commit()
            pid = project.id
        print(f"[project] {pid}")

    # --- verified case facts -> the dossier row the Director reads
    if args.dossier_json:
        from director_os.core.db import CaseDossierRow

        payload = json.loads(Path(args.dossier_json).read_text())
        with deps.sessions() as s:
            row = s.get(CaseDossierRow, pid)
            if row is None:
                s.add(CaseDossierRow(project_id=pid, dossier_json=payload))
            else:
                row.dossier_json = payload
            s.commit()
        n_facts = len(payload.get("key_facts", []))
        print(f"[dossier] verified case facts attached ({n_facts} key facts)")

    asset_id = stage_ingest(src, pid, deps)
    stage_redaction(asset_id, src, pid, deps, max_regions=args.max_face_regions)
    result = run_pipeline(src, pid, deps)

    print("\n================ PRODUCTION RUN RESULT ================")
    print(f"project:  {result.project_id}")
    print(f"final:    {result.final_path} ({result.final_path.stat().st_size / 1e6:.1f} MB)")
    print(f"audit ok: {result.audit.ok}")
    for c in result.audit.checks:
        print(f"  - {c.name}: {c.detail}")

    # --- foreign-narration report (which windows were replaced)
    foreign_windows: list[dict] = []
    with deps.sessions() as s:
        prow = s.get(PerceptionIndexRow, asset_id)
        transcript = (prow.transcript_json or {}) if prow else {}
    from director_os.agent_runtime.foreign_media import foreign_narration_ranges

    for a, b, label in foreign_narration_ranges(transcript):
        foreign_windows.append({"start_s": round(a, 2), "end_s": round(b, 2), "reason": label})
    if foreign_windows:
        print(f"[foreign-narration] {len(foreign_windows)} window(s) excluded from raw clips:")
        for r in foreign_windows:
            print(f"    {r['start_s']:8.1f}-{r['end_s']:8.1f}s  {r['reason']}")
    else:
        print("[foreign-narration] none detected — pure raw tape")

    # --- subtitle burn-in + watermark in ONE re-encode
    pipe_dir = settings.output_dir / "pipeline" / pid
    ass = pipe_dir / "subtitles.ass"
    final_out = DOWNLOAD / f"{args.slug}_master_{h}p_final.mp4"
    finalize(result.final_path, ass, final_out, args.corner)
    size_mb = final_out.stat().st_size / 1e6
    print(f"[finalize] captions + LENS & LAW badge -> {final_out.name} ({size_mb:.1f} MB)")

    for artifact, name in (
        (pipe_dir / "subtitles.srt", f"{args.slug}_captions.srt"),
        (ass, f"{args.slug}_captions.ass"),
    ):
        if artifact.exists():
            (DOWNLOAD / name).write_bytes(artifact.read_bytes())

    final_info = probe(final_out)
    report = {
        "tape": args.title,
        "slug": args.slug,
        "source": {
            "file": src.name,
            "duration_s": round(info.duration_s, 2),
            "geometry": f"{info.width}x{info.height}",
        },
        "master": {
            "file": final_out.name,
            "duration_s": round(final_info.duration_s, 2),
            "geometry": f"{final_info.width}x{final_info.height}",
        },
        "content_gate": {
            "allowed": verdict.allowed,
            "mode": verdict.mode,
            "categories": sorted(verdict.categories),
            "reason": verdict.reason,
        },
        "foreign_narration_replaced": foreign_windows,
        "caption_guard": guard,
        "qa_audit": {
            "ok": result.audit.ok,
            "checks": [{"name": c.name, "detail": c.detail} for c in result.audit.checks],
        },
        "project_id": pid,
    }
    report_path = DOWNLOAD / f"{args.slug}_production_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[report] {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
