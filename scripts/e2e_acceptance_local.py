"""Acceptance run — full Stage 1-10 on ANY source length, local ASR, no keys.

The 2026-11-19 acceptance: hook-first planning (scaled law), word-count
narration with pause compression, measured-TTS card re-anchoring, 2-bus /
loudnorm audio, post-FX on the deliverable ONLY (tracked blur incl. B-roll +
selective keep + audio enhance + burned captions), QA retention gates, ship
pack. This runner injects the locally-transcribed transcript and copies every
deliverable into /home/z/my-project/download/.

Usage:
  uv run python scripts/e2e_acceptance_local.py --source data/media/bodycam_eustis.webm \
      --transcript output/pipeline/eustis_local/transcript.json [--project-id PID] [--render WxH]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "packages" / "core" / "src"))

from director_os.core.config import get_settings  # noqa: E402
from director_os.core.db import CaseSourceRow, ProjectRow, init_models, make_engine, new_id  # noqa: E402
from director_os.core.qa_probe import probe  # noqa: E402
from director_os.otio_engine.segment_cache import VideoRenderSpec  # noqa: E402
from director_os.worker.pipeline import PipelineDeps, make_deps, run_pipeline  # noqa: E402

DOWNLOAD = Path("/home/z/my-project/download")

SOURCE_PAGE = (
    "https://commons.wikimedia.org/wiki/File:Bodycam-_Drunk_Cop_Arrested_"
    "While_On_Duty._Eustis%2C_FL._July_04%2C_2022..webm"
)


class LocalTranscript:
    """The locally-produced transcript served as the ASR callable (persisted
    once by scripts/transcribe_local.py — the pipeline's crash-resume persists
    it into perception_index on first use, re-runs never re-transcribe)."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = 0

    def __call__(self, path: Path, asset_id: str):
        self.calls += 1
        print(f"  [asr] serving local transcript ({len(self.payload.get('segments') or [])} "
              f"segments) for {asset_id[:8]}", flush=True)
        return self.payload


def chunked_build_proxies(source, asset, settings, log):  # noqa: ANN001 — e2e shim
    """Sandbox proxies: processes die at call boundaries, so the proxy build is
    chunk-resumable (a stub is enough for the acceptance render — segments cut
    from the source directly with exact_slice_args)."""
    log("  [proxy] acceptance shim: skipping proxy bundle (segments cut from source)")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--project-id", default=None)
    ap.add_argument("--render", default="1280x720")
    ap.add_argument("--budget-s", type=int, default=int(os.getenv("RUN_BUDGET_S", "480")))
    ap.add_argument("--force-replan", action="store_true")
    args = ap.parse_args()

    settings = get_settings()

    src = Path(args.source).resolve()
    if not src.exists():
        raise SystemExit(f"source missing: {src}")
    info = probe(src)
    print(
        f"[source] {src.name}: {info.duration_s / 60:.1f} min {info.width}x{info.height}"
        f"@{info.fps:.2f} ({src.stat().st_size / 1e6:.0f} MB) — budget {args.budget_s}s",
        flush=True,
    )

    payload = json.loads(Path(args.transcript).read_text())
    transcribe = LocalTranscript(payload)

    # the acceptance laws ride env BEFORE make_deps reads them
    os.environ.setdefault("AUDIO_MODE", "loudnorm_48k")  # no neural enhancer in sandbox
    os.environ.setdefault("POST_FX", "1")  # owner rule: post-FX on the DELIVERABLE only
    os.environ.setdefault("BLUR_MODE", "selective")  # primary subject visible
    os.environ.setdefault("TTS_RATE", "+14%")  # retention-law pace (Brian)
    os.environ.setdefault("DIRECTOR_AUTODRIVE", "")  # story-map session owns >= 8 min

    w, h = (int(x) for x in args.render.lower().split("x"))
    deps: PipelineDeps = make_deps(
        settings.database_url,
        transcribe=transcribe,
        video=VideoRenderSpec(w=w, h=h, fps=30, preset="veryfast", crf=23),
        log=lambda m: print(m, flush=True),
    )
    init_models(make_engine(settings.database_url))

    pid = args.project_id
    if pid:
        print(f"[project] resuming {pid}", flush=True)
    else:
        with deps.sessions() as s:
            project = ProjectRow(id=new_id(), title="Eustis FL — acceptance run (local ASR)")
            s.add(project)
            s.commit()
            pid = project.id
        with deps.sessions() as s:
            s.add(
                CaseSourceRow(
                    project_id=pid,
                    kind="text",
                    label="source title facts",
                    text=(
                        "Public-record video title: 'Bodycam: Drunk Cop Arrested While On "
                        "Duty. Eustis, FL. July 04, 2022.' Published on Wikimedia Commons "
                        "(public domain). The footage shows an on-duty police officer in "
                        "Eustis, Florida being investigated and arrested on July 4, 2022."
                    ),
                    status="pending",
                )
            )
            s.commit()
        Path(".acceptance_pid").write_text(pid)
        print(f"[project] {pid} created", flush=True)

    # sandbox: proxy build is chunk-hostile — acceptance cuts straight from source
    import director_os.worker.pipeline as pipeline_mod

    pipeline_mod.build_proxies = chunked_build_proxies

    try:
        result = run_pipeline(src, pid, deps, force_replan=args.force_replan)
    except Exception as e:  # noqa: BLE001
        print(f"[pipeline] FAILED: {type(e).__name__}: {e}", flush=True)
        raise

    print("\n================ ACCEPTANCE RESULT ================", flush=True)
    print(f"final: {result.final_path} ({result.final_path.stat().st_size / 1e6:.1f} MB)")
    print(f"render: {result.render.rendered} rendered / {result.render.cached} cached")
    for c in result.audit.checks:
        print(f"  - {c.name}: {'OK' if c.ok else 'FAIL'} — {c.detail}")
    if result.post_fx:
        pfx = result.post_fx.as_dict()
        print(
            f"post-fx: cues={pfx['n_caption_cues']} ({pfx['narration_cues']}n/{pfx['raw_cues']}r) "
            f"blur={pfx['blur_backend']} tracks={pfx['blur_tracks']} leaks={pfx['blur_leaks']} "
            f"audio={pfx['enhance_engine']} SNR {pfx['snr_before_db']}->{pfx['snr_after_db']}dB"
        )

    # ---- deliverables -> download/
    DOWNLOAD.mkdir(parents=True, exist_ok=True)
    stem = "eustis_acceptance"
    copies = {
        result.final_path: DOWNLOAD / f"{stem}_master.mp4",
        result.otio_path: DOWNLOAD / f"{stem}_timeline.otio",
    }
    out_dir = settings.output_dir / "pipeline" / pid
    for name in ("subtitles.srt", "subtitles.ass"):
        p = out_dir / name
        if p.exists():
            copies[p] = DOWNLOAD / f"{stem}_{name}"
    pfx_dir = out_dir / "post_fx"
    if pfx_dir.exists():
        ass = pfx_dir / "subtitles.ass"
        if ass.exists():
            copies[ass] = DOWNLOAD / f"{stem}_postfx_captions.ass"
    pack_dir = out_dir / "ship_pack"
    if pack_dir.exists():
        target = DOWNLOAD / f"{stem}_ship_pack"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(pack_dir, target)
        print(f"[ship] pack copied -> {target}")
    for src_p, dst_p in copies.items():
        if Path(src_p).exists():
            shutil.copyfile(src_p, dst_p)
            print(f"[ship] {dst_p.name} ({dst_p.stat().st_size / 1e6:.1f} MB)")
    print(f"[ship] project id: {pid}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
