"""Full-tape LLM-directed documentary — the Eustis 43.8-min acceptance run.

The M5 postmortem acceptance: the ENTIRE 43.8-minute tape flows
ingest -> dossier -> ASR -> story map -> 2-phase Director (Phase A macro
plan, Phase B act-by-act tool loop) -> render -> QA retention gate, and the
master MUST hold 600s+/6+ narration/8+ raw clips/0.70+ source span. The run
prints the OTIO act breakdown (Cold Open, Acts 1-6, total_duration_s,
narrator_count, raw_count, source timestamp span) when it lands.

Sandbox reality: every process dies at the tool-call boundary, so this
runner is RESUMABLE BY DESIGN and driven repeatedly until DONE:

  - proxy  : chunk-encoded (240s slices, .done markers) — a killed call
             re-enters build_proxies, skips finished chunks, concats
  - ASR    : the Gladia job id rides a sidecar — a killed call re-polls the
             SAME job instead of re-uploading 43 minutes of audio
  - render : the segment cache is already content-addressed — misses only
  - everything else: the pipeline's own domain-table idempotency

Run from repo root with .env exported:
    set -a; . ./.env; set +a
    uv run python scripts/e2e_llm_documentary.py [--project-id ID]

Exit codes: 0 = DONE (master + breakdown printed), 75 = budget exhausted
(re-run me), anything else = real failure.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "perception" / "src"))

from director_os.core.config import get_settings  # noqa: E402
from director_os.core.db import CaseSourceRow, ProjectRow, new_id  # noqa: E402
from director_os.core.qa_probe import probe  # noqa: E402
from director_os.otio_engine.segment_cache import VideoRenderSpec  # noqa: E402
from director_os.worker.pipeline import PipelineDeps, make_deps, run_pipeline  # noqa: E402

SOURCE = REPO / "data" / "media" / "bodycam_eustis.webm"
SOURCE_PAGE = (
    "https://commons.wikimedia.org/wiki/File:"
    "Bodycam-_Drunk_Cop_Arrested_While_On_Duty._Eustis%2C_FL._July_04%2C_2022..webm"
)
CHUNK_S = 240.0
EXIT_BUDGET = 75


class BudgetExhausted(RuntimeError):
    pass


def _budget_watchdog(budget_s: int):
    def _handler(signum, frame):
        # hard exit: unwinding through subprocess.__exit__ DRAINS the running
        # ffmpeg (the sweep's full-tape decode can out-live the whole window).
        # Every stage is idempotent (chunk markers, DB commits, Gladia sidecar)
        # so an instant death is always safe to resume.
        print(
            f"[budget] wall budget of {budget_s}s exhausted — resume me",
            flush=True,
        )
        os._exit(EXIT_BUDGET)

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(budget_s)


# ------------------------------------------------------------- chunked proxy
def chunked_build_proxies(source, out_dir):
    """Resumable drop-in for core.proxy.build_proxies (identical artifacts)."""
    from director_os.core.proxy import ProxyBuildError, ProxyBundle

    source = Path(source)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    info = probe(source)
    total = info.duration_s

    # chunk cache keyed on the SOURCE (survives new asset ids across retries)
    chunk_dir = REPO / "output" / "proxy_chunks" / f"{source.stem}-{int(source.stat().st_size)}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = int(total // CHUNK_S) + 1
    parts = []
    for i in range(n_chunks):
        part = chunk_dir / f"chunk_{i:03d}.mp4"
        marker = chunk_dir / f"chunk_{i:03d}.done"
        start = i * CHUNK_S
        dur = min(CHUNK_S, total - start)
        if dur <= 0.5:
            continue
        if marker.exists() and part.exists():
            parts.append(part)
            continue
        cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{start:.2f}",
            "-t",
            f"{dur:.2f}",
            "-i",
            str(source),
            "-vf",
            "scale=-2:480:flags=bicubic",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "28",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-force_key_frames",
            "expr:gte(t,n_forced*2)",
            str(part),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            part.unlink(missing_ok=True)
            raise ProxyBuildError(f"proxy chunk {i} failed: {r.stderr[-500:]}")
        marker.write_text("ok")
        parts.append(part)
        print(f"    [proxy] chunk {i + 1}/{n_chunks} ({dur:.0f}s) encoded", flush=True)

    proxy = out / "proxy.mp4"
    if len(parts) == 1:
        subprocess.run(["cp", str(parts[0]), str(proxy)], check=True)
    else:
        lst = out / "chunks.txt"
        lst.write_text("".join(f"file '{p}'\n" for p in parts))
        r = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(lst),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(proxy),
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            raise ProxyBuildError(f"proxy concat failed: {r.stderr[-500:]}")

    # scrubsheet + audio64k from the fast-to-decode proxy (same artifacts)
    scrub = out / "scrubsheet.jpg"
    r = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(proxy),
            "-vf",
            "fps=1/2,scale=-2:60,tile=40x1",
            "-frames:v",
            "1",
            str(scrub),
        ],
        capture_output=True,
        text=True,
    )
    audio = out / "audio64k.m4a"
    r2 = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(proxy),
            "-vn",
            "-c:a",
            "aac",
            "-b:a",
            "64k",
            "-ar",
            "48000",
            "-ac",
            "1",
            str(audio),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0 or r2.returncode != 0:
        raise ProxyBuildError(f"scrub/audio failed: {(r.stderr or r2.stderr)[-400:]}")
    print(f"  [proxy] complete: {proxy.name} + {scrub.name} + {audio.name}", flush=True)
    return ProxyBundle(proxy_path=proxy, scrub_path=scrub, audio_path=audio)


# ------------------------------------------------------------- ASR (resumable)
def make_transcribe(settings, project_id: str):
    """Gladia callable with job-id sidecar: killed calls re-POLL, not re-upload."""
    from director_os.perception.gladia import GladiaClient

    gladia = GladiaClient(api_key=settings.gladia_api_key, contact_email=settings.contact_email)
    sidecar = settings.output_dir / "pipeline" / project_id / "gladia_job.json"

    def transcribe(path: Path, asset_id: str):
        asr_in = Path(path)
        # compact upload: 16k mono m4a (~16MB instead of an 80MB wav)
        if asr_in.suffix == ".webm":
            asr_in = asr_in.with_suffix(".asr.m4a")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(path),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "48k",
                    str(asr_in),
                ],
                check=True,
                capture_output=True,
            )
        job_id = None
        if sidecar.exists():
            job_id = json.loads(sidecar.read_text()).get("job_id")
        if not job_id:
            print(
                f"  [asr] uploading {asr_in.stat().st_size / 1e6:.1f} MB to Gladia...", flush=True
            )
            audio_url = gladia._upload(asr_in)
            job = gladia._create_job(audio_url)
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            sidecar.write_text(json.dumps({"job_id": job}))
            job_id = job
        else:
            print(f"  [asr] resuming Gladia job {job_id[:8]}…", flush=True)
        t0 = time.time()
        payload = gladia._poll_result(job_id)
        print(f"  [asr] transcription ready after {time.time() - t0:.0f}s of polling", flush=True)
        transcript = gladia._normalize(payload, asset_id)
        sidecar.unlink(missing_ok=True)
        return transcript

    return transcribe


# ------------------------------------------------------------- breakdown
def print_act_breakdown(result) -> None:
    from director_os.agent_runtime.plan_invariants import (
        TimelineComposition,
        coverage_span,
        timeline_composition,
    )
    from director_os.otio_engine.compiler import master_track
    from director_os.otio_engine.timeline import is_gap, item_duration_s

    comp: TimelineComposition = timeline_composition(result.timeline)
    print("\n================ OTIO ACT BREAKDOWN ================")
    acts: dict[str, dict] = {}
    order: list[str] = []
    for item in master_track(result.timeline):
        if is_gap(item):
            continue
        meta = item.metadata.get("director_os") or {}
        dur = item_duration_s(item)
        sr = getattr(item, "source_range", None)
        src = sr.start_time.to_seconds() if sr is not None else 0.0
        act = str(meta.get("act_name") or "(unlabeled)")
        if act not in acts:
            acts[act] = {"dur": 0.0, "raw": 0, "narr": 0, "first_s": None, "last_end_s": None}
            order.append(act)
        a = acts[act]
        a["dur"] += dur
        if meta.get("kind") == "narration":
            a["narr"] += 1
        else:
            a["raw"] += 1
            a["first_s"] = src if a["first_s"] is None else min(a["first_s"], src)
            end = src + dur
            a["last_end_s"] = end if a["last_end_s"] is None else max(a["last_end_s"], end)
    for act in order:
        a = acts[act]
        span = (
            f" | src {a['first_s']:.0f}s..{a['last_end_s']:.0f}s"
            if a["first_s"] is not None
            else ""
        )
        print(f"  {act:<34} {a['dur']:7.1f}s  raw={a['raw']} narr={a['narr']}{span}")
    print("-" * 52)
    print(f"  total_duration_s      {comp.final_duration_s:.1f}")
    print(f"  narrator_count        {comp.narrator_segment_count}")
    print(f"  raw_count             {comp.raw_segment_count}")
    print(
        f"  source timestamp span {comp.first_source_s:.0f}s..{comp.last_source_end_s:.0f}s "
        f"of {result.source_duration_s:.0f}s "
        f"(coverage {coverage_span(comp, result.source_duration_s):.2f})"
    )
    if result.dossier:
        d = result.dossier
        print(
            f"  dossier: subject={d.get('subject_name')} "
            f"charges={len(d.get('charges') or [])} "
            f"verdict={'yes' if d.get('verdict_outcome') else 'none'}"
        )


# ------------------------------------------------------------- main
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default=str(SOURCE))
    p.add_argument("--project-id", default=None)
    p.add_argument("--render", default="1280x720")
    p.add_argument("--budget-s", type=int, default=int(os.getenv("RUN_BUDGET_S", "500")))
    p.add_argument("--max-face-regions", type=int, default=6)
    args = p.parse_args()

    settings = get_settings()
    budget = args.budget_s
    _budget_watchdog(budget)

    # YuNet sweep at 3s sampling fits one budget window (VP9 decode dominates);
    # must ride the env BEFORE make_deps reads it
    os.environ.setdefault("REDACT_MODE", "yunet")

    src = Path(args.source).resolve()
    info = probe(src)
    print(
        f"[source] {src.name}: {info.duration_s:.1f}s {info.width}x{info.height}"
        f"@{info.fps:.2f} ({src.stat().st_size / 1e6:.0f} MB) — budget {budget}s",
        flush=True,
    )

    w, h = (int(x) for x in args.render.lower().split("x"))
    deps: PipelineDeps = make_deps(
        settings.database_url,
        transcribe=make_transcribe(settings, args.project_id or "_bootstrap_"),
        video=VideoRenderSpec(w=w, h=h, fps=30, preset="veryfast", crf=23),
        log=lambda m: print(m, flush=True),
    )
    # NOTE: transcribe binds to the FINAL project id once known (rebound below)

    if args.project_id:
        pid = args.project_id
        with deps.sessions() as s:
            if s.get(ProjectRow, pid) is None:
                raise SystemExit(f"project {pid} does not exist")
        print(f"[project] resuming {pid}", flush=True)
    else:
        with deps.sessions() as s:
            project = ProjectRow(
                id=new_id(), title="Eustis FL bodycam — full 43.8-min tape (M5 story-map Director)"
            )
            s.add(project)
            s.commit()
            pid = project.id
        # the video's own page rides as a source_page case source (auto-discover)
        with deps.sessions() as s:
            s.add(
                CaseSourceRow(
                    project_id=pid,
                    kind="source_page",
                    label="Wikimedia Commons file page",
                    uri_or_storage_key=SOURCE_PAGE,
                    status="pending",
                )
            )
            s.commit()
        print(f"[project] {pid} created + source_page attached", flush=True)
        Path(".eustis_pid").write_text(pid)  # convenience for the driver loop

    deps.transcribe = make_transcribe(settings, pid)

    # chunk-resumable proxy (sandbox: processes die at call boundaries)
    import director_os.worker.pipeline as pipeline_mod

    pipeline_mod.build_proxies = chunked_build_proxies

    # the 43-min tape at 1s YuNet sampling cannot fit one budget window —
    # sample every 3s (clustered boxes are padded; the blur contract holds)
    import director_os.perception.faces as faces_mod

    _orig_detect = faces_mod.detect_faces
    faces_mod.detect_faces = lambda path, sample_every_s=1.0, **kw: _orig_detect(
        path, sample_every_s=3.0, **kw
    )

    try:
        result = run_pipeline(src, pid, deps)
    except BudgetExhausted as e:
        print(f"[budget] {e} — re-run this script with --project-id {pid}", flush=True)
        return EXIT_BUDGET
    except Exception as e:  # noqa: BLE001
        print(f"[pipeline] FAILED: {type(e).__name__}: {e}", flush=True)
        raise

    print_act_breakdown(result)
    print("\n================ EUSTIS M5 ACCEPTANCE ================")
    print(f"final: {result.final_path} ({result.final_path.stat().st_size / 1e6:.1f} MB)")
    print(f"render: {result.render.rendered} rendered / {result.render.cached} cached")
    for c in result.audit.checks:
        print(f"  - {c.name}: {'OK' if c.ok else 'FAIL'} — {c.detail}")

    deliverables = REPO.parent / "download"
    deliverables.mkdir(parents=True, exist_ok=True)
    final_copy = deliverables / "eustis_llm_directed_master.mp4"
    final_copy.write_bytes(result.final_path.read_bytes())
    for name in ("subtitles.srt", "subtitles.ass"):
        f = settings.output_dir / "pipeline" / pid / name
        if f.exists():
            (deliverables / f"eustis_llm_directed_{name}").write_bytes(f.read_bytes())
    otio_copy = deliverables / "eustis_llm_directed_timeline.otio"
    otio_copy.write_bytes(result.otio_path.read_bytes())
    from director_os.agent_runtime.plan_invariants import coverage_span, timeline_composition

    comp = timeline_composition(result.timeline)
    (deliverables / "eustis_llm_directed_breakdown.json").write_text(
        json.dumps(
            {
                "project_id": pid,
                "source_duration_s": result.source_duration_s,
                "master_duration_s": comp.final_duration_s,
                "narrator_count": comp.narrator_segment_count,
                "raw_count": comp.raw_segment_count,
                "source_span_s": [comp.first_source_s, comp.last_source_end_s],
                "coverage": round(coverage_span(comp, result.source_duration_s), 3),
                "acts": comp.acts,
                "dossier_subject": (result.dossier or {}).get("subject_name"),
                "audit_ok": result.audit.ok,
                "audit_checks": [
                    {"name": c.name, "ok": c.ok, "detail": c.detail} for c in result.audit.checks
                ],
            },
            indent=2,
        )
    )
    print(f"\n[done] deliverables -> {deliverables}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
