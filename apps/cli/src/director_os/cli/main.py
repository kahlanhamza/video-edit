"""`director` — the one-command entrypoint (PLAN §8, Milestone 1 done-when).

  uv run director render <url-or-path> [--max-seconds 180]

Pipeline: ingest (UA-guarded) -> probe -> OTIO timeline -> compile (segment cache)
-> render segments -> concat -c copy -> QA probe (audio law + duration).
"""

from __future__ import annotations

from pathlib import Path

import click
from director_os.core.config import get_settings
from director_os.core.http import download
from director_os.core.qa_probe import assert_audio_law, probe
from director_os.otio_engine.compiler import compile_timeline
from director_os.otio_engine.contract import RawClipMeta
from director_os.otio_engine.renderer import render_dag
from director_os.otio_engine.segment_cache import (
    AudioRenderSpec,
    SegmentCache,
    VideoRenderSpec,
)
from director_os.otio_engine.timeline import (
    MASTER_TRACK,
    ClipSpec,
    TimelineSpec,
    TrackSpec,
    build_timeline,
    save_otio,
)


@click.group()
def main() -> None:
    """DirectorOS — autonomous documentary editing platform."""


@main.command()
def version() -> None:
    click.echo("DirectorOS 0.1.0 (M1: modular core engine + OTIO + segment caching)")


@main.command()
@click.argument("source")
@click.option(
    "--max-seconds",
    default=180.0,
    show_default=True,
    help="Trim the demo master cut to this duration.",
)
@click.option("--width", default=640, show_default=True)
@click.option("--height", default=360, show_default=True)
@click.option("--fps", default=30, show_default=True)
def render(source: str, max_seconds: float, width: int, height: int, fps: int) -> None:
    """Render a local file or URL through the M1 engine (segment-cached)."""
    settings = get_settings()
    out_dir = settings.output_dir / "render_demo"
    settings.ingest_dir.mkdir(parents=True, exist_ok=True)

    click.echo("[1/6] ingest")
    if source.startswith(("http://", "https://")):
        name = Path(source.split("?")[0]).name or "ingest.bin"
        local = download(source, settings.ingest_dir / name)
    else:
        local = Path(source)
        if not local.exists():
            raise click.ClickException(f"source not found: {source}")
        local = local.resolve()

    click.echo("[2/6] probe")
    info = probe(local)
    dur = min(info.duration_s, max_seconds)
    click.echo(f"  source: {local.name} {info.duration_s:.1f}s -> using {dur:.1f}s")

    asset_id = "asset_1"
    click.echo("[3/6] build OTIO timeline")
    spec = TimelineSpec(
        name="director_demo",
        fps=fps,
        width=width,
        height=height,
        tracks=[
            TrackSpec(
                name=MASTER_TRACK,
                items=[
                    ClipSpec(
                        name="raw_master",
                        source_start_s=0.0,
                        dur_s=dur,
                        meta=RawClipMeta(asset_id=asset_id),
                    )
                ],
            )
        ],
    )
    tl = build_timeline(spec)
    otio_path = save_otio(tl, out_dir / "demo.otio")

    click.echo("[4/6] compile (hash-based segment cache)")
    cache = SegmentCache(settings.segments_dir)
    dag = compile_timeline(
        tl,
        assets={asset_id: str(local)},
        cache=cache,
        video=VideoRenderSpec(w=width, h=height, fps=fps),
        audio=AudioRenderSpec(),
        out_dir=out_dir,
    )
    click.echo(f"  {len(dag.segments)} segment(s), {dag.total_dur_s:.1f}s total")

    click.echo("[5/6] render + concat -c copy")
    report = render_dag(dag, cache)
    click.echo(f"  rendered={report.rendered} cached={report.cached} -> {report.final_path}")

    click.echo("[6/6] QA probe (audio law + duration)")
    final_probe = probe(report.final_path)
    assert_audio_law(final_probe)
    if abs(final_probe.duration_s - dag.total_dur_s) > max(1.0, dag.total_dur_s * 0.05):
        raise click.ClickException(
            f"duration drift: got {final_probe.duration_s:.2f}s, expected {dag.total_dur_s:.2f}s"
        )
    click.echo(
        f"DONE {report.final_path} ({final_probe.duration_s:.1f}s, "
        f"{final_probe.sample_rate}Hz/{final_probe.channels}ch) | timeline: {otio_path}"
    )
