"""Parallel segment rendering (issue #9 close-out).

- parallel render == serial render (same keys, same concat plan, QA-clean final)
- misses actually encode on multiple worker threads
- fleet tasks re-verify the segment key before touching the cache
- concat task fails loudly when chunks are missing
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from director_os.otio_engine.compiler import compile_timeline
from director_os.otio_engine.contract import RawClipMeta
from director_os.otio_engine.renderer import render_dag
from director_os.otio_engine.segment_cache import (
    AudioRenderSpec,
    SegmentCache,
    VideoRenderSpec,
)
from director_os.otio_engine.timeline import (
    ClipSpec,
    TimelineSpec,
    TrackSpec,
    build_timeline,
)


def three_clip_spec() -> TimelineSpec:
    return TimelineSpec(
        name="parallel_case",
        fps=30,
        width=640,
        height=360,
        tracks=[
            TrackSpec(
                name="V1_master",
                items=[
                    ClipSpec(
                        name=f"seg{i}",
                        source_start_s=float(i * 3),
                        dur_s=3.0,
                        meta=RawClipMeta(asset_id="a1"),
                    )
                    for i in range(3)
                ],
            )
        ],
    )


def compile_tree(tmp_path: Path, src: Path):
    tl = build_timeline(three_clip_spec())
    cache = SegmentCache(tmp_path / "segments")
    dag = compile_timeline(
        tl,
        assets={"a1": str(src)},
        cache=cache,
        video=VideoRenderSpec(w=320, h=240, fps=30, preset="ultrafast", crf=32),
        audio=AudioRenderSpec(),
        out_dir=tmp_path / "render",
    )
    return cache, dag


def test_parallel_matches_serial_plan(fixture_video: Path, tmp_path):
    cache_s, dag_s = compile_tree(tmp_path / "serial", fixture_video)
    cache_p, dag_p = compile_tree(tmp_path / "parallel", fixture_video)
    assert [s.key for s in dag_s.segments] == [s.key for s in dag_p.segments]

    rep_serial = render_dag(dag_s, cache_s, log=lambda *_: None, max_workers=1)
    rep_parallel = render_dag(dag_p, cache_p, log=lambda *_: None, max_workers=3)
    assert rep_parallel.rendered == rep_serial.rendered == 3
    assert rep_parallel.cached == rep_serial.cached == 0

    import subprocess

    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(rep_parallel.final_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert float(out.stdout.strip()) == pytest.approx(9.0, abs=0.9)


def test_parallel_encodes_on_multiple_threads(fixture_video: Path, tmp_path, monkeypatch):
    cache, dag = compile_tree(tmp_path, fixture_video)
    seen_threads: set[int] = set()

    from director_os.otio_engine import renderer as renderer_mod

    real = renderer_mod.render_segment

    def counting_render(spec, src_path, out_path):
        seen_threads.add(threading.get_ident())
        real(spec, src_path, out_path)

    monkeypatch.setattr(renderer_mod, "render_segment", counting_render)
    render_dag(dag, cache, log=lambda *_: None, max_workers=3)
    assert len(seen_threads) >= 2, "cache misses must encode on a worker pool, not serially"


def test_worker_segment_task_rejects_key_mismatch(fixture_video: Path, tmp_path):
    from director_os.worker.tasks import render_segment_task

    cache, dag = compile_tree(tmp_path, fixture_video)
    seg = dag.segments[0]
    spec_json = seg.spec.model_dump(mode="json")
    with pytest.raises(ValueError, match="segment key mismatch"):
        render_segment_task.run(spec_json, seg.src_path, "deadbeef" * 5)


def test_worker_segment_task_renders_into_cache(fixture_video: Path, tmp_path, monkeypatch):
    from director_os.worker import tasks as worker_tasks

    monkeypatch.setattr(worker_tasks, "get_segments_dir", lambda: tmp_path / "segments")
    cache, dag = compile_tree(tmp_path, fixture_video)
    seg = dag.segments[1]
    result = worker_tasks.render_segment_task.run(
        seg.spec.model_dump(mode="json"), seg.src_path, seg.key
    )
    assert result["key"] == seg.key
    assert result["bytes"] > 50_000
    assert cache.get(seg.key) is not None


def test_worker_concat_task_fails_on_missing_chunk(tmp_path, monkeypatch):
    from director_os.worker import tasks as worker_tasks

    monkeypatch.setattr(worker_tasks, "get_segments_dir", lambda: tmp_path / "segments")
    with pytest.raises(FileNotFoundError, match="missing from cache"):
        worker_tasks.concat_task.run(["nope1", "nope2"], str(tmp_path / "final.mp4"))


def test_worker_concat_task_produces_final(tmp_path, monkeypatch):
    from director_os.worker import tasks as worker_tasks

    monkeypatch.setattr(worker_tasks, "get_segments_dir", lambda: tmp_path / "segments")
    seg_dir = tmp_path / "segments"
    seg_dir.mkdir()
    chunk_a = seg_dir / "aaaa.mp4"
    chunk_b = seg_dir / "bbbb.mp4"
    # two tiny uniform mp4s (same codec params) so -c copy concat is legal
    for name, dur in (("aaaa.mp4", 1.0), ("bbbb.mp4", 1.0)):
        subprocess_ffmpeg(seg_dir / name, dur)

    out = worker_tasks.concat_task.run(["aaaa", "bbbb"], str(tmp_path / "final.mp4"))
    assert out["segments"] == 2
    assert Path(out["final"]).exists()
    assert chunk_a.exists() and chunk_b.exists()


def subprocess_ffmpeg(out: Path, dur: float) -> None:
    import subprocess

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=320x240:rate=30:duration={dur}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "32",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "30",
            "-an",
            str(out),
        ],
        check=True,
        capture_output=True,
    )


# ------------------------------------------------------------------ M5 drill
def test_drill_kill_mid_encode_resumes_from_cache(fixture_video: Path, tmp_path):
    """M5 crash-recovery drill (issue #13): kill a worker mid-encode, resume.

    Deterministic simulation of the drill: segment A admitted to the cache
    legitimately, the worker dies leaving a half-written B.tmp.mp4, segment C
    never started. The resume run must (1) hit A from cache, (2) re-encode B
    OVER the partial tmp — a partial tmp must never poison admission, (3)
    render C, (4) concat all three into a QA-clean final of the right length.
    """
    cache, dag = compile_tree(tmp_path / "drill", fixture_video)
    keys = [s.key for s in dag.segments]
    assert len(keys) == 3

    # -- the crash: A done, B half-encoded, C untouched
    from director_os.otio_engine.renderer import render_one

    render_one(dag.segments[0], cache)
    garbage_tmp = cache.root / f"{keys[1]}.tmp.mp4"
    garbage_tmp.write_bytes(b"\x00" * 100_000)  # half-written encode corpse
    assert cache.count() == 1
    assert garbage_tmp.exists()

    # -- the resume
    report = render_dag(dag, cache, log=lambda *_: None, max_workers=1)
    assert report.cached == 1  # A skipped
    assert report.rendered == 2  # B + C re-encoded
    assert cache.count() == 3
    assert not garbage_tmp.exists()  # partial tmp replaced by the real encode

    # -- the final is whole: every chunk in DAG order, correct total duration
    from director_os.core.qa_probe import probe

    info = probe(dag.final_path)
    assert abs(info.duration_s - 9.0) <= 0.5
    assert info.has_video
