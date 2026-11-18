"""Lazy kept-window audio enhancement — the cut-out footage is never enhanced."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from director_os.audio_engine.enhance import EnhancerResult
from director_os.core.qa_probe import probe
from director_os.worker.audio_enhance import (
    EnhanceInfo,
    merge_windows,
    stage_audio_enhance,
)


class FakeEnhancer:
    """Passes audio through, counts WHAT it was fed (whole vs windows)."""

    name = "fake-enhancer"
    atten_lim_db = 28.0

    def __init__(self):
        self.fed_durations: list[float] = []

    def enhance(self, src, dst):
        info = probe(src)
        self.fed_durations.append(round(info.duration_s, 3))
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-c:a", "copy", str(dst)],
            check=True,
            capture_output=True,
        )
        return EnhancerResult(
            engine=self.name,
            neural=True,
            atten_lim_db=self.atten_lim_db,
            in_path=Path(src),
            out_path=Path(dst),
            duration_in_s=info.duration_s,
            duration_out_s=info.duration_s,
        )


def _make_source(tmp_path: Path, dur: float = 12.0) -> Path:
    src = tmp_path / "src.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=64x64:rate=30:duration={dur}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={dur}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-shortest",
            str(src),
        ],
        check=True,
        capture_output=True,
    )
    return src


class Settings:
    def __init__(self, root: Path):
        self.output_dir = root


class Deps:
    def __init__(self, root: Path, persist_fail: bool = False):
        self.settings = Settings(root)
        self.logs: list[str] = []
        self.log = self.logs.append
        self.vocals_stem_resolver = None
        self.sessions = None
        self._persist_fail = persist_fail


def _skip_persist(monkeypatch):
    from director_os.worker import audio_enhance as ae

    monkeypatch.setattr(ae, "_persist", lambda *a, **k: None)


def test_merge_windows_sorts_and_merges():
    assert merge_windows([(4.0, 8.0), (0.0, 2.0), (1.5, 3.0)]) == [(0.0, 3.0), (4.0, 8.0)]
    assert merge_windows([]) == []
    assert merge_windows([(5.0, 5.0)]) == []  # empty window dropped


def test_lazy_enhance_runs_only_kept_windows(tmp_path, monkeypatch):
    _skip_persist(monkeypatch)
    src = _make_source(tmp_path)
    enhancer = FakeEnhancer()
    info = stage_audio_enhance(
        "a1",
        src,
        {"words": []},
        "p1",
        Deps(tmp_path),
        enhancer=enhancer,
        kept_windows=[(0.0, 4.0), (8.0, 12.0)],
    )
    # exactly the kept 8s processed; the discarded 4s (4-8s) never touched
    assert enhancer.fed_durations == [4.0, 4.0]
    assert info.kept_windows == [(0.0, 4.0), (8.0, 12.0)]
    assert info.kept_seconds == pytest.approx(8.0)
    assert info.skipped_seconds == pytest.approx(4.0)
    # full-timeline master, source-clock aligned
    assert probe(info.master_wav).duration_s == pytest.approx(12.0, abs=0.15)


def test_lazy_master_is_silent_over_dropped_ranges(tmp_path, monkeypatch):
    _skip_persist(monkeypatch)
    src = _make_source(tmp_path, dur=8.0)
    info = stage_audio_enhance(
        "a1",
        src,
        {"words": []},
        "p1",
        Deps(tmp_path),
        enhancer=FakeEnhancer(),
        kept_windows=[(0.0, 2.0)],
    )
    # the middle of the dropped range: digital silence
    r = subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-ss",
            "4.0",
            "-t",
            "1.0",
            "-i",
            str(info.master_wav),
            "-f",
            "md5",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    silence_md5 = r.stdout.strip()
    r2 = subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo",
            "-t",
            "1.0",
            "-f",
            "md5",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert silence_md5 == r2.stdout.strip(), "dropped range must be digital silence"


def test_lazy_enhance_second_run_hits_cache(tmp_path, monkeypatch):
    _skip_persist(monkeypatch)
    src = _make_source(tmp_path)
    deps = Deps(tmp_path)
    enhancer = FakeEnhancer()
    stage_audio_enhance(
        "a1",
        src,
        {"words": []},
        "p1",
        deps,
        enhancer=enhancer,
        kept_windows=[(0.0, 4.0), (8.0, 12.0)],
    )
    assert len(enhancer.fed_durations) == 2
    enhancer2 = FakeEnhancer()
    info2 = stage_audio_enhance(
        "a1",
        src,
        {"words": []},
        "p1",
        deps,
        enhancer=enhancer2,
        kept_windows=[(0.0, 4.0), (8.0, 12.0)],
    )
    assert enhancer2.fed_durations == [], "re-run with the same plan must be a full cache hit"
    assert isinstance(info2, EnhanceInfo)


def test_whole_asset_path_unchanged_when_no_windows(tmp_path, monkeypatch):
    _skip_persist(monkeypatch)
    src = _make_source(tmp_path, dur=6.0)
    enhancer = FakeEnhancer()
    info = stage_audio_enhance("a1", src, {"words": []}, "p1", Deps(tmp_path), enhancer=enhancer)
    assert enhancer.fed_durations == [pytest.approx(6.0, abs=0.05)]
    assert info.kept_windows == []
    assert info.master_wav.name == "master_2bus.wav"


def test_changed_plan_pays_only_new_windows(tmp_path, monkeypatch):
    _skip_persist(monkeypatch)
    src = _make_source(tmp_path)
    deps = Deps(tmp_path)
    stage_audio_enhance(
        "a1",
        src,
        {"words": []},
        "p1",
        deps,
        enhancer=FakeEnhancer(),
        kept_windows=[(0.0, 4.0), (8.0, 12.0)],
    )
    enhancer2 = FakeEnhancer()
    info2 = stage_audio_enhance(
        "a1",
        src,
        {"words": []},
        "p1",
        deps,
        enhancer=enhancer2,
        kept_windows=[(0.0, 4.0), (6.0, 10.0)],
    )
    # only the NEW window (6-10s) is processed; (0-4s) and (8-12) reuse cache
    assert len(enhancer2.fed_durations) == 1
    assert enhancer2.fed_durations[0] == pytest.approx(4.0)
    assert [tuple(w) for w in info2.kept_windows] == [(0.0, 4.0), (6.0, 10.0)]
