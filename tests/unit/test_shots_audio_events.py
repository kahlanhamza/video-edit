"""perception v1 — shot boundaries + audio events on synthetic deterministic media."""

from __future__ import annotations

import subprocess

import pytest
from director_os.perception.shots import (
    PerceptionError,
    detect_audio_events,
    detect_shots,
)

VID_W = 320
VID_H = 240
FPS = 15
HALF_S = 2.0


@pytest.fixture(scope="module")
def two_scene_video(tmp_path_factory):
    """4s video: testsrc2 (0-2s) hard-cut to smptebars (2-4s)."""
    d = tmp_path_factory.mktemp("shots")
    path = d / "two_scenes.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=duration={HALF_S}:size={VID_W}x{VID_H}:rate={FPS}",
        "-f",
        "lavfi",
        "-i",
        f"smptebars=duration={HALF_S}:size={VID_W}x{VID_H}:rate={FPS}",
        "-filter_complex",
        "[0:v][1:v]concat=n=2:v=1[outv]",
        "-map",
        "[outv]",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        str(path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return path


@pytest.fixture(scope="module")
def single_scene_video(tmp_path_factory):
    d = tmp_path_factory.mktemp("shots")
    path = d / "one_scene.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=duration=4:size={VID_W}x{VID_H}:rate={FPS}",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        str(path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return path


def _event_audio(tmp_path, kind: str) -> str:
    """1s silence + 1s 600Hz tone + 1s silence -> 3s wav."""
    path = tmp_path / f"{kind}.wav"
    expr = "if(lt(t\\,1.0)\\,0\\,if(lt(t\\,2.0)\\,sin(2*PI*600*t)\\,0))"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"aevalsrc={expr}:s=48000:d=3.0",
        "-ac",
        "2",
        "-c:a",
        "pcm_s16le",
        str(path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return str(path)


# ------------------------------------------------------------------- shots
def test_detect_shots_finds_hard_cut(two_scene_video):
    det = detect_shots(str(two_scene_video), threshold=0.2, backend="ffmpeg")
    assert det.backend == "ffmpeg-scene"
    assert len(det.cut_points) == 1
    cut = det.cut_points[0]
    assert HALF_S - 0.5 <= cut <= HALF_S + 0.5
    # normalized shot list spans the full duration without gaps
    assert det.shots[0].start_s == 0.0
    assert abs(det.shots[-1].end_s - 2 * HALF_S) < 0.3
    assert sum(s.duration_s for s in det.shots) == pytest.approx(2 * HALF_S, abs=0.3)


def test_detect_shots_single_scene_no_cut(single_scene_video):
    det = detect_shots(str(single_scene_video), threshold=0.5, backend="ffmpeg")
    assert det.cut_points == []
    assert len(det.shots) == 1


def test_detect_shots_auto_backend_falls_back_gracefully(two_scene_video):
    """backend=auto works whether or not pyscenedetect is installed."""
    det = detect_shots(str(two_scene_video), threshold=0.2, backend="auto")
    assert det.backend in ("ffmpeg-scene", "pyscenedetect")
    assert len(det.shots) >= 1


def test_detect_shots_missing_file_raises(tmp_path):
    # ffprobe runs first (check=True) and fails on the missing file
    with pytest.raises(subprocess.CalledProcessError):
        detect_shots(str(tmp_path / "nope.mp4"), backend="ffmpeg")


# ------------------------------------------------------------- audio events
def test_detect_audio_events_finds_burst(tmp_path):
    path = _event_audio(tmp_path, "burst")
    events = detect_audio_events(path, noise_db=-30.0)
    assert len(events) == 1
    ev = events[0]
    assert 0.6 <= ev.start_s <= 1.4
    assert 1.6 <= ev.end_s <= 2.4
    assert ev.kind == "sound"


def test_detect_audio_events_silence_only(tmp_path):
    path = tmp_path / "silent.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo:d=2.0",
            "-c:a",
            "pcm_s16le",
            str(path),
        ],
        capture_output=True,
        check=True,
    )
    assert detect_audio_events(str(path)) == []


def test_detect_audio_events_missing_file_raises(tmp_path):
    with pytest.raises(PerceptionError):
        detect_audio_events(str(tmp_path / "nope.wav"))
