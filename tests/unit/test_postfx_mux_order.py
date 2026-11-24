"""Post-fx burn order — blur FIRST, captions OVER the blur.

The owner's report: "the blur tracking step and watermark blur is coming after
the captions so this is a bug this might blur the captions." The mux must take
the BLURRED video as its input and burn the .ass captions on top — a caption
never sits under a blur window.
"""

from __future__ import annotations

from pathlib import Path

from director_os.worker import post_fx


def test_mux_burns_captions_over_blurred_video(monkeypatch):
    captured: list[list[str]] = []

    def fake_run(cmd, what):
        captured.append(cmd)

    monkeypatch.setattr(post_fx, "_run", fake_run)
    post_fx._mux_pass(
        Path("/work/master_blurred.mp4"),
        Path("/work/enhanced.wav"),
        Path("/work/subtitles.ass"),
        Path("/work/final.mp4"),
    )
    assert captured, "mux must run ffmpeg"
    cmd = captured[0]
    joined = " ".join(cmd)
    # the blurred master is the VIDEO input
    assert "master_blurred.mp4" in joined
    # captions burn as a -vf ASS filter over that video (never before blur)
    assert "-vf" in cmd and any("ass=" in part for part in cmd)
    # the ass filter rides the video filter chain, not a second video input
    assert cmd[cmd.index("-vf") + 1].startswith("ass=")


def test_captions_and_blur_markers_are_independent_passes():
    """A killed blur must not lose the caption build (resumable passes)."""
    names = [name for name in ("captions", "blur", "audio", "final", "cfr", "sweep")]
    work = Path("/tmp/unused")
    markers = post_fx._markers(work)
    for name in names:
        assert name in markers
