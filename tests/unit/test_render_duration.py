"""Frame-quantized duration — the concat drift regression (live-caught).

The 2627 s Eustis master rendered 329 segments of 7.9845 s each; ffmpeg
emits frames while PTS < end, so every segment became 8.000 s and the concat
ran 4.92 s long — the pipeline QA gate caught it (tolerance 1 s). Expectations
must ride the SAME quantization as ffmpeg.
"""

from __future__ import annotations

from director_os.otio_engine.compiler import render_duration_s


def test_whole_frame_slices_pass_through():
    assert render_duration_s(8.0, 30) == 8.0
    assert render_duration_s(1 / 30, 30) == 1 / 30


def test_fractional_slices_ceil_to_frames():
    # 7.9845 s @30 fps = 239.535 frames -> ffmpeg renders 240 frames = 8.0 s
    assert render_duration_s(7.9845, 30) == 8.0
    assert render_duration_s(0.016, 30) == 1 / 30  # 0.48 frame -> 1 frame
    assert render_duration_s(1.0, 30) == 1.0


def test_drift_scenario_quantizes_like_ffmpeg():
    # the live failure: 329 segments x ceil(7.9845*30)/30 = 329 x 8.0 = 2632.0
    total = sum(render_duration_s(7.9845, 30) for _ in range(329))
    assert total == 2632.0


def test_non_integer_fps_supported():
    # 23.976 fps NTSC: 1.0 s = 23.976 frames -> 24 frames
    assert render_duration_s(1.0, 23.976) == 24 / 23.976
