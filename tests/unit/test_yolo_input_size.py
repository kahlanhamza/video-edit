"""Regression: YOLO input-size reconciliation + fail-loud sweep contract.

Root cause of the real-footage tracked-redaction failure (Eustis, issue #14
follow-up): the lindevs ONNX export pins its input at 640x640, the sweep fed
960x960, onnxruntime raised INVALID_ARGUMENT, and the sweep's per-frame
`except Exception` swallowed it into an empty plan — zero redaction baked,
audit vacuously green. These tests pin the reconciliation helper and the
fail-loud contract.
"""

import numpy as np
import pytest
from director_os.vision_fx.compute.yolo_onnx import resolve_input_size


def test_fixed_model_input_wins_over_requested():
    # exactly what onnxruntime reports for yolov8n-face-lindevs.onnx
    shape = ["images", 1, 3, 640, 640]
    eff, note = resolve_input_size(shape, 960)
    assert eff == 640
    assert "fixed at 640" in note
    assert "960" in note


def test_fixed_model_matching_request_is_silent_noop():
    eff, note = resolve_input_size(["images", 1, 3, 640, 640], 640)
    assert eff == 640
    assert "honored-as" not in note


def test_dynamic_model_honors_requested():
    eff, note = resolve_input_size(["images", 1, 3, "height", "width"], 960)
    assert eff == 960
    assert "dynamic" in note


def test_short_shape_falls_back_to_requested():
    eff, _ = resolve_input_size(["images"], 960)
    assert eff == 960


def test_sweep_does_not_swallow_detector_exceptions(tmp_path_factory):
    """A detector that raises must propagate — an empty plan is a privacy hole."""
    import subprocess

    from director_os.vision_fx.compute.local_onnx import LocalOnnxCpuBackend

    class ExplodingDetector:
        name = "exploding"

        def detect(self, frame_bgr: np.ndarray):
            raise RuntimeError("INVALID_ARGUMENT: simulated shape mismatch")

    td = tmp_path_factory.mktemp("yolo")
    out = td / "tiny.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=10:duration=0.2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ],
        check=True,
    )
    backend = LocalOnnxCpuBackend(detector=ExplodingDetector(), stride=1)
    with pytest.raises(RuntimeError, match="INVALID_ARGUMENT"):
        # even frame 1 must raise, not be swallowed into an empty plan
        _ = backend.detect_faces(str(out))
