"""Compute backend selection — `COMPUTE_BACKEND` env is the single switch.

local (default) | kaggle | modal | auto. `auto` routes cheap work to CPU and
heavy work (stems, whisperx) to the free Kaggle T4 when credentials exist —
the "pay nothing, think later" default. Every branch raises
`ComputeNotAvailable` with the exact .env flips needed, never silently
degrades to a slower backend the caller didn't ask for.
"""

from __future__ import annotations

from director_os.vision_fx.compute.base import ComputeBackend, ComputeNotAvailable
from director_os.vision_fx.compute.local_onnx import LocalOnnxCpuBackend

HEAVY_KINDS = {"stems", "asr_align"}


def _mode(kind: str | None) -> str:
    from director_os.core.config import get_settings

    mode = (get_settings().compute_backend or "local").lower()
    if mode == "auto":
        s = get_settings()
        if s.kaggle_key and (kind is None or kind in HEAVY_KINDS):
            return "kaggle"
        return "local"
    if mode not in {"local", "kaggle", "modal"}:
        raise ComputeNotAvailable(f"COMPUTE_BACKEND={mode!r} is not one of local|kaggle|modal|auto")
    return mode


def get_compute_backend(kind: str | None = None) -> ComputeBackend:
    """Build the configured adapter. `kind` hints auto-routing (faces|stems|asr_align)."""
    mode = _mode(kind)
    if mode == "kaggle":
        from director_os.vision_fx.compute.kaggle_batch import KaggleBatchGpuBackend

        return KaggleBatchGpuBackend()
    if mode == "modal":
        from director_os.vision_fx.compute.modal_gpu import ModalGpuBackend

        return ModalGpuBackend()
    return LocalOnnxCpuBackend()


def available_backends() -> dict[str, dict[str, bool]]:
    """Diagnostic snapshot for logs/cost dashboards — what could run right now."""
    from director_os.core.config import get_settings

    s = get_settings()
    local = LocalOnnxCpuBackend().capabilities()
    return {
        "local-onnx-cpu": local,
        "kaggle-gpu": {"available": bool(s.kaggle_key)},
        "modal-gpu": {"available": bool(s.modal_token_id and s.modal_token_secret)},
    }
