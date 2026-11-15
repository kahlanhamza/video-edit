"""Interpolation providers (ADR-0011, ADR-0012).

Slowed content MUST NOT be frame-duplicated: v0.1.0 measured 54-98%
near-identical consecutive frames in replay sections (owner: "laggy").
Providers, in adoption order:

  framerate  ffmpeg blend interpolation — measured 2% duplicates on the
             same window, ~3x realtime in this sandbox. Default.
  rife       Practical-RIFE (MIT) via onnxruntime — phase 2 (issue #8
             backlog): needs an ONNX export + CPU throughput benchmark;
             no torch on this box (py3.12, 3GB RAM, 2 cores).

The renderer asks this module for the filter chain so swapping providers
is a config change, not a renderer rewrite.
"""
from . import config


def slow_chain(speed, fps=None):
    """Video filter chain that slows a CFR stream to `speed`x with
    interpolated output frames at `fps`."""
    fps = fps or config.FPS
    if speed >= 0.999:
        return f"fps={fps}"                       # real-time: no interpolation
    provider = getattr(config, "INTERP_PROVIDER", "framerate")
    if provider == "rife":
        raise NotImplementedError(
            "RIFE provider is phase-2 (issue #8 backlog): needs "
            "onnxruntime benchmark; framerate provider ships today")
    if provider != "framerate":
        raise ValueError(f"unknown INTERP_PROVIDER: {provider}")
    return (f"setpts=PTS/{speed:.4f},"
            f"framerate=fps={fps}:interp_start=0:interp_end=255:scene=100")
