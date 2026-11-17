"""Night-rescue + multi-cam exposure matching — subtle, realistic, no stylization.

Dark bodycam footage must be LEGIBLE, not "graded": a gentle shadow-lift /
gamma normalization so viewers can actually see what the officer saw, and
dashcam -> bodycam cuts must not jump in brightness. This module measures
per-clip luma (sampled decode, cheap) and proposes FFmpeg `eq` fragments:

- `night_rescue_filters`: lift ONLY when the clip is genuinely dark; tiny
  gamma bump + shadow lift, saturation untouched (no stylized teal-orange).
- `match_exposure_filters`: per-clip gamma that pulls each clip's median
  luma toward the SET median so cross-cam cuts stop flashing.

Both are deterministic pure functions of the measured stats — the renderer
just splices the fragments into the chain.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

# 8-bit luma thresholds (0-255)
DARK_MEDIAN = 64.0  # below this = night/interior bodycam, legibility at risk
BRIGHT_MEDIAN = 170.0  # above this = blown daylight, protect highlights instead
GAMMA_MIN = 0.8  # never more than subtle — realism is the product
GAMMA_MAX = 1.45


class LumaStats(BaseModel):
    """Sampled luma distribution of one clip."""

    median: float = Field(ge=0, le=255)
    p5: float = Field(ge=0, le=255)
    p95: float = Field(ge=0, le=255)
    n_samples: int = Field(ge=0)


class ExposureError(RuntimeError):
    pass


def luma_stats(path: str | Path, sample_every_s: float = 1.0, max_samples: int = 600) -> LumaStats:
    """Sampled median/percentile luma of a clip (one decode, stride-sampled)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ExposureError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(fps * sample_every_s)))
    values: list[float] = []
    idx = 0
    while len(values) < max_samples:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            values.append(float(np.median(gray)))
        idx += 1
    cap.release()
    if not values:
        raise ExposureError(f"no frames sampled from {path}")
    arr = np.array(values)
    return LumaStats(
        median=float(np.median(arr)),
        p5=float(np.percentile(arr, 5)),
        p95=float(np.percentile(arr, 95)),
        n_samples=len(values),
    )


def gamma_for_lift(median: float, target: float) -> float:
    """Gamma that moves `median` toward `target` (power law, clamped subtle).

    eq=gamma maps v -> v^g: g < 1 LIFTS shadows (brightens midtones),
    g > 1 DARKENS. Lift = target above median => g below 1.
    """
    if median <= 1 or target <= 1:
        return 1.0
    g = float(np.log(max(1e-3, target / 255.0)) / np.log(max(1e-3, median / 255.0)))
    return round(max(GAMMA_MIN, min(GAMMA_MAX, g)), 3)


def night_rescue_filters(stats: LumaStats, target_median: float = 78.0) -> list[str]:
    """Subtle legibility lift for dark bodycam footage (no saturation moves).

    Dark clip -> small gamma lift (g < 1) + gentle shadow emphasis via curves
    anchor. Bright/blown clip -> gamma UP to protect highlights. In-range
    clips get NOTHING — the default is to leave realistic footage alone.
    """
    if stats.n_samples == 0:
        return []
    if stats.median >= DARK_MEDIAN and stats.median <= BRIGHT_MEDIAN:
        return []  # legible already — do not stylize realistic footage
    if stats.median < DARK_MEDIAN:
        g = gamma_for_lift(stats.median, min(target_median, DARK_MEDIAN + 14))
        if g >= 0.999:  # clamp already refused the lift — nothing subtle left
            return []
        out = [f"eq=gamma={g:g}"]
        if stats.p5 < 12:  # crushed blacks: lift the shadow floor slightly
            out.append("curves=master='0/0.02 0.25/0.27 0.5/0.5 1/1'")
        return out
    # blown daylight: pull down a touch (g > 1)
    g = gamma_for_lift(stats.median, BRIGHT_MEDIAN - 12)
    return [f"eq=gamma={g:g}"] if g > 1.001 else []


def match_exposure_filters(
    stats: LumaStats,
    set_median: float,
) -> list[str]:
    """Gamma that pulls one clip toward the multi-cam set's median luma.

    `set_median` is the median of the per-clip medians (robust against one
    wildly dark cam). Only meaningful deltas (>4 luma) emit a filter, and the
    clamp keeps every correction subtle.
    """
    if stats.n_samples == 0 or set_median <= 0:
        return []
    delta = set_median - stats.median
    if abs(delta) <= 4.0:
        return []
    g = gamma_for_lift(stats.median, set_median)
    if abs(g - 1.0) < 0.01:
        return []
    return [f"eq=gamma={g:g}"]


def set_median_of(stats_list: list[LumaStats]) -> float:
    """Median of per-clip medians (the multi-cam target)."""
    meds = sorted(s.median for s in stats_list if s.n_samples > 0)
    if not meds:
        return 0.0
    return float(np.median(meds))
