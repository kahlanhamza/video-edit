"""One-Euro Filter — zero-lag velocity-aware smoothing for redaction boxes.

Why not EMA: an exponential moving average lags fast motion by 5-10 frames —
a redaction box trailing the face it must hide is a privacy leak. The
One-Euro filter (Casiez et al., CHI 2012) is velocity-aware: at low speed it
is heavily damped (no jitter), at high speed its cutoff opens up so the
output essentially rides the input — minimal lag exactly when a face is
moving fast (whip-pans, turns).

Each box channel (center x/y, width, height) is filtered independently.
"""

from __future__ import annotations

import math


class LowPassFilter:
    """Stateful exponential low-pass; first sample passes through unchanged."""

    __slots__ = ("_has_value", "value")

    def __init__(self) -> None:
        self.value = 0.0
        self._has_value = False

    @property
    def primed(self) -> bool:
        """True once the first sample has been absorbed."""
        return self._has_value

    def __call__(self, value: float, alpha: float) -> float:
        if not self._has_value:
            self.value = value
            self._has_value = True
        else:
            self.value = alpha * value + (1.0 - alpha) * self.value
        return self.value


class OneEuroFilter:
    """Adaptive-cutoff low-pass: `min_cutoff + beta * |dx_hat|`.

    `freq` is the signal cadence (fps for per-frame boxes). `min_cutoff`
    dominates slow motion (smaller = smoother), `beta` scales speed response
    (larger = tighter tracking at speed). For redaction, err toward tracking:
    beta too low re-introduces lag; beta too high re-introduces jitter — but
    the envelope pass in `trajectory.py` covers residual error anyway.
    """

    __slots__ = ("_x", "_dx", "beta", "d_cutoff", "freq", "min_cutoff")

    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 1.2,
        beta: float = 0.03,
        d_cutoff: float = 1.0,
    ):
        if freq <= 0:
            raise ValueError("freq must be positive")
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = LowPassFilter()
        self._dx = LowPassFilter()

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / dt if dt > 0 else 1.0
        return 1.0 / (1.0 + tau * te)

    def __call__(self, x: float, dt: float | None = None) -> float:
        """Filter one sample; `dt` in seconds (defaults to 1/freq)."""
        step = dt if dt and dt > 0 else 1.0 / self.freq
        prev = self._x.value if self._x.primed else x
        dx = (x - prev) / step
        edx = self._dx(dx, self._alpha(self.d_cutoff, step))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self._x(x, self._alpha(cutoff, step))

    def reset(self) -> None:
        self._x = LowPassFilter()
        self._dx = LowPassFilter()


def filter_series(
    values: list[float],
    *,
    freq: float = 30.0,
    min_cutoff: float = 1.2,
    beta: float = 0.03,
) -> list[float]:
    """Filter a whole signal at constant cadence (test + trajectory helper)."""
    f = OneEuroFilter(freq=freq, min_cutoff=min_cutoff, beta=beta)
    return [f(v) for v in values]
