"""One-Euro filter — velocity-aware smoothing replaces EMA (issue #14 Pass 2)."""

from director_os.vision_fx.one_euro import OneEuroFilter, filter_series

FPS = 30.0


def test_steady_signal_passes_through():
    """A stationary face must not wobble: constant input -> constant output."""
    out = filter_series([100.0] * 60, freq=FPS, min_cutoff=1.2, beta=0.05)
    assert max(out) - min(out) < 0.5
    assert abs(out[-1] - 100.0) < 1.0


def test_fast_ramp_lags_less_than_ema():
    """Whip-pan speed (30 px/frame): One-Euro must ride the signal, not trail it.

    An EMA with alpha=0.3 lags by (1-a)/a ~= 2.3 frames = ~70px here. The
    One-Euro with a sane beta must stay within a fraction of a frame of lag.
    """
    raw = [100.0 + 30.0 * i for i in range(60)]
    out = filter_series(raw, freq=FPS, min_cutoff=1.2, beta=0.05)
    # warmup excluded (first samples learn the velocity)
    steady_errors = [abs(o - r) for o, r in zip(out[10:], raw[10:], strict=True)]
    assert max(steady_errors) < 15.0, f"lag too big: {max(steady_errors)} px"


def test_small_step_recovers_quickly():
    """A face stepping to a new position (turn snap) re-covers within 3 frames."""
    raw = [200.0] * 30 + [260.0] * 30
    out = filter_series(raw, freq=FPS, min_cutoff=1.2, beta=0.05)
    assert abs(out[33] - 260.0) < 12.0


def test_no_overshoot():
    """Low-pass property: output never overshoots the input trend."""
    raw = [50.0 + 10.0 * i for i in range(40)] + [450.0] * 40
    out = filter_series(raw, freq=FPS, min_cutoff=1.2, beta=0.05)
    assert max(out) <= 450.0 + 1.0


def test_reset_re_primes():
    f = OneEuroFilter(freq=FPS)
    for v in [0.0, 1.0, 2.0, 3.0]:
        f(v)
    f.reset()
    assert abs(f(100.0) - 100.0) < 0.5  # first sample after reset passes through


def test_rejects_bad_freq():
    import pytest

    with pytest.raises(ValueError):
        OneEuroFilter(freq=0)
