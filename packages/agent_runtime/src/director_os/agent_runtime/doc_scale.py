"""DocScale — ONE length-adaptive law for the documentary path.

Owner rule (2026-11-19): the project must work on ANY video length, not just
the hardcoded 43-minute Eustis acceptance tape. Every number the Director
prompt, the plan pre-flight, the deterministic fallback and the QA floors
use is derived here from `source_duration_s`, so planner and QA can never
disagree and a 6-minute or 90-minute tape plans as correctly as a 43-minute
one.

The law (kept continuous, Eustis numbers preserved exactly):

- keep ratio   : the master keeps 28-40% of the raw tape (target band),
                 floor 25% — never above 900s of master, never below 75s.
- acts         : Cold Open + 3-6 body acts, scaled to the target length.
- narration    : block count scales with target; per-block WORD bounds are
                 fixed (cold 40-65 / mid 48-85 / outro 70-105 — pro TTS at
                 ~2.6-2.9 words/sec); total narration words scale with the
                 target and cap at 650 (the AI-Studio long-form law).
- raw clips    : speech-anchored stretches, count scales with target.
- hook law     : first narration <= 8s, context complete <= 32s — binds at
                 every documentary length (a hook is not a long-video rule).
"""

from __future__ import annotations

from dataclasses import dataclass

# documentary treatment applies from 8 minutes of raw (below: v1 auto-cut)
DOCUMENTARY_MIN_SOURCE_S = 480.0
# absolute master clamps (a 6-minute tape still earns a real doc, ~100s+)
MASTER_MIN_S = 75.0
MASTER_MAX_S = 900.0
# planning word pace (conservative; the post-TTS retime pass replaces the
# estimate with the real clip duration)
PLAN_WPS = 2.6
# per-block narration word bounds (fixed — pro pacing at any doc length)
WORDS_COLD = (40, 65)
WORDS_MID = (48, 85)
WORDS_OUTRO = (70, 105)
# hook law (universal)
HOOK_MAX_START_S = 8.0
HOOK_MAX_END_S = 32.0
STING_MAX_S = 4.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _clamp_i(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


@dataclass(frozen=True)
class DocScale:
    """Every length-derived planning/QA number for one source tape."""

    source_duration_s: float
    target_lo: float
    target_hi: float
    min_final_s: float
    act_min: int
    act_max: int
    narr_lo: int
    narr_hi: int
    raw_min: int
    raw_max: int
    clip_max_s: float
    words_total_min: int
    min_coverage_span: float
    hook_law: bool

    @property
    def is_documentary(self) -> bool:
        return self.source_duration_s >= DOCUMENTARY_MIN_SOURCE_S

    @property
    def word_bounds_mid(self) -> tuple[int, int]:
        return WORDS_MID

    @property
    def word_bounds_cold(self) -> tuple[int, int]:
        return WORDS_COLD

    @property
    def word_bounds_outro(self) -> tuple[int, int]:
        return WORDS_OUTRO


def doc_scale(source_duration_s: float) -> DocScale:
    """Derive the whole planning law from one number: the raw tape length."""
    src = max(0.0, float(source_duration_s))
    if src < DOCUMENTARY_MIN_SOURCE_S:
        # non-documentary sources: floors stay generous, hook law off
        return DocScale(
            source_duration_s=src,
            target_lo=_clamp(src * 0.85, 30.0, MASTER_MAX_S),
            target_hi=_clamp(src * 0.95, 40.0, MASTER_MAX_S),
            min_final_s=_clamp(src * 0.5, 20.0, 600.0),
            act_min=1,
            act_max=6,
            narr_lo=1,
            narr_hi=14,
            raw_min=1,
            raw_max=40,
            clip_max_s=8.0,
            words_total_min=0,
            min_coverage_span=0.0,
            hook_law=False,
        )
    # --- keep-ratio law: 28-40% of the raw tape, clamped to sane absolutes
    target_lo = _clamp(src * 0.28, MASTER_MIN_S, MASTER_MAX_S)
    target_hi = _clamp(src * 0.35, MASTER_MIN_S + 15.0, MASTER_MAX_S)
    min_final = _clamp(src * 0.25, 60.0, 600.0)
    act_max = 6
    act_min = _clamp_i(round(target_lo / 100), 3, act_max)
    narr_lo = _clamp_i(round(target_lo / 70), 4, 10)
    narr_hi = _clamp_i(round(target_hi / 60), narr_lo + 2, 14)
    raw_min = _clamp_i(round(target_lo / 75), 3, 8)
    raw_max = _clamp_i(raw_min + 10, 10, 18)
    clip_max_s = _clamp(round(target_lo / 12), 25.0, 55.0)
    words_total_min = _clamp_i(round(target_hi * 0.30 * PLAN_WPS), 100, 650)
    span = 0.70 if src >= 1200.0 else 0.55
    return DocScale(
        source_duration_s=src,
        target_lo=target_lo,
        target_hi=target_hi,
        min_final_s=min_final,
        act_min=act_min,
        act_max=act_max,
        narr_lo=narr_lo,
        narr_hi=narr_hi,
        raw_min=raw_min,
        raw_max=raw_max,
        clip_max_s=clip_max_s,
        words_total_min=words_total_min,
        min_coverage_span=span,
        hook_law=True,
    )
