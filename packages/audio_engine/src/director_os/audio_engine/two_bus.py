"""Broadcast Bus-1 Speech Isolation mix — raw bodycam clips (PLAN §4.5 M4 law).

Why plain loudnorm fails on bodycam: it integrates EVERYTHING — highway
traffic, wind gusts, cruiser engines — so the officer's voice arrives buried
in a "normalized" wall of noise. The pro fix is voice isolation:

    BUS 1 (voice)      neural speech extraction -> intelligibility EQ (mud
                       cut + clarity/presence/air timbre restoration) ->
                       compression -> speech-gated -14 dB duck in gaps
    BUS 2 (ambience)   OPT-IN tamed copy of the raw audio (150 Hz-4 kHz band,
                       -24 dB under the voice) so pauses keep outdoor air
                       DISABLED BY DEFAULT (AUDIO_BUS2_VOLUME=0): the 2026-11-18
                       ear verdict on the real Eustis harvest preferred the
                       pure-voice master — the raw-derived bed re-inflates the
                       exact noise the extractor removed (user: sample B
                       "clear speakers and no noises at all" vs sample C
                       "clear speakers and noise"). Set AUDIO_BUS2_VOLUME=0.06
                       to restore the outdoor-air bed.
    MASTER             Bus1 (+ opt-in Bus2) -> speech-gated gain (-12..-6 dBFS
                       window, target -11) -> -1.5 dBTP limiter -> 20 ms
                       anti-click micro-fades -> 48 kHz stereo lock

Chain of custody (per raw asset, in `stage_audio_enhance`):
  1. raw_48k.wav      full source audio, 48 kHz stereo (SNR "raw" reference)
  2. preclean.wav     highpass=f=90:poles=2 — vest rubs / engine rumble must
                      not overload the neural model (render audio only; ASR
                      keeps its raw 85 Hz-only decode)
  3. enhanced.wav     neural enhancer (MossFormer2_SE_48K / deep-filter
                      --atten-lim-db 28); optional Demucs vocals stem rides
                      here first when music/siren is flagged
  4. bus1_gated.wav   EQ + compressor in ffmpeg, then the numpy gate envelope
                      (Silero VAD + ASR words, 150 ms attack/release ramps)
  5. master_2bus.wav  the single ffmpeg mix (this module builds the cmd)

The segment renderer then slices master_2bus.wav with the exact same window
as the video and adds the 20 ms anti-click micro-fades at the cut seams.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ constants
RUMBLE_HIGHPASS_HZ = 90  # pre-ML wind-thump/rumble filter (steep, 2 poles)
DUCK_DB = -14.0  # Bus-1 non-speech attenuation
RAMP_S = 0.15  # gate attack/release ramps
BUS2_HIGHPASS_HZ = 150
BUS2_LOWPASS_HZ = 4000
# Bus-2 ambience bed level, OPT-IN (default OFF — see module docstring for the
# 2026-11-18 ear verdict). One .env line re-enables the outdoor-air bed.
BUS2_VOLUME_ENV = "AUDIO_BUS2_VOLUME"
BUS2_VOLUME_DEFAULT = 0.0
BUS2_LIMITER_LINEAR = 0.0625
# Timbre restoration law (measured, Nov 2026 — scripts/measure_tilt.py on the
# real MossFormer2_SE_48K Kaggle harvest): neural extractors preserve low-mid
# most and shave the top end hardest — presence 2-5 kHz lands -19.9 dB and air
# 5-12 kHz -23.9 dB below the 500-2k mid band (raw: -14.3/-14.3). Perceptually
# that is exactly "the voices sound deep": consonants dull, no sparkle, bass
# dominant. The Bus-1 chain therefore RESTORES the top end on top of the mud
# control instead of only de-mudding.
MUD_CUT_HZ, MUD_CUT_GAIN_DB, MUD_CUT_WIDTH_O = 300, -3.0, 1.2
CLARITY_HZ, CLARITY_GAIN_DB, CLARITY_WIDTH_O = 2800, 3.0, 1.2
PRESENCE_HZ, PRESENCE_GAIN_DB, PRESENCE_WIDTH_O = 4500, 2.5, 1.4
AIR_SHELF_HZ, AIR_SHELF_GAIN_DB = 6800, 3.0
COMP_THRESHOLD_DB = -18.0
COMP_RATIO = 3.0
COMP_ATTACK_MS = 10
COMP_RELEASE_MS = 120
MASTER_LIMITER_LINEAR = 0.8414  # -1.5 dBTP
SEGMENT_FADE_S = 0.02  # anti-click micro-fades at cut seams
# AAC reconstruction overshoots a hard-limited waveform by up to ~1.5 dB on
# decode for sibilant/high-crest material (measured: -0.66 dBFS decoded from a
# -2.1 dBFS pre-codec ceiling on a sibilance-heavy mix); the segment encoder
# needs its own ceiling so the SHIPPED mp4 stays under the -1.0 dBTP clipping
# gate at ANY crest (master bus stays at the spec'd -1.5)
ENCODE_HEADROOM_LIMITER_LINEAR = 0.7079  # -3.0 dBFS pre-codec
SAMPLE_RATE = 48_000

# duck gain as a linear factor: 10^(-14/20)
DUCK_LINEAR = round(10.0 ** (DUCK_DB / 20.0), 6)


class TwoBusMixError(RuntimeError):
    pass


def bus1_tone_chain() -> str:
    """Bus-1 intelligibility chain: mud cut -> clarity -> presence -> air -> comp.

    Applied to the neural-clean audio BEFORE the gate, so the compressor
    breathes with the voice, not with the gate. The presence bell + air shelf
    are the timbre-restoration law: they put back the 2-12 kHz band the neural
    extractor shaved (the measured "deep voice" artifact — see the constants
    block). Safe noise-wise: after the extractor the residual in those bands
    sits >20 dB under the voice, so the lift cannot re-inflate the floor.
    """
    return ",".join(
        [
            f"equalizer=f={MUD_CUT_HZ}:width_type=o:width={MUD_CUT_WIDTH_O}:g={MUD_CUT_GAIN_DB}",
            f"equalizer=f={CLARITY_HZ}:width_type=o:width={CLARITY_WIDTH_O}:g={CLARITY_GAIN_DB:+g}",
            f"equalizer=f={PRESENCE_HZ}:width_type=o:width={PRESENCE_WIDTH_O}:g={PRESENCE_GAIN_DB:+g}",
            f"highshelf=f={AIR_SHELF_HZ}:g={AIR_SHELF_GAIN_DB:+g}",
            (
                f"acompressor=threshold={COMP_THRESHOLD_DB:g}dB:ratio={COMP_RATIO:g}:"
                f"attack={COMP_ATTACK_MS}:release={COMP_RELEASE_MS}"
            ),
        ]
    )


def bus2_volume() -> float:
    """Bed level from env, read at CALL time (config, not code — cost law)."""
    return float(os.getenv(BUS2_VOLUME_ENV, str(BUS2_VOLUME_DEFAULT)))


def bus2_chain(volume: float | None = None) -> str:
    """Bus-2 tamed ambience: 150 Hz-4 kHz band, bed level, soft-limited."""
    vol = bus2_volume() if volume is None else volume
    return ",".join(
        [
            f"highpass=f={BUS2_HIGHPASS_HZ}",
            f"lowpass=f={BUS2_LOWPASS_HZ}",
            f"volume={vol:g}",
            (
                f"alimiter=level_in=1:level_out=1:level=disabled:"
                f"limit={BUS2_LIMITER_LINEAR:g}:attack=5:release=80"
            ),
        ]
    )


def master_chain(gain_db: float) -> str:
    """Master bus: speech-gated gain -> -1.5 dBTP brickwall -> 48 kHz stereo lock.

    `level=disabled` is the law: alimiter's auto-level would normalize the
    output back UP to 0 dBFS, silently destroying the calibrated ceiling.
    """
    return ",".join(
        [
            f"volume={gain_db:+g}dB",
            (
                f"alimiter=level_in=1:level_out=1:level=disabled:"
                f"limit={MASTER_LIMITER_LINEAR:g}:attack=5:release=50"
            ),
            f"aresample={SAMPLE_RATE}",
            "aformat=sample_fmts=fltp:channel_layouts=stereo",
        ]
    )


def segment_fade_chain(dur_s: float, fade_s: float = SEGMENT_FADE_S) -> str:
    """20 ms anti-click micro-fades + pre-codec headroom limiter (per segment).

    The fades kill clicks at the concat seams; the encode-headroom limiter
    (-2.1 dBFS, level=disabled) absorbs the AAC reconstruction overshoot so
    the SHIPPED mp4 decodes under the -1.0 dBTP clipping QA gate.
    """
    fade_s = min(fade_s, max(0.0, dur_s / 2.0))
    tail_start = max(0.0, dur_s - fade_s)
    return ",".join(
        [
            f"afade=t=in:st=0:d={fade_s:.4f}",
            f"afade=t=out:st={tail_start:.4f}:d={fade_s:.4f}",
            (
                f"alimiter=level_in=1:level_out=1:level=disabled:"
                f"limit={ENCODE_HEADROOM_LIMITER_LINEAR:g}:attack=5:release=50"
            ),
        ]
    )


def build_two_bus_mix_cmd(
    bus1_wav: Path,
    bus2_src_path: Path | None,
    out_path: Path,
    master_gain_db: float,
    bus2_level: float | None = None,
) -> list[str]:
    """THE master mix command (per asset). Output: 48 kHz stereo PCM WAV.

    Bus-2 rides only when `bus2_level` (or $AUDIO_BUS2_VOLUME) > 0 AND a
    source is given — the default master is pure isolated voice (the
    2026-11-18 ear verdict; see module docstring).

    Inputs:  [0] = bus1_gated.wav (neural-clean + EQ + comp + gate)
             [1] = raw asset audio (tamed ambience source, opt-in)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vol = bus2_volume() if bus2_level is None else bus2_level
    use_bus2 = vol > 0.0 and bus2_src_path is not None
    if use_bus2:
        graph = ";".join(
            [
                "[0:a]aformat=sample_fmts=fltp:channel_layouts=stereo[b1]",
                f"[1:a]aformat=sample_fmts=fltp:channel_layouts=stereo,{bus2_chain(vol)}[b2]",
                "[b1][b2]amix=inputs=2:duration=first:normalize=0[mix]",
                f"[mix]{master_chain(master_gain_db)}[m]",
            ]
        )
        inputs = ["-i", str(bus1_wav), "-i", str(bus2_src_path)]
    else:
        graph = ";".join(
            [
                "[0:a]aformat=sample_fmts=fltp:channel_layouts=stereo[b1]",
                f"[b1]{master_chain(master_gain_db)}[m]",
            ]
        )
        inputs = ["-i", str(bus1_wav)]
    return [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        *inputs,
        "-filter_complex",
        graph,
        "-map",
        "[m]",
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "2",
        "-c:a",
        "pcm_s16le",
        str(out_path),
    ]


def run_two_bus_mix(
    bus1_wav: Path,
    bus2_src_path: Path,
    out_path: Path,
    master_gain_db: float,
) -> Path:
    r = subprocess.run(
        build_two_bus_mix_cmd(bus1_wav, bus2_src_path, out_path, master_gain_db),
        capture_output=True,
    )
    if r.returncode != 0:
        raise TwoBusMixError(
            f"2-bus mix failed ({out_path}): {r.stderr[-400:].decode(errors='replace')}"
        )
    return out_path


# ------------------------------------------------------------------ the gate
def duck_envelope(
    n_samples: int,
    sr: int,
    speech_intervals: list[tuple[float, float]],
    duck_linear: float = DUCK_LINEAR,
    ramp_s: float = RAMP_S,
) -> np.ndarray:
    """Per-sample Bus-1 gain: full voice in speech, -14 dB in gaps, cosine ramps.

    Raised-cosine 150 ms attack/release so the duck never clicks or pumps; the
    envelope is deterministic (pure math on the gate intervals) and unit-tested.
    """
    env = np.full(n_samples, duck_linear, dtype=np.float32)
    ramp_n = max(1, int(ramp_s * sr))
    lo = duck_linear
    span = 1.0 - lo
    for s, e in speech_intervals:
        a, b = max(0, int(s * sr)), min(int(e * sr), n_samples)
        span_len = b - a
        if span_len <= 0:
            continue
        head = min(ramp_n, max(1, span_len // 2))
        if head <= 2:  # sub-ramp blip: stay fully open, no partial-cosine steps
            env[a:b] = 1.0
            continue
        u = np.arange(head, dtype=np.float64) / (head - 1)  # 0..1 inclusive
        ramp = (0.5 - 0.5 * np.cos(np.pi * u)).astype(np.float32)  # smooth S-curve
        env[a : a + head] = lo + span * ramp  # attack: lo -> 1.0
        env[a + head : b - head] = 1.0  # open speech span (may be empty)
        env[b - head : b] = lo + span * (1.0 - ramp)  # release: 1.0 -> lo
    return env


def apply_gate_inplace(wav_path: Path, envelope: np.ndarray) -> None:
    """Multiply a 16-bit stereo WAV by the gate envelope (in place, atomic).

    Reads with stdlib `wave` (no new deps), applies the float envelope, writes
    int16 with a hard clip guard at 0.999 amplitude. The envelope is per-mono-
    sample; interleaved stereo doubles it. A length mismatch beyond one sample
    is a caller bug and raises (never silently mis-align the gate).
    """
    import wave

    wav_path = Path(wav_path)
    try:
        with wave.open(str(wav_path), "rb") as w:
            sr = w.getframerate()
            nch = w.getnchannels()
            width = w.getsampwidth()
            if width != 2 or sr != SAMPLE_RATE:
                raise TwoBusMixError(
                    f"gate input must be 16-bit 48 kHz WAV, got {width * 8}-bit {sr} Hz: {wav_path}"
                )
            frames = w.readframes(w.getnframes())
    except (FileNotFoundError, wave.Error) as e:
        raise TwoBusMixError(f"gate input unreadable: {wav_path} ({e})") from e
    pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    env2 = np.repeat(envelope, nch)[: pcm.size]
    if env2.size < pcm.size:
        missing = pcm.size - env2.size
        if missing > nch:  # >1 frame of drift cannot happen on aligned timelines
            raise TwoBusMixError(
                f"gate envelope {envelope.size} mono samples < audio {pcm.size // nch}: "
                "enhanced wav is off-timeline"
            )
        env2 = np.concatenate([env2, np.full(missing, DUCK_LINEAR, dtype=np.float32)])
    out = np.clip(pcm * env2, -0.999, 0.999)
    data = (out * 32767.0).astype(np.int16).tobytes()
    tmp = wav_path.with_suffix(".gated.tmp.wav")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(nch)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data)
    tmp.replace(wav_path)
