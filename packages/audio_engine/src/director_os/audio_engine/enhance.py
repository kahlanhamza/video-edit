"""Neural speech enhancement — Bus-1 primary extractor (PLAN §4.5, M4 audio law).

Port-and-adapter like `LLMBackend`/`TtsBackend`: a `SpeechEnhancer` port, a
resolver (`resolve_enhancer`), and every backend zero-heavy-SDK at import time:

1. `ClearerVoiceEnhancer` — `MossFormer2_SE_48K` (Apache-2.0, native 48 kHz),
   the strongest open speech extractor for outdoor bodycam noise. Uses the
   `clearvoice` package lazily (torch-backed); not on the default path, one
   `AUDIO_ENHANCER=clearervoice` flips it on when weights are installed.
2. `DeepFilterCliEnhancer` — the `deep-filter` Rust CLI (DeepFilterNet3, MIT,
   model embedded in the release binary). THE production default: a static
   musl binary dropped into the image, `--atten-lim-db 28` (14 dB is far too
   weak for highway traffic / wind gusts / cruiser engine noise).
3. `AfftdnEnhancer` — the ffmpeg `afftdn` FFT denoiser. Explicitly a WEAK
   fallback: non-stationary noise survives it audibly, so it must NEVER be
   silently passed off as "cleaned" audio.

THE NO-SILENT-FALLBACK LAW:
- `resolve_enhancer(preference="auto")` prefers clearervoice -> deep-filter and
  RAISES `EnhancerUnavailableError` when neither exists. `afftdn` is only ever
  selected when EXPLICITLY configured (`AUDIO_ENHANCER=afftdn`) or inside a
  hermetic test env — never as an implicit auto fallback.
- `EnhancerResult.engine` names the exact engine used; the worker persists it
  in `agent_tool_logs` (`audio.enhance_speech`) and `qa.audit_technical` turns
  an afftdn run outside hermetic tests into a FAILED `enhancer_engine_used`
  check — production renders cannot ship afftdn-cleaned audio quietly.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

# env knobs (one .env line switches provider — cost law: config, not code)
ENHANCER_ENV = "AUDIO_ENHANCER"
DEEPFILTER_BIN_ENV = "DEEPFILTER_BIN"
DEEPFILTER_DEFAULT_BIN = "deep-filter"
DEEPFILTER_DEFAULT_ATTEN_LIM_DB = 28.0  # 14 dB is too weak for highway noise
AFFTDN_ENV = "DIRECTOR_OS_ALLOW_AFFTDN"  # hermetic-test escape hatch, set by CI/tests only

NEURAL_ENGINE_NAMES = ("mossformer2_se_48k", "deepfilternet3")


class EnhancerError(RuntimeError):
    pass


class EnhancerUnavailableError(EnhancerError):
    """The requested/configured enhancer does not exist in this environment."""


@dataclass
class EnhancerResult:
    """Receipt for one enhance() run — everything QA + the tool log need."""

    engine: str
    neural: bool
    atten_lim_db: float | None
    in_path: Path
    out_path: Path
    duration_in_s: float
    duration_out_s: float

    @property
    def duration_drift_s(self) -> float:
        return abs(self.duration_out_s - self.duration_in_s)


class SpeechEnhancer(Protocol):
    """Turns a noisy 48 kHz WAV into a speech-focused clean WAV, same timeline."""

    name: str
    neural: bool

    def enhance(self, wav_in: Path, wav_out: Path) -> EnhancerResult: ...


def _wav_duration_s(path: Path) -> float:
    from director_os.core.qa_probe import probe

    return probe(path).duration_s


def _run(cmd: list[str], what: str) -> None:
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        tail = r.stderr[-400:].decode(errors="replace")
        raise EnhancerError(f"{what} failed (exit {r.returncode}): {tail}")


# --------------------------------------------------------------------- backends
class DeepFilterCliEnhancer:
    """`deep-filter` Rust CLI (DeepFilterNet3, model embedded in the binary)."""

    name = "deepfilternet3"
    neural = True

    def __init__(
        self,
        bin_path: str | None = None,
        atten_lim_db: float = DEEPFILTER_DEFAULT_ATTEN_LIM_DB,
    ):
        env_bin = os.getenv(DEEPFILTER_BIN_ENV, "").strip()
        self.bin_path = bin_path or env_bin or DEEPFILTER_DEFAULT_BIN
        resolved = shutil.which(self.bin_path) or (
            Path(self.bin_path) if Path(self.bin_path).exists() else None
        )
        if resolved is None:
            raise EnhancerUnavailableError(
                f"deep-filter CLI not found (looked for '{self.bin_path}' on PATH and "
                f"${DEEPFILTER_BIN_ENV}); install per apps/worker/Dockerfile or set "
                f"{DEEPFILTER_BIN_ENV}=/path/to/deep-filter"
            )
        self.exe = str(resolved)
        self.atten_lim_db = atten_lim_db

    def enhance(self, wav_in: Path, wav_out: Path) -> EnhancerResult:
        wav_in, wav_out = Path(wav_in), Path(wav_out)
        wav_out.parent.mkdir(parents=True, exist_ok=True)
        tmp_out_dir = wav_out.parent / f".df_out_{wav_out.stem}"
        tmp_out_dir.mkdir(parents=True, exist_ok=True)
        dur_in = _wav_duration_s(wav_in)
        _run(
            [
                self.exe,
                "--atten-lim-db",
                f"{self.atten_lim_db:g}",
                str(wav_in),
                "-o",
                str(tmp_out_dir),
            ],
            what=f"deep-filter ({self.exe}, atten-lim {self.atten_lim_db:g} dB)",
        )
        produced = tmp_out_dir / wav_in.name
        if not produced.exists():
            raise EnhancerError(f"deep-filter produced no output for {wav_in.name}")
        produced.replace(wav_out)
        tmp_out_dir.rmdir()
        return EnhancerResult(
            engine=self.name,
            neural=True,
            atten_lim_db=self.atten_lim_db,
            in_path=wav_in,
            out_path=wav_out,
            duration_in_s=dur_in,
            duration_out_s=_wav_duration_s(wav_out),
        )


class ClearerVoiceEnhancer:
    """ClearerVoice-Studio `MossFormer2_SE_48K` — native 48 kHz, Apache-2.0.

    The `clearvoice` package (torch-backed) is imported lazily: zero cost for
    the default deep-filter path, full strength when installed.
    """

    name = "mossformer2_se_48k"
    neural = True

    def enhance(self, wav_in: Path, wav_out: Path) -> EnhancerResult:
        wav_in, wav_out = Path(wav_in), Path(wav_out)
        wav_out.parent.mkdir(parents=True, exist_ok=True)
        try:
            from clearvoice import ClearVoice
        except ImportError as e:  # pragma: no cover — depends on env install
            raise EnhancerUnavailableError(
                "clearvoice package not installed; install ClearerVoice-Studio "
                "(pip install clearvoice or repo install) to use MossFormer2_SE_48K"
            ) from e
        dur_in = _wav_duration_s(wav_in)
        try:
            voicer = ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])
            output = voicer(input_path=str(wav_in), online_write=False)
            voicer.write(output, output_path=str(wav_out))
        except Exception as e:  # model download/weights errors surface loudly
            raise EnhancerError(f"MossFormer2_SE_48K enhance failed: {e}") from e
        if not wav_out.exists():
            raise EnhancerError(f"clearvoice wrote no output to {wav_out}")
        return EnhancerResult(
            engine=self.name,
            neural=True,
            atten_lim_db=None,
            in_path=wav_in,
            out_path=wav_out,
            duration_in_s=dur_in,
            duration_out_s=_wav_duration_s(wav_out),
        )


class AfftdnEnhancer:
    """ffmpeg `afftdn` FFT denoiser — the WEAK fallback, never silent.

    Only reachable via explicit `AUDIO_ENHANCER=afftdn` or a hermetic test env;
    production runs get a FAILED `enhancer_engine_used` QA check so the
    operator cannot ship this as "cleaned" audio without seeing the flag.
    """

    name = "afftdn"
    neural = False

    def __init__(self, noise_floor_db: float = -25.0):
        self.noise_floor_db = noise_floor_db

    def enhance(self, wav_in: Path, wav_out: Path) -> EnhancerResult:
        wav_in, wav_out = Path(wav_in), Path(wav_out)
        wav_out.parent.mkdir(parents=True, exist_ok=True)
        dur_in = _wav_duration_s(wav_in)
        _run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(wav_in),
                "-af",
                f"afftdn=nf={self.noise_floor_db:g}:tn=1",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-c:a",
                "pcm_s16le",
                str(wav_out),
            ],
            what="afftdn fallback denoise",
        )
        return EnhancerResult(
            engine=self.name,
            neural=False,
            atten_lim_db=None,
            in_path=wav_in,
            out_path=wav_out,
            duration_in_s=dur_in,
            duration_out_s=_wav_duration_s(wav_out),
        )


# --------------------------------------------------------------------- resolver
def hermetic_test_env() -> bool:
    """True inside pytest or when the explicit test escape hatch is set."""
    return "PYTEST_CURRENT_TEST" in os.environ or os.getenv(AFFTDN_ENV) == "1"


def probe_available_engines() -> dict[str, bool]:
    """Capability scan used by `auto` and by worker startup logs."""
    available = {"clearervoice": False, "deepfilter": False}
    try:
        import clearvoice  # noqa: F401

        available["clearervoice"] = True
    except ImportError:
        pass
    df_bin = os.getenv(DEEPFILTER_BIN_ENV, "").strip() or DEEPFILTER_DEFAULT_BIN
    available["deepfilter"] = shutil.which(df_bin) is not None or Path(df_bin).exists()
    return available


def resolve_enhancer(preference: str | None = None) -> SpeechEnhancer:
    """Pick the enhancer EXPLICITLY — auto never silently degrades to afftdn.

    Preference order for `auto`: clearervoice (if importable) -> deep-filter
    (if the CLI binary exists) -> EnhancerUnavailableError with install
    instructions. `afftdn` is returned ONLY for an explicit `afftdn`
    preference; a production (non-hermetic) run is marked via
    `EnhancerResult.neural=False` and fails the QA gate downstream.
    """
    pref = ((preference or os.getenv(ENHANCER_ENV, "auto")) or "auto").strip().lower()
    available = probe_available_engines()
    if pref == "clearervoice":
        return ClearerVoiceEnhancer()
    if pref == "deepfilter":
        return DeepFilterCliEnhancer()
    if pref == "afftdn":
        return AfftdnEnhancer()
    if pref == "auto":
        if available["clearervoice"]:
            return ClearerVoiceEnhancer()
        if available["deepfilter"]:
            return DeepFilterCliEnhancer()
        raise EnhancerUnavailableError(
            "no neural speech enhancer available (clearvoice package? deep-filter "
            "binary?) — refusing to silently pass afftdn as cleaned audio; install "
            "deep-filter (see apps/worker/Dockerfile) or set "
            f"{ENHANCER_ENV}=afftdn explicitly to accept the flagged weak fallback"
        )
    raise EnhancerError(
        f"unknown {ENHANCER_ENV} preference: '{pref}' (auto|clearervoice|deepfilter|afftdn)"
    )


def engine_is_production_legal(
    engine: str, allow_afftdn: bool | None = None, *, neural: bool = True
) -> bool:
    """The QA law: afftdn outside hermetic tests is a failed check, not a fallback.

    `neural` comes from the receipt (`EnhancerResult.neural` — the adapter's
    own declaration): only `afftdn` ships as a non-neural adapter, so afftdn
    is hard-gated while genuinely-neural engines (deepfilternet3,
    mossformer2_se_48k, and hermetic test fakes) pass.
    """
    if engine == "afftdn":
        return hermetic_test_env() if allow_afftdn is None else allow_afftdn
    return neural
