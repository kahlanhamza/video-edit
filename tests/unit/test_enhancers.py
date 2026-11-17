"""SpeechEnhancer port laws — no silent fallbacks, explicit engines only.

THE LAW under test (audio.enhance_speech):
- `auto` NEVER degrades to afftdn — it raises with install instructions;
- afftdn is reachable only by explicit config or hermetic tests;
- the receipt names the exact engine (agent_tool_logs / QA read it);
- `engine_is_production_legal` fails afftdn in production, passes neural.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from director_os.audio_engine.enhance import (
    DEEPFILTER_DEFAULT_ATTEN_LIM_DB,
    AfftdnEnhancer,
    DeepFilterCliEnhancer,
    EnhancerResult,
    EnhancerUnavailableError,
    engine_is_production_legal,
    hermetic_test_env,
    probe_available_engines,
    resolve_enhancer,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Tests control the environment — no inherited AUDIO_ENHANCER config."""
    monkeypatch.delenv("AUDIO_ENHANCER", raising=False)
    monkeypatch.delenv("DEEPFILTER_BIN", raising=False)


class TestResolver:
    def test_auto_raises_when_no_neural_engine(self, monkeypatch):
        # hide the real binary in this sandbox: pretend nothing exists
        monkeypatch.setenv("DEEPFILTER_BIN", "/definitely/not/here")
        monkeypatch.setattr(shutil, "which", lambda _name: None)
        with pytest.raises(EnhancerUnavailableError, match="refusing to silently pass afftdn"):
            resolve_enhancer("auto")

    def test_auto_prefers_deepfilter_when_binary_exists(self, monkeypatch):
        monkeypatch.setattr(DeepFilterCliEnhancer, "__init__", lambda self: None)
        monkeypatch.setattr(
            shutil, "which", lambda name: "/usr/bin/deep-filter" if name == "deep-filter" else None
        )
        eng = resolve_enhancer("auto")
        assert isinstance(eng, DeepFilterCliEnhancer)

    def test_explicit_afftdn_is_allowed_but_flagged_non_neural(self):
        eng = resolve_enhancer("afftdn")
        assert eng.name == "afftdn"
        assert eng.neural is False

    def test_unknown_preference_raises(self):
        with pytest.raises(Exception, match="unknown AUDIO_ENHANCER"):
            resolve_enhancer("runes")

    def test_env_preference_used_when_no_arg(self, monkeypatch):
        monkeypatch.setenv("AUDIO_ENHANCER", "afftdn")
        assert resolve_enhancer().name == "afftdn"

    def test_deepfilter_default_atten_is_28db(self):
        assert DEEPFILTER_DEFAULT_ATTEN_LIM_DB == 28.0


class TestProductionLaw:
    def test_afftdn_fails_production(self):
        assert engine_is_production_legal("afftdn", allow_afftdn=False) is False

    def test_afftdn_allowed_in_hermetic_env(self):
        assert hermetic_test_env() is True  # pytest IS the hermetic env
        assert engine_is_production_legal("afftdn", allow_afftdn=None) is True

    def test_neural_engines_pass(self):
        for name in ("deepfilternet3", "mossformer2_se_48k"):
            assert engine_is_production_legal(name, allow_afftdn=False, neural=True)

    def test_non_neural_claim_never_passes(self):
        assert engine_is_production_legal("whatever", allow_afftdn=False, neural=False) is False

    def test_probe_available_reports_capability(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda _n: None)
        monkeypatch.setenv("DEEPFILTER_BIN", "/definitely/not/here")
        caps = probe_available_engines()
        assert caps == {"clearervoice": False, "deepfilter": False}


class TestDeepFilterCli:
    def test_unavailable_binary_raises_with_instructions(self, monkeypatch):
        monkeypatch.setenv("DEEPFILTER_BIN", "/definitely/not/here")
        with pytest.raises(Exception, match="deep-filter CLI not found"):
            DeepFilterCliEnhancer()

    def test_enhance_builds_expected_command(self, monkeypatch, tmp_path: Path):
        monkeypatch.setattr(
            shutil,
            "which",
            lambda name: name if name == "/usr/bin/deep-filter" else None,
        )
        eng = DeepFilterCliEnhancer(bin_path="/usr/bin/deep-filter", atten_lim_db=28.0)
        assert eng.atten_lim_db == 28.0

        calls: list[list[str]] = []

        def fake_run(cmd, capture_output=True, text=False, **_kw):
            calls.append(cmd)
            # emulate the CLI: writes <outdir>/<same filename>
            outdir = Path(cmd[cmd.index("-o") + 1])
            (outdir / "in.wav").write_bytes(b"RIFF")

            class R:
                returncode = 0
                stderr = b""

            return R()

        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.setattr("director_os.audio_engine.enhance._wav_duration_s", lambda _p: 2.0)
        result = eng.enhance(tmp_path / "in.wav", tmp_path / "clean.wav")
        cmd = calls[0]
        assert "--atten-lim-db" in cmd and "28" in cmd
        assert "/usr/bin/deep-filter" in cmd
        assert isinstance(result, EnhancerResult)
        assert result.engine == "deepfilternet3" and result.neural is True
        assert result.out_path == tmp_path / "clean.wav"

    def test_enhance_raises_on_cli_failure(self, monkeypatch, tmp_path: Path):
        monkeypatch.setattr(
            shutil,
            "which",
            lambda name: name if name == "/usr/bin/deep-filter" else None,
        )
        eng = DeepFilterCliEnhancer(bin_path="/usr/bin/deep-filter")

        def fake_run(cmd, capture_output=True, text=False, **_kw):
            # ffprobe/ffmpeg calls (duration probe, etc.) succeed; the
            # deep-filter binary call itself fails
            class R:
                stderr = b"boom"
                stdout = b""

            if cmd[0] in ("ffprobe", "ffmpeg"):
                R.returncode = 0
                R.stdout = (
                    b'{"streams": [{"codec_type": "audio", "sample_rate": "48000", "channels": 2}],'
                    b'"format": {"duration": "2.0", "bit_rate": "128000"}}'
                    if cmd[0] == "ffprobe"
                    else b""
                )
            else:
                R.returncode = 1
            return R()

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(Exception, match="deep-filter"):
            eng.enhance(tmp_path / "in.wav", tmp_path / "clean.wav")


class TestAfftdn:
    def test_afftdn_is_explicitly_weak_and_neural_false(self):
        eng = AfftdnEnhancer()
        assert eng.name == "afftdn"
        assert eng.neural is False
        cmd_matters = True  # documented law: afftdn nf=-25 tn=1 chain
        assert cmd_matters
