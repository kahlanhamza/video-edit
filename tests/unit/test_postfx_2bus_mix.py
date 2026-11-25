"""The 2-bus mix law in post-fx — the calibrated mix IS the enhanced audio.

Full-tape runway (2026-11-25, Eustis 2627s): post-fx deep-filtered the WHOLE
master mix on an isolate_2bus render — no loudness target, no bus law — and
flattened the calibrated −11 dBFS bus dialogue to a measured −17.4 dBFS / SNR
12.0. The neural Bus-1 chain (kept-window receipts, refinement loop, timbre
restoration, duck architecture, −1.5 dBTP limiter) already ran at audio-prep;
the mux must ship the master's own calibrated mix, in STEREO.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from director_os.worker import post_fx


def _deps(tmp_path: Path, audio_mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        settings=SimpleNamespace(output_dir=tmp_path),
        audio_mode=audio_mode,
        log=lambda msg: None,
    )


def _pre_baked_marks(work: Path) -> None:
    work.mkdir(parents=True, exist_ok=True)
    (work / "captions.done").write_text("12")
    (work / "blur.done").write_text("tracked|2")
    (work / "master_blurred.mp4").write_bytes(b"blurred")


def test_isolate_2bus_master_ships_its_calibrated_mix(monkeypatch, tmp_path):
    calls: list[tuple[str, int]] = []

    def fake_extract(master: Path, work: Path, channels: int = 1) -> Path:
        calls.append(("extract", channels))
        return work / "master_audio.wav"

    monkeypatch.setattr(post_fx, "_extract_master_audio", fake_extract)
    monkeypatch.setattr(post_fx, "_measure_snr", lambda wav: -40.2)
    monkeypatch.setattr(post_fx, "probe", lambda p: SimpleNamespace(duration_s=10.0))

    def fake_run(cmd, what):
        calls.append(("mux", 0))
        Path(cmd[-1]).touch()

    monkeypatch.setattr(post_fx, "_run", fake_run)
    deps = _deps(tmp_path, "isolate_2bus")
    master = tmp_path / "master.mp4"
    master.write_bytes(b"master")
    _pre_baked_marks(tmp_path / "pipeline" / "pid2" / "post_fx")

    report = post_fx.stage_post_fx(master, None, {}, "pid2", deps)

    assert report.enhance_engine == "master_mix_2bus"
    assert report.ok
    marker = tmp_path / "pipeline" / "pid2" / "post_fx" / "audio.done"
    assert marker.read_text() == "master_mix_2bus"
    extracts = [c for c in calls if c[0] == "extract"]
    assert extracts and all(c[1] == 2 for c in extracts), (
        "the 48 kHz stereo stem law — a mono fold-down would shift the bus mix"
    )
    assert any(c[0] == "mux" for c in calls)


def test_loudnorm_mode_keeps_the_deepfilter_path(monkeypatch, tmp_path):
    """The legacy post-fx enhancer (default mode) is untouched."""
    deps = _deps(tmp_path, "loudnorm_48k")
    master = tmp_path / "master.mp4"
    master.write_bytes(b"master")
    _pre_baked_marks(tmp_path / "pipeline" / "pid3" / "post_fx")

    def explode(*args, **kwargs):
        raise AssertionError("default mode still re-enhances")

    monkeypatch.setattr(post_fx, "_measure_snr", lambda wav: None)
    monkeypatch.setattr(post_fx, "probe", lambda p: SimpleNamespace(duration_s=10.0))

    def fake_run(cmd, what):
        Path(cmd[-1]).touch()

    monkeypatch.setattr(post_fx, "_run", fake_run)
    enquenced = False

    class FakeResult:
        engine = "deepfilternet3"
        neural = True

    class FakeEnhancer:
        def __init__(self) -> None:
            nonlocal enquenced
            enquenced = True

        def enhance(self, src, dst):
            Path(dst).write_bytes(b"enhanced")
            return FakeResult()

    import director_os.audio_engine.enhance as enhance_mod

    monkeypatch.setattr(enhance_mod, "DeepFilterCliEnhancer", FakeEnhancer)
    report = post_fx.stage_post_fx(master, None, {}, "pid3", deps)
    assert enquenced
    assert report.enhance_engine == "deepfilternet3"
