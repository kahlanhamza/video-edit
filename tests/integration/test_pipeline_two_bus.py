"""2-Bus Speech Isolation — the full worker seam, end to end.

Synthetic noisy bodycam MP4 (testsrc2 video + speech-band bursts + brown
noise), a stub ASR whose words match the bursts, and a `PerfectEnhancer` fake
(the fixture's own noise-free reference) drive run_pipeline with
AUDIO_MODE=isolate_2bus. Asserts:

- stage audio_prep produces the content-addressed master + receipt (cached on rerun)
- the compiled segments render from the 2-bus master WAV (anti-click fades)
- agent_tool_logs records the exact enhancer engine
- media_assets.r2_enhanced_key is persisted
- qa.audit_technical enforces snr_verification / zero_clipping /
  stream_uniformity / enhancer_engine_used on the final master
- the 15 s A/B raw-vs-clean pair exists for ear evidence
"""

from __future__ import annotations

import dataclasses
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest
from director_os.audio_engine.enhance import EnhancerResult
from director_os.core.db import AgentToolLogRow, JobRow, MediaAssetRow, ProjectRow
from director_os.core.models import ProjectStatus
from director_os.core.qa_probe import probe
from director_os.otio_engine.segment_cache import VideoRenderSpec
from director_os.perception.transcript import Transcript, WordTiming
from director_os.worker.pipeline import PipelineDeps, run_pipeline
from sqlalchemy import select

SR = 48_000
DUR_S = 12.0
WORDS = [(1.0, 3.0), (4.0, 6.5), (7.5, 10.5)]
SPEECH_RMS = 0.22  # -13 dBFS bursts; mixed at 0.8 -> speech lands ~ -15 dBFS raw
NOISE_RMS = 0.16  # broadband highway/static noise (-16 dBFS)


def _voiced_burst(n: int, sr: int, rng, rms_target: float) -> np.ndarray:
    """Harmonic-complex 'voice' (f0=130 Hz, 1/k rolloff) + sibilance band.

    Speech-like crest (~4 dB) instead of Gaussian noise's ~12 dB, so limiter
    behavior in the test matches real compressed dialogue rather than a
    pathological case. The 3-8 kHz band-limited noise layer stands in for
    fricatives/sibilance — real speech carries it, and the timbre gate
    (timbre_not_muffled) would rightly reject a voice with none.
    """
    t = np.arange(n) / sr
    phase = rng.uniform(0, 2 * np.pi, size=12)
    wave_ = sum(np.sin(2 * np.pi * 130 * k * t + phase[k - 1]) / k for k in range(1, 13))
    wave_ *= 0.6 + 0.4 * np.sin(2 * np.pi * 3.3 * t)  # syllabic AM
    wave_ = wave_ / (np.sqrt(np.mean(np.square(wave_))) + 1e-9) * rms_target
    hiss = _band_noise(n, sr, rng, 3000.0, 8000.0) * (rms_target * 0.22)
    # sibilance rides WITH the syllables (same AM as the voice) — continuous
    # independent white-noise would stack a 12 dB crest on the voice and the
    # limiter chain would eat real loudness, which real speech never does
    hiss *= 0.6 + 0.4 * np.sin(2 * np.pi * 3.3 * t)
    out = wave_ + hiss
    # soft-knee limiter: free-running harmonics + hiss show ~12 dB crest, a
    # crest no AGC'd bodycam dialogue ever carries. The shipped loudness is a
    # limiter equilibrium (RMS = ceiling - crest), so the fixture must bound
    # crest or the -12..-6 window audit fails for synthetic reasons. Drive is
    # tuned so the burst crest lands ~7 dB over RMS — compressed-dialogue-like.
    drive = 0.35
    out = np.tanh(drive * out / (np.sqrt(np.mean(np.square(out))) + 1e-9))
    peak = float(np.max(np.abs(out)))
    return out * (rms_target * 2.2 / peak)  # peak ~ 6.8 dB over RMS


def _band_noise(n: int, sr: int, rng, lo_hz: float, hi_hz: float) -> np.ndarray:
    x = rng.normal(size=n)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    out = np.fft.irfft(spec * ((freqs >= lo_hz) & (freqs <= hi_hz)), n=n)
    return out / (np.sqrt(np.mean(np.square(out))) + 1e-9)


@pytest.fixture(scope="module")
def noisy_bodycam(tmp_path_factory) -> Path:
    """12 s MP4: animated video + voiced bursts buried in broadband noise (48k stereo)."""
    root = tmp_path_factory.mktemp("noisy_bwc")
    rng = np.random.default_rng(42)
    n = int(DUR_S * SR)
    speech = np.zeros(n)
    for s0, s1 in WORDS:
        a, b = int(s0 * SR), int(s1 * SR)
        speech[a:b] = _voiced_burst(b - a, SR, rng, SPEECH_RMS)
    white = rng.normal(size=n)
    # gentle low tilt: bodycam noise is broadband with a mild low shelf
    noise = white / (np.sqrt(np.mean(np.square(white))) + 1e-9) * NOISE_RMS
    low = np.convolve(noise, np.ones(64) / 64, mode="same")
    noise = noise + 0.5 * low / (np.sqrt(np.mean(np.square(low))) + 1e-9) * NOISE_RMS
    mixed = np.clip(speech * 0.8 + noise, -0.99, 0.99)
    stereo = np.stack([mixed, mixed], axis=1)
    wav = root / "noisy_48k.wav"
    data = (stereo * 32767).astype(np.int16).tobytes()
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data)
    mp4 = root / "noisy_bodycam.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=30:duration=12",
            "-i",
            str(wav),
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "32",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-shortest",
            str(mp4),
        ],
        check=True,
        capture_output=True,
    )
    # the enhancer fake's reference: speech only, same mix level, 48 kHz stereo
    clean = root / "clean_reference.wav"
    cdata = (
        (np.clip(np.stack([speech * 0.8, speech * 0.8], axis=1), -0.999, 0.999) * 32767)
        .astype(np.int16)
        .tobytes()
    )
    with wave.open(str(clean), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(cdata)
    return mp4


class PerfectEnhancer:
    """Test fake: the noise is known to the fixture, so 'enhance' removes it."""

    name = "perfect_test_fake"
    neural = True

    def __init__(self, clean_ref: Path):
        self.clean_ref = clean_ref

    def enhance(self, wav_in: Path, wav_out: Path) -> EnhancerResult:
        wav_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.clean_ref, wav_out)  # same timeline, no noise
        return EnhancerResult(
            engine=self.name,
            neural=True,
            atten_lim_db=None,
            in_path=wav_in,
            out_path=wav_out,
            duration_in_s=probe(wav_in).duration_s,
            duration_out_s=probe(wav_out).duration_s,
        )


@pytest.fixture()
def deps(tmp_path: Path, noisy_bodycam: Path):
    from director_os.core.config import get_settings
    from director_os.core.db import init_models, make_engine, make_session_factory

    settings = get_settings()
    settings = dataclasses.replace(
        settings,
        output_dir=tmp_path / "output",
        ingest_dir=tmp_path / "output" / "ingest",
        segments_dir=tmp_path / "output" / "segments",
    )
    engine = make_engine(f"sqlite:///{tmp_path}/pipeline.db")
    init_models(engine)
    ref = noisy_bodycam.parent / "clean_reference.wav"
    d = PipelineDeps(
        sessions=make_session_factory(engine),
        settings=settings,
        video=VideoRenderSpec(w=320, h=240, fps=30, preset="ultrafast", crf=34),
        audio_mode="isolate_2bus",
        speech_enhancer=PerfectEnhancer(ref),
    )
    d.transcribe = _stub_transcribe
    d.log = lambda msg: None
    return d


def _stub_transcribe(path: Path, asset_id: str) -> Transcript:
    words = [
        WordTiming(text=f"w{i}", start_s=s0, end_s=s1, speaker="OFFICER_1")
        for i, (s0, s1) in enumerate(WORDS)
    ]
    return Transcript(
        asset_id=asset_id,
        language="en",
        duration_s=DUR_S,
        words=words,
        segments=[],
        provider="stub",
    )


@pytest.fixture()
def project(deps) -> str:
    with deps.sessions() as s:
        row = ProjectRow(title="Two Bus Case", status=ProjectStatus.created.value)
        s.add(row)
        s.commit()
        return row.id


def test_full_pipeline_two_bus_render_is_clean_and_proven(noisy_bodycam, project, deps):
    result = run_pipeline(noisy_bodycam, project, deps)

    assert result.audit.ok, [c.detail for c in result.audit.failures]
    by_name = {c.name: c for c in result.audit.checks}
    # --- the acoustic law is ON the report
    for gate in (
        "enhancer_engine_used",
        "snr_verification",
        "timbre_not_muffled",
        "zero_clipping",
        "stream_uniformity",
    ):
        assert gate in by_name, f"missing QA gate {gate}"
        assert by_name[gate].ok, f"{gate}: {by_name[gate].detail}"
    # --- the engine law: the fake's exact name is on record, afftdn nowhere
    assert "perfect_test_fake" in by_name["enhancer_engine_used"].detail

    # --- 48 kHz stereo law on the final master
    info = probe(result.final_path)
    assert info.sample_rate == 48_000 and info.channels == 2

    with deps.sessions() as s:
        asset = s.execute(select(MediaAssetRow)).scalars().one()
        assert asset.r2_enhanced_key, "enhanced audio key must be persisted"
        assert Path(asset.r2_enhanced_key).exists()

        logs = s.execute(select(AgentToolLogRow)).scalars().all()
        enhance_logs = [e for e in logs if e.tool_name == "audio.enhance_speech"]
        assert len(enhance_logs) == 1
        assert "enhancer_engine=perfect_test_fake" in enhance_logs[0].result_summary
        assert "snr_clean=" in enhance_logs[0].result_summary

    # --- A/B pair exported next to the pipeline artifacts
    ab_dir = deps.settings.output_dir / "pipeline" / project
    assert (ab_dir / "ab_raw_15s.wav").exists()
    assert (ab_dir / "ab_clean_15s.wav").exists()
    assert (ab_dir / "ab_clean_15s.mp4").exists()

    # --- rerun: enhance cache hit, still green, no second tool log
    result2 = run_pipeline(noisy_bodycam, project, deps, job_id=result.job_id)
    assert result2.audit.ok
    with deps.sessions() as s:
        logs = s.execute(select(AgentToolLogRow)).scalars().all()
        assert len([e for e in logs if e.tool_name == "audio.enhance_speech"]) == 1, (
            "cache hit must not re-run the enhancer"
        )


def test_missing_enhancer_skips_loudly_never_silently(
    noisy_bodycam, project, deps, monkeypatch
):
    """No neural enhancer available + auto resolution => the audio_prep step
    is SKIPPED (loudly — the step row + console say why and how to upgrade),
    the raw clips downgrade to loudnorm, and the job still reaches the human
    gate. The operator installs deep-filter (scripts/fetch_deep_filter.py)
    and presses Run again — finished stages are never redone. The old law
    (hard pipeline death at 52%) made every machine without the binary
    unusable; the skip keeps the no-SILENT-fallback promise via visibility,
    not via a dead render."""
    from director_os.audio_engine.enhance import EnhancerUnavailableError
    from director_os.core.db import JobStepRow

    monkeypatch.setenv("DEEPFILTER_BIN", "/definitely/not/here")
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    deps.speech_enhancer = None

    def _no_enhancer(*_a, **_kw):
        raise EnhancerUnavailableError("deep-filter CLI not found (test machine)")

    import director_os.audio_engine.enhance as enhance_mod

    monkeypatch.setattr(enhance_mod, "resolve_enhancer", _no_enhancer)

    result = run_pipeline(noisy_bodycam, project, deps)
    assert result.audit.ok, "the cut must complete without an enhancer"
    with deps.sessions() as s:
        job = s.execute(select(JobRow)).scalars().one()
        assert job.current_stage == "human_gate"
        assert job.error_message is None
        step = (
            s.execute(select(JobStepRow).where(JobStepRow.key == "audio_prep"))
            .scalars()
            .one()
        )
        assert step.status == "skipped"
        assert "loudnorm fallback" in (step.detail or "")


def test_isolate_2bus_compile_requires_master_wav(deps, tmp_path):
    """A timeline authored isolate_2bus without the master WAV refuses to compile."""
    from director_os.otio_engine.compiler import TimelineCompileError, compile_timeline
    from director_os.otio_engine.contract import RawClipMeta
    from director_os.otio_engine.segment_cache import SegmentCache
    from director_os.otio_engine.timeline import (
        ClipSpec,
        TimelineSpec,
        TrackSpec,
        build_timeline,
    )

    tl = build_timeline(
        TimelineSpec(
            name="t",
            fps=30,
            width=320,
            height=240,
            tracks=[
                TrackSpec(
                    name="V1_master",
                    items=[
                        ClipSpec(
                            name="a",
                            source_start_s=0.0,
                            dur_s=2.0,
                            meta=RawClipMeta(asset_id="a1", audio_mode="isolate_2bus"),
                        )
                    ],
                )
            ],
        )
    )
    cache = SegmentCache(tmp_path / "segs")
    with pytest.raises(TimelineCompileError, match="isolate_2bus"):
        compile_timeline(tl, assets={"a1": tmp_path / "x.mp4"}, cache=cache)
