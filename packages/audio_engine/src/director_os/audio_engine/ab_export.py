"""15-second A/B noise-reduction export — hear + measure the 2-bus difference.

Metrics prove SNR moved; ears still settle disputes. `export_ab_pair` picks
the 15 s window with the most ASR speech (deterministic: first window wins
ties), slices raw and clean audio at EXACTLY the same window, and writes
`ab_raw_15s.wav` + `ab_clean_15s.wav` (plus muxed MP4 pairs when a video is
given) so integration tests and review can A/B the noise reduction directly.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

WINDOW_S = 15.0


class ABExportError(RuntimeError):
    pass


@dataclass
class ABExport:
    window_start_s: float
    window_end_s: float
    raw_path: Path
    clean_path: Path
    raw_video_path: Path | None = None
    clean_video_path: Path | None = None


def best_speech_window(
    speech_intervals: list[tuple[float, float]],
    duration_s: float,
    window_s: float = WINDOW_S,
) -> tuple[float, float]:
    """Densest-speech 15 s window; first-wins ties; whole file when no ASR."""
    if duration_s <= window_s + 1e-6:
        return 0.0, duration_s
    if not speech_intervals:
        return 0.0, window_s
    # candidate windows anchored at each speech start (and t=0) — plenty dense
    starts = sorted({0.0} | {max(0.0, s - 1.0) for s, _ in speech_intervals})
    best: tuple[float, int] = (0.0, -1)
    for st in starts:
        end = min(st + window_s, duration_s)
        if end - st < min(window_s, duration_s) * 0.8:
            continue
        coverage = sum(
            max(0.0, min(e, end) - max(s, st)) for s, e in speech_intervals if e > st and s < end
        )
        if coverage > best[1]:
            best = (st, int(coverage * 1000))
    st = best[0]
    end = min(st + window_s, duration_s)
    return st, end


def _run(cmd: list[str], what: str) -> None:
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise ABExportError(f"{what} failed: {r.stderr[-300:].decode(errors='replace')}")


def _slice_audio(src: Path, start_s: float, dur_s: float, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_s:.6f}",
            "-t",
            f"{dur_s:.6f}",
            "-i",
            str(src),
            "-map",
            "0:a:0",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(out),
        ],
        what=f"ab slice {src.name}",
    )
    return out


def _mux_video(video: Path, audio: Path, start_s: float, dur_s: float, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_s:.6f}",
            "-t",
            f"{dur_s:.6f}",
            "-i",
            str(video),
            "-i",
            str(audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "28",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-shortest",
            str(out),
        ],
        what=f"ab mux {video.name}",
    )
    return out


def export_ab_pair(
    raw_audio: Path,
    clean_audio: Path,
    out_dir: Path,
    speech_intervals: list[tuple[float, float]] | None = None,
    duration_s: float | None = None,
    video_path: Path | None = None,
    window_s: float = WINDOW_S,
) -> ABExport:
    """Raw-vs-clean A/B pair over the identical window (WAVs + optional MP4s)."""
    from director_os.core.qa_probe import probe

    raw_audio, clean_audio, out_dir = Path(raw_audio), Path(clean_audio), Path(out_dir)
    dur = duration_s if duration_s is not None else probe(clean_audio).duration_s
    st, end = best_speech_window(speech_intervals or [], dur, window_s)
    dur_s = end - st
    raw_ab = _slice_audio(raw_audio, st, dur_s, out_dir / "ab_raw_15s.wav")
    clean_ab = _slice_audio(clean_audio, st, dur_s, out_dir / "ab_clean_15s.wav")
    raw_v = clean_v = None
    if video_path is not None and Path(video_path).exists():
        raw_v = _mux_video(Path(video_path), raw_ab, st, dur_s, out_dir / "ab_raw_15s.mp4")
        clean_v = _mux_video(Path(video_path), clean_ab, st, dur_s, out_dir / "ab_clean_15s.mp4")
    return ABExport(
        window_start_s=st,
        window_end_s=end,
        raw_path=raw_ab,
        clean_path=clean_ab,
        raw_video_path=raw_v,
        clean_video_path=clean_v,
    )
