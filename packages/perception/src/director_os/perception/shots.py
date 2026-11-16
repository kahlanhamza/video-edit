"""perception v1 — shot boundaries + audio events (PLAN §4.5, M2).

Shot detection: PySceneDetect (ContentDetector) when installed, otherwise a
zero-dependency ffmpeg scene-score path (`select='gt(scene,t)'` + showinfo).
Both backends return the same normalized `ShotDetection`, so callers never
branch on the provider.

Audio events: ffmpeg `silencedetect` inverted — everything that is NOT
silence above a noise floor is an event (speech bursts, gunshots, sirens).
Deterministic on synthetic inputs, hermetic, and pinned by tests.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field

from director_os.core.qa_probe import probe

DEFAULT_SCENE_THRESHOLD = 0.3


class PerceptionError(RuntimeError):
    pass


@dataclass(frozen=True)
class Shot:
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class ShotDetection:
    source: str
    backend: str
    threshold: float
    shots: list[Shot] = field(default_factory=list)

    @property
    def cut_points(self) -> list[float]:
        """Interior cut times (shot starts after the first)."""
        return [s.start_s for s in self.shots[1:]]


@dataclass(frozen=True)
class AudioEvent:
    start_s: float
    end_s: float
    kind: str = "sound"

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


# --------------------------------------------------------------------- shots
def detect_shots(
    video_path: str,
    threshold: float = DEFAULT_SCENE_THRESHOLD,
    backend: str = "auto",
) -> ShotDetection:
    """Detect shot boundaries; `backend` = auto | pyscenedetect | ffmpeg."""
    duration = probe(video_path).duration_s
    use_psd = backend == "pyscenedetect" or (backend == "auto" and _has_pyscenedetect())
    if use_psd:
        cuts = _cuts_pyscenedetect(video_path, threshold)
        used = "pyscenedetect"
    else:
        cuts = _cuts_ffmpeg_scene(video_path, threshold)
        used = "ffmpeg-scene"
    cuts = sorted(c for c in cuts if 0.0 < c < duration)
    bounds = [0.0, *cuts, duration]
    shots = [Shot(start_s=a, end_s=b) for a, b in zip(bounds, bounds[1:], strict=False) if b > a]
    return ShotDetection(source=str(video_path), backend=used, threshold=threshold, shots=shots)


def _has_pyscenedetect() -> bool:
    try:
        import scenedetect  # noqa: F401
    except ImportError:
        return False
    return True


def _cuts_ffmpeg_scene(video_path: str, threshold: float) -> list[float]:
    """Interior cut times from ffmpeg scene-score select + showinfo pts."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        video_path,
        "-vf",
        f"select='gt(scene,{threshold:.3f})',showinfo",
        "-an",
        "-f",
        "null",
        "-",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise PerceptionError(f"scene detect failed: {r.stderr[-400:]}")
    cuts: list[float] = []
    for line in r.stderr.splitlines():
        m = re.search(r"showinfo.*pts_time:([0-9]+\.?[0-9]*)", line)
        if m:
            cuts.append(float(m.group(1)))
    return cuts


def _cuts_pyscenedetect(video_path: str, threshold: float) -> list[float]:
    try:
        from scenedetect import ContentDetector, detect, video_manager  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise PerceptionError("pyscenedetect requested but not installed") from e

    vm = video_manager.VideoManager([video_path])
    scene_list = detect(vm, ContentDetector(threshold=threshold * 100.0))
    return [s[0].get_seconds() for s in scene_list[1:]] if scene_list else []


# --------------------------------------------------------------- audio events
def detect_audio_events(
    audio_or_video_path: str,
    noise_db: float = -30.0,
    min_silence_s: float = 0.2,
) -> list[AudioEvent]:
    """Non-silent intervals (events) = complement of detected silence."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        audio_or_video_path,
        "-af",
        f"silencedetect=noise={noise_db:g}dB:d={min_silence_s:g}",
        "-vn",
        "-f",
        "null",
        "-",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise PerceptionError(f"silence detect failed: {r.stderr[-400:]}")

    silences: list[tuple[float, float | None]] = []
    start: float | None = None
    for line in r.stderr.splitlines():
        m = re.search(r"silence_start:\s*(-?[0-9]+\.?[0-9]*)", line)
        if m:
            start = max(float(m.group(1)), 0.0)
        m = re.search(r"silence_end:\s*(-?[0-9]+\.?[0-9]*)", line)
        if m and start is not None:
            silences.append((start, float(m.group(1))))
            start = None
    if start is not None:
        silences.append((start, None))  # silence ran to the end

    duration = probe(audio_or_video_path).duration_s
    loud: list[tuple[float, float]] = []
    cursor = 0.0
    for s_start, s_end in silences:
        if s_start > cursor:
            loud.append((cursor, s_start))
        cursor = s_end if s_end is not None else duration
    if cursor < duration:
        loud.append((cursor, duration))

    return [
        AudioEvent(start_s=a, end_s=b)
        for a, b in loud
        if b - a > 0.05  # ignore sub-frame jitter
    ]
