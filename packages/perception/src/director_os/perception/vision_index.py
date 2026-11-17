"""Two-tier Gemini vision stack (PLAN §4.6, M4) — cheap indexing + precision QA.

The VISION_API_KEY budget is spent on two jobs, never one fat model:

- Tier 1 — INDEX: `build_contact_sheets` stitches 6-9 frames from every
  `window_s` slice into one timestamped grid and the LITE model
  (`gemini-3.5-flash-lite`, the fast/low-cost tier — 2.5-flash deprecates
  2026-12-01) turns each sheet into a timestamped visual timeline for pennies.
- Tier 2 — QA / Frames-on-Demand: when the Director wants to place a
  freeze-frame, punch-in zoom or graphic card, `frames_on_demand` pulls full-res
  frames around that exact timestamp and the FLAGSHIP model
  (`gemini-3.8-flash`) verifies visibility — that a card will not cover a face
  or the AXON overlay, that the shot is actually usable.

Both tiers ride the same `VisionAdapter`/`GeminiVisionClient` port (hermetic
fakes keep every test offline); only the model name differs.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from director_os.core.fonts import pick_font
from director_os.core.qa_probe import probe
from director_os.perception.vision import (
    FakeVisionClient,
    FrameRef,
    GeminiVisionClient,
    SceneInsight,
    VisionAdapter,
)

DEFAULT_INDEX_MODEL = "gemini-3.5-flash-lite"
DEFAULT_QA_MODEL = "gemini-3.8-flash"

INDEX_PROMPT = (
    "This is a contact sheet: N frames from one {window:.0f}s window of police "
    "bodycam footage, left-to-right top-to-bottom, each stamped with its "
    "timestamp. Describe what happens in order as a JSON array of objects: "
    '[{{"timestamp_s": float, "kind": "observation|risk|redaction_candidate|event", '
    '"description": str (one concrete sentence: who/what/where), '
    '"confidence": float 0..1}}]. Flag risks: obscured lens, dark frames, '
    "third-party faces or plates that need redaction, decisive moments."
)

QA_PROMPT = (
    "You are QA-critic for a documentary edit. Frames around {t_s:.2f}s are "
    "attached at full resolution. Proposed edit: {proposal}. Verify concretely: "
    "(1) what is visible at that moment, (2) whether the proposed overlay/card/"
    "zoom would cover a face, a license plate, or the AXON BODY timestamp "
    "overlay, (3) whether the frame is too dark/blurry/obstructed to use. "
    "Answer ONLY with a JSON array of objects: "
    '[{{"timestamp_s": float, "kind": "observation|risk|redaction_candidate|event", '
    '"description": str, "confidence": float 0..1}}].'
)


class VisionIndexError(RuntimeError):
    pass


def index_model() -> str:
    return os.getenv("VISION_INDEX_MODEL", DEFAULT_INDEX_MODEL)


def qa_model() -> str:
    return os.getenv("VISION_QA_MODEL", DEFAULT_QA_MODEL)


def make_vision_clients(api_key: str) -> tuple[VisionAdapter, VisionAdapter]:
    """(index_adapter, qa_adapter) — lite tier + flagship tier; blank key -> fakes."""
    if not api_key:
        return VisionAdapter(FakeVisionClient()), VisionAdapter(FakeVisionClient())
    return (
        VisionAdapter(GeminiVisionClient(api_key, model=index_model())),
        VisionAdapter(GeminiVisionClient(api_key, model=qa_model())),
    )


# ---------------------------------------------------------------- contact sheets
@dataclass(frozen=True)
class ContactSheet:
    """One stitched 3x3 grid covering [t_start_s, t_end_s]."""

    path: Path
    t_start_s: float
    t_end_s: float
    frame_times: tuple[float, ...] = ()


def _sheet_grid(grid: tuple[int, int]) -> int:
    return grid[0] * grid[1]


def build_contact_sheets(
    video_path: str,
    out_dir: str | Path,
    window_s: float = 30.0,
    grid: tuple[int, int] = (3, 3),
    tile_w: int = 320,
    start_s: float = 0.0,
    end_s: float | None = None,
) -> list[ContactSheet]:
    """Stitch one grid image per `window_s` slice; deterministic frame times.

    Frames are sampled evenly inside each window (never exactly on the window
    edge) and burned with their media timestamp so the model can reason in
    absolute time. Output PNGs are content-stable for a given (source, layout).
    """

    from PIL import Image, ImageDraw

    dur = probe(video_path).duration_s
    end_s = dur if end_s is None else min(end_s, dur)
    if end_s - start_s < 0.5:
        raise VisionIndexError(f"window too short: {end_s - start_s:.2f}s")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = _sheet_grid(grid)
    font_path = pick_font("DejaVu Sans")
    sheets: list[ContactSheet] = []

    win = window_s
    k = 0
    t = start_s
    while t < end_s - 0.01:
        w_end = min(t + win, end_s)
        span = w_end - t
        frame_times = tuple(round(t + span * (i + 0.5) / n, 2) for i in range(n))
        # extract the window's frames at fixed even spacing
        tmp_dir = out / f".tmp_sheet_{k}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{t:.3f}",
            "-t",
            f"{span:.3f}",
            "-i",
            str(video_path),
            "-vf",
            f"fps={n / span:.4f},scale={tile_w}:-2",
            "-frames:v",
            str(n),
            "-q:v",
            "3",
            str(tmp_dir / "f_%02d.jpg"),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise VisionIndexError(f"sheet frame extract failed: {r.stderr[-300:]}")
        files = sorted(tmp_dir.glob("f_*.jpg"))[:n]
        if not files:
            raise VisionIndexError(f"no frames extracted for window {t:.1f}-{w_end:.1f}s")

        tiles: list = []
        for i, f in enumerate(files):
            img = Image.open(f).convert("RGB")
            tiles.append((img, frame_times[i] if i < len(frame_times) else t))
        tile_h = max(im.height for im, _ in tiles) or tile_w // 2

        sheet = Image.new("RGB", (tile_w * grid[0], tile_h * grid[1]), (10, 10, 12))
        draw = ImageDraw.Draw(sheet)
        for i, (im, ts) in enumerate(tiles):
            gx, gy = i % grid[0], i // grid[0]
            sheet.paste(im, (gx * tile_w, gy * tile_h))
            label = f"{int(ts // 60):02d}:{ts % 60:04.1f}"
            box = [gx * tile_w + 4, gy * tile_h + 4, gx * tile_w + 78, gy * tile_h + 22]
            draw.rectangle(box, fill=(0, 0, 0))
            draw.text(
                (gx * tile_w + 8, gy * tile_h + 6),
                label,
                font=_pfont(font_path, 13),
                fill=(255, 255, 80),
            )

        path = out / f"sheet_{k:04d}_{int(t):05d}.png"
        sheet.save(path)
        for f in files:
            f.unlink(missing_ok=True)
        tmp_dir.rmdir()
        sheets.append(
            ContactSheet(
                path=path, t_start_s=round(t, 3), t_end_s=round(w_end, 3), frame_times=frame_times
            )
        )
        t += win
        k += 1
    return sheets


_PF_CACHE: dict[tuple[str, int], object] = {}


def _pfont(path: Path, size: int):
    from PIL import ImageFont

    key = (str(path), size)
    if key not in _PF_CACHE:
        _PF_CACHE[key] = ImageFont.truetype(str(path), size)
    return _PF_CACHE[key]


def index_timeline(
    adapter: VisionAdapter,
    sheets: list[ContactSheet],
    window_s: float = 30.0,
) -> list[SceneInsight]:
    """Tier 1: one lite-model call per sheet -> timestamped visual timeline."""
    insights: list[SceneInsight] = []
    for sheet in sheets:
        frame = FrameRef(timestamp_s=sheet.t_start_s, image_path=sheet.path)
        found = adapter.understand(INDEX_PROMPT.format(window=window_s), [frame])
        for ins in found:
            # clamp sheet-relative model answers into the sheet's absolute window
            if ins.timestamp_s < sheet.t_start_s - 1e-6 or ins.timestamp_s > sheet.t_end_s + 1e-6:
                insights.append(ins.model_copy(update={"timestamp_s": sheet.t_start_s}))
            else:
                insights.append(ins)
    return insights


# ---------------------------------------------------------------- frames on demand
def frames_on_demand(
    video_path: str,
    t_s: float,
    n: int = 5,
    span_s: float = 4.0,
    out_dir: str | Path = "/tmp/dos_fod",
    tile_w: int = 960,
) -> list[FrameRef]:
    """Tier 2 input: `n` full-res frames evenly spread across [t-span/2, t+span/2]."""
    dur = probe(video_path).duration_s
    lo = max(0.0, t_s - span_s / 2.0)
    hi = min(dur, t_s + span_s / 2.0)
    if hi - lo < 0.2:
        raise VisionIndexError(f"frames_on_demand span too short near {t_s:.2f}s")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    times = [round(lo + (hi - lo) * (i + 0.5) / n, 3) for i in range(n)]
    refs: list[FrameRef] = []
    for i, ts in enumerate(times):
        path = out / f"fod_{int(ts * 1000):09d}_{i}.jpg"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{ts:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-vf",
            f"scale={tile_w}:-2",
            "-q:v",
            "2",
            "-y",
            str(path),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not path.exists():
            continue  # a missing boundary frame is non-fatal — QA uses what it got
        refs.append(FrameRef(timestamp_s=ts, image_path=path))
    if not refs:
        raise VisionIndexError(f"no frames extracted around {t_s:.2f}s")
    return refs


def qa_placement(
    qa_adapter: VisionAdapter, frames: list[FrameRef], proposal: str, t_s: float
) -> list[SceneInsight]:
    """Tier 2: flagship-model verification of one proposed edit placement."""
    if not frames:
        raise VisionIndexError("qa_placement needs frames (frames_on_demand first)")
    return qa_adapter.understand(QA_PROMPT.format(t_s=t_s, proposal=proposal), frames)


def suggest_fod_timestamps(
    timeline_insights: list[SceneInsight],
    kinds: tuple[str, ...] = ("risk", "redaction_candidate", "event"),
) -> list[float]:
    """Timestamps worth a Frames-on-Demand pass — decisive/risky moments first."""
    picked = [i.timestamp_s for i in timeline_insights if i.kind in kinds and i.confidence >= 0.5]
    # dedupe within 5s, keep order
    out: list[float] = []
    for t in sorted(picked):
        if all(abs(t - o) > 5.0 for o in out):
            out.append(round(t, 2))
    return out
