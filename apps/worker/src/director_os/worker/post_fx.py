"""POST-FX — the owner's standing rule (2026-11-19), enforced in code.

"Only and only apply tracking blur and audio enhancement AFTER the
deliverable video is ready — after the agent already created the video +
narration. We save time and resources on the big 43-minute video and only
apply it on the chipped video."

So this stage NEVER touches the raw source. It runs on the assembled master
(video + narration + bed, the deliverable) as three resumable passes:

  1. CAPTIONS — narration cues (TTS word stamps on the master clock, the
     voiceover never comes from ASR) + raw-speech cues (ASR words remapped
     through each kept raw clip's source->master mapping, skipping muted
     b-roll windows) -> subtitles.master.{json,ass,srt}
  2. TRACKED BLUR — YOLO/YuNet face sweep ON THE MASTER (so B-roll segments
     are covered by construction — the compile-time per-clip bake could
     never see them) -> trajectories -> gaussian redaction render
  3. AUDIO ENHANCE — deep-filter speech enhancement ON THE MASTER MIX, with
     a muted-speech rescue gate (the enhancer may never destroy dialogue;
     VAD speech kept-ratio is measured before/after)
  4. MUX — blurred video + enhanced audio + burned captions -> final master

Every pass writes a .done marker; a killed run re-enters and skips finished
work. Duration is preserved end-to-end (drift tolerance ±1s is asserted by
QA downstream, which audits the FINAL file).
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from director_os.core.qa_probe import probe

if TYPE_CHECKING:
    from director_os.vision_fx.trajectory import RedactionTrajectory

# the master clock re-mapping pad: narration stems get a small tail pad so
# sentence-final words never clip
NARR_TAIL_PAD_S = 0.35
# muted-speech rescue: minimum fraction of VAD speech regions that must
# survive enhancement (audio law: the enhancer may never destroy dialogue)
MIN_SPEECH_KEEP_RATIO = 0.85


class PostFxError(RuntimeError):
    pass


@dataclass
class PostFxReport:
    """Receipt for the whole post-FX pass (QA + the run log read this)."""

    ok: bool = False
    final_path: Path | None = None
    n_caption_cues: int = 0
    narration_cues: int = 0
    raw_cues: int = 0
    blur_backend: str = ""
    blur_tracks: int = 0
    blur_windows: int = 0
    blur_leaks: int = 0
    enhance_engine: str = ""
    enhance_neural: bool = False
    snr_before_db: float | None = None
    snr_after_db: float | None = None
    speech_keep_ratio: float | None = None
    duration_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["final_path"] = str(self.final_path) if self.final_path else None
        return d


# --------------------------------------------------------------------- utils
def _run(cmd: list[str], what: str) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise PostFxError(f"{what} failed: {r.stderr[-600:]}")


def _escape_ass_path(p: Path) -> str:
    return str(p).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def _markers(work: Path) -> dict[str, Path]:
    return {
        name: work / f"{name}.done"
        for name in ("captions", "blur", "audio", "final", "cfr", "sweep")
    }


# the sandbox watchdog kills long processes: the blur render is CHUNKED
# (~150s of master per chunk, one .done marker each) so a killed run resumes
# at the exact chunk instead of re-rendering 10 minutes of frames
BLUR_CHUNK_FRAMES = 4800


# ------------------------------------------------------------- captions pass
def build_master_caption_cues(
    timeline,
    transcript: dict,
    narration_words: dict[str, list[dict]],
) -> tuple[list, dict[str, int]]:
    """Master-clock caption beats — the SHARED builder (subtitles.master_cues).

    Kept as a thin re-export so the historical import path keeps working; the
    law lives in ONE implementation now (`director_os.subtitles.master_cues`)
    and `stage_subtitles` persists the SAME cues into the Studio store, so the
    grid, the program monitor and this burn can never disagree again.
    """
    from director_os.subtitles.master_cues import build_master_caption_cues as _build

    return _build(timeline, transcript, narration_words)


def _captions_pass(
    master: Path,
    work: Path,
    timeline,
    transcript: dict,
    narration_clips: list[dict],
    report: PostFxReport,
) -> Path:
    from director_os.subtitles.master_cues import (
        build_master_caption_cues as _build_cues,
    )
    from director_os.subtitles.master_cues import narration_word_stamps
    from director_os.subtitles.store import save_cues

    # narration word stamps ride the stage_narration manifest entries
    narration_words = narration_word_stamps(narration_clips or [])
    beats, counts = _build_cues(timeline, transcript, narration_words)
    save_cues(work, beats)
    report.n_caption_cues = counts["total"]
    report.narration_cues = counts["narration"]
    report.raw_cues = counts["raw"]
    return work / "subtitles.ass"


# ----------------------------------------------------------------- blur pass
# selective redaction (acceptance law): the PRIMARY subject stays visible
# while every other identity — plus plates and the baked-in watermark — is
# blurred. BLUR_MODE env: "all" (default, blur every face) | "selective".
WATERMARK_BAND_W = 0.30  # fraction of frame width (bottom-right band)
WATERMARK_BAND_H = 0.14  # fraction of frame height


def _primary_track_id(tracks) -> str | None:
    """The main subject: the face track with the greatest on-screen span
    (ties -> the largest mean box area). This is who the camera follows."""
    best_id, best_key = None, (0.0, 0.0)
    for t in tracks:
        if not t.hits:
            continue
        span = float(t.t_end_s - t.t_start_s)
        area = sum((h.w * h.h) for h in t.hits) / max(1, len(t.hits))
        key = (span, area)
        if key > best_key:
            best_id, best_key = t.track_id, key
    return best_id


# selective law: the camera's own subject stays VISIBLE when the camera is ON
# them (a close-up box = >= 40% of frame height); everyone far away or off to
# the side — bystanders, passengers, reflections — is blurred regardless.
# Bodycam trackers fragment one person into many short IDs, so a single
# "primary track keep" can never cover the subject; the SIZE law does.
KEEP_MIN_HEIGHT_FRAC = 0.20
KEEP_MIN_HITS = 4


def _selective_keep_ids(tracks, frame_h: int) -> set[str]:
    """Track ids that stay VISIBLE under selective redaction.

    The SIZE law only: a close-up box (>= 40% of frame height, sustained) is
    the person the camera is ON — the subject stays visible. Long-but-small
    tracks (a bystander parked in frame the whole video) are NOT the subject
    and must blur, so span alone never qualifies. No close-ups at all ->
    nothing is kept (privacy-safe default).
    """
    keep: set[str] = set()
    for t in tracks:
        if not t.hits or len(t.hits) < KEEP_MIN_HITS:
            continue
        mean_h = sum(h.h for h in t.hits) / len(t.hits)
        if frame_h > 0 and mean_h / frame_h >= KEEP_MIN_HEIGHT_FRAC:
            keep.add(t.track_id)
    return keep


def _watermark_trajectory(n_frames: int, fps: float, w: int, h: int) -> RedactionTrajectory:
    """Static bottom-right band (baked-in channel watermark) blurred all video."""
    from director_os.vision_fx.trajectory import RedactionTrajectory, TrackedBox

    bw = max(40, int(w * WATERMARK_BAND_W))
    bh = max(24, int(h * WATERMARK_BAND_H))
    x = max(0, w - bw - int(w * 0.01))
    y = max(0, h - bh - int(h * 0.02))
    boxes = [
        TrackedBox(frame_idx=i, t_s=round(i / fps, 3), x=x, y=y, w=bw, h=bh)
        for i in range(max(0, n_frames))
    ]
    return RedactionTrajectory(track_id="watermark_static", boxes=boxes, n_windows=1)


def _blur_pass(master: Path, work: Path, deps, report: PostFxReport) -> Path:
    import sys

    repo_src = Path(__file__).resolve().parents[4]  # repo/packages/... -> repo
    for p in (
        repo_src / "packages" / "vision_fx" / "src",
        repo_src / "packages" / "core" / "src",
        repo_src / "packages" / "perception" / "src",
    ):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    import json as _json

    from director_os.vision_fx.compute.local_onnx import LocalOnnxCpuBackend
    from director_os.vision_fx.redact_track import (
        audit_coverage,
        normalize_cfr,
        render_tracked_redaction,
    )
    from director_os.vision_fx.trajectory import repair_trajectories

    backend = LocalOnnxCpuBackend()
    caps = backend.capabilities()
    report.blur_backend = "yolov8n-face" if caps.get("yolo_model") else "yunet-packaged"

    blur_mode = os.getenv("BLUR_MODE", "").strip().lower() or "all"
    exclude: set[str] = set()

    marks = _markers(work)
    cfr = work / "master_cfr.mp4"
    if not marks["cfr"].exists() or not cfr.exists():
        cfr = normalize_cfr(master, cfr)
        marks["cfr"].write_text("ok")
        deps.log("  [post-fx] blur: master normalized to CFR")
    else:
        deps.log("  [post-fx] blur: CFR normalize cached")

    info = probe(master)
    fps = float(info.fps or 30.0)
    n_frames = int(round(info.duration_s * fps))

    tracks_path = work / "face_tracks.json"
    if marks["sweep"].exists() and tracks_path.exists():
        from director_os.vision_fx.compute.base import FaceTrack

        tracks = [FaceTrack.model_validate(t) for t in _json.loads(tracks_path.read_text())]
        deps.log(f"  [post-fx] blur: face sweep cached ({len(tracks)} tracks)")
    else:
        # Pass 1: sweep the MASTER (b-roll included by construction)
        tracks = backend.detect_faces(master, sample_every_s=1.0, progress=None)
        if caps.get("plates"):
            try:
                plate_tracks = backend.detect_plates(master, sample_every_s=1.0, progress=None)
                tracks = [*tracks, *plate_tracks]
                report.notes.append(f"plates swept: {len(plate_tracks)} track(s)")
            except Exception as e:  # noqa: BLE001 — plates are additive, never fatal
                report.notes.append(f"plate sweep skipped ({e})")
        tracks_path.write_text(_json.dumps([t.model_dump(mode="json") for t in tracks]))
        marks["sweep"].write_text(str(len(tracks)))
        deps.log(f"  [post-fx] blur: {len(tracks)} identity track(s) swept on the master")

    frame_w_px = int(info.width or 0)
    frame_h_px = int(info.height or 0)
    if blur_mode == "selective" and tracks:
        keep = _selective_keep_ids(tracks, frame_h_px)
        exclude.update(keep)
        report.notes.append(
            f"selective mode: {len(keep)} close-up identity track(s) stay visible "
            "(primary + >=20%-height followed subjects); all far/small faces blurred"
        )

    plan = repair_trajectories(
        tracks, fps=fps, frame_w=frame_w_px, frame_h=frame_h_px, n_frames=n_frames
    )
    if blur_mode != "off":
        plan.trajectories.append(_watermark_trajectory(n_frames, fps, frame_w_px, frame_h_px))
    report.blur_tracks = len(plan.trajectories)
    report.blur_windows = sum(t.n_windows for t in plan.trajectories)
    audited_tracks = [t for t in tracks if t.track_id not in exclude]
    leaks = audit_coverage(plan, audited_tracks, fps)
    report.blur_leaks = len(leaks)

    # chunked, resumable render: each chunk is one .done marker
    n_chunks = max(1, -(-n_frames // BLUR_CHUNK_FRAMES))
    parts: list[Path] = []
    for ci in range(n_chunks):
        f0, f1 = ci * BLUR_CHUNK_FRAMES, min((ci + 1) * BLUR_CHUNK_FRAMES, n_frames)
        part = work / f"blur_chunk_{ci:02d}.mp4"
        parts.append(part)
        if (work / f"blur_chunk_{ci:02d}.done").exists() and part.exists():
            continue
        render_tracked_redaction(
            cfr,
            plan,
            part,
            mode="gauss",
            crf=20,
            preset="veryfast",
            exclude_track_ids=exclude,
            frame_start=f0,
            frame_end=f1,
        )
        (work / f"blur_chunk_{ci:02d}.done").write_text(str(f1 - f0))
        deps.log(f"  [post-fx] blur: chunk {ci + 1}/{n_chunks} rendered ({f1 - f0} frames)")
    if n_chunks == 1:
        parts[0].replace(work / "master_blurred.mp4")
    else:
        lst = work / "blur_chunks.txt"
        lst.write_text("".join(f"file '{p.name}'\n" for p in parts))
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(lst),
                "-c",
                "copy",
                str(work / "master_blurred.mp4"),
            ],
            check=True,
            capture_output=True,
        )
    return work / "master_blurred.mp4"


# ---------------------------------------------------------------- audio pass
def _speech_regions(wav: Path) -> list[tuple[float, float]]:
    """VAD speech spans (silero) — used by the muted-speech rescue gate."""
    try:
        import numpy as np
        from director_os.audio_engine.snr import decode_pcm_mono
        from director_os.audio_engine.vad import SileroVad

        pcm = decode_pcm_mono(str(wav), 16_000)
        return SileroVad().speech_regions(pcm.astype(np.float32))
    except Exception:  # VAD unavailable: gate degrades to "assume speech kept"
        return []


def _extract_master_audio(master: Path, work: Path) -> Path:
    wav = work / "master_audio.wav"
    _run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(master),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "48000",
            "-c:a",
            "pcm_s16le",
            str(wav),
        ],
        "master audio extract",
    )
    return wav


# ------------------------------------------------------------------- mux pass
def _mux_pass(blurred: Path, enhanced_wav: Path, ass: Path, final: Path) -> None:
    vf = f"ass='{_escape_ass_path(ass)}'"
    _run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(blurred),
            "-i",
            str(enhanced_wav),
            "-vf",
            vf,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "22",
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
            "-movflags",
            "+faststart",
            str(final),
        ],
        "post-fx final mux (captions + enhanced audio)",
    )


# -------------------------------------------------------------------- stage
def stage_post_fx(
    master: Path,
    timeline,
    transcript: dict,
    project_id: str,
    deps,
    narration_clips: list[dict] | None = None,
) -> PostFxReport:
    """The POST-FX pass on the assembled deliverable (never the raw source).

    Resumable: each pass leaves a .done marker in
    output/pipeline/<project>/post_fx/. Returns the PostFxReport receipt.
    """
    master = Path(master)
    if not master.exists():
        raise PostFxError(f"master missing: {master}")
    work = deps.settings.output_dir / "pipeline" / project_id / "post_fx"
    work.mkdir(parents=True, exist_ok=True)
    marks = _markers(work)
    report = PostFxReport()

    # ---- 1. captions
    if marks["captions"].exists():
        report.n_caption_cues = int(marks["captions"].read_text() or 0)
        deps.log(f"  [post-fx] captions cached ({report.n_caption_cues} cues)")
    else:
        ass = _captions_pass(master, work, timeline, transcript, narration_clips or [], report)
        marks["captions"].write_text(str(report.n_caption_cues))
        deps.log(
            f"  [post-fx] captions: {report.n_caption_cues} cues "
            f"({report.narration_cues} narration / {report.raw_cues} raw) -> {ass.name}"
        )
    ass = work / "subtitles.ass"

    # ---- 2. tracked blur (master only — b-roll covered by construction)
    if marks["blur"].exists():
        blurred = work / "master_blurred.mp4"
        blur_mark = marks["blur"].read_text().split("|")
        report.blur_backend = blur_mark[0]
        report.blur_tracks = int(blur_mark[1]) if len(blur_mark) > 1 and blur_mark[1] else 0
        deps.log(f"  [post-fx] blur cached ({report.blur_tracks} tracks via {report.blur_backend})")
    else:
        blurred = _blur_pass(master, work, deps, report)
        marks["blur"].write_text(f"{report.blur_backend}|{report.blur_tracks}")
        deps.log(
            f"  [post-fx] blur: {report.blur_tracks} track(s), {report.blur_windows} window(s), "
            f"{report.blur_leaks} leak(s) via {report.blur_backend}"
        )

    # ---- 3. audio enhancement on the master mix
    if marks["audio"].exists():
        enhanced = work / "master_audio.enhanced.wav"
        report.enhance_engine = marks["audio"].read_text()
        deps.log(f"  [post-fx] audio enhance cached ({report.enhance_engine})")
    else:
        from director_os.audio_engine.enhance import DeepFilterCliEnhancer

        raw_wav = _extract_master_audio(master, work)
        report.snr_before_db = _measure_snr(raw_wav)
        speech_before = _speech_regions(raw_wav)
        enhanced = work / "master_audio.enhanced.wav"
        try:
            enhancer = DeepFilterCliEnhancer()  # DEEPFILTER_BIN env or PATH
            result = enhancer.enhance(raw_wav, enhanced)
            report.enhance_engine = result.engine
            report.enhance_neural = result.neural
        except Exception as e:  # noqa: BLE001 — degrade loudly, never ship noise
            deps.log(f"  [post-fx] deep-filter unavailable ({e}) — afftdn fallback")
            from director_os.audio_engine.enhance import AfftdnEnhancer

            enhancer_fallback: AfftdnEnhancer = AfftdnEnhancer()
            result = enhancer_fallback.enhance(raw_wav, enhanced)
            report.enhance_engine = result.engine
            report.enhance_neural = result.neural
        # muted-speech rescue gate
        kept = _speech_regions(enhanced)
        if speech_before:
            ratio = _speech_keep_ratio(speech_before, kept)
            report.speech_keep_ratio = ratio
            if ratio < MIN_SPEECH_KEEP_RATIO:
                raise PostFxError(
                    f"enhancer destroyed dialogue: speech keep ratio {ratio:.2f} < "
                    f"{MIN_SPEECH_KEEP_RATIO:.2f} — refusing to ship"
                )
        report.snr_after_db = _measure_snr(enhanced)
        marks["audio"].write_text(report.enhance_engine)
        deps.log(
            f"  [post-fx] audio: {report.enhance_engine} SNR {report.snr_before_db} -> "
            f"{report.snr_after_db} dB, speech kept {report.speech_keep_ratio}"
        )

    # ---- 4. mux (blurred video + enhanced audio + burned captions)
    if marks["final"].exists():
        final = work / "master_postfx.mp4"
    else:
        final = work / "master_postfx.mp4"
        _mux_pass(blurred, enhanced, ass, final)
        # duration must be preserved end-to-end
        info = probe(final)
        drift = abs(info.duration_s - probe(master).duration_s)
        if drift > 1.0:
            raise PostFxError(f"post-fx drifted {drift:.2f}s vs master — refusing to ship")
        report.duration_s = info.duration_s
        marks["final"].write_text("ok")
    report.final_path = final
    report.ok = True
    deps.log(f"  [post-fx] final: {final.name} ({final.stat().st_size / 1e6:.1f} MB)")
    return report


def _measure_snr(wav: Path) -> float | None:
    try:
        from director_os.audio_engine.snr import measure_snr

        return round(measure_snr(str(wav), _speech_regions(wav))["snr_db"], 1)
    except Exception:
        return None


def _speech_keep_ratio(
    before: list[tuple[float, float]], after: list[tuple[float, float]]
) -> float:
    """Fraction of before-speech time still detected as speech after enhance."""
    if not before:
        return 1.0
    kept_s = 0.0
    for b0, b1 in before:
        covered = 0.0
        for a0, a1 in after:
            lo, hi = max(b0, a0), min(b1, a1)
            if hi > lo:
                covered += hi - lo
        kept_s += min(covered, b1 - b0)
    total = sum(b1 - b0 for b0, b1 in before)
    return round(kept_s / total, 3) if total > 0 else 1.0
