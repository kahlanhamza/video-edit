"""Stage 10 — Creator Ship Pack (PLAN §7, human-gate close-out).

YouTube auto-upload is DISABLED, so "publish" is a human act — Stage 10's job
is to make that act a copy-paste of one folder:

    ship_pack/
      final_1080p.mp4     the QA-passed master (clean first/last frames, verified)
      chapters.txt        YouTube chapter list from the timeline's cards+VO cues
      captions.srt        the transcript captions (identical to the baked .ass)
      seo.md              title / description / outcome summary / keywords
      metadata.json       machine-readable mirror of everything above
      thumbnails/         ranked candidate frames (sharpness x luma scored)
      timeline.otio       the editable source of truth (re-import anywhere)

Everything is deterministic and $0: chapters derive from overlay cards +
narration cues, SEO from the AXON overlay OCR + visual index + transcript,
thumbnails from scored frame extracts. The manifest carries the Content-ID
risk flag from the music guardrail so a flagged song can never be "forgotten"
between the mix and the upload.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from director_os.core.models import JobStage  # noqa: F401 — stage identity lives here
from director_os.otio_engine.timeline import (
    MASTER_TRACK,
    OVERLAY_TRACK,
    is_gap,
    item_duration_s,
    track_by_name,
)


class ShipPackError(RuntimeError):
    pass


@dataclass
class ShipPackResult:
    pack_dir: Path
    files: dict[str, Path] = field(default_factory=dict)
    chapters: list[tuple[float, str]] = field(default_factory=list)
    seo: dict = field(default_factory=dict)
    verification: dict = field(default_factory=dict)
    content_id_risk: dict = field(default_factory=dict)


# ------------------------------------------------------------------ helpers
def _extract_frame(video: Path, t_s: float, dst: Path, h: int = 720) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{max(0.0, t_s):.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-vf",
            f"scale=-2:{h}",
            "-q:v",
            "3",
            str(dst),
        ],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def _frame_score(path: Path) -> tuple[float, float]:
    """(luma 0..255, sharpness) — PIL-only, deterministic."""
    from PIL import Image

    with Image.open(path).convert("L") as im:
        px = list(im.getdata())
        n = len(px) or 1
        luma = sum(px) / n
        w, h = im.size
        # cheap laplacian-ish gradient energy on a 64-wide downsample
        small = im.resize((64, max(1, int(64 * h / w))))
        sp = list(small.getdata())
        sw = small.width
        energy = 0.0
        for i in range(sw, len(sp) - sw):
            energy += abs(sp[i] - sp[i - 1]) + abs(sp[i] - sp[i + sw])
        sharpness = energy / max(1, len(sp))
    return luma, sharpness


def first_last_frame_luma(video: Path) -> tuple[float, float]:
    """Mean luma of the first and last visible frames — the 'clean ends' check."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        first = Path(td) / "first.jpg"
        last = Path(td) / "last.jpg"
        _extract_frame(video, 0.0, first)
        _extract_frame(video, -0.4, last)  # -ss before input = last-frame seek
        if not first.exists() or not last.exists():
            raise ShipPackError("could not extract first/last frame for the clean-ends check")
        return _frame_score(first)[0], _frame_score(last)[0]


def _fmt_ts(t_s: float) -> str:
    t = max(0, int(round(t_s)))
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def chapter_markers(timeline) -> list[tuple[float, str]]:
    """Cards + narration cues in timeline order -> (t_s, label).

    YouTube chapters REQUIRE the first entry at 0:00 — a cold-open entry is
    prepended when the timeline starts elsewhere.
    """
    markers: list[tuple[float, str]] = []
    overlay = track_by_name(timeline, OVERLAY_TRACK)
    if overlay is not None:
        t = 0.0
        from director_os.otio_engine.contract import CardMeta, parse_clip_meta

        for item in overlay:
            dur = item_duration_s(item)
            if dur <= 0 or is_gap(item):
                t += max(0.0, dur)
                continue
            try:
                meta = parse_clip_meta(item)
            except Exception:  # noqa: BLE001 — a corrupt card never kills the pack
                t += dur
                continue
            if isinstance(meta, CardMeta):
                label = str(
                    meta.fields.get("title")
                    or meta.fields.get("subtitle")
                    or meta.template.replace("_", " ").title()
                )
                markers.append((round(t, 2), label))
            t += dur
    master = track_by_name(timeline, MASTER_TRACK)
    if master is not None:
        t = 0.0
        from director_os.otio_engine.contract import NarratorClipMeta, parse_clip_meta

        for item in master:
            dur = item_duration_s(item)
            if dur <= 0 or is_gap(item):
                t += max(0.0, dur)
                continue
            try:
                meta = parse_clip_meta(item)
            except Exception:  # noqa: BLE001
                t += dur
                continue
            if isinstance(meta, NarratorClipMeta):
                words = " ".join(meta.script_text.split()[:6])
                markers.append((round(t, 2), f"Narration: {words}"))
            t += dur
    markers.sort(key=lambda m: m[0])
    if not markers or markers[0][0] > 0.01:
        markers.insert(0, (0.0, "Cold open"))
    return markers


def build_seo(
    project_title: str,
    transcript: dict,
    insights: list[dict],
    card_subtitle: str | None,
    chapters: list[tuple[float, str]],
    duration_s: float,
) -> dict:
    """Deterministic SEO pack; `SHIP_SEO_LLM=1` adds an optional LLM polish pass."""
    location = ""
    if card_subtitle and "—" in card_subtitle:
        parts = [p.strip() for p in card_subtitle.split("—")]
        if len(parts) >= 3:
            location = parts[-1]
    hooks = [
        str(i.get("description", "")).strip()
        for i in insights
        if i.get("kind") in {"event", "risk"} and i.get("description")
    ][:3]
    words = [w.get("text", "") for w in (transcript.get("words") or [])[:80]]
    outcome = " ".join(words[-40:]) if words else ""

    title = f"{project_title} — Full Bodycam Breakdown"
    if location:
        title = f"{project_title} ({location}) — Full Bodycam Breakdown"

    lines = [
        title,
        "",
        "What happens in this footage, minute by minute:",
        *(f"- {h}" for h in hooks if h),
        "",
        f"Raw footage duration: {duration_s / 60.0:.1f} min. "
        "Faces, plates and bystander identities are redacted; speech is "
        "isolated and enhanced for clarity.",
        "",
        "Chapters:",
        *(f"{_fmt_ts(t)} {label}" for t, label in chapters),
    ]
    description = "\n".join(lines)
    keywords = sorted(
        {
            "bodycam",
            "police footage",
            "case file",
            *(location.lower() for _ in [1] if location),
            *(h.lower()[:40] for h in hooks if h),
        }
    )[:12]
    seo = {
        "title": title[:100],
        "description": description[:4900],
        "outcome_summary": outcome[:600],
        "keywords": keywords,
        "generator": "deterministic_v1",
    }
    if os_polish_flag():
        polished = polish_seo_llm(seo)
        if polished:
            seo.update(polished)
            seo["generator"] = "llm_polished+deterministic_fallback"
    return seo


def os_polish_flag() -> bool:
    import os

    return os.getenv("SHIP_SEO_LLM", "").strip().lower() in {"1", "true", "yes"}


def polish_seo_llm(seo: dict) -> dict | None:
    """Optional LLM polish — native Gemini REST (LLM_API_KEY, blank base URL).

    Any failure returns None (deterministic SEO already shipped). Cost: one
    small completion per Ship Pack build.
    """
    try:
        import httpx
        from director_os.core.config import get_settings

        s = get_settings()
        if not (s.llm_api_key and not s.llm_base_url):
            return None
        prompt = (
            "Rewrite this YouTube metadata for a bodycam documentary channel. "
            "Answer ONLY with JSON keys title, description, keywords (array). "
            "Keep it factual — do not invent events.\n\n"
            f"TITLE: {seo['title']}\n\nDESCRIPTION:\n{seo['description'][:1500]}"
        )
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.4, "maxOutputTokens": 1200},
        }
        with httpx.Client(timeout=60) as client:
            r = client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{s.llm_model}:generateContent",
                headers={"x-goog-api-key": s.llm_api_key},
                json=body,
            )
        if r.status_code != 200:
            return None
        text = "".join(
            p.get("text", "")
            for p in r.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
        ).strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        data = json.loads(text)
        out = {"title": str(data.get("title", ""))[:100] or seo["title"]}
        desc = str(data.get("description", ""))[:4900]
        if desc:
            out["description"] = desc
        if isinstance(data.get("keywords"), list):
            out["keywords"] = [str(k) for k in data["keywords"]][:12]
        return out
    except Exception:  # noqa: BLE001 — polish is strictly additive
        return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ builder
def build_ship_pack(
    project_id: str,
    deps,
    timeline,
    final_path: Path,
    transcript: dict | None = None,
    insights: list[dict] | None = None,
    card_subtitle: str | None = None,
    enhance_info=None,  # EnhanceInfo | None
    duration_s: float = 0.0,
    project_title: str | None = None,
) -> ShipPackResult:
    """Assemble the Creator Ship Pack next to the pipeline artifacts (idempotent)."""
    out_dir = Path(deps.settings.output_dir) / "pipeline" / project_id / "ship_pack"
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}

    final = out_dir / "final_1080p.mp4"
    final_path = Path(final_path)
    if final_path.resolve() != final.resolve():
        _run_copy(final_path, final)
    files["final_1080p.mp4"] = final

    # chapters
    chapters = chapter_markers(timeline)
    (out_dir / "chapters.txt").write_text(
        "\n".join(f"{_fmt_ts(t)} {label}" for t, label in chapters) + "\n", encoding="utf-8"
    )
    files["chapters.txt"] = out_dir / "chapters.txt"

    # captions (subtitles stage product — same cues the .ass baked)
    srt_src = Path(deps.settings.output_dir) / "pipeline" / project_id / "subtitles.srt"
    srt = out_dir / "captions.srt"
    if srt_src.exists():
        _run_copy(srt_src, srt)
    else:
        srt.write_text("", encoding="utf-8")
    files["captions.srt"] = srt

    # otio source of truth
    otio_src = Path(deps.settings.output_dir) / "pipeline" / project_id / "timeline.otio"
    otio = out_dir / "timeline.otio"
    if otio_src.exists():
        _run_copy(otio_src, otio)
        files["timeline.otio"] = otio

    # thumbnails: salient insight moments + even coverage, scored + ranked
    cand_times: list[float] = []
    for i in insights or []:
        if (
            i.get("kind") in {"event", "risk"}
            and float(i.get("confidence", 0)) >= 0.5
            and i.get("timestamp_s") is not None
        ):
            cand_times.append(float(i["timestamp_s"]))
    total = duration_s or 0.0
    if total <= 0:
        try:
            from director_os.core.qa_probe import probe

            total = float(probe(final).duration_s or 0.0)
        except Exception:  # noqa: BLE001
            total = 0.0
    if total > 0:
        cand_times += [total * f for f in (0.08, 0.25, 0.45, 0.65, 0.85)]
    seen: list[float] = []
    for t in cand_times:
        if all(abs(t - o) > max(2.0, total * 0.02) for o in seen):
            seen.append(t)
    thumbs_dir = out_dir / "thumbnails"
    scored: list[dict] = []
    for k, t in enumerate(seen[:10]):
        p = thumbs_dir / f"thumb_{k:02d}_{int(t):05d}.jpg"
        if not _extract_frame(final, t, p):
            continue
        luma, sharp = _frame_score(p)
        scored.append(
            {
                "path": str(p),
                "t_s": round(t, 2),
                "luma": round(luma, 1),
                "sharpness": round(sharp, 1),
            }
        )
    scored.sort(key=lambda s: (-s["sharpness"], -min(s["luma"], 170)))
    for rank, s in enumerate(scored):
        s["rank"] = rank
    if scored:
        files["thumbnails/"] = thumbs_dir

    # verification: the clean-ends law
    first_luma, last_luma = first_last_frame_luma(final)
    verification = {
        "first_frame_luma": round(first_luma, 1),
        "last_frame_luma": round(last_luma, 1),
        "clean_first_last_frames": first_luma >= 6.0 and last_luma >= 6.0,
        "duration_s": round(total, 2),
    }

    # content-id risk (music guardrail receipt)
    music_spans = [
        (float(a), float(b)) for a, b in (enhance_info.music_flagged_spans if enhance_info else [])
    ]
    content_id_risk = {
        "music_flagged": bool(music_spans),
        "music_spans_s": [[round(a, 2), round(b, 2)] for a, b in music_spans],
        "demucs_used": bool(enhance_info and enhance_info.enhancer_input == "demucs_vocals"),
        "upload_safe_note": (
            "music spans routed through Demucs vocals stem before the mix"
            if (music_spans and enhance_info and enhance_info.enhancer_input == "demucs_vocals")
            else (
                "MUSIC FLAGGED but no Demucs stem was used — review before upload"
                if music_spans
                else "no copyrighted-music spans detected by YAMNet"
            )
        ),
    }

    # seo
    seo = build_seo(
        project_title=project_title or f"Case File {project_id[:8]}",
        transcript=transcript or {},
        insights=insights or [],
        card_subtitle=card_subtitle,
        chapters=chapters,
        duration_s=total,
    )
    seo_lines = [
        f"# {seo['title']}",
        "",
        seo["description"],
        "",
        "## Outcome summary",
        seo["outcome_summary"],
        "",
        "## Keywords",
        ", ".join(seo["keywords"]),
    ]
    (out_dir / "seo.md").write_text("\n".join(seo_lines) + "\n", encoding="utf-8")
    files["seo.md"] = out_dir / "seo.md"

    manifest = {
        "stage": "ship_pack",
        "project_id": project_id,
        "files": {k: str(v) for k, v in files.items()},
        "final_sha256": _sha256(final),
        "chapters": [[t, label] for t, label in chapters],
        "seo": seo,
        "verification": verification,
        "content_id_risk": content_id_risk,
        "thumbnails": scored,
        "youtube_upload": "disabled — upload manually from this folder",
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    files["metadata.json"] = out_dir / "metadata.json"

    return ShipPackResult(
        pack_dir=out_dir,
        files=files,
        chapters=chapters,
        seo=seo,
        verification=verification,
        content_id_risk=content_id_risk,
    )


def _run_copy(src: Path, dst: Path) -> None:
    import shutil

    shutil.copy2(src, dst)
