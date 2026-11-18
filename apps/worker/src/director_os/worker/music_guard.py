"""Content-ID music guardrail — YAMNet Music spans -> Demucs routing (Stage 4->5).

Stage 4 tags `music` event spans (car radios, store PA, background songs) via
YAMNet (`perception.audio_events`, AudioSet classes 24/132/133/519). Stage 5
MUST NOT feed those ranges straight into MossFormer2: the neural enhancer
keeps whatever correlates with speech, and a copyrighted background song that
survives the mix is a YouTube Content ID claim waiting to happen.

The law: when music is flagged, the enhancer's INPUT becomes the Demucs
`vocals` stem (two-stems separation — `htdemucs`, Kaggle GPU fleet) for the
whole asset; speech rides the stem, the accompaniment never enters Bus-1.
The stem is produced OFF-PIPELINE (free GPU): `scripts/demucs_stems_run.py`
or `COMPUTE_BACKEND=kaggle` — the resolver looks for it at

    output/audio2bus/demucs/<asset_id>/vocals.wav

A missing stem never blocks the pipeline (soft degradation, the enhance
receipt + Ship Pack manifest carry the risk flag), but it is logged LOUD.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

DEMUCS_STEM_DIRNAME = "demucs"


def music_spans_from_shots(shots: dict) -> list[tuple[float, float]]:
    """`shots_json["audio_events"]` -> (start_s, end_s) spans tagged `music`."""
    out: list[tuple[float, float]] = []
    for e in (shots or {}).get("events", []):
        if e.get("kind") == "music":
            out.append((float(e["start_s"]), float(e["end_s"])))
    return sorted(out)


def demucs_stem_path(asset_id: str, settings) -> Path:
    """Canonical stem location for one asset (`vocals.wav`)."""
    return Path(settings.output_dir) / "audio2bus" / DEMUCS_STEM_DIRNAME / asset_id / "vocals.wav"


def make_demucs_resolver(
    asset_id: str, settings, log=lambda _m: None
) -> Callable[[str], Path | None]:
    """Deps.vocals_stem_resolver factory: canonical stem path or None (+ loud hint)."""

    def resolver(_asset_id: str) -> Path | None:
        stem = demucs_stem_path(asset_id, settings)
        if stem.exists() and stem.stat().st_size > 0:
            log(f"  [audio] Demucs vocals stem found: {stem}")
            return stem
        log(
            "  [audio] music flagged but no Demucs stem at "
            f"{stem} — produce it on the free GPU fleet, then re-run: "
            "uv run scripts/demucs_stems_run.py push|poll|harvest "
            "(or COMPUTE_BACKEND=kaggle); continuing without it (risk flag rides "
            "the receipt + Ship Pack manifest)"
        )
        return None

    return resolver


def resolve_music_routing(
    asset_id: str,
    shots: dict,
    deps,
) -> tuple[list[tuple[float, float]], bool]:
    """Read the vision-stage YAMNet index and wire the Demucs resolver if flagged.

    Returns (music_spans, resolver_active). Called by run_pipeline between
    vision enrichment and audio prep — the exact Stage 4 -> Stage 5 seam.
    """
    spans = music_spans_from_shots(shots)
    if not spans:
        return [], False
    deps.vocals_stem_resolver = make_demucs_resolver(asset_id, deps.settings, deps.log)
    return spans, True
