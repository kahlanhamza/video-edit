#!/usr/bin/env python3
"""Rebuild the ACCEPTED Eustis timeline from the shipped master artifacts.

The sandbox recycle lost `download/eustis_acceptance_timeline.otio`, and the
AI editor session needs the ACCEPTED cut as its working-copy base (a rollback
boundary for the agent's proposals). Faithful reconstruction, no guesses:

- master: pipeline_run/v3/master_postfx_v4.mp4 (QA 11/11 green, sha256 in
  provenance) -> ONE raw clip 0..duration on V1_master
- cards: cards_meta_v3.json times + the measured burn window 6.5 s
  (postfx_v3.phase_burn uses t0..t0+6.5) -> V2_overlay_cards gaps + CardMeta
  clips; template/fields recovered from the burned card PNGs
- provenance: sha256, QA verdict, caption v4 stats ride timeline metadata

Usage: python3 scripts/build_accept_timeline.py
Writes: download/eustis_acceptance_timeline.otio
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "otio_engine" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "core" / "src"))

from director_os.core.models import AudioContract
from director_os.otio_engine.contract import CardMeta, RawClipMeta
from director_os.otio_engine.timeline import (
    MASTER_TRACK,
    OVERLAY_TRACK,
    ClipSpec,
    GapSpec,
    TimelineSpec,
    TrackSpec,
    build_timeline,
    save_otio,
)

V3 = Path("/home/z/my-project/pipeline_run/v3")
OUT = Path("/home/z/my-project/download/eustis_acceptance_timeline.otio")

# Card content recovered from the burned PNGs (kicker/title/subtitle).
CARD_FIELDS = [
    {
        "template": "location_intro",
        "kicker": "LOCATION INTRO",
        "title": "Eustis, Florida",
        "subtitle": "July 4, 2022",
    },
    {
        "template": "evidence_fact",
        "kicker": "EVIDENCE FACT",
        "title": "Open Container",
        "subtitle": "An open beer can found in the patrol car",
    },
    {
        "template": "evidence_fact",
        "kicker": "EVIDENCE FACT",
        "title": "Admission of Guilt",
        "subtitle": "Three tall boys of Modelo beer consumed",
    },
    {
        "template": "case_outcome",
        "kicker": "CASE OUTCOME",
        "title": "Career in Ruins",
        "subtitle": "Suspended without pay; BAC 0.391g/210L (nearly 5x legal limit)",
    },
]

BURN_WINDOW_S = 6.5


def main() -> None:
    master = V3 / "master_postfx_v4.mp4"
    dur = float(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(master),
            ]
        )
        .decode()
        .strip()
    )
    sha = hashlib.sha256(master.read_bytes()).hexdigest()
    cards_meta = json.loads((V3 / "cards_meta_v3.json").read_text())
    qa = json.loads((V3 / "qa_v4.json").read_text())
    qa_green = all(c["ok"] for c in qa)
    stats = json.loads((V3 / "caption_v4_stats.json").read_text())

    # ---- V1_master: the shipped master, verbatim (post-FX burned in)
    raw = RawClipMeta(asset_id="master_postfx_v4", audio_mode="loudnorm_48k", trim_silence=False)
    master_clip = ClipSpec(
        name="eustis_acceptance_master", source_start_s=0.0, dur_s=round(dur, 3), meta=raw
    )
    master_track = TrackSpec(name=MASTER_TRACK, items=[master_clip])

    # ---- V2_overlay_cards: gaps + card clips at the measured burn times
    overlay_items: list[ClipSpec | GapSpec] = []
    t_prev = 0.0
    for (png, t0), fields in zip(cards_meta["card_times"], CARD_FIELDS, strict=True):
        gap_s = round(t0 - t_prev, 3)
        if gap_s > 0.01:
            overlay_items.append(GapSpec(dur_s=gap_s))
        meta = CardMeta(
            template=fields["template"],
            fields={
                "kicker": fields["kicker"],
                "title": fields["title"],
                "subtitle": fields["subtitle"],
            },
        )
        overlay_items.append(
            ClipSpec(
                name=f"card_{png.replace('.png', '')}",
                source_start_s=0.0,
                dur_s=BURN_WINDOW_S,
                meta=meta,
            )
        )
        t_prev = t0 + BURN_WINDOW_S

    tl = build_timeline(
        TimelineSpec(
            name="eustis_acceptance_720p",
            fps=30,
            width=1280,
            height=720,
            audio=AudioContract(sample_rate=48000, channels=2),
            tracks=[master_track, TrackSpec(name=OVERLAY_TRACK, items=overlay_items)],
        )
    )
    tl.metadata["director_os"]["provenance"] = {
        "master": "pipeline_run/v3/master_postfx_v4.mp4",
        "master_sha256": sha,
        "duration_s": round(dur, 3),
        "qa_v4_green": qa_green,
        "caption_v4_stats": stats,
        "rebuilt_from": "shipped artifacts (original acceptance timeline lost in sandbox recycle)",
    }
    OUT.write_text(save_otio(tl, str(OUT)).read_text())
    print(f"wrote {OUT} master={dur:.2f}s cards={len(CARD_FIELDS)} qa_green={qa_green}")


if __name__ == "__main__":
    main()
