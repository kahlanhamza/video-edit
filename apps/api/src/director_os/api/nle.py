"""NLE timeline snapshot — the Studio live editing timeline feed.

Assembles EVERYTHING the multi-track timeline renders from durable rows: the
active OTIO version's clips (video + card overlays + narration audio), the
caption cue store, the redaction regions (static boxes + tracked windows),
and the source asset facts. The Studio refetches this whenever a `timeline`
SSE event fires, so agent-driven cuts appear live.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from director_os.core.db import MediaAssetRow, PerceptionIndexRow, ProjectRow, TimelineVersionRow
from director_os.otio_engine.contract import meta_kind
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker


class ProjectNotFound(LookupError):
    pass


def _seconds(rt: Any) -> float:
    rate = rt.rate or 30.0
    return float(rt.value) / float(rate)


def build_nle_snapshot(sessions: sessionmaker, project_id: str, output_dir: Path) -> dict[str, Any]:
    """Everything the multi-track NLE renders, from durable state only."""
    with sessions() as s:
        project = s.get(ProjectRow, project_id)
        if project is None:
            raise ProjectNotFound(project_id)
        version_row = None
        if project.active_otio_version_id is not None:
            version_row = s.execute(
                select(TimelineVersionRow).where(
                    TimelineVersionRow.project_id == project_id,
                    TimelineVersionRow.version_num == project.active_otio_version_id,
                )
            ).scalar_one_or_none()
        asset = s.execute(
            select(MediaAssetRow)
            .where(MediaAssetRow.project_id == project_id)
            .order_by(MediaAssetRow.id)
            .limit(1)
        ).scalar_one_or_none()
        regions: list[dict[str, Any]] = []
        if asset is not None:
            prow = s.get(PerceptionIndexRow, asset.id)
            regions = list((prow.redaction_tracks_json or {}).get("regions", []) if prow else [])
        otio_json = dict(version_row.otio_json) if version_row else None
        version = version_row.version_num if version_row else None
        asset_facts = (
            {
                "id": asset.id,
                "duration_s": float(asset.duration_s) if asset else 0.0,
                "fps": float(asset.fps) if asset else 30.0,
                "width": int(asset.width) if asset else 0,
                "height": int(asset.height) if asset else 0,
                "has_proxy": bool(
                    asset and asset.r2_proxy_key and Path(asset.r2_proxy_key).exists()
                ),
            }
            if asset
            else None
        )

    clips = _clips_from_otio(otio_json)
    captions = _captions(output_dir / "pipeline" / project_id)
    duration = asset_facts["duration_s"] if asset_facts else 0.0
    for c in clips:
        duration = max(duration, c["end_s"])
    for cue in captions:
        duration = max(duration, cue["end_ms"] / 1000.0)
    return {
        "project_id": project_id,
        "version": version,
        "duration_s": round(duration, 3),
        "asset": asset_facts,
        "clips": clips,
        "captions": captions,
        "effects": [
            {
                "id": str(r.get("id", "")),
                "x": int(r.get("x", 0)),
                "y": int(r.get("y", 0)),
                "w": int(r.get("w", 0)),
                "h": int(r.get("h", 0)),
                "sigma": int(r.get("sigma", 24)),
                "label": str(r.get("label", "")),
                "mode": str(r.get("mode", "static")),
                "t_start_s": r.get("t_start_s"),
                "t_end_s": r.get("t_end_s"),
                "created_by": str(r.get("created_by", "")),
            }
            for r in regions
        ],
    }


def _clips_from_otio(otio_json: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Walk EVERY track of the active version — the NLE shows all of them.

    `master_index` mirrors the DirectorToolbox item indexing (position on the
    master track INCLUDING gaps) so the Studio's manual trim/split dispatch
    lands on the exact clip the operator clicked."""
    if not otio_json:
        return []
    try:
        import opentimelineio.adapters as ot_adapters

        timeline = ot_adapters.read_from_string(json.dumps(otio_json), "otio_json")
    except Exception:  # noqa: BLE001 — a corrupt row must not 500 the studio
        return []
    clips: list[dict[str, Any]] = []
    for track in timeline.tracks:
        counter = 0
        for item in track:
            counter += 1
            if type(item).__name__ == "Gap":
                continue
            kind = meta_kind(item)
            try:
                rng = item.trimmed_range_in_parent()
                start = _seconds(rng.start_time)
                dur = _seconds(rng.duration)
            except Exception:  # noqa: BLE001 — unparented item: best effort
                sr = getattr(item, "source_range", None)
                if sr is None:
                    continue
                start, dur = _seconds(sr.start_time), _seconds(sr.duration)
            source_start: float | None = None
            try:
                sr = item.source_range
                if sr is not None:
                    source_start = _seconds(sr.start_time)
            except Exception:  # noqa: BLE001
                source_start = None
            meta: dict[str, Any] = {}
            try:
                # otio AnyDictionary proxies are dict-like but not dict instances
                meta = dict(getattr(item, "metadata", {}).get("director_os", {}) or {})
            except Exception:  # noqa: BLE001 — metadata-less item: empty meta
                meta = {}
            clips.append(
                {
                    "track": track.name,
                    "kind": kind,
                    "name": str(getattr(item, "name", "") or kind),
                    "start_s": round(start, 3),
                    "dur_s": round(dur, 3),
                    "end_s": round(start + dur, 3),
                    "master_index": counter - 1 if track.name == "V1_master" else None,
                    "act": meta.get("act_name"),
                    "script": meta.get("script_text"),
                    "template": meta.get("template"),
                    "audio_mode": meta.get("audio_mode"),
                    "source_start_s": round(source_start, 3) if source_start is not None else None,
                }
            )
    return clips


def _captions(sub_dir: Path) -> list[dict[str, Any]]:
    """Cue store (subtitles.json) — 404/empty just means 'not built yet'."""
    path = sub_dir / "subtitles.json"
    if not path.exists():
        return []
    try:
        entries = json.loads(path.read_text(encoding="utf-8") or "[]")
    except Exception:  # noqa: BLE001 — corrupt store must not 500 the studio
        return []
    out: list[dict[str, Any]] = []
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            continue
        out.append(
            {
                "index": i,
                "start_ms": int(e.get("start_ms", 0)),
                "end_ms": int(e.get("end_ms", 0)),
                "text": str(e.get("text", "")),
                "speaker": str(e.get("speaker", "")),
            }
        )
    return out
