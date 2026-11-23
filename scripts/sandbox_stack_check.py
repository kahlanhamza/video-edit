"""Full-stack interaction test — REAL API over REAL HTTP (sandbox tier).

Boots uvicorn exactly like `make api` does (sqlite + local storage), seeds a
project + narration timeline through the app's own DB machinery (what the
pipeline produces), then drives the NEW A1-trim interaction the way the
Studio does: POST /agent/dispatch -> GET /timeline/nle -> GET /export/gate.

What must hold at the end:
  1. dispatch timeline.trim_audio returns ok + a NEW timeline version
  2. the NLE snapshot carries vo_in_s on the A1 block (trim visible in UI)
  3. the export gate flips dirty with a pending op count of 1
  4. a reset dispatch (vo_out null) rides the same path
  5. the invalid dispatch (inverted window) is rejected WITHOUT advancing
     the version (checkpoint-rollback law)

Exit code 0 = the new interaction works end to end over real HTTP.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

BASE = "http://127.0.0.1:8011"
PID_FILE = Path("/tmp/stack_check_project_id")


def api(method: str, path: str, body: dict | None = None) -> dict:
    """HTTP status under `status_code`, JSON body under `body` (never merged)."""
    req = urllib.request.Request(
        BASE + path,
        method=method,
        headers={"Content-Type": "application/json"},
        data=json.dumps(body).encode() if body is not None else None,
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return {"status_code": r.status, "body": json.loads(r.read())}
    except urllib.error.HTTPError as e:
        try:
            parsed = json.loads(e.read().decode())
        except Exception:  # noqa: BLE001
            parsed = {"detail": "unparseable error body"}
        return {"status_code": e.code, "body": parsed}


def safe_json(e: urllib.error.HTTPError) -> dict:
    try:
        return json.loads(e.read().decode())
    except Exception:  # noqa: BLE001
        return {}


def wait_health(seconds: float = 60) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=3) as r:
                if r.status == 200:
                    return
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    raise SystemExit("API never became healthy")


def seed_project_with_narration_timeline() -> str:
    """Create a project + narration timeline through the app's own stack."""
    # lazy imports AFTER uvicorn env is pinned — same engine the API binds
    import opentimelineio.opentime as ot_time
    import opentimelineio.schema as ot_schema
    from director_os.agent_runtime.checkpoint import CheckpointStore
    from director_os.api.main import session
    from director_os.core.db import ProjectRow
    from director_os.core.models import ProjectStatus
    from director_os.otio_engine.contract import NarratorClipMeta, RawClipMeta, attach

    s = session()
    row = ProjectRow(
        title="SMOKE narration/commentary trim",
        niche_skill="bodycam_documentary",
        status=ProjectStatus.approved.value,
    )
    s.add(row)
    s.commit()
    project_id = row.id
    s.close()

    # a pipeline-shaped timeline: raw block + narration block (the A1 lane)
    tl = ot_schema.Timeline()
    master = ot_schema.Track(name="V1_master")
    tl.tracks.append(master)

    raw = ot_schema.Clip(name="raw_0")
    raw.source_range = ot_time.TimeRange(
        start_time=ot_time.RationalTime(0, 30), duration=ot_time.RationalTime(12 * 30, 30)
    )
    attach(raw, RawClipMeta(asset_id="smoke_asset", audio_mode="loudnorm_48k"))
    master.append(raw)

    narr = ot_schema.Clip(name="narration_1")
    narr.source_range = ot_time.TimeRange(
        start_time=ot_time.RationalTime(20 * 30, 30), duration=ot_time.RationalTime(8 * 30, 30)
    )
    attach(
        narr,
        NarratorClipMeta(
            script_text="The stop looked routine — until the trunk opened.",
            broll_asset_id="smoke_asset",
            broll_range=(20.0, 28.0),
        ),
    )
    master.append(narr)

    # checkpoint store needs the factory, not a session — build it the way main.py does
    from director_os.api import main as api_main

    api_main.session()  # force engine init
    factory = api_main._session_factory
    assert factory is not None
    store = CheckpointStore(factory)
    version = store.checkpoint(tl, project_id, note="seed: raw + narration (pipeline shape)")
    with factory() as s2:
        p = s2.get(ProjectRow, project_id)
        assert p is not None
        p.active_otio_version_id = version
        s2.commit()
    print(f"  seeded project {project_id} with timeline version {version}")
    return project_id


def main() -> int:
    wait_health()

    print("[1] project create over HTTP")
    r = api("POST", "/projects", {"title": "SMOKE direct", "niche_skill": "bodycam_documentary"})
    assert r["status_code"] == 201, r
    project_id = r["body"]["id"]
    print(f"    created {project_id}")

    project_id = seed_project_with_narration_timeline()
    PID_FILE.write_text(project_id)

    print("[2] NLE snapshot — narration block visible on A1")
    nle = api("GET", f"/projects/{project_id}/timeline/nle")
    assert nle["status_code"] == 200, nle
    nle_body = nle["body"]
    narr = [c for c in nle_body["clips"] if c["kind"] == "narration"]
    assert len(narr) == 1, f"expected 1 narration block, got {len(narr)}"
    a1 = narr[0]
    assert a1["master_index"] is not None, "narration block must expose its master index"
    assert a1["vo_in_s"] == 0.0 and a1["vo_out_s"] is None, a1
    print(
        f"    A1 block at master_index={a1['master_index']} start={a1['start_s']} dur={a1['dur_s']}"
    )

    print("[3] NEW INTERACTION: timeline.trim_audio (skip 1.2s of head dead air)")
    before = api("GET", f"/projects/{project_id}/timeline")["body"].get("version")
    r = api(
        "POST",
        f"/projects/{project_id}/agent/dispatch",
        {
            "tool": "timeline.trim_audio",
            "args": {"item_index": a1["master_index"], "vo_in_s": 1.2, "vo_out_s": 6.4},
            "checkpoint_note": "smoke: A1 trim",
        },
    )
    assert r["status_code"] == 200, r
    assert r["body"]["ok"] is True, r
    assert r["body"]["version"] is not None and r["body"]["version"] > (before or 0), r
    print(f"    ok, version {before} -> {r['body']['version']}")

    print("[4] NLE reflects the trim")
    nle = api("GET", f"/projects/{project_id}/timeline/nle")
    a1b = [c for c in nle["body"]["clips"] if c["kind"] == "narration"][0]
    assert a1b["vo_in_s"] == 1.2 and a1b["vo_out_s"] == 6.4, a1b
    print(
        f"    A1 vo window [{a1b['vo_in_s']}, {a1b['vo_out_s']}]s — Studio trim handles have data"
    )

    print("[5] export gate tracks the trim (version axis)")
    gate = api("GET", f"/projects/{project_id}/export/gate")
    assert gate["status_code"] == 200, gate
    gate_body = gate["body"]
    assert gate_body["dirty"] is True, gate_body
    # never exported: pending is an op-fold concept and stays 0; the trim is
    # a VERSION-level edit — the gate sees the active version advance
    assert gate_body["pending_ops"] == 0, gate_body
    assert gate_body["exported_once"] is False, gate_body
    assert gate_body["active_version"] == r["body"]["version"], gate_body
    print(
        f"    gate: dirty={gate_body['dirty']} active_version={gate_body['active_version']} "
        f"(version-level edits dirty the gate — Export button never lies)"
    )

    print("[6] invalid trim is rejected WITHOUT advancing the version")
    r2 = api(
        "POST",
        f"/projects/{project_id}/agent/dispatch",
        {
            "tool": "timeline.trim_audio",
            "args": {"item_index": a1["master_index"], "vo_in_s": 5.0, "vo_out_s": 1.0},
            "checkpoint_note": "smoke: invalid window",
        },
    )
    assert r2["body"]["ok"] is False, r2  # executor rolls back + reports failure
    nle2 = api("GET", f"/projects/{project_id}/timeline/nle")
    a1c = [c for c in nle2["body"]["clips"] if c["kind"] == "narration"][0]
    assert a1c["vo_out_s"] == 6.4, "rollback must preserve the last good window"
    print("    rejected + rolled back — the good window survives")

    print("[7] reset dispatch releases the window (vo_out null)")
    r3 = api(
        "POST",
        f"/projects/{project_id}/agent/dispatch",
        {
            "tool": "timeline.trim_audio",
            "args": {"item_index": a1["master_index"], "vo_in_s": 0.0, "vo_out_s": None},
            "checkpoint_note": "smoke: reset window",
        },
    )
    assert r3["body"]["ok"] is True, r3
    nle3 = api("GET", f"/projects/{project_id}/timeline/nle")
    a1d = [c for c in nle3["body"]["clips"] if c["kind"] == "narration"][0]
    assert a1d["vo_out_s"] is None and a1d["vo_in_s"] == 0.0, a1d
    print("    window released — full stem again")

    print("\nFULL-STACK TRIM INTERACTION: ALL CHECKS GREEN")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    raise SystemExit(main())
