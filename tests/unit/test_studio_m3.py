"""M3 close-out — proxy builder, subtitle grid store, studio API surface.

Covers: ingest.build_proxies artifacts, cue store save/load/edit (difflib
re-alignment), assets/proxy/scrubsheet streaming, subtitle GET/PATCH and
redaction CRUD with clamping.
"""

from __future__ import annotations

import pytest
from director_os.core.proxy import build_proxies
from director_os.core.qa_probe import probe
from director_os.subtitles.ass_builder import CaptionBeat, WordTiming
from director_os.subtitles.store import SubtitleStoreError, edit_cue, load_cues, save_cues


# ------------------------------------------------------------------ proxies
def test_build_proxies_artifacts(fixture_video, tmp_path):
    out = tmp_path / "assets"
    bundle = build_proxies(fixture_video, out)
    assert bundle.proxy_path.exists() and bundle.proxy_path.stat().st_size > 0
    assert bundle.scrub_path.exists() and bundle.scrub_path.stat().st_size > 0
    assert bundle.audio_path.exists() and bundle.audio_path.stat().st_size > 0
    info = probe(bundle.proxy_path)
    assert info.has_video and info.height == 480
    assert (info.sample_rate or 0) == 48000  # proxy obeys the 48k law


def test_build_proxies_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_proxies(tmp_path / "nope.mp4", tmp_path / "out")


# ------------------------------------------------------------------ cue store
def _seed(base):
    save_cues(
        base,
        [
            CaptionBeat(
                text="unit one contact",
                start_ms=0,
                end_ms=1800,
                speaker="OFFICER_1",
                words=[
                    WordTiming(word="unit", start_ms=0, end_ms=400),
                    WordTiming(word="one", start_ms=400, end_ms=800),
                    WordTiming(word="contact", start_ms=800, end_ms=1800),
                ],
            ),
            CaptionBeat(text="roger that", start_ms=2000, end_ms=3000, speaker="OFFICER_2"),
        ],
    )


def test_cue_store_roundtrip_and_srt(tmp_path):
    base = tmp_path / "sub"
    _seed(base)
    cues = load_cues(base)
    assert len(cues) == 2
    srt = (base / "subtitles.srt").read_text()
    assert "unit one contact" in srt and "1" in srt
    ass = (base / "subtitles.ass").read_text()
    assert "unit" in ass and "Lane1" in ass


def test_edit_cue_realigned_via_difflib(tmp_path):
    """Guardrail 5: proofread 'unit one contact' -> 'Signal 10-8, en route' keeps ms."""
    base = tmp_path / "sub"
    _seed(base)
    beat = edit_cue(base, 0, "Signal 10-8 en route", speaker="DISPATCH")
    assert beat.speaker == "DISPATCH"
    assert len(beat.words) == 4
    # monotonic + inside the original slot
    assert beat.words[0].start_ms >= 0
    for a, b in zip(beat.words, beat.words[1:], strict=False):
        assert b.start_ms >= a.start_ms
    assert beat.end_ms <= 1800
    # artifacts rebuilt on disk
    srt = (base / "subtitles.srt").read_text()
    assert "Signal 10-8 en route" in srt


def test_edit_cue_validates(tmp_path):
    base = tmp_path / "sub"
    _seed(base)
    with pytest.raises(SubtitleStoreError, match="range"):
        edit_cue(base, 9, "nope")
    with pytest.raises(SubtitleStoreError, match="empty"):
        edit_cue(base, 0, "   ")


# ------------------------------------------------------------------ studio API
@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    import director_os.api.main as api_main
    from director_os.core.config import get_settings
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/studio.db")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    get_settings.cache_clear()
    api_main._engine = None
    with TestClient(api_main.app) as client:
        yield client, tmp_path / "output"
    get_settings.cache_clear()


@pytest.fixture()
def project_with_asset(api_client, fixture_video):
    """Project + asset with a real proxy bundle on disk."""
    from director_os.core.db import MediaAssetRow
    from director_os.core.proxy import build_proxies

    client, out_dir = api_client
    pid = client.post("/projects", json={"title": "Studio case"}).json()["id"]
    bundle = build_proxies(fixture_video, out_dir / "pipeline" / pid / "assets" / "a_studio")
    with api_main_session() as s:
        asset = MediaAssetRow(
            id="a_studio",
            project_id=pid,
            source_url=str(fixture_video),
            role="raw_bwc",
            r2_proxy_key=str(bundle.proxy_path),
            duration_s=12.0,
            fps=30.0,
            width=320,
            height=240,
        )
        s.add(asset)
        s.commit()
    return client, pid


def api_main_session():
    import director_os.api.main as api_main

    factory = api_main._session_factory
    assert factory is not None
    return factory()


def test_assets_list_and_proxy_stream(project_with_asset):
    client, pid = project_with_asset
    assets = client.get(f"/projects/{pid}/assets").json()
    assert len(assets) == 1 and assets[0]["has_proxy"] is True
    r = client.get("/assets/a_studio/proxy")
    assert r.status_code == 200 and r.headers["content-type"].startswith("video/mp4")
    assert len(r.content) > 0
    sheet = client.get("/assets/a_studio/scrubsheet")
    assert sheet.status_code == 200 and sheet.headers["content-type"] == "image/jpeg"


def test_proxy_404_without_pipeline(project_with_asset):
    client, pid = project_with_asset
    r = client.get("/assets/does_not_exist/proxy")
    assert r.status_code == 404


def test_subtitle_grid_endpoints(project_with_asset):
    from director_os.subtitles.store import save_cues

    client, pid = project_with_asset
    save_cues(
        client.get(f"/projects/{pid}/assets").json() and _sub_dir(pid),
        [CaptionBeat(text="unit one contact", start_ms=0, end_ms=1800, speaker="OFFICER_1")],
    )
    cues = client.get(f"/projects/{pid}/subtitles").json()
    assert cues and cues[0]["text"] == "unit one contact"

    edited = client.patch(
        f"/projects/{pid}/subtitles/0",
        json={"text": "Signal 10-8 en route", "speaker": "DISPATCH"},
    )
    assert edited.status_code == 200
    assert edited.json()["speaker"] == "DISPATCH"

    assert client.patch(f"/projects/{pid}/subtitles/42", json={"text": "x"}).status_code == 404
    assert client.patch(f"/projects/{pid}/subtitles/0", json={"text": "  "}).status_code == 422

    srt = client.get(f"/projects/{pid}/subtitles/srt")
    assert srt.status_code == 200 and b"Signal 10-8" in srt.content


def _sub_dir(pid: str):

    from director_os.core.config import get_settings

    return get_settings().output_dir / "pipeline" / pid


def test_redaction_crud_and_clamping(project_with_asset):
    client, pid = project_with_asset
    # valid box inside 320x240
    r = client.post(
        f"/projects/{pid}/redactions", json={"x": 200, "y": 10, "w": 100, "h": 40, "label": "axon"}
    )
    assert r.status_code == 201
    region = r.json()
    assert region["created_by"] == "human_edit"
    # fully outside -> 422
    r2 = client.post(f"/projects/{pid}/redactions", json={"x": 5000, "y": 5000, "w": 10, "h": 10})
    assert r2.status_code == 422
    # list + delete
    listed = client.get(f"/projects/{pid}/redactions").json()
    assert [x["id"] for x in listed] == [region["id"]]
    assert client.delete(f"/projects/{pid}/redactions/{region['id']}").status_code == 204
    assert client.get(f"/projects/{pid}/redactions").json() == []
    assert client.delete(f"/projects/{pid}/redactions/{region['id']}").status_code == 404


def test_drawn_redaction_reaches_render_pipeline(project_with_asset):
    """The studio stores the FLAT shape ({x, y, w, h}); the render pipeline must
    read it (plus the canonical nested {bbox: {...}} shape) into BlurRegions."""
    import director_os.api.main as api_main
    from director_os.core.db import PerceptionIndexRow
    from director_os.worker.pipeline import _project_redaction_regions

    client, pid = project_with_asset
    r = client.post(
        f"/projects/{pid}/redactions", json={"x": 10, "y": 20, "w": 50, "h": 30, "sigma": 12}
    )
    assert r.status_code == 201

    factory = api_main._session_factory
    assert factory is not None
    regions = _project_redaction_regions(factory, "a_studio")
    assert len(regions) == 1
    assert (regions[0].bbox.x, regions[0].bbox.y, regions[0].bbox.w, regions[0].bbox.h) == (
        10,
        20,
        50,
        30,
    )
    assert regions[0].sigma == 12 and regions[0].scope == "global"

    # nested canonical shape accepted; corrupt entries skipped silently
    with api_main_session() as s:
        row = s.get(PerceptionIndexRow, "a_studio")
        row.redaction_tracks_json = {
            "regions": [
                {"id": "n1", "bbox": {"x": 1, "y": 2, "w": 3, "h": 4}},
                {"id": "bad", "x": 0, "y": 0, "w": 0, "h": 0},  # zero-size -> invalid
                {"id": "flat2", "x": 5, "y": 5, "w": 10, "h": 10},
            ]
        }
        s.commit()
    regions = _project_redaction_regions(factory, "a_studio")
    assert [r.track_id for r in regions] == ["n1", "flat2"]
