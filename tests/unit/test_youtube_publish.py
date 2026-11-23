"""YouTube Private-Draft publisher — metadata rules + publisher contract + API gate.

- chapter validation (0:00 first, >=3, ascending, >=10s apart — real YouTube rules)
- description builder formats the chapter index
- DraftMetadata limits (title 100, description 5000, tags 500 total)
- InMemory publisher records uploads; missing video/srt -> FileNotFoundError
- GoogleYouTubePublisher without googleapiclient -> actionable error
- POST /projects/{id}/publish-draft: happy path, 404, chapter misuse, missing file
"""

from __future__ import annotations

from pathlib import Path

import pytest
from director_os.api.main import app, set_publisher
from director_os.api.youtube import (
    Chapter,
    ChapterError,
    DraftMetadata,
    InMemoryYouTubePublisher,
    build_description,
    validate_chapters,
)


# ------------------------------------------------------------------ chapters
def chapters_ok() -> list[Chapter]:
    return [
        Chapter(title="Cold open", start_s=0.0),
        Chapter(title="Traffic stop", start_s=30.0),
        Chapter(title="Arrest", start_s=95.0),
    ]


def test_chapter_stamp_formatting():
    assert Chapter(title="a", start_s=0.0).stamp == "0:00"
    assert Chapter(title="a", start_s=65.0).stamp == "1:05"
    assert Chapter(title="a", start_s=95.0).stamp == "1:35"
    assert Chapter(title="a", start_s=3675.0).stamp == "1:01:15"


def test_chapter_rules_enforced():
    validate_chapters(chapters_ok())  # legal
    validate_chapters([])  # no chapters = plain description, legal

    with pytest.raises(ChapterError, match="0:00"):
        validate_chapters([Chapter(title="late", start_s=5.0), Chapter(title="b", start_s=60.0)])
    with pytest.raises(ChapterError, match="at least 3"):
        validate_chapters([Chapter(title="a", start_s=0.0), Chapter(title="b", start_s=60.0)])
    with pytest.raises(ChapterError, match="ascending"):
        validate_chapters(
            [
                Chapter(title="a", start_s=0.0),
                Chapter(title="b", start_s=60.0),
                Chapter(title="c", start_s=30.0),
            ]
        )
    with pytest.raises(ChapterError, match="10s apart"):
        validate_chapters(
            [
                Chapter(title="a", start_s=0.0),
                Chapter(title="b", start_s=5.0),
                Chapter(title="c", start_s=60.0),
            ]
        )


def test_description_builder():
    desc = build_description(
        "A documentary cut of the 5th & Market stop.",
        chapters_ok(),
        footer="Auto-rendered by DirectorOS.",
    )
    assert desc.startswith("A documentary cut of the 5th & Market stop.")
    assert "Chapters:" in desc
    assert "0:00 Cold open" in desc
    assert "1:35 Arrest" in desc
    assert desc.endswith("Auto-rendered by DirectorOS.")
    assert len(desc) <= 5000


def test_description_too_long_rejected():
    with pytest.raises(ChapterError, match="too long"):
        build_description("x" * 5000, chapters_ok())


# ------------------------------------------------------------------ metadata
def test_draft_metadata_limits():
    m = DraftMetadata.build(
        title="Case 26-0417 — full stop documentation",
        base_description="Doc",
        chapters=chapters_ok(),
        tags=["bodycam", "documentary"],
    )
    assert m.privacy_status == "private"
    assert m.category_id == "25"

    with pytest.raises(Exception, match="100"):
        DraftMetadata(title="t" * 101)
    with pytest.raises(Exception, match="5000"):
        DraftMetadata(title="ok", description="d" * 5001)
    with pytest.raises(Exception, match="tags too long"):
        DraftMetadata(title="ok", tags=["x" * 400, "y" * 200])
    with pytest.raises(Exception, match="privacy_status"):
        DraftMetadata(title="ok", privacy_status="secret")


# ------------------------------------------------------------------ publishers
def test_inmemory_publisher_records(tmp_path):
    pub = InMemoryYouTubePublisher()
    video = tmp_path / "final.mp4"
    video.write_bytes(b"fake")
    srt = tmp_path / "captions.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:02,000\nhello\n")

    result = pub.upload_draft(video, DraftMetadata(title="t"), srt_path=srt)
    assert result.video_id == "draft_0001"
    assert result.privacy_status == "private"
    assert result.caption_attached is False  # InMemory records but does not "attach"
    assert len(pub.uploads) == 1
    rec = pub.uploads[0]
    assert rec["metadata"]["title"] == "t"
    assert rec["srt_path"] == str(srt)

    with pytest.raises(FileNotFoundError, match="video missing"):
        pub.upload_draft(tmp_path / "nope.mp4", DraftMetadata(title="t"))
    with pytest.raises(FileNotFoundError, match="captions file missing"):
        pub.upload_draft(video, DraftMetadata(title="t"), srt_path=tmp_path / "nope.srt")


def test_google_publisher_actionable_error_without_sdk(tmp_path):
    from director_os.api.youtube import GoogleYouTubePublisher

    pub = GoogleYouTubePublisher(credentials=object())
    video = tmp_path / "final.mp4"
    video.write_bytes(b"fake")
    with pytest.raises(RuntimeError, match="google-api-python-client"):
        pub.upload_draft(video, DraftMetadata(title="t"))


def test_google_publisher_lazy_import(tmp_path):
    """The class must be importable without googleapiclient installed."""
    from director_os.api.youtube import GoogleYouTubePublisher

    GoogleYouTubePublisher(credentials=None)  # import-time no-op


# ------------------------------------------------------------------ API gate
@pytest.fixture()
def client(tmp_path: Path, monkeypatch):
    # the sandbox exports a global DATABASE_URL; pin the API to a per-test sqlite
    from director_os.api import main as api_main
    from director_os.core.config import get_settings
    from fastapi.testclient import TestClient as _TC

    get_settings.cache_clear()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/api.db")
    api_main._engine = None
    api_main._session_factory = None
    set_publisher(InMemoryYouTubePublisher())
    yield _TC(app)
    get_settings.cache_clear()
    api_main._engine = None
    api_main._session_factory = None


@pytest.fixture()
def project_id(client):
    resp = client.post("/projects", json={"title": "Publish Gate Case"})
    assert resp.status_code == 201
    return resp.json()["id"]


def _seed_baked_gate(project_id: str, version: int = 1) -> None:
    """Simulate a completed bake: primary asset + perception row carrying the
    exact gate cursor `_stamp_export_gate` writes (ops rev + version)."""
    from director_os.api.main import session
    from director_os.core.db import MediaAssetRow, PerceptionIndexRow

    s = session()
    try:
        asset = MediaAssetRow(
            project_id=project_id, duration_s=10.0, fps=30.0, width=640, height=360
        )
        s.add(asset)
        s.flush()
        s.add(
            PerceptionIndexRow(
                asset_id=asset.id,
                redaction_tracks_json={"baked_ops_rev": 0, "baked_timeline_version": version},
            )
        )
        s.commit()
    finally:
        s.close()


def _seed_youtube_vault() -> None:
    from director_os.api.main import session
    from director_os.core.credentials import upsert_credentials

    s = session()
    try:
        upsert_credentials(
            s,
            {
                "YOUTUBE_CLIENT_ID": "cid.apps.googleusercontent.com",
                "YOUTUBE_CLIENT_SECRET": "super-secret",
                "YOUTUBE_REFRESH_TOKEN": "1//refresh-token",
            },
        )
        s.commit()
    finally:
        s.close()


def _install_google_stubs(monkeypatch) -> None:
    """googleapiclient/google-auth are optional deps absent in CI — inject
    import-level stubs so _resolve_publisher can construct OAuth credentials."""
    import sys
    import types

    creds_mod = types.ModuleType("google.oauth2.credentials")

    class Credentials:
        def __init__(self, **kw):
            self.kw = kw

    creds_mod.Credentials = Credentials
    oauth_pkg = types.ModuleType("google.oauth2")
    oauth_pkg.credentials = creds_mod
    google_pkg = types.ModuleType("google")
    google_pkg.oauth2 = oauth_pkg
    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.oauth2", oauth_pkg)
    monkeypatch.setitem(sys.modules, "google.oauth2.credentials", creds_mod)


class _StubRealPublisher:
    """Stands in for GoogleYouTubePublisher: records and returns a real-looking result."""

    last_instance: _StubRealPublisher | None = None

    def __init__(self, credentials):
        self.credentials = credentials
        self.uploads: list[dict] = []
        _StubRealPublisher.last_instance = self

    def upload_draft(self, video_path, metadata, srt_path=None):
        from director_os.api.youtube import PublishResult

        self.uploads.append(
            {
                "video_path": str(video_path),
                "title": metadata.title,
                "srt_path": str(srt_path) if srt_path else None,
                "credentials": self.credentials,
            }
        )
        return PublishResult(
            video_id="yt_real_123",
            privacy_status="private",
            caption_attached=srt_path is not None,
        )


def test_publish_draft_real_upload_when_vault_configured(client, project_id, tmp_path, monkeypatch):
    """Vault carries OAuth creds -> the REAL publisher runs, dry_run=False, and
    the project flips to published_draft (the true last mile)."""
    from director_os.api import main as api_main

    _install_google_stubs(monkeypatch)
    monkeypatch.setattr(api_main, "GoogleYouTubePublisher", _StubRealPublisher)
    _seed_youtube_vault()
    _seed_baked_gate(project_id)

    video = tmp_path / "final.mp4"
    video.write_bytes(b"fake-master")
    srt = tmp_path / "captions.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:02,000\nhello\n")
    resp = client.post(
        f"/projects/{project_id}/publish-draft",
        json={
            "video_path": str(video),
            "title": "Case 26-0417 — full documentation",
            "srt_path": str(srt),
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["dry_run"] is False
    assert data["video_id"] == "yt_real_123"

    pub = _StubRealPublisher.last_instance
    assert pub is not None and len(pub.uploads) == 1
    assert pub.uploads[0]["video_path"] == str(video)
    assert pub.uploads[0]["srt_path"] == str(srt)

    proj = client.get(f"/projects/{project_id}").json()
    assert proj["status"] == "published_draft"


def test_publish_draft_dry_run_without_vault_creds(client, project_id, tmp_path):
    """No YouTube keys in the vault -> InMemory publisher records a SIMULATION.
    The response says so (dry_run) and the project must NOT be marked published."""
    _seed_baked_gate(project_id)
    video = tmp_path / "final.mp4"
    video.write_bytes(b"fake-master")
    resp = client.post(
        f"/projects/{project_id}/publish-draft",
        json={"video_path": str(video), "title": "Dry run case"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["dry_run"] is True
    assert data["video_id"].startswith("draft_")

    proj = client.get(f"/projects/{project_id}").json()
    assert proj["status"] != "published_draft"

    from director_os.core.db import ProjectRow
    from sqlalchemy import select

    s = __import__("director_os.api.main", fromlist=["session"]).session()
    row = s.execute(select(ProjectRow).where(ProjectRow.id == project_id)).scalar_one()
    assert row.youtube_draft_id is None


def test_publish_draft_refuses_dirty_gate(client, project_id, tmp_path):
    """CR-002 law: a cut the disk does not carry must never reach YouTube —
    no stamps (nothing ever baked) => gate dirty => 409 with an actionable detail."""
    video = tmp_path / "final.mp4"
    video.write_bytes(b"fake")
    resp = client.post(
        f"/projects/{project_id}/publish-draft",
        json={"video_path": str(video), "title": "t"},
    )
    assert resp.status_code == 409
    assert "dirty" in resp.json()["detail"]


def test_publish_draft_happy_path(client, project_id, tmp_path):
    _seed_baked_gate(project_id)
    video = tmp_path / "final.mp4"
    video.write_bytes(b"fake-master")
    resp = client.post(
        f"/projects/{project_id}/publish-draft",
        json={
            "video_path": str(video),
            "title": "Case 26-0417 — full documentation",
            "base_description": "Bodycam documentary cut.",
            "chapter_titles": ["Cold open", "Stop", "Arrest"],
            "chapter_starts_s": [0.0, 30.0, 95.0],
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["privacy_status"] == "private"
    assert data["video_id"].startswith("draft_")
    assert data["dry_run"] is True  # no vault creds in this fixture = simulation

    # a dry run persists NOTHING — no fake draft id, no published status
    from director_os.core.db import ProjectRow
    from sqlalchemy import select

    s = __import__("director_os.api.main", fromlist=["session"]).session()
    row = s.execute(select(ProjectRow).where(ProjectRow.id == project_id)).scalar_one()
    assert row.youtube_draft_id is None
    assert row.status != "published_draft"


def test_publish_draft_404(client):
    resp = client.post(
        "/projects/does-not-exist/publish-draft",
        json={"video_path": "x.mp4", "title": "t"},
    )
    assert resp.status_code == 404


def test_publish_draft_rejects_bad_chapters(client, project_id, tmp_path):
    video = tmp_path / "final.mp4"
    video.write_bytes(b"fake")
    resp = client.post(
        f"/projects/{project_id}/publish-draft",
        json={
            "video_path": str(video),
            "title": "t",
            "chapter_titles": ["a", "b"],  # only 2 chapters -> YouTube rule violation
            "chapter_starts_s": [0.0, 30.0],
        },
    )
    assert resp.status_code == 422
    assert "at least 3" in resp.json()["detail"]


def test_publish_draft_rejects_mismatched_chapters(client, project_id):
    resp = client.post(
        f"/projects/{project_id}/publish-draft",
        json={
            "video_path": "x.mp4",
            "title": "t",
            "chapter_titles": ["a", "b", "c"],
            "chapter_starts_s": [0.0, 30.0],
        },
    )
    assert resp.status_code == 422
    assert "mismatch" in resp.json()["detail"]


def test_publish_draft_rejects_missing_video(client, project_id, tmp_path):
    _seed_baked_gate(project_id)  # gate clean, so the 422 is the FILE, not the gate
    resp = client.post(
        f"/projects/{project_id}/publish-draft",
        json={"video_path": str(tmp_path / "ghost.mp4"), "title": "t"},
    )
    assert resp.status_code == 422
    assert "video missing" in resp.json()["detail"]


def test_publish_draft_title_limit(client, project_id, tmp_path):
    video = tmp_path / "final.mp4"
    video.write_bytes(b"fake")
    resp = client.post(
        f"/projects/{project_id}/publish-draft",
        json={"video_path": str(video), "title": "t" * 101},
    )
    assert resp.status_code == 422


# ------------------------- deliverables (the real last mile) -------------------------
@pytest.fixture()
def deliverable_env(client, tmp_path: Path, monkeypatch):
    """Point OUTPUT_DIR at a per-test dir AFTER the client fixture pinned the
    DB, so get_settings() rebuilds with both."""
    from director_os.core.config import get_settings

    out = tmp_path / "output"
    monkeypatch.setenv("OUTPUT_DIR", str(out))
    get_settings.cache_clear()
    yield out
    get_settings.cache_clear()


def test_deliverables_prefill_from_ship_pack(client, project_id, deliverable_env):
    work = deliverable_env / "pipeline" / project_id
    pack = work / "ship_pack"
    pack.mkdir(parents=True)
    (pack / "final_1080p.mp4").write_bytes(b"pack-master")
    (pack / "captions.srt").write_text("1\n00:00:00,000 --> 00:00:02,000\nhi\n")
    (pack / "metadata.json").write_text("{}")

    resp = client.get(f"/projects/{project_id}/deliverables")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["master_path"] == str(pack / "final_1080p.mp4")
    assert data["master_size_bytes"] == len(b"pack-master")
    assert data["srt_path"] == str(pack / "captions.srt")
    assert data["pack_dir"] == str(pack)
    names = [f["name"] for f in data["pack_files"]]
    assert names == ["captions.srt", "final_1080p.mp4", "metadata.json"]
    assert data["suggested_title"] == "Publish Gate Case"
    assert data["youtube_configured"] is False
    # nothing baked yet -> the gate is dirty and publish will 409 (honest UI)
    assert data["gate_dirty"] is True


def test_deliverables_falls_back_to_postfx_then_final(client, project_id, deliverable_env):
    work = deliverable_env / "pipeline" / project_id
    work.mkdir(parents=True)
    final = work / "final.mp4"
    final.write_bytes(b"pre-fx master")

    resp = client.get(f"/projects/{project_id}/deliverables")
    data = resp.json()
    assert data["master_path"] == str(final)
    assert data["master_is_post_fx"] is False

    postfx = work / "postfx" / "master_postfx.mp4"
    postfx.parent.mkdir()
    postfx.write_bytes(b"post-fx master")
    resp = client.get(f"/projects/{project_id}/deliverables")
    data = resp.json()
    assert data["master_path"] == str(postfx)
    assert data["master_is_post_fx"] is True


def test_deliverables_404(client):
    assert client.get("/projects/does-not-exist/deliverables").status_code == 404


def test_deliverables_youtube_configured_flag(client, project_id, deliverable_env):
    _seed_youtube_vault()
    resp = client.get(f"/projects/{project_id}/deliverables")
    assert resp.status_code == 200
    assert resp.json()["youtube_configured"] is True
