"""Persistence: SQLAlchemy models on sqlite + Alembic 0001 migration."""

from __future__ import annotations

import sqlite3

import pytest
from director_os.core.db import (
    MediaAssetRow,
    ProjectRow,
    init_models,
    make_engine,
    make_session_factory,
)
from director_os.core.models import AudioContract
from sqlalchemy import select


def test_audio_contract_is_the_law():
    assert AudioContract().sample_rate == 48000
    assert AudioContract().channels == 2
    with pytest.raises(ValueError, match="audio law"):
        AudioContract(sample_rate=44100)
    with pytest.raises(ValueError, match="audio law"):
        AudioContract(channels=1)


def test_models_roundtrip_on_sqlite(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/dev.db")
    init_models(engine)
    s = make_session_factory(engine)()
    p = ProjectRow(title="Case 1 — demo", niche_skill="bodycam_documentary")
    s.add(p)
    s.commit()
    s.add(
        MediaAssetRow(
            project_id=p.id,
            source_url="https://example.org/a.webm",
            duration_s=414.4,
            fps=30.0,
            width=1280,
            height=720,
        )
    )
    s.commit()
    row = s.execute(select(ProjectRow).where(ProjectRow.id == p.id)).scalar_one()
    assert row.title == "Case 1 — demo"
    asset = s.execute(select(MediaAssetRow)).scalar_one()
    assert asset.project_id == p.id
    assert asset.duration_s == pytest.approx(414.4)


def test_alembic_upgrade_head(tmp_path, monkeypatch):
    alembic_config = pytest.importorskip("alembic.config")
    alembic_command = pytest.importorskip("alembic.command")

    from pathlib import Path

    ini = Path(__file__).resolve().parents[2] / "packages" / "core" / "alembic.ini"
    db = tmp_path / "mig.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")

    cfg = alembic_config.Config(str(ini))
    alembic_command.upgrade(cfg, "head")

    con = sqlite3.connect(db)
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    con.close()
    assert {
        "projects",
        "media_assets",
        "timeline_versions",
        "segment_cache",
        "agent_tool_logs",
        "jobs",
        "asr_cache",
        "timeline_ops",
    } <= tables
