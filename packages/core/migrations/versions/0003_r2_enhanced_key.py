"""0003 — media_assets.r2_enhanced_key (M4 audio: 2-bus speech isolation).

The per-asset 2-bus master WAV produced by `audio.enhance_speech` is persisted
on the asset row so re-plans/re-renders reuse it (content-addressed on disk,
receipt JSON alongside). Column is added only when missing, so both
migration-created and create_all-created tables converge (same law as 0002).

Revision ID: 0003_r2_enhanced_key
Revises: 0002_pgvector_tokens
Create Date: 2026-11-17
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003_r2_enhanced_key"
down_revision = "0002_pgvector_tokens"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    cols = {c["name"] for c in sa.inspect(bind).get_columns("media_assets")}
    if "r2_enhanced_key" not in cols:
        op.add_column("media_assets", sa.Column("r2_enhanced_key", sa.String(1024), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    cols = {c["name"] for c in sa.inspect(bind).get_columns("media_assets")}
    if "r2_enhanced_key" in cols:
        op.drop_column("media_assets", "r2_enhanced_key")
