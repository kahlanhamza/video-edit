"""0004 — multi-source case ingestion (case_sources + case_dossiers).

M5 "flexible Multi-Source Bundle": any combination of PDFs, URLs, and pasted
text attaches to a project alongside the video URL(s). `case_dossiers` caches
the latest synthesized CaseDossier so the Director prompt and the Studio
inspector read the same JSON. Idempotent on both create_all- and
migration-created databases (tables are only created when missing).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004_case_sources"
down_revision = "0003_r2_enhanced_key"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    names = set(sa.inspect(bind).get_table_names())

    if "case_sources" not in names:
        op.create_table(
            "case_sources",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "project_id",
                sa.String(36),
                sa.ForeignKey("projects.id"),
                index=True,
                nullable=False,
            ),
            sa.Column("kind", sa.String(20), nullable=False),
            sa.Column("label", sa.String(300), nullable=False, server_default=""),
            sa.Column("uri_or_storage_key", sa.String(2048), nullable=False, server_default=""),
            sa.Column("text", sa.Text(), nullable=False),
            sa.Column("extracted_text", sa.Text(), nullable=False),
            sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
            sa.Column("error", sa.String(1000), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
        )

    if "case_dossiers" not in names:
        op.create_table(
            "case_dossiers",
            sa.Column("project_id", sa.String(36), sa.ForeignKey("projects.id"), primary_key=True),
            sa.Column("dossier_json", sa.JSON(), nullable=False),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
        )


def downgrade() -> None:
    bind = op.get_bind()
    names = set(sa.inspect(bind).get_table_names())
    if "case_dossiers" in names:
        op.drop_table("case_dossiers")
    if "case_sources" in names:
        op.drop_table("case_sources")
