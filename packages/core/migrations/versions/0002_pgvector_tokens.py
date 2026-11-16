"""0002 — pgvector embedding column + agent tool-log token accounting (M2, ADR-0008).

Revision ID: 0002_pgvector_tokens
Revises: 0001_init
Create Date: 2026-11-17

Postgres-only pieces (pgvector extension + `perception_index.embedding vector(1536)`)
are skipped on other dialects (CI runs sqlite). Token columns are added only when
missing, so both migration-created and create_all-created tables converge.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002_pgvector_tokens"
down_revision = "0001_init"
branch_labels = None
depends_on = None

EMBEDDING_DIM = 1536


def upgrade() -> None:
    bind = op.get_bind()

    if bind.dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
        op.execute(
            "ALTER TABLE perception_index "
            f"ADD COLUMN IF NOT EXISTS embedding vector({EMBEDDING_DIM})"
        )

    cols = {c["name"] for c in sa.inspect(bind).get_columns("agent_tool_logs")}
    if "tokens_in" not in cols:
        op.add_column("agent_tool_logs", sa.Column("tokens_in", sa.Integer(), nullable=True))
    if "tokens_out" not in cols:
        op.add_column("agent_tool_logs", sa.Column("tokens_out", sa.Integer(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    cols = {c["name"] for c in sa.inspect(bind).get_columns("agent_tool_logs")}
    if "tokens_out" in cols:
        op.drop_column("agent_tool_logs", "tokens_out")
    if "tokens_in" in cols:
        op.drop_column("agent_tool_logs", "tokens_in")
    if bind.dialect.name == "postgresql":
        op.execute("ALTER TABLE perception_index DROP COLUMN IF EXISTS embedding")
