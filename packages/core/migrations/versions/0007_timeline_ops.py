"""0007 — the op log (timeline_ops).

Everything-is-an-op: an EditOp row is the atom of editing — kind, frame
window, track, params, origin (agent|human), evidence. Undo = revoke (the
row stays; the audit trail the bodycam niche legally requires). `rev` is the
monotonic cursor the SSE op stream and every client fold use. Idempotent on
both create_all- and migration-created databases.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0007_timeline_ops"
down_revision = "0006_asr_cache"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    names = set(sa.inspect(bind).get_table_names())

    if "timeline_ops" not in names:
        op.create_table(
            "timeline_ops",
            sa.Column("rev", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("id", sa.String(36), nullable=False, unique=True),
            sa.Column("project_id", sa.String(36), nullable=False, index=True),
            sa.Column("asset_id", sa.String(36), nullable=True),
            sa.Column("kind", sa.String(40), nullable=False),
            sa.Column("frame_start", sa.Integer(), nullable=True),
            sa.Column("frame_end", sa.Integer(), nullable=True),
            sa.Column("track", sa.String(60), nullable=False, server_default="V1_master"),
            sa.Column("params", sa.JSON(), nullable=False, server_default="{}"),
            sa.Column("origin", sa.String(20), nullable=False, server_default="human"),
            sa.Column("evidence", sa.JSON(), nullable=False, server_default="{}"),
            sa.Column("session_id", sa.String(64), nullable=True),
            sa.Column("note", sa.String(1000), nullable=False, server_default=""),
            sa.Column("revokes_op_id", sa.String(36), nullable=True),
            sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
        )


def downgrade() -> None:
    bind = op.get_bind()
    names = set(sa.inspect(bind).get_table_names())
    if "timeline_ops" in names:
        op.drop_table("timeline_ops")
