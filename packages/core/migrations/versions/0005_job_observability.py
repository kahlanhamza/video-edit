"""0005 — job observability (job_steps + job_logs + job_events).

The full-visibility Studio mandate: every pipeline step streams its own
progress bar, every log line lands in the embedded sandbox console, and the
NLE timeline pushes revisions live. `job_events` is the append-only SSE
cursor feed; `job_steps` / `job_logs` are the durable snapshots behind it.
Idempotent on both create_all- and migration-created databases.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0005_job_observability"
down_revision = "0004_case_sources"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    names = set(sa.inspect(bind).get_table_names())

    if "job_steps" not in names:
        op.create_table(
            "job_steps",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column(
                "job_id",
                sa.String(36),
                sa.ForeignKey("jobs.id"),
                index=True,
                nullable=False,
            ),
            sa.Column("project_id", sa.String(36), index=True, nullable=False),
            sa.Column("key", sa.String(80), nullable=False),
            sa.Column("label", sa.String(200), nullable=False),
            sa.Column("status", sa.String(20), nullable=False, server_default="running"),
            sa.Column("pct", sa.Float(), nullable=False, server_default="0"),
            sa.Column("detail", sa.String(1000), nullable=False, server_default=""),
            sa.Column("order", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("duration_ms", sa.Integer(), nullable=False, server_default="0"),
        )

    if "job_logs" not in names:
        op.create_table(
            "job_logs",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column(
                "job_id",
                sa.String(36),
                sa.ForeignKey("jobs.id"),
                index=True,
                nullable=False,
            ),
            sa.Column("project_id", sa.String(36), index=True, nullable=False),
            sa.Column("level", sa.String(10), nullable=False, server_default="info"),
            sa.Column("message", sa.Text(), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
        )

    if "job_events" not in names:
        op.create_table(
            "job_events",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column(
                "job_id",
                sa.String(36),
                sa.ForeignKey("jobs.id"),
                index=True,
                nullable=False,
            ),
            sa.Column("project_id", sa.String(36), index=True, nullable=False),
            sa.Column("kind", sa.String(20), nullable=False),
            sa.Column("payload", sa.JSON(), nullable=False),
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
    for table in ("job_events", "job_logs", "job_steps"):
        if table in names:
            op.drop_table(table)
