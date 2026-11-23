"""0008 — unique (project_id, version_num) on timeline_versions (CR-007).

Two concurrent agent dispatches SELECT-max then INSERT the next checkpoint
version; without a constraint both could claim the same version_num and the
rollback chain corrupts silently (the gate reads `active > baked` — duplicate
numbers make those comparisons lie). The store retries on IntegrityError.

Idempotent: dedupes existing duplicate rows first (keep the newest id per
(project_id, version_num) — the later checkpoint is the one the live timeline
continued from), then creates the unique index only when absent.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0008_version_unique"
down_revision = "0007_timeline_ops"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    names = set(sa.inspect(bind).get_table_names())
    if "timeline_versions" not in names:
        return

    # dedupe: keep the row with the greatest id (the most recent write) —
    # checkpoints are content snapshots, the newest one is what the timeline
    # actually continued from
    bind.execute(
        sa.text(
            """
        DELETE FROM timeline_versions
        WHERE id NOT IN (
            SELECT MAX(id) FROM timeline_versions GROUP BY project_id, version_num
        )
        """
        )
    )

    indexes = {ix["name"] for ix in sa.inspect(bind).get_indexes("timeline_versions")}
    if "uq_versions_project_num" not in indexes:
        op.create_index(
            "uq_versions_project_num",
            "timeline_versions",
            ["project_id", "version_num"],
            unique=True,
        )


def downgrade() -> None:
    bind = op.get_bind()
    names = set(sa.inspect(bind).get_table_names())
    if "timeline_versions" not in names:
        return
    indexes = {ix["name"] for ix in sa.inspect(bind).get_indexes("timeline_versions")}
    if "uq_versions_project_num" in indexes:
        op.drop_index("uq_versions_project_num", table_name="timeline_versions")
