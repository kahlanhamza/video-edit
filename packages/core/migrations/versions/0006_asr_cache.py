"""0006 — content-addressed ASR cache (asr_cache).

Gladia keeps finished transcriptions addressable by id, so re-uploading the
same media to re-run ASR is re-billed waste. `asr_cache` binds
(provider, sha256(media)) -> (external transcription id, status, normalized
payload) so a retry ADOPTS the existing transcription (or serves a pure local
hit) instead of uploading again. Idempotent on both create_all- and
migration-created databases.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0006_asr_cache"
down_revision = "0005_job_observability"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    names = set(sa.inspect(bind).get_table_names())

    if "asr_cache" not in names:
        op.create_table(
            "asr_cache",
            sa.Column("provider", sa.String(40), primary_key=True),
            sa.Column("content_sha256", sa.String(64), primary_key=True),
            sa.Column("external_id", sa.String(120), nullable=True),
            sa.Column("status", sa.String(20), nullable=False, server_default="processing"),
            sa.Column("language", sa.String(10), nullable=True),
            sa.Column("transcript_json", sa.JSON(), nullable=True),
            sa.Column("error", sa.String(1000), nullable=True),
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
    if "asr_cache" in names:
        op.drop_table("asr_cache")
