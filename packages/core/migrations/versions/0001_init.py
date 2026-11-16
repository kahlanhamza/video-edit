"""0001 — initial schema (PLAN §5.1).

Revision ID: 0001_init
Revises: -
Create Date: 2026-11-17
"""

from __future__ import annotations

from alembic import op
from director_os.core.db import Base

revision = "0001_init"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    Base.metadata.create_all(bind)


def downgrade() -> None:
    bind = op.get_bind()
    Base.metadata.drop_all(bind)
