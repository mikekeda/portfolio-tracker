"""add instruments_yahoo.derived_metrics

Fund-level fundamentals aggregated from etf_holdings constituents. Lives on the
Yahoo cache row rather than in its own table because update_data writes that row
field by field, so a derived column survives the 2-hourly refresh, and because
widening instruments_metrics_daily would be a fifteen-column migration for a
display cache that deliberately keeps no history.

Revision ID: b3e5f7c21a48
Revises: a7c31e5b90d4
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "b3e5f7c21a48"
down_revision = "a7c31e5b90d4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "instruments_yahoo",
        sa.Column("derived_metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("instruments_yahoo", "derived_metrics")
