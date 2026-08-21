"""add etf_holdings

Moves ETF look-through holdings out of the data.py literal and into the DB:
they are fetched from issuer APIs quarterly, not hand-maintained, and keying on
the issuer's identifier lets constituents that are not tracked instruments be
stored instead of discarded at fetch time.

Revision ID: a7c31e5b90d4
Revises: f2a8c1d04e91
"""

import sqlalchemy as sa
from alembic import op

revision = "a7c31e5b90d4"
down_revision = "f2a8c1d04e91"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "etf_holdings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("etf_instrument_id", sa.Integer(), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("source_key", sa.String(length=20), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=True),
        sa.Column("weight_pct", sa.Float(), nullable=False),
        sa.Column("instrument_id", sa.Integer(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["etf_instrument_id"], ["instruments.id"]),
        sa.ForeignKeyConstraint(["instrument_id"], ["instruments.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("etf_instrument_id", "date", "source_key", name="uq_etf_holding"),
    )
    op.create_index("ix_etf_holdings_date", "etf_holdings", ["date"])
    op.create_index("ix_etf_holdings_instrument_id", "etf_holdings", ["instrument_id"])


def downgrade() -> None:
    op.drop_index("ix_etf_holdings_instrument_id", table_name="etf_holdings")
    op.drop_index("ix_etf_holdings_date", table_name="etf_holdings")
    op.drop_table("etf_holdings")
