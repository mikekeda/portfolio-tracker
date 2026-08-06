"""Add sec_companyfacts and sec_features_daily tables.

SEC companyfacts-backed PIT fundamentals for CIK instruments. Kept separate
from features_daily so the Yahoo nightly writer remains single-writer.

Revision ID: f2a8c1d04e91
Revises: e1f6a3d9c257
Create Date: 2026-08-06 08:45:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "f2a8c1d04e91"
down_revision: Union[str, Sequence[str], None] = "e1f6a3d9c257"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "sec_companyfacts",
        sa.Column("instrument_id", sa.Integer(), nullable=False),
        sa.Column("cik", sa.String(length=10), nullable=False),
        sa.Column("facts", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("roic_as_restated", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("fetched_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["instrument_id"], ["instruments.id"]),
        sa.PrimaryKeyConstraint("instrument_id"),
    )
    op.create_table(
        "sec_features_daily",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("instrument_id", sa.Integer(), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("roic", sa.Float(), nullable=True),
        sa.Column("roic_ttm", sa.Float(), nullable=True),
        sa.Column("roic_ttm_trend", sa.Float(), nullable=True),
        sa.Column("gross_margin", sa.Float(), nullable=True),
        sa.Column("operating_margin", sa.Float(), nullable=True),
        sa.Column("profit_margin", sa.Float(), nullable=True),
        sa.Column("revenue_growth", sa.Float(), nullable=True),
        sa.Column("fcf_yield", sa.Float(), nullable=True),
        sa.Column("debt_to_equity", sa.Float(), nullable=True),
        sa.Column("rule_of_40", sa.Float(), nullable=True),
        sa.Column("f_score", sa.Integer(), nullable=True),
        sa.Column("gross_margin_trend_4q", sa.Float(), nullable=True),
        sa.Column("operating_margin_trend_4q", sa.Float(), nullable=True),
        sa.Column("revenue_growth_4q_avg", sa.Float(), nullable=True),
        sa.Column("period_end", sa.Date(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["instrument_id"], ["instruments.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("instrument_id", "date", name="uq_sec_features_instrument_date"),
    )
    op.create_index(op.f("ix_sec_features_daily_date"), "sec_features_daily", ["date"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_sec_features_daily_date"), table_name="sec_features_daily")
    op.drop_table("sec_features_daily")
    op.drop_table("sec_companyfacts")
