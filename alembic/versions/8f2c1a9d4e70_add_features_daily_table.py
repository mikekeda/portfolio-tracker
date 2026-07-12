"""Add features_daily table

Revision ID: 8f2c1a9d4e70
Revises: 5713db02e7c6
Create Date: 2026-07-12 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '8f2c1a9d4e70'
down_revision: Union[str, Sequence[str], None] = '5713db02e7c6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'features_daily',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('instrument_id', sa.Integer(), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('roic', sa.Float(), nullable=True),
        sa.Column('gross_margin', sa.Float(), nullable=True),
        sa.Column('operating_margin', sa.Float(), nullable=True),
        sa.Column('profit_margin', sa.Float(), nullable=True),
        sa.Column('revenue_growth', sa.Float(), nullable=True),
        sa.Column('fcf_yield', sa.Float(), nullable=True),
        sa.Column('debt_to_equity', sa.Float(), nullable=True),
        sa.Column('short_percent_float', sa.Float(), nullable=True),
        sa.Column('analyst_rec_mean', sa.Float(), nullable=True),
        sa.Column('analyst_count', sa.Integer(), nullable=True),
        sa.Column('analyst_target_upside', sa.Float(), nullable=True),
        sa.Column('forward_pe', sa.Float(), nullable=True),
        sa.Column('peg', sa.Float(), nullable=True),
        sa.Column('ps_ratio', sa.Float(), nullable=True),
        sa.Column('pe_5y_avg_vs_current_pct', sa.Float(), nullable=True),
        sa.Column('dcf_price', sa.Float(), nullable=True),
        sa.Column('dcf_diff', sa.Float(), nullable=True),
        sa.Column('dcf_implied_growth', sa.Float(), nullable=True),
        sa.Column('rule_of_40', sa.Float(), nullable=True),
        sa.Column('f_score', sa.Integer(), nullable=True),
        sa.Column('screener_score', sa.Float(), nullable=True),
        sa.Column('passed_screeners', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('thesis_rule_eval', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('extras', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['instrument_id'], ['instruments.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('instrument_id', 'date', name='uq_features_instrument_date'),
    )
    op.create_index(op.f('ix_features_daily_date'), 'features_daily', ['date'], unique=False)
    op.create_index('idx_features_instrument_date', 'features_daily', ['instrument_id', 'date'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('idx_features_instrument_date', table_name='features_daily')
    op.drop_index(op.f('ix_features_daily_date'), table_name='features_daily')
    op.drop_table('features_daily')
