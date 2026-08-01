"""Add quarterly statements, analyst estimates, and derived trend features

Quarterly statements are the only source of multi-period fundamentals — annual
payloads stay because Piotroski and roic_3y_min depend on them.

Revision ID: b7e3f2a91c04
Revises: a4c91f7b2d38
Create Date: 2026-08-01 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'b7e3f2a91c04'
down_revision: Union[str, Sequence[str], None] = 'a4c91f7b2d38'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_YAHOO_JSONB_COLUMNS = (
    'quarterly_cashflow',
    'quarterly_balance_sheet',
    'quarterly_income_stmt',
    'estimates',
)

_FEATURE_COLUMNS = (
    'roic_ttm',
    'roic_ttm_trend',
    'gross_margin_trend_4q',
    'operating_margin_trend_4q',
    'revenue_growth_4q_avg',
    'eps_revision_ratio_30d',
    'eps_next_q_growth',
)


def upgrade() -> None:
    """Upgrade schema."""
    for name in _YAHOO_JSONB_COLUMNS:
        op.add_column(
            'instruments_yahoo',
            sa.Column(name, postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        )
    op.add_column('instruments_yahoo', sa.Column('estimates_fetched_at', sa.DateTime(), nullable=True))

    for name in _FEATURE_COLUMNS:
        op.add_column('features_daily', sa.Column(name, sa.Float(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    for name in _FEATURE_COLUMNS:
        op.drop_column('features_daily', name)

    op.drop_column('instruments_yahoo', 'estimates_fetched_at')
    for name in _YAHOO_JSONB_COLUMNS:
        op.drop_column('instruments_yahoo', name)
