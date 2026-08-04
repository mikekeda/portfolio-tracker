"""Add fair-value inputs to features_daily

Absolute analyst target levels, the DCF sensitivity band, and the
fair-price-from-multiples inputs (5y harmonic avg P/E, basis flag, EPS)
exist only in the daily-overwritten InstrumentYahoo cache today, so none
of them can be backtested. Persist the raw inputs point-in-time.

Revision ID: e1f6a3d9c257
Revises: c8d4a1b6e903
Create Date: 2026-08-04 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e1f6a3d9c257'
down_revision: Union[str, Sequence[str], None] = 'c8d4a1b6e903'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_COLUMNS = (
    sa.Column('target_low', sa.Float(), nullable=True),
    sa.Column('target_median', sa.Float(), nullable=True),
    sa.Column('target_high', sa.Float(), nullable=True),
    sa.Column('dcf_low', sa.Float(), nullable=True),
    sa.Column('dcf_high', sa.Float(), nullable=True),
    sa.Column('avg_pe_5y', sa.Float(), nullable=True),
    sa.Column('pe_basis_matches', sa.Boolean(), nullable=True),
    sa.Column('trailing_eps', sa.Float(), nullable=True),
    sa.Column('forward_eps', sa.Float(), nullable=True),
)


def upgrade() -> None:
    """Upgrade schema."""
    for column in _COLUMNS:
        op.add_column('features_daily', column)


def downgrade() -> None:
    """Downgrade schema."""
    for column in reversed(_COLUMNS):
        op.drop_column('features_daily', column.name)
