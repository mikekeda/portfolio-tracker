"""Add screener_score_max to features_daily

Sectors excluded from several screeners can never reach the same raw
screener_score, so the agent needs the achievable maximum to z-score fairly.

Revision ID: a4c91f7b2d38
Revises: 7523599802f3
Create Date: 2026-07-30 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a4c91f7b2d38'
down_revision: Union[str, Sequence[str], None] = '7523599802f3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('features_daily', sa.Column('screener_score_max', sa.Float(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('features_daily', 'screener_score_max')
