"""Add profile_fetched_at to InstrumentYahoo

Revision ID: a7e21c40f9b3
Revises: 4c95414e08d1
Create Date: 2026-07-03 00:15:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a7e21c40f9b3'
down_revision: Union[str, Sequence[str], None] = '4c95414e08d1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('instruments_yahoo', sa.Column('profile_fetched_at', sa.DateTime(), nullable=True))
    # No backfill from updated_at on purpose: the nightly PE scrapers have been
    # bumping updated_at without fetching, so it overstates freshness. NULL rows
    # sort to the front of the staleness queue and the whole universe re-fetches
    # within ~a day (9 runs x 150 budget vs ~1035 instruments).


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('instruments_yahoo', 'profile_fetched_at')
