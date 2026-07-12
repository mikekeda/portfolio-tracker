"""Add trade_suggestions table

Revision ID: b41d7c25e9a1
Revises: 8f2c1a9d4e70
Create Date: 2026-07-12 12:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'b41d7c25e9a1'
down_revision: Union[str, Sequence[str], None] = '8f2c1a9d4e70'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'trade_suggestions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('instrument_id', sa.Integer(), nullable=False),
        sa.Column('strategy', sa.String(length=30), nullable=False),
        sa.Column('action', sa.String(length=10), nullable=False),
        sa.Column('quantity', sa.Float(), nullable=True),
        sa.Column('value_gbp', sa.Float(), nullable=False),
        sa.Column('weight_before', sa.Float(), nullable=True),
        sa.Column('weight_after', sa.Float(), nullable=True),
        sa.Column('score', sa.Float(), nullable=True),
        sa.Column('fee_gbp', sa.Float(), nullable=True),
        sa.Column('rationale', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('constraint_adjustments', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('status', sa.String(length=10), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['instrument_id'], ['instruments.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('date', 'instrument_id', 'strategy', name='uq_suggestion_date_instrument_strategy'),
    )
    op.create_index(op.f('ix_trade_suggestions_date'), 'trade_suggestions', ['date'], unique=False)
    op.create_index(op.f('ix_trade_suggestions_instrument_id'), 'trade_suggestions', ['instrument_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_trade_suggestions_instrument_id'), table_name='trade_suggestions')
    op.drop_index(op.f('ix_trade_suggestions_date'), table_name='trade_suggestions')
    op.drop_table('trade_suggestions')
