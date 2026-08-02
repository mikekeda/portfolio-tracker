"""Drop indexes duplicated by a unique constraint or covered by a composite

models.py declared both a UniqueConstraint and an Index on the same columns for
six tables, so Postgres maintained two identical btrees for each. prices_daily
carried 111 MB of pure duplicate: one of its three indexes, rewritten on every
non-HOT update, and that table has taken 9.7M updates at a 23% HOT ratio.

Reads are unaffected — the surviving unique index has identical structure, so
the planner uses it for the same lookups. Scan counts were split arbitrarily
between the twins (idx_symbol_date 2.7M vs uq_symbol_date 11.7M) and simply
consolidate onto the survivor.

The three ix_*_instrument_id / ix_*_csv_id drops are single-column indexes whose
column already leads a composite (or a unique constraint) on the same table.

Revision ID: c8d4a1b6e903
Revises: b7e3f2a91c04
Create Date: 2026-08-02 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'c8d4a1b6e903'
down_revision: Union[str, Sequence[str], None] = 'b7e3f2a91c04'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# (index name, table, columns — the columns exist only to rebuild on downgrade)
_REDUNDANT_INDEXES = (
    ('idx_symbol_date', 'prices_daily', ['symbol', 'date']),
    ('idx_metrics_instrument_date', 'instruments_metrics_daily', ['instrument_id', 'date']),
    ('idx_features_instrument_date', 'features_daily', ['instrument_id', 'date']),
    ('idx_holding_instrument_date', 'holdings_daily', ['instrument_id', 'date']),
    ('idx_earnings_reports_instrument_date', 'earnings_reports', ['instrument_id', 'date']),
    ('ix_transaction_history_csv_id', 'transaction_history', ['csv_id']),
    ('ix_earnings_reports_instrument_id', 'earnings_reports', ['instrument_id']),
    ('ix_position_reviews_instrument_id', 'position_reviews', ['instrument_id']),
)


def upgrade() -> None:
    # CONCURRENTLY avoids the ACCESS EXCLUSIVE lock: a plain DROP waits for the
    # in-flight holdings query and queues new ones behind it. Needs autocommit.
    with op.get_context().autocommit_block():
        for index, table, _ in _REDUNDANT_INDEXES:
            op.drop_index(index, table_name=table, postgresql_concurrently=True, if_exists=True)


def downgrade() -> None:
    with op.get_context().autocommit_block():
        for index, table, columns in _REDUNDANT_INDEXES:
            op.create_index(index, table, columns, postgresql_concurrently=True, if_not_exists=True)
