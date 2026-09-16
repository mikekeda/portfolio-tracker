"""Record successful, skipped and failed trade-agent attempts.

Revision ID: d91f7b203ac6
Revises: b3e5f7c21a48
"""

from alembic import op
import sqlalchemy as sa

revision = "d91f7b203ac6"
down_revision = "b3e5f7c21a48"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "trade_agent_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("as_of_date", sa.Date(), nullable=True),
        sa.Column("strategy", sa.String(30), nullable=False),
        sa.Column("ran_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(10), nullable=False),
        sa.Column("intent_count", sa.Integer(), nullable=True),
        sa.Column("order_count", sa.Integer(), nullable=True),
        sa.Column("executable_count", sa.Integer(), nullable=True),
        sa.Column("reason", sa.String(200), nullable=True),
    )
    op.create_index("idx_trade_agent_runs_strategy_ran_at", "trade_agent_runs", ["strategy", "ran_at"])


def downgrade() -> None:
    op.drop_index("idx_trade_agent_runs_strategy_ran_at", table_name="trade_agent_runs")
    op.drop_table("trade_agent_runs")
