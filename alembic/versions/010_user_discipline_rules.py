"""Add per-user discipline rules + trading philosophy (Phase 10.1)

Revision ID: 010
Revises: 009
Create Date: 2026-04-18

Architectural Intent:
- `discipline_rules` is a JSON array of user-written rule strings. The
  pre-trade AI veto layer evaluates every proposed order against every
  rule; any violation produces a structured veto the UI surfaces with an
  explicit override path.
- `trading_philosophy` is a free-text paragraph describing the user's
  high-level approach. The AI uses it as context for ambiguous orders
  where a narrow rule doesn't fire but the trade still feels off.
- Both columns are nullable / empty by default; users who set nothing
  see zero behaviour change.
"""
from alembic import op
import sqlalchemy as sa


revision = '010'
down_revision = '009'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'users',
        sa.Column(
            'discipline_rules',
            sa.JSON,
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
    )
    op.add_column(
        'users',
        sa.Column(
            'trading_philosophy',
            sa.Text,
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column('users', 'trading_philosophy')
    op.drop_column('users', 'discipline_rules')
