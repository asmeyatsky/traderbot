"""Add user-configurable auto-trading controls: stop-loss, take-profit, confidence, max position

Revision ID: 003
Revises: 002
Create Date: 2026-02-23
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '003'
down_revision: Union[str, None] = '002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('users', sa.Column('stop_loss_pct', sa.Numeric(5, 2), nullable=False, server_default='5'))
    op.add_column('users', sa.Column('take_profit_pct', sa.Numeric(5, 2), nullable=False, server_default='10'))
    op.add_column('users', sa.Column('confidence_threshold', sa.Numeric(3, 2), nullable=False, server_default='0.6'))
    op.add_column('users', sa.Column('max_position_pct', sa.Numeric(5, 2), nullable=False, server_default='20'))


def downgrade() -> None:
    op.drop_column('users', 'max_position_pct')
    op.drop_column('users', 'confidence_threshold')
    op.drop_column('users', 'take_profit_pct')
    op.drop_column('users', 'stop_loss_pct')
