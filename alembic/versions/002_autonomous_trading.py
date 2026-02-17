"""Add autonomous trading support - user auto-trading fields, broker_order_id, activity log

Revision ID: 002
Revises: 001
Create Date: 2026-02-17
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add auto-trading columns to users table
    op.add_column('users', sa.Column('auto_trading_enabled', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('users', sa.Column('watchlist', sa.JSON(), nullable=False, server_default='[]'))
    op.add_column('users', sa.Column('trading_budget', sa.Numeric(15, 2), nullable=True))

    # Add broker_order_id to orders table
    op.add_column('orders', sa.Column('broker_order_id', sa.String(255), nullable=True))

    # Create trading_activity_log table
    op.create_table(
        'trading_activity_log',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('event_type', sa.String(50), nullable=False),
        sa.Column('symbol', sa.String(10), nullable=True),
        sa.Column('signal', sa.String(20), nullable=True),
        sa.Column('confidence', sa.Numeric(5, 4), nullable=True),
        sa.Column('order_id', sa.String(36), sa.ForeignKey('orders.id'), nullable=True),
        sa.Column('broker_order_id', sa.String(255), nullable=True),
        sa.Column('quantity', sa.Numeric(12, 2), nullable=True),
        sa.Column('price', sa.Numeric(12, 4), nullable=True),
        sa.Column('message', sa.String(1000), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('occurred_at', sa.DateTime(), nullable=False),
        sa.Column('recorded_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('idx_activity_user_id', 'trading_activity_log', ['user_id'])
    op.create_index('idx_activity_event_type', 'trading_activity_log', ['event_type'])
    op.create_index('idx_activity_occurred_at', 'trading_activity_log', ['occurred_at'])
    op.create_index('idx_activity_user_event', 'trading_activity_log', ['user_id', 'event_type'])


def downgrade() -> None:
    op.drop_table('trading_activity_log')
    op.drop_column('orders', 'broker_order_id')
    op.drop_column('users', 'trading_budget')
    op.drop_column('users', 'watchlist')
    op.drop_column('users', 'auto_trading_enabled')
