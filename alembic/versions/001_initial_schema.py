"""Initial schema - users, portfolios, orders, positions, domain_events

Revision ID: 001
Revises: None
Create Date: 2026-02-16
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Users table
    op.create_table(
        'users',
        sa.Column('id', sa.String(36), primary_key=True, index=True),
        sa.Column('email', sa.String(255), unique=True, nullable=False, index=True),
        sa.Column('first_name', sa.String(255), nullable=False),
        sa.Column('last_name', sa.String(255), nullable=False),
        sa.Column('password_hash', sa.String(255), nullable=False),
        sa.Column('risk_tolerance', sa.Enum('CONSERVATIVE', 'MODERATE', 'AGGRESSIVE', name='risktolerance'), nullable=False, server_default='MODERATE'),
        sa.Column('investment_goal', sa.Enum('CAPITAL_PRESERVATION', 'BALANCED_GROWTH', 'MAXIMUM_RETURNS', name='investmentgoal'), nullable=False, server_default='BALANCED_GROWTH'),
        sa.Column('max_position_size_percentage', sa.Numeric(5, 2), nullable=False, server_default='5'),
        sa.Column('daily_loss_limit', sa.Numeric(12, 2), nullable=True),
        sa.Column('weekly_loss_limit', sa.Numeric(12, 2), nullable=True),
        sa.Column('monthly_loss_limit', sa.Numeric(12, 2), nullable=True),
        sa.Column('sector_preferences', sa.JSON, nullable=False, server_default='[]'),
        sa.Column('sector_exclusions', sa.JSON, nullable=False, server_default='[]'),
        sa.Column('is_active', sa.Boolean, nullable=False, server_default='true'),
        sa.Column('email_notifications_enabled', sa.Boolean, nullable=False, server_default='true'),
        sa.Column('sms_notifications_enabled', sa.Boolean, nullable=False, server_default='false'),
        sa.Column('approval_mode_enabled', sa.Boolean, nullable=False, server_default='false'),
        sa.Column('terms_accepted_at', sa.DateTime, nullable=True),
        sa.Column('privacy_accepted_at', sa.DateTime, nullable=True),
        sa.Column('marketing_consent', sa.Boolean, nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index('idx_user_email', 'users', ['email'])
    op.create_index('idx_user_created_at', 'users', ['created_at'])

    # Portfolios table
    op.create_table(
        'portfolios',
        sa.Column('id', sa.String(36), primary_key=True, index=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), nullable=False, index=True),
        sa.Column('total_value', sa.Numeric(15, 2), nullable=False, server_default='0'),
        sa.Column('cash_balance', sa.Numeric(15, 2), nullable=False, server_default='0'),
        sa.Column('invested_value', sa.Numeric(15, 2), nullable=False, server_default='0'),
        sa.Column('total_gain_loss', sa.Numeric(15, 2), nullable=False, server_default='0'),
        sa.Column('total_return_percentage', sa.Numeric(5, 2), nullable=False, server_default='0'),
        sa.Column('ytd_return_percentage', sa.Numeric(5, 2), nullable=False, server_default='0'),
        sa.Column('current_drawdown', sa.Numeric(5, 2), nullable=False, server_default='0'),
        sa.Column('peak_value', sa.Numeric(15, 2), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index('idx_portfolio_user_id', 'portfolios', ['user_id'])
    op.create_index('idx_portfolio_created_at', 'portfolios', ['created_at'])

    # Orders table
    op.create_table(
        'orders',
        sa.Column('id', sa.String(36), primary_key=True, index=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), nullable=False, index=True),
        sa.Column('symbol', sa.String(10), nullable=False, index=True),
        sa.Column('order_type', sa.Enum('MARKET', 'LIMIT', 'STOP_LOSS', 'TRAILING_STOP', name='ordertype'), nullable=False),
        sa.Column('position_type', sa.Enum('LONG', 'SHORT', name='positiontype'), nullable=False),
        sa.Column('quantity', sa.Numeric(12, 2), nullable=False),
        sa.Column('status', sa.Enum('PENDING', 'EXECUTED', 'CANCELLED', 'FAILED', name='orderstatus'), nullable=False, server_default='PENDING', index=True),
        sa.Column('price', sa.Numeric(12, 4), nullable=True),
        sa.Column('limit_price', sa.Numeric(12, 4), nullable=True),
        sa.Column('stop_price', sa.Numeric(12, 4), nullable=True),
        sa.Column('commission', sa.Numeric(12, 4), nullable=True, server_default='0'),
        sa.Column('filled_quantity', sa.Numeric(12, 2), nullable=False, server_default='0'),
        sa.Column('placed_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('executed_at', sa.DateTime, nullable=True),
        sa.Column('notes', sa.String(500), nullable=True),
    )
    op.create_index('idx_order_user_id', 'orders', ['user_id'])
    op.create_index('idx_order_status', 'orders', ['status'])
    op.create_index('idx_order_symbol', 'orders', ['symbol'])
    op.create_index('idx_order_user_status', 'orders', ['user_id', 'status'])
    op.create_index('idx_order_placed_at', 'orders', ['placed_at'])

    # Positions table
    op.create_table(
        'positions',
        sa.Column('id', sa.String(36), primary_key=True, index=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), nullable=False, index=True),
        sa.Column('symbol', sa.String(10), nullable=False, index=True),
        sa.Column('quantity', sa.Numeric(12, 2), nullable=False),
        sa.Column('position_type', sa.Enum('LONG', 'SHORT', name='positiontype', create_type=False), nullable=False),
        sa.Column('average_entry_price', sa.Numeric(12, 4), nullable=False),
        sa.Column('current_price', sa.Numeric(12, 4), nullable=True),
        sa.Column('unrealized_gain_loss', sa.Numeric(15, 2), nullable=False, server_default='0'),
        sa.Column('realized_gain_loss', sa.Numeric(15, 2), nullable=False, server_default='0'),
        sa.Column('opened_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('closed_at', sa.DateTime, nullable=True),
        sa.Column('updated_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index('idx_position_user_id', 'positions', ['user_id'])
    op.create_index('idx_position_symbol', 'positions', ['symbol'])
    op.create_index('idx_position_user_symbol', 'positions', ['user_id', 'symbol'])
    op.create_index('idx_position_opened_at', 'positions', ['opened_at'])

    # Domain Events table
    op.create_table(
        'domain_events',
        sa.Column('id', sa.String(36), primary_key=True, index=True),
        sa.Column('aggregate_type', sa.String(100), nullable=False, index=True),
        sa.Column('aggregate_id', sa.String(36), nullable=False, index=True),
        sa.Column('event_type', sa.String(100), nullable=False, index=True),
        sa.Column('event_data', sa.JSON, nullable=False),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), nullable=True, index=True),
        sa.Column('occurred_at', sa.DateTime, nullable=False, index=True),
        sa.Column('recorded_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index('idx_event_aggregate', 'domain_events', ['aggregate_type', 'aggregate_id'])
    op.create_index('idx_event_type', 'domain_events', ['event_type'])
    op.create_index('idx_event_user_id', 'domain_events', ['user_id'])
    op.create_index('idx_event_occurred_at', 'domain_events', ['occurred_at'])


def downgrade() -> None:
    op.drop_table('domain_events')
    op.drop_table('positions')
    op.drop_table('orders')
    op.drop_table('portfolios')
    op.drop_table('users')

    # Drop enum types
    op.execute("DROP TYPE IF EXISTS risktolerance")
    op.execute("DROP TYPE IF EXISTS investmentgoal")
    op.execute("DROP TYPE IF EXISTS ordertype")
    op.execute("DROP TYPE IF EXISTS positiontype")
    op.execute("DROP TYPE IF EXISTS orderstatus")
