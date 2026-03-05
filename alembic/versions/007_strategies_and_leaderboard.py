"""Add saved_strategies, backtest_results, and strategy_follows tables

Revision ID: 007
Revises: 006
Create Date: 2026-03-05
"""
from alembic import op
import sqlalchemy as sa

revision = '007'
down_revision = '006'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'saved_strategies',
        sa.Column('id', sa.String(36), primary_key=True, index=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), nullable=False, index=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.String(1000), nullable=False, server_default=''),
        sa.Column('strategy_type', sa.String(50), nullable=False),
        sa.Column('parameters', sa.JSON(), nullable=False, server_default='{}'),
        sa.Column('symbol', sa.String(10), nullable=False, server_default='AAPL'),
        sa.Column('is_public', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('fork_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )
    op.create_index('idx_strategy_user_id', 'saved_strategies', ['user_id'])
    op.create_index('idx_strategy_public', 'saved_strategies', ['is_public'])

    op.create_table(
        'backtest_results',
        sa.Column('id', sa.String(36), primary_key=True, index=True),
        sa.Column('strategy_id', sa.String(36), sa.ForeignKey('saved_strategies.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), nullable=False, index=True),
        sa.Column('symbol', sa.String(10), nullable=False),
        sa.Column('initial_capital', sa.Numeric(15, 2), nullable=False),
        sa.Column('final_value', sa.Numeric(15, 2), nullable=False),
        sa.Column('total_return_pct', sa.Numeric(8, 2), nullable=False),
        sa.Column('sharpe_ratio', sa.Numeric(8, 4), nullable=False),
        sa.Column('max_drawdown_pct', sa.Numeric(8, 2), nullable=False),
        sa.Column('win_rate', sa.Numeric(5, 2), nullable=False),
        sa.Column('total_trades', sa.Integer(), nullable=False),
        sa.Column('volatility', sa.Numeric(8, 2), nullable=False),
        sa.Column('profit_factor', sa.Numeric(8, 4), nullable=False),
        sa.Column('run_at', sa.DateTime(), nullable=False),
    )
    op.create_index('idx_backtest_strategy_id', 'backtest_results', ['strategy_id'])
    op.create_index('idx_backtest_user_id', 'backtest_results', ['user_id'])

    op.create_table(
        'strategy_follows',
        sa.Column('id', sa.String(36), primary_key=True, index=True),
        sa.Column('follower_user_id', sa.String(36), sa.ForeignKey('users.id'), nullable=False, index=True),
        sa.Column('strategy_id', sa.String(36), sa.ForeignKey('saved_strategies.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
    )
    op.create_index('idx_follow_user_strategy', 'strategy_follows', ['follower_user_id', 'strategy_id'], unique=True)


def downgrade() -> None:
    op.drop_table('strategy_follows')
    op.drop_table('backtest_results')
    op.drop_table('saved_strategies')
