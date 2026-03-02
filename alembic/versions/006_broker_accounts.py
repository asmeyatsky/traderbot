"""Add broker_accounts table for per-user broker linking

Revision ID: 006
Revises: 005
Create Date: 2026-03-02
"""
from alembic import op
import sqlalchemy as sa

revision = '006'
down_revision = '005'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'broker_accounts',
        sa.Column('id', sa.String(36), primary_key=True, index=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), nullable=False, index=True),
        sa.Column('broker_type', sa.String(50), nullable=False),
        sa.Column('encrypted_api_key', sa.String(500), nullable=False),
        sa.Column('encrypted_secret_key', sa.String(500), nullable=False),
        sa.Column('paper_trading', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('label', sa.String(255), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )
    op.create_index(
        'idx_broker_account_user_broker',
        'broker_accounts',
        ['user_id', 'broker_type'],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index('idx_broker_account_user_broker', table_name='broker_accounts')
    op.drop_table('broker_accounts')
