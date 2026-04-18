"""Add per-user live-trading mode columns

Revision ID: 009
Revises: 008
Create Date: 2026-04-18

Architectural Intent (ADR-002):
- `trading_mode` defaults to 'paper' for every user. Flipping to 'live'
  requires the full enable-live-mode flow (KYC attestation + TOTP + daily
  loss cap + risk acknowledgement phrase).
- `daily_loss_cap_usd` caps how much a user can lose before they're
  auto-reverted to 'paper'.
- `kyc_attestation_hash` stores a sha256 of the attestation payload +
  timestamp + IP so we have tamper evidence without keeping raw PII.
- `totp_secret_encrypted` stores the per-user TOTP secret; encryption key
  is JWT_SECRET_KEY for simplicity today — Phase 8 can swap to a dedicated
  KMS key if regulatory scrutiny demands it.
"""
from alembic import op
import sqlalchemy as sa

revision = '009'
down_revision = '008'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'users',
        sa.Column(
            'trading_mode',
            sa.String(10),
            nullable=False,
            server_default='paper',
        ),
    )
    op.add_column(
        'users',
        sa.Column(
            'daily_loss_cap_usd',
            sa.Numeric(12, 2),
            nullable=True,
        ),
    )
    op.add_column(
        'users',
        sa.Column(
            'kyc_attestation_hash',
            sa.String(64),
            nullable=True,
        ),
    )
    op.add_column(
        'users',
        sa.Column(
            'totp_secret_encrypted',
            sa.String(255),
            nullable=True,
        ),
    )
    op.add_column(
        'users',
        sa.Column(
            'live_mode_enabled_at',
            sa.DateTime(),
            nullable=True,
        ),
    )

    # CHECK constraint: trading_mode is one of 'paper' / 'live'. SQLite and
    # Postgres both honour CHECK constraints from Alembic's DDL API.
    op.create_check_constraint(
        'ck_users_trading_mode',
        'users',
        "trading_mode IN ('paper', 'live')",
    )


def downgrade() -> None:
    op.drop_constraint('ck_users_trading_mode', 'users', type_='check')
    op.drop_column('users', 'live_mode_enabled_at')
    op.drop_column('users', 'totp_secret_encrypted')
    op.drop_column('users', 'kyc_attestation_hash')
    op.drop_column('users', 'daily_loss_cap_usd')
    op.drop_column('users', 'trading_mode')
