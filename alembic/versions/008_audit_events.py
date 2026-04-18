"""Add append-only audit_events table with INSERT-only role

Revision ID: 008
Revises: 007
Create Date: 2026-04-18

Architectural Intent:
- Append-only security audit log required by 2026 rules §4.
- Separate DB role `traderbot_audit_writer` has INSERT only — the application
  uses this role when writing audit events so a code bug cannot tamper with
  historical records.
- Reader role `traderbot_audit_reader` has SELECT only — for compliance tools.

The role creation is idempotent and skipped on SQLite (test environment) where
roles don't apply.
"""
from alembic import op
import sqlalchemy as sa

revision = '008'
down_revision = '007'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'audit_events',
        sa.Column('id', sa.String(36), primary_key=True, index=True),
        sa.Column('actor_user_id', sa.String(36), nullable=True, index=True),
        sa.Column('action', sa.String(100), nullable=False, index=True),
        sa.Column('aggregate_type', sa.String(100), nullable=False, index=True),
        sa.Column('aggregate_id', sa.String(100), nullable=False, index=True),
        sa.Column('before_hash', sa.String(64), nullable=True),
        sa.Column('after_hash', sa.String(64), nullable=True),
        sa.Column('payload_json', sa.JSON(), nullable=False),
        sa.Column('occurred_at', sa.DateTime(), nullable=False),
        sa.Column('correlation_id', sa.String(64), nullable=True, index=True),
        sa.Column('client_ip', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.String(500), nullable=True),
    )
    op.create_index('idx_audit_actor', 'audit_events', ['actor_user_id'])
    op.create_index('idx_audit_action', 'audit_events', ['action'])
    op.create_index('idx_audit_aggregate', 'audit_events', ['aggregate_type', 'aggregate_id'])
    op.create_index('idx_audit_occurred_at', 'audit_events', ['occurred_at'])
    op.create_index('idx_audit_correlation', 'audit_events', ['correlation_id'])

    # Postgres-only: create append-only role + reader role. SQLite/MySQL skip.
    bind = op.get_bind()
    dialect = bind.dialect.name
    if dialect == 'postgresql':
        op.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'traderbot_audit_writer') THEN
                    CREATE ROLE traderbot_audit_writer NOLOGIN;
                END IF;
                IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'traderbot_audit_reader') THEN
                    CREATE ROLE traderbot_audit_reader NOLOGIN;
                END IF;
            END
            $$;
            """
        )
        op.execute("GRANT INSERT ON audit_events TO traderbot_audit_writer;")
        op.execute("GRANT SELECT ON audit_events TO traderbot_audit_reader;")
        op.execute(
            "REVOKE UPDATE, DELETE, TRUNCATE ON audit_events FROM traderbot_audit_writer;"
        )


def downgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    if dialect == 'postgresql':
        op.execute("REVOKE ALL ON audit_events FROM traderbot_audit_writer;")
        op.execute("REVOKE ALL ON audit_events FROM traderbot_audit_reader;")

    op.drop_index('idx_audit_correlation', table_name='audit_events')
    op.drop_index('idx_audit_occurred_at', table_name='audit_events')
    op.drop_index('idx_audit_aggregate', table_name='audit_events')
    op.drop_index('idx_audit_action', table_name='audit_events')
    op.drop_index('idx_audit_actor', table_name='audit_events')
    op.drop_table('audit_events')
