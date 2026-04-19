# ============================================================================
# Data plane — RDS Postgres, ElastiCache Valkey, Secrets Manager
#
# Architectural Intent:
# - RDS is the managed replacement for the Docker Postgres volume. PITR
#   covers 7 days; the off-site backup cron from Phase 7 is dropped.
# - ElastiCache Valkey is Redis-compatible wire-protocol (KeyDB is too,
#   so zero app changes). Single-node t4g.micro; add replicas if the
#   cache becomes load-bearing.
# - One Secrets Manager secret for the app, one for broker keys. The
#   Phase 3 settings module already reads via AWS_SECRETS_NAME.
# ============================================================================

# ── Random passwords ──────────────────────────────────────────────────────
resource "random_password" "db_master" {
  length  = 32
  special = true
  # Certain characters break Postgres connection strings when urlencoded
  # sloppily; avoid them.
  override_special = "!@#$%^&*()-_=+"
}

# ── RDS Postgres ───────────────────────────────────────────────────────────
resource "aws_db_instance" "postgres" {
  identifier     = "${local.name_prefix}-postgres"
  engine         = "postgres"
  engine_version = "15"

  instance_class        = var.db_instance_class
  allocated_storage     = var.db_allocated_storage
  max_allocated_storage = var.db_max_allocated_storage
  storage_type          = "gp3"
  storage_encrypted     = true

  db_name  = "traderbot"
  username = "trading"
  password = random_password.db_master.result
  port     = 5432

  db_subnet_group_name   = aws_db_subnet_group.rds.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  publicly_accessible    = false
  multi_az               = var.db_multi_az

  # PITR + daily snapshots. Hour chosen in a typical low-traffic window.
  backup_retention_period  = var.db_backup_retention_days
  backup_window            = "03:00-04:00"
  maintenance_window       = "sun:04:00-sun:05:00"
  delete_automated_backups = false

  # Protection from 'terraform destroy' catastrophes. Disable manually
  # only during a planned migration away.
  deletion_protection = true
  skip_final_snapshot = false
  final_snapshot_identifier = "${local.name_prefix}-postgres-final-${formatdate("YYYYMMDDhhmm", timestamp())}"

  # Performance Insights is free for 7 days of history on t4g.micro.
  performance_insights_enabled          = true
  performance_insights_retention_period = 7

  # Logs shipped to CloudWatch.
  enabled_cloudwatch_logs_exports = ["postgresql"]

  tags = { Name = "${local.name_prefix}-postgres" }

  lifecycle {
    ignore_changes = [
      # Final snapshot identifier recomputes on every plan due to timestamp().
      # We only want it at destroy time, not as a diff.
      final_snapshot_identifier,
    ]
  }
}

# ── ElastiCache Valkey ─────────────────────────────────────────────────────
# Using replication_group even for a single-node setup so future replica
# adds don't require destroying the primary.
resource "aws_elasticache_replication_group" "cache" {
  replication_group_id = "${local.name_prefix}-cache"
  description          = "TraderBot cache (rate limits, TOTP nonces, WS fanout)"

  engine               = "valkey"
  engine_version       = "7.2"
  node_type            = var.cache_node_type
  port                 = 6379
  parameter_group_name = "default.valkey7"

  num_node_groups         = 1
  replicas_per_node_group = 0 # single-node; add replicas when needed

  subnet_group_name  = aws_elasticache_subnet_group.cache.name
  security_group_ids = [aws_security_group.cache.id]

  automatic_failover_enabled = false # requires replicas; upgrade later
  at_rest_encryption_enabled = true
  transit_encryption_enabled = false # app doesn't use TLS to cache today

  maintenance_window       = "sun:05:00-sun:06:00"
  snapshot_retention_limit = 0 # cache data is rebuildable; don't pay for snapshots

  tags = { Name = "${local.name_prefix}-cache" }
}

# ── Secrets Manager ────────────────────────────────────────────────────────
# ONE secret per concern. The app reads this via AWS_SECRETS_NAME, which
# the Phase 3 settings module already supports. We create the secret
# shells here; populating values is a one-shot CLI step operators run
# once (documented in the README).

resource "aws_secretsmanager_secret" "app" {
  name        = "${local.name_prefix}/app"
  description = "Core app secrets: JWT_SECRET_KEY, ANTHROPIC_API_KEY, POLYGON_API_KEY, etc."

  # Secrets Manager has a hard recovery window; 7 days is the minimum
  # that still lets operators recover from an accidental delete without
  # leaving dead secrets lying around.
  recovery_window_in_days = 7

  tags = { Name = "${local.name_prefix}-secret-app" }
}

resource "aws_secretsmanager_secret" "broker" {
  name                    = "${local.name_prefix}/broker"
  description             = "Alpaca API keys — isolated so broker rotation is a single-secret operation."
  recovery_window_in_days = 7

  tags = { Name = "${local.name_prefix}-secret-broker" }
}

# Terraform doesn't store the secret *value*; operator runs:
#   aws secretsmanager put-secret-value \
#       --secret-id traderbot-prod/app \
#       --secret-string '{"JWT_SECRET_KEY": "...", "ANTHROPIC_API_KEY": "..."}'

# We write the DB connection URL into the app secret alongside JWT/Anthropic.
# Terraform manages this one value (passwords are random); operators edit
# the rest via the CLI. The version is identified by a stable key so
# re-applies don't churn versions.
resource "aws_secretsmanager_secret_version" "app_db_url" {
  secret_id = aws_secretsmanager_secret.app.id
  secret_string = jsonencode({
    DATABASE_URL = "postgresql://${aws_db_instance.postgres.username}:${urlencode(random_password.db_master.result)}@${aws_db_instance.postgres.address}:${aws_db_instance.postgres.port}/${aws_db_instance.postgres.db_name}?sslmode=require"
    REDIS_URL    = "redis://${aws_elasticache_replication_group.cache.primary_endpoint_address}:6379/0"
  })

  lifecycle {
    # Operators merge other keys into the secret via CLI; don't clobber them.
    ignore_changes = [secret_string]
  }
}
