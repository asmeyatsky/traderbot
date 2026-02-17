# =============================================================================
# KMS Customer-Managed Key for TraderBot
#
# Provides encryption at rest for RDS, S3, CloudWatch Logs, Secrets Manager,
# and other AWS services. Key rotation is enabled for compliance.
#
# Architectural Intent:
# - Single CMK for all encryption needs (cost-effective)
# - Key policy grants access to ECS task role and AWS services
# - Automatic annual key rotation for PCI DSS / ISO 27001 compliance
# =============================================================================

resource "aws_kms_key" "main" {
  description             = "TraderBot CMK for encryption at rest"
  deletion_window_in_days = 30
  enable_key_rotation     = true
  multi_region            = false

  policy = jsonencode({
    Version = "2012-10-17"
    Id      = "traderbot-key-policy"
    Statement = [
      {
        Sid    = "EnableRootAccountFullAccess"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${local.aws_account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "AllowECSTaskRoleUsage"
        Effect = "Allow"
        Principal = {
          AWS = aws_iam_role.ecs_task.arn
        }
        Action = [
          "kms:Decrypt",
          "kms:DescribeKey",
          "kms:GenerateDataKey*",
        ]
        Resource = "*"
      },
      {
        Sid    = "AllowECSExecutionRoleUsage"
        Effect = "Allow"
        Principal = {
          AWS = aws_iam_role.ecs_task_execution.arn
        }
        Action = [
          "kms:Decrypt",
          "kms:DescribeKey",
          "kms:GenerateDataKey*",
        ]
        Resource = "*"
      },
      {
        Sid    = "AllowCloudWatchLogsUsage"
        Effect = "Allow"
        Principal = {
          Service = "logs.${var.aws_region}.amazonaws.com"
        }
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey",
        ]
        Resource = "*"
        Condition = {
          ArnLike = {
            "kms:EncryptionContext:aws:logs:arn" = "arn:aws:logs:${var.aws_region}:${local.aws_account_id}:log-group:*"
          }
        }
      },
      {
        Sid    = "AllowS3BucketUsage"
        Effect = "Allow"
        Principal = {
          Service = "s3.amazonaws.com"
        }
        Action = [
          "kms:Decrypt",
          "kms:GenerateDataKey*",
        ]
        Resource = "*"
      },
      {
        Sid    = "AllowCloudTrailUsage"
        Effect = "Allow"
        Principal = {
          Service = "cloudtrail.amazonaws.com"
        }
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey",
        ]
        Resource = "*"
        Condition = {
          StringLike = {
            "kms:EncryptionContext:aws:cloudtrail:arn" = "arn:aws:cloudtrail:${var.aws_region}:${local.aws_account_id}:trail/*"
          }
        }
      },
    ]
  })

  tags = {
    Name = "${local.name_prefix}-cmk"
  }
}

resource "aws_kms_alias" "main" {
  name          = "alias/${local.name_prefix}"
  target_key_id = aws_kms_key.main.key_id
}
