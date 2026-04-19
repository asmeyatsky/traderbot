# ============================================================================
# Outputs — values CI needs + things operators will paste into consoles
# ============================================================================

output "ecr_repository_url" {
  description = "Full ECR URL. CI pushes images here."
  value       = aws_ecr_repository.api.repository_url
}

output "ecs_cluster_name" {
  description = "ECS cluster the API service runs in."
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "ECS service name. CI calls `update-service` against this."
  value       = aws_ecs_service.api.name
}

output "ecs_task_execution_role_arn" {
  description = "Role the ECS agent assumes to pull images + secrets."
  value       = aws_iam_role.task_execution.arn
}

output "ecs_task_role_arn" {
  description = "Role the running container assumes."
  value       = aws_iam_role.task.arn
}

output "github_deploy_role_arn" {
  description = "IAM role GitHub Actions assumes via OIDC to deploy. Paste into the workflow's role-to-assume input."
  value       = aws_iam_role.github_deploy.arn
}

output "alb_dns_name" {
  description = "ALB DNS — internal. CloudFront uses this. Not published to users."
  value       = aws_lb.api.dns_name
}

output "cloudfront_distribution_id" {
  description = "Distribution ID. CI calls CreateInvalidation against this."
  value       = aws_cloudfront_distribution.spa.id
}

output "cloudfront_domain_name" {
  description = "CloudFront domain — the Route 53 apex aliases here."
  value       = aws_cloudfront_distribution.spa.domain_name
}

output "spa_bucket_name" {
  description = "S3 bucket for SPA assets. CI syncs `frontend/dist/` here."
  value       = aws_s3_bucket.spa.id
}

output "app_secret_name" {
  description = "Secrets Manager secret for app config. Passed to ECS as AWS_SECRETS_NAME."
  value       = aws_secretsmanager_secret.app.name
}

output "app_secret_arn" {
  description = "ARN of the app secret — needed for `aws secretsmanager put-secret-value`."
  value       = aws_secretsmanager_secret.app.arn
}

output "broker_secret_arn" {
  description = "ARN of the broker secret (Alpaca keys)."
  value       = aws_secretsmanager_secret.broker.arn
}

output "db_endpoint" {
  description = "RDS endpoint. Operators connect here for migrations or ad-hoc SQL."
  value       = aws_db_instance.postgres.address
}

output "db_port" {
  value = aws_db_instance.postgres.port
}

output "cache_endpoint" {
  description = "ElastiCache primary endpoint."
  value       = aws_elasticache_replication_group.cache.primary_endpoint_address
}

output "db_master_password" {
  description = "Initial DB master password. Rotate via SecretsManager + RDS modify_password after first use."
  value       = random_password.db_master.result
  sensitive   = true
}
