# ============================================================================
# IAM — Fargate task roles + GitHub OIDC deploy role
#
# Architectural Intent:
# - Task execution role: what ECS needs to START a task (pull from ECR,
#   write logs, fetch secrets at boot).
# - Task role: what the app code needs at RUNTIME — today just Secrets
#   Manager read for the specific secrets this stack owns.
# - Deploy role: what GitHub Actions assumes via OIDC to push images and
#   update the service. No long-lived AWS credentials in GitHub.
# ============================================================================

# ── Task execution role ────────────────────────────────────────────────────
# Used by the ECS agent itself to pull images, write logs, pull secrets
# (when injected via the `secrets` block in the task definition).
data "aws_iam_policy_document" "task_exec_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "task_execution" {
  name               = "${local.name_prefix}-task-execution"
  assume_role_policy = data.aws_iam_policy_document.task_exec_assume.json
  tags               = { Name = "${local.name_prefix}-task-execution" }
}

resource "aws_iam_role_policy_attachment" "task_execution_base" {
  role       = aws_iam_role.task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Allow the ECS agent to resolve the specific secrets this stack owns.
data "aws_iam_policy_document" "task_exec_secrets" {
  statement {
    effect = "Allow"
    actions = [
      "secretsmanager:GetSecretValue",
      "secretsmanager:DescribeSecret",
    ]
    resources = [
      aws_secretsmanager_secret.app.arn,
      aws_secretsmanager_secret.broker.arn,
    ]
  }
}

resource "aws_iam_role_policy" "task_execution_secrets" {
  name   = "secrets-read"
  role   = aws_iam_role.task_execution.id
  policy = data.aws_iam_policy_document.task_exec_secrets.json
}

# ── Task role (runtime) ────────────────────────────────────────────────────
# What the app code itself can do. Today: read the secrets it was given
# at boot (so it can refresh during a long-lived task if we ever rotate
# a key). Nothing else.
resource "aws_iam_role" "task" {
  name               = "${local.name_prefix}-task"
  assume_role_policy = data.aws_iam_policy_document.task_exec_assume.json
  tags               = { Name = "${local.name_prefix}-task" }
}

resource "aws_iam_role_policy" "task_secrets" {
  name   = "runtime-secrets-read"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.task_exec_secrets.json
}

# ECS Exec uses SSM. We attach the managed policy so operators can exec
# into a running task without opening SSH anywhere.
resource "aws_iam_role_policy_attachment" "task_ssm" {
  role       = aws_iam_role.task.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# ── GitHub OIDC provider + deploy role ─────────────────────────────────────
# Lets GitHub Actions assume an AWS role without any long-lived secret.
# Thumbprint is GitHub's current OIDC signing cert (stable — rotates
# rarely). AWS publishes the current thumbprint in their docs.
resource "aws_iam_openid_connect_provider" "github" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]

  tags = { Name = "github-oidc" }
}

data "aws_iam_policy_document" "github_deploy_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }

    # Only jobs running from main of the exact repo can assume. Tighten
    # further (per-branch, per-workflow) when we add staging / PR preview.
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repository}:ref:refs/heads/main"]
    }
  }
}

resource "aws_iam_role" "github_deploy" {
  name               = "${local.name_prefix}-github-deploy"
  assume_role_policy = data.aws_iam_policy_document.github_deploy_assume.json
  tags               = { Name = "${local.name_prefix}-github-deploy" }
}

# Deploy permissions: push to ECR, update THIS service's task definition,
# roll the deployment, read/write the CI status of the specific service.
data "aws_iam_policy_document" "github_deploy_policy" {
  statement {
    sid    = "EcrAuth"
    effect = "Allow"
    actions = [
      "ecr:GetAuthorizationToken",
    ]
    resources = ["*"]
  }

  statement {
    sid    = "EcrPushPull"
    effect = "Allow"
    actions = [
      "ecr:BatchGetImage",
      "ecr:BatchCheckLayerAvailability",
      "ecr:CompleteLayerUpload",
      "ecr:GetDownloadUrlForLayer",
      "ecr:InitiateLayerUpload",
      "ecr:PutImage",
      "ecr:UploadLayerPart",
    ]
    resources = [aws_ecr_repository.api.arn]
  }

  statement {
    sid    = "EcsDeployApi"
    effect = "Allow"
    actions = [
      "ecs:DescribeServices",
      "ecs:UpdateService",
      "ecs:RegisterTaskDefinition",
      "ecs:DescribeTaskDefinition",
      "ecs:DeregisterTaskDefinition",
      "ecs:ListTaskDefinitions",
    ]
    resources = ["*"] # RegisterTaskDefinition has no resource-level ARN
  }

  statement {
    sid       = "PassTaskRoles"
    effect    = "Allow"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.task_execution.arn, aws_iam_role.task.arn]
  }

  statement {
    sid    = "S3SpaUpload"
    effect = "Allow"
    actions = [
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket",
    ]
    resources = [
      aws_s3_bucket.spa.arn,
      "${aws_s3_bucket.spa.arn}/*",
    ]
  }

  statement {
    sid       = "CloudFrontInvalidate"
    effect    = "Allow"
    actions   = ["cloudfront:CreateInvalidation"]
    resources = [aws_cloudfront_distribution.spa.arn]
  }
}

resource "aws_iam_role_policy" "github_deploy" {
  name   = "deploy"
  role   = aws_iam_role.github_deploy.id
  policy = data.aws_iam_policy_document.github_deploy_policy.json
}
