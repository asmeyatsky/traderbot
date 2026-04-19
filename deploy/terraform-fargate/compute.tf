# ============================================================================
# Compute plane — ECR, ECS cluster, task definition, service, autoscaling
#
# Architectural Intent:
# - Single ECS cluster on Fargate (no EC2 backing). Multiple services
#   could share it later (background workers, staging).
# - Task definition pulls image by tag from ECR. CI pins the tag per-
#   commit via `aws ecs register-task-definition` and then calls
#   `update-service` with `--force-new-deployment` to roll.
# - Rolling deploy with circuit_breaker enabled: if the new task fails
#   healthchecks, ECS auto-rolls back to the last known-good task def.
# - Secrets injected at container start via the `secrets` block —
#   populated from Secrets Manager, never shown in task definition JSON.
# ============================================================================

# ── ECR ────────────────────────────────────────────────────────────────────
resource "aws_ecr_repository" "api" {
  name                 = "${local.name_prefix}-api"
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = { Name = "${local.name_prefix}-api-ecr" }
}

# Keep only the last N tagged images to bound cost. Untagged layers
# orphaned by a failed push are removed after 1 day.
resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep the last ${var.ecr_lifecycle_keep_images} tagged images"
        selection = {
          tagStatus     = "tagged"
          tagPatternList = ["*"]
          countType     = "imageCountMoreThan"
          countNumber   = var.ecr_lifecycle_keep_images
        }
        action = { type = "expire" }
      },
      {
        rulePriority = 2
        description  = "Expire untagged layers after 1 day"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 1
        }
        action = { type = "expire" }
      },
    ]
  })
}

# ── ECS cluster ────────────────────────────────────────────────────────────
resource "aws_ecs_cluster" "main" {
  name = "${local.name_prefix}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled" # CloudWatch deep metrics. Costs ~$0.30 per task per month.
  }

  tags = { Name = "${local.name_prefix}-cluster" }
}

# Use only the on-demand Fargate provider today. Spot is a Phase 8
# optimization — added via a second capacity_provider_strategy when
# we have >2 tasks so a Spot preemption doesn't halve capacity.
resource "aws_ecs_cluster_capacity_providers" "main" {
  cluster_name       = aws_ecs_cluster.main.name
  capacity_providers = ["FARGATE"]

  default_capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight            = 100
    base              = 0
  }
}

# ── CloudWatch log group for task stdout ───────────────────────────────────
resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${local.name_prefix}-api"
  retention_in_days = 14 # balances operator forensics with cost

  tags = { Name = "${local.name_prefix}-api-logs" }
}

# ── Task definition ────────────────────────────────────────────────────────
resource "aws_ecs_task_definition" "api" {
  family                   = "${local.name_prefix}-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.api_cpu
  memory                   = var.api_memory

  execution_role_arn = aws_iam_role.task_execution.arn
  task_role_arn      = aws_iam_role.task.arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64" # Dockerfile targets python:3.11-slim (multi-arch but prefer x86 for CI parity)
  }

  container_definitions = jsonencode([
    {
      name      = "api"
      image     = "${aws_ecr_repository.api.repository_url}:${var.api_image_tag}"
      essential = true

      portMappings = [
        {
          containerPort = 8000
          protocol      = "tcp"
        }
      ]

      environment = [
        { name = "ENVIRONMENT", value = "production" },
        { name = "LOG_LEVEL", value = "INFO" },
        { name = "AWS_REGION", value = var.aws_region },
        # Phase 3 secrets path — app resolves AWS_SECRETS_NAME via boto3.
        { name = "AWS_SECRETS_NAME", value = aws_secretsmanager_secret.app.name },
        # Kill switches (ADR-002). Defaults keep LIVE users on paper.
        { name = "EMERGENCY_HALT", value = "false" },
        { name = "ENABLE_LIVE_TRADING", value = "false" },
      ]

      # Anything in the app secret lands as an env var via this block;
      # the ECS agent does the Secrets Manager lookup at container start
      # using the task-execution role.
      secrets = [
        {
          name      = "JWT_SECRET_KEY"
          valueFrom = "${aws_secretsmanager_secret.app.arn}:JWT_SECRET_KEY::"
        },
        {
          name      = "ANTHROPIC_API_KEY"
          valueFrom = "${aws_secretsmanager_secret.app.arn}:ANTHROPIC_API_KEY::"
        },
        {
          name      = "DATABASE_URL"
          valueFrom = "${aws_secretsmanager_secret.app.arn}:DATABASE_URL::"
        },
        {
          name      = "REDIS_URL"
          valueFrom = "${aws_secretsmanager_secret.app.arn}:REDIS_URL::"
        },
        {
          name      = "ALPACA_API_KEY"
          valueFrom = "${aws_secretsmanager_secret.broker.arn}:ALPACA_API_KEY::"
        },
        {
          name      = "ALPACA_SECRET_KEY"
          valueFrom = "${aws_secretsmanager_secret.broker.arn}:ALPACA_SECRET_KEY::"
        },
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.api.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "api"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8000/ready || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60 # Python + uvicorn + DI container boot
      }

      readonlyRootFilesystem = false # alembic writes temp files
      linuxParameters = {
        initProcessEnabled = true # tini → clean shutdown
      }
    }
  ])

  tags = { Name = "${local.name_prefix}-api-taskdef" }
}

# ── Service ────────────────────────────────────────────────────────────────
resource "aws_ecs_service" "api" {
  name                               = "${local.name_prefix}-api"
  cluster                            = aws_ecs_cluster.main.id
  task_definition                    = aws_ecs_task_definition.api.arn
  desired_count                      = var.api_desired_count
  launch_type                        = "FARGATE"
  deployment_minimum_healthy_percent = 100 # never drop below desired during rollouts
  deployment_maximum_percent         = 200
  enable_execute_command             = true # operator ECS Exec
  platform_version                   = "LATEST"
  propagate_tags                     = "SERVICE"
  force_new_deployment               = false # CI sets force-new via update-service

  network_configuration {
    subnets          = [for s in aws_subnet.public : s.id]
    security_groups  = [aws_security_group.fargate.id]
    assign_public_ip = true # no NAT GW; egress via VPC endpoints + public route
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8000
  }

  # Deployment circuit-breaker: if the new task set fails healthchecks,
  # auto-rollback to the previous task def. No operator intervention.
  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  lifecycle {
    # CI updates task_definition on every deploy; Terraform shouldn't
    # diff against it or re-apply on every plan.
    ignore_changes = [task_definition, desired_count]
  }

  tags = { Name = "${local.name_prefix}-api-service" }

  depends_on = [aws_lb_listener.https]
}

# ── Autoscaling ────────────────────────────────────────────────────────────
resource "aws_appautoscaling_target" "api" {
  max_capacity       = var.api_max_count
  min_capacity       = var.api_desired_count
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.api.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "api_cpu" {
  name               = "${local.name_prefix}-api-cpu"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.api.resource_id
  scalable_dimension = aws_appautoscaling_target.api.scalable_dimension
  service_namespace  = aws_appautoscaling_target.api.service_namespace

  target_tracking_scaling_policy_configuration {
    target_value = 60.0
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    scale_in_cooldown  = 300 # 5 min — slow to scale down so a traffic dip doesn't flap
    scale_out_cooldown = 60  # fast to scale up on a spike
  }
}
