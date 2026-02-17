# ╔══════════════════════════════════════════════════════════════════╗
# ║  WARNING: STAGING vs PRODUCTION NETWORKING DIFFERENCES          ║
# ║                                                                  ║
# ║  Staging uses cost-optimized networking (~$69/mo):               ║
# ║    - t4g.nano NAT instance instead of NAT Gateway                ║
# ║    - No VPC Interface Endpoints (traffic routes via NAT)         ║
# ║                                                                  ║
# ║  Production MUST use full networking (~$168/mo):                  ║
# ║    - Managed NAT Gateway (HA, no patching, auto-scaling)         ║
# ║    - VPC Endpoints for ECR, Secrets Manager, CloudWatch Logs     ║
# ║      (keeps traffic off public internet, required for PCI DSS)   ║
# ║                                                                  ║
# ║  DO NOT copy staging tfvars/config to production.                ║
# ║  Always deploy production with: -var="environment=production"    ║
# ╚══════════════════════════════════════════════════════════════════╝

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  backend "s3" {
    bucket         = "traderbot-terraform-state"
    key            = "traderbot/terraform.tfstate"
    region         = "eu-west-2"
    encrypt        = true
    dynamodb_table = "traderbot-terraform-locks"
  }
}

# DynamoDB table for Terraform state locking
resource "aws_dynamodb_table" "terraform_locks" {
  name         = "traderbot-terraform-locks"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }

  tags = {
    Name = "traderbot-terraform-locks"
  }
}

variable "environment" {
  description = "Environment name (staging/production)"
  type        = string
  default     = "staging"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "eu-west-2"
}

variable "domain_name" {
  description = "Domain name for ACM certificate (leave empty to skip HTTPS setup)"
  type        = string
  default     = "traderbotapp.com"
}

data "aws_caller_identity" "current" {}

locals {
  name_prefix    = "traderbot-${var.environment}"
  aws_account_id = data.aws_caller_identity.current.account_id

  default_tags = {
    Environment        = var.environment
    Project            = "traderbot"
    DataClassification = "confidential"
    Compliance         = "pci-gdpr-iso27001"
    ManagedBy          = "terraform"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = local.default_tags
  }
}

# Production safeguard: ensure NAT Gateway and VPC endpoints exist in production
check "production_networking" {
  assert {
    condition = (
      var.environment != "production" ||
      (length(aws_nat_gateway.main) > 0 && length(aws_vpc_endpoint.ecr_dkr) > 0)
    )
    error_message = "Production MUST have NAT Gateway and VPC endpoints. Check environment variable."
  }
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}


# =============================================================================
# VPC
# =============================================================================

resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "${local.name_prefix}-vpc"
  }
}

# Private Subnets (RDS, Redis, ECS tasks)
resource "aws_subnet" "private_1" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 4, 0)
  availability_zone       = "${var.aws_region}a"
  map_public_ip_on_launch = false

  tags = { Name = "${local.name_prefix}-private-1" }
}

resource "aws_subnet" "private_2" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 4, 1)
  availability_zone       = "${var.aws_region}b"
  map_public_ip_on_launch = false

  tags = { Name = "${local.name_prefix}-private-2" }
}

# Public Subnets (ALB only)
resource "aws_subnet" "public_1" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 4, 2)
  availability_zone       = "${var.aws_region}a"
  map_public_ip_on_launch = true

  tags = { Name = "${local.name_prefix}-public-1" }
}

resource "aws_subnet" "public_2" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 4, 3)
  availability_zone       = "${var.aws_region}b"
  map_public_ip_on_launch = true

  tags = { Name = "${local.name_prefix}-public-2" }
}

# Public Route Table
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = { Name = "${local.name_prefix}-public-rt" }
}

resource "aws_route_table_association" "public_1" {
  subnet_id      = aws_subnet.public_1.id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "public_2" {
  subnet_id      = aws_subnet.public_2.id
  route_table_id = aws_route_table.public.id
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = { Name = "${local.name_prefix}-igw" }
}

# NAT Gateway (production) — managed HA, auto-scaling, no patching required
# Production-only: staging uses a t3.nano NAT instance to save ~$31/mo
resource "aws_eip" "nat" {
  domain = "vpc"

  tags = { Name = "${local.name_prefix}-eip" }
}

resource "aws_nat_gateway" "main" {
  count = var.environment == "production" ? 1 : 0

  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public_1.id

  tags = { Name = "${local.name_prefix}-nat-gw" }
}

# ---------------------------------------------------------------------------
# NAT Instance (staging) — t3.nano fck-nat AMI, ~$4/mo vs $35/mo NAT Gateway
# Staging-only: production uses managed NAT Gateway for HA and compliance
# ---------------------------------------------------------------------------

data "aws_ami" "fck_nat" {
  count = var.environment != "production" ? 1 : 0

  most_recent = true
  owners      = ["568608671756"] # fck-nat project

  filter {
    name   = "name"
    values = ["fck-nat-al2023-*-arm64-ebs"]
  }

  filter {
    name   = "architecture"
    values = ["arm64"]
  }
}

resource "aws_security_group" "nat_instance" {
  count = var.environment != "production" ? 1 : 0

  name        = "${local.name_prefix}-nat-instance"
  description = "Security group for NAT instance (staging)"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "All traffic from VPC"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = [var.vpc_cidr]
  }

  egress {
    description = "All outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${local.name_prefix}-nat-instance-sg" }
}

resource "aws_instance" "nat" {
  count = var.environment != "production" ? 1 : 0

  ami                    = data.aws_ami.fck_nat[0].id
  instance_type          = "t4g.nano"
  subnet_id              = aws_subnet.public_1.id
  vpc_security_group_ids = [aws_security_group.nat_instance[0].id]
  source_dest_check      = false

  tags = { Name = "${local.name_prefix}-nat-instance" }
}

resource "aws_eip_association" "nat_instance" {
  count = var.environment != "production" ? 1 : 0

  instance_id   = aws_instance.nat[0].id
  allocation_id = aws_eip.nat.id
}

# Private Route Table — routes to NAT Gateway (production) or NAT instance (staging)
resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id

  tags = { Name = "${local.name_prefix}-private-rt" }
}

resource "aws_route" "private_nat" {
  route_table_id         = aws_route_table.private.id
  destination_cidr_block = "0.0.0.0/0"
  nat_gateway_id         = var.environment == "production" ? aws_nat_gateway.main[0].id : null
  network_interface_id   = var.environment != "production" ? aws_instance.nat[0].primary_network_interface_id : null
}

resource "aws_route_table_association" "private_1" {
  subnet_id      = aws_subnet.private_1.id
  route_table_id = aws_route_table.private.id
}

resource "aws_route_table_association" "private_2" {
  subnet_id      = aws_subnet.private_2.id
  route_table_id = aws_route_table.private.id
}

# =============================================================================
# VPC Endpoints (private connectivity for ECS in private subnets)
# =============================================================================

# Security group for VPC Interface Endpoints
# Production-only: staging routes ECR/Secrets/Logs traffic through NAT instance
resource "aws_security_group" "vpc_endpoints" {
  count = var.environment == "production" ? 1 : 0

  name        = "${local.name_prefix}-vpc-endpoints"
  description = "Security group for VPC Interface Endpoints"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 443
    to_port         = 443
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs.id]
  }

  tags = { Name = "${local.name_prefix}-vpc-endpoints-sg" }
}

# ECR Docker endpoint (for pulling images)
# Production-only: keeps ECR traffic off public internet (PCI DSS requirement)
resource "aws_vpc_endpoint" "ecr_dkr" {
  count = var.environment == "production" ? 1 : 0

  vpc_id              = aws_vpc.main.id
  service_name        = "com.amazonaws.${var.aws_region}.ecr.dkr"
  vpc_endpoint_type   = "Interface"
  private_dns_enabled = true
  subnet_ids          = [aws_subnet.private_1.id, aws_subnet.private_2.id]
  security_group_ids  = [aws_security_group.vpc_endpoints[0].id]

  tags = { Name = "${local.name_prefix}-ecr-dkr-endpoint" }
}

# ECR API endpoint
# Production-only: keeps ECR API traffic off public internet (PCI DSS requirement)
resource "aws_vpc_endpoint" "ecr_api" {
  count = var.environment == "production" ? 1 : 0

  vpc_id              = aws_vpc.main.id
  service_name        = "com.amazonaws.${var.aws_region}.ecr.api"
  vpc_endpoint_type   = "Interface"
  private_dns_enabled = true
  subnet_ids          = [aws_subnet.private_1.id, aws_subnet.private_2.id]
  security_group_ids  = [aws_security_group.vpc_endpoints[0].id]

  tags = { Name = "${local.name_prefix}-ecr-api-endpoint" }
}

# Secrets Manager endpoint
# Production-only: keeps secrets traffic off public internet (PCI DSS requirement)
resource "aws_vpc_endpoint" "secretsmanager" {
  count = var.environment == "production" ? 1 : 0

  vpc_id              = aws_vpc.main.id
  service_name        = "com.amazonaws.${var.aws_region}.secretsmanager"
  vpc_endpoint_type   = "Interface"
  private_dns_enabled = true
  subnet_ids          = [aws_subnet.private_1.id, aws_subnet.private_2.id]
  security_group_ids  = [aws_security_group.vpc_endpoints[0].id]

  tags = { Name = "${local.name_prefix}-secretsmanager-endpoint" }
}

# CloudWatch Logs endpoint
# Production-only: keeps log traffic off public internet (PCI DSS requirement)
resource "aws_vpc_endpoint" "logs" {
  count = var.environment == "production" ? 1 : 0

  vpc_id              = aws_vpc.main.id
  service_name        = "com.amazonaws.${var.aws_region}.logs"
  vpc_endpoint_type   = "Interface"
  private_dns_enabled = true
  subnet_ids          = [aws_subnet.private_1.id, aws_subnet.private_2.id]
  security_group_ids  = [aws_security_group.vpc_endpoints[0].id]

  tags = { Name = "${local.name_prefix}-logs-endpoint" }
}

# S3 Gateway endpoint (for ECR image layers and ALB logs)
resource "aws_vpc_endpoint" "s3" {
  vpc_id            = aws_vpc.main.id
  service_name      = "com.amazonaws.${var.aws_region}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = [aws_route_table.private.id]
}

# =============================================================================
# Subnet Groups
# =============================================================================

resource "aws_db_subnet_group" "main" {
  name       = local.name_prefix
  subnet_ids = [aws_subnet.private_1.id, aws_subnet.private_2.id]

  tags = { Name = "${local.name_prefix}-db-subnet" }
}

resource "aws_elasticache_subnet_group" "main" {
  name       = local.name_prefix
  subnet_ids = [aws_subnet.private_1.id, aws_subnet.private_2.id]
}

# =============================================================================
# RDS PostgreSQL (Hardened)
# =============================================================================

resource "aws_db_parameter_group" "main" {
  name   = "${local.name_prefix}-pg15"
  family = "postgres15"

  parameter {
    name  = "rds.force_ssl"
    value = "1"
  }

  parameter {
    name         = "shared_preload_libraries"
    value        = "pgaudit"
    apply_method = "pending-reboot"
  }

  parameter {
    name  = "pgaudit.log"
    value = "ddl,role"
  }

  parameter {
    name  = "log_connections"
    value = "1"
  }

  parameter {
    name  = "log_disconnections"
    value = "1"
  }

  parameter {
    name  = "log_statement"
    value = "ddl"
  }

  tags = { Name = "${local.name_prefix}-pg-params" }
}

resource "aws_db_instance" "main" {
  identifier            = local.name_prefix
  engine                = "postgres"
  engine_version        = "15.16"
  instance_class        = var.environment == "production" ? "db.t3.medium" : "db.t3.micro"
  allocated_storage     = 20
  max_allocated_storage = 100
  storage_encrypted     = true
  kms_key_id            = aws_kms_key.main.arn

  username               = "trading"
  password               = random_password.db_password.result
  db_name                = "traderbot"
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  parameter_group_name   = aws_db_parameter_group.main.name

  multi_az                = var.environment == "production"
  backup_retention_period = var.environment == "production" ? 35 : 7
  backup_window           = "03:00-04:00"
  maintenance_window      = "mon:04:00-mon:05:00"

  performance_insights_enabled    = true
  performance_insights_kms_key_id = aws_kms_key.main.arn

  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]

  skip_final_snapshot       = var.environment != "production"
  final_snapshot_identifier = var.environment == "production" ? "${local.name_prefix}-final" : null

  tags = { Name = local.name_prefix }
}

# =============================================================================
# Security Groups
# =============================================================================

resource "aws_security_group" "rds" {
  name        = "${local.name_prefix}-rds"
  description = "RDS security group"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs.id]
  }

  tags = { Name = "${local.name_prefix}-rds-sg" }
}

resource "aws_security_group" "redis" {
  name        = "${local.name_prefix}-redis"
  description = "ElastiCache security group"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs.id]
  }

  tags = { Name = "${local.name_prefix}-redis-sg" }
}

resource "aws_security_group" "ecs" {
  name        = "${local.name_prefix}-ecs"
  description = "ECS tasks security group"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "ALB to ECS"
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  # Egress to VPC CIDR (RDS, Redis, VPC endpoints)
  egress {
    description = "VPC internal traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = [var.vpc_cidr]
  }

  # Egress to NAT Gateway for external API calls (market data, etc.)
  egress {
    description = "HTTPS to internet via NAT"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${local.name_prefix}-ecs-sg" }
}

resource "aws_security_group" "alb" {
  name        = "${local.name_prefix}-alb"
  description = "ALB security group"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTP (redirects to HTTPS)"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${local.name_prefix}-alb-sg" }
}

# =============================================================================
# ALB (with HTTPS, access logs)
# =============================================================================

resource "aws_lb" "main" {
  name               = local.name_prefix
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = [aws_subnet.public_1.id, aws_subnet.public_2.id]

  enable_deletion_protection = var.environment == "production"

  access_logs {
    bucket  = aws_s3_bucket.alb_logs.id
    enabled = true
  }

  tags = { Name = local.name_prefix }

  depends_on = [aws_s3_bucket_policy.alb_logs]
}

# ALB Target Group
resource "aws_lb_target_group" "main" {
  name        = local.name_prefix
  port        = 80
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout             = 5
    interval            = 30
    path                = "/health"
    matcher             = "200"
  }
}

# ACM Certificate (only if domain_name is provided)
resource "aws_acm_certificate" "main" {
  count             = var.domain_name != "" ? 1 : 0
  domain_name       = var.domain_name
  validation_method = "DNS"

  subject_alternative_names = ["*.${var.domain_name}"]

  lifecycle {
    create_before_destroy = true
  }

  tags = { Name = "${local.name_prefix}-cert" }
}

# DNS validation records for ACM certificate
data "aws_route53_zone" "main" {
  count = var.domain_name != "" ? 1 : 0
  name  = var.domain_name
}

resource "aws_route53_record" "cert_validation" {
  for_each = {
    for dvo in(var.domain_name != "" ? aws_acm_certificate.main[0].domain_validation_options : []) :
    dvo.domain_name => {
      name   = dvo.resource_record_name
      record = dvo.resource_record_value
      type   = dvo.resource_record_type
    }
  }

  allow_overwrite = true
  name            = each.value.name
  records         = [each.value.record]
  ttl             = 60
  type            = each.value.type
  zone_id         = data.aws_route53_zone.main[0].zone_id
}

resource "aws_acm_certificate_validation" "main" {
  count                   = var.domain_name != "" ? 1 : 0
  certificate_arn         = aws_acm_certificate.main[0].arn
  validation_record_fqdns = [for record in aws_route53_record.cert_validation : record.fqdn]
}

# HTTPS Listener (443) — only if certificate is validated
resource "aws_lb_listener" "https" {
  count             = var.domain_name != "" ? 1 : 0
  load_balancer_arn = aws_lb.main.arn
  port              = "443"
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = aws_acm_certificate_validation.main[0].certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.main.arn
  }
}

# HTTP Listener — redirect to HTTPS if certificate exists, otherwise forward
resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type = var.domain_name != "" ? "redirect" : "forward"

    # Redirect config (used when domain is set)
    dynamic "redirect" {
      for_each = var.domain_name != "" ? [1] : []
      content {
        port        = "443"
        protocol    = "HTTPS"
        status_code = "HTTP_301"
      }
    }

    # Forward config (used when no domain/cert)
    target_group_arn = var.domain_name == "" ? aws_lb_target_group.main.arn : null
  }
}

# =============================================================================
# ECS Cluster
# =============================================================================

resource "aws_ecs_cluster" "main" {
  name = local.name_prefix

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = { Name = local.name_prefix }
}

# =============================================================================
# IAM Roles
# =============================================================================

resource "aws_iam_role" "ecs_task_execution" {
  name = "${local.name_prefix}-ecs-task-execution"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "secrets_manager_execution" {
  name = "${local.name_prefix}-secrets-manager-execution"
  role = aws_iam_role.ecs_task_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = aws_secretsmanager_secret.config.arn
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:DescribeKey",
        ]
        Resource = aws_kms_key.main.arn
      }
    ]
  })
}

resource "aws_iam_role" "ecs_task" {
  name = "${local.name_prefix}-ecs-task"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy" "secrets_manager_task" {
  name = "${local.name_prefix}-secrets-manager-task"
  role = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = aws_secretsmanager_secret.config.arn
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:DescribeKey",
        ]
        Resource = aws_kms_key.main.arn
      }
    ]
  })
}

# =============================================================================
# ECS Service (Private Subnets)
# =============================================================================

resource "aws_ecs_service" "main" {
  name            = "${local.name_prefix}-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.main.arn
  desired_count   = var.environment == "production" ? 2 : 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = [aws_subnet.private_1.id, aws_subnet.private_2.id]
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.main.arn
    container_name   = "trading-api"
    container_port   = 8000
  }

  depends_on = [aws_lb_listener.http]

  deployment_controller {
    type = "ECS"
  }
}

# ECS Task Definition
resource "aws_ecs_task_definition" "main" {
  family                   = "traderbot-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.environment == "production" ? "1024" : "512"
  memory                   = var.environment == "production" ? "2048" : "1024"
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name      = "trading-api"
      image     = "${local.aws_account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/traderbot:latest"
      essential = true
      portMappings = [{
        containerPort = 8000
        protocol      = "tcp"
      }]
      environment = [
        { name = "ENVIRONMENT", value = var.environment },
        { name = "AWS_SECRETS_NAME", value = "${local.name_prefix}/config" },
        { name = "AWS_REGION", value = var.aws_region },
        { name = "AWS_DEFAULT_REGION", value = var.aws_region }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = "/ecs/${local.name_prefix}"
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }
      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
        interval    = 30
        timeout     = 10
        retries     = 3
        startPeriod = 60
      }
    }
  ])
}

# =============================================================================
# ECR Repository (with scanning and lifecycle)
# =============================================================================

resource "aws_ecr_repository" "main" {
  name                 = "traderbot"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "KMS"
    kms_key         = aws_kms_key.main.arn
  }

  tags = { Name = "${local.name_prefix}-ecr" }
}

resource "aws_ecr_lifecycle_policy" "main" {
  repository = aws_ecr_repository.main.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

# =============================================================================
# Secrets Manager (Hardened with CMK and resource policy)
# =============================================================================

resource "aws_secretsmanager_secret" "config" {
  name        = "${local.name_prefix}/config"
  description = "TraderBot API configuration"
  kms_key_id  = aws_kms_key.main.arn

  recovery_window_in_days = var.environment == "production" ? 30 : 7

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowECSTaskAccess"
        Effect = "Allow"
        Principal = {
          AWS = [
            aws_iam_role.ecs_task.arn,
            aws_iam_role.ecs_task_execution.arn,
          ]
        }
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret",
        ]
        Resource = "*"
      },
      {
        Sid       = "DenyAllOthers"
        Effect    = "Deny"
        Principal = "*"
        Action = [
          "secretsmanager:GetSecretValue",
        ]
        Resource = "*"
        Condition = {
          StringNotEquals = {
            "aws:PrincipalArn" = [
              aws_iam_role.ecs_task.arn,
              aws_iam_role.ecs_task_execution.arn,
              "arn:aws:iam::${local.aws_account_id}:root",
              data.aws_caller_identity.current.arn,
            ]
          }
        }
      },
    ]
  })

  tags = { Name = "${local.name_prefix}-secrets" }
}

resource "aws_secretsmanager_secret_version" "config" {
  secret_id = aws_secretsmanager_secret.config.id

  secret_string = jsonencode({
    DATABASE_URL          = "postgresql://${aws_db_instance.main.username}:${random_password.db_password.result}@${aws_db_instance.main.endpoint}/${aws_db_instance.main.db_name}"
    REDIS_URL             = "redis://localhost:6379/0"
    POLYGON_API_KEY       = ""
    ALPHA_VANTAGE_API_KEY = ""
    MARKETAUX_API_KEY     = ""
    FINNHUB_API_KEY       = ""
    ALPACA_API_KEY        = ""
    ALPACA_SECRET_KEY     = ""
    JWT_SECRET_KEY        = random_password.jwt_secret.result
  })

  lifecycle {
    ignore_changes = [secret_string]
  }
}

resource "random_password" "jwt_secret" {
  length  = 64
  special = false
}

# =============================================================================
# CloudWatch Log Group (KMS encrypted)
# =============================================================================

resource "aws_cloudwatch_log_group" "main" {
  name              = "/ecs/${local.name_prefix}"
  retention_in_days = var.environment == "production" ? 90 : 14
  kms_key_id        = aws_kms_key.main.arn

  tags = { Name = "${local.name_prefix}-logs" }
}

# Random DB Password
resource "random_password" "db_password" {
  length  = 32
  special = false
}

# =============================================================================
# Outputs
# =============================================================================

output "alb_dns_name" {
  value = aws_lb.main.dns_name
}

output "rds_endpoint" {
  value     = aws_db_instance.main.endpoint
  sensitive = true
}

output "ecs_cluster_name" {
  value = aws_ecs_cluster.main.name
}

output "secrets_arn" {
  value = aws_secretsmanager_secret.config.arn
}

output "ecr_repository" {
  value = aws_ecr_repository.main.repository_url
}

output "kms_key_arn" {
  value = aws_kms_key.main.arn
}
