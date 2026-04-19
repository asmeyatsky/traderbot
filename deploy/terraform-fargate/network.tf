# ============================================================================
# VPC + Subnets + Security Groups + VPC Endpoints
#
# Architectural Intent (ADR-003):
# - Two AZs, two public subnets, two private subnets.
# - Fargate runs in public subnets with assign_public_ip=true so we skip
#   the NAT Gateway ($32/mo + per-GB egress). Security comes from tight
#   security-group rules, not network isolation.
# - RDS and ElastiCache live in the private subnets and only accept
#   traffic from the Fargate security group.
# - VPC endpoints (gateway for S3, interface for ECR/Logs/SecretsManager)
#   keep container pulls and log writes off the public internet.
# ============================================================================

# ── VPC ────────────────────────────────────────────────────────────────────
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = { Name = "${local.name_prefix}-vpc" }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "${local.name_prefix}-igw" }
}

# ── Subnets ────────────────────────────────────────────────────────────────
resource "aws_subnet" "public" {
  for_each = { for idx, az in var.azs : idx => az }

  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, each.key)
  availability_zone       = each.value
  map_public_ip_on_launch = true

  tags = {
    Name = "${local.name_prefix}-public-${each.value}"
    Tier = "public"
  }
}

resource "aws_subnet" "private" {
  for_each = { for idx, az in var.azs : idx => az }

  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, each.key + 10)
  availability_zone = each.value

  tags = {
    Name = "${local.name_prefix}-private-${each.value}"
    Tier = "private"
  }
}

# ── Routing ────────────────────────────────────────────────────────────────
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "${local.name_prefix}-public-rt" }
}

resource "aws_route" "public_default" {
  route_table_id         = aws_route_table.public.id
  destination_cidr_block = "0.0.0.0/0"
  gateway_id             = aws_internet_gateway.main.id
}

resource "aws_route_table_association" "public" {
  for_each       = aws_subnet.public
  subnet_id      = each.value.id
  route_table_id = aws_route_table.public.id
}

# Private route table — no default route. Egress only via VPC endpoints.
# If we later add a NAT Gateway, add an aws_route pointing 0.0.0.0/0 at it.
resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "${local.name_prefix}-private-rt" }
}

resource "aws_route_table_association" "private" {
  for_each       = aws_subnet.private
  subnet_id      = each.value.id
  route_table_id = aws_route_table.private.id
}

# ── Security Groups ────────────────────────────────────────────────────────

# ALB: 80/443 open to the world.
resource "aws_security_group" "alb" {
  name        = "${local.name_prefix}-alb-sg"
  description = "Inbound 80/443 from internet to ALB"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${local.name_prefix}-alb-sg" }
}

# Fargate tasks: only accept inbound from the ALB SG on the app port.
resource "aws_security_group" "fargate" {
  name        = "${local.name_prefix}-fargate-sg"
  description = "Fargate tasks — inbound from ALB only"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "App port from ALB"
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    description = "All outbound (ECR, SecretsManager, DB, cache, Anthropic, Alpaca)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${local.name_prefix}-fargate-sg" }
}

# RDS: only accept from Fargate SG.
resource "aws_security_group" "rds" {
  name        = "${local.name_prefix}-rds-sg"
  description = "Postgres — inbound from Fargate only"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "Postgres"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.fargate.id]
  }

  tags = { Name = "${local.name_prefix}-rds-sg" }
}

# ElastiCache: only accept from Fargate SG.
resource "aws_security_group" "cache" {
  name        = "${local.name_prefix}-cache-sg"
  description = "Valkey — inbound from Fargate only"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "Redis protocol"
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.fargate.id]
  }

  tags = { Name = "${local.name_prefix}-cache-sg" }
}

# Interface-endpoint SG: Fargate → endpoints on 443.
resource "aws_security_group" "vpc_endpoints" {
  name        = "${local.name_prefix}-vpce-sg"
  description = "VPC interface endpoints (443 from Fargate)"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "HTTPS from Fargate"
    from_port       = 443
    to_port         = 443
    protocol        = "tcp"
    security_groups = [aws_security_group.fargate.id]
  }

  tags = { Name = "${local.name_prefix}-vpce-sg" }
}

# ── VPC Endpoints ──────────────────────────────────────────────────────────
# Gateway endpoint for S3 — free, attaches to route tables.
resource "aws_vpc_endpoint" "s3" {
  vpc_id            = aws_vpc.main.id
  service_name      = "com.amazonaws.${var.aws_region}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = [aws_route_table.public.id, aws_route_table.private.id]
  tags              = { Name = "${local.name_prefix}-vpce-s3" }
}

# Interface endpoints — $0.01/hr each + per-GB. Critical ones for Fargate:
#   ECR API + ECR DKR (image pulls)
#   CloudWatch Logs (task log writes)
#   Secrets Manager (boot-time secret fetch)
locals {
  interface_endpoint_services = toset([
    "ecr.api",
    "ecr.dkr",
    "logs",
    "secretsmanager",
  ])
}

resource "aws_vpc_endpoint" "interface" {
  for_each = local.interface_endpoint_services

  vpc_id              = aws_vpc.main.id
  service_name        = "com.amazonaws.${var.aws_region}.${each.key}"
  vpc_endpoint_type   = "Interface"
  subnet_ids          = [for s in aws_subnet.public : s.id]
  security_group_ids  = [aws_security_group.vpc_endpoints.id]
  private_dns_enabled = true

  tags = { Name = "${local.name_prefix}-vpce-${replace(each.key, ".", "-")}" }
}

# ── Subnet group helpers ───────────────────────────────────────────────────
resource "aws_db_subnet_group" "rds" {
  name       = "${local.name_prefix}-rds"
  subnet_ids = [for s in aws_subnet.private : s.id]
  tags       = { Name = "${local.name_prefix}-rds-subnets" }
}

resource "aws_elasticache_subnet_group" "cache" {
  name       = "${local.name_prefix}-cache"
  subnet_ids = [for s in aws_subnet.private : s.id]
}
