# ============================================================================
# Input Variables
# ============================================================================

variable "aws_region" {
  description = "Primary AWS region (ALB, ECS, RDS, ElastiCache all live here)."
  type        = string
  default     = "eu-west-2"
}

variable "project_name" {
  description = "Short name used as a prefix for every resource."
  type        = string
  default     = "traderbot"
}

variable "environment" {
  description = "Environment label — prod, staging, etc. Drives sizing defaults."
  type        = string
  default     = "prod"
  validation {
    condition     = contains(["prod", "staging", "dev"], var.environment)
    error_message = "environment must be one of: prod, staging, dev."
  }
}

variable "domain" {
  description = "Root domain for the application (e.g. traderbotapp.com)."
  type        = string
}

variable "github_repository" {
  description = "GitHub repo as <org>/<name>. Used by the OIDC deploy role's trust policy."
  type        = string
  default     = "asmeyatsky-personal/traderbot"
}

# ── VPC ────────────────────────────────────────────────────────────────────
variable "vpc_cidr" {
  description = "CIDR block for the VPC."
  type        = string
  default     = "10.20.0.0/16"
}

variable "azs" {
  description = "AZs used for subnet distribution. Fargate wants at least two for HA."
  type        = list(string)
  default     = ["eu-west-2a", "eu-west-2b"]
}

# ── Fargate ────────────────────────────────────────────────────────────────
variable "api_cpu" {
  description = "CPU units per Fargate task (1024 = 1 vCPU)."
  type        = number
  default     = 512
}

variable "api_memory" {
  description = "Memory (MiB) per Fargate task."
  type        = number
  default     = 1024
}

variable "api_desired_count" {
  description = "Baseline Fargate task count (autoscaling adds more as load rises)."
  type        = number
  default     = 2
}

variable "api_max_count" {
  description = "Upper bound on Fargate autoscaling."
  type        = number
  default     = 6
}

variable "api_image_tag" {
  description = "Container image tag to deploy. CI overwrites this per-commit; default is the latest tag as a break-glass."
  type        = string
  default     = "latest"
}

# ── RDS Postgres ───────────────────────────────────────────────────────────
variable "db_instance_class" {
  description = "RDS instance class. t4g.micro is the cheapest ARM-Graviton Postgres option."
  type        = string
  default     = "db.t4g.micro"
}

variable "db_allocated_storage" {
  description = "Initial RDS storage in GB (autoscales up to db_max_allocated_storage)."
  type        = number
  default     = 20
}

variable "db_max_allocated_storage" {
  description = "Ceiling for RDS storage autoscaling."
  type        = number
  default     = 100
}

variable "db_backup_retention_days" {
  description = "Automated backup retention window. Minimum for PITR."
  type        = number
  default     = 7
}

variable "db_multi_az" {
  description = "Multi-AZ for RDS. Off in prod t4g.micro saves ~$15/mo; flip on when revenue justifies the extra cost."
  type        = bool
  default     = false
}

# ── ElastiCache ────────────────────────────────────────────────────────────
variable "cache_node_type" {
  description = "ElastiCache node type. cache.t4g.micro is fine for our current key volume."
  type        = string
  default     = "cache.t4g.micro"
}

# ── Deploy knobs ───────────────────────────────────────────────────────────
variable "ecr_lifecycle_keep_images" {
  description = "Number of tagged images to retain in ECR before the lifecycle policy prunes."
  type        = number
  default     = 10
}
