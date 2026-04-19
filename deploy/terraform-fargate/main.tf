# ============================================================================
# TraderBot — Fargate Terraform Root Module
#
# Lives alongside deploy/terraform/ (the EC2 module). They share an S3
# backend bucket but use distinct keys so `terraform apply` in one never
# touches the other. Migrate by building this up, validating, cutting DNS
# over, then destroying the EC2 module.
#
# Usage:
#   cp terraform.tfvars.example terraform.tfvars   # fill in real values
#   terraform init
#   terraform plan                                  # review before apply
#   terraform apply
#
# See docs/adr/ADR-003-fargate-migration.md for the full rationale.
# ============================================================================

terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Same bucket as deploy/terraform/ but a distinct key so the two modules
  # never clobber each other. Lock table gives us plan/apply serialization.
  backend "s3" {
    bucket         = "traderbot-terraform-state"
    key            = "traderbot-fargate/terraform.tfstate"
    region         = "eu-west-2"
    encrypt        = true
    dynamodb_table = "traderbot-terraform-locks"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "traderbot"
      ManagedBy   = "terraform"
      Environment = var.environment
      Module      = "fargate"
    }
  }
}

# us-east-1 provider alias is required for ACM certs fronting CloudFront;
# regional certs don't work there. Only used for the SPA CloudFront cert.
provider "aws" {
  alias  = "us_east_1"
  region = "us-east-1"

  default_tags {
    tags = {
      Project     = "traderbot"
      ManagedBy   = "terraform"
      Environment = var.environment
      Module      = "fargate"
    }
  }
}

locals {
  name_prefix = "${var.project_name}-${var.environment}"

  common_tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}
