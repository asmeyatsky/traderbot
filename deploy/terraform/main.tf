# ============================================================================
# TraderBot — Terraform Root Module
#
# Provisions a single EC2 instance with Elastic IP and Route 53 DNS for
# running the full stack via Docker Compose + Caddy auto-HTTPS.
#
# Usage:
#   cp terraform.tfvars.example terraform.tfvars  # edit values
#   terraform init
#   terraform plan
#   terraform apply
# ============================================================================

terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket = "traderbot-terraform-state"
    key    = "traderbot-ec2/terraform.tfstate"
    region = "eu-west-2"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project   = "traderbot"
      ManagedBy = "terraform"
    }
  }
}
