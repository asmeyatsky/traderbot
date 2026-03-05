# ============================================================================
# Input Variables
# ============================================================================

variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "domain" {
  description = "Root domain for the application (e.g. traderbotapp.com)"
  type        = string
}

variable "ssh_public_key" {
  description = "SSH public key for EC2 access (contents of ~/.ssh/id_ed25519.pub)"
  type        = string
}

variable "ssh_allowed_cidr" {
  description = "CIDR block allowed to SSH into the instance (e.g. your IP/32)"
  type        = string
  default     = "0.0.0.0/0"
}

variable "instance_type" {
  description = "EC2 instance type (ARM64 Graviton)"
  type        = string
  default     = "t4g.medium"
}

variable "volume_size" {
  description = "Root EBS volume size in GB"
  type        = number
  default     = 20
}

variable "github_repo" {
  description = "GitHub repository URL for cloning on first boot"
  type        = string
  default     = "https://github.com/asmeyatsky/traderbot.git"
}
