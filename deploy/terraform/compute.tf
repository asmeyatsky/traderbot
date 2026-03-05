# ============================================================================
# Compute — EC2 Instance, Security Group, Key Pair, Elastic IP
#
# Ubuntu 24.04 ARM64 on t4g.medium (2 vCPU, 4 GB RAM).
# Termination protection enabled to prevent accidental destroy.
# AMI and user_data changes are ignored to avoid instance replacement.
# ============================================================================

# ── AMI lookup: latest Ubuntu 24.04 ARM64 ──────────────────────────────────

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-arm64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  filter {
    name   = "architecture"
    values = ["arm64"]
  }
}

# ── SSH key pair ────────────────────────────────────────────────────────────

resource "aws_key_pair" "deployer" {
  key_name   = "traderbot-deployer"
  public_key = var.ssh_public_key

  tags = { Name = "traderbot-deployer" }
}

# ── Security group ─────────────────────────────────────────────────────────

resource "aws_security_group" "instance" {
  name        = "traderbot-instance"
  description = "TraderBot EC2 - SSH, HTTP, HTTPS"
  vpc_id      = aws_vpc.main.id

  # SSH
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_allowed_cidr]
  }

  # HTTP (Caddy redirects to HTTPS)
  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # HTTPS (TCP)
  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # HTTPS (UDP — HTTP/3 / QUIC)
  ingress {
    description = "HTTP/3"
    from_port   = 443
    to_port     = 443
    protocol    = "udp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # All outbound
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "traderbot-instance" }
}

# ── EC2 instance ────────────────────────────────────────────────────────────

resource "aws_instance" "app" {
  ami                     = data.aws_ami.ubuntu.id
  instance_type           = var.instance_type
  key_name                = aws_key_pair.deployer.key_name
  subnet_id               = aws_subnet.public.id
  vpc_security_group_ids  = [aws_security_group.instance.id]
  disable_api_termination = true

  root_block_device {
    volume_size           = var.volume_size
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true
  }

  user_data = templatefile("${path.module}/user_data.sh.tpl", {
    github_repo = var.github_repo
  })

  # Don't replace instance when AMI updates or user_data changes
  lifecycle {
    ignore_changes = [ami, user_data]
  }

  tags = { Name = "traderbot-app" }
}

# ── Elastic IP ──────────────────────────────────────────────────────────────

resource "aws_eip" "app" {
  instance = aws_instance.app.id
  domain   = "vpc"

  tags = { Name = "traderbot-eip" }
}
