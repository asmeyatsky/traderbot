#!/usr/bin/env bash
# ============================================================================
# EC2 Instance Bootstrap — Ubuntu 24.04 on t4g.small (ARM64)
#
# Run once after launching the instance:
#   ssh ubuntu@<ip> 'bash -s' < setup.sh
#
# Prerequisites:
#   - EC2 t4g.small with Ubuntu 24.04 ARM64 AMI
#   - Security group allows inbound 80, 443 (and 22 for SSH)
#   - Elastic IP attached (needed for DNS A record)
# ============================================================================
set -euo pipefail

echo "==> Updating system packages"
sudo apt-get update && sudo apt-get upgrade -y

echo "==> Installing Docker"
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "==> Adding ubuntu user to docker group"
sudo usermod -aG docker ubuntu

echo "==> Enabling swap (1 GB) — important for 2 GB RAM instance"
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

echo "==> Setting up automatic security updates"
sudo apt-get install -y unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades

echo "==> Creating app directory"
sudo mkdir -p /opt/traderbot
sudo chown ubuntu:ubuntu /opt/traderbot

echo "==> Setup complete. Next steps:"
echo "  1. Log out and back in (for docker group to take effect)"
echo "  2. Clone repo:  cd /opt/traderbot && git clone <repo-url> ."
echo "  3. Copy env:    cp deploy/.env.prod.example deploy/.env.prod"
echo "  4. Edit env:    nano deploy/.env.prod"
echo "  5. Deploy:      cd deploy && bash deploy.sh"
