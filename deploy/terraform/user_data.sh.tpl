#!/usr/bin/env bash
# ============================================================================
# EC2 User Data — Ubuntu 24.04 ARM64 Bootstrap
#
# Runs automatically on first boot. Logs to /var/log/user-data.log.
# Installs Docker, creates swap, clones repo, and prepares for deployment.
# ============================================================================
set -euo pipefail
exec > >(tee /var/log/user-data.log) 2>&1

echo "==> [$(date)] Starting bootstrap"

echo "==> Updating system packages"
apt-get update && apt-get upgrade -y

echo "==> Installing Docker"
apt-get install -y ca-certificates curl gnupg
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "==> Adding ubuntu user to docker group"
usermod -aG docker ubuntu

echo "==> Enabling swap (1 GB)"
fallocate -l 1G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab

echo "==> Setting up automatic security updates"
export DEBIAN_FRONTEND=noninteractive
apt-get install -y unattended-upgrades
echo 'Unattended-Upgrade::Automatic-Reboot "false";' > /etc/apt/apt.conf.d/51auto-upgrades
echo 'APT::Periodic::Update-Package-Lists "1";' >> /etc/apt/apt.conf.d/51auto-upgrades
echo 'APT::Periodic::Unattended-Upgrade "1";' >> /etc/apt/apt.conf.d/51auto-upgrades

echo "==> Cloning repository"
mkdir -p /opt/traderbot
chown ubuntu:ubuntu /opt/traderbot
su - ubuntu -c "git clone ${github_repo} /opt/traderbot"

echo "==> [$(date)] Bootstrap complete"
echo ""
echo "Next steps:"
echo "  1. SSH in:         ssh ubuntu@<elastic-ip>"
echo "  2. Configure env:  cp /opt/traderbot/deploy/.env.prod.example /opt/traderbot/deploy/.env.prod"
echo "  3. Edit env:       nano /opt/traderbot/deploy/.env.prod"
echo "  4. Deploy:         cd /opt/traderbot/deploy && bash deploy.sh"
