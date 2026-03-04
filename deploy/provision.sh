#!/usr/bin/env bash
# ============================================================================
# TraderBot — Full Provisioning Script
#
# Runs Terraform, waits for EC2 bootstrap, prompts for .env.prod,
# deploys the stack, and optionally configures GitHub Actions secrets.
#
# Usage:
#   cd deploy && bash provision.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TF_DIR="${SCRIPT_DIR}/terraform"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}==> $1${NC}"; }
ok()    { echo -e "${GREEN}==> $1${NC}"; }
warn()  { echo -e "${YELLOW}==> $1${NC}"; }
fail()  { echo -e "${RED}==> ERROR: $1${NC}"; exit 1; }

# ── Step 0: Prerequisites ──────────────────────────────────────────────────

info "Checking prerequisites..."

command -v terraform >/dev/null || fail "terraform not found. Install: https://developer.hashicorp.com/terraform/install"
command -v aws >/dev/null       || fail "aws CLI not found. Install: https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html"
command -v ssh >/dev/null       || fail "ssh not found"

aws sts get-caller-identity >/dev/null 2>&1 || fail "AWS credentials not configured. Run: aws configure"

if [ ! -f "$SSH_KEY" ]; then
  warn "SSH key not found at $SSH_KEY"
  echo "  Generate one with: ssh-keygen -t ed25519"
  echo "  Or set SSH_KEY=/path/to/your/key before running this script"
  exit 1
fi

ok "Prerequisites OK (terraform, aws, ssh, key at $SSH_KEY)"

# ── Step 1: Terraform ──────────────────────────────────────────────────────

cd "$TF_DIR"

if [ ! -f terraform.tfvars ]; then
  info "Creating terraform.tfvars..."
  SSH_PUB_KEY=$(cat "${SSH_KEY}.pub")

  read -rp "Domain [traderbotapp.com]: " DOMAIN
  DOMAIN="${DOMAIN:-traderbotapp.com}"

  read -rp "AWS region [us-east-1]: " REGION
  REGION="${REGION:-us-east-1}"

  read -rp "Restrict SSH to your IP? (y/n) [y]: " RESTRICT_SSH
  RESTRICT_SSH="${RESTRICT_SSH:-y}"
  if [[ "$RESTRICT_SSH" =~ ^[Yy] ]]; then
    MY_IP=$(curl -s https://checkip.amazonaws.com)
    SSH_CIDR="${MY_IP}/32"
    info "SSH will be restricted to $SSH_CIDR"
  else
    SSH_CIDR="0.0.0.0/0"
  fi

  cat > terraform.tfvars <<EOF
aws_region       = "${REGION}"
domain           = "${DOMAIN}"
github_repo      = "https://github.com/allansmeyatsky/traderbot.git"
ssh_public_key   = "${SSH_PUB_KEY}"
ssh_allowed_cidr = "${SSH_CIDR}"
instance_type    = "t4g.small"
volume_size      = 20
EOF

  ok "terraform.tfvars created"
else
  ok "terraform.tfvars already exists"
  DOMAIN=$(grep '^domain' terraform.tfvars | sed 's/.*= *"\(.*\)"/\1/')
fi

info "Running terraform init..."
terraform init

info "Running terraform plan..."
terraform plan -out=tfplan

echo ""
read -rp "Apply this plan? (y/n) [y]: " APPLY
APPLY="${APPLY:-y}"
[[ "$APPLY" =~ ^[Yy] ]] || { warn "Aborted."; exit 0; }

info "Running terraform apply..."
terraform apply tfplan
rm -f tfplan

ELASTIC_IP=$(terraform output -raw elastic_ip)
NAMESERVERS=$(terraform output -json nameservers | tr -d '[]"' | tr ',' '\n' | sed 's/^ */  /')

ok "Infrastructure provisioned!"
echo ""
echo -e "${CYAN}Elastic IP:${NC}   $ELASTIC_IP"
echo -e "${CYAN}Nameservers:${NC}"
echo "$NAMESERVERS"
echo ""
warn "ACTION REQUIRED: Update your domain registrar's NS records with the nameservers above."
read -rp "Press Enter once NS records are updated (or skip for now)..."

# ── Step 2: Wait for EC2 to accept SSH ─────────────────────────────────────

info "Waiting for EC2 instance to accept SSH connections..."

for i in $(seq 1 30); do
  if ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 -i "$SSH_KEY" ubuntu@"$ELASTIC_IP" "echo ok" >/dev/null 2>&1; then
    ok "SSH connection established"
    break
  fi
  if [ "$i" -eq 30 ]; then
    fail "Timed out waiting for SSH (5 minutes). Check the instance in AWS console."
  fi
  echo "  Attempt $i/30 — retrying in 10s..."
  sleep 10
done

# ── Step 3: Wait for bootstrap to finish ───────────────────────────────────

info "Waiting for user_data bootstrap to complete..."

for i in $(seq 1 30); do
  if ssh -i "$SSH_KEY" ubuntu@"$ELASTIC_IP" "grep -q 'Bootstrap complete' /var/log/user-data.log 2>/dev/null"; then
    ok "Bootstrap complete"
    break
  fi
  if [ "$i" -eq 30 ]; then
    fail "Bootstrap timed out (5 minutes). SSH in and check: tail /var/log/user-data.log"
  fi
  echo "  Attempt $i/30 — retrying in 10s..."
  sleep 10
done

# ── Step 4: Create .env.prod on EC2 ────────────────────────────────────────

info "Checking for .env.prod on EC2..."

HAS_ENV=$(ssh -i "$SSH_KEY" ubuntu@"$ELASTIC_IP" "test -f /opt/traderbot/deploy/.env.prod && echo yes || echo no")

if [ "$HAS_ENV" = "no" ]; then
  warn ".env.prod not found on EC2. Creating from template..."

  read -rsp "POSTGRES_PASSWORD (or Enter to auto-generate): " PG_PASS
  echo ""
  if [ -z "$PG_PASS" ]; then
    PG_PASS=$(openssl rand -base64 24)
    info "Generated POSTGRES_PASSWORD"
  fi

  read -rsp "JWT_SECRET_KEY (or Enter to auto-generate): " JWT_SECRET
  echo ""
  if [ -z "$JWT_SECRET" ]; then
    JWT_SECRET=$(openssl rand -hex 32)
    info "Generated JWT_SECRET_KEY"
  fi

  read -rp "DOMAIN [$DOMAIN]: " ENV_DOMAIN
  ENV_DOMAIN="${ENV_DOMAIN:-${DOMAIN:-traderbotapp.com}}"

  read -rsp "ANTHROPIC_API_KEY (or Enter to skip): " ANTHROPIC_KEY
  echo ""

  read -rp "POLYGON_API_KEY (or Enter to skip): " POLYGON_KEY
  read -rp "ALPHA_VANTAGE_API_KEY (or Enter to skip): " AV_KEY
  read -rp "FINNHUB_API_KEY (or Enter to skip): " FINNHUB_KEY
  read -rp "ALPACA_API_KEY (or Enter to skip): " ALPACA_KEY
  read -rsp "ALPACA_SECRET_KEY (or Enter to skip): " ALPACA_SECRET
  echo ""

  ssh -i "$SSH_KEY" ubuntu@"$ELASTIC_IP" "cat > /opt/traderbot/deploy/.env.prod" <<EOF
DOMAIN=${ENV_DOMAIN}
POSTGRES_USER=trading
POSTGRES_PASSWORD=${PG_PASS}
POSTGRES_DB=traderbot
JWT_SECRET_KEY=${JWT_SECRET}
ANTHROPIC_API_KEY=${ANTHROPIC_KEY}
CHAT_MODEL=claude-sonnet-4-20250514
POLYGON_API_KEY=${POLYGON_KEY}
ALPHA_VANTAGE_API_KEY=${AV_KEY}
FINNHUB_API_KEY=${FINNHUB_KEY}
ALPACA_API_KEY=${ALPACA_KEY}
ALPACA_SECRET_KEY=${ALPACA_SECRET}
EOF

  ok ".env.prod created on EC2"
else
  ok ".env.prod already exists on EC2"
fi

# ── Step 5: Deploy ─────────────────────────────────────────────────────────

info "Running deploy.sh on EC2..."

ssh -i "$SSH_KEY" ubuntu@"$ELASTIC_IP" "cd /opt/traderbot/deploy && bash deploy.sh"

ok "Deployment complete!"

# ── Step 6: Health check ───────────────────────────────────────────────────

info "Running health check..."
sleep 10

if curl -sf --max-time 10 "https://${DOMAIN:-traderbotapp.com}/health" >/dev/null 2>&1; then
  ok "Health check passed! Site is live at https://${DOMAIN:-traderbotapp.com}"
elif curl -sf --max-time 10 "http://${ELASTIC_IP}/ready" >/dev/null 2>&1; then
  ok "API is responding (HTTPS may still be provisioning — Caddy needs DNS to resolve)"
else
  warn "Health check didn't pass yet. This is normal if DNS hasn't propagated."
  echo "  Try manually: curl https://${DOMAIN:-traderbotapp.com}/health"
fi

# ── Step 7: GitHub Actions secrets ─────────────────────────────────────────

echo ""
if command -v gh >/dev/null 2>&1; then
  read -rp "Set up GitHub Actions secrets for auto-deploy? (y/n) [y]: " SETUP_GH
  SETUP_GH="${SETUP_GH:-y}"
  if [[ "$SETUP_GH" =~ ^[Yy] ]]; then
    info "Setting EC2_HOST secret..."
    echo "$ELASTIC_IP" | gh secret set EC2_HOST

    info "Setting EC2_SSH_KEY secret..."
    gh secret set EC2_SSH_KEY < "$SSH_KEY"

    ok "GitHub Actions secrets configured. Future pushes to main will auto-deploy."
  fi
else
  warn "gh CLI not installed — set GitHub secrets manually:"
  echo "  EC2_HOST = $ELASTIC_IP"
  echo "  EC2_SSH_KEY = contents of $SSH_KEY"
fi

echo ""
ok "All done! Your TraderBot is live."
echo ""
echo "  Site:  https://${DOMAIN:-traderbotapp.com}"
echo "  SSH:   ssh -i $SSH_KEY ubuntu@$ELASTIC_IP"
echo "  Logs:  ssh -i $SSH_KEY ubuntu@$ELASTIC_IP 'cd /opt/traderbot/deploy && bash deploy.sh logs'"
