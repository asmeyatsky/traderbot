# TraderBot — Low-Cost AWS Deployment

Single EC2 instance running the full stack via Docker Compose with Caddy for automatic HTTPS.

## Architecture

```
Internet → Caddy (:443 auto-TLS)
              ├─ /api/* , /ws  → FastAPI (:8000)
              └─ /*            → Nginx SPA (:80)
           PostgreSQL 15 (internal network)
           KeyDB (internal network)
```

## Cost Estimate

| Resource | Monthly Cost |
|----------|-------------|
| EC2 t4g.medium (2 vCPU, 4 GB) | ~$24/mo |
| EBS gp3 20 GB | ~$1.60/mo |
| Route 53 hosted zone | ~$0.50/mo |
| Route 53 queries | ~$0.10/mo |
| S3 state bucket | ~$0.01/mo |
| Elastic IP (while attached) | $0 |
| **Total** | **~$26.21/mo** |

## Prerequisites

- AWS account with credentials configured (`aws configure`)
- Terraform >= 1.5 installed
- An S3 bucket named `traderbot-terraform-state` for Terraform state
- A domain registered (e.g. `traderbotapp.com`)

## Infrastructure Setup (Terraform)

```bash
cd deploy/terraform

# Configure variables
cp terraform.tfvars.example terraform.tfvars
nano terraform.tfvars   # fill in ssh_public_key, domain, etc.

# Provision infrastructure
terraform init
terraform plan          # review changes
terraform apply         # creates VPC, EC2, Elastic IP, Route 53
```

After `terraform apply`:

1. Copy the **nameservers** from the output → update your domain registrar's NS records
2. Wait for DNS propagation (can take up to 48 hours, usually minutes)

## First-Time Deployment

```bash
# 1. SSH into the instance (command shown in terraform output)
ssh ubuntu@<elastic-ip>

# 2. Check bootstrap completed (user_data runs on first boot)
tail -f /var/log/user-data.log

# 3. Configure environment
cp /opt/traderbot/deploy/.env.prod.example /opt/traderbot/deploy/.env.prod
nano /opt/traderbot/deploy/.env.prod   # fill in all CHANGE_ME values

# 4. Deploy
cd /opt/traderbot/deploy
bash deploy.sh

# 5. Verify
curl https://traderbotapp.com/health
```

Caddy will automatically obtain a Let's Encrypt TLS certificate for your domain.

## CI/CD (GitHub Actions)

Pushes to `main` automatically deploy after tests pass.

### Setup

Add these secrets to your GitHub repository (`Settings → Secrets → Actions`):

| Secret | Value |
|--------|-------|
| `EC2_HOST` | Elastic IP address (from `terraform output elastic_ip`) |
| `EC2_SSH_KEY` | Private SSH key for the `ubuntu` user |

### How It Works

```
Push to main → Run Tests (test.yml) → Deploy (deploy.yml) → Health Check
```

- `deploy.yml` triggers via `workflow_run` after "Run Tests" succeeds
- SSHes into EC2 and runs `deploy.sh` (git pull, rebuild, migrate)
- Runs a health check against `https://<EC2_HOST>/health`
- Concurrency group prevents parallel deploys

## Day-to-Day Operations

```bash
cd /opt/traderbot/deploy

bash deploy.sh              # Pull latest code, rebuild, migrate, restart
bash deploy.sh logs         # Tail container logs
bash deploy.sh status       # Show container status + disk/memory usage
bash deploy.sh restart      # Restart API + frontend + Caddy
bash deploy.sh backup       # Dump PostgreSQL to deploy/backups/ (keeps last 7)
bash deploy.sh migrate      # Run Alembic migrations only
bash deploy.sh down         # Stop everything
```

## Backup Strategy

- `bash deploy.sh backup` creates a gzipped SQL dump in `deploy/backups/`
- Only the 7 most recent backups are kept (auto-pruned)
- For offsite backups, consider syncing `deploy/backups/` to S3:
  ```bash
  aws s3 sync deploy/backups/ s3://your-bucket/traderbot-backups/
  ```

## Scaling Up

When you outgrow a single instance:

1. **Vertical**: upgrade to t4g.medium (4 GB RAM, ~$24/mo)
2. **Database out**: migrate PostgreSQL to RDS ($15-30/mo) to free RAM
3. **CDN**: put CloudFront in front of Caddy for global static asset caching (~$1/mo)
4. **Full split**: move to ECS Fargate when you need horizontal scaling (~$50+/mo)
