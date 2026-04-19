# TraderBot — Fargate Infrastructure (terraform-fargate)

Terraform module for the Fargate-based production stack. See
[`docs/adr/ADR-003-fargate-migration.md`](../../docs/adr/ADR-003-fargate-migration.md)
for the full rationale.

Lives alongside `deploy/terraform/` (the legacy EC2 module). They share
the same S3 backend bucket but use distinct keys, so one `terraform apply`
never touches the other.

---

## Prerequisites

1. **AWS credentials** — `AWS_PROFILE` or the default chain resolves to an
   identity with admin (or at least VPC/IAM/ECS/RDS/S3/CloudFront/Route 53).
2. **Terraform state backend** already exists from the EC2 module:
   - S3 bucket: `traderbot-terraform-state`
   - DynamoDB lock table: `traderbot-terraform-locks`
   If they don't, `bash deploy/provision.sh` in the legacy module creates them.
3. **Route 53 hosted zone** for `traderbotapp.com` exists in the same account.
4. **GitHub repository** set to `asmeyatsky-personal/traderbot` or overridden
   via `github_repository` variable — controls the OIDC trust policy.

---

## First-time apply

```bash
cd deploy/terraform-fargate/
cp terraform.tfvars.example terraform.tfvars
# edit terraform.tfvars — at minimum, set the domain

terraform init
terraform plan -out=tfplan
# review carefully
terraform apply tfplan
```

First apply takes ~20 minutes (RDS is slow to provision; CloudFront + ACM
validation need a DNS round-trip).

---

## After apply: populate the secrets

Terraform creates empty Secrets Manager secrets. You must populate them
once before the first deploy can boot:

```bash
APP_ARN=$(terraform output -raw app_secret_arn)
BROKER_ARN=$(terraform output -raw broker_secret_arn)

# DATABASE_URL and REDIS_URL are already set by Terraform — don't overwrite.
# Merge in the rest:
aws secretsmanager get-secret-value --secret-id "$APP_ARN" --query SecretString --output text > /tmp/app.json
jq --arg jwt "$(openssl rand -hex 32)" \
   --arg ant "$ANTHROPIC_API_KEY" \
   --arg poly "$POLYGON_API_KEY" \
   '.JWT_SECRET_KEY=$jwt | .ANTHROPIC_API_KEY=$ant | .POLYGON_API_KEY=$poly' \
   /tmp/app.json > /tmp/app.new.json
aws secretsmanager put-secret-value --secret-id "$APP_ARN" --secret-string file:///tmp/app.new.json

aws secretsmanager put-secret-value --secret-id "$BROKER_ARN" \
  --secret-string "{\"ALPACA_API_KEY\":\"$ALPACA_API_KEY\",\"ALPACA_SECRET_KEY\":\"$ALPACA_SECRET_KEY\"}"
```

After this, the next Fargate task-start will resolve all env vars from
Secrets Manager.

---

## First-time CI wiring

Take the GitHub deploy role ARN from the output and paste it into the
GitHub workflow (`.github/workflows/deploy-fargate.yml` — skeleton in
progress). No long-lived AWS credentials go into GitHub secrets — OIDC
handles auth.

---

## Cost ceiling

~$95/mo steady-state (ADR-003 § Cost). To stay inside that:

- Keep `api_desired_count=2`, `api_max_count=6`.
- Keep `db_multi_az=false` until revenue justifies doubling the RDS bill.
- Don't add a NAT Gateway unless a security review demands it.

---

## Destroying

Protected resources require explicit unprotect before destroy:

1. In the AWS console, disable **deletion protection** on the RDS instance
   and ALB.
2. `terraform destroy`.

Do NOT destroy while users still depend on the stack. The legacy EC2 module
is the rollback target until this one has been live for ~7 days without
incident.
