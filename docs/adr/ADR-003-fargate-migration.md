# ADR-003: Migrate prod from EC2 + Docker Compose to Fargate + managed data services

**Status:** Proposed
**Date:** 2026-04-19
**Supersedes:** Part of ADR-001 (which kept us on AWS EC2 at launch scale)
**Related:** rewrite181226.md Phase 3 (secrets), Phase 6 (live trading), Phase 7 (launch readiness)

---

## Context

The current production stack is a single EC2 t4g.medium (eu-west-2, ARM64, 4 GB RAM, 19 GB root) running Docker Compose with:

- Caddy (reverse proxy + auto-HTTPS)
- React SPA (nginx)
- FastAPI (uvicorn)
- PostgreSQL 15 (on Docker volume)
- KeyDB (on Docker volume)

ADR-001 signed off on this as the **minimum viable launch stack** at a single-operator budget (~$26/mo). As we move toward actual launch (Phase 8), the stack's limitations are now biting:

### What's broken

| Symptom | Root cause |
|---|---|
| 10+ consecutive deploy failures over the past 48 hours | Single 19 GB root volume, Docker image + build cache accumulation, and a single RUN layer that installs ~9 GB of ML wheels |
| Operators must SSH in to unwedge the host (`docker system prune -af`) | No ephemeral build environment; state and build share the same disk |
| Host-side `.env.prod` needs manual edits to flip feature flags | Config is baked into the host, not environment |
| Zero horizontal scaling | Single instance; max RPS bounded by 2 vCPU |
| Single point of failure | Host dies → site dies |
| Postgres backups are manual (`deploy.sh backup` + cron); restore is operator-driven | No managed PITR, no cross-AZ replication |
| TLS certs are local to the host (Caddy data volume) | Cert renewal and rotation depend on the host being alive |
| Deploy = SSH + `git pull` + `docker build` | Slow (~5 min), fragile (disk/network on a 2 vCPU box), and needs a persistent SSH key |

### What the 2026 Rules say

§2 of the 2026 rules accepts AWS deviation from the default GCP with a one-paragraph justification — ADR-001 did that. Nothing in the rules forces Fargate specifically, but §4 (Security) requires "proper secrets handling; no env-based secrets in prod" which the Phase 3 boot guard enforces, and §6 (Observability) asks for OTel tracing + RED metrics, both of which are dramatically easier with CloudWatch integration than with Docker logs on a single host.

### What we actually need at launch

- 99.5% availability (can tolerate a brief outage during a deploy, but not multi-hour)
- Deploy without operator intervention (a green CI = a live deploy, always)
- Managed backups for Postgres (PITR + cross-AZ)
- Scale API horizontally when beta users grow past ~10 → 100
- TLS that renews itself without touching the host
- A place to ship OTel traces and CloudWatch metrics without running Grafana/Prometheus ourselves

EC2 + Docker Compose can meet the first and the last with enough hand-waving. The middle three are structural — they need managed services.

---

## Decision

**Move to AWS Fargate for compute, with managed data services for state.**

### Components

| Concern | Current | Proposed |
|---|---|---|
| Ingress + TLS | Caddy (on host) | **ALB + ACM** |
| API runtime | `uvicorn` in Docker Compose | **Fargate service**, 2 tasks, target-tracking autoscaling on CPU |
| Frontend | nginx in Docker Compose | **S3 + CloudFront** with OAC |
| Database | Postgres in Docker volume | **RDS PostgreSQL 15** t4g.micro, encrypted, 7-day PITR |
| Cache | KeyDB in Docker volume | **ElastiCache for Valkey** t4g.micro (Redis-compatible — no code change) |
| Secrets | `.env.prod` on disk | **AWS Secrets Manager** (Phase 3 scaffolding already wired) |
| Deploy | SSH → `git pull` → `docker build` | **GitHub OIDC → ECR push → ECS update-service** |
| Image registry | Docker Hub / local builds | **ECR** with lifecycle policy (keep 10 images) |
| Logs | `docker compose logs` | **CloudWatch Logs** + OTLP → Grafana Cloud |
| DNS | Route 53 → EIP | Route 53 → ALB alias |

### Networking

- VPC with 2 public subnets + 2 private subnets across 2 AZs.
- Fargate tasks in **public subnets with public IPs** and SG-restricted inbound (from ALB only). This skips a NAT Gateway ($32/mo + per-GB egress). Trade-off explicit below.
- VPC endpoints for ECR, S3, CloudWatch Logs, Secrets Manager so task egress stays on the VPC backbone without NAT.
- RDS and ElastiCache in private subnets with SGs allowing only Fargate SG.

### Security posture

- IAM task execution role: pull from ECR, write CloudWatch logs, read Secrets Manager.
- IAM task role (separate, least-privileged): whatever the app itself needs — today that's Secrets Manager read for the specific secret ARN, and nothing else.
- GitHub OIDC identity provider + role: push to ECR, update specific ECS service only. **No long-lived AWS credentials stored in GitHub secrets** — replaces the `EC2_SSH_KEY` secret.
- Secrets Manager: one secret for app secrets (JWT, Alpaca, Anthropic, Postgres URL). Rotated by operator, tracked via audit.
- ALB: WAF managed rule group (core rule set) attached later when we open to public beta. Not needed pre-launch.
- No SSH anywhere. If an operator needs to exec into a task, **ECS Exec** (via SSM) — auditable, no key management.

### Deploy flow

```
git push main → GitHub Actions
                 │
                 ├─ Run Tests (unchanged)
                 │
                 └─ Build + Push (workflow_run on success):
                     1. assume role via OIDC
                     2. docker build -t ${ECR_URI}:${COMMIT_SHA}
                     3. aws ecr push
                     4. aws ecs register-task-definition (image URI)
                     5. aws ecs update-service --force-new-deployment
                     6. aws ecs wait services-stable  ← smoke test gate
                     7. on timeout: aws ecs update-service --task-definition $PREV
```

Rollbacks are a one-line `aws ecs update-service --task-definition <prev-arn>`. No SSH, no `git revert`.

### Cost (eu-west-2, realistic steady-state)

| Item | Spec | ~USD/mo |
|---|---|---|
| Fargate API | 2× (0.5 vCPU / 1 GB), 24×7 | 36 |
| ALB | 1 ALB, baseline LCUs | 18 |
| RDS Postgres | db.t4g.micro, 20 GB gp3, 7-day backups | 18 |
| ElastiCache Valkey | cache.t4g.micro | 12 |
| S3 + CloudFront | SPA bundle + low-traffic distribution | 2 |
| Secrets Manager | 2 secrets (app, broker) + ~1k API calls | 2 |
| Route 53 + ACM | 1 hosted zone | 1 |
| CloudWatch Logs | ~1 GB/mo ingestion + retention | 3 |
| VPC endpoints | ECR + S3 + Logs + SM interface endpoints | ~3 |
| **Total (no NAT GW)** | | **~95** |

vs. current ~$26/mo → **3.6× increase**. Adding a NAT Gateway for proper private-subnet egress adds ~$32/mo ($127/mo total). My recommendation is **skip the NAT Gateway** — Fargate tasks in public subnets with tight SG rules is an acceptable posture for a pre-launch app, and we can add NAT later if the security review for public-facing launch demands it.

---

## Consequences

### Pros

- **Reliable deploys.** No host disk to fill up. Every deploy is a fresh container on fresh compute.
- **Horizontal scaling.** Target-tracking on CPU gives us 2 → N tasks without touching config.
- **Managed Postgres backups.** Automated snapshots + PITR. Restore is a console click or one CLI call.
- **Managed TLS.** ACM handles cert provisioning and renewal. No Caddy state to lose.
- **No SSH keys.** GitHub OIDC + ECS Exec via SSM. Revoke access by removing one IAM trust entry.
- **Observability wins.** CloudWatch integrates directly with CloudTrail, X-Ray, etc. when we want them.
- **Phase 3 completes.** AWS Secrets Manager becomes the default, not an opt-out.

### Cons

- **Higher cost.** ~$95/mo vs. ~$26/mo. Real money pre-revenue.
- **More moving parts.** VPC, RDS, ElastiCache, ALB, S3, CloudFront, ECR, ECS, Secrets Manager, OIDC role. Each is a thing to understand. Terraform helps but doesn't eliminate complexity.
- **Cold-start latency on new tasks.** Fargate takes 30–60s to pull the image and pass healthchecks. Not a user-facing issue at steady state; matters for quick rollouts.
- **ElastiCache lacks KeyDB-specific features.** We're using KeyDB today but only as a Redis-compatible cache. Valkey (AWS's Redis fork) is a drop-in.
- **Region lock-in intensifies.** Cross-region migration becomes non-trivial.
- **Data migration is a cut-over.** A maintenance window with DNS flip, Postgres dump + restore, and Redis re-warm.

### Deferred decisions

- **Aurora Serverless v2** vs. RDS t4g.micro. Serverless has scale-to-zero but $43/mo idle floor; t4g.micro is cheaper at our baseline. Revisit if traffic becomes bursty.
- **MemoryDB** vs. ElastiCache. MemoryDB is durable Redis; we don't need durability in our cache today (WebSocket state, rate-limit counters, TOTP nonce blacklist). Start with ElastiCache; upgrade if we add truly durable cache state.
- **Fargate Spot**. 70% cheaper, but tasks can be stopped with 2 min notice. Fine for background workers; risky for user-facing API. Revisit when we have >2 tasks.
- **CloudFront → S3 SPA vs. Fargate nginx.** Proposing S3+CloudFront. The only thing we lose is a single-host deploy story for the frontend; gains are cache + global CDN + ~$10/mo saved.

---

## Migration plan

Five working days, broken into shippable slices. Nothing in the existing prod is touched until step 5.

### Day 1 — Terraform data plane

- Provision VPC, subnets, SGs, VPC endpoints.
- Provision RDS Postgres, ElastiCache Valkey, Secrets Manager (empty secrets).
- Apply `terraform plan` + `apply` side-by-side with existing prod (separate state key).
- Populate Secrets Manager from current `.env.prod` values.

### Day 2 — Terraform compute plane

- Provision ECR repository.
- GitHub OIDC identity provider + deploy role.
- ECS cluster, task definition, service (with 0 desired count initially so nothing runs yet).
- ALB + target group + listener (HTTP 80 for now; HTTPS waits for cert validation).
- ACM cert request + DNS validation record (in existing Route 53 zone, so validates immediately).

### Day 3 — CI/CD

- New GitHub Actions workflow: build → ECR push → ECS update-service → `wait services-stable`.
- Run against the empty service on the new infra. Verify image builds in CI, lands in ECR, task definition updates.
- Scale service to 1 desired; smoke-test via ALB DNS name directly.

### Day 4 — Frontend + DNS prep

- S3 bucket + CloudFront distribution with OAC.
- Frontend build step in CI pushes `dist/` to S3 + creates CloudFront invalidation.
- Add a second Route 53 record (`staging.traderbotapp.com`) pointing to ALB — exercise the full path without touching the live domain.

### Day 5 — Cut-over (maintenance window)

- Announce ~15 min maintenance.
- On live EC2: set `EMERGENCY_HALT=true`, `pg_dump` + S3 upload.
- Restore dump to RDS.
- Update Fargate env to use new DB URL; scale desired → 2.
- Flip Route 53 `traderbotapp.com` A record from EIP → ALB alias.
- Verify `/api/v1/healthz`, a login, a chat message, an order round-trip.
- Keep the EC2 running for 7 days as rollback.
- After 7 clean days: `terraform destroy` on the EC2 module.

---

## Rollback

If any post-cutover check fails within 15 minutes, flip the Route 53 record back to the EC2's EIP. TTL is 60s. Worst case, ~1 min of user-visible 404s.

If failure is discovered >15 min post-cutover and the EC2 is behind on writes: stop the Fargate service, `pg_dump` from RDS, restore back to EC2, flip DNS. Worst case: users see their own data roll back to the cutover snapshot — flagged to beta users via email.

---

## Open questions (for operator review)

1. **Domain**: Does `traderbotapp.com` stay, or do we consolidate with `asmeyatsky-personal` org branding?
2. **Region**: Stay in `eu-west-2`? All resources today are there. If we ever add a second region, cheaper to plan now than retrofit.
3. **Budget ceiling**: ~$95/mo acceptable? If no, I'll add a §C "lean variant" that keeps Docker Compose for Postgres + Redis on a cheaper EC2 and moves only API + frontend to Fargate (saves ~$30/mo, keeps one pet host).
4. **NAT Gateway**: Accept the "Fargate in public subnet, SG-locked" posture, or spend $32/mo for proper private-subnet isolation?
5. **When**: Block Phase 8 launch on this migration, or launch on EC2 with a plan to migrate after initial beta signal?

---

**End of proposal.** If approved, Terraform skeleton lives at `deploy/terraform-fargate/` (separate state from the existing `deploy/terraform/`). Nothing is provisioned until we `terraform apply`.
