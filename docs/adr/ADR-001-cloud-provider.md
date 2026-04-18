# ADR-001: Cloud Provider — Stay on AWS

- **Status:** Accepted
- **Date:** 2026-04-18
- **Deciders:** Allan Smeyatsky (solo operator)
- **Supersedes:** N/A
- **Referenced by:** `rewrite181226.md`, Phase 0

## Context

The 2026 Architectural Rules (§1) mandate GCP (Cloud Run, Secret Manager, Workload Identity) as the default cloud stack. §1 permits deviation with a one-paragraph ADR citing why the default fails.

TraderBot currently runs on AWS: single EC2 t4g.medium in eu-west-2, Docker Compose with Caddy + Postgres + KeyDB + FastAPI + React frontend. Terraform manages VPC / EC2 / Route 53 / Elastic IP. Current monthly cost ≈ $26. Domain `traderbotapp.com` is AWS-registered via Route 53. CI/CD deploys via `appleboy/ssh-action` on push to `main`.

The rules allow the deviation; the question is whether to migrate now, migrate later, or stay on AWS indefinitely.

## Decision

**Stay on AWS through launch and the first 6–12 months of production.** Revisit when monthly cost > $250, when multi-region latency becomes a real user complaint, or when team size grows beyond one operator.

## Rationale

1. **Cost.** Current footprint is $26/mo. Equivalent GCP Cloud Run + Cloud SQL db-f1-micro + Memorystore Basic + Cloud DNS would run ≈ $70–110/mo before traffic. The rules recognise cost as a legitimate reason for deviation.
2. **Operator simplicity.** Solo operator; existing mental model of EC2 + Docker Compose is load-bearing. Migrating to Cloud Run requires learning Workload Identity, Cloud Build, Cloud SQL proxy, and a different IAM model — time that is better spent on live-trading wiring (Phase 6 of the rebuild plan).
3. **No stack-specific requirement is unmet.** AWS Secrets Manager satisfies "Secret Manager + Workload Identity only" in spirit (IAM Roles for EC2 provide the Workload-Identity-equivalent). The rules list GCP as a default, not a compliance requirement.
4. **Scale profile.** Beta launches with 3 users capped at $1,000 each. Single-AZ single-instance is acceptable for this risk level; the blast radius of a hardware failure is a one-hour outage, not a regulatory incident.
5. **Migration is reversible.** Terraform + Docker Compose means the entire stack is one `terraform apply` away from running anywhere. If we hit the revisit triggers, migration is a week of work, not a rebuild.

## Consequences

**Kept:**
- AWS EC2 + Docker Compose + Caddy + Postgres + KeyDB.
- Terraform state in `s3://traderbot-terraform-state` (eu-west-2).
- GitHub Actions SSH-based deploy.

**Must compensate for (tracked in `rewrite181226.md`):**
- Phase 3: move production secrets off `.env.prod` disk file into AWS Secrets Manager; container fetches at startup. This closes the §4 "no secrets in env defaults" gap.
- Phase 7: add off-site Postgres backup (`aws s3 sync deploy/backups/ s3://traderbotapp-backups/`) since single-AZ offers no DR by default.
- Phase 7: stand up a staging environment — currently prod-only.

## Revisit Triggers

Open a new ADR (ADR-00X: cloud provider migration) if **any** fire:

- Monthly infra cost > $250.
- p95 latency > 500ms sustained for users outside EU (GCP global network advantage becomes material).
- Second operator joins (onboarding cost of EC2 + Docker Compose vs. Cloud Run flips).
- A dependency (e.g. BigQuery analytics requirement in §1) becomes launch-critical and Cloud-Run-adjacent services are materially cheaper than AWS equivalents.
- A compliance requirement mandates a provider switch.

## Related

- `Architectural Rules — 2026.md` §1, §4
- `rewrite181226.md` Phase 0, Phase 3, Phase 7
- `deploy/README.md`
