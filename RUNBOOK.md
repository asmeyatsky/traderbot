# TraderBot Operational Runbook

Single-page reference for operators. If it's 3 AM and something is on fire, start here.

**Status at a glance:** live at https://traderbotapp.com (prod EC2 t4g.medium, eu-west-2). One host. Caddy fronts the stack. Postgres + KeyDB are internal-only.

---

## 1. Emergency halt (stop real-money trading NOW)

The platform-level kill switch bypasses every user's `trading_mode` and forces paper routing at `BrokerServiceFactory`. Nothing live goes through after the next request.

```bash
# On the EC2 host, in deploy/
echo "EMERGENCY_HALT=true" >> .env.prod
bash deploy.sh restart
```

To resume:

```bash
sed -i '/^EMERGENCY_HALT=/d' .env.prod  # or set =false
bash deploy.sh restart
```

**Verify it took effect** by tailing api logs and watching for `broker_routed_to_paper reason=emergency_halt` lines on the next order attempt.

---

## 2. Feature-flag gating of live trading

`ENABLE_LIVE_TRADING` is the second, coarser kill switch. Unset or `false` forces every user (including LIVE-mode users) to paper via the factory; no code change required.

```bash
# Flip on (paired with Phase 6 / Phase 8 gated rollout)
sed -i 's/^ENABLE_LIVE_TRADING=.*/ENABLE_LIVE_TRADING=true/' .env.prod
bash deploy.sh restart

# Flip off (safest default — no user can place a live order)
sed -i 's/^ENABLE_LIVE_TRADING=.*/ENABLE_LIVE_TRADING=false/' .env.prod
bash deploy.sh restart
```

Rule of thumb: if you're unsure, turn it off. Paper users don't notice; live users see "Live trading is currently disabled" on the enable-live-mode endpoint.

---

## 3. Service restart / redeploy

```bash
bash deploy.sh restart     # restart api/frontend/caddy only; keeps db/cache up
bash deploy.sh deploy      # full rebuild + migrations + deploy
bash deploy.sh migrate     # run alembic upgrade head only
bash deploy.sh status      # ps + disk + memory
bash deploy.sh logs        # tail compose logs (Ctrl-C to exit)
bash deploy.sh down        # stop the whole stack
```

---

## 4. Postgres backup & restore

**Manual backup** (gzip'd pg_dump into `deploy/backups/`, last 7 kept):

```bash
bash deploy.sh backup
```

**Off-site S3 sync** (runs nightly via cron on the EC2 host; see § 5):

```bash
bash deploy/backup_offsite.sh
```

**Restore from local backup** (destroys current DB — do not run casually):

```bash
cd deploy
LATEST=$(ls -t backups/traderbot_*.sql.gz | head -1)
echo "Restoring from $LATEST — this will overwrite the current database. Ctrl-C to abort."
read -r
docker compose -f docker-compose.prod.yml --env-file .env.prod exec -T db \
  psql -U trading -c "DROP DATABASE IF EXISTS traderbot;"
docker compose -f docker-compose.prod.yml --env-file .env.prod exec -T db \
  psql -U trading -c "CREATE DATABASE traderbot OWNER trading;"
gunzip -c "$LATEST" | \
  docker compose -f docker-compose.prod.yml --env-file .env.prod exec -T db \
    psql -U trading traderbot
bash deploy.sh restart
```

**Restore from S3** — pull locally first, then follow the local restore above:

```bash
aws s3 cp s3://traderbotapp-backups/traderbot_YYYYMMDD_HHMMSS.sql.gz deploy/backups/
```

RPO: 24h. RTO: ~30 min once someone is on the host.

---

## 5. Off-site backups (nightly cron)

The off-site script syncs `deploy/backups/` to `s3://traderbotapp-backups/` with 30-day object lifecycle. Cron entry on the EC2 host:

```cron
# Nightly at 03:17 UTC (after US market close + buffer)
17 3 * * * cd /home/ubuntu/traderbot/deploy && bash deploy.sh backup && bash backup_offsite.sh >> /var/log/traderbot-backup.log 2>&1
```

**Install:**

```bash
crontab -e
# paste the line above, save, exit
```

**Prereqs on the host:** `awscli` installed and an IAM role or `~/.aws/credentials` with `s3:PutObject` on the bucket. Bucket lifecycle policy handles retention.

---

## 6. JWT secret rotation

Rotating invalidates every outstanding session — users will be forced to log in again.

```bash
# On the host
python3 -c 'import secrets; print(secrets.token_urlsafe(48))'
# Copy the output, then:
sed -i "s|^JWT_SECRET_KEY=.*|JWT_SECRET_KEY=<new-value>|" .env.prod
bash deploy.sh restart
```

If you suspect the previous key is compromised, also **flush the Redis blacklist** (stale entries become harmless but unbounded):

```bash
docker compose -f docker-compose.prod.yml --env-file .env.prod exec cache keydb-cli --eval 'for _,k in ipairs(redis.call("keys","blacklist:token:*")) do redis.call("del", k) end return "ok"' 0
```

---

## 7. Circuit breakers

There are two. Know which one you're debugging.

- **Market volatility breaker** (`src/domain/services/risk_management.py:510`). Scope: autonomous trading loop. Triggers on extreme-volatility signals. Reset: 30 min. Operator action: usually nothing — self-resets.
- **Broker circuit breaker** (`src/domain/services/broker_circuit_breaker.py`). Scope: live place_order calls. Trips after 3 consecutive 5xx / auth / network failures. Cool-off: 5 min. Symptom: LIVE users see `LiveTradingHaltedError: Broker circuit breaker open …`. Usually means Alpaca is down or our keys are rotated.

Forcing a reset isn't exposed via an admin endpoint — if you need to clear it faster than 5 min, `bash deploy.sh restart` drops the in-process state.

---

## 8. Quick diagnostics

```bash
# Is anything serving traffic?
curl -sI https://traderbotapp.com/api/v1/healthz

# Are containers healthy?
docker compose -f docker-compose.prod.yml --env-file .env.prod ps

# What's the api saying right now?
docker compose -f docker-compose.prod.yml --env-file .env.prod logs --tail 100 api

# Is the DB reachable?
docker compose -f docker-compose.prod.yml --env-file .env.prod exec db psql -U trading -c 'SELECT 1;'

# Any recent audit events for a user?
docker compose -f docker-compose.prod.yml --env-file .env.prod exec db psql -U trading -d traderbot \
  -c "SELECT action, occurred_at FROM audit_events WHERE actor_user_id='<user-id>' ORDER BY occurred_at DESC LIMIT 20;"
```

---

## 9. Contacts

- **Alpaca support:** https://alpaca.markets/support (ticket) or support@alpaca.markets. Account ID lives in `.env.prod` as a comment.
- **AWS account:** see `deploy/terraform/main.tf` for region + account info. Elastic IP: 18.134.37.4.
- **Domain:** Route 53 (same AWS account). If DNS goes sideways, check the hosted zone for `traderbotapp.com`.
- **On-call:** Allan (asmeyatsky@hotmail.com). There is no on-call rotation — a major incident will wake Allan up.

---

## 10. Known issues

- **SSE streaming render** (Phase 7 — in progress): chat responses sometimes require a page refresh to appear. If a user reports "AI hangs," ask them to refresh once; the response is already saved server-side.
- **Local dev on Python 3.14**: PyTorch / TensorFlow wheels don't exist yet. Use Docker for end-to-end dev, or a 3.11 venv for backend-only work.

---

**Last updated:** 2026-04-18 (Phase 6 merged, Phase 7 in progress).
