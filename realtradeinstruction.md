# Real-Trading Enablement Runbook

**Status:** Draft — written before the first live trade.
**Audience:** The operator who will press the button.
**Related:** [`docs/adr/ADR-002-live-trading-gate.md`](docs/adr/ADR-002-live-trading-gate.md), [`rewrite181226.md`](rewrite181226.md) Phases 6–8, [`RUNBOOK.md`](RUNBOOK.md).

This document walks through **every step** required to take TraderBot from its current all-paper state to accepting real-money orders, for one user at a time. Every step is intentionally narrow: the time it takes to read an extra line of this doc is cheaper than a single bad order.

**Rule of thumb:** if you're unsure about any step, stop. The kill-switch is one `.env.prod` edit and a restart away. There is no urgency that justifies skipping verification.

---

## Part 0 — What "live" actually means in this codebase

The path a real-money order takes:

1. User submits order via chat or `/api/v1/orders/create`.
2. `CreateOrderUseCase.execute` loads user, runs deterministic validation, runs discipline veto (Phase 10.1).
3. Checks `user.trading_mode`. If `PAPER`: in-memory auto-fill, done.
4. If `LIVE`: calls `self.broker_routing.for_user(user)` — the `BrokerServiceFactory`.
5. `BrokerServiceFactory.for_user` checks in order:
   - `EMERGENCY_HALT=true` env → returns paper broker (force-downgrade, log only).
   - `ENABLE_LIVE_TRADING != 'true'` env → returns paper broker.
   - `user.trading_mode == PAPER` → paper (defensive re-check).
   - Daily-loss cap breached → raises `DailyLossCapBreachedError`.
   - Broker circuit breaker open → raises `LiveTradingHaltedError`.
   - Otherwise → returns the **live** Alpaca broker.
6. Pre-trade balance check hits Alpaca live account. Refuses if insufficient buying power.
7. Order is sent to Alpaca live. Audit event `LiveOrderPlaced` is emitted.

Translation: **three independent gates** must be on for any real order to fire — the platform flag, the user's mode, and the safety caps. Flipping one gate is not enough. This is intentional.

---

## Part 1 — Pre-flight (days before, not hours)

Do not attempt live trading until every box below is checked. No exceptions.

### 1.1 Legal and disclosure

- [ ] Terms of service published at `traderbotapp.com/terms` (and linked from footer + onboarding).
- [ ] Risk disclosure published — must include "you can lose more than you invest" where relevant, "not FDIC insured", and "past performance does not guarantee future results". Legal-reviewed if budget allows; self-drafted with reference to Alpaca's broker-dealer disclosures if not.
- [ ] Privacy policy updated to mention real-money trading data retention (7 years for audit events per general broker-dealer standards; confirm with Alpaca what they store on their side).
- [ ] Beta landing-page banner in place: `"Beta — you can lose real money"` visible on every page once the user's `trading_mode = live`.

### 1.2 Infrastructure

- [ ] Staging environment (`staging.traderbotapp.com`) running the same codebase. Ran against Alpaca **paper** with `ENABLE_LIVE_TRADING=true` for a 7-day soak. Zero unhandled exceptions in logs.
- [ ] Off-site Postgres backups verified restorable end-to-end. [`deploy/backup_offsite.sh`](deploy/backup_offsite.sh) cron active on the EC2 host.
- [ ] SSE streaming bug fix verified in staging (chat messages render live without refresh). Fix landed in commit `0ee5886`.
- [ ] `RUNBOOK.md` read end-to-end by the operator performing this enablement. If that's the same person who wrote it, someone *else* reads it.

### 1.3 Alpaca account

- [ ] Alpaca **live** account approved (not just paper). This can take days — start early.
- [ ] Alpaca live account funded, OR the user funding their own accounts has been verified via Alpaca's ACH process.
- [ ] Alpaca account settings review: pattern-day-trader rules understood, margin enabled / disabled per plan, shortability per plan.
- [ ] Alpaca live **API key pair** generated and stored securely. Do not reuse paper keys — Alpaca uses a different API base URL for live (`https://api.alpaca.markets` vs `https://paper-api.alpaca.markets`) and the key scopes are separate.

### 1.4 Observability

- [ ] CloudWatch / Grafana dashboard showing: request rate, error rate, p95 latency, Claude token cost per hour.
- [ ] Alert wired for `LiveOrderFailed` audit events (should be zero; any occurrence pages the operator).
- [ ] Alert wired for broker circuit breaker trips.
- [ ] Alert wired for daily-loss-cap breaches.
- [ ] Alert wired for `/metrics` endpoint returning 5xx (means observability itself is down).

### 1.5 Support readiness

- [ ] Support email published on the landing page (`support@traderbotapp.com` or equivalent).
- [ ] Auto-responder confirms receipt within 5 min.
- [ ] Operator(s) have notifications on for support email for the first 48 hours after enablement.

### 1.6 Beta whitelist

- [ ] List of beta user emails + identities. Real people you know or can reach, not anonymous signups.
- [ ] Maximum **3 users** for the first real trade. Scale to 10 only after 7 clean days.
- [ ] Each beta user has been told: their initial cap is $1,000, you'll be watching every trade, they agree to report anomalies within 24 h.

---

## Part 2 — Operator actions to flip the platform flag

With Part 1 complete, you are now authorized to flip the platform-level flag. **This alone does not open live trading** — it just enables the code path that each user must still opt into.

### 2.1 Move Alpaca live keys into secrets storage

```bash
# On the EC2 host (/opt/traderbot/deploy/):
# 1. Back up the current .env.prod (paper keys)
cp .env.prod .env.prod.pre-live.$(date +%Y%m%d_%H%M%S)

# 2. Replace the paper keys with live keys
#    (paper keys typically start with PK... and live keys start with AK...)
sed -i 's/^ALPACA_API_KEY=.*/ALPACA_API_KEY=<LIVE_KEY_FROM_ALPACA>/' .env.prod
sed -i 's/^ALPACA_SECRET_KEY=.*/ALPACA_SECRET_KEY=<LIVE_SECRET_FROM_ALPACA>/' .env.prod

# 3. Verify (check prefixes, not values)
grep '^ALPACA_' .env.prod
```

- [ ] Live keys present, paper keys backed up.

### 2.2 Flip the platform flag

```bash
# Still in .env.prod:
sed -i 's/^ENABLE_LIVE_TRADING=.*/ENABLE_LIVE_TRADING=true/' .env.prod

# Confirm EMERGENCY_HALT is off (belt-and-suspenders — it should already be false)
grep '^EMERGENCY_HALT=' .env.prod    # expect: EMERGENCY_HALT=false
```

- [ ] `ENABLE_LIVE_TRADING=true` present in `.env.prod`.
- [ ] `EMERGENCY_HALT=false`.

### 2.3 Restart the API and verify

```bash
bash deploy.sh restart

# Wait ~30s, then verify the API is healthy
docker compose -f docker-compose.prod.yml --env-file .env.prod ps | grep traderbot-api
# expect: Up (healthy)
```

- [ ] API restarted with new env.
- [ ] Healthcheck green.

### 2.4 Negative verification — paper users still get paper

This is critical. With the platform flag on, **any existing user whose `trading_mode` is still `PAPER`** must still route to paper. Let's prove it.

```bash
# Connect to the prod DB
docker compose -f docker-compose.prod.yml --env-file .env.prod exec db \
  psql -U trading traderbot -c \
  "SELECT id, email, trading_mode FROM users LIMIT 5;"
# Expect every row: trading_mode = 'paper'
```

- [ ] Every existing user is still `trading_mode = 'paper'`.

Now log into the app as one of those paper users and place a small market order (e.g., 1 share of AAPL). Check the audit log:

```bash
docker compose -f docker-compose.prod.yml --env-file .env.prod exec db \
  psql -U trading traderbot -c \
  "SELECT action, occurred_at FROM audit_events WHERE aggregate_type='order' ORDER BY occurred_at DESC LIMIT 5;"
```

- [ ] No `LiveOrderPlaced` events.  Only `OrderPlaced` or `TradeRecommendationCreated` — paper paths.

**If you see `LiveOrderPlaced` at this stage, flip `ENABLE_LIVE_TRADING=false` immediately and investigate.** The user routing is the last line of defence.

### 2.5 Positive verification — factory refuses live for paper users

The live endpoint smoke test:

```bash
# On the host:
curl -sX POST https://traderbotapp.com/api/v1/users/me/live-mode-status \
     -H "Authorization: Bearer <paper-user-JWT>" | jq

# Expect:
# { "trading_mode": "paper", "daily_loss_cap_usd": null, ... "has_totp": false }
```

- [ ] Paper user's `live-mode-status` returns `paper`.

---

## Part 3 — Enable a specific user for live trading

The operator does **nothing** in this section — the user does it. But the operator monitors each step and verifies the audit trail after.

### 3.1 User enrols TOTP

1. User logs in, navigates to Settings → Live Trading.
2. Clicks **Enable Live Mode**. Frontend calls `POST /api/v1/users/me/totp/enroll`.
3. Response contains a plaintext TOTP secret + provisioning URI. Frontend renders a QR code.
4. User scans with Google Authenticator / Authy / 1Password.
5. Frontend displays the 6-digit code input and the 4 gates (KYC / risk phrase / cap / TOTP code).

Operator verification:

```bash
docker compose -f docker-compose.prod.yml --env-file .env.prod exec db \
  psql -U trading traderbot -c \
  "SELECT id, email, totp_secret_encrypted IS NOT NULL AS totp_enrolled FROM users WHERE email='<beta-email>';"
# expect: totp_enrolled = t
```

- [ ] User's `totp_secret_encrypted` is now populated.

### 3.2 User submits the enable-live-mode form

The UI collects:
- **KYC attestation** — a free-text payload (≥ 20 chars) confirming legal name + jurisdiction + age + funds ownership. Hashed server-side.
- **Daily loss cap** — USD amount. Launch cap: `1000`. Must be > 0 and ≤ 10000.
- **TOTP code** — the current 6-digit code from their authenticator.
- **Risk acknowledgement** — user must type exactly `"I understand I will lose real money."` (case-sensitive, trailing period required).

On submit the frontend calls `POST /api/v1/users/me/enable-live-mode`. Gates run server-side in this order:

1. `ENABLE_LIVE_TRADING=true` at the platform level (403 if off).
2. User has TOTP enrolled (400 `totp_not_enrolled`).
3. TOTP code verifies (400 `totp_invalid`).
4. Risk phrase matches exactly (400 `risk_phrase_mismatch`).
5. KYC payload non-trivial (Pydantic-validated at boundary).
6. Daily-loss cap in range (Pydantic-validated).

Every outcome — success or any failure — emits an audit event.

Operator verification:

```bash
docker compose -f docker-compose.prod.yml --env-file .env.prod exec db \
  psql -U trading traderbot -c \
  "SELECT action, occurred_at, payload_json->>'reason' AS reason
   FROM audit_events WHERE actor_user_id='<user-id>'
   ORDER BY occurred_at DESC LIMIT 10;"
```

Expected final row: `action = LiveModeEnabled`. If instead you see `LiveModeEnableRejected` rows only, something gated them. The `reason` column tells you which.

- [ ] `LiveModeEnabled` audit event present for the user.
- [ ] `users.trading_mode = 'live'` for the user:
  ```bash
  docker compose -f docker-compose.prod.yml --env-file .env.prod exec db \
    psql -U trading traderbot -c \
    "SELECT trading_mode, daily_loss_cap_usd, live_mode_enabled_at, kyc_attestation_hash IS NOT NULL
     FROM users WHERE id='<user-id>';"
  ```
- [ ] `daily_loss_cap_usd` is exactly the cap the user entered.

### 3.3 Do not skip: fund the account

Alpaca requires the live brokerage account to have funds. This is **separate** from our `daily_loss_cap_usd` — our cap is how much the user is allowed to *lose*, not how much they can trade with. Alpaca's live account balance is the hard ceiling.

- [ ] The user's Alpaca live account has funds equal to at least the first planned order's notional (plus buffer).

---

## Part 4 — First live order: dry-run ritual

Before the beta user places any order, **the operator** places a $5-range test order from their own beta account to prove the full path works end to end. Do not hand the keys to users before this.

### 4.1 Place the test order

Log in as your own beta account (must also be in `live` mode). Place the smallest meaningful order:

- Symbol: `AAPL` (liquid enough that spread is pennies).
- Action: `BUY`.
- Quantity: `1`.
- Order type: `MARKET`.

When the UI prompts for a TOTP code, enter the current code.

### 4.2 Verify the order

Within 30 seconds:

```bash
# 1. Our database should show status=EXECUTED and a broker_order_id note
docker compose -f docker-compose.prod.yml --env-file .env.prod exec db \
  psql -U trading traderbot -c \
  "SELECT id, symbol, status, filled_quantity, notes FROM orders
   WHERE user_id='<your-user-id>' ORDER BY placed_at DESC LIMIT 1;"

# 2. Audit event LiveOrderPlaced should be present
docker compose -f docker-compose.prod.yml --env-file .env.prod exec db \
  psql -U trading traderbot -c \
  "SELECT action, payload_json FROM audit_events
   WHERE aggregate_type='live_order' AND actor_user_id='<your-user-id>'
   ORDER BY occurred_at DESC LIMIT 1;"
# expect: action = 'LiveOrderPlaced', payload.broker_order_id populated
```

- [ ] `orders.status = 'EXECUTED'`.
- [ ] `orders.notes` contains `broker_order_id=<something that is NOT a 'simulated_' prefix>`. If it starts with `simulated_`, the silent-stub fallback is still live — STOP and investigate.
- [ ] `LiveOrderPlaced` audit event present.

### 4.3 Cross-verify on Alpaca's side

1. Log into `https://app.alpaca.markets` (not the paper one).
2. Orders → find the order from seconds ago.
3. Positions → confirm 1 share AAPL held.

- [ ] Order visible in Alpaca live dashboard with matching broker_order_id.
- [ ] Position visible.

### 4.4 Close the position

Place a `SELL` order for 1 share of AAPL. Repeat the verification. Confirm you are back to zero AAPL in Alpaca's dashboard.

- [ ] Position closed in our DB.
- [ ] Position closed in Alpaca.
- [ ] Round-trip P&L is within pennies of expectation (spread + commission).

### 4.5 Cost check

Log into Alpaca's statements. You should see two trades. Compare to the `cost_usd` field in the Claude call logs (if you used chat to place the order) — everything adds up.

- [ ] Fees reasonable (Alpaca has no per-trade commission for most retail users).
- [ ] Claude API cost for the chat interaction < $0.05.

**Only after all of Part 4 passes** is the platform considered "live" in any meaningful sense. At this point the operator may invite the first beta user.

---

## Part 5 — Kill switches (memorise before you flip anything on)

### 5.1 Platform-wide immediate halt

Stops every user's live trading. Users mid-session see the next order refused.

```bash
ssh ubuntu@traderbotapp.com
cd /opt/traderbot/deploy
sed -i 's/^EMERGENCY_HALT=.*/EMERGENCY_HALT=true/' .env.prod
bash deploy.sh restart
```

Verify: in the next API log line you see `broker_routed_to_paper reason=emergency_halt`.

### 5.2 Feature-flag off

Less aggressive — existing sessions might have already routed; new orders will be paper.

```bash
sed -i 's/^ENABLE_LIVE_TRADING=.*/ENABLE_LIVE_TRADING=false/' .env.prod
bash deploy.sh restart
```

### 5.3 Revert one specific user to paper

```bash
# Via the API (user themselves can also do this via the frontend):
curl -sX POST https://traderbotapp.com/api/v1/users/me/disable-live-mode \
     -H "Authorization: Bearer <user-JWT>"
```

Verify:

```bash
docker compose -f docker-compose.prod.yml --env-file .env.prod exec db \
  psql -U trading traderbot -c \
  "SELECT trading_mode FROM users WHERE id='<user-id>';"
# expect: paper
```

### 5.4 The nuclear option — stop the whole API

```bash
docker compose -f docker-compose.prod.yml --env-file .env.prod stop api
# The site returns 502 from Caddy. No orders possible.
```

Use only if you cannot reach your keyboard fast enough for a clean halt. Customers see errors.

---

## Part 6 — Monitoring: the first 48 hours

The operator watches **continuously** for the first 48 hours after enabling the first beta user. Leave the tabs open.

### 6.1 Dashboards to keep open

1. **Alpaca live positions** page — is anything open you didn't expect?
2. **`audit_events` tail**:
   ```bash
   docker compose -f docker-compose.prod.yml --env-file .env.prod exec db \
     psql -U trading traderbot -c \
     "SELECT occurred_at, actor_user_id, action, payload_json->>'reason' AS reason
      FROM audit_events ORDER BY occurred_at DESC LIMIT 20;"
   ```
   Refresh every few minutes. Any `LiveOrderFailed` = investigate. Any `LiveOrderRejected` with `reason=auth_rejected` = Alpaca keys are wrong, hit the kill switch.
3. **API logs:** `docker compose logs -f api --tail=100`.
4. **Grafana CloudWatch dashboard** if wired.

### 6.2 Daily checks (for the first 7 days)

Every morning at market open (14:30 UTC for US markets):

- [ ] Compare `orders` count in our DB to Alpaca's daily trade count. They MUST match.
- [ ] Compare per-user P&L in our cached `Portfolio` to Alpaca's account value. Drift of more than a few cents means our cache has a bug; investigate same day.
- [ ] Review `audit_events` from the last 24 h. Any row you don't immediately understand is a bug.
- [ ] Check Claude API spend. If it's more than 2× yesterday, throttle or investigate.

### 6.3 What to do when something looks wrong

| Symptom | Immediate action | Investigate |
|---|---|---|
| Unexpected `LiveOrderPlaced` event | Halt platform (5.1). | Who placed it? Was chat involved? Review conversation log. |
| `LiveOrderRejected` with `auth_rejected` | Halt platform (5.1). | Alpaca keys rotated or revoked. Re-issue. |
| Our `Portfolio.cash_balance` ≠ Alpaca `buying_power` by more than $1 | Stop accepting orders (5.4) until reconciled. | Find the missing or ghost transaction. Likely a webhook / polling race. |
| Broker circuit breaker trips | No action needed (auto-recovers in 5 min). | Look at the 3 failures that tripped it. Alpaca outage? Rate limited? |
| Daily-loss cap breached for a user | Cap enforces itself (user is auto-reverted to paper). | Email the user that their cap triggered. Ask if they want to raise it. |
| User emails support claiming order didn't execute / double-executed | Verify against Alpaca dashboard FIRST, DB second. Alpaca is authoritative. | If our DB says filled but Alpaca shows nothing, there's a webhook/polling gap. If Alpaca says filled but our DB doesn't, run `poll_pending_orders()` manually. |

---

## Part 7 — Widening beyond the first 3 users

Before adding user #4:

- [ ] 7 consecutive days of live trading without any `LiveOrderFailed` events.
- [ ] Daily P&L reconciliation has never drifted more than $1 from Alpaca.
- [ ] Zero user-reported issues about order fills.
- [ ] Zero silent stub responses (`broker_order_id` never starts with `simulated_`).
- [ ] Operator has been able to sleep the last 3 nights without incident.

Only then add user #4. Stop at 10 users until revenue / scale justifies deeper monitoring.

Before widening past 10:

- Add automated alerting on every row in the table in §6.3.
- Move prod off EC2 to the Fargate stack (ADR-003). Single-host is acceptable for 10 users, not for 100.
- Legal review of terms of service by an actual attorney.

---

## Part 8 — Off-ramp: disabling live for everyone

If the app is sunsetting, or a critical bug requires a full halt longer than a weekend:

1. `ENABLE_LIVE_TRADING=false` in `.env.prod`, restart.
2. Email every live user: "live trading temporarily disabled, effective <timestamp>. Existing positions remain open in your Alpaca account — close them there directly if desired."
3. For each live user: flip them back to paper in the DB:
   ```bash
   docker compose -f docker-compose.prod.yml --env-file .env.prod exec db \
     psql -U trading traderbot -c \
     "UPDATE users SET trading_mode='paper' WHERE trading_mode='live';"
   ```
4. Audit row for each flip recorded manually (write a reason into `audit_events` via operator script).
5. Revoke Alpaca live keys. Don't leave them in `.env.prod` if the code can't use them.

---

## Appendix A — The checklist as a single-page printout

Pre-flight — Part 1:
- [ ] Terms + risk + privacy published  [ ] Staging 7-day soak  [ ] Off-site backups verified
- [ ] Alpaca live account approved + funded  [ ] Live API keys issued + stored
- [ ] Observability alerts armed  [ ] Support email live  [ ] Beta whitelist (≤ 3 names)

Platform flip — Part 2:
- [ ] Alpaca live keys in `.env.prod`  [ ] `ENABLE_LIVE_TRADING=true`  [ ] `EMERGENCY_HALT=false`
- [ ] API restarted + healthy  [ ] Paper users still route paper (verified)

Per-user enable — Part 3:
- [ ] User enrolled TOTP  [ ] User submitted enable-live-mode + passed 6 gates
- [ ] DB shows `trading_mode='live'`  [ ] `LiveModeEnabled` audit event present
- [ ] User's Alpaca account funded

First order dry-run — Part 4 (operator only):
- [ ] Placed 1-share AAPL market BUY  [ ] Our DB shows EXECUTED, real broker_order_id
- [ ] Alpaca dashboard shows the order  [ ] Closed the position  [ ] Round-trip P&L within pennies

Kill switches committed to memory:
- [ ] `EMERGENCY_HALT=true` + restart
- [ ] `ENABLE_LIVE_TRADING=false` + restart
- [ ] `POST /users/me/disable-live-mode`
- [ ] `docker compose stop api`

Monitoring (next 48 h):
- [ ] Alpaca positions tab open  [ ] `audit_events` tail open
- [ ] API logs open  [ ] Grafana dashboard open

---

**End of runbook.** If you haven't ticked every box in §1–§4 before the first live trade, go back and finish. The single most common failure in an enablement like this is haste.
