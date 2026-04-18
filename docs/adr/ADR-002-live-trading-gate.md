# ADR-002: Live-Trading Go/No-Go Criteria

- **Status:** Accepted
- **Date:** 2026-04-18
- **Deciders:** Allan Smeyatsky (solo operator)
- **Supersedes:** N/A
- **Referenced by:** `rewrite181226.md`, Phase 6, Phase 8

## Context

TraderBot has run in paper-trading mode against Alpaca's sandbox. The rebuild plan (`rewrite181226.md`) wires real-money execution in Phase 6 and launches a closed beta in Phase 8. Before writing live-mode code, we need explicit, testable criteria for:

1. When a user is allowed to enable live mode.
2. What guards prevent a runaway bot from blowing up an account.
3. What kill-switches halt live trading without a deploy.
4. What regulatory posture TraderBot takes.

This ADR records those decisions so future implementation has a fixed target and future reviews have a baseline to evaluate drift against.

## Decision

### Eligibility (per user)

A user may flip from `trading_mode='paper'` to `trading_mode='live'` only when **all four** are true:

1. **KYC attestation** recorded. User has explicitly confirmed: name, residency, age ≥ 18, "I am trading my own money, not on behalf of others." Hash of the attestation payload + timestamp + IP stored in `users.kyc_attestation_hash`.
2. **2FA enabled.** TOTP via `pyotp`, verified once during enablement. Stored secret is per-user and encrypted at rest (AWS Secrets Manager after Phase 3, AES-GCM with a per-user salt until then).
3. **Daily-loss cap set.** User selects a per-account USD cap (default $100, max $1,000 for beta). Stored in `users.daily_loss_cap_usd`.
4. **Explicit risk acknowledgement.** User confirms the exact phrase "I understand I will lose real money." in the UI. The confirmation event is persisted to `audit_events` as `LiveModeEnabled`.

A reversal (live → paper) requires no gate and is free at any time.

### Per-order guards

Every live order must pass **all** of the following before reaching the broker:

1. **Re-authentication.** TOTP challenge OR fresh re-auth within the last 5 minutes. AI-initiated orders cannot bypass this — the existing human-confirmation UI carries the TOTP challenge for live-mode users.
2. **Live balance check.** Pre-trade notional ≤ live Alpaca account `buying_power` (fetched live, not cached `Portfolio`).
3. **Daily-loss check.** If realised + unrealised loss for the current UTC day exceeds `daily_loss_cap_usd`, live orders are rejected and the user is auto-flipped to `paper` mode. Re-enabling live mode requires re-running the full eligibility flow.
4. **Position concentration check.** Existing `RiskManager` pre-trade validation applies unchanged.

### Kill-switches (no-deploy halts)

1. **Global:** environment variable `EMERGENCY_HALT=true` short-circuits every live order. Startup logs the current value; a dashboard tile surfaces it.
2. **Per-broker circuit breaker:** 3 consecutive 5xx or 401/403 responses from Alpaca → halt live orders for 5 minutes, alert via structured log (and Grafana alerting once wired in Phase 2).
3. **Per-user:** user can set `trading_mode='paper'` at any time via a "halt live trading" button on the dashboard.

### Regulatory posture

TraderBot is positioned as an **advisor dashboard + broker client**, not a broker-dealer:

- Alpaca holds the broker-dealer registration and custody relationship. TraderBot routes orders to Alpaca; funds never touch our infrastructure.
- Users authenticate into their own Alpaca accounts (via API key + secret entered in TraderBot settings). We do not execute on behalf of pooled funds.
- Terms of service + risk disclosure published on the landing page before Phase 8 launch. Must include: "not FDIC insured", "past performance not indicative", "AI can be wrong, every trade requires your confirmation".
- No marketing language implies guaranteed returns or suggests TraderBot is licensed as an investment adviser. If/when that changes, open a new ADR.

### Beta scope (Phase 8)

- Maximum 3 beta users in live mode at launch.
- Each capped at $1,000 initial capital (enforced via `daily_loss_cap_usd` = $100 and user-acknowledged total cap).
- 7-day soak on all three before widening.
- Expansion gates: zero unhandled exceptions in logs, zero broker-side rejections that the bot didn't anticipate, zero user-reported incorrect trades.

## Consequences

**Must build (tracked in `rewrite181226.md` Phase 6):**
- `users.trading_mode`, `users.kyc_attestation_hash`, `users.daily_loss_cap_usd`, `users.totp_secret_encrypted` columns + migration.
- `POST /users/me/enable-live-mode` endpoint validating all four eligibility gates.
- TOTP challenge middleware on live-order paths.
- `LiveModeEnabled`, `LiveOrderPlaced`, `LiveOrderRejected`, `DailyLossCapBreached` domain events (audit trail per `rewrite181226.md` Phase 3).
- `EMERGENCY_HALT` env check in `TradingExecutionPort` implementations.
- Circuit breaker integration with Alpaca response codes.
- Landing-page TOS and risk disclosure before Phase 8.

**Explicitly deferred:**
- Multi-broker support (IBKR, Schwab). ADR-003 when added.
- Pooled-fund / managed-account features. Would trigger broker-dealer registration requirements — out of scope.
- Paper-mode users following live-mode users' strategies via copy-trading. Domain model supports it but the beta does not.

## Revisit Triggers

Open a new ADR if any of the following:

- Beta expands beyond 3 users → capital caps need to scale.
- Regulatory framework changes (e.g. SEC rule on robo-advisers) materially affects posture.
- A second broker is added → authentication model and circuit-breaker scope change.
- An incident reveals a guard that did not fire when it should have.

## Related

- `Architectural Rules — 2026.md` §3.4 (AI output validation), §4 (security hard rules)
- `rewrite181226.md` Phase 6, Phase 8, Appendix C
- `src/infrastructure/broker_integration.py` (stub fallbacks removed per Phase 0 quick-win, 2026-04-18)
