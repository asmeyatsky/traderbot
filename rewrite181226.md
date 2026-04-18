# TraderBot — 2026 Launch Rebuild Plan

**Date:** 2026-04-18
**Author:** Claude (Opus 4.7) + Allan Smeyatsky
**Status:** Approved for execution
**Reference:** `Architectural Rules — 2026.md` (v3, 2026-04)

---

## Executive Summary

TraderBot has been in paper-trading test mode and is ready to be rebuilt against the 2026 Architectural Rules and launched with real broker integration. This document captures (1) a gap assessment of the current codebase versus the 2026 rules, (2) a nine-phase rebuild plan, and (3) explicit launch-readiness criteria that gate go-live.

**Cloud decision (ADR-001):** Stay on AWS. Rationale recorded in `memory://decisions/ADR-001-cloud-provider.md`. The 2026 rules allow deviation from GCP-default with a one-paragraph ADR citing cost + single-operator simplicity. Revisit when revenue justifies migration.

**Estimated effort:** ~27 working days (~5.5 weeks) solo.

---

## Part 1 — Assessment: Current vs. 2026 Rules

### 1.1 Stack Alignment (§1 of Rules)

| Rule | Current | Gap |
|------|---------|-----|
| GCP (Cloud Run, Secret Manager, Workload Identity) | AWS EC2 + Docker Compose | **ADR-001 — deviation approved, staying AWS** |
| Python 3.12+ | 3.11 (Docker), 3.14 (local) | Minor — bump Dockerfile to 3.12 |
| TypeScript + React frontend | ✅ React 19 + TS + Vite | OK |
| Postgres primary / Redis cache | ✅ Postgres + KeyDB (Redis-compatible) | OK |
| Firestore for documents | None | Not required yet — defer until needed |
| BigQuery analytics | None | Defer — not launch-critical |
| Protobuf or JSON Schema for cross-service contracts | None | **HIGH** — MCP servers (Phase 4) will enforce JSON Schema |

### 1.2 Layer Direction & Non-Negotiables (§2, §3)

| Rule | Current | Status |
|------|---------|--------|
| `domain ← application ← infrastructure ← presentation` enforced by linter | DDD structure exists; no import-linter, no ESLint boundaries plugin | ❌ "Rule not in CI = rule not real" |
| One MCP server per bounded context | **Zero MCP servers.** 13 tools wired directly into `chat.py` use case | ❌ Bounded contexts leak |
| Every external dep has a port | Most do (`TradingExecutionPort`, `AIChatPort`, `ConversationRepositoryPort`) | ✅ Mostly |
| Immutable domain models | Chat entities frozen; some services carry mutable state | ⚠️ Partial |
| Business logic out of infrastructure | `trading.py:112-118` auto-fills orders inside a use case (paper simulation in wrong layer) | ⚠️ |
| MCP tools = writes, resources = reads | N/A — no MCP | ❌ |
| Every external call has timeout | `requests` calls have no timeout; Anthropic stream no timeout | ❌ |

### 1.3 Security (§4) — CI-Enforced

| Rule | State |
|------|-------|
| Secrets via Secret Manager + Workload Identity | AWS Secrets Manager supported in code, but prod uses **`.env.prod` file on disk** → ❌ |
| Pydantic validation at boundary | ✅ Strong — all routers enforce schemas |
| Audit event per write (actor, action, before/after hash, append-only, separate IAM) | ⚠️ Request logs exist; domain events inconsistent on write paths |
| Timeout + circuit breaker on every external call | ⚠️ Partial (aiohttp OK; `requests` and Anthropic missing) |
| AI output validated against schema before state mutation | ✅ `TradeAction` validated; user-confirm gate exists |
| Scoped, time-boxed MCP tool access | ❌ No MCP = no scoping granularity |
| Supply chain: lockfiles, dep scan, signed commits, SBOM | ❌ None configured in CI |

### 1.4 Testing Floor (§5)

| Rule | Current | Gap |
|------|---------|-----|
| Overall ≥ 80% | CI fails below **15%** | ❌ 5× under floor |
| Domain ≥ 95%, zero mocks | Domain tests exist, no split coverage report | ❌ Unverifiable |
| Application ≥ 85%, mock ports only | No application-layer coverage gate | ❌ |
| MCP schema + round-trip tests | No MCP | ❌ |

### 1.5 Observability (§6)

| Rule | State |
|------|-------|
| OpenTelemetry tracing propagated through MCP | ❌ Not installed |
| RED metrics per endpoint and per MCP tool | ❌ No Prometheus/OTel metrics |
| Structured JSON logs with correlation IDs, zero PII | ⚠️ JSON formatter + PII filter exist; **correlation IDs not populated** (`logging.py:41-42` schema ready) |
| Per-AI-call log: model, prompt hash, tokens in/out, latency, cost | ❌ **Zero telemetry** in `claude_chat_adapter.py:122` — biggest launch blocker for AI spend |

### 1.6 Trading Reality (launch blockers)

- `src/infrastructure/di_container.py:169` — `paper_trading=True` **hardcoded**, no override.
- `src/infrastructure/broker_integration.py:141-148` — on Alpaca 403 (bad key) silently returns `broker_order_id=simulated_*`. **Most dangerous line in the codebase.**
- No `PAPER_TRADING` / `LIVE_MODE` env switch, no per-user mode.
- Only Alpaca wired; IBKR env vars exist but no adapter.
- SSE streaming known broken in frontend — AI responses don't render live (memory note).

### 1.7 Anti-patterns Observed

From §7 of the rules:
- Partial business-logic-in-infrastructure (order auto-fill in use case, not domain service) — #1
- No mechanical enforcement of anything declared — #10
- Untyped strings from AI used without MCP schema gate — #6 (mitigated by TradeAction validation but inconsistent)
- Sequential execution of independent tool calls in chat loop — #5 (potential optimization)

---

## Part 2 — Rebuild Plan (9 Phases)

Each phase ends with a deployable increment and CI gates that prevent regression.

### Phase 0 — Decision Gate (0.5 day)

- **ADR-001 — Cloud provider: AWS.** Already decided. Write the one-paragraph ADR in `memory://decisions/ADR-001-cloud-provider.md` citing: existing $26/mo AWS footprint, Caddy + Docker Compose operational simplicity for single operator, no multi-region need at launch scale, migration deferred until revenue justifies.
- **ADR-002 — Live-trading go/no-go criteria.** Define:
  - Initial capital cap: $1,000 per beta user.
  - KYC attestation + 2FA required before enabling live mode.
  - Global `EMERGENCY_HALT` env kill-switch.
  - Per-order re-authentication challenge.
  - Daily-loss hard stop per account.
  - Regulatory posture: Alpaca handles broker-dealer compliance; TraderBot is an "advisor dashboard" not a broker. Terms of service + risk disclosure on landing page.

### Phase 1 — Architecture Enforcement (3 days)

Make the rules real by putting them in CI.

**Backend:**
- Add `import-linter` with a `.importlinter` config:
  - Contract 1 (layers): `domain → application → infrastructure → presentation` — forbids reverse imports.
  - Contract 2 (domain purity): `domain` forbids imports from `anthropic`, `alpaca_trade_api`, `sqlalchemy`, `fastapi`, `httpx`, `redis`, any SDK.
  - Contract 3 (application purity): `application` may import only `domain`.
- Add `lint-imports` step to `.github/workflows/test.yml` — required check.

**Frontend:**
- Add `eslint-plugin-boundaries`. Define layers: `components/`, `api/`, `stores/`, `hooks/`, `views/`.
- Rule: `stores/` cannot import `components/`; `api/` cannot import `components/` or `stores/`.
- CI step: `eslint --max-warnings 0`.

**Expected fallout:** Several violations will surface. Chat tool handling will need to be moved out of `use_cases/chat.py` into domain services — this is the foundation for Phase 4 (MCP).

**Exit gate:** `lint-imports` green in CI, `eslint --max-warnings 0` green.

### Phase 2 — Observability Floor (3 days)

Launch-blocking for AI spend visibility.

**OpenTelemetry:**
- Install `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-instrumentation-fastapi`, `opentelemetry-instrumentation-sqlalchemy`, `opentelemetry-exporter-otlp`.
- Add request-ID middleware that populates the existing `correlation_id` field already wired in `src/infrastructure/logging.py:41-42`.
- Propagate correlation ID through Claude tool calls so an end-to-end trace connects chat request → tool call → broker API.

**Per-Claude-call telemetry (highest priority):**
- Wrap `claude_chat_adapter.py:122` final-message handler to log JSON with fields:
  - `model` (e.g., `claude-haiku-4-5-20251001`)
  - `prompt_hash` (sha256 of rendered prompt + tools)
  - `input_tokens`, `output_tokens`
  - `cache_read_input_tokens`, `cache_creation_input_tokens`
  - `latency_ms`
  - `cost_usd` (computed from Haiku 4.5 pricing table: $1/MTok input, $5/MTok output, $0.08/MTok cache read, $1.25/MTok cache write)
  - `user_id`, `conversation_id`, `correlation_id`

**RED metrics:**
- Install `prometheus_fastapi_instrumentator`.
- Expose `/metrics` endpoint (internal network only).
- Metrics: request count, error count, p50/p95/p99 duration per endpoint; same per MCP tool once Phase 4 lands.
- Scrape from Grafana Cloud free tier (no new infra cost).

**Exit gate:** `/metrics` returns non-empty in smoke test; every Claude call produces a telemetry log line; Grafana dashboard shows live RED charts.

### Phase 3 — Security Hardening (3 days)

Close the four high-priority gaps from the audit.

**Timeouts:**
- Add `timeout=10` to every `requests.get()` in `src/infrastructure/api_clients/market_data.py`.
- Construct Anthropic SDK client with explicit `httpx.Timeout(30.0, connect=5.0)`.
- All async HTTP clients get per-call timeouts, not just client-level defaults.

**Rate limiting:**
- Add `slowapi` dependency.
- Apply `@limiter.limit("5/minute")` to `/users/login`.
- Apply `@limiter.limit("60/minute")` to `/chat/messages` per user.

**Trusted hosts:**
- Set `TrustedHostMiddleware(allow_hosts=["traderbotapp.com", "www.traderbotapp.com"])` in production. Remove the permissive `["*"]`.

**Audit events:**
- New `audit_events` table: append-only, separate DB role (`traderbot_audit_writer`) with INSERT-only grant.
- Columns: `id`, `actor_user_id`, `action`, `aggregate_type`, `aggregate_id`, `before_hash`, `after_hash`, `payload_json`, `occurred_at`, `correlation_id`.
- Emit on every write path: `OrderPlaced`, `OrderCancelled`, `StrategyFollowed`, `StrategyForked`, `PositionClosed`, `LiveModeEnabled`, `LoginSucceeded`, `LoginFailed`.
- Handler lives in application layer; domain emits the event, infrastructure persists it.

**Supply chain:**
- Add `pip-audit` to CI — fail build on HIGH/CRITICAL CVEs.
- Add `npm audit --production --audit-level=high` to CI.
- Enable branch protection on `main`: signed commits required, passing CI required, 1 review (or self-review for solo).
- Generate SBOM with `syft` at build time, attach to release artifact.

**Secret rotation:**
- Document JWT secret rotation runbook.
- Move prod secrets from `.env.prod` on disk into AWS Secrets Manager; container fetches at startup. This is explicitly called out in §4 — "no secrets in code, config, env defaults."

**Exit gate:** `pip-audit` + `npm audit` green in CI; signed commits enforced; no `.env.prod` on disk in production.

### Phase 4 — MCP Bounded-Context Refactor (5 days)

Biggest structural change. Replace the 13-tool monolith in `chat.py` with MCP servers — one per bounded context, per §3.5.

**Three MCP servers:**

1. **`mcp-market-data`** (reads)
   - Resources: `price://{symbol}`, `news://{symbol}`, `technical://{symbol}/{timeframe}`, `market-status://`
   - Wraps YahooFinance, Polygon, Finnhub clients.
   - No tools — pure reads.

2. **`mcp-portfolio`** (reads + writes)
   - Tools: `place_order`, `cancel_order` (state mutations).
   - Resources: `portfolio://current`, `positions://{symbol}`, `orders://open`, `position-details://{symbol}`.
   - Tool boundary enforces risk pre-checks via domain service.
   - Every tool invocation emits an audit event.

3. **`mcp-research`** (reads + writes)
   - Tools: `run_backtest`, `save_strategy`, `fork_strategy`.
   - Resources: `strategy://{id}`, `backtest://{id}`, `leaderboard://`, `screener://{screen_name}`.

**Per-server structure (strict §3.7 compliance):**

```
src/infrastructure/mcp/<context>/
├── server.py          # MCP server entrypoint (wraps use cases)
├── tools.py           # tool definitions + JSON Schema
├── resources.py       # resource definitions
├── schemas/           # JSON Schema files for all inputs/outputs
└── tests/
    ├── test_schema_compliance.py
    └── test_round_trip.py
```

**Module header for each (§3.7):**
```python
"""
Portfolio MCP Server

Layer: infrastructure
Ports used: TradingExecutionPort, PortfolioRepositoryPort
MCP integration: exposes 2 tools (writes), 4 resources (reads)
Stack choice: stdlib MCP SDK (canonical for Python bounded-context servers)
"""
```

**Chat use case refactor:**
- Replace inline tool handling in `ClaudeChatAdapter` with MCP client calls.
- Agent gets a scoped MCP session per chat turn: market-data (read-only), portfolio (write-scoped to user's own account), research (write-scoped to user's own strategies).
- Session tokens are time-boxed to the chat turn duration.

**Risk callout:** This is the biggest "works now → breaks in refactor" risk in the plan. Mitigate by keeping the existing tool implementations under a feature flag and swapping MCP in per-server with integration tests passing at each step.

**Exit gate:** All 13 existing chat tool behaviors pass integration tests via MCP; round-trip + schema-compliance tests pass for all 3 servers.

### Phase 5 — Testing Floor (4 days)

Raise coverage without chasing meaningless numbers.

**Layer-split coverage:**
- Configure pytest coverage per layer in `pyproject.toml`:
  - `pytest --cov=src/domain --cov-fail-under=95 tests/domain/`
  - `pytest --cov=src/application --cov-fail-under=85 tests/application/`
  - `pytest --cov=src --cov-fail-under=80` (aggregate)
- Update `.github/workflows/test.yml` to run all three and require all to pass.

**Apply markers that are already defined in `pytest.ini` but never used:**
- Decorate every test with `@pytest.mark.domain|application|infrastructure|integration`.
- Enforce in CI: `pytest --strict-markers`.

**Property-based tests:**
- Add `hypothesis`.
- Domain invariants: order total always positive; position quantity always ≥ 0; `Money` addition commutative; `Symbol` round-trip through normalization is idempotent.

**Frontend coverage:**
- Add vitest coverage config.
- Floor: 70% on `stores/` and `api/`; UI components excluded (no business logic).

**MCP round-trip:**
- For each tool on each server, test that `(input → tool → output)` survives schema validation both ways.
- Test that invalid inputs are rejected before reaching application layer.

**Exit gate:** CI enforces all three coverage gates; every test file uses a layer marker; `--strict-markers` passes.

### Phase 6 — Live Trading (5 days)

The actual reason for the rebuild. Gate carefully.

**Per-user mode:**
- Add `trading_mode: Literal["paper", "live"]` to `users` table (default `paper`).
- Migration adds column + `live_mode_enabled_at`, `daily_loss_cap_usd`, `kyc_attestation_hash`.

**Remove hardcoded paper mode:**
- `src/infrastructure/di_container.py:169` — replace `paper_trading=True` with factory that resolves per request based on authenticated user's `trading_mode`.

**Remove silent stub fallback:**
- `src/infrastructure/broker_integration.py:141-148` — on Alpaca 403 in live mode, raise `BrokerAuthenticationError` and emit `LiveOrderFailed` audit event. No `simulated_*` response ever returned in live mode.

**Live-mode gate (UI + backend):**
- User must:
  1. Complete KYC attestation (store hash + timestamp + IP).
  2. Enable 2FA (TOTP via `pyotp`).
  3. Set daily-loss cap.
  4. Explicitly confirm "I understand I will lose real money."
- Backend endpoint `POST /users/me/enable-live-mode` validates all four and flips `trading_mode`.

**Per-order confirmation:**
- Every live order requires a TOTP challenge OR fresh re-auth within last 5 minutes.
- AI cannot flip this bit — human confirmation required in UI as today, but the confirmation now carries an additional cryptographic challenge.

**Real-money guards:**
- Pre-trade check hits **live Alpaca account balance**, not the cached app `Portfolio`.
- `CircuitBreakerService` integrated with live Alpaca order rejection rate: 3 consecutive 5xx / auth errors = halt live orders for 5 min + alert.
- Global `EMERGENCY_HALT=true` env short-circuits all live orders regardless of user mode. Startup log must confirm current value.
- Daily-loss hard stop: application-level check before every live order; if exceeded, auto-flip user to `paper` and require re-attestation.

**Audit:**
- Every live order emits `LiveOrderPlaced` domain event with: actor, symbol, side, qty, limit, broker response hash, correlation ID, client-fingerprint.
- Audit events stored in separate append-only table (from Phase 3).

**Feature flag:**
- Ship behind `ENABLE_LIVE_TRADING=false`.
- Flip only after Phase 7 completes.

**Exit gate:** Live-mode flow tested end-to-end in staging against Alpaca paper (with live code paths); kill-switch verified; all audit events emit.

### Phase 7 — Launch Readiness (3 days)

**Staging environment:**
- Spin up a second EC2 instance (or reuse with Docker Compose project name) running from `staging` branch.
- Runs against Alpaca paper with `ENABLE_LIVE_TRADING=true` to exercise live-mode code paths without real money risk.
- Separate subdomain: `staging.traderbotapp.com`.
- Cost: ~$15/mo (stopped when not in use).

**Load test:**
- `locust` against staging: 50 concurrent chat sessions.
- Verify p95 < 2s for non-AI endpoints.
- Track Anthropic rate-limit headroom.

**Disaster recovery:**
- Test Postgres restore from `deploy.sh backup` dump — end-to-end, in staging.
- Add off-site backup: nightly cron syncs `deploy/backups/` to `s3://traderbotapp-backups/` with 30-day retention.
- Document restore procedure with timing: RPO = 24h, RTO = 1h.

**Runbook:**
- Create `RUNBOOK.md` in repo root. One-page reference:
  - How to halt trading (`EMERGENCY_HALT=true`)
  - How to restart services (`deploy.sh restart`)
  - How to rotate JWT secret
  - How to restore DB from backup
  - Alpaca support contact + account ID
  - AWS account ID + region
  - Who to call at 3 AM

**Fix known SSE streaming bug:**
- Symptom: AI responses require page refresh to render.
- Likely cause: `flush_interval -1` is set in Caddyfile but `frontend/src/api/chat.ts` async generator isn't flushing to React state correctly.
- Fix approach: verify `TextDecoderStream` usage; ensure React state updates via `flushSync` or equivalent on each chunk boundary.

**Exit gate:** Staging soak for 7 days with zero unhandled exceptions; load test passes; DR restore verified; runbook tested by someone other than the author (or fully re-read by author under timer).

### Phase 8 — Launch (1 day)

**Beta rollout:**
- Flip `ENABLE_LIVE_TRADING=true` in production.
- Whitelist 3 beta users (manual DB update to `trading_mode='live'` after KYC).
- Each capped at $1,000 initial capital.
- Watch dashboards for 7 days before widening.

**Public-facing:**
- Publish terms of service + risk disclosure on landing page.
- Add prominent "Beta — not FDIC insured — you can lose money" banner when in live mode.
- Support email published.

**Communication:**
- Email beta users with onboarding instructions.
- Monitor support inbox actively for first 48 hours.

**Rollback plan:**
- If major issue surfaces, set `ENABLE_LIVE_TRADING=false` — instantly reverts all users to paper without code change.
- All beta users' positions preserved; they can manually close via Alpaca dashboard.

### Phase 9 — Post-Launch Hardening (ongoing)

- Quarterly ADR review — revisit AWS vs. GCP decision against revenue.
- Introduce Protobuf contracts when a second service is added (not before — premature).
- Add BigQuery export of `audit_events` for compliance analytics once SEC/FCA engagement begins.
- Grow beta cohort from 3 → 10 → 50 → public as confidence builds.
- Add second broker (IBKR) once Alpaca volume proves the abstraction.

---

## Part 3 — Launch Readiness Criteria

Block launch until **all** of these are green:

| # | Gate | Pass Condition |
|---|------|----------------|
| 1 | CI: import-linter | 0 violations, enforced on PR |
| 2 | CI: ESLint boundaries | 0 warnings, enforced on PR |
| 3 | CI: coverage | domain ≥ 95%, application ≥ 85%, overall ≥ 80% |
| 4 | CI: supply chain | `pip-audit` + `npm audit` 0 HIGH/CRITICAL |
| 5 | CI: signed commits | Enforced on `main` |
| 6 | AI observability | Every Claude call logs model / prompt-hash / tokens / cost / latency |
| 7 | OTel tracing | End-to-end trace from request → tool → broker visible in Grafana |
| 8 | Secrets | Zero `.env.prod` on disk; AWS Secrets Manager in all prod code paths |
| 9 | Trading mode | Per-user `trading_mode`; stub fallback removed; hardcoded `paper_trading=True` replaced |
| 10 | Kill-switch | `EMERGENCY_HALT` tested in staging |
| 11 | Audit | `OrderPlaced` / `LiveOrderPlaced` / `StrategyFollowed` events emitted and queryable |
| 12 | MCP | 3 servers running; schema + round-trip tests pass |
| 13 | Backup | Off-site Postgres backup verified restorable end-to-end |
| 14 | Staging | 7-day soak with zero unhandled exceptions |
| 15 | Runbook | Written, tested once end-to-end |
| 16 | SSE | Live streaming renders without page refresh |
| 17 | Terms of service | Published and linked from landing page |

---

## Appendix A — Phase Dependencies

```
Phase 0 (ADRs)
    │
    ├─→ Phase 1 (Architecture enforcement)
    │       │
    │       ├─→ Phase 4 (MCP refactor) — needs layer rules in place
    │       │       │
    │       │       └─→ Phase 5 (Testing floor) — needs MCP contracts to test
    │       │
    │       └─→ Phase 2 (Observability)
    │               │
    │               └─→ Phase 6 (Live trading) — needs telemetry before real money
    │
    └─→ Phase 3 (Security hardening) — independent, runs in parallel
            │
            └─→ Phase 6 (Live trading) — needs audit events + Secrets Manager

Phase 6 → Phase 7 (Launch readiness) → Phase 8 (Launch) → Phase 9 (Post-launch)
```

## Appendix B — Estimated Timeline

| Phase | Effort | Cumulative |
|-------|--------|-----------|
| 0 | 0.5 day | 0.5 |
| 1 | 3 days | 3.5 |
| 2 | 3 days | 6.5 |
| 3 | 3 days | 9.5 |
| 4 | 5 days | 14.5 |
| 5 | 4 days | 18.5 |
| 6 | 5 days | 23.5 |
| 7 | 3 days | 26.5 |
| 8 | 1 day | 27.5 |

**~5.5 weeks solo effort.** Parallelize Phase 2 and Phase 3 to save ~3 days if capacity allows.

## Appendix C — Highest-Leverage Quick Wins (Do Tonight)

1. **Remove the silent stub fallback** at `src/infrastructure/broker_integration.py:141-148`. Single most dangerous line in the codebase for going live. Replace with `raise BrokerAuthenticationError`.
2. **Add per-Claude-call logging** to `src/infrastructure/adapters/claude_chat_adapter.py:122`. ~30 lines. You need the cost data before raising traffic.
3. **Draft ADR-001 + ADR-002** in `memory://decisions/`. Unblocks Phases 1–6.

---

**End of plan.**
