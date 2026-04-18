# Phase 10 — Discipline Coach (Post-Launch)

**Status:** Proposed
**Follows:** Phase 8 (launch), Phase 9 (post-launch hardening)
**Estimated effort:** ~6 working weeks solo
**Thesis:** Every other trading app is a recommendation engine. TraderBot's moat is being the only one willing to *refuse* trades the user has stated they don't want to make, and to remember every lesson for them.

---

## Part 1 — Why This Is The Single Most Accretive Move

The retail-trading market is saturated with signal engines. Robinhood, TradingView, ChatGPT plugins, Magnifi, Composer — everyone is racing to tell users what to buy. Yet 90% of retail traders still lose money, and the dominant failure modes are behavioural: over-trading, revenge trades, position sizing drift, emotional exits.

No major trading app is aligned with fixing this, because every broker's P&L is positively correlated with trade volume. TraderBot's chat-native architecture, MCP tool boundary, and existing audit trail make it uniquely positioned to be the first trading app whose AI's primary job is **protecting the user from themselves.**

This is:
- **Genuinely differentiated** — nobody is doing it.
- **Anti-copyable** — requires being willing to reduce trading volume, which competitors can't do without cannibalising their own revenue.
- **Compounding** — longer use = more personal memory = better coaching = more retention.
- **Architecturally cheap** — foundations (MCP, audit events, conversations, per-user data) already exist.

---

## Part 2 — Five Concrete Manifestations

### 10.1 — Pre-trade veto layer (1 week)

Every order — whether initiated via chat or the Orders UI — passes through a `DisciplineCheckPort` before reaching the broker. The AI checks the order against:

- The user's stated position-size cap (`max_position_size_percentage`, already on the `User` entity).
- The user's sector/asset rules (`sector_exclusions`, already modelled).
- Recent emotional-state heuristics (§10.3).
- A free-form `DisciplineRule` the user has written themselves ("never buy on Mondays", "no options", "never average down on losers").

Veto reasons are structured: `{rule_id, rule_text, evidence}`. The UI surfaces them as a modal the user can override with an explicit "I understand this breaks my rule X" confirmation — logged to audit.

**Wiring point:** already exists. Add the check as the first step inside `CreateOrderUseCase.execute` after user loading. A new `DisciplineCheckPort` in `src/domain/ports/` with an infrastructure adapter that calls Haiku with the proposed order + user rules.

**Exit gate:** 100% of orders pass through the check; veto reasons captured in audit; override rate tracked per user.

---

### 10.2 — Episodic memory (2 weeks)

Every significant event becomes a structured episode:

```python
@dataclass(frozen=True)
class Episode:
    id: str
    user_id: str
    occurred_at: datetime
    episode_type: Literal["trade_opened", "trade_closed", "rule_vetoed",
                          "override_used", "reflection_logged"]
    symbol: Optional[str]
    context_snapshot: dict  # portfolio, signals, news at time of event
    rationale_text: str     # user's stated reason OR AI's reasoning
    outcome: Optional[dict] # filled in when position closes
    user_reflection: Optional[str]  # user can self-label post-hoc
    embedding: list[float]  # for semantic recall
```

Stored in Postgres with `pgvector` for semantic search. On every AI turn, the chat use case retrieves the top-K semantically relevant episodes for the current conversation context and injects them into the system prompt:

> "Relevant history: 2026-01-14 — user bought RIVN after 3-day dip, rationale: 'I always buy the dip on EV names'. Outcome: -18% over 30 days. 2026-02-03 — same user attempted similar trade on LCID, AI vetoed, user overrode. Outcome: -22% over 21 days. Current request: same pattern on PSNY."

**Wiring point:** new `EpisodicMemoryPort` in domain, adapter that uses the existing Postgres connection + the `pgvector` extension. One new MCP resource in the portfolio server: `episodes://recent/{symbol}` and `episodes://similar/{rationale_text}`.

**Exit gate:** every trade creates two episodes (open + close). AI prompts include recalled episodes. Users can view their own memory timeline.

---

### 10.3 — Emotional-pattern detection (1 week)

The AI tracks two low-signal-to-noise but high-impact patterns:

**Revenge trade detection.** A trade attempt within 30 minutes of a losing position close, in a similar asset class, with position size ≥ the closed position — triggers a mandatory cooling-off. The UI shows a 10-minute countdown + the AI's framing: "You just closed a loss on X. The trade you're about to place fits the pattern of 2 prior revenge trades in your history, both of which lost." Cannot be skipped; can be cancelled.

**Scale-up-on-losses detection.** Averaging down more than 2x in 7 days on a single position — AI flags in weekly review and locks further adds to that position without explicit override.

Both are computed server-side from the order stream; no ML model needed.

**Exit gate:** patterns detected in a deterministic evaluator (unit-testable); cooling-off enforced in the router; audit events emitted.

---

### 10.4 — Trade journaling with rationale-outcome closure (1 week)

Every order at placement requires (or strongly nudges for) a one-sentence rationale. On position close, the AI asks: "You said this trade was '<rationale>'. It closed at <outcome>. What did you learn?" The user's response becomes the `user_reflection` on the episode.

The chat AI references the rationale during exit conversations: "You opened this position because you believed XYZ. Has that thesis changed?"

**Exit gate:** rationale capture rate ≥ 80% by week 4; reflection capture rate ≥ 40% (people are lazy post-loss).

---

### 10.5 — Weekly discipline report (1 week)

A Monday-morning scheduled task runs per-user analytics:

- Position-size drift (avg vs. stated cap).
- Rule-override count and outcomes.
- Win rate by rationale category ("momentum plays", "news-driven", "dip-buying").
- Most-cited past episode by the AI this week (i.e. which mistakes is the AI still teaching from).
- One-sentence AI verdict: "This week, you followed your rules. Performance: +1.2%, below benchmark. Keep going." *or* "You overrode your position-cap rule 3x. Performance: -4.1%. Consider tightening or relaxing the rule — but pick one."

Delivered as an email + in-app card. Not a notification spam — one report per week.

**Exit gate:** report ships every Monday at 09:00 local time per user; open rate ≥ 40% after 4 weeks.

---

## Part 3 — What This Unlocks

- **Marketing story:** "The only AI trading app that will tell you not to trade." — a headline no incumbent can steal.
- **Regulatory defensibility:** every veto, override, and reflection is auditable. If the SEC / FCA ever asks how the AI makes recommendations, the answer is "from the user's own stated rules and historical behaviour, with a full rationale trail."
- **Data moat:** competitors can copy the UI in weeks. They cannot copy 18 months of a user's reflected-on trade history. That's the switching cost.
- **Retention mechanics:** the longer you use TraderBot, the more expensive it is to leave — because your trading memory is in it.
- **Pricing power:** justifies a premium tier ("Coach") without needing to add more recommendation features — which most users don't want anyway.

---

## Part 4 — What It Costs

- ~6 weeks of engineering, phased in 1–2 week increments shippable independently.
- Postgres + `pgvector` — already running; just need the extension enabled on the prod instance.
- Claude API spend: modest growth from injected episodic memory into prompts, offset by the prompt-caching we already have on system prompts. Estimate: ~15–25% increase in tokens-per-turn.
- Support load: probably net-down. Users who trade less complain less.

---

## Part 5 — What This Doesn't Replace

- The backtesting engine, screener, technical-analysis tools, ensemble AI — all still needed. They're *inputs* to the discipline coach, not replaced by it.
- The launch-critical work (Phase 6 live trading, Phase 7 staging soak). This is strictly post-launch.

---

## Part 6 — Sequencing

1. Launch (Phase 8) at ~$1k/user cap. Run for 60 days. Collect real behaviour.
2. Ship 10.1 (veto layer) + 10.4 (journaling) first — they work with zero accumulated history.
3. Ship 10.2 (episodic memory) once there are ≥30 days of trade history to recall from.
4. Ship 10.3 (pattern detection) + 10.5 (weekly report) last — they benefit most from accumulated data.

---

**End of proposal.**
