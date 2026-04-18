"""
Discipline check adapter — Claude-backed implementation of the pre-trade
veto port (Phase 10.1).

Architectural Intent:
- Implements `DisciplineCheckPort` by asking Claude Haiku to evaluate the
  proposed order against every rule in `user.discipline_rules` plus the
  `user.trading_philosophy` paragraph. We ask for a structured JSON
  response, parse strictly, and reject any malformed output to a clean
  result (fail-open with a logged error) rather than blocking the order
  on a model misbehaviour.
- Users who have configured nothing (empty `discipline_rules`, None
  `trading_philosophy`) bypass the API call entirely — no cost, no
  latency, no audit noise.
- Timeouts are aggressive: the order flow is user-facing and sync, so we
  cap at 6 seconds total. If the check times out, we fail-open with a
  logged warning. The deterministic guards in `validate_order` still
  catch the dangerous cases.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import anthropic

from src.domain.entities.trading import Order, Portfolio
from src.domain.entities.user import User
from src.domain.ports.discipline_check import (
    DisciplineCheckPort,
    DisciplineCheckResult,
    DisciplineVeto,
)
from src.infrastructure.config.settings import settings

logger = logging.getLogger(__name__)


_CHECK_MODEL = "claude-haiku-4-5-20251001"

_SYSTEM_PROMPT = """\
You are the TraderBot pre-trade discipline coach. Your job is to check whether
a proposed order violates any of the user's own stated trading rules or their
trading philosophy.

You will be given:
- The proposed order (symbol, action, quantity, price, order type)
- The user's current portfolio snapshot
- The user's numbered discipline rules (self-written, in their own words)
- The user's trading philosophy (free-form paragraph)

Return a JSON object with this exact shape:
{
  "vetoes": [
    {
      "rule_id": "1" | "2" | ... | "philosophy",
      "rule_text": "the literal rule text",
      "evidence": "one sentence explaining why this order violates the rule"
    }
  ]
}

Rules:
- Only veto if the proposed order clearly contradicts a rule or the philosophy.
- Do not invent rules the user did not state.
- Do not veto based on your own opinion of whether the trade is wise — only
  enforce what the user said they want enforced.
- If no rules are violated, return {"vetoes": []}.
- Return only the JSON. No prose, no markdown.
"""


class ClaudeDisciplineCheckAdapter(DisciplineCheckPort):
    """Claude-backed pre-trade veto evaluator."""

    def __init__(self, api_key: str | None = None, model: str = _CHECK_MODEL):
        self._model = model
        self._client = anthropic.Anthropic(
            api_key=api_key or settings.ANTHROPIC_API_KEY,
            # Total budget for the check is 6s; connect guard is tighter so a
            # dead endpoint fails fast rather than blocking the order path.
            timeout=anthropic.Timeout(6.0, connect=2.0),
            max_retries=0,
        )

    def check(
        self,
        user: User,
        order: Order,
        portfolio: Portfolio,
    ) -> DisciplineCheckResult:
        if not user.discipline_rules and not user.trading_philosophy:
            # Common case — user hasn't configured anything. Skip the API call.
            return DisciplineCheckResult.clean()

        try:
            user_payload = self._build_user_payload(user, order, portfolio)
        except Exception:
            logger.exception("discipline_check_payload_build_failed user_id=%s", user.id)
            return DisciplineCheckResult.clean()

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=512,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_payload}],
            )
        except anthropic.APIError as exc:
            logger.warning(
                "discipline_check_api_error user_id=%s type=%s — failing open",
                user.id, type(exc).__name__,
            )
            return DisciplineCheckResult.clean()
        except Exception:
            logger.exception(
                "discipline_check_unexpected_error user_id=%s — failing open", user.id,
            )
            return DisciplineCheckResult.clean()

        return self._parse_response(response, user)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_user_payload(
        self, user: User, order: Order, portfolio: Portfolio,
    ) -> str:
        """Render the user-side prompt as a single JSON-ish block.

        Keeping the shape stable (rather than natural language) lets the
        model reason about the fields without needing to parse sentences.
        """
        rules_block = "\n".join(
            f"{i}. {rule}" for i, rule in enumerate(user.discipline_rules, start=1)
        ) or "(none)"

        portfolio_block = {
            "cash_balance": str(portfolio.cash_balance.amount),
            "num_positions": len(portfolio.positions),
            "positions": [
                {
                    "symbol": str(p.symbol),
                    "quantity": p.quantity,
                    "avg_buy_price": str(p.average_buy_price.amount),
                }
                for p in portfolio.positions
            ],
        }

        order_block = {
            "symbol": str(order.symbol),
            "action": "BUY" if order.position_type.value == "LONG" else "SELL",
            "quantity": order.quantity,
            "order_type": order.order_type.value,
            "price": str(order.price.amount) if order.price else None,
        }

        return (
            f"PROPOSED ORDER:\n{json.dumps(order_block, indent=2)}\n\n"
            f"PORTFOLIO SNAPSHOT:\n{json.dumps(portfolio_block, indent=2)}\n\n"
            f"USER DISCIPLINE RULES:\n{rules_block}\n\n"
            f"USER TRADING PHILOSOPHY:\n{user.trading_philosophy or '(none)'}"
        )

    def _parse_response(
        self, response: Any, user: User,
    ) -> DisciplineCheckResult:
        """Extract text, parse JSON, validate, and build the result.

        Any parse error is logged and treated as fail-open — we don't want
        a model misbehaviour to block legitimate orders. The deterministic
        guards upstream still catch dangerous cases.
        """
        try:
            text = "".join(
                block.text for block in response.content
                if getattr(block, "type", None) == "text"
            )
        except Exception:
            logger.warning(
                "discipline_check_response_shape_unexpected user_id=%s", user.id,
            )
            return DisciplineCheckResult.clean()

        try:
            parsed = json.loads(text.strip())
        except json.JSONDecodeError:
            logger.warning(
                "discipline_check_response_not_json user_id=%s preview=%r",
                user.id, text[:200],
            )
            return DisciplineCheckResult.clean()

        raw_vetoes = parsed.get("vetoes") if isinstance(parsed, dict) else None
        if not isinstance(raw_vetoes, list):
            logger.warning(
                "discipline_check_response_bad_shape user_id=%s payload=%r",
                user.id, parsed,
            )
            return DisciplineCheckResult.clean()

        vetoes: List[DisciplineVeto] = []
        for raw in raw_vetoes:
            if not isinstance(raw, dict):
                continue
            rule_id = str(raw.get("rule_id", "")).strip()
            rule_text = str(raw.get("rule_text", "")).strip()
            evidence = str(raw.get("evidence", "")).strip()
            if not rule_id or not rule_text or not evidence:
                continue
            vetoes.append(
                DisciplineVeto(
                    rule_id=rule_id, rule_text=rule_text, evidence=evidence,
                )
            )

        if vetoes:
            logger.info(
                "discipline_check_vetoed user_id=%s count=%d", user.id, len(vetoes),
            )
        return DisciplineCheckResult(approved=not vetoes, vetoes=vetoes)
