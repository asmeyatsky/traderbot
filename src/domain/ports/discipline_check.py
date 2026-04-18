"""
Discipline check port — pre-trade veto layer (Phase 10.1).

Architectural Intent:
- Every order, whether chat-initiated or UI-initiated, passes through this
  port BEFORE reaching the broker routing layer. The check evaluates the
  proposed order against the user's stated discipline rules and trading
  philosophy; if any rule is violated, the check returns a structured
  veto that the caller surfaces to the user.
- The deterministic guards (position-size cap, sector exclusions) already
  live in `TradingDomainService.validate_order`. This port is specifically
  for the things the user has written in their own words — free-form
  rules and philosophy — which require an LLM to evaluate.
- Vetoes are overridable by explicit user confirmation (the router gates
  this with a dedicated flag; every override emits an audit event). The
  port itself does not decide whether to override; it only reports what
  rules were violated.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from src.domain.entities.trading import Order, Portfolio
from src.domain.entities.user import User


@dataclass(frozen=True)
class DisciplineVeto:
    """A single rule violation.

    `rule_id` is a stable identifier: for user-written rules it is the
    1-based index into `user.discipline_rules`; for philosophy checks it is
    the literal string 'philosophy'.
    `rule_text` is the human-readable rule the user wrote.
    `evidence` is a one-sentence explanation of why the proposed order
    violates the rule, grounded in the order details (symbol, quantity,
    action, portfolio state).
    """
    rule_id: str
    rule_text: str
    evidence: str


@dataclass(frozen=True)
class DisciplineCheckResult:
    """Outcome of a discipline check.

    `approved=True` with an empty `vetoes` list means the order may proceed.
    `approved=False` means the caller MUST refuse the order unless the user
    has explicitly opted to override — in which case the override path
    bypasses this check entirely and emits an audit event with the
    violated rule IDs attached.
    """
    approved: bool
    vetoes: List[DisciplineVeto]

    @classmethod
    def clean(cls) -> 'DisciplineCheckResult':
        """Convenience: no rules configured, no checks needed, always approve."""
        return cls(approved=True, vetoes=[])


class DisciplineCheckPort(ABC):
    """Evaluate a proposed order against the user's discipline rules."""

    @abstractmethod
    def check(
        self,
        user: User,
        order: Order,
        portfolio: Portfolio,
    ) -> DisciplineCheckResult:
        """Return a result describing whether the order should proceed.

        Implementations should fail-open (return `DisciplineCheckResult.clean()`)
        on transient errors — the deterministic guards still catch the
        dangerous cases, and refusing every order when the LLM is down would
        be a denial-of-service on users with rules configured. Errors MUST
        be logged so operators can see the check isn't running.
        """
