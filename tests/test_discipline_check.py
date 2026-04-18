"""
Tests for the pre-trade discipline veto layer (Phase 10.1).

Covers:
1. The port contract (DisciplineCheckResult semantics).
2. CreateOrderUseCase integration — veto raises DisciplineVetoError, override
   flag bypasses the check, clean result proceeds, missing port proceeds.
3. ClaudeDisciplineCheckAdapter fail-open behaviour on common failure modes
   (API error, malformed JSON, no rules configured).

The LLM itself is mocked — we assert the contract between the adapter and
Anthropic, not the model's judgement.
"""
from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.application.use_cases.trading import (
    CreateOrderUseCase,
    DisciplineVetoError,
)
from src.domain.entities.trading import (
    Order, OrderStatus, OrderType, Portfolio, PositionType,
)
from src.domain.entities.user import (
    InvestmentGoal, RiskTolerance, User,
)
from src.domain.ports.discipline_check import (
    DisciplineCheckPort, DisciplineCheckResult, DisciplineVeto,
)
from src.domain.value_objects import Money, Price, Symbol


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _money(v: str) -> Money:
    return Money(Decimal(v), "USD")


def _price(v: str) -> Price:
    return Price(Decimal(v), "USD")


def _user(**over) -> User:
    base = dict(
        id="u-1",
        email="a@b.com",
        first_name="A",
        last_name="B",
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 1),
        risk_tolerance=RiskTolerance.MODERATE,
        investment_goal=InvestmentGoal.BALANCED_GROWTH,
        max_position_size_percentage=Decimal("100"),
    )
    base.update(over)
    return User(**base)


def _portfolio() -> Portfolio:
    return Portfolio(
        id="p-1",
        user_id="u-1",
        positions=[],
        cash_balance=_money("10000"),
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 1),
    )


def _build_uc(discipline_check=None):
    user = _user()

    user_repo = MagicMock()
    user_repo.get_by_id.return_value = user

    port_repo = MagicMock()
    port_repo.get_by_user_id.return_value = _portfolio()

    market_data = MagicMock()
    market_data.get_current_price.return_value = _price("150")

    order_repo = MagicMock()
    order_repo.save.side_effect = lambda o: o

    position_repo = MagicMock()
    position_repo.get_by_symbol.return_value = None
    position_repo.save.side_effect = lambda p: p

    from src.domain.services.trading import DefaultTradingDomainService
    return CreateOrderUseCase(
        order_repository=order_repo,
        portfolio_repository=port_repo,
        user_repository=user_repo,
        trading_service=DefaultTradingDomainService(),
        market_data_service=market_data,
        position_repository=position_repo,
        discipline_check=discipline_check,
    )


# ---------------------------------------------------------------------------
# Port contract
# ---------------------------------------------------------------------------


class TestResultSemantics:
    def test_clean_is_approved_with_no_vetoes(self):
        r = DisciplineCheckResult.clean()
        assert r.approved is True
        assert r.vetoes == []

    def test_approved_false_implies_vetoes_present(self):
        v = DisciplineVeto(rule_id="1", rule_text="no", evidence="because")
        r = DisciplineCheckResult(approved=False, vetoes=[v])
        assert len(r.vetoes) == 1
        assert r.vetoes[0].rule_id == "1"


# ---------------------------------------------------------------------------
# CreateOrderUseCase integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_no_check_port_proceeds(self):
        """Backward-compatible: if nobody wires a check port, orders flow."""
        uc = _build_uc(discipline_check=None)
        order = uc.execute("u-1", Symbol("AAPL"), "MARKET", "LONG", 10)
        assert order.status == OrderStatus.EXECUTED

    def test_clean_result_proceeds(self):
        check = MagicMock(spec=DisciplineCheckPort)
        check.check.return_value = DisciplineCheckResult.clean()

        uc = _build_uc(discipline_check=check)
        order = uc.execute("u-1", Symbol("AAPL"), "MARKET", "LONG", 10)

        check.check.assert_called_once()
        assert order.status == OrderStatus.EXECUTED

    def test_veto_raises_discipline_veto_error(self):
        veto = DisciplineVeto(
            rule_id="1",
            rule_text="Never buy AAPL on a Monday",
            evidence="Today is a Monday and the symbol is AAPL.",
        )
        check = MagicMock(spec=DisciplineCheckPort)
        check.check.return_value = DisciplineCheckResult(approved=False, vetoes=[veto])

        uc = _build_uc(discipline_check=check)

        with pytest.raises(DisciplineVetoError) as exc_info:
            uc.execute("u-1", Symbol("AAPL"), "MARKET", "LONG", 10)

        # The exception must carry the structured vetoes the router needs to
        # render the "break this rule?" modal.
        assert len(exc_info.value.vetoes) == 1
        assert exc_info.value.vetoes[0].rule_id == "1"
        assert "AAPL on a Monday" in str(exc_info.value)

    def test_override_flag_bypasses_check(self):
        """Router gates this flag behind explicit user confirmation + audit."""
        check = MagicMock(spec=DisciplineCheckPort)

        uc = _build_uc(discipline_check=check)
        order = uc.execute(
            "u-1", Symbol("AAPL"), "MARKET", "LONG", 10,
            override_discipline_vetoes=True,
        )

        # When the override flag is set we MUST NOT even call the port —
        # otherwise the model burns tokens deciding something the user
        # explicitly opted to override.
        check.check.assert_not_called()
        assert order.status == OrderStatus.EXECUTED


# ---------------------------------------------------------------------------
# ClaudeDisciplineCheckAdapter — contract tests (model is mocked)
# ---------------------------------------------------------------------------


def _adapter():
    """Construct the adapter with a mocked Anthropic client so the constructor
    doesn't need a live API key."""
    from src.infrastructure.adapters.discipline_check import (
        ClaudeDisciplineCheckAdapter,
    )
    a = ClaudeDisciplineCheckAdapter.__new__(ClaudeDisciplineCheckAdapter)
    a._model = "claude-haiku-4-5-20251001"
    a._client = MagicMock()
    return a


def _order(symbol: str = "AAPL", qty: int = 10) -> Order:
    return Order(
        id="o-1",
        user_id="u-1",
        symbol=Symbol(symbol),
        order_type=OrderType.MARKET,
        position_type=PositionType.LONG,
        quantity=qty,
        status=OrderStatus.PENDING,
        placed_at=datetime(2026, 4, 18),
        price=_price("150"),
    )


class TestClaudeAdapterBehaviour:
    def test_empty_rules_and_no_philosophy_skips_api(self):
        a = _adapter()
        user = _user(discipline_rules=[], trading_philosophy=None)

        result = a.check(user, _order(), _portfolio())

        assert result.approved is True
        # Critical: don't burn API tokens when there's nothing to check.
        a._client.messages.create.assert_not_called()

    def test_malformed_json_fails_open(self):
        a = _adapter()
        user = _user(discipline_rules=["No AAPL"])
        a._client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="this is not JSON")],
        )

        result = a.check(user, _order(), _portfolio())

        # Fail-open: rather than block orders because the model hallucinated
        # prose, we approve and rely on the deterministic upstream guards.
        assert result.approved is True

    def test_api_error_fails_open(self):
        import anthropic
        a = _adapter()
        user = _user(discipline_rules=["No AAPL"])
        # APIError requires specific args; APIConnectionError is a simpler
        # subclass to instantiate for this test.
        a._client.messages.create.side_effect = anthropic.APIConnectionError(
            message="down", request=MagicMock(),
        )

        result = a.check(user, _order(), _portfolio())
        assert result.approved is True

    def test_structured_veto_is_parsed(self):
        a = _adapter()
        user = _user(discipline_rules=["Never buy AAPL on Mondays"])
        body = json.dumps({
            "vetoes": [
                {
                    "rule_id": "1",
                    "rule_text": "Never buy AAPL on Mondays",
                    "evidence": "Today is Monday and the symbol is AAPL.",
                }
            ]
        })
        a._client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text=body)],
        )

        result = a.check(user, _order(), _portfolio())

        assert result.approved is False
        assert len(result.vetoes) == 1
        assert result.vetoes[0].rule_id == "1"

    def test_empty_vetoes_list_is_approval(self):
        a = _adapter()
        user = _user(discipline_rules=["Never buy AAPL on Mondays"])
        a._client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text='{"vetoes": []}')],
        )

        result = a.check(user, _order(), _portfolio())

        assert result.approved is True
        assert result.vetoes == []

    def test_partial_veto_entries_are_dropped(self):
        """Vetoes missing any of rule_id / rule_text / evidence are ignored —
        we'd rather approve on a malformed model response than cite an
        incomplete reason to the user."""
        a = _adapter()
        user = _user(discipline_rules=["No AAPL"])
        body = json.dumps({
            "vetoes": [
                {"rule_id": "1"},  # missing rule_text and evidence
                {"rule_id": "", "rule_text": "x", "evidence": "y"},  # blank rule_id
            ]
        })
        a._client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text=body)],
        )

        result = a.check(user, _order(), _portfolio())
        assert result.approved is True
        assert result.vetoes == []
