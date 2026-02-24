"""
CreateOrderUseCase — Full Order Flow Tests

End-to-end regression suite covering all 3 production bugs:
1. Decimal precision overflow (Price 4dp → Money 2dp)
2. Weekend market data gap (price unavailable)
3. Cash validation inverted (LONG orders must check cash)
"""
import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock, call

from src.application.use_cases.trading import CreateOrderUseCase
from src.domain.entities.trading import (
    Order, Position, Portfolio, OrderType, PositionType, OrderStatus,
)
from src.domain.entities.user import User, RiskTolerance, InvestmentGoal
from src.domain.value_objects import Money, Symbol, Price


def _money(value: str) -> Money:
    return Money(Decimal(value), "USD")


def _price(value: str) -> Price:
    return Price(Decimal(value), "USD")


def _make_user(**overrides) -> User:
    defaults = dict(
        id="user-1",
        email="test@example.com",
        first_name="Test",
        last_name="User",
        created_at=datetime(2025, 1, 1),
        updated_at=datetime(2025, 1, 1),
        risk_tolerance=RiskTolerance.MODERATE,
        investment_goal=InvestmentGoal.BALANCED_GROWTH,
        max_position_size_percentage=Decimal("100"),
    )
    defaults.update(overrides)
    return User(**defaults)


def _make_portfolio(**overrides) -> Portfolio:
    defaults = dict(
        id="port-1",
        user_id="user-1",
        positions=[],
        cash_balance=_money("10000.00"),
        created_at=datetime(2025, 1, 1),
        updated_at=datetime(2025, 1, 1),
    )
    defaults.update(overrides)
    return Portfolio(**defaults)


def _build_use_case(
    user=None,
    portfolio=None,
    current_price=None,
    existing_position=None,
    include_position_repo=True,
):
    """Wire up CreateOrderUseCase with mocked ports."""
    user = user or _make_user()
    portfolio = portfolio or _make_portfolio()
    current_price = current_price or _price("150.00")

    user_repo = MagicMock()
    user_repo.get_by_id.return_value = user

    portfolio_repo = MagicMock()
    portfolio_repo.get_by_user_id.return_value = portfolio

    market_data = MagicMock()
    market_data.get_current_price.return_value = current_price

    # Trading service — use real implementation for integration-style tests
    from src.domain.services.trading import DefaultTradingDomainService
    trading_service = DefaultTradingDomainService()

    order_repo = MagicMock()
    order_repo.save.side_effect = lambda o: o  # pass-through

    position_repo = None
    if include_position_repo:
        position_repo = MagicMock()
        position_repo.get_by_symbol.return_value = existing_position
        position_repo.save.side_effect = lambda p: p
        position_repo.update.side_effect = lambda p: p

    uc = CreateOrderUseCase(
        order_repository=order_repo,
        portfolio_repository=portfolio_repo,
        user_repository=user_repo,
        trading_service=trading_service,
        market_data_service=market_data,
        position_repository=position_repo,
    )
    return uc, {
        "order_repo": order_repo,
        "portfolio_repo": portfolio_repo,
        "position_repo": position_repo,
        "market_data": market_data,
        "user_repo": user_repo,
    }


class TestMarketOrderAutoFill:
    def test_market_order_status_executed(self):
        uc, _ = _build_use_case()
        order = uc.execute(
            user_id="user-1",
            symbol=Symbol("AAPL"),
            order_type="MARKET",
            position_type="LONG",
            quantity=10,
        )
        assert order.status == OrderStatus.EXECUTED
        assert order.filled_quantity == 10

    def test_limit_order_stays_pending(self):
        uc, _ = _build_use_case()
        order = uc.execute(
            user_id="user-1",
            symbol=Symbol("AAPL"),
            order_type="LIMIT",
            position_type="LONG",
            quantity=10,
            limit_price=145.00,
        )
        assert order.status == OrderStatus.PENDING
        assert order.filled_quantity == 0


class TestSettleFill:
    def test_creates_position_for_new_long(self):
        uc, mocks = _build_use_case()
        uc.execute(
            user_id="user-1",
            symbol=Symbol("AAPL"),
            order_type="MARKET",
            position_type="LONG",
            quantity=10,
        )
        mocks["position_repo"].save.assert_called_once()
        saved_pos = mocks["position_repo"].save.call_args[0][0]
        assert saved_pos.quantity == 10
        assert saved_pos.average_buy_price.amount == Decimal("150.00")

    def test_updates_existing_position(self):
        existing = Position(
            id="pos-1",
            user_id="user-1",
            symbol=Symbol("AAPL"),
            position_type=PositionType.LONG,
            quantity=10,
            average_buy_price=_money("140.00"),
            current_price=_money("140.00"),
            created_at=datetime(2025, 1, 1),
            updated_at=datetime(2025, 1, 1),
        )
        uc, mocks = _build_use_case(existing_position=existing)
        uc.execute(
            user_id="user-1",
            symbol=Symbol("AAPL"),
            order_type="MARKET",
            position_type="LONG",
            quantity=10,
        )
        mocks["position_repo"].update.assert_called_once()
        updated_pos = mocks["position_repo"].update.call_args[0][0]
        assert updated_pos.quantity == 20
        # avg = (140*10 + 150*10) / 20 = 145.00
        assert updated_pos.average_buy_price.amount == Decimal("145.00")

    def test_deducts_cash_for_long(self):
        portfolio = _make_portfolio(cash_balance=_money("10000.00"))
        uc, mocks = _build_use_case(portfolio=portfolio)
        uc.execute(
            user_id="user-1",
            symbol=Symbol("AAPL"),
            order_type="MARKET",
            position_type="LONG",
            quantity=10,
        )
        mocks["portfolio_repo"].update.assert_called_once()
        updated_port = mocks["portfolio_repo"].update.call_args[0][0]
        # 10000 - (150 * 10) = 8500
        assert updated_port.cash_balance.amount == Decimal("8500.00")

    def test_short_order_adds_cash(self):
        portfolio = _make_portfolio(cash_balance=_money("5000.00"))
        uc, mocks = _build_use_case(portfolio=portfolio)
        uc.execute(
            user_id="user-1",
            symbol=Symbol("AAPL"),
            order_type="MARKET",
            position_type="SHORT",
            quantity=10,
        )
        mocks["portfolio_repo"].update.assert_called_once()
        updated_port = mocks["portfolio_repo"].update.call_args[0][0]
        # 5000 + (150 * 10) = 6500
        assert updated_port.cash_balance.amount == Decimal("6500.00")


class TestSettleFillPrecision:
    """Regression for Bug #1: 4dp Price must be quantized to 2dp Money."""

    def test_fill_price_quantized_to_2dp(self):
        """Even with a 4dp market price, position gets 2dp Money."""
        uc, mocks = _build_use_case(current_price=_price("150.1234"))
        uc.execute(
            user_id="user-1",
            symbol=Symbol("AAPL"),
            order_type="MARKET",
            position_type="LONG",
            quantity=10,
        )
        saved_pos = mocks["position_repo"].save.call_args[0][0]
        assert saved_pos.average_buy_price.amount == Decimal("150.12")
        assert saved_pos.average_buy_price.amount.as_tuple().exponent >= -2

    def test_trade_value_quantized_to_2dp(self):
        """trade_value (price * qty) must be 2dp for cash balance update."""
        uc, mocks = _build_use_case(current_price=_price("150.1234"))
        uc.execute(
            user_id="user-1",
            symbol=Symbol("AAPL"),
            order_type="MARKET",
            position_type="LONG",
            quantity=10,
        )
        updated_port = mocks["portfolio_repo"].update.call_args[0][0]
        cash = updated_port.cash_balance.amount
        assert cash.as_tuple().exponent >= -2


class TestErrorCases:
    def test_user_not_found_raises(self):
        uc, mocks = _build_use_case()
        mocks["user_repo"].get_by_id.return_value = None
        with pytest.raises(ValueError, match="User not found"):
            uc.execute("bad-id", Symbol("AAPL"), "MARKET", "LONG", 10)

    def test_portfolio_not_found_raises(self):
        uc, mocks = _build_use_case()
        mocks["portfolio_repo"].get_by_user_id.return_value = None
        with pytest.raises(ValueError, match="Portfolio not found"):
            uc.execute("user-1", Symbol("AAPL"), "MARKET", "LONG", 10)

    def test_price_unavailable_raises(self):
        uc, mocks = _build_use_case()
        mocks["market_data"].get_current_price.return_value = None
        with pytest.raises(ValueError, match="Could not get current price"):
            uc.execute("user-1", Symbol("AAPL"), "MARKET", "LONG", 10)

    def test_validation_failure_raises(self):
        """Insufficient cash → ValueError with descriptive message."""
        portfolio = _make_portfolio(cash_balance=_money("100.00"))
        uc, _ = _build_use_case(portfolio=portfolio)
        with pytest.raises(ValueError, match="Insufficient cash"):
            uc.execute("user-1", Symbol("AAPL"), "MARKET", "LONG", 100)

    def test_no_position_repo_skips_settlement(self):
        """When position_repository=None, order is saved but no position created."""
        uc, mocks = _build_use_case(include_position_repo=False)
        order = uc.execute(
            user_id="user-1",
            symbol=Symbol("AAPL"),
            order_type="MARKET",
            position_type="LONG",
            quantity=10,
        )
        assert order.status == OrderStatus.EXECUTED
        mocks["order_repo"].save.assert_called_once()
        # No position repo means no position or portfolio updates
        mocks["portfolio_repo"].update.assert_not_called()
