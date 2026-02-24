"""
Position Entity Math Tests

Ensures all Position computations produce valid Money values (2dp)
even when source prices have 4 decimal places.
"""
import pytest
from datetime import datetime
from decimal import Decimal

from src.domain.entities.trading import Position, PositionType
from src.domain.value_objects import Money, Symbol


def _money(value: str) -> Money:
    return Money(Decimal(value), "USD")


def _make_position(**overrides) -> Position:
    defaults = dict(
        id="pos-1",
        user_id="user-1",
        symbol=Symbol("AAPL"),
        position_type=PositionType.LONG,
        quantity=100,
        average_buy_price=_money("150.00"),
        current_price=_money("155.00"),
        created_at=datetime(2025, 1, 1),
        updated_at=datetime(2025, 1, 1),
    )
    defaults.update(overrides)
    return Position(**defaults)


class TestMarketValue:
    def test_market_value_quantized_to_2dp(self):
        """market_value must always be 2dp Money, even with odd price."""
        pos = _make_position(current_price=_money("33.33"), quantity=3)
        mv = pos.market_value
        assert mv.amount == Decimal("99.99")
        assert mv.amount.as_tuple().exponent >= -2

    def test_market_value_standard(self):
        pos = _make_position(current_price=_money("155.00"), quantity=100)
        assert pos.market_value.amount == Decimal("15500.00")


class TestTotalCost:
    def test_total_cost_quantized_to_2dp(self):
        pos = _make_position(average_buy_price=_money("33.33"), quantity=3)
        tc = pos.total_cost
        assert tc.amount == Decimal("99.99")
        assert tc.amount.as_tuple().exponent >= -2

    def test_total_cost_standard(self):
        pos = _make_position(average_buy_price=_money("150.00"), quantity=100)
        assert pos.total_cost.amount == Decimal("15000.00")


class TestAdjustQuantity:
    def test_add_shares_computes_average_to_2dp(self):
        pos = _make_position(
            average_buy_price=_money("150.00"), quantity=100
        )
        updated = pos.adjust_quantity(50, _money("160.00"))
        assert updated.quantity == 150
        # avg = (150*100 + 160*50) / 150 = 23000/150 = 153.33...
        assert updated.average_buy_price.amount == Decimal("153.33")
        assert updated.average_buy_price.amount.as_tuple().exponent >= -2

    def test_close_position_returns_zero_quantity(self):
        pos = _make_position(quantity=100)
        closed = pos.adjust_quantity(-100, _money("160.00"))
        assert closed.quantity == 0
        assert closed.average_buy_price.amount == Decimal("0")

    def test_partial_sell_reduces_quantity(self):
        pos = _make_position(quantity=100, average_buy_price=_money("150.00"))
        updated = pos.adjust_quantity(-30, _money("160.00"))
        assert updated.quantity == 70


class TestUnrealizedPnl:
    def test_unrealized_pnl_calculated(self):
        pos = _make_position(
            average_buy_price=_money("150.00"),
            current_price=_money("160.00"),
            quantity=10,
        )
        pnl = pos.unrealized_pnl_amount
        # (1600.00 - 1500.00) = 100.00
        assert pnl.amount == Decimal("100.00")

    def test_unrealized_pnl_override(self):
        """When unrealized_pnl is explicitly set, the property returns it."""
        override = _money("42.00")
        pos = _make_position(unrealized_pnl=override)
        assert pos.unrealized_pnl_amount is override

    def test_unrealized_pnl_negative(self):
        pos = _make_position(
            average_buy_price=_money("160.00"),
            current_price=_money("150.00"),
            quantity=10,
        )
        pnl = pos.unrealized_pnl_amount
        assert pnl.amount == Decimal("-100.00")


class TestUpdatePrice:
    def test_returns_new_instance(self):
        pos = _make_position()
        new = pos.update_price(_money("200.00"))
        assert new is not pos
        assert new.current_price.amount == Decimal("200.00")
        assert pos.current_price.amount == Decimal("155.00")  # original unchanged
