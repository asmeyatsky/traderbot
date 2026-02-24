"""
Money/Price Value Object Boundary Tests

Regression suite for Bug #1: Decimal precision overflow.
yfinance returns prices with 4 decimal places, but Money validates
at max 2dp. These tests ensure the boundary between Price (4dp)
and Money (2dp) is handled correctly throughout the codebase.
"""
import pytest
from decimal import Decimal

from src.domain.value_objects import Money, Price


class TestMoneyDecimalValidation:
    """Money must reject amounts with more than 2 decimal places."""

    def test_money_rejects_more_than_2dp(self):
        with pytest.raises(ValueError, match="maximum 2 decimal places"):
            Money(Decimal("1.234"), "USD")

    def test_money_accepts_exact_2dp(self):
        m = Money(Decimal("1.23"), "USD")
        assert m.amount == Decimal("1.23")

    def test_money_accepts_integer(self):
        m = Money(Decimal("100"), "USD")
        assert m.amount == Decimal("100")

    def test_money_accepts_1dp(self):
        m = Money(Decimal("9.5"), "USD")
        assert m.amount == Decimal("9.5")


class TestPriceDecimalValidation:
    """Price allows up to 4 decimal places."""

    def test_price_allows_up_to_4dp(self):
        p = Price(Decimal("1.2345"), "USD")
        assert p.amount == Decimal("1.2345")

    def test_price_rejects_more_than_4dp(self):
        with pytest.raises(ValueError, match="maximum 4 decimal places"):
            Price(Decimal("1.23456"), "USD")

    def test_price_accepts_2dp(self):
        p = Price(Decimal("150.25"), "USD")
        assert p.amount == Decimal("150.25")


class TestPriceToMoneyConversion:
    """Converting Price → Money requires explicit quantization to 2dp."""

    def test_price_to_money_requires_quantize(self):
        price = Price(Decimal("150.1234"), "USD")
        # Direct construction without quantize must fail
        with pytest.raises(ValueError, match="maximum 2 decimal places"):
            Money(price.amount, "USD")

    def test_price_to_money_quantized_succeeds(self):
        price = Price(Decimal("150.1234"), "USD")
        money = Money(price.amount.quantize(Decimal("0.01")), "USD")
        assert money.amount == Decimal("150.12")

    def test_multiplication_increases_decimal_places(self):
        """Decimal multiplication can produce more dp than either operand."""
        a = Decimal("1.2345")  # 4dp
        b = Decimal("1.23")    # 2dp
        result = a * b
        # Product can have up to 6dp — must quantize before Money()
        assert result.as_tuple().exponent <= -4
        with pytest.raises(ValueError):
            Money(result, "USD")

    def test_price_times_quantity_quantized_to_money(self):
        """The standard pattern: (price * qty).quantize() → Money."""
        price = Price(Decimal("150.1234"), "USD")
        qty = 10
        raw = price.amount * Decimal(qty)  # 1501.2340 — 4dp
        money = Money(raw.quantize(Decimal("0.01")), "USD")
        assert money.amount == Decimal("1501.23")
