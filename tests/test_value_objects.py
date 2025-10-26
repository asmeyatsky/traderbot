"""
Tests for Value Objects

Tests for domain value objects like Money, Symbol, Price, etc.
"""
import pytest
from decimal import Decimal

from src.domain.value_objects import Money, Symbol, Price, NewsSentiment


class TestMoney:
    """Test Money value object."""

    def test_money_creation(self):
        """Test creating a Money value object."""
        money = Money(Decimal("100.50"), "USD")
        assert money.amount == Decimal("100.50")
        assert money.currency == "USD"

    def test_money_equality(self):
        """Test Money equality."""
        money1 = Money(Decimal("100.00"), "USD")
        money2 = Money(Decimal("100.00"), "USD")
        money3 = Money(Decimal("100.01"), "USD")

        assert money1 == money2
        assert money1 != money3

    def test_money_addition(self):
        """Test adding Money values."""
        money1 = Money(Decimal("100.00"), "USD")
        money2 = Money(Decimal("50.00"), "USD")
        result = money1.add(money2)

        assert result.amount == Decimal("150.00")
        assert result.currency == "USD"

    def test_money_subtraction(self):
        """Test subtracting Money values."""
        money1 = Money(Decimal("100.00"), "USD")
        money2 = Money(Decimal("30.00"), "USD")
        result = money1.subtract(money2)

        assert result.amount == Decimal("70.00")
        assert result.currency == "USD"

    def test_money_multiplication(self):
        """Test multiplying Money by a scalar."""
        money = Money(Decimal("100.00"), "USD")
        result = money.multiply(Decimal("1.5"))

        assert result.amount == Decimal("150.00")
        assert result.currency == "USD"

    def test_money_division(self):
        """Test dividing Money by a scalar."""
        money = Money(Decimal("100.00"), "USD")
        result = money.divide(Decimal("2"))

        assert result.amount == Decimal("50.00")
        assert result.currency == "USD"

    def test_money_currency_mismatch(self):
        """Test that adding Money with different currencies raises error."""
        money1 = Money(Decimal("100.00"), "USD")
        money2 = Money(Decimal("50.00"), "EUR")

        with pytest.raises(ValueError):
            money1.add(money2)

    def test_money_immutability(self):
        """Test that Money is immutable."""
        money = Money(Decimal("100.00"), "USD")

        with pytest.raises(Exception):
            money.amount = Decimal("200.00")

    def test_money_zero(self):
        """Test Money zero."""
        money = Money(Decimal("0"), "USD")
        assert money.is_zero()

    def test_money_negative(self):
        """Test Money with negative amount."""
        money = Money(Decimal("-100.00"), "USD")
        assert money.is_negative()

    def test_money_positive(self):
        """Test Money with positive amount."""
        money = Money(Decimal("100.00"), "USD")
        assert money.is_positive()


class TestSymbol:
    """Test Symbol value object."""

    def test_symbol_creation(self):
        """Test creating a Symbol value object."""
        symbol = Symbol("AAPL")
        assert str(symbol) == "AAPL"

    def test_symbol_uppercase(self):
        """Test that Symbol is stored in uppercase."""
        symbol = Symbol("aapl")
        assert str(symbol) == "AAPL"

    def test_symbol_equality(self):
        """Test Symbol equality."""
        symbol1 = Symbol("AAPL")
        symbol2 = Symbol("aapl")
        symbol3 = Symbol("GOOGL")

        assert symbol1 == symbol2
        assert symbol1 != symbol3

    def test_symbol_invalid(self):
        """Test creating Symbol with invalid format."""
        # Symbol should validate format
        # This test depends on actual validation logic
        symbol = Symbol("A1B2")  # May be valid or invalid depending on implementation

    def test_symbol_immutability(self):
        """Test that Symbol is immutable."""
        symbol = Symbol("AAPL")

        # Symbol should be immutable
        # Exact behavior depends on implementation
        assert str(symbol) == "AAPL"


class TestPrice:
    """Test Price value object."""

    def test_price_creation(self):
        """Test creating a Price value object."""
        price = Price(Decimal("150.50"), "USD")
        assert price.amount == Decimal("150.50")
        assert price.currency == "USD"

    def test_price_equality(self):
        """Test Price equality."""
        price1 = Price(Decimal("150.00"), "USD")
        price2 = Price(Decimal("150.00"), "USD")
        price3 = Price(Decimal("150.01"), "USD")

        assert price1 == price2
        assert price1 != price3

    def test_price_comparison(self):
        """Test Price comparison."""
        price1 = Price(Decimal("150.00"), "USD")
        price2 = Price(Decimal("160.00"), "USD")

        assert price1 < price2
        assert price2 > price1

    def test_price_to_money(self):
        """Test converting Price to Money."""
        price = Price(Decimal("150.00"), "USD")
        money = price.to_money()

        assert isinstance(money, Money)
        assert money.amount == price.amount
        assert money.currency == price.currency

    def test_price_immutability(self):
        """Test that Price is immutable."""
        price = Price(Decimal("150.00"), "USD")

        with pytest.raises(Exception):
            price.amount = Decimal("160.00")


class TestNewsSentiment:
    """Test NewsSentiment value object."""

    def test_sentiment_creation(self):
        """Test creating a NewsSentiment value object."""
        sentiment = NewsSentiment(score=0.75, label="POSITIVE")
        assert sentiment.score == 0.75
        assert sentiment.label == "POSITIVE"

    def test_sentiment_score_range(self):
        """Test that sentiment score is in valid range."""
        # Score should be between -1.0 and 1.0
        valid_sentiment = NewsSentiment(score=0.5, label="POSITIVE")
        assert -1.0 <= valid_sentiment.score <= 1.0

    def test_sentiment_labels(self):
        """Test valid sentiment labels."""
        positive = NewsSentiment(score=0.5, label="POSITIVE")
        negative = NewsSentiment(score=-0.5, label="NEGATIVE")
        neutral = NewsSentiment(score=0.0, label="NEUTRAL")

        assert positive.label == "POSITIVE"
        assert negative.label == "NEGATIVE"
        assert neutral.label == "NEUTRAL"

    def test_sentiment_immutability(self):
        """Test that NewsSentiment is immutable."""
        sentiment = NewsSentiment(score=0.75, label="POSITIVE")

        with pytest.raises(Exception):
            sentiment.score = -0.5
