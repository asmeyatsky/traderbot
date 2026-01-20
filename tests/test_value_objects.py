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
        """Test adding Money values using operator."""
        money1 = Money(Decimal("100.00"), "USD")
        money2 = Money(Decimal("50.00"), "USD")
        result = money1 + money2

        assert result.amount == Decimal("150.00")
        assert result.currency == "USD"

    def test_money_subtraction(self):
        """Test subtracting Money values using operator."""
        money1 = Money(Decimal("100.00"), "USD")
        money2 = Money(Decimal("30.00"), "USD")
        result = money1 - money2

        assert result.amount == Decimal("70.00")
        assert result.currency == "USD"

    def test_money_multiplication(self):
        """Test multiplying Money by a scalar using operator."""
        money = Money(Decimal("100.00"), "USD")
        result = money * Decimal("2")

        assert result.amount == Decimal("200.00")
        assert result.currency == "USD"

    def test_money_division(self):
        """Test dividing Money by a scalar using operator."""
        money = Money(Decimal("100.00"), "USD")
        result = money / Decimal("2")

        assert result.amount == Decimal("50.00")
        assert result.currency == "USD"

    def test_money_currency_mismatch(self):
        """Test that adding Money with different currencies raises error."""
        money1 = Money(Decimal("100.00"), "USD")
        money2 = Money(Decimal("50.00"), "EUR")

        with pytest.raises(ValueError):
            _ = money1 + money2

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

    def test_symbol_uppercase_required(self):
        """Test that Symbol requires uppercase input."""
        # Symbol validation expects uppercase
        with pytest.raises(ValueError):
            Symbol("aapl")

    def test_symbol_equality(self):
        """Test Symbol equality."""
        symbol1 = Symbol("AAPL")
        symbol2 = Symbol("AAPL")
        symbol3 = Symbol("GOOGL")

        assert symbol1 == symbol2
        assert symbol1 != symbol3

    def test_symbol_invalid(self):
        """Test creating Symbol with invalid format."""
        # Invalid symbol should raise ValueError
        with pytest.raises(ValueError):
            Symbol("toolongname")

    def test_symbol_immutability(self):
        """Test that Symbol is immutable."""
        symbol = Symbol("AAPL")

        # Symbol should be immutable
        with pytest.raises(Exception):
            symbol.value = "GOOGL"


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

    def test_price_multiplication(self):
        """Test Price multiplication by scalar."""
        price = Price(Decimal("150.00"), "USD")
        result = price * Decimal("2")

        assert result.amount == Decimal("300.00")

    def test_price_immutability(self):
        """Test that Price is immutable."""
        price = Price(Decimal("150.00"), "USD")

        with pytest.raises(Exception):
            price.amount = Decimal("160.00")


class TestNewsSentiment:
    """Test NewsSentiment value object."""

    def test_sentiment_creation(self):
        """Test creating a NewsSentiment value object."""
        sentiment = NewsSentiment(
            score=Decimal("75"),
            confidence=Decimal("85"),
            source="Reuters"
        )
        assert sentiment.score == Decimal("75")
        assert sentiment.confidence == Decimal("85")
        assert sentiment.source == "Reuters"

    def test_sentiment_score_range(self):
        """Test that sentiment score is in valid range (-100 to 100)."""
        valid_sentiment = NewsSentiment(
            score=Decimal("50"),
            confidence=Decimal("80"),
            source="test"
        )
        assert -100 <= valid_sentiment.score <= 100

    def test_sentiment_is_positive(self):
        """Test sentiment is_positive property."""
        positive = NewsSentiment(
            score=Decimal("50"),
            confidence=Decimal("80"),
            source="test"
        )
        assert positive.is_positive is True

    def test_sentiment_is_negative(self):
        """Test sentiment is_negative property."""
        negative = NewsSentiment(
            score=Decimal("-50"),
            confidence=Decimal("80"),
            source="test"
        )
        assert negative.is_negative is True

    def test_sentiment_is_neutral(self):
        """Test sentiment is_neutral property."""
        neutral = NewsSentiment(
            score=Decimal("0"),
            confidence=Decimal("80"),
            source="test"
        )
        assert neutral.is_neutral is True

    def test_sentiment_immutability(self):
        """Test that NewsSentiment is immutable."""
        sentiment = NewsSentiment(
            score=Decimal("75"),
            confidence=Decimal("85"),
            source="test"
        )

        with pytest.raises(Exception):
            sentiment.score = Decimal("-50")
