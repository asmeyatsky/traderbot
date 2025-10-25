"""
Basic functionality tests for the AI Trading Platform.

This module contains basic unit tests for the core components
to validate that they work as expected.
"""
import unittest
from datetime import datetime
from decimal import Decimal

from src.domain.entities.trading import Order, Position, Portfolio
from src.domain.entities.trading import OrderType, PositionType, OrderStatus
from src.domain.entities.user import User, RiskTolerance, InvestmentGoal
from src.domain.value_objects import Money, Symbol, Price, NewsSentiment
from src.domain.services.trading import DefaultTradingDomainService
from src.infrastructure.data_processing.sentiment_analysis import FinSentimentAnalyzer


class TestValueObjects(unittest.TestCase):
    """Test value objects to ensure they work correctly."""
    
    def test_money_operations(self):
        """Test basic operations on Money value object."""
        money1 = Money(Decimal('100.50'), 'USD')
        money2 = Money(Decimal('50.25'), 'USD')
        
        # Test addition
        result = money1 + money2
        self.assertEqual(result.amount, Decimal('150.75'))
        self.assertEqual(result.currency, 'USD')
        
        # Test subtraction
        result = money1 - money2
        self.assertEqual(result.amount, Decimal('50.25'))
        
        # Test multiplication
        result = money1 * Decimal('2')
        self.assertEqual(result.amount, Decimal('201.00'))
        
        # Test division
        result = money1 / Decimal('2')
        self.assertEqual(result.amount, Decimal('50.25'))
    
    def test_symbol_validation(self):
        """Test symbol validation rules."""
        # Valid symbols
        valid_symbols = ['AAPL', 'GOOGL', 'TSLA', 'BRK.B', 'MSFT']
        for sym in valid_symbols:
            symbol = Symbol(sym)
            self.assertEqual(str(symbol), sym)
        
        # Invalid symbols should raise an error
        invalid_symbols = ['123INVALID', 'INVALID_SYMBOL', 'verylongsymbolname']
        for sym in invalid_symbols:
            with self.assertRaises(ValueError, msg=f"Symbol {sym} should be invalid"):
                Symbol(sym)
    
    def test_news_sentiment(self):
        """Test NewsSentiment value object."""
        sentiment = NewsSentiment(score=Decimal('75.5'), confidence=Decimal('85.0'), source='Test')
        
        self.assertTrue(sentiment.is_positive)
        self.assertFalse(sentiment.is_negative)
        self.assertFalse(sentiment.is_neutral)
        
        # Test combining sentiments
        other_sentiment = NewsSentiment(score=Decimal('-20.0'), confidence=Decimal('90.0'), source='Test2')
        combined = sentiment.combine_with(other_sentiment)
        
        expected_score = (75.5 * 85.0 + (-20.0) * 90.0) / (85.0 + 90.0)
        self.assertAlmostEqual(float(combined.score), expected_score, places=1)


class TestEntities(unittest.TestCase):
    """Test domain entities to ensure they work correctly."""
    
    def test_order_lifecycle(self):
        """Test order creation and lifecycle methods."""
        order = Order(
            id="123",
            user_id="user123",
            symbol=Symbol("AAPL"),
            order_type=OrderType.MARKET,
            position_type=PositionType.LONG,
            quantity=10,
            status=OrderStatus.PENDING,
            placed_at=datetime.now()
        )
        
        # Test validation
        errors = order.validate()
        self.assertEqual(len(errors), 0, f"Order validation failed: {errors}")
        
        # Test execution
        execution_price = Price(Decimal('150.00'), 'USD')
        executed_order = order.execute(execution_price, datetime.now(), 10)
        
        self.assertEqual(executed_order.status, OrderStatus.EXECUTED)
        self.assertEqual(executed_order.price.amount, Decimal('150.00'))
        self.assertEqual(executed_order.filled_quantity, 10)
    
    def test_position_management(self):
        """Test position creation and management."""
        initial_price = Money(Decimal('100.00'), 'USD')
        current_price = Money(Decimal('120.00'), 'USD')
        
        position = Position(
            id="pos123",
            user_id="user123",
            symbol=Symbol("AAPL"),
            position_type=PositionType.LONG,
            quantity=10,
            average_buy_price=initial_price,
            current_price=current_price,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Test market value calculation
        expected_value = current_price.amount * 10
        self.assertEqual(position.market_value.amount, expected_value)
        
        # Test P&L calculation
        expected_pnl = (current_price.amount - initial_price.amount) * 10
        self.assertEqual(position.unrealized_pnl_amount.amount, expected_pnl)
        
        # Test price update
        new_price = Money(Decimal('125.00'), 'USD')
        updated_position = position.update_price(new_price)
        self.assertEqual(updated_position.current_price.amount, Decimal('125.00'))
    
    def test_portfolio_operations(self):
        """Test portfolio operations."""
        portfolio = Portfolio(
            id="port123",
            user_id="user123",
            cash_balance=Money(Decimal('10000.00'), 'USD')
        )
        
        # Initial state
        self.assertEqual(portfolio.total_value.amount, Decimal('10000.00'))
        
        # Add a position
        position = Position(
            id="pos123",
            user_id="user123",
            symbol=Symbol("AAPL"),
            position_type=PositionType.LONG,
            quantity=10,
            average_buy_price=Money(Decimal('100.00'), 'USD'),
            current_price=Money(Decimal('120.00'), 'USD'),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        updated_portfolio = portfolio.add_position(position)
        expected_total = Decimal('10000.00') + (Decimal('120.00') * 10)  # Cash + position value
        self.assertEqual(updated_portfolio.total_value.amount, expected_total)


class TestServices(unittest.TestCase):
    """Test domain services."""
    
    def test_trading_domain_service(self):
        """Test the trading domain service."""
        service = DefaultTradingDomainService()
        
        # Create test objects
        user = User(
            id="user123",
            email="test@example.com",
            first_name="Test",
            last_name="User",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            risk_tolerance=RiskTolerance.MODERATE,
            investment_goal=InvestmentGoal.BALANCED_GROWTH,
            max_position_size_percentage=Decimal('10.0')
        )
        
        portfolio = Portfolio(
            id="port123",
            user_id="user123",
            cash_balance=Money(Decimal('50000.00'), 'USD')
        )
        
        order = Order(
            id="order123",
            user_id="user123",
            symbol=Symbol("AAPL"),
            order_type=OrderType.MARKET,
            position_type=PositionType.LONG,
            quantity=100,
            status=OrderStatus.PENDING,
            placed_at=datetime.now(),
            price=Price(Decimal('150.00'), 'USD')
        )
        
        # Validate order
        errors = service.validate_order(order, user, portfolio)
        self.assertEqual(len(errors), 0, f"Order validation failed: {errors}")
        
        # Calculate position sizing
        position_size = service.calculate_position_sizing(user, portfolio, Symbol("AAPL"))
        # Position value = 150 * position_size should be <= 10% of portfolio value
        portfolio_value = portfolio.total_value.amount  # 50000
        max_position_value = (Decimal('10.0') / 100) * portfolio_value  # 5000
        max_shares = int(max_position_value / Decimal('100'))  # Assuming $100 per share
        self.assertLessEqual(position_size, max_shares)


class TestSentimentAnalyzer(unittest.TestCase):
    """Test sentiment analysis functionality."""
    
    def test_sentiment_analysis(self):
        """Test the sentiment analyzer."""
        analyzer = FinSentimentAnalyzer()
        
        # Test positive sentiment
        positive_text = "Company reports strong earnings growth and revenue beats expectations"
        sentiment = analyzer.analyze_sentiment(positive_text)
        self.assertIsInstance(sentiment, NewsSentiment)
        self.assertGreater(sentiment.score, 0, "Positive text should have positive sentiment")
        
        # Test negative sentiment
        negative_text = "Company misses earnings expectations and faces regulatory challenges"
        sentiment = analyzer.analyze_sentiment(negative_text)
        self.assertLess(sentiment.score, 0, "Negative text should have negative sentiment")
        
        # Test neutral sentiment
        neutral_text = "Company announces regular quarterly dividend"
        sentiment = analyzer.analyze_sentiment(neutral_text)
        # Allow some range around 0 for neutral
        self.assertGreaterEqual(sentiment.score, -20, "Neutral text should have score near 0")
        self.assertLessEqual(sentiment.score, 20, "Neutral text should have score near 0")


if __name__ == '__main__':
    unittest.main()