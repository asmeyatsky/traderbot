"""
Ports for External Dependencies

This module defines the interfaces (ports) for external services
that the domain layer depends on, following the ports and adapters pattern.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime, date

from src.domain.entities.trading import Order, Position, Portfolio
from src.domain.entities.user import User
from src.domain.value_objects import Symbol, Price, NewsSentiment, Money


class OrderRepositoryPort(ABC):
    """
    Port for order persistence operations.
    
    Architectural Intent:
    - Defines the interface for storing and retrieving orders
    - Domain layer depends on this abstraction, not concrete implementation
    - Enables dependency inversion principle
    """
    
    @abstractmethod
    def save(self, order: Order) -> Order:
        """Save an order to persistent storage."""
        pass
    
    @abstractmethod
    def get_by_id(self, order_id: str) -> Optional[Order]:
        """Retrieve an order by its ID."""
        pass
    
    @abstractmethod
    def get_by_user_id(self, user_id: str) -> List[Order]:
        """Retrieve all orders for a user."""
        pass
    
    @abstractmethod
    def get_active_orders(self, user_id: str) -> List[Order]:
        """Retrieve all active orders for a user."""
        pass


class PositionRepositoryPort(ABC):
    """
    Port for position persistence operations.
    """
    
    @abstractmethod
    def save(self, position: Position) -> Position:
        """Save a position to persistent storage."""
        pass
    
    @abstractmethod
    def get_by_id(self, position_id: str) -> Optional[Position]:
        """Retrieve a position by its ID."""
        pass
    
    @abstractmethod
    def get_by_user_id(self, user_id: str) -> List[Position]:
        """Retrieve all positions for a user."""
        pass
    
    @abstractmethod
    def get_by_symbol(self, user_id: str, symbol: Symbol) -> Optional[Position]:
        """Retrieve a specific position by user and symbol."""
        pass


class PortfolioRepositoryPort(ABC):
    """
    Port for portfolio persistence operations.
    """
    
    @abstractmethod
    def save(self, portfolio: Portfolio) -> Portfolio:
        """Save a portfolio to persistent storage."""
        pass
    
    @abstractmethod
    def get_by_user_id(self, user_id: str) -> Optional[Portfolio]:
        """Retrieve a portfolio by user ID."""
        pass


class UserRepositoryPort(ABC):
    """
    Port for user persistence operations.
    """
    
    @abstractmethod
    def save(self, user: User) -> User:
        """Save a user to persistent storage."""
        pass
    
    @abstractmethod
    def get_by_id(self, user_id: str) -> Optional[User]:
        """Retrieve a user by ID."""
        pass
    
    @abstractmethod
    def get_by_email(self, email: str) -> Optional[User]:
        """Retrieve a user by email."""
        pass


class MarketDataPort(ABC):
    """
    Port for market data operations.
    """
    
    @abstractmethod
    def get_current_price(self, symbol: Symbol) -> Optional[Price]:
        """Get the current price for a symbol."""
        pass
    
    @abstractmethod
    def get_historical_prices(
        self, 
        symbol: Symbol, 
        start_date: date, 
        end_date: date
    ) -> List[Price]:
        """Get historical prices for a symbol within a date range."""
        pass
    
    @abstractmethod
    def get_market_news(self, symbol: Symbol) -> List[str]:
        """Get recent news for a symbol."""
        pass


class NewsAnalysisPort(ABC):
    """
    Port for news analysis and sentiment operations.
    """
    
    @abstractmethod
    def analyze_sentiment(self, text: str) -> NewsSentiment:
        """Analyze sentiment of a text."""
        pass
    
    @abstractmethod
    def batch_analyze_sentiment(self, texts: List[str]) -> List[NewsSentiment]:
        """Analyze sentiment of multiple texts."""
        pass
    
    @abstractmethod
    def extract_symbols_from_news(self, news_text: str) -> List[Symbol]:
        """Extract stock symbols mentioned in news text."""
        pass


class TradingExecutionPort(ABC):
    """
    Port for executing trades with brokers.
    """
    
    @abstractmethod
    def place_order(self, order: Order) -> str:
        """Place an order with the broker and return order ID."""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order with the broker."""
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Order:
        """Get the current status of an order."""
        pass
    
    @abstractmethod
    def get_account_balance(self, user_id: str) -> Money:
        """Get the account balance for a user."""
        pass


class NotificationPort(ABC):
    """
    Port for sending notifications to users.
    """
    
    @abstractmethod
    def send_trade_notification(self, user: User, order: Order) -> bool:
        """Send a notification about a trade execution."""
        pass
    
    @abstractmethod
    def send_risk_alert(self, user: User, message: str) -> bool:
        """Send a risk alert to a user."""
        pass
    
    @abstractmethod
    def send_market_alert(self, user: User, symbol: Symbol, message: str) -> bool:
        """Send a market alert to a user."""
        pass


class AIModelPort(ABC):
    """
    Port for AI/ML model operations.
    """
    
    @abstractmethod
    def predict_price_movement(self, symbol: Symbol, days: int = 1) -> float:
        """Predict price movement for a symbol."""
        pass
    
    @abstractmethod
    def get_trading_signal(self, symbol: Symbol) -> str:  # Returns 'BUY', 'SELL', 'HOLD'
        """Get a trading signal for a symbol."""
        pass
    
    @abstractmethod
    def analyze_portfolio_risk(self, portfolio: Portfolio) -> float:
        """Analyze risk level of a portfolio (0-1 scale)."""
        pass