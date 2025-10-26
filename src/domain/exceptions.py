"""
Domain Exceptions

This module defines custom exceptions for the trading domain following DDD principles.
All exceptions should be raised from the domain layer and caught by the application layer.
"""


class DomainException(Exception):
    """
    Base exception for all domain-level errors.

    Architectural Intent:
    - Provides a clear exception hierarchy
    - Allows catching all domain exceptions at application layer
    - Makes error handling explicit and type-safe
    """
    pass


class OrderException(DomainException):
    """Base exception for order-related errors."""
    pass


class OrderValidationException(OrderException):
    """Raised when order validation fails."""
    pass


class InsufficientFundsException(OrderException):
    """Raised when user has insufficient cash to execute an order."""
    pass


class PositionSizeViolationException(OrderException):
    """Raised when order would violate position size constraints."""
    pass


class OrderNotFound(OrderException):
    """Raised when an order cannot be found."""
    pass


class PortfolioException(DomainException):
    """Base exception for portfolio-related errors."""
    pass


class PortfolioNotFound(PortfolioException):
    """Raised when a portfolio cannot be found."""
    pass


class InvalidPortfolioOperation(PortfolioException):
    """Raised when an invalid operation is attempted on a portfolio."""
    pass


class UserException(DomainException):
    """Base exception for user-related errors."""
    pass


class UserNotFound(UserException):
    """Raised when a user cannot be found."""
    pass


class InvalidUserConfiguration(UserException):
    """Raised when user configuration is invalid."""
    pass


class RiskManagementException(DomainException):
    """Base exception for risk management violations."""
    pass


class RiskLimitExceededException(RiskManagementException):
    """Raised when a risk limit is violated."""
    pass


class DailyLossLimitExceeded(RiskLimitExceededException):
    """Raised when daily loss limit is exceeded."""
    pass


class WeeklyLossLimitExceeded(RiskLimitExceededException):
    """Raised when weekly loss limit is exceeded."""
    pass


class MonthlyLossLimitExceeded(RiskLimitExceededException):
    """Raised when monthly loss limit is exceeded."""
    pass


class DrawdownLimitExceeded(RiskLimitExceededException):
    """Raised when maximum drawdown limit is exceeded."""
    pass


class SectorConstraintViolation(RiskManagementException):
    """Raised when order violates sector constraints."""
    pass


class TradingException(DomainException):
    """Base exception for general trading errors."""
    pass


class MarketDataException(TradingException):
    """Raised when market data cannot be retrieved."""
    pass


class PriceUnavailableException(MarketDataException):
    """Raised when price data is unavailable for a symbol."""
    pass


class InvalidSymbolException(TradingException):
    """Raised when an invalid stock symbol is used."""
    pass


class AISignalException(TradingException):
    """Raised when AI signal generation fails."""
    pass


class BrokerException(TradingException):
    """Base exception for broker-related errors."""
    pass


class BrokerOrderExecutionException(BrokerException):
    """Raised when order execution fails at the broker."""
    pass


class BrokerConnectionException(BrokerException):
    """Raised when connection to broker fails."""
    pass


class InsufficientBrokerBalance(BrokerException):
    """Raised when broker account has insufficient balance."""
    pass
