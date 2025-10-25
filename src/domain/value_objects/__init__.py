"""
Value Objects for the Trading Domain

This module contains value objects used in the trading platform domain,
following DDD principles and clean architecture patterns.
"""
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional
import re


@dataclass(frozen=True)
class Money:
    """
    Money Value Object
    
    Represents a monetary amount with currency.
    
    Architectural Intent:
    - Provides type safety for monetary values
    - Prevents mixing of different currencies
    - Encapsulates monetary operations
    """
    amount: Decimal
    currency: str
    
    def __post_init__(self):
        """Validate money after initialization"""
        if self.amount.as_tuple().exponent < -2:
            raise ValueError("Currency amounts should have maximum 2 decimal places")
        
        if len(self.currency) != 3:
            raise ValueError("Currency must be 3-letter ISO code")
    
    def __add__(self, other: 'Money') -> 'Money':
        """Add two Money values of the same currency"""
        if self.currency != other.currency:
            raise ValueError(f"Cannot add different currencies: {self.currency} and {other.currency}")
        
        return Money(self.amount + other.amount, self.currency)
    
    def __sub__(self, other: 'Money') -> 'Money':
        """Subtract two Money values of the same currency"""
        if self.currency != other.currency:
            raise ValueError(f"Cannot subtract different currencies: {self.currency} and {other.currency}")
        
        return Money(self.amount - other.amount, self.currency)
    
    def __mul__(self, factor: Decimal) -> 'Money':
        """Multiply Money by a factor"""
        return Money(self.amount * factor, self.currency)
    
    def __truediv__(self, divisor: Decimal) -> 'Money':
        """Divide Money by a divisor"""
        if divisor == Decimal('0'):
            raise ValueError("Cannot divide by zero")
        return Money(self.amount / divisor, self.currency)
    
    def __neg__(self) -> 'Money':
        """Negate the amount"""
        return Money(-self.amount, self.currency)
    
    def is_positive(self) -> bool:
        """Check if the amount is positive"""
        return self.amount > Decimal('0')
    
    def is_negative(self) -> bool:
        """Check if the amount is negative"""
        return self.amount < Decimal('0')
    
    def is_zero(self) -> bool:
        """Check if the amount is zero"""
        return self.amount == Decimal('0')


@dataclass(frozen=True)
class Symbol:
    """
    Stock Symbol Value Object
    
    Represents a stock symbol with validation.
    
    Architectural Intent:
    - Provides type safety for stock symbols
    - Validates symbol format
    - Encapsulates symbol operations
    """
    value: str
    
    def __post_init__(self):
        """Validate symbol after initialization"""
        # Stock symbols are typically 1-5 uppercase letters (sometimes with dots for special cases like BRK.B)
        if not re.match(r'^[A-Z][A-Z0-9\.\-]{0,4}$', self.value):
            raise ValueError(f"Invalid stock symbol format: {self.value}")
    
    def __str__(self) -> str:
        return self.value
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Symbol):
            return False
        return self.value.upper() == other.value.upper()
    
    def __hash__(self) -> int:
        return hash(self.value.upper())


@dataclass(frozen=True)
class TradingVolume:
    """
    Trading Volume Value Object
    
    Represents trading volume data.
    
    Architectural Intent:
    - Provides type safety for trading volume
    - Encapsulates volume-related operations
    """
    volume: int
    
    def __post_init__(self):
        """Validate volume after initialization"""
        if self.volume < 0:
            raise ValueError("Volume cannot be negative")
    
    def __add__(self, other: 'TradingVolume') -> 'TradingVolume':
        """Add two volume values"""
        return TradingVolume(self.volume + other.volume)
    
    def __sub__(self, other: 'TradingVolume') -> 'TradingVolume':
        """Subtract two volume values"""
        result = self.volume - other.volume
        if result < 0:
            raise ValueError("Resulting volume cannot be negative")
        return TradingVolume(result)
    
    def weighted_average(self, other: 'TradingVolume', self_weight: Decimal, other_weight: Decimal) -> 'TradingVolume':
        """Calculate weighted average of volumes"""
        if self_weight < 0 or other_weight < 0:
            raise ValueError("Weights must be non-negative")
        
        total_weight = self_weight + other_weight
        if total_weight == 0:
            raise ValueError("Sum of weights cannot be zero")
        
        weighted_avg = (self.volume * self_weight + other.volume * other_weight) / total_weight
        return TradingVolume(int(weighted_avg))


@dataclass(frozen=True)
class Price:
    """
    Price Value Object
    
    Represents a price with validation and operations.
    
    Architectural Intent:
    - Provides type safety for price values
    - Validates price format and range
    - Encapsulates price-related operations
    """
    amount: Decimal
    currency: str
    
    def __post_init__(self):
        """Validate price after initialization"""
        if self.amount < Decimal('0'):
            raise ValueError("Price cannot be negative")
        
        if len(self.currency) != 3:
            raise ValueError("Currency must be 3-letter ISO code")
        
        if self.amount.as_tuple().exponent < -4:
            raise ValueError("Price should have maximum 4 decimal places")
    
    def calculate_change(self, other: 'Price') -> Decimal:
        """Calculate percentage change from another price"""
        if self.currency != other.currency:
            raise ValueError("Prices must be in the same currency to calculate change")
        
        if other.amount == Decimal('0'):
            raise ValueError("Cannot calculate change from zero price")
        
        return (self.amount - other.amount) / other.amount * 100
    
    def __add__(self, other: 'Price') -> 'Price':
        """Add two prices of the same currency"""
        if self.currency != other.currency:
            raise ValueError(f"Cannot add different currencies: {self.currency} and {other.currency}")
        
        return Price(self.amount + other.amount, self.currency)
    
    def __sub__(self, other: 'Price') -> 'Price':
        """Subtract two prices of the same currency"""
        if self.currency != other.currency:
            raise ValueError(f"Cannot subtract different currencies: {self.currency} and {other.currency}")
        
        result = self.amount - other.amount
        if result < 0:
            raise ValueError("Resulting price cannot be negative")
        
        return Price(result, self.currency)
    
    def __mul__(self, factor: Decimal) -> 'Price':
        """Multiply price by a factor"""
        return Price(self.amount * factor, self.currency)
    
    def __truediv__(self, divisor: Decimal) -> 'Price':
        """Divide price by a divisor"""
        if divisor == Decimal('0'):
            raise ValueError("Cannot divide by zero")
        return Price(self.amount / divisor, self.currency)
    
    def __lt__(self, other: 'Price') -> bool:
        """Compare prices of the same currency"""
        if self.currency != other.currency:
            raise ValueError(f"Cannot compare different currencies: {self.currency} and {other.currency}")
        return self.amount < other.amount
    
    def __le__(self, other: 'Price') -> bool:
        """Compare prices of the same currency"""
        if self.currency != other.currency:
            raise ValueError(f"Cannot compare different currencies: {self.currency} and {other.currency}")
        return self.amount <= other.amount
    
    def __gt__(self, other: 'Price') -> bool:
        """Compare prices of the same currency"""
        if self.currency != other.currency:
            raise ValueError(f"Cannot compare different currencies: {self.currency} and {other.currency}")
        return self.amount > other.amount
    
    def __ge__(self, other: 'Price') -> bool:
        """Compare prices of the same currency"""
        if self.currency != other.currency:
            raise ValueError(f"Cannot compare different currencies: {self.currency} and {other.currency}")
        return self.amount >= other.amount


@dataclass(frozen=True)
class NewsSentiment:
    """
    News Sentiment Value Object
    
    Represents sentiment analysis of news articles.
    
    Architectural Intent:
    - Provides type safety for sentiment scores
    - Validates sentiment range (-100 to 100)
    - Encapsulates sentiment-related operations
    """
    score: Decimal  # -100 (extremely negative) to +100 (extremely positive)
    confidence: Decimal  # 0 to 100 percent confidence
    source: str  # Source of the sentiment analysis
    
    def __post_init__(self):
        """Validate sentiment after initialization"""
        if not (-100 <= self.score <= 100):
            raise ValueError("Sentiment score must be between -100 and 100")
        
        if not (0 <= self.confidence <= 100):
            raise ValueError("Confidence must be between 0 and 100")
    
    @property
    def is_positive(self) -> bool:
        """Check if sentiment is positive"""
        return self.score > 0
    
    @property
    def is_negative(self) -> bool:
        """Check if sentiment is negative"""
        return self.score < 0
    
    @property
    def is_neutral(self) -> bool:
        """Check if sentiment is neutral (close to 0)"""
        return abs(self.score) <= 5  # Consider scores between -5 and 5 as neutral
    
    def combine_with(self, other: 'NewsSentiment') -> 'NewsSentiment':
        """Combine two sentiment scores using weighted average"""
        total_confidence = self.confidence + other.confidence
        if total_confidence == 0:
            return NewsSentiment(
                score=self.score,
                confidence=self.confidence,
                source=f"Combined({self.source},{other.source})"
            )
        
        combined_score = (
            (self.score * self.confidence + other.score * other.confidence) / 
            total_confidence
        )
        
        return NewsSentiment(
            score=combined_score,
            confidence=max(self.confidence, other.confidence),
            source=f"Combined({self.source},{other.source})"
        )