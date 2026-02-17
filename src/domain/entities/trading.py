"""
Trading Domain Entities

This module contains the core domain entities for the trading platform,
following DDD principles and clean architecture patterns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import List, Optional
from enum import Enum
from src.domain.value_objects import Money, Symbol, TradingVolume


class OrderStatus(Enum):
    PENDING = "PENDING"
    EXECUTED = "EXECUTED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TRAILING_STOP = "TRAILING_STOP"


class PositionType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass(frozen=True)
class Order:
    """
    Order Domain Entity
    
    Represents a trading order with all relevant details.
    
    Architectural Intent:
    - This entity encapsulates all business rules related to trading orders
    - Orders are immutable after creation to prevent accidental state changes
    - Contains validation logic in factory methods
    """
    id: str
    user_id: str
    symbol: Symbol
    order_type: OrderType
    position_type: PositionType
    quantity: int
    status: OrderStatus
    placed_at: datetime
    executed_at: Optional[datetime] = None
    price: Optional[Money] = None
    stop_price: Optional[Money] = None
    filled_quantity: int = 0
    commission: Optional[Money] = None
    notes: Optional[str] = None
    broker_order_id: Optional[str] = None
    
    def execute(self, execution_price: Money, executed_at: datetime, filled_qty: int) -> 'Order':
        """Execute the order and return a new instance"""
        from dataclasses import replace
        return replace(
            self,
            status=OrderStatus.EXECUTED,
            executed_at=executed_at,
            price=execution_price,
            filled_quantity=filled_qty,
        )
    
    def cancel(self) -> 'Order':
        """Cancel the order and return a new instance"""
        from dataclasses import replace
        return replace(self, status=OrderStatus.CANCELLED)
    
    @property
    def is_filled(self) -> bool:
        return self.filled_quantity >= self.quantity
    
    @property
    def is_active(self) -> bool:
        return self.status in [OrderStatus.PENDING]
    
    def validate(self) -> List[str]:
        """Validate the order and return a list of validation errors"""
        errors = []
        
        if self.quantity <= 0:
            errors.append("Quantity must be positive")
        
        if self.price and self.price.amount < Decimal('0'):
            errors.append("Price must be positive")
            
        if self.stop_price and self.stop_price.amount < Decimal('0'):
            errors.append("Stop price must be positive")
        
        return errors


@dataclass(frozen=True)
class Position:
    """
    Position Domain Entity
    
    Represents a user's position in a particular security.
    
    Architectural Intent:
    - Maintains consistency of position state
    - Encapsulates position-related business logic
    - Immutable to prevent accidental state corruption
    """
    id: str
    user_id: str
    symbol: Symbol
    position_type: PositionType
    quantity: int
    average_buy_price: Money
    current_price: Money
    created_at: datetime
    updated_at: datetime
    unrealized_pnl: Optional[Money] = None
    realized_pnl: Money = Money(Decimal('0'), 'USD')
    
    @property
    def market_value(self) -> Money:
        """Calculate the current market value of the position"""
        amount = self.current_price.amount * Decimal(self.quantity)
        return Money(amount, self.current_price.currency)
    
    @property
    def total_cost(self) -> Money:
        """Calculate the total cost of the position"""
        amount = self.average_buy_price.amount * Decimal(self.quantity)
        return Money(amount, self.average_buy_price.currency)
    
    @property
    def unrealized_pnl_amount(self) -> Money:
        """Calculate unrealized profit or loss"""
        if self.unrealized_pnl is not None:
            return self.unrealized_pnl
        
        market_value = self.market_value
        total_cost = self.total_cost
        pnl_amount = market_value.amount - total_cost.amount
        return Money(pnl_amount, market_value.currency)
    
    def update_price(self, new_price: Money) -> 'Position':
        """Update the current price and return a new position instance"""
        return Position(
            id=self.id,
            user_id=self.user_id,
            symbol=self.symbol,
            position_type=self.position_type,
            quantity=self.quantity,
            average_buy_price=self.average_buy_price,
            current_price=new_price,
            created_at=self.created_at,
            updated_at=datetime.now(),
            unrealized_pnl=self.unrealized_pnl,
            realized_pnl=self.realized_pnl
        )
    
    def adjust_quantity(self, quantity_change: int, execution_price: Money) -> 'Position':
        """
        Adjust the position quantity (for adding or reducing position)
        
        For simplicity, this updates average price based on new execution.
        In real trading, this would need to consider FIFO, LIFO, or average cost basis.
        """
        new_quantity = self.quantity + quantity_change
        
        # If position is being closed (quantity becomes 0)
        if new_quantity == 0:
            return Position(
                id=self.id,
                user_id=self.user_id,
                symbol=self.symbol,
                position_type=self.position_type,
                quantity=0,
                average_buy_price=Money(Decimal('0'), 'USD'),
                current_price=execution_price,
                created_at=self.created_at,
                updated_at=datetime.now(),
                unrealized_pnl=Money(Decimal('0'), 'USD'),
                realized_pnl=self.realized_pnl
            )
        
        # Calculate new average price
        total_cost_before = self.average_buy_price.amount * Decimal(self.quantity)
        cost_of_new_shares = execution_price.amount * Decimal(quantity_change)
        new_total_cost = total_cost_before + cost_of_new_shares
        new_avg_price = new_total_cost / Decimal(new_quantity)
        
        return Position(
            id=self.id,
            user_id=self.user_id,
            symbol=self.symbol,
            position_type=self.position_type,
            quantity=new_quantity,
            average_buy_price=Money(new_avg_price, self.average_buy_price.currency),
            current_price=execution_price,
            created_at=self.created_at,
            updated_at=datetime.now(),
            unrealized_pnl=self.unrealized_pnl,
            realized_pnl=self.realized_pnl
        )


@dataclass(frozen=True)
class Portfolio:
    """
    Portfolio Domain Entity

    Aggregates all positions for a user and manages portfolio-level operations.

    Architectural Intent:
    - Serves as an aggregate root for positions
    - Maintains portfolio-level invariants
    - Encapsulates portfolio-related business logic
    """
    id: str
    user_id: str
    positions: List[Position] = field(default_factory=list)  # Properly initialized mutable default
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    cash_balance: Money = field(default_factory=lambda: Money(Decimal('0'), 'USD'))
    
    @property
    def total_value(self) -> Money:
        """Calculate total portfolio value (cash + all positions)"""
        positions_value = sum((pos.market_value.amount for pos in self.positions), Decimal('0'))
        total_amount = positions_value + self.cash_balance.amount
        return Money(total_amount, self.cash_balance.currency)
    
    @property
    def positions_value(self) -> Money:
        """Calculate value of all positions"""
        positions_value = sum((pos.market_value.amount for pos in self.positions), Decimal('0'))
        return Money(positions_value, self.cash_balance.currency)
    
    def add_position(self, position: Position) -> 'Portfolio':
        """Add a new position to the portfolio"""
        new_positions = [pos for pos in self.positions if pos.symbol != position.symbol]
        new_positions.append(position)
        
        return Portfolio(
            id=self.id,
            user_id=self.user_id,
            positions=new_positions,
            created_at=self.created_at,
            updated_at=datetime.now(),
            cash_balance=self.cash_balance
        )
    
    def remove_position(self, symbol: Symbol) -> 'Portfolio':
        """Remove a position from the portfolio"""
        new_positions = [pos for pos in self.positions if pos.symbol != symbol]
        
        return Portfolio(
            id=self.id,
            user_id=self.user_id,
            positions=new_positions,
            created_at=self.created_at,
            updated_at=datetime.now(),
            cash_balance=self.cash_balance
        )
    
    def update_position(self, updated_position: Position) -> 'Portfolio':
        """Update an existing position"""
        new_positions = []
        for pos in self.positions:
            if pos.symbol == updated_position.symbol:
                new_positions.append(updated_position)
            else:
                new_positions.append(pos)
        
        return Portfolio(
            id=self.id,
            user_id=self.user_id,
            positions=new_positions,
            created_at=self.created_at,
            updated_at=datetime.now(),
            cash_balance=self.cash_balance
        )
    
    def update_cash_balance(self, new_balance: Money) -> 'Portfolio':
        """Update the cash balance"""
        return Portfolio(
            id=self.id,
            user_id=self.user_id,
            positions=self.positions,
            created_at=self.created_at,
            updated_at=datetime.now(),
            cash_balance=new_balance
        )
    
    def get_position(self, symbol: Symbol) -> Optional[Position]:
        """Get a specific position by symbol"""
        for pos in self.positions:
            if pos.symbol == symbol:
                return pos
        return None