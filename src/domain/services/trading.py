"""
Domain Services for the Trading Platform

This module contains domain services that implement business logic
that doesn't naturally fit within an entity or value object.
Following DDD principles and clean architecture patterns.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime
from decimal import Decimal
import uuid

from src.domain.entities.trading import (
    Order, Position, Portfolio, OrderType, PositionType, OrderStatus
)
from src.domain.entities.user import User, RiskTolerance, InvestmentGoal
from src.domain.value_objects import Money, Symbol, Price, NewsSentiment


class TradingDomainService(ABC):
    """
    Abstract base class for trading domain services.
    
    Architectural Intent:
    - Defines interfaces for core trading domain services
    - Separates domain logic from infrastructure concerns
    - Enables dependency injection and testing
    """
    
    @abstractmethod
    def validate_order(self, order: Order, user: User, portfolio: Portfolio) -> List[str]:
        """
        Validate that an order is allowed based on user constraints and portfolio state.
        """
        pass
    
    @abstractmethod
    def calculate_position_sizing(self, user: User, portfolio: Portfolio, symbol: Symbol) -> int:
        """
        Calculate the appropriate position size based on user constraints and portfolio allocation.
        """
        pass
    
    @abstractmethod
    def execute_order(self, order: Order, current_price: Price) -> Order:
        """
        Execute an order and return the updated order state.
        """
        pass


class DefaultTradingDomainService(TradingDomainService):
    """
    Default implementation of trading domain services.
    
    Implements core trading business logic following the PRD requirements.
    """
    
    def validate_order(self, order: Order, user: User, portfolio: Portfolio) -> List[str]:
        """
        Validate that an order is allowed based on user constraints and portfolio state.
        """
        errors = []
        
        # Validate order-specific constraints
        order_errors = order.validate()
        errors.extend(order_errors)
        
        # Check if user has sufficient cash for buy orders
        if order.position_type == PositionType.LONG:
            required_amount = (order.price.amount * Decimal(order.quantity)).quantize(Decimal("0.01")) if order.price else Decimal("0")
            if required_amount > portfolio.cash_balance.amount:
                errors.append(f"Insufficient cash balance. Required: ${required_amount:.2f}, Available: ${portfolio.cash_balance.amount:.2f}")

        # Check position size constraints
        portfolio_value = portfolio.total_value.amount
        if portfolio_value > 0:
            position_value = (order.price.amount * Decimal(order.quantity)).quantize(Decimal("0.01")) if order.price else Decimal("0")
            position_percentage = (position_value / portfolio_value) * 100
            
            if position_percentage > user.max_position_size_percentage:
                errors.append(
                    f"Position size ({position_percentage:.2f}%) exceeds user's maximum ({user.max_position_size_percentage}%)"
                )
        
        # Check sector constraints
        symbol_str = str(order.symbol).upper()
        if user.sector_exclusions:
            # This would require mapping symbols to sectors, simplified for now
            # In real implementation, we'd look up sector information for the symbol
            pass
        
        return errors
    
    def calculate_position_sizing(self, user: User, portfolio: Portfolio, symbol: Symbol) -> int:
        """
        Calculate the appropriate position size based on user constraints and portfolio allocation.
        """
        # Calculate max position size based on percentage constraint
        max_portfolio_percentage = user.max_position_size_percentage
        portfolio_value = portfolio.total_value.amount
        max_position_value = (max_portfolio_percentage / Decimal('100')) * portfolio_value
        
        # For this calculation, we'll assume we have the current price of the symbol
        # In a real implementation, this would come from market data
        # For now, we'll return an arbitrary value based on the max position value
        # assuming an average stock price of $100
        average_stock_price = Decimal('100.00')
        max_shares = int(max_position_value / average_stock_price)
        
        return max_shares
    
    def execute_order(self, order: Order, current_price: Price) -> Order:
        """
        Execute an order and return the updated order state.
        """
        import datetime
        execution_time = datetime.datetime.now()
        
        # Calculate filled quantity (simplified - assumes full fill for now)
        filled_qty = order.quantity
        
        # Update commission (simplified calculation)
        commission_amount = Decimal('0.00')  # In real implementation, use broker-specific commission
        
        # Return updated order with execution details
        return Order(
            id=order.id,
            user_id=order.user_id,
            symbol=order.symbol,
            order_type=order.order_type,
            position_type=order.position_type,
            quantity=order.quantity,
            status=OrderStatus.EXECUTED,
            placed_at=order.placed_at,
            executed_at=execution_time,
            price=Money(current_price.amount, current_price.currency),
            stop_price=order.stop_price,
            filled_quantity=filled_qty,
            commission=Money(commission_amount, current_price.currency),
            notes=order.notes
        )


class RiskManagementDomainService(ABC):
    """
    Abstract base class for risk management domain services.
    """
    
    @abstractmethod
    def check_portfolio_risk_limits(self, portfolio: Portfolio, user: User) -> List[str]:
        """
        Check if the portfolio violates any of the user's risk limits.
        """
        pass
    
    @abstractmethod
    def should_pause_trading(self, portfolio: Portfolio, user: User) -> bool:
        """
        Determine if trading should be paused based on risk conditions.
        """
        pass


class DefaultRiskManagementDomainService(RiskManagementDomainService):
    """
    Default implementation of risk management domain services.
    """
    
    def check_portfolio_risk_limits(self, portfolio: Portfolio, user: User) -> List[str]:
        """
        Check if the portfolio violates any of the user's risk limits.
        """
        errors = []
        
        # Check daily loss limit
        if user.daily_loss_limit and portfolio.total_value.amount < (
            portfolio.total_value.amount - user.daily_loss_limit.amount
        ):
            errors.append(f"Daily loss limit exceeded: {user.daily_loss_limit}")
        
        # Check weekly loss limit
        if user.weekly_loss_limit and portfolio.total_value.amount < (
            portfolio.total_value.amount - user.weekly_loss_limit.amount
        ):
            errors.append(f"Weekly loss limit exceeded: {user.weekly_loss_limit}")
        
        # Check monthly loss limit
        if user.monthly_loss_limit and portfolio.total_value.amount < (
            portfolio.total_value.amount - user.monthly_loss_limit.amount
        ):
            errors.append(f"Monthly loss limit exceeded: {user.monthly_loss_limit}")
        
        # Check drawdown limit based on risk tolerance
        # This is a simplified check - in real implementation, we'd track peak values over time
        max_drawdown = {
            RiskTolerance.CONSERVATIVE: Decimal('10.0'),
            RiskTolerance.MODERATE: Decimal('15.0'),
            RiskTolerance.AGGRESSIVE: Decimal('25.0')
        }
        
        if user.risk_tolerance:
            # This calculation would require historical portfolio values
            # Simplified for now
            pass
        
        return errors
    
    def should_pause_trading(self, portfolio: Portfolio, user: User) -> bool:
        """
        Determine if trading should be paused based on risk conditions.
        """
        limit_errors = self.check_portfolio_risk_limits(portfolio, user)
        return len(limit_errors) > 0


class PortfolioOptimizationDomainService(ABC):
    """
    Abstract base class for portfolio optimization domain services.
    """
    
    @abstractmethod
    def rebalance_portfolio(self, portfolio: Portfolio, user: User) -> List[Order]:
        """
        Generate rebalancing orders for the portfolio based on target allocation.
        """
        pass
    
    @abstractmethod
    def optimize_allocation(self, portfolio: Portfolio, user: User) -> dict:
        """
        Optimize the portfolio allocation based on the user's risk tolerance and goals.
        """
        pass


class DefaultPortfolioOptimizationDomainService(PortfolioOptimizationDomainService):
    """
    Default implementation of portfolio optimization domain services.
    """
    
    def rebalance_portfolio(self, portfolio: Portfolio, user: User) -> List[Order]:
        """
        Generate rebalancing orders for the portfolio based on target allocation.
        """
        orders = []
        
        # Simplified rebalancing logic - in a real implementation this would be much more complex
        # and would consider various factors like correlation, volatility, etc.
        
        # For now, return an empty list as a placeholder
        return orders
    
    def optimize_allocation(self, portfolio: Portfolio, user: User) -> dict:
        """
        Optimize the portfolio allocation based on the user's risk tolerance and goals.
        """
        # Calculate current allocation
        current_value = portfolio.total_value.amount
        if current_value <= 0:
            return {}
        
        # Simplified optimization based on user's goals
        optimization_targets = {
            InvestmentGoal.CAPITAL_PRESERVATION: {
                "cash_percentage": 40,
                "stable_stocks_percentage": 40,
                "growth_stocks_percentage": 20
            },
            InvestmentGoal.BALANCED_GROWTH: {
                "cash_percentage": 20,
                "stable_stocks_percentage": 40,
                "growth_stocks_percentage": 40
            },
            InvestmentGoal.MAXIMUM_RETURNS: {
                "cash_percentage": 5,
                "stable_stocks_percentage": 25,
                "growth_stocks_percentage": 70
            }
        }
        
        # This is a simplified calculation - real optimization would use advanced algorithms
        return optimization_targets.get(user.investment_goal, {})