"""
Application Layer Use Cases

This module contains the application use cases that orchestrate
domain entities and services to implement business functionality.
Following clean architecture principles and DDD patterns.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import List, Optional
import uuid

from src.domain.entities.trading import Order, Position, Portfolio
from src.domain.entities.user import User
from src.domain.value_objects import Symbol, Money, Price
from src.domain.ports import (
    OrderRepositoryPort, PositionRepositoryPort, PortfolioRepositoryPort, 
    UserRepositoryPort, MarketDataPort, NewsAnalysisPort, 
    TradingExecutionPort, NotificationPort, AIModelPort
)
from src.domain.services.trading import TradingDomainService, RiskManagementDomainService
from src.infrastructure.data_processing.sentiment_analysis import sentiment_analyzer


class CreateOrderUseCase:
    """
    Use Case: Create a new trading order.
    
    Orchestrates the creation of a trading order with validation
    and persistence.
    """
    
    def __init__(
        self,
        order_repository: OrderRepositoryPort,
        portfolio_repository: PortfolioRepositoryPort,
        user_repository: UserRepositoryPort,
        trading_service: TradingDomainService,
        market_data_service: MarketDataPort,
        position_repository: Optional[PositionRepositoryPort] = None,
    ):
        self.order_repository = order_repository
        self.portfolio_repository = portfolio_repository
        self.user_repository = user_repository
        self.trading_service = trading_service
        self.market_data_service = market_data_service
        self.position_repository = position_repository

    def execute(
        self,
        user_id: str,
        symbol: Symbol,
        order_type: str,
        position_type: str,
        quantity: int,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Optional[Order]:
        """
        Execute the create order use case.

        In paper-trading mode, market orders are auto-filled immediately:
        the order is marked EXECUTED, a position is created/updated, and
        the portfolio cash balance is adjusted.
        """
        # Retrieve user and portfolio
        user = self.user_repository.get_by_id(user_id)
        if not user:
            raise ValueError(f"User not found: {user_id}")

        portfolio = self.portfolio_repository.get_by_user_id(user_id)
        if not portfolio:
            raise ValueError(f"Portfolio not found for user: {user_id}")

        # Get current price for the symbol
        current_price = self.market_data_service.get_current_price(symbol)
        if not current_price:
            raise ValueError(f"Could not get current price for symbol: {symbol}")

        from src.domain.entities.trading import OrderType, PositionType as PT, OrderStatus
        from dataclasses import replace as dc_replace

        order_type_enum = OrderType[order_type.upper()]
        position_type_enum = PT[position_type.upper()]

        order = Order(
            id=str(uuid.uuid4()),
            user_id=user_id,
            symbol=symbol,
            order_type=order_type_enum,
            position_type=position_type_enum,
            quantity=quantity,
            status=OrderStatus.PENDING,
            placed_at=datetime.now(),
            price=Price(Decimal(str(limit_price)), 'USD') if limit_price else current_price,
            stop_price=Price(Decimal(str(stop_price)), 'USD') if stop_price else None,
            filled_quantity=0,
            commission=None,
            notes=None
        )

        # Validate order against user constraints
        validation_errors = self.trading_service.validate_order(order, user, portfolio)
        if validation_errors:
            raise ValueError(f"Order validation failed: {'; '.join(validation_errors)}")

        # Paper trading: auto-fill market orders immediately
        if order_type_enum == OrderType.MARKET:
            order = dc_replace(
                order,
                status=OrderStatus.EXECUTED,
                filled_quantity=quantity,
                executed_at=datetime.now(),
            )

        saved_order = self.order_repository.save(order)

        # If filled, update position and cash balance
        if saved_order.status == OrderStatus.EXECUTED and self.position_repository:
            self._settle_fill(saved_order, portfolio)

        return saved_order

    def _settle_fill(self, order: Order, portfolio) -> None:
        """Create/update position and adjust cash after a paper-trade fill."""
        from src.domain.entities.trading import PositionType as PT
        from dataclasses import replace as dc_replace

        TWO_PLACES = Decimal("0.01")
        raw_price = order.price or Money(Decimal("0"), "USD")
        # Ensure fill price is a Money (2-decimal) regardless of source type
        fill_price = Money(raw_price.amount.quantize(TWO_PLACES), "USD")
        trade_value = (fill_price.amount * Decimal(order.quantity)).quantize(TWO_PLACES)

        # Update position
        existing = self.position_repository.get_by_symbol(order.user_id, order.symbol)

        if order.position_type == PT.LONG:
            if existing:
                updated = existing.adjust_quantity(order.quantity, fill_price)
                self.position_repository.update(updated)
            else:
                new_position = Position(
                    id=str(uuid.uuid4()),
                    user_id=order.user_id,
                    symbol=order.symbol,
                    position_type=PT.LONG,
                    quantity=order.quantity,
                    average_buy_price=fill_price,
                    current_price=fill_price,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )
                self.position_repository.save(new_position)
            new_cash = (portfolio.cash_balance.amount - trade_value).quantize(TWO_PLACES)
        else:
            if existing:
                updated = existing.adjust_quantity(-order.quantity, fill_price)
                self.position_repository.update(updated)
            new_cash = (portfolio.cash_balance.amount + trade_value).quantize(TWO_PLACES)

        updated_portfolio = portfolio.update_cash_balance(Money(new_cash, "USD"))
        self.portfolio_repository.update(updated_portfolio)


class ExecuteTradeUseCase:
    """
    Use Case: Execute a trade based on AI signals and market conditions.
    """
    
    def __init__(
        self,
        order_repository: OrderRepositoryPort,
        portfolio_repository: PortfolioRepositoryPort,
        user_repository: UserRepositoryPort,
        trading_service: TradingDomainService,
        risk_service: RiskManagementDomainService,
        market_data_service: MarketDataPort,
        trading_execution_service: TradingExecutionPort,
        notification_service: NotificationPort,
        ai_model_service: AIModelPort
    ):
        self.order_repository = order_repository
        self.portfolio_repository = portfolio_repository
        self.user_repository = user_repository
        self.trading_service = trading_service
        self.risk_service = risk_service
        self.market_data_service = market_data_service
        self.trading_execution_service = trading_execution_service
        self.notification_service = notification_service
        self.ai_model_service = ai_model_service
    
    def execute(self, user_id: str, symbol: Symbol) -> Optional[Order]:
        """
        Execute a trade based on AI signals and market conditions.
        """
        # Retrieve user and portfolio
        user = self.user_repository.get_by_id(user_id)
        if not user:
            raise ValueError(f"User not found: {user_id}")
        
        portfolio = self.portfolio_repository.get_by_user_id(user_id)
        if not portfolio:
            raise ValueError(f"Portfolio not found for user: {user_id}")
        
        # Check risk limits
        risk_violations = self.risk_service.check_portfolio_risk_limits(portfolio, user)
        if risk_violations:
            # Send risk alert
            self.notification_service.send_risk_alert(user, '; '.join(risk_violations))
            raise ValueError(f"Risk limits violated: {'; '.join(risk_violations)}")
        
        # Get AI trading signal
        signal = self.ai_model_service.get_trading_signal(symbol)
        if signal == 'HOLD':
            return None  # No trade to execute
        
        # Get current market data
        current_price = self.market_data_service.get_current_price(symbol)
        if not current_price:
            raise ValueError(f"Could not get current price for symbol: {symbol}")
        
        # Calculate position size based on user constraints
        position_size = self.trading_service.calculate_position_sizing(user, portfolio, symbol)
        
        # Create appropriate order based on signal
        from src.domain.entities.trading import OrderType, PositionType, OrderStatus
        
        order_type = OrderType.MARKET
        position_type = PositionType.LONG if signal == 'BUY' else PositionType.SHORT
        
        order = Order(
            id=str(uuid.uuid4()),
            user_id=user_id,
            symbol=symbol,
            order_type=order_type,
            position_type=position_type,
            quantity=position_size,
            status=OrderStatus.PENDING,
            placed_at=datetime.now(),
            price=current_price,
            stop_price=None,
            filled_quantity=0,
            commission=None,
            notes=f"AI signal: {signal}"
        )
        
        # Validate order
        validation_errors = self.trading_service.validate_order(order, user, portfolio)
        if validation_errors:
            raise ValueError(f"Order validation failed: {'; '.join(validation_errors)}")
        
        # Save the order
        saved_order = self.order_repository.save(order)
        
        # Execute with broker (in real implementation, this might be async)
        broker_order_id = self.trading_execution_service.place_order(saved_order)
        
        # Update order status after broker execution
        # In real implementation, this would be handled by an event or async process
        executed_order = self.trading_service.execute_order(saved_order, current_price)
        updated_order = self.order_repository.save(executed_order)
        
        # Send notification
        self.notification_service.send_trade_notification(user, updated_order)
        
        return updated_order


class AnalyzeNewsSentimentUseCase:
    """
    Use Case: Analyze sentiment of financial news for trading decisions.
    """
    
    def __init__(
        self,
        news_analysis_service: NewsAnalysisPort,
        market_data_service: MarketDataPort,
        portfolio_repository: PortfolioRepositoryPort
    ):
        self.news_analysis_service = news_analysis_service
        self.market_data_service = market_data_service
        self.portfolio_repository = portfolio_repository
    
    def execute(self, symbol: Symbol) -> List[dict]:
        """
        Analyze sentiment of recent news for a symbol.

        Returns:
            List of dictionaries containing article, sentiment, and timestamp
        """
        # Get recent news for the symbol
        news_articles = self.market_data_service.get_market_news(symbol)

        # Analyze sentiment for each article
        sentiment_results = []
        for article in news_articles:
            sentiment = sentiment_analyzer.analyze_sentiment(article)
            sentiment_results.append({
                'article': article,
                'sentiment': sentiment,
                'timestamp': datetime.now()
            })

        # Calculate aggregate sentiment
        if sentiment_results:
            avg_sentiment = sum(item['sentiment'].score for item in sentiment_results) / len(sentiment_results)
            overall_sentiment = type('NewsSentiment', (), {
                'score': avg_sentiment,
                'confidence': 80,
                'source': 'Aggregate'
            })()

            sentiment_results.append({
                'article': 'AGGREGATE',
                'sentiment': overall_sentiment,
                'timestamp': datetime.now()
            })

        return sentiment_results


class GetPortfolioPerformanceUseCase:
    """
    Use Case: Calculate and return portfolio performance metrics.
    """
    
    def __init__(
        self,
        portfolio_repository: PortfolioRepositoryPort,
        market_data_service: MarketDataPort
    ):
        self.portfolio_repository = portfolio_repository
        self.market_data_service = market_data_service
    
    def execute(self, user_id: str) -> dict:
        """
        Calculate portfolio performance metrics for a user.
        """
        portfolio = self.portfolio_repository.get_by_user_id(user_id)
        if not portfolio:
            raise ValueError(f"Portfolio not found for user: {user_id}")
        
        # Update position prices with current market prices
        updated_positions = []
        for pos in portfolio.positions:
            current_price = self.market_data_service.get_current_price(pos.symbol)
            if current_price:
                updated_pos = pos.update_price(current_price)
                updated_positions.append(updated_pos)
            else:
                updated_positions.append(pos)  # Keep original if price unavailable
        
        # Calculate metrics
        total_value = portfolio.total_value
        positions_value = portfolio.positions_value
        
        # Calculate daily/weekly/monthly changes if historical data is available
        # For now, we'll return basic metrics
        return {
            'total_value': float(total_value.amount),
            'cash_balance': float(portfolio.cash_balance.amount),
            'positions_value': float(positions_value.amount),
            'position_count': len(portfolio.positions),
            'positions': [
                {
                    'symbol': str(pos.symbol),
                    'quantity': pos.quantity,
                    'average_buy_price': float(pos.average_buy_price.amount),
                    'current_price': float(pos.current_price.amount),
                    'market_value': float(pos.market_value.amount),
                    'unrealized_pnl': float(pos.unrealized_pnl_amount.amount),
                    'pnl_percentage': float((pos.current_price.amount - pos.average_buy_price.amount) / pos.average_buy_price.amount * 100) if pos.average_buy_price.amount > 0 else 0
                }
                for pos in updated_positions
            ]
        }


class GetUserPreferencesUseCase:
    """
    Use Case: Get and update user preferences and risk settings.
    """
    
    def __init__(
        self,
        user_repository: UserRepositoryPort
    ):
        self.user_repository = user_repository
    
    def get_user_preferences(self, user_id: str) -> Optional[User]:
        """
        Get user preferences and settings.
        """
        return self.user_repository.get_by_id(user_id)
    
    def update_user_risk_tolerance(self, user_id: str, risk_tolerance: str) -> User:
        """
        Update user's risk tolerance.
        """
        user = self.user_repository.get_by_id(user_id)
        if not user:
            raise ValueError(f"User not found: {user_id}")
        
        from src.domain.entities.user import RiskTolerance
        risk_enum = RiskTolerance[risk_tolerance.upper()]
        
        updated_user = user.update_risk_tolerance(risk_enum)
        return self.user_repository.save(updated_user)
    
    def update_user_investment_goal(self, user_id: str, investment_goal: str) -> User:
        """
        Update user's investment goal.
        """
        user = self.user_repository.get_by_id(user_id)
        if not user:
            raise ValueError(f"User not found: {user_id}")
        
        from src.domain.entities.user import InvestmentGoal
        goal_enum = InvestmentGoal[investment_goal.upper()]
        
        updated_user = user.update_investment_goal(goal_enum)
        return self.user_repository.save(updated_user)