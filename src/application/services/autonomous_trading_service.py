"""
Autonomous Trading Service

Application-layer service that orchestrates the autonomous trading loop.
Connects ML predictions, risk management, broker execution, and activity logging.

Architectural Intent:
- Orchestrates domain services without containing business logic itself
- Each user is processed in isolation with independent error handling
- All decisions are logged for auditability
- Respects circuit breaker, risk limits, and confidence thresholds
"""
from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime
from decimal import Decimal
from typing import Optional
import uuid

from src.domain.entities.trading import (
    Order, OrderType, OrderStatus, PositionType, Position,
)
from src.domain.entities.user import User
from src.domain.value_objects import Symbol, Money
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.infrastructure.repositories.user_repository import UserRepository
    from src.infrastructure.repositories.portfolio_repository import PortfolioRepository
    from src.infrastructure.repositories.position_repository import PositionRepository
    from src.infrastructure.repositories.order_repository import OrderRepository
    from src.infrastructure.repositories.activity_log_repository import ActivityLogRepository
    from src.infrastructure.data_processing.ml_model_service import EnsembleModelService
    from src.infrastructure.broker_integration import AlpacaBrokerService
    from src.domain.services.risk_management import RiskManager, CircuitBreakerService
    from src.infrastructure.api_clients.market_data import MarketDataService

logger = logging.getLogger(__name__)

# Activity log event types
SIGNAL_GENERATED = "SIGNAL_GENERATED"
ORDER_PLACED = "ORDER_PLACED"
ORDER_FILLED = "ORDER_FILLED"
ORDER_FAILED = "ORDER_FAILED"
RISK_BLOCKED = "RISK_BLOCKED"
CIRCUIT_BREAKER = "CIRCUIT_BREAKER"


class AutonomousTradingService:
    """
    Orchestrates the autonomous trading loop.

    Two entry points called by the scheduler:
    - run_trading_cycle(): Generate signals and place orders (every 15 min during market hours)
    - poll_pending_orders(): Check broker for fill status (every 5 min)
    """

    def __init__(
        self,
        user_repository: UserRepository,
        portfolio_repository: PortfolioRepository,
        position_repository: PositionRepository,
        order_repository: OrderRepository,
        activity_log_repository: ActivityLogRepository,
        ml_model_service: EnsembleModelService,
        broker_service: AlpacaBrokerService,
        risk_manager: RiskManager,
        circuit_breaker: CircuitBreakerService,
        market_data_service: MarketDataService,
        confidence_threshold: float = 0.6,
    ):
        self.user_repo = user_repository
        self.portfolio_repo = portfolio_repository
        self.position_repo = position_repository
        self.order_repo = order_repository
        self.activity_log = activity_log_repository
        self.ml_service = ml_model_service
        self.broker = broker_service
        self.risk_manager = risk_manager
        self.circuit_breaker = circuit_breaker
        self.market_data = market_data_service
        self.confidence_threshold = confidence_threshold

    # ------------------------------------------------------------------
    # Trading cycle
    # ------------------------------------------------------------------

    def run_trading_cycle(self) -> None:
        """Execute one trading cycle for all auto-trading users."""
        logger.info("Autonomous trading cycle starting")

        # Check circuit breaker first
        if not self.circuit_breaker.is_trading_allowed():
            logger.warning("Circuit breaker active — skipping trading cycle")
            return

        users = self.user_repo.get_auto_trading_users()
        logger.info(f"Processing {len(users)} auto-trading users")

        for user in users:
            try:
                self._process_user(user)
            except Exception as e:
                logger.error(f"Error processing user {user.id}: {e}", exc_info=True)

        logger.info("Autonomous trading cycle complete")

    def _process_user(self, user: User) -> None:
        """Process a single user's trading cycle."""
        portfolio = self.portfolio_repo.get_by_user_id(user.id)
        if not portfolio:
            logger.warning(f"No portfolio for user {user.id}, skipping")
            return

        # Check risk limits
        if self.risk_manager.should_pause_trading(portfolio, user):
            self.activity_log.log_event(
                user_id=user.id,
                event_type=RISK_BLOCKED,
                message="Trading paused — risk limits exceeded",
            )
            return

        # Determine per-symbol budget
        total_budget = (
            user.trading_budget.amount
            if user.trading_budget
            else portfolio.cash_balance.amount
        )
        if not user.watchlist:
            return
        budget_per_symbol = total_budget / Decimal(len(user.watchlist))

        for ticker in user.watchlist:
            try:
                self._process_symbol(user, portfolio, ticker, budget_per_symbol)
            except Exception as e:
                logger.error(
                    f"Error processing {ticker} for user {user.id}: {e}",
                    exc_info=True,
                )

    def _process_symbol(
        self,
        user: User,
        portfolio,
        ticker: str,
        budget_per_symbol: Decimal,
    ) -> None:
        """Generate signal and potentially place an order for one symbol."""
        symbol = Symbol(ticker)

        # Get ML signal
        signal = self.ml_service.predict_price_direction(symbol)
        signal_str = signal.signal.upper()
        confidence = signal.confidence

        self.activity_log.log_event(
            user_id=user.id,
            event_type=SIGNAL_GENERATED,
            symbol=ticker,
            signal=signal_str,
            confidence=confidence,
            message=signal.explanation,
        )

        # Skip HOLD or low-confidence signals
        if signal_str == "HOLD" or confidence < self.confidence_threshold:
            return

        if signal_str in ("BUY", "STRONG_BUY"):
            self._handle_buy(user, portfolio, symbol, ticker, budget_per_symbol, signal)
        elif signal_str in ("SELL", "STRONG_SELL"):
            self._handle_sell(user, portfolio, symbol, ticker, signal)

    # ------------------------------------------------------------------
    # Order handling
    # ------------------------------------------------------------------

    def _handle_buy(
        self, user, portfolio, symbol, ticker, budget, signal
    ) -> None:
        current_price = self.market_data.get_current_price(symbol)
        if not current_price or current_price.amount <= 0:
            logger.warning(f"No price available for {ticker}")
            return

        shares = int(budget / current_price.amount)
        if shares <= 0:
            return

        price_money = Money(current_price.amount, "USD")
        order = Order(
            id=str(uuid.uuid4()),
            user_id=user.id,
            symbol=symbol,
            order_type=OrderType.MARKET,
            position_type=PositionType.LONG,
            quantity=shares,
            status=OrderStatus.PENDING,
            placed_at=datetime.utcnow(),
            price=price_money,
        )

        # Risk check
        errors = self.risk_manager.validate_order(order, user, portfolio)
        if errors:
            self.activity_log.log_event(
                user_id=user.id,
                event_type=RISK_BLOCKED,
                symbol=ticker,
                signal=signal.signal,
                confidence=signal.confidence,
                message=f"Risk blocked: {'; '.join(errors)}",
            )
            return

        self._submit_order(user, order, signal)

    def _handle_sell(self, user, portfolio, symbol, ticker, signal) -> None:
        position = portfolio.get_position(symbol)
        if not position or position.quantity <= 0:
            return

        current_price = self.market_data.get_current_price(symbol)
        price_money = (
            Money(current_price.amount, "USD")
            if current_price
            else position.current_price
        )

        order = Order(
            id=str(uuid.uuid4()),
            user_id=user.id,
            symbol=symbol,
            order_type=OrderType.MARKET,
            position_type=PositionType.SHORT,
            quantity=position.quantity,
            status=OrderStatus.PENDING,
            placed_at=datetime.utcnow(),
            price=price_money,
        )
        self._submit_order(user, order, signal)

    def _submit_order(self, user, order, signal) -> None:
        """Place order with broker and persist."""
        try:
            response = self.broker.place_order(order)
            order = replace(order, broker_order_id=response.broker_order_id)
            self.order_repo.save(order)

            self.activity_log.log_event(
                user_id=user.id,
                event_type=ORDER_PLACED,
                symbol=str(order.symbol),
                signal=signal.signal,
                confidence=signal.confidence,
                order_id=order.id,
                broker_order_id=response.broker_order_id,
                quantity=order.quantity,
                price=order.price.amount if order.price else None,
                message=f"{order.position_type.value} {order.quantity} {order.symbol} "
                        f"@ {order.price}",
            )
        except Exception as e:
            self.activity_log.log_event(
                user_id=user.id,
                event_type=ORDER_FAILED,
                symbol=str(order.symbol),
                message="Order submission failed",
            )
            logger.error(f"Failed to submit order for {order.symbol}: {e}")

    # ------------------------------------------------------------------
    # Order polling
    # ------------------------------------------------------------------

    def poll_pending_orders(self) -> None:
        """Check broker for status updates on all pending orders."""
        logger.info("Polling pending orders")
        pending_orders = self.order_repo.get_pending_with_broker_id()
        logger.info(f"Found {len(pending_orders)} pending orders to poll")

        for order in pending_orders:
            try:
                self._check_order_status(order)
            except Exception as e:
                logger.error(
                    f"Error polling order {order.id}: {e}", exc_info=True
                )

    def _check_order_status(self, order: Order) -> None:
        """Check a single order's status with the broker and update accordingly."""
        status = self.broker.get_order_status(order.broker_order_id)

        if status == "filled":
            filled_order = replace(
                order,
                status=OrderStatus.EXECUTED,
                filled_quantity=order.quantity,
                executed_at=datetime.utcnow(),
            )
            self.order_repo.update_order(filled_order)

            # Upsert position
            self._upsert_position_on_fill(filled_order)

            # Update portfolio cash
            self._update_portfolio_cash(filled_order)

            self.activity_log.log_event(
                user_id=order.user_id,
                event_type=ORDER_FILLED,
                symbol=str(order.symbol),
                order_id=order.id,
                broker_order_id=order.broker_order_id,
                quantity=order.quantity,
                price=order.price.amount if order.price else None,
                message=f"Order filled: {order.position_type.value} {order.quantity} {order.symbol}",
            )

        elif status in ("cancelled", "rejected", "expired"):
            failed_order = replace(order, status=OrderStatus.FAILED)
            self.order_repo.update_order(failed_order)

            self.activity_log.log_event(
                user_id=order.user_id,
                event_type=ORDER_FAILED,
                symbol=str(order.symbol),
                order_id=order.id,
                broker_order_id=order.broker_order_id,
                message=f"Order {status}: {order.symbol}",
            )

    def _upsert_position_on_fill(self, order: Order) -> None:
        """Create or update a position after an order fills."""
        symbol = order.symbol
        existing = self.position_repo.get_by_symbol(order.user_id, symbol)
        fill_price = order.price or Money(Decimal("0"), "USD")

        if order.position_type == PositionType.LONG:
            # Buying — create or add to position
            if existing:
                updated = existing.adjust_quantity(order.quantity, fill_price)
                self.position_repo.update(updated)
            else:
                new_position = Position(
                    id=str(uuid.uuid4()),
                    user_id=order.user_id,
                    symbol=symbol,
                    position_type=PositionType.LONG,
                    quantity=order.quantity,
                    average_buy_price=fill_price,
                    current_price=fill_price,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
                self.position_repo.save(new_position)
        else:
            # Selling — reduce or close position
            if existing:
                updated = existing.adjust_quantity(-order.quantity, fill_price)
                self.position_repo.update(updated)

    def _update_portfolio_cash(self, order: Order) -> None:
        """Adjust portfolio cash balance after a fill."""
        portfolio = self.portfolio_repo.get_by_user_id(order.user_id)
        if not portfolio or not order.price:
            return

        trade_value = order.price.amount * Decimal(order.quantity)

        if order.position_type == PositionType.LONG:
            new_cash = portfolio.cash_balance.amount - trade_value
        else:
            new_cash = portfolio.cash_balance.amount + trade_value

        updated = portfolio.update_cash_balance(Money(new_cash, "USD"))
        self.portfolio_repo.update(updated)
