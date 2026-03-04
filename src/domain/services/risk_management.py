"""
Risk Management System

Architectural Intent:
- This module implements the risk management functionality required by the PRD,
  including pre-trade risk checks, real-time monitoring, and automated controls.
- All classes follow DDD principles: domain services are pure business logic,
  free from infrastructure concerns (no settings imports, no threading, no I/O).
- AccountState is an immutable frozen dataclass; all mutations return new instances.
- RiskManager, StopLossService, and CircuitBreakerService accept configuration
  via constructor parameters (dependency injection), never from infrastructure.
- Scheduling and thread management belong in the application/infrastructure layers.

Key Design Decisions:
1. Risk limits are injected as a dict keyed by RiskTolerance enum values.
2. CircuitBreakerService accepts volatility_threshold and reset_after as constructor params.
3. StopLossService uses the Price value object instead of dynamic mock objects.
4. All timestamps use datetime.now(timezone.utc) for consistency.
5. Logging replaces all print() statements for production observability.
"""
import logging
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Callable, Dict, List, Optional, Tuple

from src.domain.entities.trading import (
    Order,
    OrderStatus,
    OrderType,
    Portfolio,
    Position,
    PositionType,
)
from src.domain.entities.user import RiskTolerance, User
from src.domain.ports import NotificationPort
from src.domain.services.trading import RiskManagementDomainService
from src.domain.value_objects import Money, Price, Symbol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AccountState:
    """
    Immutable snapshot of an account's state for risk monitoring purposes.

    Invariants:
    - peak_value is always >= 0
    - value_history is an immutable tuple of (datetime, Decimal) pairs
    - All state transitions produce a new AccountState instance
    """

    user: User
    portfolio: Portfolio
    current_value: Money
    peak_value: Decimal
    value_history: Tuple[Tuple[datetime, Decimal], ...] = ()
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @staticmethod
    def create(user: User, portfolio: Portfolio) -> "AccountState":
        """
        Factory method to create a new AccountState from a user and portfolio.
        """
        now = datetime.now(timezone.utc)
        return AccountState(
            user=user,
            portfolio=portfolio,
            current_value=portfolio.total_value,
            peak_value=portfolio.total_value.amount,
            value_history=((now, portfolio.total_value.amount),),
            last_update=now,
        )

    def update_portfolio(self, new_portfolio: Portfolio) -> "AccountState":
        """
        Return a new AccountState reflecting an updated portfolio.

        Keeps the last 100 history entries for memory efficiency.
        """
        now = datetime.now(timezone.utc)
        new_peak = max(self.peak_value, new_portfolio.total_value.amount)

        # Append to history, keeping last 100 entries
        updated_history = self.value_history + ((now, new_portfolio.total_value.amount),)
        if len(updated_history) > 100:
            updated_history = updated_history[-100:]

        return replace(
            self,
            portfolio=new_portfolio,
            current_value=new_portfolio.total_value,
            peak_value=new_peak,
            value_history=updated_history,
            last_update=now,
        )

    def calculate_drawdown(self) -> Decimal:
        """
        Calculate the current drawdown percentage from peak value.

        Returns a non-negative Decimal representing the percentage drawdown.
        """
        if self.peak_value <= 0:
            return Decimal("0")

        current = self.current_value.amount
        drawdown = ((self.peak_value - current) / self.peak_value) * 100
        return max(Decimal("0"), drawdown)


class RiskManager(RiskManagementDomainService):
    """
    Risk Management Service implementing the RiskManagementDomainService interface.

    This service implements all the risk management requirements from the PRD:
    - Pre-trade risk checks
    - Real-time position monitoring
    - Portfolio-level risk controls

    Architectural Intent:
    - risk_limits are injected via the constructor (no infrastructure dependency).
    - Monitoring loops are NOT managed here; the application layer is responsible
      for scheduling calls to check_account_risks.
    - All side-effect communication goes through the NotificationPort.
    """

    def __init__(
        self,
        notification_service: NotificationPort,
        risk_limits: Optional[Dict[RiskTolerance, dict]] = None,
    ):
        self.notification_service = notification_service
        self.risk_limits = risk_limits or self._default_risk_limits()

    @staticmethod
    def _default_risk_limits() -> Dict[RiskTolerance, dict]:
        """
        Provide sensible default risk limits when none are injected.

        In production, these should be supplied by the infrastructure/config layer
        via constructor injection.
        """
        return {
            RiskTolerance.CONSERVATIVE: {
                "max_drawdown": Decimal("10.0"),
                "position_limit_percentage": Decimal("5.0"),
                "volatility_threshold": Decimal("2.0"),
            },
            RiskTolerance.MODERATE: {
                "max_drawdown": Decimal("15.0"),
                "position_limit_percentage": Decimal("10.0"),
                "volatility_threshold": Decimal("2.5"),
            },
            RiskTolerance.AGGRESSIVE: {
                "max_drawdown": Decimal("25.0"),
                "position_limit_percentage": Decimal("20.0"),
                "volatility_threshold": Decimal("3.0"),
            },
        }

    # ------------------------------------------------------------------
    # Account-level risk checks (called by the application layer scheduler)
    # ------------------------------------------------------------------

    def check_account_risks(
        self, user_id: str, account_state: AccountState
    ) -> List[str]:
        """
        Check for risk violations for a specific account.

        Returns a list of triggered risk violation messages.
        The application layer should call this periodically.
        """
        violations: List[str] = []

        if self._is_drawdown_limit_exceeded(account_state):
            msg = self._trigger_drawdown_protection(user_id, account_state)
            violations.append(msg)

        if self._is_position_concentration_exceeded(account_state):
            msg = self._trigger_concentration_protection(user_id, account_state)
            violations.append(msg)

        if self._is_volatility_threshold_exceeded(account_state):
            msg = self._trigger_volatility_protection(user_id, account_state)
            violations.append(msg)

        return violations

    def _is_drawdown_limit_exceeded(self, account_state: AccountState) -> bool:
        """Check if drawdown limits are exceeded."""
        if not account_state.peak_value or account_state.peak_value <= 0:
            return False

        current_value = account_state.current_value.amount
        drawdown_pct = (
            (account_state.peak_value - current_value) / account_state.peak_value
        ) * 100

        user_limits = self.risk_limits[account_state.user.risk_tolerance]
        return drawdown_pct > user_limits["max_drawdown"]

    def _is_position_concentration_exceeded(
        self, account_state: AccountState
    ) -> bool:
        """Check if position concentration limits are exceeded."""
        if (
            not account_state.portfolio
            or account_state.portfolio.total_value.amount <= 0
        ):
            return False

        portfolio_value = account_state.portfolio.total_value.amount

        for position in account_state.portfolio.positions:
            position_value = position.market_value.amount
            position_pct = (position_value / portfolio_value) * 100

            user_limits = self.risk_limits[account_state.user.risk_tolerance]
            if position_pct > user_limits["position_limit_percentage"]:
                return True

        return False

    def _is_volatility_threshold_exceeded(
        self, account_state: AccountState
    ) -> bool:
        """
        Check if market volatility thresholds are exceeded.

        This would require market data to calculate volatility.
        Placeholder returning False until market data port is integrated.
        """
        return False

    def _trigger_drawdown_protection(
        self, user_id: str, account_state: AccountState
    ) -> str:
        """Trigger drawdown protection measures and return the alert message."""
        message = (
            f"Drawdown protection triggered for account {user_id}. "
            "Current drawdown exceeds limits."
        )
        self.notification_service.send_risk_alert(account_state.user, message)
        logger.warning(message)
        return message

    def _trigger_concentration_protection(
        self, user_id: str, account_state: AccountState
    ) -> str:
        """Trigger position concentration protection measures and return the alert message."""
        message = (
            f"Position concentration protection triggered for account {user_id}. "
            "Position size exceeds limits."
        )
        self.notification_service.send_risk_alert(account_state.user, message)
        logger.warning(message)
        return message

    def _trigger_volatility_protection(
        self, user_id: str, account_state: AccountState
    ) -> str:
        """Trigger volatility protection measures and return the alert message."""
        message = (
            f"Volatility protection triggered for account {user_id}. "
            "Market volatility exceeds thresholds."
        )
        self.notification_service.send_risk_alert(account_state.user, message)
        logger.warning(message)
        return message

    # ------------------------------------------------------------------
    # Interface methods from RiskManagementDomainService
    # ------------------------------------------------------------------

    def validate_order(
        self, order: Order, user: User, portfolio: Portfolio
    ) -> List[str]:
        """
        Perform pre-trade risk validation.

        Implements the interface method from RiskManagementDomainService.
        """
        errors: List[str] = []

        # Check if order value exceeds position limits
        if order.price:
            order_value = order.price.amount * Decimal(order.quantity)
            portfolio_value = portfolio.total_value.amount

            if portfolio_value > 0:
                order_pct = (order_value / portfolio_value) * 100
                user_limits = self.risk_limits[user.risk_tolerance]

                if order_pct > user_limits["position_limit_percentage"]:
                    errors.append(
                        f"Order value ({order_pct:.2f}%) exceeds position limit "
                        f"({user_limits['position_limit_percentage']}%)"
                    )

        # Check if user has sufficient cash for the order
        required_cash = Decimal("0")
        if order.position_type.name == "LONG":
            required_cash = (
                order.price.amount * Decimal(order.quantity)
                if order.price
                else Decimal("0")
            )

        if required_cash > portfolio.cash_balance.amount:
            errors.append("Insufficient cash balance for this order")

        # Check if the symbol is in user's excluded sectors (simplified check)
        if user.sector_exclusions:
            # This would require mapping symbols to sectors
            # For now, we skip this check
            pass

        return errors

    def check_portfolio_risk_limits(
        self, portfolio: Portfolio, user: User
    ) -> List[str]:
        """
        Check if the portfolio violates any of the user's risk limits.

        Implements the interface method from RiskManagementDomainService.
        """
        errors: List[str] = []
        portfolio_value = portfolio.total_value.amount

        # Check daily loss limit
        if user.daily_loss_limit and portfolio_value < (
            portfolio_value + user.daily_loss_limit.amount
        ):
            errors.append(f"Daily loss limit exceeded: {user.daily_loss_limit}")

        # Check weekly loss limit
        if user.weekly_loss_limit and portfolio_value < (
            portfolio_value + user.weekly_loss_limit.amount
        ):
            errors.append(f"Weekly loss limit exceeded: {user.weekly_loss_limit}")

        # Check monthly loss limit
        if user.monthly_loss_limit and portfolio_value < (
            portfolio_value + user.monthly_loss_limit.amount
        ):
            errors.append(f"Monthly loss limit exceeded: {user.monthly_loss_limit}")

        # Check position concentration
        for position in portfolio.positions:
            position_value = position.market_value.amount
            position_pct = (position_value / portfolio_value) * 100

            user_limits = self.risk_limits[user.risk_tolerance]
            if position_pct > user_limits["position_limit_percentage"]:
                errors.append(
                    f"Position concentration limit exceeded for {position.symbol}: "
                    f"{position_pct:.2f}% > {user_limits['position_limit_percentage']}%"
                )

        return errors

    def should_pause_trading(self, portfolio: Portfolio, user: User) -> bool:
        """
        Determine if trading should be paused based on risk conditions.

        Implements the interface method from RiskManagementDomainService.
        """
        risk_violations = self.check_portfolio_risk_limits(portfolio, user)
        return len(risk_violations) > 0


class StopLossService:
    """
    Service to manage stop-loss orders.

    Architectural Intent:
    - Provides pure business logic for stop-loss evaluation and order creation.
    - Does NOT manage its own monitoring threads; the application layer calls
      check_stop_losses() on a schedule.
    - Uses the Price value object from the domain layer for stop prices.
    """

    def __init__(self, trading_service, notification_service: NotificationPort):
        self.trading_service = trading_service
        self.notification_service = notification_service
        self.stop_losses: Dict[str, dict] = {}

    def check_stop_losses(self) -> List[str]:
        """
        Check if any stop-loss conditions are met and execute triggered orders.

        Returns a list of position IDs that were triggered.
        The application layer should call this periodically.
        """
        current_time = datetime.now(timezone.utc)
        triggered_position_ids: List[str] = []

        for position_id, stop_loss_config in list(self.stop_losses.items()):
            position: Position = stop_loss_config["position"]
            stop_price: Price = stop_loss_config["stop_price"]
            current_price_func: Callable[[], Optional[Price]] = stop_loss_config[
                "current_price_func"
            ]
            current_price = current_price_func()

            if current_price and current_price.amount <= stop_price.amount:
                # Stop-loss triggered, create sell order
                stop_loss_order = Order(
                    id=str(uuid.uuid4()),
                    user_id=position.user_id,
                    symbol=position.symbol,
                    order_type=OrderType.MARKET,
                    position_type=PositionType.SHORT,
                    quantity=position.quantity,
                    status=OrderStatus.PENDING,
                    placed_at=current_time,
                    price=None,
                    stop_price=Money(stop_price.amount, stop_price.currency),
                    filled_quantity=0,
                    commission=None,
                    notes=(
                        f"Stop-loss triggered for position {position.id} "
                        f"at price {current_price.amount}"
                    ),
                )

                try:
                    self.trading_service.submit_order(stop_loss_order)
                    triggered_position_ids.append(position_id)

                    message = (
                        f"Stop-loss triggered for {position.symbol} "
                        f"at ${current_price.amount}"
                    )
                    user = stop_loss_config.get("user")
                    if user:
                        self.notification_service.send_risk_alert(user, message)

                    logger.info(message)
                except Exception as e:
                    logger.error(
                        "Error executing stop-loss for position %s: %s",
                        position_id,
                        e,
                    )

        # Remove triggered stop-losses
        for position_id in triggered_position_ids:
            self.stop_losses.pop(position_id, None)

        return triggered_position_ids

    def add_stop_loss(
        self,
        position_id: str,
        position: Position,
        stop_percentage: Decimal,
        user: User,
        current_price_func: Callable[[], Optional[Price]],
    ) -> bool:
        """
        Add a stop-loss for a position.

        The stop price is calculated as stop_percentage below the position's
        current price and stored as a proper Price value object.
        """
        try:
            stop_price_value = position.current_price.amount * (
                1 - stop_percentage / 100
            )
            stop_price = Price(
                amount=stop_price_value,
                currency=position.current_price.currency,
            )

            self.stop_losses[position_id] = {
                "position": position,
                "stop_price": stop_price,
                "current_price_func": current_price_func,
                "user": user,
                "created_at": datetime.now(timezone.utc),
            }

            logger.info(
                "Stop-loss added for position %s at price %s",
                position_id,
                stop_price.amount,
            )
            return True
        except Exception as e:
            logger.error(
                "Error adding stop-loss for position %s: %s", position_id, e
            )
            return False

    def remove_stop_loss(self, position_id: str) -> bool:
        """Remove a stop-loss for a position."""
        if position_id in self.stop_losses:
            del self.stop_losses[position_id]
            logger.info("Stop-loss removed for position %s", position_id)
            return True
        return False


class CircuitBreakerService:
    """
    Service to implement circuit breakers for extreme market conditions.

    Architectural Intent:
    - Accepts volatility_threshold and reset_after as constructor params
      (no infrastructure config dependency).
    - Does NOT manage its own monitoring threads; the application layer calls
      check_circuit_breaker() on a schedule.
    - Provides is_trading_allowed() as a pure query method.
    """

    def __init__(
        self,
        notification_service: NotificationPort,
        volatility_threshold: Decimal = Decimal("5.0"),
        reset_after: timedelta = timedelta(minutes=30),
    ):
        self.notification_service = notification_service
        self.extreme_volatility_threshold = volatility_threshold
        self.reset_after = reset_after
        self.circuit_breaker_triggered: bool = False
        self.trigger_time: Optional[datetime] = None

    def check_circuit_breaker(self) -> Optional[str]:
        """
        Check circuit breaker conditions and take action if needed.

        Returns a message string if a state transition occurred, None otherwise.
        The application layer should call this periodically.
        """
        if self.circuit_breaker_triggered:
            now = datetime.now(timezone.utc)
            if self.trigger_time and (now - self.trigger_time) > self.reset_after:
                return self._reset_circuit_breaker()
        else:
            if self._is_extreme_market_volatility():
                return self._trigger_circuit_breaker()

        return None

    def _is_extreme_market_volatility(self) -> bool:
        """
        Check if there is extreme market volatility that should trigger the circuit breaker.

        This would check market indices or other broad market indicators.
        Placeholder returning False until market data port is integrated.
        """
        return False

    def _trigger_circuit_breaker(self) -> str:
        """Trigger the circuit breaker and return the alert message."""
        self.circuit_breaker_triggered = True
        self.trigger_time = datetime.now(timezone.utc)

        message = (
            "Circuit breaker activated due to extreme market volatility. "
            "Trading paused."
        )
        logger.warning(message)
        return message

    def _reset_circuit_breaker(self) -> str:
        """Reset the circuit breaker and return the status message."""
        self.circuit_breaker_triggered = False
        self.trigger_time = None

        message = "Circuit breaker reset. Trading resumed."
        logger.info(message)
        return message

    def is_trading_allowed(self) -> bool:
        """Check if trading is allowed (circuit breaker not triggered)."""
        return not self.circuit_breaker_triggered
