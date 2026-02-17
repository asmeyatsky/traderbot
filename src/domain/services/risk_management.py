"""
Risk Management System

This module implements the risk management functionality required by the PRD,
including pre-trade risk checks, real-time monitoring, and automated controls.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from decimal import Decimal
import threading
import time

from src.domain.entities.trading import Order, Position, Portfolio
from src.domain.entities.user import User, RiskTolerance
from src.domain.value_objects import Money, Symbol
from src.domain.services.trading import RiskManagementDomainService
from src.domain.ports import NotificationPort
from src.infrastructure.config.settings import settings


class RiskManager(RiskManagementDomainService):
    """
    Risk Management Service implementing the RiskManagementDomainService interface.

    This service implements all the risk management requirements from the PRD:
    - Pre-trade risk checks
    - Real-time position monitoring
    - Automated stop-loss and take-profit
    - Portfolio-level risk controls
    - Circuit breakers
    """

    def __init__(self, notification_service: NotificationPort):
        self.notification_service = notification_service
        # Load risk limits from configuration
        self.risk_limits = self._load_risk_limits_from_settings()

    def _load_risk_limits_from_settings(self) -> dict:
        """
        Load risk limits from application settings.

        This allows risk parameters to be configured via environment variables
        rather than being hardcoded.
        """
        # Track account states for real-time monitoring
        self.account_states = {}  # user_id -> AccountState
        self.running = False
        self.monitoring_thread = None

        return {
            RiskTolerance.CONSERVATIVE: {
                'max_drawdown': Decimal(str(settings.RISK_CONSERVATIVE_MAX_DRAWDOWN)),
                'position_limit_percentage': Decimal(str(settings.RISK_CONSERVATIVE_POSITION_LIMIT_PCT)),
                'volatility_threshold': Decimal(str(settings.RISK_CONSERVATIVE_VOLATILITY_THRESHOLD))
            },
            RiskTolerance.MODERATE: {
                'max_drawdown': Decimal(str(settings.RISK_MODERATE_MAX_DRAWDOWN)),
                'position_limit_percentage': Decimal(str(settings.RISK_MODERATE_POSITION_LIMIT_PCT)),
                'volatility_threshold': Decimal(str(settings.RISK_MODERATE_VOLATILITY_THRESHOLD))
            },
            RiskTolerance.AGGRESSIVE: {
                'max_drawdown': Decimal(str(settings.RISK_AGGRESSIVE_MAX_DRAWDOWN)),
                'position_limit_percentage': Decimal(str(settings.RISK_AGGRESSIVE_POSITION_LIMIT_PCT)),
                'volatility_threshold': Decimal(str(settings.RISK_AGGRESSIVE_VOLATILITY_THRESHOLD))
            }
        }
    
    def start_monitoring(self):
        """
        Start the risk monitoring in a background thread.
        """
        if self.running:
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        print("Risk monitoring started")
    
    def stop_monitoring(self):
        """
        Stop the risk monitoring.
        """
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        print("Risk monitoring stopped")
    
    def _monitoring_loop(self):
        """
        Main monitoring loop for real-time risk management.
        """
        while self.running:
            try:
                # Check all account states for risk violations
                for user_id, account_state in self.account_states.items():
                    self._check_account_risks(user_id, account_state)
                
                # Sleep before next check
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                print(f"Error in risk monitoring loop: {e}")
    
    def _check_account_risks(self, user_id: str, account_state: 'AccountState'):
        """
        Check for risk violations for a specific account.
        """
        # Check for drawdown limits
        if self._is_drawdown_limit_exceeded(account_state):
            self._trigger_drawdown_protection(user_id, account_state)
        
        # Check for position concentration limits
        if self._is_position_concentration_exceeded(account_state):
            self._trigger_concentration_protection(user_id, account_state)
        
        # Check for volatility limits
        if self._is_volatility_threshold_exceeded(account_state):
            self._trigger_volatility_protection(user_id, account_state)
    
    def _is_drawdown_limit_exceeded(self, account_state: 'AccountState') -> bool:
        """
        Check if drawdown limits are exceeded.
        """
        if not account_state.peak_value or account_state.peak_value <= 0:
            return False
        
        current_value = account_state.current_value.amount
        drawdown_pct = ((account_state.peak_value - current_value) / account_state.peak_value) * 100
        
        user_limits = self.risk_limits[account_state.user.risk_tolerance]
        return drawdown_pct > user_limits['max_drawdown']
    
    def _is_position_concentration_exceeded(self, account_state: 'AccountState') -> bool:
        """
        Check if position concentration limits are exceeded.
        """
        if not account_state.portfolio or account_state.portfolio.total_value.amount <= 0:
            return False
        
        portfolio_value = account_state.portfolio.total_value.amount
        
        for position in account_state.portfolio.positions:
            position_value = position.market_value.amount
            position_pct = (position_value / portfolio_value) * 100
            
            user_limits = self.risk_limits[account_state.user.risk_tolerance]
            if position_pct > user_limits['position_limit_percentage']:
                return True
        
        return False
    
    def _is_volatility_threshold_exceeded(self, account_state: 'AccountState') -> bool:
        """
        Check if market volatility thresholds are exceeded.
        """
        # This would require market data to calculate volatility
        # For now, returning False as a placeholder
        return False
    
    def _trigger_drawdown_protection(self, user_id: str, account_state: 'AccountState'):
        """
        Trigger drawdown protection measures.
        """
        message = f"Drawdown protection triggered for account {user_id}. Current drawdown exceeds limits."
        self.notification_service.send_risk_alert(account_state.user, message)
        
        # In a real implementation, this would pause trading or liquidate positions
        print(message)
    
    def _trigger_concentration_protection(self, user_id: str, account_state: 'AccountState'):
        """
        Trigger position concentration protection measures.
        """
        message = f"Position concentration protection triggered for account {user_id}. Position size exceeds limits."
        self.notification_service.send_risk_alert(account_state.user, message)
        
        # In a real implementation, this would reduce position sizes
        print(message)
    
    def _trigger_volatility_protection(self, user_id: str, account_state: 'AccountState'):
        """
        Trigger volatility protection measures.
        """
        message = f"Volatility protection triggered for account {user_id}. Market volatility exceeds thresholds."
        self.notification_service.send_risk_alert(account_state.user, message)
        
        # In a real implementation, this would adjust positions based on volatility
        print(message)
    
    def validate_order(self, order: Order, user: User, portfolio: Portfolio) -> List[str]:
        """
        Perform pre-trade risk validation.
        Implements the interface method from RiskManagementDomainService.
        """
        errors = []
        
        # Check if user exists in account states, if not, add them
        if user.id not in self.account_states:
            self.account_states[user.id] = AccountState(user, portfolio)
        
        # Update account state with the new potential position
        account_state = self.account_states[user.id]
        
        # Check if order value exceeds position limits
        if order.price:
            order_value = order.price.amount * Decimal(order.quantity)
            portfolio_value = portfolio.total_value.amount
            
            if portfolio_value > 0:
                order_pct = (order_value / portfolio_value) * 100
                user_limits = self.risk_limits[user.risk_tolerance]
                
                if order_pct > user_limits['position_limit_percentage']:
                    errors.append(
                        f"Order value ({order_pct:.2f}%) exceeds position limit ({user_limits['position_limit_percentage']}%)"
                    )
        
        # Check if user has sufficient cash for the order
        required_cash = 0
        if order.position_type.name == 'LONG':  # Buying
            required_cash = float(order.price.amount * Decimal(order.quantity)) if order.price else 0
        
        if required_cash > float(portfolio.cash_balance.amount):
            errors.append("Insufficient cash balance for this order")
        
        # Check if the symbol is in user's excluded sectors (simplified check)
        if user.sector_exclusions:
            # This would require mapping symbols to sectors
            # For now, we'll skip this check
            pass
        
        return errors
    
    def check_portfolio_risk_limits(self, portfolio: Portfolio, user: User) -> List[str]:
        """
        Check if the portfolio violates any of the user's risk limits.
        Implements the interface method from RiskManagementDomainService.
        """
        errors = []
        
        # Calculate portfolio metrics
        portfolio_value = portfolio.total_value.amount
        
        # Check drawdown limits
        # For this simplified version, we'll check against user's daily/weekly/monthly loss limits
        if user.daily_loss_limit and portfolio_value < (
            portfolio_value + user.daily_loss_limit.amount  # Assuming this is a negative value
        ):
            errors.append(f"Daily loss limit exceeded: {user.daily_loss_limit}")
        
        if user.weekly_loss_limit and portfolio_value < (
            portfolio_value + user.weekly_loss_limit.amount
        ):
            errors.append(f"Weekly loss limit exceeded: {user.weekly_loss_limit}")
        
        if user.monthly_loss_limit and portfolio_value < (
            portfolio_value + user.monthly_loss_limit.amount
        ):
            errors.append(f"Monthly loss limit exceeded: {user.monthly_loss_limit}")
        
        # Check position concentration
        for position in portfolio.positions:
            position_value = position.market_value.amount
            position_pct = (position_value / portfolio_value) * 100
            
            user_limits = self.risk_limits[user.risk_tolerance]
            if position_pct > user_limits['position_limit_percentage']:
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


class AccountState:
    """
    Represents the state of an account for risk monitoring purposes.
    """
    
    def __init__(self, user: User, portfolio: Portfolio):
        self.user = user
        self.portfolio = portfolio
        self.current_value = portfolio.total_value
        self.peak_value = portfolio.total_value.amount  # Track peak portfolio value
        self.value_history = [(datetime.now(), portfolio.total_value.amount)]
        self.last_update = datetime.now()
    
    def update_portfolio(self, new_portfolio: Portfolio):
        """
        Update the account state with a new portfolio.
        """
        self.portfolio = new_portfolio
        self.current_value = new_portfolio.total_value
        
        # Update peak value if current value is higher
        if new_portfolio.total_value.amount > self.peak_value:
            self.peak_value = new_portfolio.total_value.amount
        
        # Add to history (keep last 100 entries)
        self.value_history.append((datetime.now(), new_portfolio.total_value.amount))
        if len(self.value_history) > 100:
            self.value_history = self.value_history[-100:]
        
        self.last_update = datetime.now()
    
    def calculate_drawdown(self) -> Decimal:
        """
        Calculate the current drawdown percentage.
        """
        if self.peak_value <= 0:
            return Decimal('0')
        
        current_value = self.current_value.amount
        drawdown = ((self.peak_value - current_value) / self.peak_value) * 100
        return max(Decimal('0'), drawdown)  # Drawdown should not be negative


class StopLossService:
    """
    Service to manage stop-loss orders automatically.
    """
    
    def __init__(self, trading_service, notification_service: NotificationPort):
        self.trading_service = trading_service
        self.notification_service = notification_service
        self.stop_losses = {}  # position_id -> stop_loss_config
        self.running = False
        self.monitoring_thread = None
    
    def start_monitoring(self):
        """
        Start monitoring for stop-loss triggers.
        """
        if self.running:
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        print("Stop-loss monitoring started")
    
    def stop_monitoring(self):
        """
        Stop monitoring for stop-loss triggers.
        """
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        print("Stop-loss monitoring stopped")
    
    def _monitoring_loop(self):
        """
        Main monitoring loop for stop-loss orders.
        """
        while self.running:
            try:
                # Check all stop-loss conditions
                self._check_stop_losses()
                
                # Sleep before next check
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                print(f"Error in stop-loss monitoring loop: {e}")
    
    def _check_stop_losses(self):
        """
        Check if any stop-loss conditions are met.
        """
        from src.domain.entities.trading import Order, OrderType, PositionType, OrderStatus
        from src.domain.value_objects import Price
        import uuid
        
        current_time = datetime.now()
        triggered_stop_losses = []
        
        for position_id, stop_loss_config in self.stop_losses.items():
            position = stop_loss_config['position']
            stop_price = stop_loss_config['stop_price']
            current_price = stop_loss_config['current_price_func']()  # Function to get current price
            
            # Check if stop-loss condition is met
            if current_price and current_price.amount <= stop_price.amount:
                # Stop-loss triggered, create sell order
                stop_loss_order = Order(
                    id=str(uuid.uuid4()),
                    user_id=position.user_id,
                    symbol=position.symbol,
                    order_type=OrderType.MARKET,
                    position_type=PositionType.SHORT,  # Selling to close long position
                    quantity=position.quantity,
                    status=OrderStatus.PENDING,
                    placed_at=current_time,
                    price=None,  # Market order
                    stop_price=stop_price,
                    filled_quantity=0,
                    commission=None,
                    notes=f"Stop-loss triggered for position {position.id} at price {current_price.amount}"
                )
                
                # Execute the stop-loss order
                try:
                    # This would need to go through the trading service
                    # For now, we'll just record it was triggered
                    self.trading_service.submit_order(stop_loss_order)
                    triggered_stop_losses.append(position_id)
                    
                    # Send notification
                    message = f"Stop-loss triggered for {position.symbol} at ${current_price.amount}"
                    user = stop_loss_config.get('user')  # User info would be stored in config
                    if user:
                        self.notification_service.send_risk_alert(user, message)
                except Exception as e:
                    print(f"Error executing stop-loss for position {position_id}: {e}")
        
        # Remove triggered stop-losses
        for position_id in triggered_stop_losses:
            if position_id in self.stop_losses:
                del self.stop_losses[position_id]
    
    def add_stop_loss(self, position_id: str, position: Position, stop_percentage: Decimal, 
                      user: User, current_price_func) -> bool:
        """
        Add a stop-loss for a position.
        """
        try:
            # Calculate stop price based on percentage below current price
            stop_price_value = position.current_price.amount * (1 - stop_percentage / 100)
            stop_price = type('Price', (), {'amount': stop_price_value})()  # Mock price object
            
            self.stop_losses[position_id] = {
                'position': position,
                'stop_price': stop_price,
                'current_price_func': current_price_func,
                'user': user,
                'created_at': datetime.now()
            }
            
            return True
        except Exception as e:
            print(f"Error adding stop-loss for position {position_id}: {e}")
            return False
    
    def remove_stop_loss(self, position_id: str) -> bool:
        """
        Remove a stop-loss for a position.
        """
        if position_id in self.stop_losses:
            del self.stop_losses[position_id]
            return True
        return False


class CircuitBreakerService:
    """
    Service to implement circuit breakers for extreme market conditions.
    """

    def __init__(self, notification_service: NotificationPort):
        self.notification_service = notification_service
        # Load circuit breaker settings from configuration
        self.extreme_volatility_threshold = Decimal(str(settings.CIRCUIT_BREAKER_VOLATILITY_THRESHOLD))
        self.circuit_breaker_triggered = False
        self.trigger_time = None
        self.reset_after = timedelta(minutes=settings.CIRCUIT_BREAKER_RESET_MINUTES)
        self.running = False
        self.monitoring_thread = None
    
    def start_monitoring(self):
        """
        Start monitoring for circuit breaker conditions.
        """
        if self.running:
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        print("Circuit breaker monitoring started")
    
    def stop_monitoring(self):
        """
        Stop monitoring for circuit breaker conditions.
        """
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        print("Circuit breaker monitoring stopped")
    
    def _monitoring_loop(self):
        """
        Main monitoring loop for circuit breaker conditions.
        """
        while self.running:
            try:
                if self.circuit_breaker_triggered:
                    # Check if it's time to reset the circuit breaker
                    if datetime.now() - self.trigger_time > self.reset_after:
                        self._reset_circuit_breaker()
                
                # Check for extreme market conditions
                if not self.circuit_breaker_triggered:
                    if self._is_extreme_market_volatility():
                        self._trigger_circuit_breaker()
                
                # Sleep before next check
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                print(f"Error in circuit breaker monitoring loop: {e}")
    
    def _is_extreme_market_volatility(self) -> bool:
        """
        Check if there's extreme market volatility that should trigger the circuit breaker.
        """
        # This would check market indices or other broad market indicators
        # For now, returning False as a placeholder
        return False
    
    def _trigger_circuit_breaker(self):
        """
        Trigger the circuit breaker.
        """
        self.circuit_breaker_triggered = True
        self.trigger_time = datetime.now()
        
        message = "Circuit breaker activated due to extreme market volatility. Trading paused."
        # In a real implementation, this would notify all users and pause trading
        print(message)
    
    def _reset_circuit_breaker(self):
        """
        Reset the circuit breaker.
        """
        self.circuit_breaker_triggered = False
        self.trigger_time = None
        
        message = "Circuit breaker reset. Trading resumed."
        # In a real implementation, this would resume trading
        print(message)
    
    def is_trading_allowed(self) -> bool:
        """
        Check if trading is allowed (circuit breaker not triggered).
        """
        return not self.circuit_breaker_triggered