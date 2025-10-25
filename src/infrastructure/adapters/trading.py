"""
Trading Engine with Broker Integration

This module implements the trading engine that executes trades
through various broker APIs as required by the PRD.
"""
import alpaca_trade_api as tradeapi
from alpaca_trade_api.entity import Order as AlpacaOrder
import threading
import time
from typing import Dict, List, Optional
from datetime import datetime

from src.domain.entities.trading import Order, OrderStatus, OrderType
from src.domain.value_objects import Money, Symbol
from src.domain.ports import TradingExecutionPort
from src.infrastructure.config.settings import settings


class AlpacaTradingAdapter(TradingExecutionPort):
    """
    Adapter for Alpaca Broker API.
    
    Implements TradingExecutionPort interface to execute trades through Alpaca.
    """
    
    def __init__(self):
        self.api_key = settings.ALPACA_API_KEY
        self.api_secret = settings.ALPACA_SECRET_KEY
        self.base_url = "https://paper-api.alpaca.markets" if settings.ENVIRONMENT == "development" else "https://api.alpaca.markets"
        
        # Initialize Alpaca API
        self.api = tradeapi.REST(
            key_id=self.api_key,
            secret_key=self.api_secret,
            base_url=self.base_url,
            api_version='v2'
        )
    
    def place_order(self, order: Order) -> str:
        """
        Place an order with Alpaca and return the order ID.
        """
        try:
            # Map domain order type to Alpaca order type
            alpaca_order_type = self._map_order_type(order.order_type)
            
            # Map domain order to Alpaca order
            alpaca_order = self.api.submit_order(
                symbol=str(order.symbol),
                qty=order.quantity,
                side='buy' if order.position_type.name == 'LONG' else 'sell',
                type=alpaca_order_type,
                time_in_force='gtc',  # Good till canceled
                limit_price=str(order.price.amount) if order.price else None,
                stop_price=str(order.stop_price.amount) if order.stop_price else None
            )
            
            return alpaca_order.id
        except Exception as e:
            print(f"Error placing order with Alpaca: {e}")
            raise
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order with Alpaca.
        """
        try:
            self.api.cancel_order(order_id)
            return True
        except Exception as e:
            print(f"Error canceling order with Alpaca: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Order:
        """
        Get the current status of an order from Alpaca.
        This requires mapping Alpaca response to our domain Order entity.
        """
        try:
            alpaca_order = self.api.get_order(order_id)
            
            # Map Alpaca status to domain status
            status_map = {
                'new': OrderStatus.PENDING,
                'partially_filled': OrderStatus.PENDING,
                'filled': OrderStatus.EXECUTED,
                'done_for_day': OrderStatus.EXECUTED,
                'canceled': OrderStatus.CANCELLED,
                'expired': OrderStatus.CANCELLED,
                'replaced': OrderStatus.CANCELLED,  # Simplified mapping
                'pending_cancel': OrderStatus.CANCELLED,
                'pending_new': OrderStatus.PENDING,
                'accepted': OrderStatus.PENDING,
                'pending_replace': OrderStatus.PENDING,
                'calculated': OrderStatus.EXECUTED,
                'stopped': OrderStatus.FAILED,
                'rejected': OrderStatus.FAILED,
                'suspended': OrderStatus.PENDING,
                'open': OrderStatus.PENDING,
                'filled_day': OrderStatus.EXECUTED,
                'filled_day_one_canceled_rest': OrderStatus.EXECUTED,
                'filled_next_day': OrderStatus.EXECUTED,
                'fill_or_kill_rejected': OrderStatus.FAILED
            }
            
            status = status_map.get(alpaca_order.status.lower(), OrderStatus.PENDING)
            
            # Create domain Order object
            # Note: This is a simplified mapping - in real impl you'd need more fields
            from src.domain.entities.trading import PositionType
            
            position_type = PositionType.LONG if alpaca_order.side == 'buy' else PositionType.SHORT
            order_type = self._reverse_map_order_type(alpaca_order.type)
            
            return Order(
                id=alpaca_order.id,
                user_id="",  # Would need to be passed from context
                symbol=Symbol(alpaca_order.symbol),
                order_type=order_type,
                position_type=position_type,
                quantity=int(alpaca_order.qty),
                status=status,
                placed_at=alpaca_order.created_at,
                executed_at=alpaca_order.filled_at,
                price=Money(float(alpaca_order.filled_avg_price), 'USD') if alpaca_order.filled_avg_price else None,
                stop_price=Money(float(alpaca_order.stop_price), 'USD') if alpaca_order.stop_price else None,
                filled_quantity=int(alpaca_order.filled_qty) if alpaca_order.filled_qty else 0,
                commission=Money(float(alpaca_order.commission), 'USD') if alpaca_order.commission else None,
                notes=alpaca_order.client_order_id
            )
        except Exception as e:
            print(f"Error getting order status from Alpaca: {e}")
            # Return a minimal order with failed status
            return Order(
                id=order_id,
                user_id="",
                symbol=Symbol("UNKNOWN"),
                order_type=OrderType.MARKET,
                position_type=PositionType.LONG,
                quantity=0,
                status=OrderStatus.FAILED,
                placed_at=datetime.now(),
                executed_at=None,
                price=None,
                stop_price=None,
                filled_quantity=0,
                commission=None,
                notes=f"Error: {str(e)}"
            )
    
    def get_account_balance(self, user_id: str) -> Money:
        """
        Get the account balance for a user from Alpaca.
        """
        try:
            account = self.api.get_account()
            cash_amount = float(account.cash)
            return Money(cash_amount, 'USD')
        except Exception as e:
            print(f"Error getting account balance from Alpaca: {e}")
            return Money(0, 'USD')
    
    def _map_order_type(self, order_type: OrderType) -> str:
        """
        Map domain OrderType to Alpaca order type.
        """
        mapping = {
            OrderType.MARKET: 'market',
            OrderType.LIMIT: 'limit',
            OrderType.STOP_LOSS: 'stop',
            OrderType.TRAILING_STOP: 'trailing_stop'
        }
        return mapping.get(order_type, 'market')
    
    def _reverse_map_order_type(self, alpaca_type: str) -> OrderType:
        """
        Map Alpaca order type back to domain OrderType.
        """
        mapping = {
            'market': OrderType.MARKET,
            'limit': OrderType.LIMIT,
            'stop': OrderType.STOP_LOSS,
            'trailing_stop': OrderType.TRAILING_STOP
        }
        return mapping.get(alpaca_type, OrderType.MARKET)


class TradingEngine:
    """
    Core trading engine that manages the trading workflow.
    
    This engine coordinates between the various services to execute trades
    based on signals from the AI models while respecting risk constraints.
    """
    
    def __init__(
        self,
        trading_execution_service: TradingExecutionPort,
        ai_model_service,
        risk_management_service,
        market_data_service,
        notification_service
    ):
        self.trading_execution_service = trading_execution_service
        self.ai_model_service = ai_model_service
        self.risk_management_service = risk_management_service
        self.market_data_service = market_data_service
        self.notification_service = notification_service
        
        # Keep track of active orders to manage them
        self.active_orders = {}
        self.running = False
        self.execution_thread = None
    
    def start_engine(self):
        """
        Start the trading engine in a background thread.
        """
        if self.running:
            return
        
        self.running = True
        self.execution_thread = threading.Thread(target=self._run_trading_loop, daemon=True)
        self.execution_thread.start()
        print("Trading engine started")
    
    def stop_engine(self):
        """
        Stop the trading engine.
        """
        self.running = False
        if self.execution_thread:
            self.execution_thread.join()
        print("Trading engine stopped")
    
    def _run_trading_loop(self):
        """
        Main trading loop that continuously looks for trading opportunities.
        """
        while self.running:
            try:
                # In a real implementation, this would fetch signals for multiple symbols
                # For this example, we'll just simulate a periodic check
                time.sleep(30)  # Check every 30 seconds
                
                # This is where you'd typically get signals and execute trades
                # self._process_trading_signals()
                
            except Exception as e:
                print(f"Error in trading loop: {e}")
    
    def submit_order(self, order: Order) -> str:
        """
        Submit an order for execution.
        """
        try:
            # Place the order with the broker
            broker_order_id = self.trading_execution_service.place_order(order)
            
            # Store order reference
            self.active_orders[broker_order_id] = order
            
            return broker_order_id
        except Exception as e:
            print(f"Error submitting order: {e}")
            raise
    
    def monitor_order(self, order_id: str) -> Order:
        """
        Monitor an order and return its current status.
        """
        try:
            # Get status from broker
            current_order = self.trading_execution_service.get_order_status(order_id)
            
            # Update internal tracking
            if order_id in self.active_orders:
                self.active_orders[order_id] = current_order
            
            return current_order
        except Exception as e:
            print(f"Error monitoring order: {e}")
            # Return a failed order if monitoring fails
            return Order(
                id=order_id,
                user_id="",
                symbol=Symbol("UNKNOWN"),
                order_type=OrderType.MARKET,
                position_type=PositionType.LONG,
                quantity=0,
                status=OrderStatus.FAILED,
                placed_at=datetime.now(),
                executed_at=None,
                price=None,
                stop_price=None,
                filled_quantity=0,
                commission=None,
                notes=f"Monitoring failed: {str(e)}"
            )
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        """
        try:
            # Cancel with broker
            success = self.trading_execution_service.cancel_order(order_id)
            
            # Remove from active orders if successful
            if success and order_id in self.active_orders:
                del self.active_orders[order_id]
            
            return success
        except Exception as e:
            print(f"Error canceling order: {e}")
            return False
    
    def get_account_balance(self, user_id: str) -> Money:
        """
        Get the account balance for a user.
        """
        return self.trading_execution_service.get_account_balance(user_id)


class PortfolioTracker:
    """
    Service to track and update portfolio positions based on executed trades.
    
    This service keeps the portfolio state synchronized with actual positions
    held in the broker account.
    """
    
    def __init__(self, trading_engine: TradingEngine, portfolio_repository):
        self.trading_engine = trading_engine
        self.portfolio_repository = portfolio_repository
        self.sync_interval = 60  # Sync every minute
        self.running = False
        self.sync_thread = None
    
    def start_sync(self):
        """
        Start the portfolio synchronization in a background thread.
        """
        if self.running:
            return
        
        self.running = True
        self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.sync_thread.start()
        print("Portfolio sync started")
    
    def stop_sync(self):
        """
        Stop the portfolio synchronization.
        """
        self.running = False
        if self.sync_thread:
            self.sync_thread.join()
        print("Portfolio sync stopped")
    
    def _sync_loop(self):
        """
        Main synchronization loop to keep portfolio in sync with broker.
        """
        while self.running:
            try:
                # In a real implementation, this would sync with the broker
                # to get actual positions and update the portfolio
                time.sleep(self.sync_interval)
            except Exception as e:
                print(f"Error in portfolio sync: {e}")
    
    def update_portfolio_from_order(self, order: Order):
        """
        Update the portfolio based on an executed order.
        """
        # This would update the portfolio with the new position information
        # based on the executed order
        pass