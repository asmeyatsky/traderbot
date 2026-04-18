"""
Broker Integration Service

This module implements the broker integration layer that connects to various
brokerage APIs (Alpaca, Interactive Brokers, etc.) to execute trades as
outlined in the PRD. It handles order routing, execution, and position management.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import logging
import asyncio
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime
from decimal import Decimal
import json

from src.domain.entities.trading import Order, Position, Portfolio
from src.domain.entities.user import User
from src.domain.value_objects import Symbol, Money, Price
from enum import Enum


logger = logging.getLogger(__name__)


class BrokerAPIException(Exception):
    """Custom exception for broker API errors."""
    pass


class BrokerAuthenticationError(BrokerAPIException):
    """Raised when the broker rejects our credentials (HTTP 401/403).

    Never silently simulated — mis-auth in live mode would lose real money.
    """
    pass


class BrokerOrderResponse:
    """Response object for broker order operations."""
    def __init__(self, broker_order_id: str, status: str, filled_qty: int = 0, avg_fill_price: Optional[float] = None):
        self.broker_order_id = broker_order_id
        self.status = status  # 'pending', 'filled', 'partial_filled', 'cancelled', 'rejected'
        self.filled_qty = filled_qty
        self.avg_fill_price = avg_fill_price


class TradingExecutionPort(ABC):
    """Abstract base class for trading execution services."""

    @abstractmethod
    def place_order(self, order: Order) -> BrokerOrderResponse:
        """Place an order with the broker."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order with the broker."""
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> str:
        """Get the current status of an order."""
        pass

    @abstractmethod
    def get_positions(self, user_id: str) -> List[Position]:
        """Get current positions for a user."""
        pass

    @abstractmethod
    def get_account_info(self, user_id: str) -> Dict:
        """Get account information for a user."""
        pass


class AlpacaBrokerService(TradingExecutionPort):
    """Alpaca Broker API Integration Service."""

    def __init__(self, api_key: str, secret_key: str, paper_trading: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        
        # Alpaca base URLs
        if paper_trading:
            self.base_url = "https://paper-api.alpaca.markets/v2"
            self.data_url = "https://data.alpaca.markets/v2"
        else:
            self.base_url = "https://api.alpaca.markets/v2"
            self.data_url = "https://data.alpaca.markets/v2"
        
        # Setup session with authentication and retry logic
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Add authentication headers
        self.session.headers.update({
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json"
        })
        
        self.paper_trading = paper_trading
        logger.info(f"AlpacaBrokerService initialized (paper_trading={paper_trading})")

    def place_order(self, order: Order) -> BrokerOrderResponse:
        """
        Place an order with Alpaca Broker.

        Emits a LiveOrderPlaced audit event on success for live (non-paper)
        orders. Paper orders are not audited — they have no real-money impact.
        """
        from src.infrastructure.adapters.audit_event_sink import audit_event_sink
        from src.domain.ports.audit_event_sink import AuditEvent, hash_state
        from datetime import datetime as _dt
        import uuid as _uuid
        try:
            from src.infrastructure.observability import get_correlation_id
            _correlation_id = get_correlation_id() or None
        except Exception:
            _correlation_id = None

        def _emit_live_audit(action: str, payload: dict, after_hash: str | None = None) -> None:
            if self.paper_trading:
                return  # paper orders are not real-money events
            try:
                audit_event_sink.append(
                    AuditEvent(
                        id=str(_uuid.uuid4()),
                        actor_user_id=order.user_id,
                        action=action,
                        aggregate_type="live_order",
                        aggregate_id=str(order.id),
                        before_hash=None,
                        after_hash=after_hash,
                        payload_json=payload,
                        occurred_at=_dt.utcnow(),
                        correlation_id=_correlation_id,
                    )
                )
            except Exception:  # noqa: BLE001 — never let audit break execution
                logger.exception("audit emission failed for %s", action)

        try:
            # Map our order to Alpaca format
            alpaca_order = {
                "symbol": str(order.symbol),
                "qty": str(order.quantity),
                "side": "buy" if order.position_type.value == "LONG" else "sell",
                "type": self._map_order_type(order.order_type.value.lower()),
                "time_in_force": "day" if order.order_type.value != "GTC" else "gtc",
            }
            
            # Add price fields based on order type
            if order.order_type.value in ["LIMIT", "STOP_LIMIT"]:
                alpaca_order["limit_price"] = f"{order.price.amount:.2f}" if order.price else None
            if order.order_type.value in ["STOP_LOSS", "STOP_LIMIT"]:
                alpaca_order["stop_price"] = f"{order.stop_price.amount:.2f}" if order.stop_price else None
            
            # Place the order
            response = self.session.post(f"{self.base_url}/orders", json=alpaca_order, timeout=10)

            if response.status_code in (401, 403):
                logger.error(
                    "Alpaca rejected credentials placing order %s (status=%s, paper=%s)",
                    order.id, response.status_code, self.paper_trading,
                )
                _emit_live_audit(
                    "LiveOrderRejected",
                    {"reason": "auth_rejected", "http_status": response.status_code},
                )
                raise BrokerAuthenticationError(
                    f"Alpaca credentials rejected (HTTP {response.status_code}); "
                    "refusing to simulate — verify ALPACA_API_KEY / ALPACA_SECRET_KEY."
                )

            response.raise_for_status()
            order_data = response.json()

            result = BrokerOrderResponse(
                broker_order_id=order_data["id"],
                status=order_data["status"],
                filled_qty=int(order_data.get("filled_qty", 0)),
                avg_fill_price=float(order_data.get("filled_avg_price", 0)) if order_data.get("filled_avg_price") else None
            )
            _emit_live_audit(
                "LiveOrderPlaced",
                {
                    "symbol": str(order.symbol),
                    "quantity": order.quantity,
                    "side": alpaca_order["side"],
                    "type": alpaca_order["type"],
                    "broker_order_id": result.broker_order_id,
                    "status": result.status,
                },
                after_hash=hash_state({
                    "broker_order_id": result.broker_order_id,
                    "status": result.status,
                    "filled_qty": result.filled_qty,
                }),
            )
            return result

        except requests.exceptions.HTTPError as e:
            error_details = e.response.json() if e.response.content else {}
            logger.error(f"Alpaca API error placing order: {error_details}")
            _emit_live_audit(
                "LiveOrderRejected",
                {"reason": "http_error", "details": str(error_details)[:500]},
            )
            raise BrokerAPIException(f"Failed to place order: {error_details}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error placing order: {e}")
            _emit_live_audit(
                "LiveOrderRejected",
                {"reason": "network_error", "message": str(e)[:500]},
            )
            raise BrokerAPIException(f"Network error placing order: {e}")
        except BrokerAuthenticationError:
            # Already audited above; re-raise unchanged.
            raise
        except Exception as e:
            logger.error(f"Unexpected error placing order: {e}")
            _emit_live_audit(
                "LiveOrderRejected",
                {"reason": "unexpected_error", "message": str(e)[:500]},
            )
            raise BrokerAPIException(f"Unexpected error placing order: {e}")

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order with Alpaca."""
        try:
            response = self.session.delete(f"{self.base_url}/orders/{order_id}", timeout=10)
            
            if response.status_code == 404:
                # Order already filled or doesn't exist
                return True
            if response.status_code in (401, 403):
                logger.error(
                    "Alpaca rejected credentials cancelling order %s (status=%s)",
                    order_id, response.status_code,
                )
                raise BrokerAuthenticationError(
                    f"Alpaca credentials rejected (HTTP {response.status_code}) cancelling {order_id}."
                )

            response.raise_for_status()
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    def get_order_status(self, order_id: str) -> str:
        """Get the current status of an order."""
        try:
            response = self.session.get(f"{self.base_url}/orders/{order_id}", timeout=10)

            if response.status_code in (401, 403):
                logger.error(
                    "Alpaca rejected credentials reading order %s (status=%s)",
                    order_id, response.status_code,
                )
                raise BrokerAuthenticationError(
                    f"Alpaca credentials rejected (HTTP {response.status_code}) fetching {order_id}."
                )
            if response.status_code == 404:
                return "not_found"
            
            response.raise_for_status()
            order_data = response.json()
            return order_data["status"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting order status for {order_id}: {e}")
            return "unknown"

    def get_positions(self, user_id: str) -> List[Position]:
        """Get current positions from Alpaca."""
        try:
            response = self.session.get(f"{self.base_url}/positions", timeout=10)

            if response.status_code in (401, 403):
                logger.error(
                    "Alpaca rejected credentials reading positions for user %s (status=%s)",
                    user_id, response.status_code,
                )
                raise BrokerAuthenticationError(
                    f"Alpaca credentials rejected (HTTP {response.status_code}) reading positions."
                )
            
            response.raise_for_status()
            positions_data = response.json()
            
            positions = []
            for pos_data in positions_data:
                # Convert Alpaca position to our domain Position
                quantity = int(float(pos_data["qty"]))
                avg_entry_price = Decimal(pos_data["avg_entry_price"])
                current_price = Decimal(pos_data.get("current_price", pos_data["avg_entry_price"]))
                market_value = current_price * Decimal(quantity)
                cost_basis = avg_entry_price * Decimal(quantity)
                
                # Convert position type
                pos_type = "LONG" if float(pos_data["qty"]) > 0 else "SHORT"
                
                from src.domain.entities.trading import PositionType
                position_type = PositionType.LONG if pos_type == "LONG" else PositionType.SHORT
                
                position = Position(
                    id=pos_data["symbol"],  # Using symbol as ID for simplicity in simulation
                    user_id=user_id,
                    symbol=Symbol(pos_data["symbol"]),
                    position_type=position_type,
                    quantity=quantity,
                    average_buy_price=Money(avg_entry_price, "USD"),
                    current_price=Money(current_price, "USD"),
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    unrealized_pnl=Money(market_value - cost_basis, "USD")
                )
                positions.append(position)
            
            return positions
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting positions: {e}")
            return []

    def get_account_info(self, user_id: str) -> Dict:
        """Get account information from Alpaca."""
        try:
            response = self.session.get(f"{self.base_url}/account", timeout=10)

            if response.status_code in (401, 403):
                logger.error(
                    "Alpaca rejected credentials reading account for user %s (status=%s)",
                    user_id, response.status_code,
                )
                raise BrokerAuthenticationError(
                    f"Alpaca credentials rejected (HTTP {response.status_code}) reading account."
                )
            
            response.raise_for_status()
            account_data = response.json()
            
            return {
                "account_number": account_data["account_number"],
                "buying_power": float(account_data["buying_power"]),
                "cash": float(account_data["cash"]),
                "portfolio_value": float(account_data["portfolio_value"]),
                "status": account_data["status"],
                "trading_blocked": account_data["trading_blocked"],
                "transfers_blocked": account_data["transfers_blocked"]
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting account info: {e}")
            return {}

    def _map_order_type(self, order_type: str) -> str:
        """Map our order types to Alpaca order types."""
        mapping = {
            'market': 'market',
            'limit': 'limit',
            'stop_loss': 'stop',
            'trailing_stop': 'trailing_stop'
        }
        return mapping.get(order_type, 'market')


class InteractiveBrokersService(TradingExecutionPort):
    """
    Interactive Brokers API Integration Service.
    
    Note: IB API typically uses a more complex socket-based connection.
    This is a simplified implementation for the purpose of this platform.
    """
    
    def __init__(self, api_endpoint: str, username: str, password: str, account_id: str):
        self.api_endpoint = api_endpoint
        self.username = username
        self.password = password
        self.account_id = account_id
        self.session = requests.Session()
        logger.info("InteractiveBrokersService initialized")

    def place_order(self, order: Order) -> BrokerOrderResponse:
        """
        Place an order with Interactive Brokers.
        In a real implementation, this would use the IB API (socket connection).
        """
        logger.info(f"Placing order with Interactive Brokers: {order.symbol} {order.quantity} shares")
        
        # For now, return a simulated successful response
        # In real implementation, would connect to the IB Gateway/TWS API
        return BrokerOrderResponse(
            broker_order_id=f"IB_{order.id}",
            status="pending",
            filled_qty=order.quantity,
            avg_fill_price=float(order.price.amount) if order.price else 100.0
        )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order with Interactive Brokers."""
        logger.info(f"Cancelling order with Interactive Brokers: {order_id}")
        # Simulated success
        return True

    def get_order_status(self, order_id: str) -> str:
        """Get order status from Interactive Brokers."""
        logger.info(f"Getting order status for: {order_id}")
        # Simulated status
        return "filled"

    def get_positions(self, user_id: str) -> List[Position]:
        """Get positions from Interactive Brokers."""
        logger.info(f"Getting positions for user: {user_id}")
        # Simulated positions
        return []

    def get_account_info(self, user_id: str) -> Dict:
        """Get account information from Interactive Brokers."""
        logger.info(f"Getting account info for user: {user_id}")
        # Simulated account info
        return {
            "account_number": self.account_id,
            "cash": 50000.0,
            "portfolio_value": 150000.0,
            "status": "ACTIVE"
        }


class SmartOrderRoutingService:
    """Intelligent order routing service that chooses the best broker for execution."""

    def __init__(self, brokers: Dict[str, TradingExecutionPort]):
        self.brokers = brokers
        self.execution_preferences = {}  # Could store user preferences per symbol/broker
        logger.info("SmartOrderRoutingService initialized")

    def route_order(self, order: Order, user: User) -> BrokerOrderResponse:
        """
        Route order to the most appropriate broker based on various factors.
        
        Factors considered:
        - Cost (commissions, fees)
        - Speed of execution
        - Market access (certain brokers have better access to specific markets)
        - User preferences
        - Current market conditions
        """
        # Get the best broker for this order based on various criteria
        best_broker = self._determine_best_broker(order, user)
        
        logger.info(f"Routing order {order.id} to {best_broker}")
        
        try:
            # Execute the order with the chosen broker
            broker_service = self.brokers[best_broker]
            return broker_service.place_order(order)
        except Exception as e:
            logger.error(f"Error executing order with {best_broker}: {e}")
            # If primary broker fails, try with another broker
            for broker_name, broker_service in self.brokers.items():
                if broker_name != best_broker:
                    try:
                        logger.info(f"Trying alternate broker {broker_name}")
                        return broker_service.place_order(order)
                    except Exception as fallback_error:
                        logger.error(f"Fallback broker {broker_name} also failed: {fallback_error}")
                        continue
            
            # If all brokers fail, raise the original exception
            raise

    def _determine_best_broker(self, order: Order, user: User) -> str:
        """Determine which broker to use based on various factors."""
        # Simple algorithm - in real implementation, this would be more sophisticated
        # Considerations:
        # - Commission rates
        # - Execution speed
        # - Market access
        # - User preferences
        # - Current account balances
        
        # If user has a preferred broker, use that
        user_preferred_broker = getattr(user, 'preferred_broker', None)
        if user_preferred_broker and user_preferred_broker in self.brokers:
            return user_preferred_broker
        
        # Otherwise, for now, return the first available broker
        # In a real implementation, we'd have more sophisticated logic here
        return next(iter(self.brokers.keys()))


class VWAPExecutionService:
    """VWAP (Volume Weighted Average Price) execution service for large orders."""

    def __init__(self, broker_service: TradingExecutionPort):
        self.broker_service = broker_service
        logger.info("VWAPExecutionService initialized")

    def execute_vwap_order(self, order: Order, time_window_minutes: int = 60) -> List[BrokerOrderResponse]:
        """
        Execute a large order using VWAP algorithm to minimize market impact.
        
        Args:
            order: The large order to execute
            time_window_minutes: Time window over which to execute
            
        Returns:
            List of individual order responses
        """
        # Calculate average volume for this symbol (simplified)
        avg_volume = self._get_average_volume(order.symbol)
        
        # Calculate how many shares to execute per time slice
        # This is a simplified VWAP implementation
        time_slices = time_window_minutes
        shares_per_slice = order.quantity // time_slices
        
        responses = []
        remaining_quantity = order.quantity
        
        for i in range(time_slices):
            if remaining_quantity <= 0:
                break
                
            # Calculate quantity for this slice (could vary based on volume patterns)
            slice_quantity = min(shares_per_slice, remaining_quantity)
            
            # Create sub-order with the calculated quantity
            sub_order = Order(
                id=f"{order.id}_slice_{i}",
                user_id=order.user_id,
                symbol=order.symbol,
                order_type=order.order_type,
                position_type=order.position_type,
                quantity=slice_quantity,
                status=order.status,
                placed_at=order.placed_at,
                executed_at=order.executed_at,
                price=order.price,
                stop_price=order.stop_price,
                filled_quantity=order.filled_quantity,
                commission=order.commission,
                notes=f"VWAP slice {i+1}/{time_slices}"
            )
            
            try:
                response = self.broker_service.place_order(sub_order)
                responses.append(response)
                remaining_quantity -= slice_quantity
            except Exception as e:
                logger.error(f"Failed to execute VWAP slice {i}: {e}")
                # Continue with remaining slices
                continue
        
        return responses

    def _get_average_volume(self, symbol: Symbol) -> int:
        """
        Get average trading volume for a symbol.
        In a real implementation, this would come from market data.
        """
        # Simulated average volume - in real implementation, fetch from market data provider
        return 1000000  # 1M shares per day average


class BrokerIntegrationService:
    """
    Main broker integration service that manages multiple brokers and provides
    a unified interface to the trading system.
    """

    def __init__(self, 
                 alpaca_service: Optional[AlpacaBrokerService] = None,
                 ib_service: Optional[InteractiveBrokersService] = None):
        self.brokers = {}
        
        if alpaca_service:
            self.brokers['alpaca'] = alpaca_service
        if ib_service:
            self.brokers['interactive_brokers'] = ib_service
            
        self.smart_router = SmartOrderRoutingService(self.brokers) if self.brokers else None
        self.vwap_service = None  # Initialize when needed
        logger.info("BrokerIntegrationService initialized with available brokers")

    def execute_order(self, order: Order, user: User) -> BrokerOrderResponse:
        """Execute an order using the smart routing service."""
        if not self.smart_router:
            raise BrokerAPIException("No brokers configured")
        
        return self.smart_router.route_order(order, user)

    def get_all_positions(self, user_id: str) -> Dict[str, List[Position]]:
        """Get positions from all connected brokers."""
        all_positions = {}
        
        for broker_name, broker_service in self.brokers.items():
            try:
                positions = broker_service.get_positions(user_id)
                all_positions[broker_name] = positions
            except Exception as e:
                logger.error(f"Error getting positions from {broker_name}: {e}")
                all_positions[broker_name] = []
        
        return all_positions

    def get_unified_account_info(self, user_id: str) -> Dict:
        """Get combined account information from all brokers."""
        unified_info = {
            'total_cash': 0,
            'total_portfolio_value': 0,
            'total_buying_power': 0,
            'broker_accounts': {}
        }
        
        for broker_name, broker_service in self.brokers.items():
            try:
                account_info = broker_service.get_account_info(user_id)
                unified_info['broker_accounts'][broker_name] = account_info
                unified_info['total_cash'] += account_info.get('cash', 0)
                unified_info['total_portfolio_value'] += account_info.get('portfolio_value', 0)
                unified_info['total_buying_power'] += account_info.get('buying_power', 0)
            except Exception as e:
                logger.error(f"Error getting account info from {broker_name}: {e}")
        
        return unified_info

    def get_available_brokers(self) -> List[str]:
        """Get list of available broker integrations."""
        return list(self.brokers.keys())

    def supports_vwap_execution(self) -> bool:
        """Check if VWAP execution is supported."""
        return self.vwap_service is not None


class BrokerType(str, Enum):
    """Enumeration of supported broker types."""
    ALPACA = "alpaca"
    INTERACTIVE_BROKERS = "interactive_brokers"
    TD_AMERITRADE = "td_ameritrade"
    FIDELITY = "fidelity"
    SCHWAB = "schwab"


class BrokerOrder:
    """Represents an order in the broker's system."""
    def __init__(self, broker_order_id: str, client_order_id: str, symbol: str, quantity: int,
                 side: str, order_type: str, status: str, time_in_force: str = "day",
                 limit_price: Optional[float] = None, stop_price: Optional[float] = None,
                 created_at: Optional[datetime] = None):
        self.broker_order_id = broker_order_id
        self.client_order_id = client_order_id
        self.symbol = symbol
        self.quantity = quantity
        self.side = side  # 'buy' or 'sell'
        self.order_type = order_type  # 'market', 'limit', 'stop', etc
        self.status = status  # 'pending', 'filled', 'partial_filled', 'cancelled', 'rejected'
        self.time_in_force = time_in_force
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.created_at = created_at or datetime.now()


class BrokerAdapterManager:
    """Manager for different broker adapters with routing capabilities."""

    def __init__(self):
        """Initialize the adapter manager with available broker services."""
        from src.infrastructure.config.settings import settings

        self.brokers = {}

        # Initialize Alpaca service.
        # BrokerAdapterManager is used only by legacy pathways that have NOT
        # been migrated to the per-user BrokerServiceFactory yet. Paper-mode
        # is the safe default — do not change without a migration of callers.
        # See src/infrastructure/adapters/broker_factory.py (ADR-002).
        self.brokers[BrokerType.ALPACA] = AlpacaBrokerService(
            api_key=settings.ALPACA_API_KEY,
            secret_key=settings.ALPACA_SECRET_KEY,
            paper_trading=True,
        )

        # Initialize Interactive Brokers service if API key exists
        if settings.INTERACTIVE_BROKERS_API_KEY:
            self.brokers[BrokerType.INTERACTIVE_BROKERS] = InteractiveBrokersService(
                api_key=settings.INTERACTIVE_BROKERS_API_KEY
            )

    def get_broker_service(self, broker_type: BrokerType):
        """Get broker service instance for the specified broker type."""
        if broker_type not in self.brokers:
            raise BrokerAPIException(f"Broker type {broker_type} not supported")
        return self.brokers[broker_type]

    def execute_order(self, order: Order, user: User, broker_type: BrokerType) -> BrokerOrder:
        """Execute an order through the specified broker."""
        broker_service = self.get_broker_service(broker_type)
        return broker_service.execute_order(order, user)

    def get_positions(self, user: User, broker_type: BrokerType) -> List[Position]:
        """Get positions from the specified broker."""
        broker_service = self.get_broker_service(broker_type)
        return broker_service.get_positions(user)

    def get_account_info(self, user: User, broker_type: BrokerType) -> Any:
        """Get account information from the specified broker."""
        broker_service = self.get_broker_service(broker_type)
        return broker_service.get_account_info(user)

    def get_available_brokers(self) -> List[BrokerType]:
        """Get list of available brokers."""
        return list(self.brokers.keys())


# Global broker integration service instance
broker_integration_service = BrokerIntegrationService()