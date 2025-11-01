"""
Multi-Broker Integration Service

Implements unified interface for multiple broker integrations
including Alpaca, Interactive Brokers, and mock implementations
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Any
from decimal import Decimal
import uuid
import requests
from enum import Enum

from src.domain.entities.trading import Order, OrderType, Position
from src.domain.entities.user import User
from src.domain.value_objects import Symbol, Money


class BrokerType(Enum):
    MOCK = "mock"
    ALPACA = "alpaca"
    INTERACTIVE_BROKERS = "interactive_brokers"
    TD_AMERITRADE = "td_ameritrade"
    FTX = "ftx"  # Note: FTX is defunct, using as example


class BrokerOrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class BrokerPositionType(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class BrokerOrder:
    """Data class representing an order in broker-specific format"""
    broker_order_id: str
    client_order_id: str
    symbol: str
    order_type: BrokerOrderType
    position_type: BrokerPositionType
    quantity: int
    side: str  # "buy" or "sell"
    time_in_force: str  # "day", "gtc", "ioc", etc.
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    status: str = "pending"
    filled_qty: int = 0
    avg_fill_price: Optional[Decimal] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    broker_specific_data: Optional[Dict[str, Any]] = None


@dataclass
class BrokerPosition:
    """Data class representing a position in broker-specific format"""
    symbol: str
    quantity: int
    avg_entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    market_value: Decimal
    position_type: BrokerPositionType
    broker_specific_data: Optional[Dict[str, Any]] = None


@dataclass
class BrokerAccountInfo:
    """Data class for broker account information"""
    account_id: str
    account_number: str
    account_type: str
    buying_power: Money
    cash_balance: Money
    portfolio_value: Money
    day_trade_count: int
    pattern_day_trader: bool
    trading_blocked: bool
    transfers_blocked: bool
    account_blocked: bool
    created_at: datetime
    updated_at: datetime
    broker_specific_data: Optional[Dict[str, Any]] = None


class BrokerIntegrationService(ABC):
    """
    Abstract base class for broker integration services.
    """
    
    @abstractmethod
    def place_order(self, order: Order, user: User) -> BrokerOrder:
        """Place an order with the broker"""
        pass
    
    @abstractmethod
    def cancel_order(self, broker_order_id: str, user: User) -> bool:
        """Cancel an order with the broker"""
        pass
    
    @abstractmethod
    def get_order_status(self, broker_order_id: str, user: User) -> BrokerOrder:
        """Get the status of an order from the broker"""
        pass
    
    @abstractmethod
    def get_positions(self, user: User) -> List[BrokerPosition]:
        """Get current positions from the broker"""
        pass
    
    @abstractmethod
    def get_account_info(self, user: User) -> BrokerAccountInfo:
        """Get account information from the broker"""
        pass
    
    @abstractmethod
    def get_historical_orders(self, user: User, limit: int = 100) -> List[BrokerOrder]:
        """Get historical orders from the broker"""
        pass


class MockBrokerService(BrokerIntegrationService):
    """
    Mock broker service for testing and development
    """
    
    def __init__(self):
        self._orders = {}
        self._positions = {}
        self._accounts = {}
        self._order_id_counter = 1000
    
    def place_order(self, order: Order, user: User) -> BrokerOrder:
        """Place an order with the mock broker"""
        broker_order_id = f"mock_{self._order_id_counter}"
        self._order_id_counter += 1
        
        # Determine side based on position type
        side = "buy" if order.position_type.name == 'LONG' else 'sell'
        
        # Determine broker order type
        broker_order_type = BrokerOrderType.MARKET  # Default to market
        if order.order_type.name == 'LIMIT':
            broker_order_type = BrokerOrderType.LIMIT
        elif order.order_type.name == 'STOP_LOSS':
            broker_order_type = BrokerOrderType.STOP
        
        broker_order = BrokerOrder(
            broker_order_id=broker_order_id,
            client_order_id=order.id,
            symbol=str(order.symbol),
            order_type=broker_order_type,
            position_type=BrokerPositionType.LONG if order.position_type.name == 'LONG' else BrokerPositionType.SHORT,
            quantity=order.quantity,
            side=side,
            time_in_force="day",
            limit_price=order.price.amount if order.price else None,
            stop_price=order.stop_price.amount if order.stop_price else None,
            status="accepted",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Store the order
        self._orders[broker_order_id] = broker_order
        
        return broker_order
    
    def cancel_order(self, broker_order_id: str, user: User) -> bool:
        """Cancel an order with the mock broker"""
        if broker_order_id in self._orders:
            order = self._orders[broker_order_id]
            order.status = "cancelled"
            order.updated_at = datetime.now()
            return True
        return False
    
    def get_order_status(self, broker_order_id: str, user: User) -> BrokerOrder:
        """Get the status of an order from the mock broker"""
        if broker_order_id in self._orders:
            return self._orders[broker_order_id]
        else:
            raise ValueError(f"Order {broker_order_id} not found")
    
    def get_positions(self, user: User) -> List[BrokerPosition]:
        """Get current positions from the mock broker"""
        # For demo, return mock positions
        positions = [
            BrokerPosition(
                symbol="AAPL",
                quantity=100,
                avg_entry_price=Decimal('150.00'),
                current_price=Decimal('175.50'),
                unrealized_pnl=Decimal('2550.00'),
                market_value=Decimal('17550.00'),
                position_type=BrokerPositionType.LONG
            ),
            BrokerPosition(
                symbol="GOOGL",
                quantity=50,
                avg_entry_price=Decimal('2500.00'),
                current_price=Decimal('2750.25'),
                unrealized_pnl=Decimal('12512.50'),
                market_value=Decimal('137512.50'),
                position_type=BrokerPositionType.LONG
            )
        ]
        
        return positions
    
    def get_account_info(self, user: User) -> BrokerAccountInfo:
        """Get account information from the mock broker"""
        account_id = f"mock_acc_{user.id}"
        
        account_info = BrokerAccountInfo(
            account_id=account_id,
            account_number=f"ACCT_{str(uuid.uuid4())[:8].upper()}",
            account_type="cash",
            buying_power=Money(Decimal('50000.00'), 'USD'),
            cash_balance=Money(Decimal('30000.00'), 'USD'),
            portfolio_value=Money(Decimal('170000.00'), 'USD'),
            day_trade_count=0,
            pattern_day_trader=False,
            trading_blocked=False,
            transfers_blocked=False,
            account_blocked=False,
            created_at=datetime.now() - timedelta(days=30),
            updated_at=datetime.now()
        )
        
        self._accounts[account_id] = account_info
        return account_info
    
    def get_historical_orders(self, user: User, limit: int = 100) -> List[BrokerOrder]:
        """Get historical orders from the mock broker"""
        # Return all stored orders up to the limit
        orders = list(self._orders.values())
        return orders[-limit:]  # Return most recent orders


class AlpacaBrokerService(BrokerIntegrationService):
    """
    Alpaca broker service implementation
    Note: This is a simplified mock implementation - real implementation would use Alpaca API
    """
    
    def __init__(self, api_key: str, secret_key: str, base_url: str = "https://api.alpaca.markets"):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'APCA-API-KEY-ID': api_key,
            'APCA-API-SECRET-KEY': secret_key,
            'Content-Type': 'application/json'
        })
    
    def place_order(self, order: Order, user: User) -> BrokerOrder:
        """Place an order with Alpaca broker"""
        # In a real implementation, this would call Alpaca API
        # For mock implementation, we'll simulate the API call
        broker_order_id = f"alp_{uuid.uuid4()}"
        
        # Map our domain order to Alpaca order format
        side = "buy" if order.position_type.name == 'LONG' else 'sell'
        order_type_map = {
            'MARKET': 'market',
            'LIMIT': 'limit',
            'STOP_LOSS': 'stop',
            'TRAILING_STOP': 'trailing_stop'
        }
        
        alpaca_order_type = order_type_map.get(order.order_type.name, 'market')
        
        # For mock implementation, return a simulated broker order
        return BrokerOrder(
            broker_order_id=broker_order_id,
            client_order_id=order.id,
            symbol=str(order.symbol),
            order_type=BrokerOrderType(alpaca_order_type),
            position_type=BrokerPositionType.LONG if order.position_type.name == 'LONG' else BrokerPositionType.SHORT,
            quantity=order.quantity,
            side=side,
            time_in_force="day",
            limit_price=order.price.amount if order.price else None,
            stop_price=order.stop_price.amount if order.stop_price else None,
            status="pending",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    def cancel_order(self, broker_order_id: str, user: User) -> bool:
        """Cancel an order with Alpaca broker"""
        # In a real implementation, this would call Alpaca API
        # For mock, just return True
        return True
    
    def get_order_status(self, broker_order_id: str, user: User) -> BrokerOrder:
        """Get the status of an order from Alpaca broker"""
        # In a real implementation, this would call Alpaca API
        # For mock, return a simulated order status
        return BrokerOrder(
            broker_order_id=broker_order_id,
            client_order_id=uuid.uuid4(),
            symbol="AAPL",
            order_type=BrokerOrderType.MARKET,
            position_type=BrokerPositionType.LONG,
            quantity=100,
            side="buy",
            time_in_force="day",
            status="filled",
            filled_qty=100,
            avg_fill_price=Decimal('175.25'),
            created_at=datetime.now() - timedelta(minutes=5),
            updated_at=datetime.now()
        )
    
    def get_positions(self, user: User) -> List[BrokerPosition]:
        """Get current positions from Alpaca broker"""
        # In a real implementation, this would call Alpaca API
        # For mock, return sample positions
        return [
            BrokerPosition(
                symbol="AAPL",
                quantity=100,
                avg_entry_price=Decimal('150.00'),
                current_price=Decimal('175.50'),
                unrealized_pnl=Decimal('2550.00'),
                market_value=Decimal('17550.00'),
                position_type=BrokerPositionType.LONG
            )
        ]
    
    def get_account_info(self, user: User) -> BrokerAccountInfo:
        """Get account information from Alpaca broker"""
        # In a real implementation, this would call Alpaca API
        # For mock, return sample account info
        return BrokerAccountInfo(
            account_id=f"alpaca_acc_{user.id}",
            account_number="PA12345678",
            account_type="cash",
            buying_power=Money(Decimal('50000.00'), 'USD'),
            cash_balance=Money(Decimal('30000.00'), 'USD'),
            portfolio_value=Money(Decimal('85000.00'), 'USD'),
            day_trade_count=2,
            pattern_day_trader=False,
            trading_blocked=False,
            transfers_blocked=False,
            account_blocked=False,
            created_at=datetime.now() - timedelta(days=45),
            updated_at=datetime.now()
        )
    
    def get_historical_orders(self, user: User, limit: int = 100) -> List[BrokerOrder]:
        """Get historical orders from Alpaca broker"""
        # In a real implementation, this would call Alpaca API
        # For mock, return sample orders
        return [
            BrokerOrder(
                broker_order_id=f"alp_historical_{i}",
                client_order_id=uuid.uuid4(),
                symbol="AAPL",
                order_type=BrokerOrderType.MARKET,
                position_type=BrokerPositionType.LONG,
                quantity=50,
                side="buy",
                time_in_force="day",
                status="filled",
                filled_qty=50,
                avg_fill_price=Decimal('170.00'),
                created_at=datetime.now() - timedelta(days=i),
                updated_at=datetime.now() - timedelta(days=i)
            )
            for i in range(1, min(limit, 5) + 1)  # Return up to 5 historical orders
        ]


class BrokerAdapterManager:
    """
    Adapter manager to handle multiple broker integrations
    """
    
    def __init__(self):
        self._brokers = {
            BrokerType.MOCK: MockBrokerService(),
            # In a real implementation, you'd initialize with actual API credentials
            # BrokerType.ALPACA: AlpacaBrokerService(api_key, secret_key)
        }
    
    def get_broker_service(self, broker_type: BrokerType) -> BrokerIntegrationService:
        """Get the appropriate broker service based on broker type"""
        if broker_type in self._brokers:
            return self._brokers[broker_type]
        else:
            raise ValueError(f"Broker type {broker_type} not supported")
    
    def get_available_brokers(self) -> List[BrokerType]:
        """Get list of available broker types"""
        return list(self._brokers.keys())
    
    def execute_order(self, order: Order, user: User, broker_type: BrokerType) -> BrokerOrder:
        """Execute an order through the specified broker"""
        broker_service = self.get_broker_service(broker_type)
        return broker_service.place_order(order, user)
    
    def get_account_info(self, user: User, broker_type: BrokerType) -> BrokerAccountInfo:
        """Get account information from the specified broker"""
        broker_service = self.get_broker_service(broker_type)
        return broker_service.get_account_info(user)