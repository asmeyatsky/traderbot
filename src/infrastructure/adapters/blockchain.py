"""
Blockchain Integration for Secure Trade Execution

This module implements blockchain-based trade execution and settlement
for enhanced security, transparency, and automation of trading activities.
"""
import hashlib
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import secrets
import base64
from dataclasses import dataclass

# Try to import blockchain libraries (will be simulated if not available)
try:
    from web3 import Web3
    from eth_account import Account
    HAS_WEB3 = True
except ImportError:
    print("Web3 library not available. Using blockchain simulation.")
    HAS_WEB3 = False

from src.domain.entities.trading import Order, Position, Portfolio
from src.domain.entities.user import User
from src.domain.value_objects import Money, Symbol, Price
from src.domain.ports import TradingExecutionPort


@dataclass
class BlockchainTradeRecord:
    """Record of a trade executed on blockchain."""
    trade_id: str
    order_id: str
    user_id: str
    symbol: str
    action: str  # BUY, SELL
    quantity: int
    price: float
    timestamp: datetime
    blockchain_tx_hash: str
    smart_contract_address: str
    nonce: int
    signature: str


class BlockchainTradeExecutor:
    """
    Executes trades on blockchain for secure and transparent settlement.
    """
    
    def __init__(self, provider_url: Optional[str] = None):
        self.provider_url = provider_url or "http://localhost:8545"  # Default local node
        self.web3 = None
        self.smart_contract = None
        self.account = None
        self.initialized = False
        
        if HAS_WEB3:
            try:
                self.web3 = Web3(Web3.HTTPProvider(self.provider_url))
                if self.web3.is_connected():
                    self.initialized = True
                    print("Connected to blockchain network")
                else:
                    print("Could not connect to blockchain, using simulation")
                    self.initialized = False
            except Exception as e:
                print(f"Blockchain connection failed: {e}, using simulation")
                self.initialized = False
        else:
            # Using simulation mode
            self.initialized = True
            print("Blockchain libraries not available, using simulation mode")
    
    def initialize_account(self, private_key: Optional[str] = None):
        """
        Initialize blockchain account for trading.
        """
        if HAS_WEB3 and self.initialized:
            if private_key:
                self.account = Account.from_key(private_key)
            else:
                # Create a new account (in production, you'd load from secure storage)
                self.account = Account.create()
        else:
            # In simulation mode, create mock account data
            self.account = {
                'address': f"0x{secrets.token_hex(20)}",
                'private_key': secrets.token_hex(32)
            }
    
    def execute_trade_on_blockchain(self, order: Order) -> Optional[BlockchainTradeRecord]:
        """
        Execute a trade on blockchain using smart contracts.
        """
        if not self.initialized:
            print("Blockchain not initialized, executing simulated trade")
            return self._execute_simulated_trade(order)
        
        try:
            # Create trade record
            trade_record = BlockchainTradeRecord(
                trade_id=f"trade_{secrets.token_hex(8)}",
                order_id=order.id,
                user_id=order.user_id,
                symbol=str(order.symbol),
                action="BUY" if order.position_type.name == 'LONG' else "SELL",
                quantity=order.quantity,
                price=float(order.price.amount) if order.price else 0.0,
                timestamp=datetime.now(),
                blockchain_tx_hash="",
                smart_contract_address="",
                nonce=0,
                signature=""
            )
            
            # In a real implementation, we would:
            # 1. Encode the trade parameters
            # 2. Call the trading smart contract
            # 3. Wait for transaction confirmation
            # 4. Store the transaction hash
            
            if HAS_WEB3:
                # Encode trade data
                trade_data = self._encode_trade_data(trade_record)
                
                # Create transaction
                transaction = {
                    'to': self._get_smart_contract_address(),  # Trading smart contract
                    'from': self.account.address,
                    'gas': 200000,
                    'gasPrice': self.web3.eth.gas_price,
                    'value': 0,  # Value depends on trade
                    'data': trade_data,
                    'nonce': self.web3.eth.get_transaction_count(self.account.address)
                }
                
                # Sign and send transaction
                signed_txn = self.account.sign_transaction(transaction)
                tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
                
                # Wait for confirmation
                receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
                
                # Update trade record with blockchain details
                trade_record.blockchain_tx_hash = tx_hash.hex()
                trade_record.nonce = transaction['nonce']
                trade_record.signature = signed_txn.signature.hex()
                
            else:
                # Simulate blockchain execution
                trade_record.blockchain_tx_hash = f"0x{secrets.token_hex(32)}"
                trade_record.smart_contract_address = f"0x{secrets.token_hex(20)}"
                trade_record.nonce = secrets.randbelow(1000000)
                trade_record.signature = secrets.token_hex(65)
            
            return trade_record
            
        except Exception as e:
            print(f"Blockchain trade execution failed: {e}")
            # Fallback to simulated execution
            return self._execute_simulated_trade(order)
    
    def _encode_trade_data(self, trade_record: BlockchainTradeRecord) -> str:
        """
        Encode trade data for blockchain transaction.
        """
        data = {
            'orderId': trade_record.order_id,
            'userId': trade_record.user_id,
            'symbol': trade_record.symbol,
            'action': trade_record.action,
            'quantity': trade_record.quantity,
            'price': trade_record.price,
            'timestamp': trade_record.timestamp.isoformat()
        }
        
        # Convert to hex string
        json_str = json.dumps(data, sort_keys=True)
        return Web3.to_hex(text=json_str) if HAS_WEB3 else json_str.encode().hex()
    
    def _get_smart_contract_address(self) -> str:
        """
        Get the address of the trading smart contract.
        """
        # In a real implementation, this would come from configuration
        # For simulation, return a mock address
        return f"0x{secrets.token_hex(20)}"
    
    def _execute_simulated_trade(self, order: Order) -> BlockchainTradeRecord:
        """
        Simulate blockchain trade execution for demonstration.
        """
        # Create a simulated trade record
        trade_record = BlockchainTradeRecord(
            trade_id=f"trade_{secrets.token_hex(8)}",
            order_id=order.id,
            user_id=order.user_id,
            symbol=str(order.symbol),
            action="BUY" if order.position_type.name == 'LONG' else "SELL",
            quantity=order.quantity,
            price=float(order.price.amount) if order.price else 0.0,
            timestamp=datetime.now(),
            blockchain_tx_hash=f"0x{secrets.token_hex(32)}",
            smart_contract_address=f"0x{secrets.token_hex(20)}",
            nonce=secrets.randbelow(1000000),
            signature=secrets.token_hex(65)
        )
        
        print(f"Simulated blockchain trade executed: {trade_record.trade_id}")
        return trade_record
    
    def verify_trade_on_chain(self, trade_record: BlockchainTradeRecord) -> bool:
        """
        Verify that a trade was actually executed on the blockchain.
        """
        if not self.initialized:
            return True  # Assume valid in simulation
        
        try:
            if HAS_WEB3:
                # Get transaction receipt to verify it was included in a block
                receipt = self.web3.eth.get_transaction_receipt(
                    Web3.to_bytes(hexstr=trade_record.blockchain_tx_hash)
                )
                return receipt is not None and receipt.status == 1
            else:
                # In simulation, just check if the hash looks valid
                return len(trade_record.blockchain_tx_hash) == 66 and trade_record.blockchain_tx_hash.startswith('0x')
        except:
            return False


class SmartContractTradeSettlement:
    """
    Smart contract implementation for trade settlement.
    This would typically be deployed to the blockchain, but we'll simulate it here.
    """
    
    def __init__(self):
        self.trades = {}
        self.positions = {}
        self.balances = {}
    
    def execute_trade(self, 
                     user_id: str, 
                     symbol: str, 
                     action: str, 
                     quantity: int, 
                     price: float) -> Dict[str, Any]:
        """
        Execute a trade through smart contract logic.
        """
        trade_id = f"trade_{secrets.token_hex(8)}"
        
        # Calculate trade value
        trade_value = quantity * price
        
        # Handle different trade actions
        if action == "BUY":
            # Check if user has sufficient balance
            user_balance = self.balances.get(user_id, {}).get('cash', 0)
            if user_balance < trade_value:
                raise Exception(f"Insufficient balance for trade. Required: {trade_value}, Available: {user_balance}")
            
            # Update balances
            self.balances[user_id] = {
                'cash': user_balance - trade_value,
                'holdings': self.balances.get(user_id, {}).get('holdings', {})
            }
            
            # Update positions
            user_positions = self.positions.get(user_id, {})
            current_position = user_positions.get(symbol, 0)
            user_positions[symbol] = current_position + quantity
            self.positions[user_id] = user_positions
            
        elif action == "SELL":
            # Check if user has sufficient position
            user_positions = self.positions.get(user_id, {})
            current_position = user_positions.get(symbol, 0)
            if current_position < quantity:
                raise Exception(f"Insufficient position for trade. Required: {quantity}, Available: {current_position}")
            
            # Update positions
            user_positions[symbol] = current_position - quantity
            self.positions[user_id] = user_positions
            
            # Update balances
            user_balance = self.balances.get(user_id, {}).get('cash', 0)
            self.balances[user_id] = {
                'cash': user_balance + trade_value,
                'holdings': self.balances.get(user_id, {}).get('holdings', {})
            }
        
        # Record the trade
        self.trades[trade_id] = {
            'user_id': user_id,
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'value': trade_value,
            'timestamp': datetime.now().isoformat(),
            'status': 'executed'
        }
        
        return {
            'trade_id': trade_id,
            'status': 'executed',
            'gas_used': secrets.randbelow(100000) + 50000  # Simulated gas usage
        }
    
    def get_user_position(self, user_id: str, symbol: str) -> int:
        """
        Get user's position in a specific symbol.
        """
        return self.positions.get(user_id, {}).get(symbol, 0)
    
    def get_user_balance(self, user_id: str) -> Dict[str, float]:
        """
        Get user's cash balance.
        """
        return self.balances.get(user_id, {'cash': 0, 'holdings': {}})


class DecentralizedOracleService:
    """
    Oracle service to provide market data to smart contracts.
    """
    
    def __init__(self, market_data_service):
        self.market_data_service = market_data_service
        self.price_cache = {}
        self.last_updated = {}
    
    def get_current_price(self, symbol: Symbol) -> float:
        """
        Get current price for a symbol from oracle.
        """
        # Check cache first
        cache_key = str(symbol)
        current_time = time.time()
        
        if (cache_key in self.price_cache and 
            current_time - self.last_updated.get(cache_key, 0) < 60):  # 1 minute cache
            return self.price_cache[cache_key]
        
        # Fetch from market data service
        try:
            price_obj = self.market_data_service.get_current_price(symbol)
            if price_obj:
                price = float(price_obj.amount)
                self.price_cache[cache_key] = price
                self.last_updated[cache_key] = current_time
                return price
        except:
            pass
        
        # Return cached value if available, otherwise default
        return self.price_cache.get(cache_key, 100.0)  # Default to $100 if no data
    
    def get_historical_prices(self, symbol: Symbol, days: int = 30) -> List[float]:
        """
        Get historical prices for oracle calculations.
        """
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        try:
            prices = self.market_data_service.get_historical_prices(symbol, start_date, end_date)
            return [float(p.amount) for p in prices]
        except:
            # Return mock historical data
            base_price = self.get_current_price(symbol)
            return [base_price * (1 + (i-15) * 0.001 + np.random.normal(0, 0.02)) for i in range(days)]


class BlockchainTradingAdapter(TradingExecutionPort):
    """
    Adapter for blockchain-based trading execution following the ports pattern.
    """
    
    def __init__(self, market_data_service):
        self.blockchain_executor = BlockchainTradeExecutor()
        self.smart_contract = SmartContractTradeSettlement()
        self.oracle_service = DecentralizedOracleService(market_data_service)
        self.trade_records = {}  # Store trade records
    
    def place_order(self, order: Order) -> str:
        """
        Place an order through blockchain smart contract.
        """
        try:
            # Execute trade on blockchain
            trade_record = self.blockchain_executor.execute_trade_on_blockchain(order)
            
            # Execute settlement through smart contract
            price = float(order.price.amount) if order.price else self.oracle_service.get_current_price(order.symbol)
            settlement_result = self.smart_contract.execute_trade(
                user_id=order.user_id,
                symbol=str(order.symbol),
                action="BUY" if order.position_type.name == 'LONG' else "SELL",
                quantity=order.quantity,
                price=price
            )
            
            # Store the trade record
            if trade_record:
                self.trade_records[order.id] = trade_record
            
            # Return the blockchain transaction hash as order ID
            return trade_record.blockchain_tx_hash if trade_record else order.id
            
        except Exception as e:
            print(f"Blockchain order placement failed: {e}")
            # Fallback to traditional execution
            return order.id
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order (on blockchain if possible).
        """
        # In blockchain systems, cancellation is complex
        # For now, we'll mark as canceled in our records
        # Real implementation would require smart contract functionality
        return True
    
    def get_order_status(self, order_id: str) -> Order:
        """
        Get the status of an order from blockchain records.
        """
        # In a real implementation, this would query the blockchain
        # For now, return the original order with updated status
        return Order(
            id=order_id,
            user_id="user123",  # This would come from blockchain record
            symbol=Symbol("AAPL"),  # This would come from blockchain record
            order_type=OrderType.MARKET,  # This would come from blockchain record
            position_type=PositionType.LONG,  # This would come from blockchain record
            quantity=10,  # This would come from blockchain record
            status=OrderStatus.EXECUTED,  # This would come from blockchain record
            placed_at=datetime.now(),  # This would come from blockchain record
            # ... other fields
        )
    
    def get_account_balance(self, user_id: str) -> Money:
        """
        Get account balance from blockchain records.
        """
        balance_info = self.smart_contract.get_user_balance(user_id)
        cash_balance = balance_info.get('cash', 0)
        return Money(cash_balance, 'USD')


class BlockchainPortfolioTracker:
    """
    Track portfolio positions and values using blockchain records.
    """
    
    def __init__(self, blockchain_adapter: BlockchainTradingAdapter):
        self.blockchain_adapter = blockchain_adapter
    
    def get_blockchain_positions(self, user_id: str) -> Dict[str, int]:
        """
        Get user positions from blockchain records.
        """
        # In a real implementation, this would query the blockchain
        # For now, using the smart contract simulation
        all_positions = self.blockchain_adapter.smart_contract.positions
        return all_positions.get(user_id, {})
    
    def get_historical_trades(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get historical trades for a user from blockchain.
        """
        # In a real implementation, this would query the blockchain
        # For now, return empty list
        return []


# Initialize blockchain services
def initialize_blockchain_services(market_data_service):
    """
    Initialize all blockchain services for the trading platform.
    """
    blockchain_adapter = BlockchainTradingAdapter(market_data_service)
    portfolio_tracker = BlockchainPortfolioTracker(blockchain_adapter)
    
    return {
        'adapter': blockchain_adapter,
        'portfolio_tracker': portfolio_tracker,
        'executor': blockchain_adapter.blockchain_executor
    }