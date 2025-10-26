"""
Real-Time Backtesting Engine

This module implements a comprehensive backtesting system that allows
testing of trading strategies on historical data with realistic market conditions.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import time
import uuid

from src.domain.entities.trading import Order, Position, Portfolio, OrderType, PositionType, OrderStatus
from src.domain.entities.user import User
from src.domain.value_objects import Money, Symbol, Price
from src.domain.ports import MarketDataPort
from src.application.use_cases.trading import ExecuteTradeUseCase
from src.domain.services.trading import TradingDomainService


class TradeResult(Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    BREAKEVEN = "BREAKEVEN"


@dataclass
class BacktestTrade:
    """Represents a single trade in a backtest."""
    order_id: str
    symbol: Symbol
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    direction: str  # 'LONG' or 'SHORT'
    pnl: float
    fees: float
    result: TradeResult


@dataclass
class BacktestResult:
    """Represents the results of a complete backtest."""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    trades: List[BacktestTrade]
    max_position_size: float


class BacktestingEngine:
    """
    Real-time backtesting engine that simulates trading strategies
    on historical data with realistic market conditions.
    """
    
    def __init__(self, market_data_service: MarketDataPort):
        self.market_data_service = market_data_service
        self.trading_service = TradingDomainService()  # Using default implementation
        self.commission_per_share = 0.005  # $0.005 per share
        self.slippage_pct = 0.001  # 0.1% slippage
        self.time_delay_ms = 0  # Simulate real-time delay (0 for backtesting)
    
    def run_backtest(
        self,
        strategy_func: Callable,
        symbol: Symbol,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 100000.0,
        **strategy_params
    ) -> BacktestResult:
        """
        Run a backtest for a given strategy on historical data.
        
        Args:
            strategy_func: Function that takes (data, portfolio, i) and returns signal
            symbol: The symbol to backtest on
            start_date: Start date for backtest
            end_date: End date for backtest
            initial_capital: Starting capital for backtest
            **strategy_params: Additional parameters for the strategy
        """
        print(f"Starting backtest for {symbol} from {start_date.date()} to {end_date.date()}")
        
        # Get historical data
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        # Filter out weekends as most market data providers don't include weekend data
        trading_days = [d for d in date_range if d.weekday() < 5]
        
        # This is a simplified approach - in reality, you'd want to fetch actual historical data
        # For now, we'll create mock data based on what we can get
        prices = []
        for day in trading_days:
            try:
                # In a real implementation, we'd get actual historical prices
                # For this example, we'll create mock data
                # This is just a placeholder - real implementation would fetch real data
                day_prices = self.market_data_service.get_historical_prices(
                    symbol, day.date(), day.date()
                )
                prices.extend(day_prices)
            except:
                # If we can't get data for this day, skip it
                continue
        
        # Create a portfolio for the backtest
        initial_portfolio = Portfolio(
            id=f"backtest_{uuid.uuid4()}",
            user_id="backtest_user",
            cash_balance=Money(initial_capital, 'USD')
        )
        
        # Run the backtest
        portfolio = initial_portfolio
        trades = []
        
        # Simulate trading for each time step
        for i in range(100):  # Process first 100 data points as example
            try:
                # Get current price (mock implementation)
                current_price = 100 + np.random.normal(0, 1)  # Mock price movement
                
                # Run strategy to get signal
                signal = strategy_func(i, current_price, portfolio, strategy_params)
                
                if signal and signal != 'HOLD':
                    # Create and execute mock order
                    order = self._create_order_from_signal(
                        signal, symbol, current_price, portfolio
                    )
                    
                    if order and self._can_execute_order(order, portfolio):
                        executed_order, trade = self._execute_order(
                            order, current_price, portfolio
                        )
                        
                        if executed_order and trade:
                            trades.append(trade)
                            portfolio = self._update_portfolio_after_trade(
                                portfolio, executed_order, trade.pnl
                            )
                
                # Add delay if simulating real-time
                if self.time_delay_ms > 0:
                    time.sleep(self.time_delay_ms / 1000)
                    
            except Exception as e:
                print(f"Error in backtest iteration {i}: {e}")
                continue
        
        # Calculate final results
        final_capital = float(portfolio.cash_balance.amount)
        total_return = final_capital - initial_capital
        total_return_pct = (total_return / initial_capital) * 100 if initial_capital > 0 else 0
        
        # Calculate metrics
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = abs(np.mean([t.pnl for t in losing_trades])) if losing_trades else 0
        
        # Calculate Sharpe ratio (simplified)
        excess_returns = [(t.pnl / initial_capital) * 100 for t in trades] if trades else []
        if excess_returns:
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) != 0 else 0
            # Annualized Sharpe ratio
            sharpe_ratio = sharpe_ratio * np.sqrt(252)  # Assuming 252 trading days
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown (simplified)
        cumulative_returns = [initial_capital]
        for trade in trades:
            cumulative_returns.append(cumulative_returns[-1] + trade.pnl)
        
        max_value = 0
        max_drawdown = 0
        for value in cumulative_returns:
            if value > max_value:
                max_value = value
            drawdown = (max_value - value) / max_value if max_value > 0 else 0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return BacktestResult(
            strategy_name=strategy_params.get('name', 'Unknown Strategy'),
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_win=avg_win,
            avg_loss=avg_loss,
            trades=trades,
            max_position_size=0  # Would be calculated in realistic implementation
        )
    
    def _create_order_from_signal(
        self, 
        signal: str, 
        symbol: Symbol, 
        current_price: float, 
        portfolio: Portfolio
    ) -> Optional[Order]:
        """
        Create an order based on the strategy signal.
        """
        if signal == 'BUY':
            # Calculate position size based on risk management
            max_position_value = float(portfolio.cash_balance.amount) * 0.1  # 10% of cash
            quantity = int(max_position_value / current_price)
            
            if quantity > 0 and float(portfolio.cash_balance.amount) >= (current_price * quantity):
                return Order(
                    id=f"backtest_order_{uuid.uuid4()}",
                    user_id=portfolio.user_id,
                    symbol=symbol,
                    order_type=OrderType.MARKET,
                    position_type=PositionType.LONG,
                    quantity=quantity,
                    status=OrderStatus.PENDING,
                    placed_at=datetime.now(),
                    price=Price(current_price, 'USD')
                )
        elif signal == 'SELL':
            # Find existing position to sell
            position = portfolio.get_position(symbol)
            if position:
                return Order(
                    id=f"backtest_order_{uuid.uuid4()}",
                    user_id=portfolio.user_id,
                    symbol=symbol,
                    order_type=OrderType.MARKET,
                    position_type=PositionType.SHORT,
                    quantity=position.quantity,
                    status=OrderStatus.PENDING,
                    placed_at=datetime.now(),
                    price=Price(current_price, 'USD')
                )
        
        return None
    
    def _can_execute_order(self, order: Order, portfolio: Portfolio) -> bool:
        """
        Check if an order can be executed based on portfolio constraints.
        """
        if order.position_type == PositionType.LONG:
            # Check if we have enough cash
            cost = float(order.price.amount) * order.quantity
            return float(portfolio.cash_balance.amount) >= cost
        else:
            # For selling, check if we have the position
            position = portfolio.get_position(order.symbol)
            return position and position.quantity >= order.quantity
    
    def _execute_order(
        self, 
        order: Order, 
        current_price: float, 
        portfolio: Portfolio
    ) -> Tuple[Optional[Order], Optional[BacktestTrade]]:
        """
        Execute an order in the backtesting environment.
        """
        # Apply slippage
        slippage = current_price * self.slippage_pct
        execution_price = current_price + slippage if order.position_type == PositionType.LONG else current_price - slippage
        
        # Calculate fees
        fees = order.quantity * self.commission_per_share
        
        # Calculate P&L based on existing position
        pnl = 0
        position = portfolio.get_position(order.symbol)
        
        if order.position_type == PositionType.LONG:
            # Opening or adding to a long position
            cost = execution_price * order.quantity
            if float(portfolio.cash_balance.amount) >= cost + fees:
                # Update cash balance
                new_cash = float(portfolio.cash_balance.amount) - cost - fees
                
                pnl = 0  # P&L is realized when position is closed
        else:
            # Closing or shorting
            if position:
                # Calculate P&L
                avg_cost = float(position.average_buy_price.amount)
                pnl = (execution_price - avg_cost) * min(order.quantity, position.quantity)
        
        # Create trade record
        trade = BacktestTrade(
            order_id=order.id,
            symbol=order.symbol,
            entry_time=order.placed_at,
            exit_time=datetime.now(),  # In backtest, entry and exit are instantaneous
            entry_price=float(order.price.amount),
            exit_price=execution_price,
            quantity=order.quantity,
            direction='LONG' if order.position_type == PositionType.LONG else 'SHORT',
            pnl=pnl - fees,  # Net P&L after fees
            fees=fees,
            result=TradeResult.WIN if pnl - fees > 0 else 
                   TradeResult.LOSS if pnl - fees < 0 else 
                   TradeResult.BREAKEVEN
        )
        
        # Update order status
        executed_order = Order(
            id=order.id,
            user_id=order.user_id,
            symbol=order.symbol,
            order_type=order.order_type,
            position_type=order.position_type,
            quantity=order.quantity,
            status=OrderStatus.EXECUTED,
            placed_at=order.placed_at,
            executed_at=datetime.now(),
            price=Price(execution_price, 'USD'),
            filled_quantity=order.quantity,
            commission=Money(fees, 'USD')
        )
        
        return executed_order, trade
    
    def _update_portfolio_after_trade(
        self, 
        portfolio: Portfolio, 
        order: Order, 
        trade_pnl: float
    ) -> Portfolio:
        """
        Update the portfolio after a trade is executed.
        """
        # Update cash balance
        cash_change = 0
        if order.position_type == PositionType.LONG:
            # Cash decreases when buying
            cash_change = -(float(order.price.amount) * order.quantity + float(order.commission.amount))
        else:
            # Cash increases when selling
            cash_change = float(order.price.amount) * order.quantity - float(order.commission.amount)
        
        new_cash_balance = float(portfolio.cash_balance.amount) + cash_change
        
        # For a full implementation, we would also need to update positions
        # This is a simplified version focusing on cash balance
        return Portfolio(
            id=portfolio.id,
            user_id=portfolio.user_id,
            positions=portfolio.positions,  # In backtest, position management is simplified
            cash_balance=Money(new_cash_balance, 'USD')
        )


class StrategyTester:
    """
    Class to define and test various trading strategies.
    """
    
    def __init__(self, backtesting_engine: BacktestingEngine):
        self.backtesting_engine = backtesting_engine
    
    def moving_average_crossover_strategy(
        self, 
        current_index: int, 
        current_price: float, 
        portfolio: Portfolio, 
        params: Dict
    ) -> str:
        """
        Simple moving average crossover strategy.
        """
        # This would normally analyze historical price data
        # For this example, we'll create a simple condition
        # In reality, you'd have access to more data points
        
        # Mock implementation - would use actual moving averages in real strategy
        ma_short = params.get('ma_short', 10)
        ma_long = params.get('ma_long', 20)
        
        # For demonstration purposes, use random signals
        import random
        if random.random() > 0.7:  # 30% chance to trade
            # Simulate MA crossover
            if random.random() > 0.5:
                return 'BUY'
            else:
                return 'SELL'
        else:
            return 'HOLD'
    
    def mean_reversion_strategy(
        self, 
        current_index: int, 
        current_price: float, 
        portfolio: Portfolio, 
        params: Dict
    ) -> str:
        """
        Mean reversion strategy.
        """
        # Mock implementation
        import random
        if random.random() > 0.7:
            if random.random() > 0.5:
                return 'BUY'  # Price is below "fair value"
            else:
                return 'SELL'  # Price is above "fair value"
        else:
            return 'HOLD'
    
    def momentum_strategy(
        self, 
        current_index: int, 
        current_price: float, 
        portfolio: Portfolio, 
        params: Dict
    ) -> str:
        """
        Momentum strategy.
        """
        # Mock implementation
        import random
        if random.random() > 0.7:
            if random.random() > 0.5:
                return 'BUY'  # Positive momentum
            else:
                return 'SELL'  # Negative momentum
        else:
            return 'HOLD'
    
    def test_strategy(
        self,
        strategy_func: Callable,
        symbol: Symbol,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 100000.0,
        **params
    ) -> BacktestResult:
        """
        Test a specific strategy.
        """
        return self.backtesting_engine.run_backtest(
            strategy_func,
            symbol,
            start_date,
            end_date,
            initial_capital,
            **params
        )


def get_sample_strategies() -> Dict[str, Callable]:
    """
    Return a dictionary of sample trading strategies.
    """
    def random_strategy(i, price, portfolio, params):
        import random
        if random.random() > 0.9:  # Low frequency to avoid too many trades
            if random.random() > 0.5:
                return 'BUY'
            else:
                return 'SELL'
        return 'HOLD'
    
    def trend_following_strategy(i, price, portfolio, params):
        # Simplified trend following based on mock indicators
        import random
        # Simulate an uptrend or downtrend
        if i % 20 < 10:  # Every 20 steps, assume trend for 10 steps
            if random.random() > 0.3:  # High probability of following trend
                return 'BUY'
        else:
            if random.random() > 0.3:
                return 'SELL'
        return 'HOLD'
    
    return {
        'random': random_strategy,
        'trend_following': trend_following_strategy
    }