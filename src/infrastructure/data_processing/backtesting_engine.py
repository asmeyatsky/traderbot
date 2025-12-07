"""
Backtesting Engine for the AI Trading Platform

This module implements a comprehensive backtesting framework that allows
trading strategies to be tested against historical data as required by the PRD.
It supports various data sources, performance metrics, and scenario analysis.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Callable
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
import yfinance as yf
from enum import Enum

from src.domain.entities.trading import Order, Position, Portfolio
from src.domain.entities.user import User
from src.domain.value_objects import Symbol, Money, Price
from src.application.use_cases.trading import ExecuteTradeUseCase
from src.infrastructure.data_processing.ml_model_service import MLModelService, TradingSignal


logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Result of a backtesting run."""
    final_portfolio_value: float
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    profit_factor: float
    trades: List[Dict]  # Detailed trade records


@dataclass
class BacktestConfiguration:
    """Configuration for a backtesting run."""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    symbols: List[Symbol]
    commission_rate: float = 0.005  # 0.5% commission
    slippage_rate: float = 0.001    # 0.1% slippage
    rebalance_frequency: str = 'daily'  # daily, weekly, monthly
    risk_per_trade: float = 0.02      # 2% risk per trade


class DataProvider(ABC):
    """Abstract base class for data providers."""

    @abstractmethod
    def get_historical_data(self, symbol: Symbol, start: datetime, end: datetime) -> pd.DataFrame:
        """Get historical price data for a symbol."""
        pass

    @abstractmethod
    def get_dividend_data(self, symbol: Symbol, start: datetime, end: datetime) -> pd.DataFrame:
        """Get dividend data for a symbol."""
        pass


class YahooFinanceDataProvider(DataProvider):
    """Data provider implementation using Yahoo Finance."""

    def get_historical_data(self, symbol: Symbol, start: datetime, end: datetime) -> pd.DataFrame:
        """Get historical price data from Yahoo Finance."""
        try:
            # Format dates for yfinance
            start_str = start.strftime('%Y-%m-%d')
            end_str = end.strftime('%Y-%m-%d')
            
            # Fetch data
            ticker = yf.Ticker(str(symbol))
            data = ticker.history(start=start_str, end=end_str, interval="1d")
            
            if data.empty:
                logger.warning(f"No data found for {symbol} in period {start} to {end}")
                return pd.DataFrame()
            
            # Ensure required columns exist
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in data.columns:
                    data[col] = np.nan
            
            # Add symbol column for multi-symbol processing
            data['Symbol'] = str(symbol)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def get_dividend_data(self, symbol: Symbol, start: datetime, end: datetime) -> pd.DataFrame:
        """Get dividend data from Yahoo Finance."""
        try:
            start_str = start.strftime('%Y-%m-%d')
            end_str = end.strftime('%Y-%m-%d')
            
            ticker = yf.Ticker(str(symbol))
            dividends = ticker.dividends
            
            # Filter dividends within date range
            mask = (dividends.index >= start) & (dividends.index <= end)
            filtered_dividends = dividends[mask]
            
            return pd.DataFrame({
                'Date': filtered_dividends.index,
                'Dividend': filtered_dividends.values
            })
            
        except Exception as e:
            logger.error(f"Error fetching dividends for {symbol}: {e}")
            return pd.DataFrame()


class Strategy(ABC):
    """Abstract base class for trading strategies."""

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, state: Dict) -> List[Tuple[datetime, str, Dict]]:
        """
        Generate trading signals based on data.
        
        Args:
            data: Historical market data
            state: Current portfolio state and other context
            
        Returns:
            List of (timestamp, signal, additional_info) tuples
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of the strategy."""
        pass


class SMACrossoverStrategy(Strategy):
    """Simple Moving Average Crossover Strategy."""

    def __init__(self, short_window: int = 20, long_window: int = 50):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data: pd.DataFrame, state: Dict) -> List[Tuple[datetime, str, Dict]]:
        """Generate buy/sell signals based on SMA crossover."""
        if len(data) < self.long_window:
            return []
        
        # Calculate moving averages
        data['SMA_short'] = data['Close'].rolling(window=self.short_window).mean()
        data['SMA_long'] = data['Close'].rolling(window=self.long_window).mean()
        
        signals = []
        
        for i in range(self.long_window, len(data)):
            current = data.iloc[i]
            previous = data.iloc[i-1]
            
            # Check for crossover
            if (previous['SMA_short'] <= previous['SMA_long'] and 
                current['SMA_short'] > current['SMA_long']):
                # Golden cross - buy signal
                signals.append((current.name, 'BUY', {
                    'reason': 'SMA_short crossed above SMA_long',
                    'price': current['Close'],
                    'SMA_short': current['SMA_short'],
                    'SMA_long': current['SMA_long']
                }))
            elif (previous['SMA_short'] >= previous['SMA_long'] and 
                  current['SMA_short'] < current['SMA_long']):
                # Death cross - sell signal
                signals.append((current.name, 'SELL', {
                    'reason': 'SMA_short crossed below SMA_long',
                    'price': current['Close'],
                    'SMA_short': current['SMA_short'],
                    'SMA_long': current['SMA_long']
                }))
        
        return signals

    def get_strategy_name(self) -> str:
        return f"SMA Crossover ({self.short_window}/{self.long_window})"


class MLStrategy(Strategy):
    """Machine Learning-based trading strategy."""

    def __init__(self, ml_service: MLModelService, confidence_threshold: float = 0.7):
        self.ml_service = ml_service
        self.confidence_threshold = confidence_threshold

    def generate_signals(self, data: pd.DataFrame, state: Dict) -> List[Tuple[datetime, str, Dict]]:
        """Generate signals using ML model predictions."""
        if data.empty:
            return []
        
        symbol = Symbol(data['Symbol'].iloc[0] if 'Symbol' in data.columns else 'UNKNOWN')
        
        signals = []
        
        # For each day, get ML prediction and generate signal if confidence is high enough
        for index, row in data.iterrows():
            try:
                # Get ML signal for this date
                ml_signal = self.ml_service.predict_price_direction(symbol)
                
                if ml_signal.confidence >= self.confidence_threshold:
                    if ml_signal.signal == 'BUY' and ml_signal.score > 0.1:
                        signals.append((index, 'BUY', {
                            'reason': f'ML signal: {ml_signal.explanation}',
                            'confidence': ml_signal.confidence,
                            'score': ml_signal.score,
                            'price': row['Close']
                        }))
                    elif ml_signal.signal == 'SELL' and ml_signal.score < -0.1:
                        signals.append((index, 'SELL', {
                            'reason': f'ML signal: {ml_signal.explanation}',
                            'confidence': ml_signal.confidence,
                            'score': ml_signal.score,
                            'price': row['Close']
                        }))
            except Exception as e:
                logger.error(f"Error generating ML signal for {symbol} on {index}: {e}")
                continue
        
        return signals

    def get_strategy_name(self) -> str:
        return f"ML Strategy (threshold={self.confidence_threshold})"


class BacktestingEngine:
    """Main backtesting engine that executes strategies against historical data."""

    def __init__(self, 
                 data_provider: DataProvider,
                 strategy: Strategy,
                 config: BacktestConfiguration):
        self.data_provider = data_provider
        self.strategy = strategy
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def run_backtest(self) -> BacktestResult:
        """Run the backtest and return results."""
        self.logger.info(f"Starting backtest: {self.strategy.get_strategy_name()}")
        self.logger.info(f"Period: {self.config.start_date} to {self.config.end_date}")
        self.logger.info(f"Initial capital: ${self.config.initial_capital:,.2f}")
        
        # Initialize portfolio
        portfolio = Portfolio(
            id="backtest_portfolio",
            user_id="backtest_user",
            cash_balance=Money(self.config.initial_capital, 'USD')
        )
        
        # Fetch historical data for all symbols
        all_data = {}
        for symbol in self.config.symbols:
            data = self.data_provider.get_historical_data(
                symbol, self.config.start_date, self.config.end_date
            )
            if not data.empty:
                all_data[str(symbol)] = data
            else:
                self.logger.warning(f"No data available for symbol {symbol}")
        
        if not all_data:
            raise ValueError("No historical data available for any symbols")
        
        # Combine data for all symbols if we have multiple
        if len(all_data) == 1:
            combined_data = next(iter(all_data.values()))
            combined_data = combined_data.reset_index()
            combined_data['Symbol'] = next(iter(all_data.keys()))
        else:
            # For multi-symbol backtesting, we need to align dates and process accordingly
            # This is a simplified approach - real implementation would be more sophisticated
            all_dfs = []
            for symbol, data in all_data.items():
                data_copy = data.reset_index()
                data_copy['Symbol'] = symbol
                all_dfs.append(data_copy)
            
            combined_data = pd.concat(all_dfs, ignore_index=True)
            combined_data = combined_data.sort_values(['Date', 'Symbol'])
        
        # Initialize tracking variables
        portfolio_values = []
        trades = []
        holding_positions = {}
        
        # Process each date
        for date in pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq='D'
        ):
            if date.weekday() > 4:  # Skip weekends
                continue
                
            # Get data for this date
            date_data = combined_data[
                pd.to_datetime(combined_data['Date']).dt.date == date.date()
            ]
            
            if date_data.empty:
                continue
            
            # Update portfolio value
            current_portfolio_value = self._calculate_portfolio_value(
                portfolio, date_data, holding_positions
            )
            portfolio_values.append({
                'date': date,
                'value': current_portfolio_value,
                'cash': float(portfolio.cash_balance.amount)
            })
            
            # Get available cash for trading
            available_cash = float(portfolio.cash_balance.amount)
            
            # Generate signals for this date
            for _, row in date_data.iterrows():
                symbol = Symbol(row['Symbol'])
                
                # Prepare data slice for signal generation
                symbol_data = combined_data[
                    combined_data['Symbol'] == str(symbol)
                ].set_index('Date')
                
                # Get a slice of data up to current date
                symbol_data_slice = symbol_data[symbol_data.index <= date]
                
                if len(symbol_data_slice) < 50:  # Need sufficient data for indicators
                    continue
                
                current_state = {
                    'portfolio': portfolio,
                    'holding_positions': holding_positions,
                    'available_cash': available_cash
                }
                
                # Generate signals using the strategy
                signals = self.strategy.generate_signals(symbol_data_slice, current_state)
                
                # Process signals for current date
                for signal_time, signal_type, signal_info in signals:
                    if signal_time.date() == date.date():
                        trade_result = self._execute_trade(
                            portfolio, 
                            holding_positions, 
                            symbol, 
                            signal_type, 
                            signal_info,
                            row
                        )
                        
                        if trade_result:
                            trades.append(trade_result)
        
        # Calculate final results
        return self._calculate_results(portfolio_values, trades, self.config.initial_capital)

    def _calculate_portfolio_value(self, portfolio: Portfolio, current_data: pd.DataFrame, positions: Dict) -> float:
        """Calculate current portfolio value."""
        value = float(portfolio.cash_balance.amount)
        
        for symbol_str, position in positions.items():
            if position.quantity > 0:
                # Find current price for this symbol
                symbol_data = current_data[current_data['Symbol'] == symbol_str]
                if not symbol_data.empty:
                    current_price = symbol_data.iloc[0]['Close']
                    market_value = position.quantity * current_price
                    value += market_value
        
        return value

    def _execute_trade(self, 
                      portfolio: Portfolio, 
                      positions: Dict,
                      symbol: Symbol,
                      signal_type: str,
                      signal_info: Dict,
                      current_data: pd.Series) -> Optional[Dict]:
        """Execute a trade based on signal."""
        try:
            price = float(current_data['Close'])
            commission = price * self.config.commission_rate
            execution_price = price * (1 + self.config.slippage_rate) if signal_type == 'BUY' else price * (1 - self.config.slippage_rate)
            
            # Calculate position size based on risk management
            portfolio_value = float(portfolio.cash_balance.amount) + sum(
                pos.quantity * float(pos.average_buy_price.amount) for pos in positions.values()
            )
            max_risk_amount = portfolio_value * self.config.risk_per_trade
            
            if signal_type == 'BUY':
                # Calculate how much to buy based on available cash and risk
                if portfolio.cash_balance.amount < execution_price:
                    return None  # Not enough cash
                
                # For now, buy one share - in real strategy this would be more sophisticated
                quantity = 1  # This would be calculated based on strategy and risk
                
                if quantity * execution_price > float(portfolio.cash_balance.amount):
                    # Reduce quantity to affordable amount
                    quantity = int(float(portfolio.cash_balance.amount) / execution_price)
                
                if quantity <= 0:
                    return None
                
                # Deduct from cash
                cost = quantity * execution_price + commission
                if cost > float(portfolio.cash_balance.amount):
                    return None
                
                # Update portfolio cash
                portfolio.cash_balance = Money(
                    portfolio.cash_balance.amount - Decimal(str(cost)),
                    portfolio.cash_balance.currency
                )
                
                # Update or create position
                symbol_str = str(symbol)
                if symbol_str in positions:
                    # Average down or add to position
                    existing_pos = positions[symbol_str]
                    new_quantity = existing_pos.quantity + quantity
                    new_avg_price = (
                        (existing_pos.average_buy_price.amount * Decimal(str(existing_pos.quantity))) + 
                        (Decimal(str(execution_price)) * Decimal(str(quantity)))
                    ) / Decimal(str(new_quantity))
                    
                    positions[symbol_str] = Position(
                        id=existing_pos.id,
                        user_id=existing_pos.user_id,
                        symbol=existing_pos.symbol,
                        position_type=existing_pos.position_type,
                        quantity=new_quantity,
                        average_buy_price=Money(new_avg_price, 'USD'),
                        current_price=Money(Decimal(str(execution_price)), 'USD'),
                        created_at=existing_pos.created_at,
                        updated_at=datetime.now(),
                        unrealized_pnl=existing_pos.unrealized_pnl
                    )
                else:
                    positions[symbol_str] = Position(
                        id=f"{symbol_str}_position",
                        user_id=portfolio.user_id,
                        symbol=symbol,
                        position_type=PositionType.LONG,
                        quantity=quantity,
                        average_buy_price=Money(Decimal(str(execution_price)), 'USD'),
                        current_price=Money(Decimal(str(execution_price)), 'USD'),
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )
                
                return {
                    'date': current_data.name,
                    'symbol': str(symbol),
                    'action': signal_type,
                    'quantity': quantity,
                    'price': execution_price,
                    'commission': commission,
                    'reason': signal_info.get('reason', 'Strategy signal'),
                    'value': quantity * execution_price
                }
            
            elif signal_type == 'SELL':
                symbol_str = str(symbol)
                if symbol_str not in positions or positions[symbol_str].quantity <= 0:
                    return None  # No position to sell
                
                # For now, sell all shares - in real strategy this might be partial
                quantity = positions[symbol_str].quantity
                proceeds = quantity * execution_price - commission
                
                # Update portfolio cash
                portfolio.cash_balance = Money(
                    portfolio.cash_balance.amount + Decimal(str(proceeds)),
                    portfolio.cash_balance.currency
                )
                
                # Remove position
                del positions[symbol_str]
                
                return {
                    'date': current_data.name,
                    'symbol': str(symbol),
                    'action': signal_type,
                    'quantity': quantity,
                    'price': execution_price,
                    'commission': commission,
                    'reason': signal_info.get('reason', 'Strategy signal'),
                    'value': quantity * execution_price
                }
        
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return None
        
        return None

    def _calculate_results(self, portfolio_values: List[Dict], trades: List[Dict], initial_capital: float) -> BacktestResult:
        """Calculate performance metrics from backtest results."""
        if not portfolio_values:
            return BacktestResult(
                final_portfolio_value=initial_capital,
                total_return=0,
                annualized_return=0,
                volatility=0,
                sharpe_ratio=0,
                max_drawdown=0,
                win_rate=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                profit_factor=0,
                trades=trades
            )
        
        # Extract portfolio values over time
        values = [pv['value'] for pv in portfolio_values]
        returns = np.diff(values) / values[:-1]  # Daily returns
        
        if len(returns) == 0:
            return BacktestResult(
                final_portfolio_value=portfolio_values[-1]['value'],
                total_return=0,
                annualized_return=0,
                volatility=0,
                sharpe_ratio=0,
                max_drawdown=0,
                win_rate=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                profit_factor=0,
                trades=trades
            )
        
        # Calculate metrics
        final_value = portfolio_values[-1]['value']
        total_return = (final_value - initial_capital) / initial_capital
        annualized_return = self._annualize_return(total_return, len(portfolio_values))
        
        volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
        risk_free_rate = 0.02  # 2% risk-free rate
        sharpe_ratio = (np.mean(returns) * 252 - risk_free_rate) / volatility if volatility > 0 else 0
        
        max_drawdown = self._calculate_max_drawdown(values)
        
        # Trade analysis
        winning_trades = 0
        losing_trades = 0
        gross_profit = 0
        gross_loss = 0
        
        for trade in trades:
            if trade['action'] == 'SELL' and 'pnl' in trade:
                pnl = trade['pnl']
                if pnl > 0:
                    winning_trades += 1
                    gross_profit += pnl
                else:
                    losing_trades += 1
                    gross_loss += abs(pnl)
        
        total_trades = len([t for t in trades if t['action'] in ['BUY', 'SELL']])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return BacktestResult(
            final_portfolio_value=final_value,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            profit_factor=profit_factor,
            trades=trades
        )

    def _annualize_return(self, total_return: float, num_days: int) -> float:
        """Convert total return to annualized return."""
        years = num_days / 252  # Trading days per year
        if years <= 0:
            return 0
        return (1 + total_return) ** (1 / years) - 1

    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown."""
        if len(portfolio_values) <= 1:
            return 0
        
        portfolio_values = np.array(portfolio_values)
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - running_max) / running_max
        return min(0, float(drawdowns.min()))


class StrategyComparator:
    """Compare multiple strategies against each other."""

    def __init__(self, data_provider: DataProvider):
        self.data_provider = data_provider

    def compare_strategies(self, 
                          strategies: List[Strategy],
                          config: BacktestConfiguration) -> Dict[str, BacktestResult]:
        """Compare multiple strategies with the same configuration."""
        results = {}
        
        for strategy in strategies:
            engine = BacktestingEngine(self.data_provider, strategy, config)
            result = engine.run_backtest()
            results[strategy.get_strategy_name()] = result
            
            # Log result summary
            logger.info(f"Strategy: {strategy.get_strategy_name()}")
            logger.info(f"  Final Value: ${result.final_portfolio_value:,.2f}")
            logger.info(f"  Total Return: {result.total_return * 100:.2f}%")
            logger.info(f"  Annualized Return: {result.annualized_return * 100:.2f}%")
            logger.info(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
            logger.info(f"  Max Drawdown: {result.max_drawdown * 100:.2f}%")
            logger.info(f"  Win Rate: {result.win_rate * 100:.2f}%")
            logger.info("---")
        
        return results


class MonteCarloBacktester:
    """Perform Monte Carlo simulation on backtest results."""

    def __init__(self, base_result: BacktestResult):
        self.base_result = base_result

    def run_simulation(self, num_simulations: int = 1000, confidence_level: float = 0.05) -> Dict:
        """Run Monte Carlo simulation to assess strategy robustness."""
        # Extract trade returns from the base result
        trade_returns = []
        for trade in self.base_result.trades:
            # Calculate return for this trade (this is simplified)
            if trade['action'] == 'SELL' and 'pnl' in trade and 'cost' in trade:
                return_pct = trade['pnl'] / trade['cost']
                trade_returns.append(return_pct)
        
        if not trade_returns:
            return {
                'error': 'No trade returns available for simulation',
                'original_result': self.base_result
            }
        
        # Run Monte Carlo simulations
        simulation_results = []
        initial_capital = self.base_result.final_portfolio_value  # Use as proxy for starting value
        
        for _ in range(num_simulations):
            # Randomly sample from historical trade returns
            sampled_returns = np.random.choice(trade_returns, size=len(trade_returns), replace=True)
            
            # Calculate portfolio value over time
            portfolio_value = initial_capital
            values = [portfolio_value]
            
            for ret in sampled_returns:
                portfolio_value *= (1 + ret)
                values.append(portfolio_value)
            
            final_value = values[-1]
            total_return = (final_value - initial_capital) / initial_capital
            simulation_results.append({
                'final_value': final_value,
                'total_return': total_return,
                'max_drawdown': self._calculate_max_drawdown(values)
            })
        
        # Calculate confidence intervals
        returns = [sim['total_return'] for sim in simulation_results]
        lower_bound = np.percentile(returns, confidence_level * 100)
        upper_bound = np.percentile(returns, (1 - confidence_level) * 100)
        
        return {
            'original_result': self.base_result,
            'simulations_run': num_simulations,
            'confidence_level': confidence_level,
            'return_confidence_interval': (lower_bound, upper_bound),
            'expected_return': np.mean(returns),
            'return_std_dev': np.std(returns),
            'worst_case': min(returns),
            'best_case': max(returns),
            'probability_positive': sum(1 for r in returns if r > 0) / len(returns)
        }

    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown."""
        if len(portfolio_values) <= 1:
            return 0
        
        portfolio_values = np.array(portfolio_values)
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - running_max) / running_max
        return min(0, float(drawdowns.min()))