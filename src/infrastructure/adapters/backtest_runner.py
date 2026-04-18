"""
Backtest runner adapter — wraps the in-house BacktestingEngine.

Architectural Intent:
- Implements `BacktestRunnerPort` so the application layer never touches
  YahooFinanceDataProvider or the strategy classes directly.
- Strategy selection lives here — a string name maps to a concrete strategy
  instance. Adding a new strategy means editing this file plus the enum in
  the port's accepted names.
"""
from __future__ import annotations

import logging

from src.domain.ports.backtest_runner import (
    BacktestRequest,
    BacktestResult,
    BacktestRunnerPort,
)
from src.infrastructure.data_processing.backtesting_engine import (
    BacktestConfiguration,
    BacktestingEngine,
    MomentumStrategy,
    RSIMeanReversionStrategy,
    SMACrossoverStrategy,
    YahooFinanceDataProvider,
)

logger = logging.getLogger(__name__)


def _strategy_for(name: str):
    if name == "rsi_mean_reversion":
        return RSIMeanReversionStrategy()
    if name == "momentum":
        return MomentumStrategy()
    return SMACrossoverStrategy()  # default + "sma_crossover"


class YahooBacktestRunner(BacktestRunnerPort):
    """Default adapter — pulls historical data from Yahoo Finance."""

    def __init__(self) -> None:
        self._data_provider = YahooFinanceDataProvider()

    def run(self, request: BacktestRequest) -> BacktestResult:
        strategy = _strategy_for(request.strategy_name)
        config = BacktestConfiguration(
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            symbols=[request.symbol],
        )
        engine = BacktestingEngine(self._data_provider, strategy, config)
        engine_result = engine.run_backtest()
        return BacktestResult(
            strategy_name=strategy.get_strategy_name(),
            symbol=str(request.symbol),
            initial_capital=request.initial_capital,
            final_portfolio_value=engine_result.final_portfolio_value,
            total_return=engine_result.total_return,
            annualized_return=engine_result.annualized_return,
            volatility=engine_result.volatility,
            sharpe_ratio=engine_result.sharpe_ratio,
            max_drawdown=engine_result.max_drawdown,
            total_trades=engine_result.total_trades,
            win_rate=engine_result.win_rate,
            profit_factor=engine_result.profit_factor,
            trades=engine_result.trades[:50],  # cap for JSON size
        )
