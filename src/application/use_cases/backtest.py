"""
Run Backtest Use Case

Architectural Intent:
- Orchestrates strategy selection, engine execution, and result formatting
- Single use case class, no business logic in infrastructure
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict

from src.domain.entities.backtest import BacktestConfig, BacktestSummary, StrategyType
from src.domain.value_objects import Symbol
from src.infrastructure.data_processing.backtesting_engine import (
    BacktestConfiguration,
    BacktestingEngine,
    MomentumStrategy,
    RSIMeanReversionStrategy,
    SMACrossoverStrategy,
    YahooFinanceDataProvider,
)

logger = logging.getLogger(__name__)


class RunBacktestUseCase:
    """Execute a backtest and return a JSON-serialisable summary."""

    def __init__(self):
        self._data_provider = YahooFinanceDataProvider()

    def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Entry point called by the chat tool. Accepts a dict with:
        strategy, symbol, start_date?, end_date?, initial_capital?
        """
        strategy_name = args.get("strategy", "sma_crossover")
        symbol_str = args.get("symbol", "AAPL").upper()
        initial_capital = args.get("initial_capital", 10000.0)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        if args.get("start_date"):
            try:
                start_date = datetime.strptime(args["start_date"], "%Y-%m-%d")
            except ValueError:
                pass
        if args.get("end_date"):
            try:
                end_date = datetime.strptime(args["end_date"], "%Y-%m-%d")
            except ValueError:
                pass

        symbol = Symbol(symbol_str)

        # Select strategy
        if strategy_name == "rsi_mean_reversion":
            strategy = RSIMeanReversionStrategy()
        elif strategy_name == "momentum":
            strategy = MomentumStrategy()
        else:
            strategy = SMACrossoverStrategy()

        config = BacktestConfiguration(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            symbols=[symbol],
        )

        try:
            engine = BacktestingEngine(self._data_provider, strategy, config)
            result = engine.run_backtest()

            return {
                "strategy": strategy.get_strategy_name(),
                "symbol": symbol_str,
                "initial_capital": initial_capital,
                "final_value": round(result.final_portfolio_value, 2),
                "total_return_pct": round(result.total_return * 100, 2),
                "annualized_return_pct": round(result.annualized_return * 100, 2),
                "volatility": round(result.volatility * 100, 2),
                "sharpe_ratio": round(result.sharpe_ratio, 2),
                "max_drawdown_pct": round(result.max_drawdown * 100, 2),
                "total_trades": result.total_trades,
                "win_rate": round(result.win_rate * 100, 1),
                "profit_factor": round(result.profit_factor, 2) if result.profit_factor != float('inf') else 999.0,
                "trades": result.trades[:50],  # Cap for JSON size
            }
        except Exception as exc:
            logger.error("Backtest failed for %s: %s", symbol_str, exc)
            return {"error": str(exc), "strategy": strategy_name, "symbol": symbol_str}
