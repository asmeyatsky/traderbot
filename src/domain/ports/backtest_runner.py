"""
Backtest Runner — port.

Architectural Intent:
- The application use case needs one thing from infrastructure: "given these
  backtest parameters, give me a result." Everything about data fetching,
  strategy implementations, and engine mechanics is an infrastructure detail.
- Lifting this port eliminates the grandfathered `ignore_imports` entry for
  `src.application.use_cases.backtest` (Phase 1 → Phase 4 burn-down).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass
from typing import Any, List, Optional

from src.domain.value_objects import Symbol


@dataclass(frozen=True)
class BacktestRequest:
    """Inputs to a single backtest run."""
    strategy_name: str  # "sma_crossover" | "rsi_mean_reversion" | "momentum"
    symbol: Symbol
    start_date: datetime
    end_date: datetime
    initial_capital: float


@dataclass(frozen=True)
class BacktestResult:
    """Flat result summary. `trades` is capped by infrastructure for size."""
    strategy_name: str
    symbol: str
    initial_capital: float
    final_portfolio_value: float
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    profit_factor: float
    trades: List[Any]


class BacktestRunnerPort(ABC):
    """Runs a backtest against historical data and returns a summary."""

    @abstractmethod
    def run(self, request: BacktestRequest) -> BacktestResult:
        """Execute the backtest. Raise on unrecoverable errors."""
