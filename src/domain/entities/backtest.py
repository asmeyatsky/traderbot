"""
Backtest Domain Entities

Architectural Intent:
- Frozen dataclasses for backtest configuration and results
- StrategyType enum for supported strategies
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class StrategyType(Enum):
    SMA_CROSSOVER = "sma_crossover"
    RSI_MEAN_REVERSION = "rsi_mean_reversion"
    MOMENTUM = "momentum"


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for a backtest run."""
    strategy: StrategyType
    symbol: str
    start_date: Optional[str] = None   # YYYY-MM-DD
    end_date: Optional[str] = None     # YYYY-MM-DD
    initial_capital: float = 10000.0


@dataclass(frozen=True)
class BacktestSummary:
    """Summary of a completed backtest."""
    strategy: str
    symbol: str
    initial_capital: float
    final_value: float
    total_return_pct: float
    annualized_return_pct: float
    volatility: float
    sharpe_ratio: float
    max_drawdown_pct: float
    total_trades: int
    win_rate: float
    profit_factor: float
    trades: List[Dict] = field(default_factory=list)
    equity_curve: List[Dict] = field(default_factory=list)
