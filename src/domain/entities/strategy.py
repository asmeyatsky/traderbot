"""
Strategy Domain Entity

Represents a saved trading strategy configuration with backtesting results,
supporting the strategy marketplace and copy trading features.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import List, Optional, Dict, Any


@dataclass(frozen=True)
class SavedStrategy:
    """
    A user's saved strategy configuration.

    Architectural Intent:
    - Represents a reusable, shareable trading strategy
    - Can be made public for the strategy marketplace
    - Tracks performance via linked backtest results
    """
    id: str
    user_id: str
    name: str
    description: str
    strategy_type: str  # sma_crossover, rsi_mean_reversion, momentum
    parameters: Dict[str, Any] = field(default_factory=dict)
    symbol: str = "AAPL"
    is_public: bool = False
    fork_count: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def make_public(self) -> SavedStrategy:
        from dataclasses import replace
        return replace(self, is_public=True, updated_at=datetime.utcnow())

    def make_private(self) -> SavedStrategy:
        from dataclasses import replace
        return replace(self, is_public=False, updated_at=datetime.utcnow())


@dataclass(frozen=True)
class BacktestResult:
    """
    A persisted backtest result linked to a strategy.

    Architectural Intent:
    - Immutable record of strategy performance
    - Used for leaderboard ranking and strategy comparison
    """
    id: str
    strategy_id: str
    user_id: str
    symbol: str
    initial_capital: Decimal
    final_value: Decimal
    total_return_pct: Decimal
    sharpe_ratio: Decimal
    max_drawdown_pct: Decimal
    win_rate: Decimal
    total_trades: int
    volatility: Decimal
    profit_factor: Decimal
    run_at: Optional[datetime] = None


@dataclass(frozen=True)
class StrategyFollow:
    """
    Represents a user following (copy trading) a strategy.

    Architectural Intent:
    - Links a follower to a public strategy
    - Enables copy trading and social features
    """
    id: str
    follower_user_id: str
    strategy_id: str
    created_at: Optional[datetime] = None
