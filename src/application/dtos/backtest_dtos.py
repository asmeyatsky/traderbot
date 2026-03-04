"""
Backtest DTOs

Pydantic models for API request/response validation.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class BacktestRequest(BaseModel):
    strategy: str = "sma_crossover"
    symbol: str = "AAPL"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    initial_capital: float = 10000.0


class BacktestTradeDTO(BaseModel):
    date: Optional[str] = None
    symbol: str
    action: str
    quantity: int
    price: float
    commission: float = 0.0
    reason: str = ""
    value: float = 0.0


class BacktestResponse(BaseModel):
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
    trades: List[dict] = []
    error: Optional[str] = None


class StrategyInfo(BaseModel):
    name: str
    label: str
    description: str
