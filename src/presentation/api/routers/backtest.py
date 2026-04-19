"""
Backtest API Router

Endpoints for running backtests and listing available strategies.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from src.application.dtos.backtest_dtos import BacktestRequest
from src.application.use_cases.backtest import RunBacktestUseCase
from src.infrastructure.security import get_current_user

router = APIRouter(prefix="/api/v1/backtest", tags=["backtest"])


def _get_backtest_use_case() -> RunBacktestUseCase:
    """Lazy resolve from DI — avoids constructing the use case at module
    import time (Phase 4 split the use case from its infrastructure runner
    port, so it needs a wired container)."""
    from src.infrastructure.di_container import container
    return container.use_cases.backtest_use_case()


@router.post("/run")
async def run_backtest(
    request: BacktestRequest,
    user_id: str = Depends(get_current_user),
    use_case: RunBacktestUseCase = Depends(_get_backtest_use_case),
):
    """Run a backtest with the specified strategy and parameters."""
    return use_case.run(request.model_dump())


@router.get("/strategies")
async def list_strategies(user_id: str = Depends(get_current_user)):
    """List available backtesting strategies."""
    return {
        "strategies": [
            {
                "name": "sma_crossover",
                "label": "SMA Crossover",
                "description": "Buy on golden cross (SMA 20 > SMA 50), sell on death cross",
            },
            {
                "name": "rsi_mean_reversion",
                "label": "RSI Mean Reversion",
                "description": "Buy when RSI < 30 (oversold), sell when RSI > 70 (overbought)",
            },
            {
                "name": "momentum",
                "label": "Momentum Breakout",
                "description": "Buy when price breaks 20-day high, sell when price breaks 20-day low",
            },
        ]
    }
