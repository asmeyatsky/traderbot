"""
Tests for Backtest Domain Entities and RunBacktestUseCase

Tests cover:
- BacktestConfig and BacktestSummary frozen dataclasses
- StrategyType enum values
- RSIMeanReversionStrategy signal generation
- MomentumStrategy signal generation
"""
import pytest
from datetime import datetime

from src.domain.entities.backtest import BacktestConfig, BacktestSummary, StrategyType

try:
    import pandas as pd
    import numpy as np
    from src.infrastructure.data_processing.backtesting_engine import (
        RSIMeanReversionStrategy,
        MomentumStrategy,
        SMACrossoverStrategy,
    )
    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False


class TestBacktestEntities:
    def test_strategy_type_enum(self):
        assert StrategyType.SMA_CROSSOVER.value == "sma_crossover"
        assert StrategyType.RSI_MEAN_REVERSION.value == "rsi_mean_reversion"
        assert StrategyType.MOMENTUM.value == "momentum"

    def test_backtest_config_frozen(self):
        config = BacktestConfig(strategy=StrategyType.SMA_CROSSOVER, symbol="AAPL")
        with pytest.raises(AttributeError):
            config.symbol = "MSFT"  # type: ignore

    def test_backtest_config_defaults(self):
        config = BacktestConfig(strategy=StrategyType.MOMENTUM, symbol="TSLA")
        assert config.initial_capital == 10000.0
        assert config.start_date is None

    def test_backtest_summary_frozen(self):
        summary = BacktestSummary(
            strategy="SMA", symbol="AAPL", initial_capital=10000,
            final_value=11000, total_return_pct=10.0, annualized_return_pct=10.0,
            volatility=0.15, sharpe_ratio=1.2, max_drawdown_pct=-5.0,
            total_trades=20, win_rate=0.6, profit_factor=1.5,
        )
        with pytest.raises(AttributeError):
            summary.final_value = 12000  # type: ignore


def _make_ohlcv(n=100):
    """Generate synthetic OHLCV data."""
    dates = pd.date_range(start="2024-01-01", periods=n, freq="B")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "Open": close - 0.5,
        "High": close + 1.0,
        "Low": close - 1.0,
        "Close": close,
        "Volume": np.random.randint(100_000, 1_000_000, n),
    }, index=dates)


@pytest.mark.skipif(not _HAS_DEPS, reason="pandas/numpy not installed")
class TestRSIMeanReversionStrategy:
    def test_generates_signals(self):
        strategy = RSIMeanReversionStrategy(rsi_period=14)
        data = _make_ohlcv(100)
        signals = strategy.generate_signals(data, {})
        # Should produce at least some signals on random data
        assert isinstance(signals, list)
        for ts, signal_type, info in signals:
            assert signal_type in ("BUY", "SELL")
            assert "rsi" in info

    def test_strategy_name(self):
        strategy = RSIMeanReversionStrategy()
        assert "RSI" in strategy.get_strategy_name()


@pytest.mark.skipif(not _HAS_DEPS, reason="pandas/numpy not installed")
class TestMomentumStrategy:
    def test_generates_signals(self):
        strategy = MomentumStrategy(lookback=20)
        data = _make_ohlcv(100)
        signals = strategy.generate_signals(data, {})
        assert isinstance(signals, list)
        for ts, signal_type, info in signals:
            assert signal_type in ("BUY", "SELL")

    def test_strategy_name(self):
        strategy = MomentumStrategy()
        assert "Momentum" in strategy.get_strategy_name()

    def test_insufficient_data(self):
        strategy = MomentumStrategy(lookback=20)
        data = _make_ohlcv(10)
        signals = strategy.generate_signals(data, {})
        assert signals == []


@pytest.mark.skipif(not _HAS_DEPS, reason="pandas/numpy not installed")
class TestSMACrossoverStrategy:
    def test_strategy_name(self):
        strategy = SMACrossoverStrategy(short_window=20, long_window=50)
        assert "SMA" in strategy.get_strategy_name()
