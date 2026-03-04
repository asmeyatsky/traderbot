"""
Tests for Technical Analysis Domain Service and Signal Generation

Tests cover:
- TechnicalIndicators frozen dataclass
- generate_signal_summary() pure function with various indicator ranges
- TechnicalAnalysisPort interface contract
"""
import pytest

from src.domain.services.technical_analysis import (
    TechnicalIndicators,
    generate_signal_summary,
)


class TestTechnicalIndicators:
    def test_frozen_dataclass(self):
        indicators = TechnicalIndicators(symbol="AAPL", rsi_14=45.0)
        with pytest.raises(AttributeError):
            indicators.rsi_14 = 50.0  # type: ignore

    def test_default_none_fields(self):
        indicators = TechnicalIndicators(symbol="AAPL")
        assert indicators.rsi_14 is None
        assert indicators.macd_line is None
        assert indicators.current_price is None


class TestGenerateSignalSummary:
    def test_oversold_rsi(self):
        indicators = TechnicalIndicators(symbol="AAPL", rsi_14=25.0)
        signals = generate_signal_summary(indicators)
        assert signals["RSI"] == "OVERSOLD"

    def test_overbought_rsi(self):
        indicators = TechnicalIndicators(symbol="AAPL", rsi_14=75.0)
        signals = generate_signal_summary(indicators)
        assert signals["RSI"] == "OVERBOUGHT"

    def test_neutral_rsi(self):
        indicators = TechnicalIndicators(symbol="AAPL", rsi_14=50.0)
        signals = generate_signal_summary(indicators)
        assert signals["RSI"] == "NEUTRAL"

    def test_bullish_macd(self):
        indicators = TechnicalIndicators(symbol="AAPL", macd_histogram=1.5)
        signals = generate_signal_summary(indicators)
        assert signals["MACD"] == "BULLISH"

    def test_bearish_macd(self):
        indicators = TechnicalIndicators(symbol="AAPL", macd_histogram=-0.5)
        signals = generate_signal_summary(indicators)
        assert signals["MACD"] == "BEARISH"

    def test_bullish_ma(self):
        indicators = TechnicalIndicators(symbol="AAPL", current_price=150.0, sma_50=140.0)
        signals = generate_signal_summary(indicators)
        assert signals["Moving_Average"] == "BULLISH"

    def test_bearish_ma(self):
        indicators = TechnicalIndicators(symbol="AAPL", current_price=130.0, sma_50=140.0)
        signals = generate_signal_summary(indicators)
        assert signals["Moving_Average"] == "BEARISH"

    def test_bollinger_overbought(self):
        indicators = TechnicalIndicators(
            symbol="AAPL", current_price=160.0, bb_upper=155.0, bb_lower=135.0
        )
        signals = generate_signal_summary(indicators)
        assert signals["Bollinger_Bands"] == "OVERBOUGHT"

    def test_bollinger_oversold(self):
        indicators = TechnicalIndicators(
            symbol="AAPL", current_price=130.0, bb_upper=155.0, bb_lower=135.0
        )
        signals = generate_signal_summary(indicators)
        assert signals["Bollinger_Bands"] == "OVERSOLD"

    def test_stochastic_oversold(self):
        indicators = TechnicalIndicators(symbol="AAPL", stoch_k=15.0)
        signals = generate_signal_summary(indicators)
        assert signals["Stochastic"] == "OVERSOLD"

    def test_adx_strong_trend(self):
        indicators = TechnicalIndicators(symbol="AAPL", adx=30.0)
        signals = generate_signal_summary(indicators)
        assert signals["ADX"] == "STRONG_TREND"

    def test_adx_weak_trend(self):
        indicators = TechnicalIndicators(symbol="AAPL", adx=15.0)
        signals = generate_signal_summary(indicators)
        assert signals["ADX"] == "WEAK_TREND"

    def test_empty_indicators_returns_empty_signals(self):
        indicators = TechnicalIndicators(symbol="AAPL")
        signals = generate_signal_summary(indicators)
        assert signals == {}

    def test_full_indicators(self):
        indicators = TechnicalIndicators(
            symbol="AAPL",
            rsi_14=45.0,
            macd_histogram=0.5,
            current_price=150.0,
            sma_50=145.0,
            bb_upper=160.0,
            bb_lower=140.0,
            stoch_k=50.0,
            adx=30.0,
        )
        signals = generate_signal_summary(indicators)
        assert len(signals) >= 6
        assert signals["RSI"] == "NEUTRAL"
        assert signals["MACD"] == "BULLISH"
        assert signals["Moving_Average"] == "BULLISH"
        assert signals["Bollinger_Bands"] == "NEUTRAL"
        assert signals["Stochastic"] == "NEUTRAL"
        assert signals["ADX"] == "STRONG_TREND"
