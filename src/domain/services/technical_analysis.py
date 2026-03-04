"""
Technical Analysis Domain Service

Architectural Intent:
- Defines the TechnicalIndicators value object and TechnicalAnalysisPort interface
- Pure domain logic for generating signal summaries from indicator values
- No infrastructure dependencies — adapters implement the port
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

from src.domain.value_objects import Symbol


@dataclass(frozen=True)
class TechnicalIndicators:
    """
    Immutable value object containing computed technical indicators for a symbol.

    All fields are Optional because not every indicator can always be computed
    (e.g., insufficient data for a 200-day SMA).
    """
    symbol: str
    rsi_14: Optional[float] = None
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    atr_14: Optional[float] = None
    stoch_k: Optional[float] = None
    stoch_d: Optional[float] = None
    adx: Optional[float] = None
    current_price: Optional[float] = None


def generate_signal_summary(indicators: TechnicalIndicators) -> Dict[str, str]:
    """
    Pure function: derive human-readable signal labels from raw indicator values.

    Returns a dict like {"RSI": "OVERSOLD", "MACD": "BULLISH", ...}.
    """
    signals: Dict[str, str] = {}

    # RSI
    if indicators.rsi_14 is not None:
        if indicators.rsi_14 < 30:
            signals["RSI"] = "OVERSOLD"
        elif indicators.rsi_14 > 70:
            signals["RSI"] = "OVERBOUGHT"
        else:
            signals["RSI"] = "NEUTRAL"

    # MACD
    if indicators.macd_histogram is not None:
        if indicators.macd_histogram > 0:
            signals["MACD"] = "BULLISH"
        elif indicators.macd_histogram < 0:
            signals["MACD"] = "BEARISH"
        else:
            signals["MACD"] = "NEUTRAL"

    # Moving Average trend (price vs SMA 50)
    if indicators.current_price is not None and indicators.sma_50 is not None:
        if indicators.current_price > indicators.sma_50:
            signals["Moving_Average"] = "BULLISH"
        elif indicators.current_price < indicators.sma_50:
            signals["Moving_Average"] = "BEARISH"
        else:
            signals["Moving_Average"] = "NEUTRAL"

    # Bollinger Bands
    if (
        indicators.current_price is not None
        and indicators.bb_upper is not None
        and indicators.bb_lower is not None
    ):
        if indicators.current_price >= indicators.bb_upper:
            signals["Bollinger_Bands"] = "OVERBOUGHT"
        elif indicators.current_price <= indicators.bb_lower:
            signals["Bollinger_Bands"] = "OVERSOLD"
        else:
            signals["Bollinger_Bands"] = "NEUTRAL"

    # Stochastic
    if indicators.stoch_k is not None:
        if indicators.stoch_k < 20:
            signals["Stochastic"] = "OVERSOLD"
        elif indicators.stoch_k > 80:
            signals["Stochastic"] = "OVERBOUGHT"
        else:
            signals["Stochastic"] = "NEUTRAL"

    # ADX (trend strength)
    if indicators.adx is not None:
        if indicators.adx > 25:
            signals["ADX"] = "STRONG_TREND"
        else:
            signals["ADX"] = "WEAK_TREND"

    return signals


class TechnicalAnalysisPort(ABC):
    """Port for computing technical indicators from market data."""

    @abstractmethod
    def compute_indicators(self, symbol: Symbol) -> TechnicalIndicators:
        """Compute all technical indicators for *symbol* using recent OHLCV data."""
        pass
