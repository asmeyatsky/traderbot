"""
Pandas-TA Technical Analysis Adapter

Architectural Intent:
- Implements TechnicalAnalysisPort using pandas-ta for real indicator computation
- Fetches OHLCV data via yfinance (infrastructure concern, not domain)
- Returns a frozen TechnicalIndicators value object to the domain layer
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import yfinance as yf

from src.domain.services.technical_analysis import TechnicalAnalysisPort, TechnicalIndicators
from src.domain.value_objects import Symbol

logger = logging.getLogger(__name__)

# pandas-ta may not be installed in every environment; import defensively
try:
    import pandas_ta as ta  # noqa: F401
    _HAS_PANDAS_TA = True
except ImportError:
    _HAS_PANDAS_TA = False
    logger.warning("pandas-ta not installed; technical indicators will be unavailable")


class PandasTATechnicalAnalysisAdapter(TechnicalAnalysisPort):
    """
    Adapter: computes RSI, MACD, SMA, EMA, Bollinger Bands, ATR, Stochastic, ADX
    from Yahoo Finance OHLCV data using pandas-ta.
    """

    def __init__(self, lookback_days: int = 250):
        self._lookback_days = lookback_days

    def _fetch_ohlcv(self, symbol: Symbol) -> Optional[pd.DataFrame]:
        """Fetch daily OHLCV from Yahoo Finance."""
        try:
            ticker = yf.Ticker(str(symbol))
            df = ticker.history(period=f"{self._lookback_days}d", interval="1d")
            if df.empty or len(df) < 30:
                return None
            return df
        except Exception as exc:
            logger.error("Failed to fetch OHLCV for %s: %s", symbol, exc)
            return None

    def compute_indicators(self, symbol: Symbol) -> TechnicalIndicators:
        """Compute all technical indicators for a symbol."""
        df = self._fetch_ohlcv(symbol)
        if df is None or not _HAS_PANDAS_TA:
            return TechnicalIndicators(symbol=str(symbol))

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        current_price = float(close.iloc[-1])

        def _safe_last(series: Optional[pd.Series]) -> Optional[float]:
            if series is None or series.empty:
                return None
            val = series.iloc[-1]
            if pd.isna(val):
                return None
            return round(float(val), 4)

        # RSI
        rsi = ta.rsi(close, length=14)

        # MACD
        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        macd_line = macd_signal = macd_hist = None
        if macd_df is not None and not macd_df.empty:
            macd_line = _safe_last(macd_df.iloc[:, 0])
            macd_hist = _safe_last(macd_df.iloc[:, 1])
            macd_signal = _safe_last(macd_df.iloc[:, 2])

        # SMAs
        sma_20 = _safe_last(ta.sma(close, length=20))
        sma_50 = _safe_last(ta.sma(close, length=50))
        sma_200 = _safe_last(ta.sma(close, length=200)) if len(close) >= 200 else None

        # EMAs
        ema_12 = _safe_last(ta.ema(close, length=12))
        ema_26 = _safe_last(ta.ema(close, length=26))

        # Bollinger Bands
        bb_df = ta.bbands(close, length=20, std=2)
        bb_lower = bb_middle = bb_upper = None
        if bb_df is not None and not bb_df.empty:
            bb_lower = _safe_last(bb_df.iloc[:, 0])
            bb_middle = _safe_last(bb_df.iloc[:, 1])
            bb_upper = _safe_last(bb_df.iloc[:, 2])

        # ATR
        atr = _safe_last(ta.atr(high, low, close, length=14))

        # Stochastic
        stoch_df = ta.stoch(high, low, close)
        stoch_k = stoch_d = None
        if stoch_df is not None and not stoch_df.empty:
            stoch_k = _safe_last(stoch_df.iloc[:, 0])
            stoch_d = _safe_last(stoch_df.iloc[:, 1])

        # ADX
        adx_df = ta.adx(high, low, close, length=14)
        adx_val = None
        if adx_df is not None and not adx_df.empty:
            adx_val = _safe_last(adx_df.iloc[:, 0])

        return TechnicalIndicators(
            symbol=str(symbol),
            rsi_14=_safe_last(rsi),
            macd_line=macd_line,
            macd_signal=macd_signal,
            macd_histogram=macd_hist,
            sma_20=sma_20,
            sma_50=sma_50,
            sma_200=sma_200,
            ema_12=ema_12,
            ema_26=ema_26,
            bb_upper=bb_upper,
            bb_middle=bb_middle,
            bb_lower=bb_lower,
            atr_14=atr,
            stoch_k=stoch_k,
            stoch_d=stoch_d,
            adx=adx_val,
            current_price=current_price,
        )
