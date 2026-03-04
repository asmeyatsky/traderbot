"""
Technical Analysis Adapter (using ``ta`` library)

Architectural Intent:
- Implements TechnicalAnalysisPort using the ``ta`` library for real indicator computation
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

try:
    import ta as ta_lib  # noqa: F401
    _HAS_TA = True
except ImportError:
    _HAS_TA = False
    logger.warning("ta library not installed; technical indicators will be unavailable")


class PandasTATechnicalAnalysisAdapter(TechnicalAnalysisPort):
    """
    Adapter: computes RSI, MACD, SMA, EMA, Bollinger Bands, ATR, Stochastic, ADX
    from Yahoo Finance OHLCV data using the ``ta`` library.
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
        if df is None or not _HAS_TA:
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
        rsi = ta_lib.momentum.RSIIndicator(close, window=14).rsi()

        # MACD
        macd_ind = ta_lib.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        macd_line = _safe_last(macd_ind.macd())
        macd_signal = _safe_last(macd_ind.macd_signal())
        macd_hist = _safe_last(macd_ind.macd_diff())

        # SMAs
        sma_20 = _safe_last(ta_lib.trend.SMAIndicator(close, window=20).sma_indicator())
        sma_50 = _safe_last(ta_lib.trend.SMAIndicator(close, window=50).sma_indicator())
        sma_200 = _safe_last(ta_lib.trend.SMAIndicator(close, window=200).sma_indicator()) if len(close) >= 200 else None

        # EMAs
        ema_12 = _safe_last(ta_lib.trend.EMAIndicator(close, window=12).ema_indicator())
        ema_26 = _safe_last(ta_lib.trend.EMAIndicator(close, window=26).ema_indicator())

        # Bollinger Bands
        bb = ta_lib.volatility.BollingerBands(close, window=20, window_dev=2)
        bb_lower = _safe_last(bb.bollinger_lband())
        bb_middle = _safe_last(bb.bollinger_mavg())
        bb_upper = _safe_last(bb.bollinger_hband())

        # ATR
        atr = _safe_last(ta_lib.volatility.AverageTrueRange(high, low, close, window=14).average_true_range())

        # Stochastic
        stoch = ta_lib.momentum.StochasticOscillator(high, low, close)
        stoch_k = _safe_last(stoch.stoch())
        stoch_d = _safe_last(stoch.stoch_signal())

        # ADX
        adx_ind = ta_lib.trend.ADXIndicator(high, low, close, window=14)
        adx_val = _safe_last(adx_ind.adx())

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
