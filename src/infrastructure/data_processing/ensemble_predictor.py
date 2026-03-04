"""
Ensemble Predictor Service

Architectural Intent:
- Builds features from technical indicators (via pandas-ta)
- Trains RandomForest + GradientBoosting on historical data
- Predicts next-day direction with SHAP TreeExplainer for feature attribution
- Returns a PredictionResult frozen dataclass to the domain layer
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.domain.entities.prediction import PredictionResult
from src.domain.value_objects import Symbol

logger = logging.getLogger(__name__)

try:
    import pandas_ta as ta
    _HAS_TA = True
except ImportError:
    _HAS_TA = False

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute feature columns from OHLCV data using pandas-ta."""
    if not _HAS_TA:
        return pd.DataFrame()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    feats = pd.DataFrame(index=df.index)
    feats["RSI_14"] = ta.rsi(close, length=14)

    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None:
        feats["MACD_hist"] = macd.iloc[:, 1]

    feats["SMA_20"] = ta.sma(close, length=20)
    feats["SMA_50"] = ta.sma(close, length=50)
    feats["EMA_12"] = ta.ema(close, length=12)

    bb = ta.bbands(close, length=20, std=2)
    if bb is not None:
        feats["BB_width"] = bb.iloc[:, 2] - bb.iloc[:, 0]

    feats["ATR_14"] = ta.atr(high, low, close, length=14)

    stoch = ta.stoch(high, low, close)
    if stoch is not None:
        feats["Stoch_K"] = stoch.iloc[:, 0]

    adx = ta.adx(high, low, close, length=14)
    if adx is not None:
        feats["ADX"] = adx.iloc[:, 0]

    # Price-derived
    feats["Return_1d"] = close.pct_change(1)
    feats["Return_5d"] = close.pct_change(5)
    feats["Volatility_10d"] = close.pct_change().rolling(10).std()

    return feats


class EnsemblePredictorService:
    """Trains RF + GB on recent data and predicts next-day direction with SHAP explanations."""

    def __init__(self, lookback_days: int = 500):
        self._lookback = lookback_days

    def predict(self, symbol: Symbol) -> PredictionResult:
        """Run the full pipeline: fetch data -> build features -> train -> predict -> explain."""
        symbol_str = str(symbol)
        try:
            df = yf.Ticker(symbol_str).history(period=f"{self._lookback}d", interval="1d")
        except Exception as exc:
            logger.error("Failed to fetch data for %s: %s", symbol_str, exc)
            return self._empty_result(symbol_str)

        if df is None or len(df) < 100:
            return self._empty_result(symbol_str)

        features = _build_features(df)
        if features.empty:
            return self._empty_result(symbol_str)

        # Target: next-day return sign  (1 = UP, 0 = DOWN/FLAT)
        close = df["Close"]
        target = (close.shift(-1) > close).astype(int)

        combined = features.copy()
        combined["target"] = target
        combined.dropna(inplace=True)

        if len(combined) < 60:
            return self._empty_result(symbol_str)

        X = combined.drop(columns=["target"])
        y = combined["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # Train models
        rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
        rf.fit(X_train, y_train)
        gb.fit(X_train, y_train)

        # Predict on latest row
        latest = X.iloc[[-1]]
        rf_pred = int(rf.predict(latest)[0])
        gb_pred = int(gb.predict(latest)[0])
        rf_prob = float(rf.predict_proba(latest)[0].max())
        gb_prob = float(gb.predict_proba(latest)[0].max())

        # Ensemble vote
        votes = {"RandomForest": "UP" if rf_pred == 1 else "DOWN",
                 "GradientBoosting": "UP" if gb_pred == 1 else "DOWN"}
        avg_prob = (rf_prob + gb_prob) / 2
        direction = "UP" if (rf_pred + gb_pred) >= 1 else "DOWN"

        # SHAP explanations
        top_features = self._explain(rf, latest, X.columns.tolist())

        # Estimate predicted change %
        predicted_change = round(float(close.pct_change().tail(20).mean()) * 100, 2)
        if direction == "DOWN":
            predicted_change = -abs(predicted_change)

        return PredictionResult(
            symbol=symbol_str,
            direction=direction,
            confidence=round(avg_prob, 3),
            predicted_change_pct=predicted_change,
            model_votes=votes,
            top_features=top_features,
        )

    def _explain(self, model, latest: pd.DataFrame, feature_names: List[str]) -> List[Dict[str, Any]]:
        """Use SHAP TreeExplainer to get top feature attributions."""
        if not _HAS_SHAP:
            return []
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(latest)
            # shap_values may be a list (one per class); take class 1 (UP)
            if isinstance(shap_values, list):
                vals = shap_values[1][0]
            else:
                vals = shap_values[0]

            pairs = list(zip(feature_names, vals))
            pairs.sort(key=lambda p: abs(p[1]), reverse=True)

            return [
                {
                    "feature": name,
                    "importance": round(abs(float(val)), 4),
                    "direction": "positive" if val > 0 else "negative",
                }
                for name, val in pairs[:5]
            ]
        except Exception as exc:
            logger.warning("SHAP explanation failed: %s", exc)
            return []

    def _empty_result(self, symbol: str) -> PredictionResult:
        return PredictionResult(
            symbol=symbol,
            direction="FLAT",
            confidence=0.0,
            predicted_change_pct=0.0,
            model_votes={},
            top_features=[],
        )
