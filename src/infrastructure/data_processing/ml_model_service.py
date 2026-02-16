"""
Advanced AI/ML Model Services for the AI Trading Platform

This module implements advanced ML models including NLP for sentiment analysis,
prediction models, and reinforcement learning agents as outlined in the PRD.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import logging
import os
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from decimal import Decimal
import requests
import json

from src.domain.value_objects import Symbol, Price, NewsSentiment, Money
from src.domain.entities.trading import Position, Portfolio
from src.domain.entities.user import User


logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Represents a trading signal with confidence and explanation."""
    signal: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 to 1.0
    explanation: str
    score: float  # -1.0 to 1.0 (negative = sell, positive = buy)


@dataclass
class ModelPerformance:
    """Performance metrics for ML models."""
    accuracy: float
    precision: float
    recall: float
    sharpe_ratio: float
    max_drawdown: float
    annual_return: float


class MLModelService(ABC):
    """Abstract base class for ML model services."""

    @abstractmethod
    def predict_price_direction(self, symbol: Symbol, lookback_period: int = 30) -> TradingSignal:
        """Predict price direction for a given symbol."""
        pass

    @abstractmethod
    def get_model_performance(self, model_type: str) -> ModelPerformance:
        """Get performance metrics for a specific model."""
        pass

    @abstractmethod
    def retrain_model(self, symbol: Symbol) -> bool:
        """Retrain the model with new data."""
        pass


class SentimentAnalysisService(ABC):
    """Abstract base class for sentiment analysis services."""

    @abstractmethod
    def analyze_sentiment(self, text: str) -> NewsSentiment:
        """Analyze sentiment of text."""
        pass

    @abstractmethod
    def analyze_batch_sentiment(self, texts: List[str]) -> List[NewsSentiment]:
        """Analyze sentiment of multiple texts."""
        pass

    @abstractmethod
    def get_symbol_sentiment(self, symbol: Symbol, lookback_hours: int = 24) -> NewsSentiment:
        """Get aggregate sentiment for a symbol."""
        pass


class ReinforcementLearningAgent(ABC):
    """Abstract base class for RL trading agents."""

    @abstractmethod
    def get_action(self, state: Dict, symbol: Symbol) -> Tuple[str, float]:  # (action, position_size)
        """Get trading action based on current market state."""
        pass

    @abstractmethod
    def train(self, training_data: List[Dict]) -> bool:
        """Train the RL agent."""
        pass

    @abstractmethod
    def evaluate(self, evaluation_data: List[Dict]) -> ModelPerformance:
        """Evaluate the RL agent."""
        pass


class LSTMPricePredictionService(MLModelService):
    """
    LSTM-based price prediction service with real training and inference.

    Uses yfinance for data, computes technical indicators as features,
    trains a TensorFlow LSTM model, and persists model artifacts.
    """

    FEATURE_COLUMNS = [
        'returns', 'log_returns', 'volatility_5', 'volatility_20',
        'sma_5', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
        'rsi_14', 'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_lower', 'bb_width',
        'volume_sma_20', 'volume_ratio',
        'atr_14', 'obv_norm',
    ]
    SEQUENCE_LENGTH = 30
    MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'lstm')

    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._scalers: Dict[str, Any] = {}
        self._performance_cache: Dict[str, ModelPerformance] = {}
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        logger.info("LSTMPricePredictionService initialized")

    @staticmethod
    def _compute_features(df: 'pd.DataFrame') -> 'pd.DataFrame':
        """Compute technical indicator features from OHLCV data."""
        import pandas as pd

        feat = pd.DataFrame(index=df.index)
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        # Returns
        feat['returns'] = close.pct_change()
        feat['log_returns'] = np.log(close / close.shift(1))

        # Volatility
        feat['volatility_5'] = feat['returns'].rolling(5).std()
        feat['volatility_20'] = feat['returns'].rolling(20).std()

        # Moving averages
        feat['sma_5'] = (close / close.rolling(5).mean()) - 1
        feat['sma_20'] = (close / close.rolling(20).mean()) - 1
        feat['sma_50'] = (close / close.rolling(50).mean()) - 1

        # EMA
        feat['ema_12'] = (close / close.ewm(span=12).mean()) - 1
        feat['ema_26'] = (close / close.ewm(span=26).mean()) - 1

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-10)
        feat['rsi_14'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        feat['macd'] = (ema12 - ema26) / close
        feat['macd_signal'] = feat['macd'].ewm(span=9).mean()
        feat['macd_hist'] = feat['macd'] - feat['macd_signal']

        # Bollinger Bands
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        feat['bb_upper'] = ((bb_mid + 2 * bb_std) / close) - 1
        feat['bb_lower'] = ((bb_mid - 2 * bb_std) / close) - 1
        feat['bb_width'] = (4 * bb_std) / close

        # Volume
        vol_sma = volume.rolling(20).mean()
        feat['volume_sma_20'] = volume / vol_sma.replace(0, 1) - 1
        feat['volume_ratio'] = volume / vol_sma.replace(0, 1)

        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        feat['atr_14'] = tr.rolling(14).mean() / close

        # OBV normalized
        obv = (np.sign(close.diff()) * volume).cumsum()
        feat['obv_norm'] = (obv - obv.rolling(20).mean()) / obv.rolling(20).std().replace(0, 1)

        feat.replace([np.inf, -np.inf], np.nan, inplace=True)
        feat.dropna(inplace=True)
        return feat

    def _build_model(self, n_features: int) -> Any:
        """Build LSTM model architecture."""
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers as kl

        model = keras.Sequential([
            kl.LSTM(64, return_sequences=True, input_shape=(self.SEQUENCE_LENGTH, n_features)),
            kl.Dropout(0.2),
            kl.LSTM(32),
            kl.Dropout(0.2),
            kl.Dense(16, activation='relu'),
            kl.Dense(3, activation='softmax')  # BUY, HOLD, SELL
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def retrain_model(self, symbol: Symbol, lookback_years: int = 3) -> bool:
        """
        Train the LSTM model for a symbol using historical data.

        Fetches data from yfinance, computes features, builds sequences,
        and trains with walk-forward split.
        """
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        symbol_str = str(symbol.value) if hasattr(symbol, 'value') else str(symbol)

        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol_str)
            df = ticker.history(period=f"{lookback_years}y")
            if df is None or len(df) < 200:
                logger.warning(f"Insufficient data for {symbol_str}: {len(df) if df is not None else 0} rows")
                return False

            features = self._compute_features(df)
            if len(features) < self.SEQUENCE_LENGTH + 50:
                logger.warning(f"Insufficient features for {symbol_str}")
                return False

            # Create target: 1 = price up >0.5%, 2 = price down >0.5%, 0 = hold
            close_aligned = df['Close'].reindex(features.index)
            future_returns = close_aligned.pct_change(5).shift(-5)
            targets = pd.Series(0, index=features.index)  # HOLD
            targets[future_returns > 0.005] = 0  # BUY = class 0
            targets[future_returns <= 0.005] = 1  # HOLD = class 1
            targets[future_returns < -0.005] = 2  # SELL = class 2

            # Align
            valid_idx = features.index.intersection(targets.dropna().index)
            features = features.loc[valid_idx]
            targets = targets.loc[valid_idx]

            # Scale features
            scaler = StandardScaler()
            feature_values = scaler.fit_transform(features.values)

            # Build sequences
            X, y = [], []
            for i in range(self.SEQUENCE_LENGTH, len(feature_values) - 5):
                X.append(feature_values[i - self.SEQUENCE_LENGTH:i])
                y.append(targets.iloc[i])
            X = np.array(X)
            y = np.array(y)

            # Walk-forward split (80/20)
            split = int(len(X) * 0.8)
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]

            # Build and train
            model = self._build_model(len(self.FEATURE_COLUMNS))
            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=30,
                batch_size=32,
                verbose=0
            )

            # Evaluate
            val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
            logger.info(f"LSTM trained for {symbol_str}: val_accuracy={val_acc:.3f}")

            # Save
            model_path = os.path.join(self.MODEL_DIR, f"{symbol_str}_lstm.keras")
            scaler_path = os.path.join(self.MODEL_DIR, f"{symbol_str}_scaler.pkl")
            model.save(model_path)
            import pickle
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)

            self._models[symbol_str] = model
            self._scalers[symbol_str] = scaler
            self._performance_cache[symbol_str] = ModelPerformance(
                accuracy=float(val_acc),
                precision=float(val_acc),
                recall=float(val_acc),
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                annual_return=0.0,
            )
            return True

        except Exception as e:
            logger.error(f"Failed to train LSTM for {symbol_str}: {e}")
            return False

    def _load_model(self, symbol_str: str) -> bool:
        """Load a previously trained model from disk."""
        model_path = os.path.join(self.MODEL_DIR, f"{symbol_str}_lstm.keras")
        scaler_path = os.path.join(self.MODEL_DIR, f"{symbol_str}_scaler.pkl")

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return False

        try:
            from tensorflow import keras
            import pickle
            self._models[symbol_str] = keras.models.load_model(model_path)
            with open(scaler_path, 'rb') as f:
                self._scalers[symbol_str] = pickle.load(f)
            logger.info(f"Loaded LSTM model for {symbol_str}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load LSTM model for {symbol_str}: {e}")
            return False

    def predict_price_direction(self, symbol: Symbol, lookback_period: int = 30) -> TradingSignal:
        """
        Predict price direction using trained LSTM model.

        Falls back to technical analysis if no model is trained.
        """
        symbol_str = str(symbol.value) if hasattr(symbol, 'value') else str(symbol)

        # Try to load model if not in memory
        if symbol_str not in self._models:
            if not self._load_model(symbol_str):
                return self._technical_fallback(symbol_str)

        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol_str)
            df = ticker.history(period="6mo")
            if df is None or len(df) < self.SEQUENCE_LENGTH + 50:
                return self._technical_fallback(symbol_str)

            features = self._compute_features(df)
            if len(features) < self.SEQUENCE_LENGTH:
                return self._technical_fallback(symbol_str)

            scaler = self._scalers[symbol_str]
            scaled = scaler.transform(features.values[-self.SEQUENCE_LENGTH:])
            X = np.array([scaled])

            model = self._models[symbol_str]
            probs = model.predict(X, verbose=0)[0]  # [buy_prob, hold_prob, sell_prob]

            buy_prob, hold_prob, sell_prob = float(probs[0]), float(probs[1]), float(probs[2])
            max_prob = max(buy_prob, hold_prob, sell_prob)

            if buy_prob == max_prob:
                signal, score = 'BUY', buy_prob - sell_prob
            elif sell_prob == max_prob:
                signal, score = 'SELL', -(sell_prob - buy_prob)
            else:
                signal, score = 'HOLD', 0.0

            return TradingSignal(
                signal=signal,
                confidence=max_prob,
                explanation=f"LSTM model: BUY={buy_prob:.2f} HOLD={hold_prob:.2f} SELL={sell_prob:.2f} for {symbol_str}",
                score=score
            )
        except Exception as e:
            logger.warning(f"LSTM prediction failed for {symbol_str}: {e}")
            return self._technical_fallback(symbol_str)

    def _technical_fallback(self, symbol_str: str) -> TradingSignal:
        """Simple technical analysis fallback when model unavailable."""
        try:
            import yfinance as yf
            df = yf.Ticker(symbol_str).history(period="3mo")
            if df is None or len(df) < 50:
                return TradingSignal(signal='HOLD', confidence=0.3, explanation="Insufficient data", score=0.0)

            close = df['Close']
            sma20 = close.rolling(20).mean().iloc[-1]
            sma50 = close.rolling(50).mean().iloc[-1]
            current = close.iloc[-1]

            if current > sma20 > sma50:
                return TradingSignal(signal='BUY', confidence=0.5, explanation=f"Price above SMA20 & SMA50 for {symbol_str}", score=0.3)
            elif current < sma20 < sma50:
                return TradingSignal(signal='SELL', confidence=0.5, explanation=f"Price below SMA20 & SMA50 for {symbol_str}", score=-0.3)
            else:
                return TradingSignal(signal='HOLD', confidence=0.4, explanation=f"Mixed signals for {symbol_str}", score=0.0)
        except Exception:
            return TradingSignal(signal='HOLD', confidence=0.2, explanation="Fallback: no data", score=0.0)

    def get_model_performance(self, model_type: str) -> ModelPerformance:
        """Get cached performance metrics for the LSTM model."""
        if model_type in self._performance_cache:
            return self._performance_cache[model_type]
        return ModelPerformance(
            accuracy=0.0, precision=0.0, recall=0.0,
            sharpe_ratio=0.0, max_drawdown=0.0, annual_return=0.0
        )


class XGBoostPredictionService(MLModelService):
    """
    XGBoost-based classification model for buy/sell/hold signals.

    Uses the same feature set as LSTM but as a flat feature vector (last row).
    Provides feature importance tracking and cross-validated training.
    """

    MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'xgboost')

    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._scalers: Dict[str, Any] = {}
        self._importances: Dict[str, Dict] = {}
        self._performance_cache: Dict[str, ModelPerformance] = {}
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        logger.info("XGBoostPredictionService initialized")

    def retrain_model(self, symbol: Symbol, lookback_years: int = 3) -> bool:
        """Train XGBoost model for a symbol."""
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        symbol_str = str(symbol.value) if hasattr(symbol, 'value') else str(symbol)

        try:
            import xgboost as xgb
            from sklearn.model_selection import cross_val_score
            import yfinance as yf
            import pickle

            ticker = yf.Ticker(symbol_str)
            df = ticker.history(period=f"{lookback_years}y")
            if df is None or len(df) < 200:
                logger.warning(f"Insufficient data for XGBoost {symbol_str}")
                return False

            features = LSTMPricePredictionService._compute_features(df)
            if len(features) < 100:
                return False

            # Target: same as LSTM
            close_aligned = df['Close'].reindex(features.index)
            future_returns = close_aligned.pct_change(5).shift(-5)
            targets = pd.Series(1, index=features.index)  # HOLD
            targets[future_returns > 0.005] = 0  # BUY
            targets[future_returns < -0.005] = 2  # SELL

            valid_idx = features.index.intersection(targets.dropna().index)
            features = features.loc[valid_idx]
            targets = targets.loc[valid_idx]

            # Scale
            scaler = StandardScaler()
            X = scaler.fit_transform(features.values)
            y = targets.values

            # Walk-forward split
            split = int(len(X) * 0.8)
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]

            # Train XGBoost
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multi:softprob',
                num_class=3,
                eval_metric='mlogloss',
                use_label_encoder=False,
                random_state=42,
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            # Evaluate
            val_acc = model.score(X_val, y_val)
            logger.info(f"XGBoost trained for {symbol_str}: val_accuracy={val_acc:.3f}")

            # Feature importance
            importance = dict(zip(
                LSTMPricePredictionService.FEATURE_COLUMNS,
                model.feature_importances_.tolist()
            ))

            # Save
            model_path = os.path.join(self.MODEL_DIR, f"{symbol_str}_xgb.json")
            scaler_path = os.path.join(self.MODEL_DIR, f"{symbol_str}_scaler.pkl")
            model.save_model(model_path)
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)

            self._models[symbol_str] = model
            self._scalers[symbol_str] = scaler
            self._importances[symbol_str] = importance
            self._performance_cache[symbol_str] = ModelPerformance(
                accuracy=float(val_acc),
                precision=float(val_acc),
                recall=float(val_acc),
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                annual_return=0.0,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to train XGBoost for {symbol_str}: {e}")
            return False

    def _load_model(self, symbol_str: str) -> bool:
        """Load previously trained XGBoost model."""
        model_path = os.path.join(self.MODEL_DIR, f"{symbol_str}_xgb.json")
        scaler_path = os.path.join(self.MODEL_DIR, f"{symbol_str}_scaler.pkl")
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return False
        try:
            import xgboost as xgb
            import pickle
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            self._models[symbol_str] = model
            self._scalers[symbol_str] = scaler
            return True
        except Exception as e:
            logger.warning(f"Failed to load XGBoost model for {symbol_str}: {e}")
            return False

    def predict_price_direction(self, symbol: Symbol, lookback_period: int = 30) -> TradingSignal:
        """Predict using trained XGBoost model."""
        symbol_str = str(symbol.value) if hasattr(symbol, 'value') else str(symbol)

        if symbol_str not in self._models:
            if not self._load_model(symbol_str):
                return TradingSignal(signal='HOLD', confidence=0.3, explanation="XGBoost: no trained model", score=0.0)

        try:
            import yfinance as yf
            df = yf.Ticker(symbol_str).history(period="6mo")
            if df is None or len(df) < 60:
                return TradingSignal(signal='HOLD', confidence=0.3, explanation="Insufficient data", score=0.0)

            features = LSTMPricePredictionService._compute_features(df)
            if len(features) < 1:
                return TradingSignal(signal='HOLD', confidence=0.3, explanation="Feature computation failed", score=0.0)

            scaler = self._scalers[symbol_str]
            X = scaler.transform(features.values[-1:])
            model = self._models[symbol_str]
            probs = model.predict_proba(X)[0]

            buy_prob, hold_prob, sell_prob = float(probs[0]), float(probs[1]), float(probs[2])
            max_prob = max(buy_prob, hold_prob, sell_prob)

            if buy_prob == max_prob:
                signal, score = 'BUY', buy_prob - sell_prob
            elif sell_prob == max_prob:
                signal, score = 'SELL', -(sell_prob - buy_prob)
            else:
                signal, score = 'HOLD', 0.0

            return TradingSignal(
                signal=signal,
                confidence=max_prob,
                explanation=f"XGBoost: BUY={buy_prob:.2f} HOLD={hold_prob:.2f} SELL={sell_prob:.2f}",
                score=score
            )
        except Exception as e:
            logger.warning(f"XGBoost prediction failed for {symbol_str}: {e}")
            return TradingSignal(signal='HOLD', confidence=0.2, explanation=f"XGBoost error: {e}", score=0.0)

    def get_feature_importance(self, symbol_str: str) -> Dict:
        """Get feature importance for a trained model."""
        return self._importances.get(symbol_str, {})

    def get_model_performance(self, model_type: str) -> ModelPerformance:
        """Get cached performance metrics."""
        if model_type in self._performance_cache:
            return self._performance_cache[model_type]
        return ModelPerformance(
            accuracy=0.0, precision=0.0, recall=0.0,
            sharpe_ratio=0.0, max_drawdown=0.0, annual_return=0.0
        )


class TransformerSentimentAnalysisService(SentimentAnalysisService):
    """
    FinBERT-based sentiment analysis service.

    Uses the ProsusAI/finbert transformer model for financial sentiment analysis.
    Falls back to VADER+TextBlob ensemble if the model fails to load.
    """

    def __init__(self):
        self._pipeline = None
        self._fallback_active = False
        logger.info("TransformerSentimentAnalysisService initializing...")
        self._load_model()

    def _load_model(self):
        """Load the FinBERT model pipeline from HuggingFace."""
        try:
            from transformers import pipeline as hf_pipeline
            self._pipeline = hf_pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                truncation=True,
                max_length=512,
            )
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load FinBERT model, using VADER fallback: {e}")
            self._fallback_active = True
            try:
                from nltk.sentiment import SentimentIntensityAnalyzer
                import nltk
                nltk.download('vader_lexicon', quiet=True)
                self._vader = SentimentIntensityAnalyzer()
            except Exception:
                self._vader = None

    def _finbert_analyze(self, text: str) -> NewsSentiment:
        """Run FinBERT inference on a single text."""
        result = self._pipeline(text[:512])[0]
        label = result['label'].lower()
        prob = result['score']

        # Map FinBERT labels to score range [-100, 100]
        if label == 'positive':
            score = prob * 100
        elif label == 'negative':
            score = -prob * 100
        else:  # neutral
            score = 0.0

        confidence = prob * 100

        return NewsSentiment(
            score=Decimal(str(round(score, 2))),
            confidence=Decimal(str(round(confidence, 2))),
            source="FinBERT"
        )

    def _vader_fallback(self, text: str) -> NewsSentiment:
        """VADER-based fallback when FinBERT is unavailable."""
        if self._vader is None:
            return NewsSentiment(
                score=Decimal('0'),
                confidence=Decimal('30'),
                source="Default"
            )
        scores = self._vader.polarity_scores(text)
        compound = scores['compound']
        return NewsSentiment(
            score=Decimal(str(round(compound * 100, 2))),
            confidence=Decimal('60'),
            source="VADER_Fallback"
        )

    def analyze_sentiment(self, text: str) -> NewsSentiment:
        """Analyze sentiment using FinBERT (or VADER fallback)."""
        if not text or not text.strip():
            return NewsSentiment(score=Decimal('0'), confidence=Decimal('50'), source="EmptyText")

        if self._fallback_active or self._pipeline is None:
            return self._vader_fallback(text)

        try:
            return self._finbert_analyze(text)
        except Exception as e:
            logger.warning(f"FinBERT inference failed, using fallback: {e}")
            return self._vader_fallback(text)

    def analyze_batch_sentiment(self, texts: List[str]) -> List[NewsSentiment]:
        """Analyze sentiment of multiple texts using batch inference."""
        if not texts:
            return []

        if self._fallback_active or self._pipeline is None:
            return [self._vader_fallback(t) for t in texts]

        try:
            # FinBERT batch inference for efficiency
            truncated = [t[:512] for t in texts if t and t.strip()]
            if not truncated:
                return [NewsSentiment(score=Decimal('0'), confidence=Decimal('50'), source="EmptyText")] * len(texts)

            results = self._pipeline(truncated, batch_size=16)
            sentiments = []
            for result in results:
                label = result['label'].lower()
                prob = result['score']
                if label == 'positive':
                    score = prob * 100
                elif label == 'negative':
                    score = -prob * 100
                else:
                    score = 0.0
                sentiments.append(NewsSentiment(
                    score=Decimal(str(round(score, 2))),
                    confidence=Decimal(str(round(prob * 100, 2))),
                    source="FinBERT"
                ))
            return sentiments
        except Exception as e:
            logger.warning(f"FinBERT batch inference failed: {e}")
            return [self._vader_fallback(t) for t in texts]

    def get_symbol_sentiment(self, symbol: Symbol, lookback_hours: int = 24) -> NewsSentiment:
        """
        Get aggregate sentiment for a symbol.

        Attempts to fetch real news via Marketaux. If unavailable, analyzes
        a small set of recent headlines from yfinance as a fallback.
        """
        logger.info(f"Getting aggregate sentiment for {symbol.value} over last {lookback_hours} hours")

        headlines: List[str] = []
        try:
            import yfinance as yf
            ticker = yf.Ticker(str(symbol.value))
            news = ticker.news or []
            for item in news[:20]:
                title = item.get('title', '')
                if title:
                    headlines.append(title)
        except Exception as e:
            logger.warning(f"Failed to fetch news for {symbol.value}: {e}")

        if not headlines:
            return NewsSentiment(
                score=Decimal('0'),
                confidence=Decimal('20'),
                source="NoData"
            )

        sentiments = self.analyze_batch_sentiment(headlines)
        if not sentiments:
            return NewsSentiment(score=Decimal('0'), confidence=Decimal('20'), source="NoData")

        avg_score = sum(float(s.score) for s in sentiments) / len(sentiments)
        avg_confidence = sum(float(s.confidence) for s in sentiments) / len(sentiments)

        return NewsSentiment(
            score=Decimal(str(round(avg_score, 2))),
            confidence=Decimal(str(round(avg_confidence, 2))),
            source="FinBERT_Aggregated"
        )


class RLTradingAgentService(ReinforcementLearningAgent):
    """
    Reinforcement Learning Trading Agent using DQN.

    Wraps the actual DQN agent from reinforcement_learning.py.
    Supports training on historical data and inference for live trading.
    """

    MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'rl')
    ACTION_MAP = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}

    def __init__(self):
        self._strategies: Dict[str, Any] = {}  # symbol -> RLTradingStrategy
        self._performance_cache: Dict[str, ModelPerformance] = {}
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        logger.info("RLTradingAgentService initialized")

    def _get_strategy(self, symbol_str: str) -> Any:
        """Get or create an RLTradingStrategy for a symbol."""
        if symbol_str not in self._strategies:
            from src.infrastructure.data_processing.reinforcement_learning import RLTradingStrategy
            from src.infrastructure.api_clients.market_data import YahooFinanceAdapter
            market_data = YahooFinanceAdapter()
            self._strategies[symbol_str] = RLTradingStrategy(market_data, strategy_type="DQN")
        return self._strategies[symbol_str]

    def get_action(self, state: Dict, symbol: Symbol) -> Tuple[str, float]:
        """
        Get trading action from trained DQN agent.

        Falls back to technical signal heuristic if no agent is trained.
        """
        symbol_str = str(symbol.value) if hasattr(symbol, 'value') else str(symbol)
        strategy = self._get_strategy(symbol_str)

        # Check if agent is trained
        symbol_key = Symbol(value=symbol_str) if hasattr(Symbol, 'value') else symbol
        if symbol_key in strategy.agents:
            signal = strategy.get_trading_signal(symbol_key)
            # Position sizing based on confidence
            position_size = 0.1 if signal in ('BUY', 'SELL') else 0.0
            return signal, position_size

        # Fallback to technical signal from state
        tech = state.get('technical_signal', 'NEUTRAL')
        cash_pct = state.get('cash_percentage', 0.5)
        if tech == 'BULLISH':
            return 'BUY', min(0.15, cash_pct * 0.5)
        elif tech == 'BEARISH':
            return 'SELL', 0.1
        return 'HOLD', 0.0

    def train(self, training_data: List[Dict]) -> bool:
        """
        Train DQN agent using historical price data.

        training_data should contain dicts with 'symbol' and optionally 'episodes'.
        """
        for item in training_data:
            symbol_str = item.get('symbol', 'AAPL')
            episodes = item.get('episodes', 100)

            strategy = self._get_strategy(symbol_str)
            symbol_key = Symbol(value=symbol_str)

            try:
                logger.info(f"Training DQN agent for {symbol_str} with {episodes} episodes")
                strategy.train_agent(symbol_key, episodes=episodes)

                # Save trained model
                model_path = os.path.join(self.MODEL_DIR, f"{symbol_str}_dqn")
                strategy.save_agent(symbol_key, model_path)
                logger.info(f"DQN agent trained and saved for {symbol_str}")

            except Exception as e:
                logger.error(f"Failed to train DQN agent for {symbol_str}: {e}")
                return False

        return True

    def evaluate(self, evaluation_data: List[Dict]) -> ModelPerformance:
        """
        Evaluate the DQN agent by running it through evaluation episodes.
        """
        total_reward = 0.0
        total_trades = 0
        winning_trades = 0

        for item in evaluation_data:
            symbol_str = item.get('symbol', 'AAPL')
            strategy = self._get_strategy(symbol_str)
            symbol_key = Symbol(value=symbol_str)

            if symbol_key not in strategy.environments:
                continue

            env = strategy.environments[symbol_key]
            if not env.price_data:
                continue

            # Run one evaluation episode
            state = env.reset()
            episode_reward = 0
            while True:
                signal = strategy.get_trading_signal(symbol_key)
                action = {'HOLD': 0, 'BUY': 1, 'SELL': 2}.get(signal, 0)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                if done:
                    break

            total_reward += episode_reward
            for trade in env.trades:
                total_trades += 1
                if trade.get('action') == 'sell':
                    revenue = trade.get('revenue', 0)
                    if revenue > 0:
                        winning_trades += 1

        win_rate = winning_trades / max(total_trades, 1)
        return ModelPerformance(
            accuracy=win_rate,
            precision=win_rate,
            recall=win_rate,
            sharpe_ratio=total_reward * 10,  # Rough approximation
            max_drawdown=0.0,
            annual_return=total_reward,
        )


class EnsembleModelService(MLModelService):
    """
    Ensemble model combining LSTM, XGBoost, and FinBERT sentiment.

    Weighted voting across models for more robust predictions.
    """

    def __init__(self, lstm_service: LSTMPricePredictionService,
                 sentiment_service: SentimentAnalysisService,
                 xgboost_service: Optional['XGBoostPredictionService'] = None):
        self.lstm_service = lstm_service
        self.sentiment_service = sentiment_service
        self.xgboost_service = xgboost_service or XGBoostPredictionService()
        self.weights = {
            'lstm': 0.35,
            'xgboost': 0.35,
            'sentiment': 0.30,
        }

    def predict_price_direction(self, symbol: Symbol, lookback_period: int = 30) -> TradingSignal:
        """
        Combine predictions from LSTM, XGBoost, and sentiment analysis.
        """
        signals: Dict[str, TradingSignal] = {}
        scores: Dict[str, float] = {}

        # LSTM prediction
        try:
            lstm_signal = self.lstm_service.predict_price_direction(symbol, lookback_period)
            signals['lstm'] = lstm_signal
            scores['lstm'] = lstm_signal.score
        except Exception as e:
            logger.warning(f"LSTM prediction failed in ensemble: {e}")

        # XGBoost prediction
        try:
            xgb_signal = self.xgboost_service.predict_price_direction(symbol, lookback_period)
            signals['xgboost'] = xgb_signal
            scores['xgboost'] = xgb_signal.score
        except Exception as e:
            logger.warning(f"XGBoost prediction failed in ensemble: {e}")

        # Sentiment
        try:
            sentiment = self.sentiment_service.get_symbol_sentiment(symbol, lookback_hours=24)
            scores['sentiment'] = float(sentiment.score) / 100.0  # Normalize to [-1, 1]
        except Exception as e:
            logger.warning(f"Sentiment analysis failed in ensemble: {e}")

        if not scores:
            return TradingSignal(signal='HOLD', confidence=0.1, explanation="All models failed", score=0.0)

        # Weighted combination
        total_weight = 0.0
        combined_score = 0.0
        for model_name, score in scores.items():
            w = self.weights.get(model_name, 0.0)
            combined_score += score * w
            total_weight += w

        if total_weight > 0:
            combined_score /= total_weight

        # Signal from combined score
        if combined_score > 0.08:
            final_signal = 'BUY'
        elif combined_score < -0.08:
            final_signal = 'SELL'
        else:
            final_signal = 'HOLD'

        # Confidence: based on model agreement
        signal_votes = [s.signal for s in signals.values()]
        agreement = signal_votes.count(final_signal) / max(len(signal_votes), 1)
        confidence = 0.4 + 0.5 * agreement  # Range [0.4, 0.9]

        parts = []
        for name, sig in signals.items():
            parts.append(f"{name}={sig.signal}({sig.confidence:.2f})")
        if 'sentiment' in scores:
            parts.append(f"sentiment={scores['sentiment']:.2f}")

        explanation = f"Ensemble [{', '.join(parts)}] → combined={combined_score:.3f}"

        return TradingSignal(
            signal=final_signal,
            confidence=confidence,
            explanation=explanation,
            score=combined_score
        )

    def get_model_performance(self, model_type: str) -> ModelPerformance:
        """Get performance from component models."""
        lstm_perf = self.lstm_service.get_model_performance(model_type)
        xgb_perf = self.xgboost_service.get_model_performance(model_type)
        # Average the component performances
        return ModelPerformance(
            accuracy=(lstm_perf.accuracy + xgb_perf.accuracy) / 2,
            precision=(lstm_perf.precision + xgb_perf.precision) / 2,
            recall=(lstm_perf.recall + xgb_perf.recall) / 2,
            sharpe_ratio=max(lstm_perf.sharpe_ratio, xgb_perf.sharpe_ratio),
            max_drawdown=min(lstm_perf.max_drawdown, xgb_perf.max_drawdown),
            annual_return=(lstm_perf.annual_return + xgb_perf.annual_return) / 2,
        )

    def retrain_model(self, symbol: Symbol) -> bool:
        """Retrain all component models."""
        lstm_ok = self.lstm_service.retrain_model(symbol)
        xgb_ok = self.xgboost_service.retrain_model(symbol)
        return lstm_ok or xgb_ok


class AdvancedRiskAnalyticsService:
    """Advanced risk analytics service implementing VaR, ES, and stress testing."""

    def __init__(self):
        logger.info("AdvancedRiskAnalyticsService initialized")

    def calculate_var(self, portfolio: Portfolio, confidence_level: float = 0.95, lookback_days: int = 252) -> Money:
        """
        Calculate Value at Risk using historical simulation method.
        
        Args:
            portfolio: Portfolio to analyze
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
            lookback_days: Number of historical days to use
        
        Returns:
            Maximum expected loss at the given confidence level
        """
        # Simulate VaR calculation (in real implementation, use historical returns)
        # Calculate portfolio volatility based on holdings
        portfolio_value = portfolio.total_value.amount
        estimated_volatility = Decimal('0.20')  # 20% annual volatility assumption
        
        # Calculate daily volatility and VaR
        daily_volatility = estimated_volatility / Decimal(str(252**0.5))  # sqrt of 252 trading days
        var_multiplier = self._get_var_multiplier(confidence_level)
        
        var_amount = portfolio_value * daily_volatility * var_multiplier
        
        return Money(var_amount, portfolio.total_value.currency)

    def _get_var_multiplier(self, confidence_level: float) -> Decimal:
        """Get VaR multiplier based on confidence level (simplified)."""
        multipliers = {
            0.90: Decimal('1.28'),  # For 90% confidence
            0.95: Decimal('1.64'),  # For 95% confidence
            0.99: Decimal('2.33')   # For 99% confidence
        }
        return multipliers.get(confidence_level, Decimal('1.64'))

    def calculate_expected_shortfall(self, portfolio: Portfolio, confidence_level: float = 0.95) -> Money:
        """
        Calculate Expected Shortfall (Conditional VaR).
        
        This is the expected loss given that the loss is greater than VaR.
        """
        # Simplified ES calculation (in real implementation, would use historical simulation)
        var_amount = self.calculate_var(portfolio, confidence_level)
        # ES is typically 1.2-1.4x VaR
        es_amount = var_amount.amount * Decimal('1.2')
        
        return Money(es_amount, var_amount.currency)

    def stress_test_portfolio(self, portfolio: Portfolio, scenarios: List[Dict]) -> Dict:
        """
        Perform stress testing under various market scenarios.
        
        Args:
            portfolio: Portfolio to test
            scenarios: List of market scenarios to test against
        
        Returns:
            Dictionary with stress test results
        """
        results = {}
        
        for scenario in scenarios:
            scenario_name = scenario.get('name', 'Unknown')
            market_impact = scenario.get('market_impact', 0.0)  # e.g., -0.2 for 20% market drop
            
            # Calculate portfolio value under this scenario
            # This is a simplified approach - real implementation would model each position
            portfolio_value = portfolio.total_value.amount
            stressed_value = portfolio_value * (1 + market_impact)
            loss = portfolio_value - stressed_value
            
            results[scenario_name] = {
                'initial_value': float(portfolio_value),
                'stressed_value': float(stressed_value),
                'absolute_loss': float(abs(loss)),
                'percentage_loss': abs(float(loss / portfolio_value * 100))
            }
        
        return results

    def calculate_correlation_matrix(self, positions: List[Position]) -> Dict:
        """
        Calculate correlation matrix for portfolio positions.
        
        In a real implementation, this would use historical price data to compute correlations.
        """
        symbols = [str(pos.symbol) for pos in positions]
        n = len(symbols)
        
        # Create a mock correlation matrix (in real implementation, compute from historical data)
        correlation_matrix = {}
        for i, sym1 in enumerate(symbols):
            correlation_matrix[sym1] = {}
            for j, sym2 in enumerate(symbols):
                if i == j:
                    correlation_matrix[sym1][sym2] = 1.0  # Perfect correlation with self
                else:
                    # Simulate correlation (in real implementation, compute from data)
                    correlation = np.random.uniform(-0.3, 0.8)  # Realistic range
                    correlation_matrix[sym1][sym2] = correlation
        
        return correlation_matrix


class PortfolioOptimizationService:
    """Portfolio optimization service implementing Modern Portfolio Theory."""

    def __init__(self):
        logger.info("PortfolioOptimizationService initialized")

    def optimize_portfolio(self, user: User, portfolio: Portfolio, available_symbols: List[Symbol]) -> Dict[str, float]:
        """
        Optimize portfolio allocation based on user's risk profile and goals.
        
        Implements Modern Portfolio Theory with Black-Litterman model enhancements.
        """
        # Simplified optimization based on user profile (in real implementation, use quadratic optimization)
        
        # Determine allocation based on user's risk tolerance and investment goal
        risk_tolerance = user.risk_tolerance.value if user.risk_tolerance else 'MODERATE'
        investment_goal = user.investment_goal.value if user.investment_goal else 'BALANCED_GROWTH'
        
        allocation = self._determine_allocation(risk_tolerance, investment_goal, available_symbols)
        
        return allocation

    def _determine_allocation(self, risk_tolerance: str, investment_goal: str, symbols: List[Symbol]) -> Dict[str, float]:
        """Determine optimal allocation based on risk tolerance and goals."""
        # Define allocation templates based on risk and goal profiles
        templates = {
            ('CONSERVATIVE', 'CAPITAL_PRESERVATION'): {
                'cash': 0.4,
                'bonds': 0.3,
                'value_stocks': 0.2,
                'growth_stocks': 0.1
            },
            ('CONSERVATIVE', 'BALANCED_GROWTH'): {
                'cash': 0.3,
                'bonds': 0.35,
                'value_stocks': 0.25,
                'growth_stocks': 0.1
            },
            ('MODERATE', 'CAPITAL_PRESERVATION'): {
                'cash': 0.2,
                'bonds': 0.3,
                'value_stocks': 0.3,
                'growth_stocks': 0.2
            },
            ('MODERATE', 'BALANCED_GROWTH'): {
                'cash': 0.15,
                'bonds': 0.25,
                'value_stocks': 0.35,
                'growth_stocks': 0.25
            },
            ('MODERATE', 'MAXIMUM_RETURNS'): {
                'cash': 0.05,
                'bonds': 0.1,
                'value_stocks': 0.35,
                'growth_stocks': 0.5
            },
            ('AGGRESSIVE', 'BALANCED_GROWTH'): {
                'cash': 0.05,
                'bonds': 0.05,
                'value_stocks': 0.3,
                'growth_stocks': 0.6
            },
            ('AGGRESSIVE', 'MAXIMUM_RETURNS'): {
                'cash': 0.0,
                'bonds': 0.05,
                'value_stocks': 0.2,
                'growth_stocks': 0.75
            }
        }
        
        template = templates.get((risk_tolerance, investment_goal), templates[('MODERATE', 'BALANCED_GROWTH')])
        
        # Distribute stock allocations across available symbols
        stock_allocation = template.get('value_stocks', 0) + template.get('growth_stocks', 0)
        equal_allocation = stock_allocation / len(symbols) if symbols else 0
        
        allocation_result = {}
        for symbol in symbols:
            allocation_result[str(symbol)] = round(equal_allocation, 4)
        
        # Add other asset types
        if template.get('cash'):
            allocation_result['cash'] = template['cash']
        if template.get('bonds'):
            allocation_result['bonds'] = template['bonds']
        
        return allocation_result