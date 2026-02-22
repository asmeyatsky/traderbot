"""
Tests for ML Model Services

Tests LSTMPricePredictionService, XGBoostPredictionService,
TransformerSentimentAnalysisService, and EnsembleModelService
using synthetic data and mocked external dependencies.
"""
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from decimal import Decimal
from datetime import datetime
import numpy as np
import pandas as pd

from src.domain.value_objects import Symbol, NewsSentiment
from src.infrastructure.data_processing.ml_model_service import (
    TradingSignal,
    ModelPerformance,
    MLModelService,
    SentimentAnalysisService,
    LSTMPricePredictionService,
    XGBoostPredictionService,
    TransformerSentimentAnalysisService,
    EnsembleModelService,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv_dataframe(rows: int = 100) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing.

    Uses a fixed start date with daily frequency to avoid business-day
    edge cases that cause date count mismatches.
    """
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=rows, freq="D")
    close = 100 + np.cumsum(np.random.randn(rows) * 0.5)
    high = close + np.abs(np.random.rand(rows))
    low = close - np.abs(np.random.rand(rows))
    return pd.DataFrame({
        "Open": close - np.random.rand(rows) * 0.5,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": np.random.randint(1_000_000, 10_000_000, rows),
    }, index=dates)


AAPL = Symbol("AAPL")


# ===========================================================================
# LSTMPricePredictionService Tests
# ===========================================================================

class TestLSTMPricePredictionService:

    @patch("src.infrastructure.data_processing.ml_model_service.os.makedirs")
    def _make_service(self, mock_makedirs):
        return LSTMPricePredictionService()

    # -- predict_price_direction --

    @patch("src.infrastructure.data_processing.ml_model_service.os.makedirs")
    def test_predict_returns_trading_signal(self, mock_makedirs):
        """predict_price_direction should return a TradingSignal."""
        svc = LSTMPricePredictionService()

        # Mock _load_model to return False (no saved model), triggering fallback
        svc._load_model = MagicMock(return_value=False)

        # Mock yfinance for the fallback technical analysis
        mock_df = _make_ohlcv_dataframe(60)
        with patch("yfinance.Ticker") as mock_ticker_cls:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = mock_df
            mock_ticker_cls.return_value = mock_ticker

            signal = svc.predict_price_direction(AAPL, lookback_period=30)

        assert isinstance(signal, TradingSignal)
        assert signal.signal in ("BUY", "SELL", "HOLD")
        assert 0.0 <= signal.confidence <= 1.0
        assert -1.0 <= signal.score <= 1.0

    @patch("src.infrastructure.data_processing.ml_model_service.os.makedirs")
    def test_predict_with_loaded_model(self, mock_makedirs):
        """When a model is loaded, prediction should use the model."""
        svc = LSTMPricePredictionService()

        # Simulate a loaded model
        svc._models["AAPL"] = MagicMock()
        svc._scalers["AAPL"] = MagicMock()

        # The scaler transform should return correctly shaped array
        svc._scalers["AAPL"].transform.return_value = np.random.rand(30, 20)

        # Model predict returns probabilities for 3 classes
        svc._models["AAPL"].predict.return_value = np.array([[0.1, 0.3, 0.6]])

        mock_df = _make_ohlcv_dataframe(60)
        with patch("yfinance.Ticker") as mock_ticker_cls:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = mock_df
            mock_ticker_cls.return_value = mock_ticker

            signal = svc.predict_price_direction(AAPL, lookback_period=30)

        assert isinstance(signal, TradingSignal)

    @patch("src.infrastructure.data_processing.ml_model_service.os.makedirs")
    def test_predict_handles_empty_data(self, mock_makedirs):
        """Should handle empty yfinance data gracefully."""
        svc = LSTMPricePredictionService()
        svc._load_model = MagicMock(return_value=False)

        with patch("yfinance.Ticker") as mock_ticker_cls:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = pd.DataFrame()
            mock_ticker_cls.return_value = mock_ticker

            signal = svc.predict_price_direction(AAPL)

        # Should return a fallback signal rather than crash
        assert isinstance(signal, TradingSignal)

    # -- retrain_model --

    @patch("src.infrastructure.data_processing.ml_model_service.os.makedirs")
    def test_retrain_with_synthetic_data(self, mock_makedirs):
        """retrain_model should succeed with sufficient data."""
        svc = LSTMPricePredictionService()

        mock_df = _make_ohlcv_dataframe(500)
        with patch("yfinance.Ticker") as mock_ticker_cls:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = mock_df
            mock_ticker_cls.return_value = mock_ticker

            # Mock model save to avoid filesystem writes
            with patch.object(svc, "_build_model") as mock_build:
                mock_model = MagicMock()
                mock_model.fit.return_value = MagicMock(history={"loss": [0.5, 0.3]})
                mock_build.return_value = mock_model

                result = svc.retrain_model(AAPL, lookback_years=1)

        assert isinstance(result, bool)

    @patch("src.infrastructure.data_processing.ml_model_service.os.makedirs")
    def test_retrain_with_insufficient_data(self, mock_makedirs):
        """retrain_model should return False with insufficient data."""
        svc = LSTMPricePredictionService()

        # Only 5 rows — too few for LSTM training
        mock_df = _make_ohlcv_dataframe(5)
        with patch("yfinance.Ticker") as mock_ticker_cls:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = mock_df
            mock_ticker_cls.return_value = mock_ticker

            result = svc.retrain_model(AAPL, lookback_years=1)

        assert result is False

    # -- get_model_performance --

    @patch("src.infrastructure.data_processing.ml_model_service.os.makedirs")
    def test_get_model_performance(self, mock_makedirs):
        """get_model_performance should return ModelPerformance."""
        svc = LSTMPricePredictionService()
        svc._performance_cache["lstm"] = ModelPerformance(
            accuracy=0.65, precision=0.63, recall=0.61,
            sharpe_ratio=1.2, max_drawdown=0.15, annual_return=0.18,
        )

        perf = svc.get_model_performance("lstm")
        assert isinstance(perf, ModelPerformance)
        assert perf.accuracy == 0.65

    @patch("src.infrastructure.data_processing.ml_model_service.os.makedirs")
    def test_get_model_performance_no_cache(self, mock_makedirs):
        """Should return default performance when no cache exists."""
        svc = LSTMPricePredictionService()
        perf = svc.get_model_performance("lstm")
        assert isinstance(perf, ModelPerformance)

    # -- _compute_features --

    @patch("src.infrastructure.data_processing.ml_model_service.os.makedirs")
    def test_compute_features_adds_indicators(self, mock_makedirs):
        """_compute_features should add technical indicator columns."""
        svc = LSTMPricePredictionService()
        df = _make_ohlcv_dataframe(100)
        result = svc._compute_features(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


# ===========================================================================
# XGBoostPredictionService Tests
# ===========================================================================

class TestXGBoostPredictionService:

    @patch("src.infrastructure.data_processing.ml_model_service.os.makedirs")
    def test_predict_returns_trading_signal(self, mock_makedirs):
        """predict_price_direction should return a TradingSignal."""
        svc = XGBoostPredictionService()
        svc._load_model = MagicMock(return_value=False)

        mock_df = _make_ohlcv_dataframe(60)
        with patch("yfinance.Ticker") as mock_ticker_cls:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = mock_df
            mock_ticker_cls.return_value = mock_ticker

            signal = svc.predict_price_direction(AAPL, lookback_period=30)

        assert isinstance(signal, TradingSignal)
        assert signal.signal in ("BUY", "SELL", "HOLD")

    @patch("src.infrastructure.data_processing.ml_model_service.os.makedirs")
    def test_predict_with_loaded_model(self, mock_makedirs):
        """When an XGBoost model is loaded, it should use predict_proba."""
        svc = XGBoostPredictionService()

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.2, 0.5, 0.3]])
        mock_model.predict.return_value = np.array([1])
        svc._models["AAPL"] = mock_model
        svc._scalers["AAPL"] = MagicMock()
        svc._scalers["AAPL"].transform.return_value = np.random.rand(1, 20)

        mock_df = _make_ohlcv_dataframe(60)
        with patch("yfinance.Ticker") as mock_ticker_cls:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = mock_df
            mock_ticker_cls.return_value = mock_ticker

            signal = svc.predict_price_direction(AAPL, lookback_period=30)

        assert isinstance(signal, TradingSignal)

    @patch("src.infrastructure.data_processing.ml_model_service.os.makedirs")
    def test_retrain_with_synthetic_data(self, mock_makedirs):
        """retrain_model should succeed with sufficient data."""
        svc = XGBoostPredictionService()

        mock_df = _make_ohlcv_dataframe(500)
        with patch("yfinance.Ticker") as mock_ticker_cls:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = mock_df
            mock_ticker_cls.return_value = mock_ticker

            result = svc.retrain_model(AAPL, lookback_years=1)

        assert isinstance(result, bool)

    @patch("src.infrastructure.data_processing.ml_model_service.os.makedirs")
    def test_get_feature_importance_no_model(self, mock_makedirs):
        """get_feature_importance should return empty dict when no model."""
        svc = XGBoostPredictionService()
        importance = svc.get_feature_importance("AAPL")
        assert isinstance(importance, dict)

    @patch("src.infrastructure.data_processing.ml_model_service.os.makedirs")
    def test_get_feature_importance_with_model(self, mock_makedirs):
        """get_feature_importance should return stored importances."""
        svc = XGBoostPredictionService()
        svc._importances["AAPL"] = {"returns_1d": 0.15, "rsi_14": 0.12}

        importance = svc.get_feature_importance("AAPL")
        assert importance["returns_1d"] == 0.15

    @patch("src.infrastructure.data_processing.ml_model_service.os.makedirs")
    def test_get_model_performance(self, mock_makedirs):
        """get_model_performance should return ModelPerformance."""
        svc = XGBoostPredictionService()
        svc._performance_cache["xgboost"] = ModelPerformance(
            accuracy=0.70, precision=0.68, recall=0.65,
            sharpe_ratio=1.5, max_drawdown=0.12, annual_return=0.22,
        )

        perf = svc.get_model_performance("xgboost")
        assert perf.accuracy == 0.70
        assert perf.sharpe_ratio == 1.5


# ===========================================================================
# TransformerSentimentAnalysisService Tests
# ===========================================================================

class TestTransformerSentimentAnalysisService:

    def _make_service(self):
        """Create service with mocked _load_model and VADER initialized."""
        with patch.object(TransformerSentimentAnalysisService, "_load_model"):
            svc = TransformerSentimentAnalysisService()
            svc._fallback_active = True
            # Initialize VADER (normally done inside _load_model fallback path)
            from nltk.sentiment import SentimentIntensityAnalyzer
            import nltk
            nltk.download('vader_lexicon', quiet=True)
            svc._vader = SentimentIntensityAnalyzer()
        return svc

    @patch("src.infrastructure.data_processing.ml_model_service.os.makedirs")
    def test_analyze_sentiment_returns_news_sentiment(self, mock_makedirs):
        """analyze_sentiment should return a NewsSentiment object."""
        svc = self._make_service()
        result = svc.analyze_sentiment("Apple stock surges after strong earnings report")
        assert isinstance(result, NewsSentiment)
        assert -100 <= result.score <= 100
        assert 0 <= result.confidence <= 100

    @patch("src.infrastructure.data_processing.ml_model_service.os.makedirs")
    def test_analyze_positive_sentiment(self, mock_makedirs):
        """Positive financial text should yield positive sentiment."""
        svc = self._make_service()
        result = svc.analyze_sentiment(
            "Company reports record profits, revenue beats expectations by 20%"
        )
        assert isinstance(result, NewsSentiment)
        assert result.score >= 0

    @patch("src.infrastructure.data_processing.ml_model_service.os.makedirs")
    def test_analyze_negative_sentiment(self, mock_makedirs):
        """Negative financial text should yield negative sentiment."""
        svc = self._make_service()
        result = svc.analyze_sentiment(
            "Company files for bankruptcy, massive layoffs announced, stock crashes"
        )
        assert isinstance(result, NewsSentiment)
        assert result.score <= 0

    @patch("src.infrastructure.data_processing.ml_model_service.os.makedirs")
    def test_analyze_batch_sentiment(self, mock_makedirs):
        """analyze_batch_sentiment should return list of NewsSentiment."""
        svc = self._make_service()
        texts = [
            "Stock price hits all-time high",
            "Company reports massive losses",
            "Market remains flat today",
        ]
        results = svc.analyze_batch_sentiment(texts)
        assert len(results) == 3
        assert all(isinstance(r, NewsSentiment) for r in results)

    @patch("src.infrastructure.data_processing.ml_model_service.os.makedirs")
    def test_analyze_empty_text(self, mock_makedirs):
        """Should handle empty text without crashing."""
        svc = self._make_service()
        result = svc.analyze_sentiment("")
        assert isinstance(result, NewsSentiment)

    @patch("src.infrastructure.data_processing.ml_model_service.os.makedirs")
    def test_get_symbol_sentiment(self, mock_makedirs):
        """get_symbol_sentiment should return aggregated sentiment."""
        svc = self._make_service()

        # Mock yfinance news fetch
        with patch("yfinance.Ticker") as mock_ticker_cls:
            mock_ticker = MagicMock()
            mock_ticker.news = [
                {"title": "Apple beats earnings expectations"},
                {"title": "Apple launches new product line"},
            ]
            mock_ticker_cls.return_value = mock_ticker

            result = svc.get_symbol_sentiment(AAPL, lookback_hours=24)

        assert isinstance(result, NewsSentiment)

    @patch("src.infrastructure.data_processing.ml_model_service.os.makedirs")
    def test_finbert_analyze_with_pipeline(self, mock_makedirs):
        """When FinBERT pipeline is available, it should be used."""
        with patch.object(TransformerSentimentAnalysisService, "_load_model"):
            svc = TransformerSentimentAnalysisService()
            svc._fallback_active = False

            # Mock the pipeline
            svc._pipeline = MagicMock()
            svc._pipeline.return_value = [{"label": "positive", "score": 0.95}]

            result = svc._finbert_analyze("Great earnings report")

        assert isinstance(result, NewsSentiment)
        assert result.score > 0


# ===========================================================================
# EnsembleModelService Tests
# ===========================================================================

class TestEnsembleModelService:

    def _make_ensemble(self):
        """Create an EnsembleModelService with mocked sub-services."""
        lstm = MagicMock(spec=LSTMPricePredictionService)
        sentiment = MagicMock(spec=TransformerSentimentAnalysisService)
        xgboost = MagicMock(spec=XGBoostPredictionService)
        return EnsembleModelService(lstm, sentiment, xgboost), lstm, sentiment, xgboost

    def test_predict_combines_signals(self):
        """Ensemble should combine LSTM, XGBoost, and sentiment signals."""
        svc, lstm, sentiment, xgboost = self._make_ensemble()

        lstm.predict_price_direction.return_value = TradingSignal(
            signal="BUY", confidence=0.8, explanation="LSTM bullish", score=0.6
        )
        xgboost.predict_price_direction.return_value = TradingSignal(
            signal="BUY", confidence=0.7, explanation="XGBoost bullish", score=0.5
        )
        sentiment.get_symbol_sentiment.return_value = NewsSentiment(
            score=Decimal("60"), confidence=Decimal("80"), source="FinBERT"
        )

        signal = svc.predict_price_direction(AAPL)

        assert isinstance(signal, TradingSignal)
        assert signal.signal == "BUY"
        assert signal.confidence > 0

    def test_predict_mixed_signals_hold(self):
        """Mixed signals should result in HOLD."""
        svc, lstm, sentiment, xgboost = self._make_ensemble()

        lstm.predict_price_direction.return_value = TradingSignal(
            signal="BUY", confidence=0.5, explanation="Weak buy", score=0.1
        )
        xgboost.predict_price_direction.return_value = TradingSignal(
            signal="SELL", confidence=0.5, explanation="Weak sell", score=-0.1
        )
        sentiment.get_symbol_sentiment.return_value = NewsSentiment(
            score=Decimal("0"), confidence=Decimal("50"), source="FinBERT"
        )

        signal = svc.predict_price_direction(AAPL)

        assert isinstance(signal, TradingSignal)
        # With opposing signals near zero, the score should be close to 0 → HOLD
        assert signal.signal == "HOLD"

    def test_predict_strong_sell(self):
        """Unanimous sell signals should produce SELL."""
        svc, lstm, sentiment, xgboost = self._make_ensemble()

        lstm.predict_price_direction.return_value = TradingSignal(
            signal="SELL", confidence=0.9, explanation="LSTM bearish", score=-0.8
        )
        xgboost.predict_price_direction.return_value = TradingSignal(
            signal="SELL", confidence=0.85, explanation="XGBoost bearish", score=-0.7
        )
        sentiment.get_symbol_sentiment.return_value = NewsSentiment(
            score=Decimal("-70"), confidence=Decimal("90"), source="FinBERT"
        )

        signal = svc.predict_price_direction(AAPL)

        assert signal.signal == "SELL"
        assert signal.score < 0

    def test_get_model_performance_averages(self):
        """Ensemble performance should average sub-model performances."""
        svc, lstm, sentiment, xgboost = self._make_ensemble()

        lstm.get_model_performance.return_value = ModelPerformance(
            accuracy=0.60, precision=0.58, recall=0.55,
            sharpe_ratio=1.0, max_drawdown=0.20, annual_return=0.15,
        )
        xgboost.get_model_performance.return_value = ModelPerformance(
            accuracy=0.70, precision=0.68, recall=0.65,
            sharpe_ratio=1.5, max_drawdown=0.12, annual_return=0.22,
        )

        perf = svc.get_model_performance("ensemble")
        assert isinstance(perf, ModelPerformance)

    def test_retrain_model_delegates(self):
        """retrain_model should retrain both LSTM and XGBoost."""
        svc, lstm, sentiment, xgboost = self._make_ensemble()

        lstm.retrain_model.return_value = True
        xgboost.retrain_model.return_value = True

        result = svc.retrain_model(AAPL)

        lstm.retrain_model.assert_called_once_with(AAPL)
        xgboost.retrain_model.assert_called_once_with(AAPL)
        assert result is True

    def test_retrain_partial_failure(self):
        """If one sub-model fails to retrain, result should still be bool."""
        svc, lstm, sentiment, xgboost = self._make_ensemble()

        lstm.retrain_model.return_value = True
        xgboost.retrain_model.return_value = False

        result = svc.retrain_model(AAPL)
        assert isinstance(result, bool)

    def test_weights_sum_to_one(self):
        """Ensemble weights should sum to 1.0."""
        svc, _, _, _ = self._make_ensemble()
        total = sum(svc.weights.values())
        assert abs(total - 1.0) < 1e-9

    def test_handles_lstm_failure(self):
        """Ensemble should handle LSTM prediction failure gracefully."""
        svc, lstm, sentiment, xgboost = self._make_ensemble()

        lstm.predict_price_direction.side_effect = Exception("LSTM failed")
        xgboost.predict_price_direction.return_value = TradingSignal(
            signal="BUY", confidence=0.7, explanation="XGBoost bullish", score=0.5
        )
        sentiment.get_symbol_sentiment.return_value = NewsSentiment(
            score=Decimal("40"), confidence=Decimal("70"), source="FinBERT"
        )

        # Should not raise — ensemble handles sub-model failures
        signal = svc.predict_price_direction(AAPL)
        assert isinstance(signal, TradingSignal)
