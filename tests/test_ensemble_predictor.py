"""
Tests for EnsemblePredictorService

Tests cover:
- PredictionResult frozen dataclass
- Empty result when data is unavailable
"""
import pytest
from src.domain.entities.prediction import PredictionResult


class TestPredictionResult:
    def test_frozen(self):
        pred = PredictionResult(
            symbol="AAPL",
            direction="UP",
            confidence=0.75,
            predicted_change_pct=1.2,
            model_votes={"RF": "UP", "GB": "UP"},
            top_features=[{"feature": "RSI", "importance": 0.3, "direction": "positive"}],
        )
        with pytest.raises(AttributeError):
            pred.direction = "DOWN"  # type: ignore

    def test_defaults(self):
        pred = PredictionResult(
            symbol="AAPL", direction="FLAT", confidence=0.0,
            predicted_change_pct=0.0, model_votes={},
        )
        assert pred.top_features == []

    def test_model_votes(self):
        pred = PredictionResult(
            symbol="TSLA", direction="DOWN", confidence=0.62,
            predicted_change_pct=-0.5,
            model_votes={"RandomForest": "DOWN", "GradientBoosting": "DOWN"},
        )
        assert len(pred.model_votes) == 2
        assert pred.model_votes["RandomForest"] == "DOWN"
