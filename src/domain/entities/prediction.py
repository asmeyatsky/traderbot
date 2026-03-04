"""
Prediction Domain Entity

Architectural Intent:
- Frozen dataclass for ML prediction results with feature explanations
- Used by the ensemble predictor and surfaced in chat tool responses
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class PredictionResult:
    """Result of an ensemble ML prediction with explainability."""
    symbol: str
    direction: str           # "UP", "DOWN", "FLAT"
    confidence: float        # 0.0 – 1.0
    predicted_change_pct: float
    model_votes: Dict[str, str]  # e.g. {"RandomForest": "UP", "GradientBoosting": "DOWN"}
    top_features: List[Dict[str, float]] = field(default_factory=list)
    # Each item: {"feature": "RSI_14", "importance": 0.23, "direction": "positive"}
