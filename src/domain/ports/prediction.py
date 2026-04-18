"""
Price-prediction port.

Architectural Intent:
- Narrow port for "given a symbol, what does the ensemble model think?"
- Separate from `AIModelPort` because that port exposes multiple concerns
  (trading signal string, price movement float, portfolio risk float) and
  this one's surface is a single call returning the richer `TradingSignal`
  value object the autonomous trading loop consumes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from src.domain.value_objects import Symbol

SignalDirection = Literal[
    "BUY", "SELL", "HOLD", "STRONG_BUY", "STRONG_SELL",
]


@dataclass(frozen=True)
class TradingSignal:
    """AI trading signal with reasoning and confidence.

    Immutable. Fields mirror what the infrastructure-side EnsembleModelService
    returns today; moving the dataclass here gives the domain layer a stable
    contract regardless of which ML adapter is behind it.
    """
    signal: SignalDirection
    confidence: float
    explanation: str
    score: float = 0.0


class PredictionPort(ABC):
    """Produce a trading signal for a given symbol."""

    @abstractmethod
    def predict_price_direction(self, symbol: Symbol) -> TradingSignal:
        """Return the model's forecast for the next trading window.

        Implementations should return a HOLD signal with low confidence
        when they cannot meaningfully predict (missing data, stale model)
        rather than raise — the autonomous loop treats low confidence as
        "skip" which is the right behaviour on data gaps.
        """
