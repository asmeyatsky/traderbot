"""
Advanced ML/AI Model Service

Implements advanced machine learning and AI capabilities for trading including:
- Predictive models
- Pattern recognition
- Market regime detection
- Portfolio optimization
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from decimal import Decimal
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import json
from enum import Enum

from src.domain.entities.trading import Portfolio, Position, Order
from src.domain.entities.user import User
from src.domain.value_objects import Symbol, Price


class MLModelType(Enum):
    PRICE_PREDICTION = "price_prediction"
    VOLATILITY_FORECAST = "volatility_forecast"
    REGIME_DETECTION = "regime_detection"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    RISK_PREDICTION = "risk_prediction"
    MOMENTUM_DETECTION = "momentum_detection"


class TradingSignal(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


@dataclass
class PredictionResult:
    """Data class for prediction results"""
    symbol: Symbol
    predicted_price: Price
    confidence: Decimal  # 0-100
    prediction_horizon: str  # "1d", "5d", "22d" (1 day, 1 week, 1 month)
    model_used: str
    features_used: List[str]
    prediction_timestamp: datetime
    technical_indicators: Dict[str, Any]
    market_regime: str


@dataclass
class MarketRegime:
    """Data class for market regime detection"""
    regime_type: str  # "bull", "bear", "high_volatility", "low_volatility", "trending", "sideways"
    confidence: Decimal  # 0-100
    start_date: datetime
    end_date: Optional[datetime] = None
    characteristics: Dict[str, Any] = None  # Volatility, correlation, trend strength, etc.


@dataclass
class ModelPerformance:
    """Data class for model performance metrics"""
    model_type: MLModelType
    accuracy: Decimal
    precision: Decimal
    recall: Decimal
    f1_score: Decimal
    sharpe_ratio: Decimal
    max_drawdown: Decimal
    backtest_return: Decimal
    training_date: datetime
    model_version: str
    features_importance: Dict[str, float]


class MLModelService(ABC):
    """
    Abstract base class for ML/AI model services.
    """
    
    @abstractmethod
    def predict_price(self, symbol: Symbol, days_ahead: int = 1) -> PredictionResult:
        """Predict future price for a symbol"""
        pass
    
    @abstractmethod
    def detect_market_regime(self, symbol: Symbol) -> MarketRegime:
        """Detect current market regime for a symbol"""
        pass
    
    @abstractmethod
    def generate_trading_signal(self, symbol: Symbol, user: User) -> Tuple[TradingSignal, Decimal]:
        """Generate trading signal for a symbol"""
        pass
    
    @abstractmethod
    def optimize_portfolio(self, portfolio: Portfolio, user: User) -> Dict[str, Any]:
        """Optimize portfolio allocation"""
        pass
    
    @abstractmethod
    def forecast_volatility(self, symbol: Symbol, days: int = 30) -> Decimal:
        """Forecast volatility for a symbol"""
        pass
    
    @abstractmethod
    def get_model_performance(self, model_type: MLModelType) -> ModelPerformance:
        """Get performance metrics for a model type"""
        pass
    
    @abstractmethod
    def retrain_model(self, model_type: MLModelType) -> bool:
        """Retrain a specific model"""
        pass


class DefaultMLModelService(MLModelService):
    """
    Default implementation of ML/AI model services.
    Note: This is a simplified implementation using mock data and basic models - 
    in production, this would use sophisticated models and real data
    """
    
    def __init__(self):
        self._price_history = self._generate_mock_price_history()
        self._models = {}
        self._model_performance = {}
        self._market_regimes = self._generate_mock_market_regimes()
        
        # Initialize basic models
        self._initialize_models()
    
    def _generate_mock_price_history(self) -> Dict[str, List[Tuple[datetime, Decimal]]]:
        """Generate mock historical price data for common symbols"""
        np.random.seed(42)  # For reproducible mock results
        
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'SPY', 'QQQ', 'DIS', 'MCD']
        history = {}
        
        for symbol in symbols:
            # Generate mock historical prices (simulating realistic price movements)
            base_price = np.random.uniform(50, 300)  # Random base price
            prices = []
            current_date = datetime.now() - timedelta(days=252)  # 1 year of data
            
            for day in range(252):
                # Generate price change (typically 0-3% daily movement)
                change_pct = np.random.normal(0.001, 0.02)  # 0.1% average daily return, 2% std dev
                new_price = base_price * (1 + change_pct)
                
                # Ensure price doesn't go below $1
                if new_price < 1.0:
                    new_price = 1.0
                
                prices.append((current_date, Decimal(str(round(new_price, 2)))))
                base_price = new_price
                current_date += timedelta(days=1)
            
            history[symbol] = prices
        
        return history
    
    def _generate_mock_market_regimes(self) -> Dict[str, List[MarketRegime]]:
        """Generate mock market regime data"""
        symbols = ['AAPL', 'GOOGL', 'SPY', 'QQQ']
        regimes = {}
        
        for symbol in symbols:
            regimes[symbol] = [
                MarketRegime(
                    regime_type=np.random.choice(["bull", "bear", "high_volatility", "low_volatility"]),
                    confidence=Decimal(str(round(np.random.uniform(60, 95), 2))),
                    start_date=datetime.now() - timedelta(days=30),
                    characteristics={
                        "volatility": round(np.random.uniform(0.15, 0.40), 4),
                        "correlation": round(np.random.uniform(0.3, 0.8), 3),
                        "trend_strength": round(np.random.uniform(0.1, 0.9), 3)
                    }
                )
            ]
        
        return regimes
    
    def _initialize_models(self):
        """Initialize basic ML models"""
        # In a real implementation, this would train actual models
        # For mock, we'll just create placeholder model information
        self._models[MLModelType.PRICE_PREDICTION] = {
            'type': 'RandomForest',
            'features': ['sma_20', 'sma_50', 'rsi', 'macd', 'volume'],
            'last_trained': datetime.now() - timedelta(days=1)
        }
        
        self._models[MLModelType.VOLATILITY_FORECAST] = {
            'type': 'GARCH',
            'features': ['historical_vol', 'volume', 'price_range'],
            'last_trained': datetime.now() - timedelta(days=1)
        }
        
        self._models[MLModelType.REGIME_DETECTION] = {
            'type': 'KMeans',
            'features': ['volatility', 'correlation', 'trend'],
            'last_trained': datetime.now() - timedelta(days=1)
        }
    
    def predict_price(self, symbol: Symbol, days_ahead: int = 1) -> PredictionResult:
        """Predict future price for a symbol"""
        # Get historical data for feature engineering
        hist_data = self._price_history.get(str(symbol), [])
        if not hist_data:
            # If no history for symbol, create mock based on random walk
            hist_data = [(datetime.now(), Decimal('100.00'))]
        
        # Calculate some technical indicators as features
        prices = [price for _, price in hist_data[-20:]]  # Last 20 days
        if len(prices) < 2:
            current_price = Decimal('100.00')
        else:
            current_price = prices[-1]
        
        # Calculate simple technical indicators
        sma_20 = sum(prices) / len(prices) if prices else current_price
        sma_5 = sum(prices[-5:]) / min(5, len(prices)) if prices else current_price
        
        # Simple mock prediction: add random walk with drift
        drift = np.random.normal(0.001, 0.02)  # Daily drift
        predicted_price = current_price * (1 + Decimal(str(drift * days_ahead)))
        
        # Calculate confidence based on model performance (mock)
        confidence = Decimal(str(round(np.random.uniform(65, 85), 2)))
        
        return PredictionResult(
            symbol=symbol,
            predicted_price=Price(predicted_price, 'USD'),
            confidence=confidence,
            prediction_horizon=f"{days_ahead}d",
            model_used="RandomWalkDrift",
            features_used=['sma_20', 'sma_5', 'current_price', 'volatility'],
            prediction_timestamp=datetime.now(),
            technical_indicators={
                'sma_20': float(sma_20),
                'sma_5': float(sma_5),
                'current_price': float(current_price)
            },
            market_regime=self.detect_market_regime(symbol).regime_type
        )
    
    def detect_market_regime(self, symbol: Symbol) -> MarketRegime:
        """Detect current market regime for a symbol"""
        # Get the latest regime from our mock data
        regimes = self._market_regimes.get(str(symbol), [])
        if regimes:
            return regimes[0]
        
        # If no regime data, return a mock regime
        return MarketRegime(
            regime_type=np.random.choice(["bull", "bear", "high_volatility", "low_volatility", "trending", "sideways"]),
            confidence=Decimal(str(round(np.random.uniform(65, 90), 2))),
            start_date=datetime.now() - timedelta(days=np.random.randint(10, 90)),
            characteristics={
                "volatility": round(np.random.uniform(0.15, 0.40), 4),
                "correlation": round(np.random.uniform(0.3, 0.8), 3),
                "trend_strength": round(np.random.uniform(0.1, 0.9), 3)
            }
        )
    
    def generate_trading_signal(self, symbol: Symbol, user: User) -> Tuple[TradingSignal, Decimal]:
        """Generate trading signal for a symbol"""
        # Get price prediction
        pred_result = self.predict_price(symbol, days_ahead=5)
        
        # Compare prediction to current price to generate signal
        hist_data = self._price_history.get(str(symbol), [])
        if not hist_data:
            return TradingSignal.HOLD, Decimal('50.0')
        
        current_price = hist_data[-1][1]  # Last price
        predicted_price = pred_result.predicted_price.amount
        
        # Calculate expected return
        expected_return = (predicted_price - current_price) / current_price * 100
        
        # Determine signal based on expected return and user's risk tolerance
        confidence = pred_result.confidence
        
        if expected_return > Decimal('3.0'):  # If expect >3% return
            signal = TradingSignal.STRONG_BUY if expected_return > Decimal('5.0') else TradingSignal.BUY
        elif expected_return < Decimal('-3.0'):  # If expect <-3% return
            signal = TradingSignal.STRONG_SELL if expected_return < Decimal('-5.0') else TradingSignal.SELL
        else:
            signal = TradingSignal.HOLD
        
        # Adjust confidence based on user risk profile
        if user.risk_tolerance.name == 'CONSERVATIVE':
            # Conservative users get lower confidence signals
            confidence = confidence * Decimal('0.7')
        elif user.risk_tolerance.name == 'AGGRESSIVE':
            # Aggressive users get higher confidence signals
            confidence = min(confidence * Decimal('1.2'), Decimal('95.0'))
        
        return signal, confidence
    
    def optimize_portfolio(self, portfolio: Portfolio, user: User) -> Dict[str, Any]:
        """Optimize portfolio allocation"""
        # This is a mock portfolio optimization using Modern Portfolio Theory principles
        
        # Get current portfolio weights
        total_value = portfolio.total_value.amount
        if total_value <= 0:
            return {"allocation": {}, "risk_metrics": {}, "status": "error", "message": "Portfolio has no value to optimize"}
        
        # Calculate current weights
        current_weights = {}
        for position in portfolio.positions:
            weight = (position.market_value.amount / total_value) if total_value > 0 else 0
            current_weights[str(position.symbol)] = float(weight)
        
        # Generate mock optimal weights based on user profile
        target_weights = {}
        
        # Adjust allocation based on user's investment goal
        if user.investment_goal.name == 'CAPITAL_PRESERVATION':
            # More conservative allocation
            target_weights = {sym: weight * 0.8 if weight > 0.1 else weight for sym, weight in current_weights.items()}
            # Add 20% to cash if available
        elif user.investment_goal.name == 'MAXIMUM_RETURNS':
            # More aggressive allocation
            target_weights = {sym: min(weight * 1.2, 0.3) for sym, weight in current_weights.items()}  # Max 30% per stock
        else:  # BALANCED_GROWTH
            # Keep allocation similar to current
            target_weights = current_weights.copy()
        
        # Calculate mock risk metrics
        volatility = Decimal(str(round(np.random.uniform(0.12, 0.25), 4)))
        sharpe_ratio = Decimal(str(round(np.random.uniform(0.8, 1.8), 3)))
        
        return {
            "allocation": target_weights,
            "risk_metrics": {
                "expected_volatility": float(volatility),
                "expected_sharpe_ratio": float(sharpe_ratio),
                "diversification_score": round(np.random.uniform(0.6, 0.95), 3)
            },
            "rebalancing_suggestions": self._generate_rebalancing_suggestions(portfolio, target_weights),
            "optimization_method": "Markowitz Mean-Variance Optimization (Mock)",
            "calculation_timestamp": datetime.now().isoformat()
        }
    
    def forecast_volatility(self, symbol: Symbol, days: int = 30) -> Decimal:
        """Forecast volatility for a symbol"""
        # Get historical data
        hist_data = self._price_history.get(str(symbol), [])
        if len(hist_data) < 30:
            # If not enough history, return a default volatility
            return Decimal('0.20')  # 20% annualized
        
        # Calculate returns
        prices = [price for _, price in hist_data[-60:]]  # Use 60 days to have enough returns
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] != 0:
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(float(ret))
        
        if returns:
            # Calculate annualized volatility
            daily_vol = np.std(returns)
            annualized_vol = daily_vol * np.sqrt(252)  # 252 trading days
            return Decimal(str(round(min(annualized_vol, 1.0), 4)))  # Cap at 100%
        
        return Decimal('0.20')
    
    def get_model_performance(self, model_type: MLModelType) -> ModelPerformance:
        """Get performance metrics for a model type"""
        # Generate mock performance metrics
        accuracy = Decimal(str(round(np.random.uniform(0.55, 0.85), 4)))
        precision = Decimal(str(round(np.random.uniform(0.50, 0.80), 4)))
        recall = Decimal(str(round(np.random.uniform(0.50, 0.80), 4)))
        f1_score = (precision + recall) / 2  # F1 is the harmonic mean of precision and recall
        
        sharpe_ratio = Decimal(str(round(np.random.uniform(0.8, 2.0), 3)))
        max_drawdown = Decimal(str(round(np.random.uniform(5, 25), 2)))
        backtest_return = Decimal(str(round(np.random.uniform(8, 25), 2)))
        
        # Feature importance for mock
        feature_importance = {
            "sma_20": np.random.uniform(0.2, 0.3),
            "rsi": np.random.uniform(0.15, 0.25),
            "volume": np.random.uniform(0.1, 0.2),
            "macd": np.random.uniform(0.1, 0.2),
            "bollinger": np.random.uniform(0.05, 0.15)
        }
        
        return ModelPerformance(
            model_type=model_type,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            backtest_return=backtest_return,
            training_date=datetime.now() - timedelta(days=np.random.randint(1, 7)),
            model_version="1.0.0",
            features_importance=feature_importance
        )
    
    def retrain_model(self, model_type: MLModelType) -> bool:
        """Retrain a specific model"""
        # In a real implementation, this would retrain the model
        # For mock, just update the last trained timestamp
        if model_type in self._models:
            self._models[model_type]['last_trained'] = datetime.now()
            # Update mock performance metrics
            self._model_performance[model_type] = self.get_model_performance(model_type)
            return True
        return False
    
    def _generate_rebalancing_suggestions(self, portfolio: Portfolio, target_weights: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate rebalancing suggestions based on target weights"""
        suggestions = []
        
        # Compare target weights to current positions
        total_value = portfolio.total_value.amount
        if total_value <= 0:
            return suggestions
        
        for position in portfolio.positions:
            symbol = str(position.symbol)
            current_weight = float(position.market_value.amount / total_value) if total_value > 0 else 0
            target_weight = target_weights.get(symbol, 0)
            
            if abs(current_weight - target_weight) > 0.05:  # If diff > 5%, suggest rebalance
                value_difference = (target_weight - current_weight) * total_value
                shares_difference = int(value_difference / float(position.current_price.amount)) if position.current_price.amount > 0 else 0
                
                suggestion = {
                    "symbol": symbol,
                    "current_weight": current_weight,
                    "target_weight": target_weight,
                    "value_difference": float(value_difference),
                    "shares_difference": shares_difference,
                    "action": "buy" if shares_difference > 0 else "sell" if shares_difference < 0 else "hold"
                }
                
                suggestions.append(suggestion)
        
        return suggestions