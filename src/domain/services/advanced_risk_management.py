"""
Advanced Risk Management Service

Implements advanced risk metrics including VaR, ES, stress testing, and correlation analysis
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from decimal import Decimal
import numpy as np
import pandas as pd
from scipy import stats
from enum import Enum

from src.domain.entities.trading import Portfolio, Position
from src.domain.entities.user import User
from src.domain.value_objects import Money


class RiskMetric(Enum):
    VALUE_AT_RISK = "VaR"
    EXPECTED_SHORTFALL = "ES"
    MAX_DRAWDOWN = "MaxDD"
    VOLATILITY = "Volatility"
    BETA = "Beta"
    SHARPE_RATIO = "SharpeRatio"
    SORTINO_RATIO = "SortinoRatio"


@dataclass
class RiskMetrics:
    """Data class for storing risk metrics"""
    value_at_risk: Optional[Money] = None
    expected_shortfall: Optional[Money] = None
    max_drawdown: Optional[Decimal] = None
    volatility: Optional[Decimal] = None
    beta: Optional[Decimal] = None
    sharpe_ratio: Optional[Decimal] = None
    sortino_ratio: Optional[Decimal] = None
    correlation_matrix: Optional[Dict[str, Dict[str, Decimal]]] = None
    stress_test_results: Optional[Dict[str, Money]] = None
    portfolio_at_risk: Optional[Dict[str, Decimal]] = None


@dataclass
class StressTestScenario:
    """Data class for stress testing scenarios"""
    name: str
    description: str
    market_move: Dict[str, Decimal]  # Asset -> percentage move
    probability: Decimal


class AdvancedRiskManagementService(ABC):
    """
    Abstract base class for advanced risk management services.
    """
    
    @abstractmethod
    def calculate_var(self, portfolio: Portfolio, confidence_level: Decimal = Decimal('95.0'), 
                     time_horizon: int = 1) -> Money:
        """Calculate Value at Risk for the portfolio"""
        pass
    
    @abstractmethod
    def calculate_expected_shortfall(self, portfolio: Portfolio, confidence_level: Decimal = Decimal('95.0')) -> Money:
        """Calculate Expected Shortfall (Conditional VaR)"""
        pass
    
    @abstractmethod
    def calculate_portfolio_metrics(self, portfolio: Portfolio, lookback_days: int = 252) -> RiskMetrics:
        """Calculate comprehensive risk metrics for the portfolio"""
        pass
    
    @abstractmethod
    def perform_stress_test(self, portfolio: Portfolio, scenario: StressTestScenario) -> Money:
        """Perform stress testing under specific market conditions"""
        pass
    
    @abstractmethod
    def calculate_correlation_matrix(self, portfolio: Portfolio) -> Dict[str, Dict[str, Decimal]]:
        """Calculate correlation matrix between portfolio holdings"""
        pass
    
    @abstractmethod
    def calculate_risk_contribution(self, portfolio: Portfolio) -> Dict[str, Decimal]:
        """Calculate risk contribution by asset"""
        pass


class DefaultAdvancedRiskManagementService(AdvancedRiskManagementService):
    """
    Default implementation of advanced risk management services.
    Note: This is a simplified implementation using mock data - in production,
    this would connect to market data and use actual historical returns
    """
    
    def __init__(self):
        self._historical_returns_cache = {}
        self._rng = np.random.RandomState(42)  # Dedicated RNG for deterministic results
        self._market_data = self._get_mock_market_data()
    
    def _get_mock_market_data(self) -> Dict:
        """Mock market data for demonstration purposes"""
        # Simulated historical returns for various assets
        # Mock historical returns (daily) for different assets
        assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'SPY', 'QQQ']
        returns_data = {}

        rng = np.random.RandomState(42)  # Separate seed for market data generation
        for asset in assets:
            # Generate correlated returns with some realistic volatility
            base_returns = rng.normal(0.0005, 0.02, 252)  # Daily returns ~0.05% avg, 2% std
            returns_data[asset] = base_returns
        
        return returns_data
    
    def calculate_var(self, portfolio: Portfolio, confidence_level: Decimal = Decimal('95.0'), 
                     time_horizon: int = 1) -> Money:
        """
        Calculate Value at Risk using historical simulation method
        """
        # For this mock implementation, we'll use the portfolio's current value
        # and assume a standard deviation based on the portfolio's holdings
        
        # Calculate portfolio volatility based on holdings
        portfolio_value = portfolio.total_value.amount
        portfolio_weights = self._calculate_portfolio_weights(portfolio)
        
        # Mock volatility calculation (in reality, this would use historical returns)
        portfolio_volatility = self._calculate_mock_portfolio_volatility(portfolio, portfolio_weights)
        
        # Apply confidence level (for normal distribution)
        z_score = stats.norm.ppf(float(confidence_level / 100))
        
        # Calculate VaR
        var_amount = portfolio_value * Decimal(str(z_score)) * portfolio_volatility * Decimal(str(time_horizon ** 0.5))
        
        # Return absolute value (VaR should be positive)
        return Money(abs(var_amount).quantize(Decimal('0.01')), portfolio.total_value.currency)
    
    def calculate_expected_shortfall(self, portfolio: Portfolio, confidence_level: Decimal = Decimal('95.0')) -> Money:
        """
        Calculate Expected Shortfall (Conditional VaR) using historical simulation
        """
        # Mock implementation - in reality, this would use historical returns
        var_result = self.calculate_var(portfolio, confidence_level)
        
        # ES is typically ~10-20% higher than VaR for normal distributions
        es_amount = var_result.amount * Decimal('1.15')
        
        return Money(es_amount.quantize(Decimal('0.01')), var_result.currency)
    
    def calculate_portfolio_metrics(self, portfolio: Portfolio, lookback_days: int = 252) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for the portfolio
        """
        # Reset RNG so results are deterministic across requests for the same portfolio
        self._rng = np.random.RandomState(42)

        # Calculate individual metrics
        var = self.calculate_var(portfolio)
        es = self.calculate_expected_shortfall(portfolio)
        
        # Mock other metrics
        max_drawdown = self._calculate_mock_max_drawdown(portfolio)
        volatility = self._calculate_mock_volatility(portfolio)
        beta = self._calculate_mock_beta(portfolio)
        sharpe = self._calculate_mock_sharpe_ratio(portfolio)
        sortino = self._calculate_mock_sortino_ratio(portfolio)
        
        correlation_matrix = self.calculate_correlation_matrix(portfolio)
        risk_contribution = self.calculate_risk_contribution(portfolio)
        
        # Mock stress test results with a few scenarios
        stress_test_results = {
            "2008 Financial Crisis": Money(Decimal('-50000'), 'USD'),
            "COVID-19 Crash": Money(Decimal('-30000'), 'USD'),
            "Dot-com Bubble": Money(Decimal('-40000'), 'USD')
        }
        
        return RiskMetrics(
            value_at_risk=var,
            expected_shortfall=es,
            max_drawdown=max_drawdown,
            volatility=volatility,
            beta=beta,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            correlation_matrix=correlation_matrix,
            stress_test_results=stress_test_results,
            portfolio_at_risk=risk_contribution
        )
    
    def perform_stress_test(self, portfolio: Portfolio, scenario: StressTestScenario) -> Money:
        """
        Perform stress testing under specific market conditions
        """
        # Mock implementation: apply the scenario's market moves to portfolio
        portfolio_value = portfolio.total_value.amount
        stress_impact = Decimal('0.0')
        
        # Apply market moves based on portfolio holdings
        for symbol, move in scenario.market_move.items():
            position = portfolio.get_position(symbol)
            if position:
                # Calculate weighted impact based on position size
                position_weight = (position.market_value.amount / portfolio_value) if portfolio_value > 0 else Decimal('0')
                stress_impact += move * position_weight
        
        # Apply the total impact to portfolio value
        stressed_value = portfolio_value * (Decimal('1') + stress_impact / Decimal('100'))
        loss = portfolio_value - stressed_value
        
        return Money(abs(loss).quantize(Decimal('0.01')), portfolio.total_value.currency)
    
    def calculate_correlation_matrix(self, portfolio: Portfolio) -> Dict[str, Dict[str, Decimal]]:
        """
        Calculate correlation matrix between portfolio holdings
        """
        symbols = [str(pos.symbol) for pos in portfolio.positions]
        correlation_matrix = {}
        
        # Generate mock correlation matrix (in reality, this would use historical returns)
        n = len(symbols)
        if n == 0:
            return {}
        
        # Create a mock correlation matrix
        for i, sym1 in enumerate(symbols):
            correlation_matrix[sym1] = {}
            for j, sym2 in enumerate(symbols):
                if i == j:
                    correlation_matrix[sym1][sym2] = Decimal('1.0')  # Perfect correlation with self
                else:
                    # Generate deterministic correlated value between 0.1 and 0.8
                    # Use hash of symbol pair for deterministic results across requests
                    pair_seed = hash((sym1, sym2)) % (2**31)
                    pair_rng = np.random.RandomState(pair_seed)
                    correlation_matrix[sym1][sym2] = Decimal(str(round(pair_rng.uniform(0.1, 0.8), 3)))
        
        return correlation_matrix
    
    def calculate_risk_contribution(self, portfolio: Portfolio) -> Dict[str, Decimal]:
        """
        Calculate risk contribution by asset
        """
        contributions = {}
        
        # Mock implementation - in reality, this would be based on marginal VaR
        total_risk = Decimal('1.0')  # Placeholder for total portfolio risk
        
        for position in portfolio.positions:
            # Calculate weight-based risk contribution (simplified)
            position_value = position.market_value.amount
            total_value = portfolio.total_value.amount
            weight = position_value / total_value if total_value > 0 else Decimal('0')
            
            # Assign a risk factor (mock - would be calculated based on volatility and correlation)
            risk_factor = Decimal(str(round(self._rng.uniform(0.8, 1.2), 3)))
            
            risk_contribution = weight * risk_factor
            contributions[str(position.symbol)] = risk_contribution
        
        # Normalize so contributions sum to 1
        total_contribution = sum(contributions.values())
        if total_contribution > 0:
            for symbol in contributions:
                contributions[symbol] = contributions[symbol] / total_contribution
        
        return contributions
    
    def _calculate_portfolio_weights(self, portfolio: Portfolio) -> Dict[str, Decimal]:
        """Calculate portfolio weights for each position"""
        weights = {}
        total_value = portfolio.total_value.amount
        
        if total_value <= 0:
            return weights
        
        for position in portfolio.positions:
            weight = position.market_value.amount / total_value
            weights[str(position.symbol)] = weight
        
        return weights
    
    def _calculate_mock_portfolio_volatility(self, portfolio: Portfolio, weights: Dict[str, Decimal]) -> Decimal:
        """Calculate mock portfolio volatility based on weights"""
        # In reality, this would use historical returns and correlation matrix
        # For mock, we'll use a simplified approach
        if not weights:
            return Decimal('0.15')  # 15% annualized volatility as default
        
        # Mock volatility calculation
        total_vol = Decimal('0.0')
        for symbol, weight in weights.items():
            # Assign each asset a mock volatility
            asset_vol = Decimal(str(round(self._rng.uniform(0.12, 0.40), 3)))  # 12-40% volatility
            total_vol += weight * asset_vol
        
        return total_vol

    def _calculate_mock_max_drawdown(self, portfolio: Portfolio) -> Decimal:
        """Mock calculation of maximum drawdown"""
        # In reality, this would require historical portfolio values
        # For mock, generate a reasonable value
        return Decimal(str(round(self._rng.uniform(5, 25), 2)))  # 5-25% drawdown

    def _calculate_mock_volatility(self, portfolio: Portfolio) -> Decimal:
        """Mock calculation of portfolio volatility"""
        # Simulated annualized volatility
        return Decimal(str(round(self._rng.uniform(12, 30), 2)))  # 12-30% volatility

    def _calculate_mock_beta(self, portfolio: Portfolio) -> Decimal:
        """Mock calculation of portfolio beta"""
        # Simulated beta relative to market (SPY)
        return Decimal(str(round(self._rng.uniform(0.8, 1.5), 3)))  # 0.8-1.5 beta

    def _calculate_mock_sharpe_ratio(self, portfolio: Portfolio) -> Decimal:
        """Mock calculation of Sharpe ratio"""
        # Simulated Sharpe ratio (risk-adjusted return)
        return Decimal(str(round(self._rng.uniform(0.5, 2.0), 3)))  # 0.5-2.0 Sharpe

    def _calculate_mock_sortino_ratio(self, portfolio: Portfolio) -> Decimal:
        """Mock calculation of Sortino ratio"""
        # Simulated Sortino ratio (downside risk-adjusted return)
        return Decimal(str(round(self._rng.uniform(0.7, 2.5), 3)))  # 0.7-2.5 Sortino