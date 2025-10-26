"""
Advanced Risk Management System

This module implements sophisticated risk management functionality
beyond basic risk management, including advanced analytics, VaR,
correlation analysis, and dynamic risk controls.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings
from scipy import stats
from dataclasses import dataclass

from src.domain.entities.trading import Order, Position, Portfolio
from src.domain.entities.user import User, RiskTolerance
from src.domain.value_objects import Money, Symbol, Price
from src.domain.ports import MarketDataPort, NotificationPort
from src.domain.services.risk_management import RiskManager
from src.infrastructure.config.settings import settings


@dataclass
class RiskMetrics:
    """Data class for comprehensive risk metrics."""
    volatility: float
    value_at_risk: float
    conditional_value_at_risk: float
    beta: float
    alpha: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    correlation_matrix: Optional[np.array] = None
    stress_test_loss: Optional[float] = None
    concentration_risk: Optional[float] = None


class AdvancedRiskManager(RiskManager):
    """
    Extended risk manager with advanced analytics and risk metrics.
    """
    
    def __init__(self, notification_service: NotificationPort):
        super().__init__(notification_service)
        self.lookback_period = 252  # 1 year of trading days
        self.confidence_level = 0.95
        self.stress_scenarios = [
            {'name': '2008 Financial Crisis', 'shock': -0.30},
            {'name': '2020 COVID Crash', 'shock': -0.35},
            {'name': 'Dot-com Bubble', 'shock': -0.25}
        ]
    
    def calculate_comprehensive_risk_metrics(
        self, 
        portfolio: Portfolio, 
        market_data_service: MarketDataPort
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for the portfolio.
        """
        # Get historical data for all positions in the portfolio
        symbols = [str(pos.symbol) for pos in portfolio.positions]
        if not symbols:
            return RiskMetrics(
                volatility=0.0,
                value_at_risk=0.0,
                conditional_value_at_risk=0.0,
                beta=0.0,
                alpha=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0
            )
        
        # Fetch historical returns for portfolio components
        portfolio_returns = self._calculate_portfolio_returns(portfolio, market_data_service)
        
        # Calculate metrics
        volatility = self._calculate_volatility(portfolio_returns)
        var_95 = self._calculate_value_at_risk(portfolio_returns, confidence_level=0.95)
        cvar_95 = self._calculate_conditional_var(portfolio_returns, confidence_level=0.95)
        beta = self._calculate_beta(portfolio_returns)
        alpha = self._calculate_alpha(portfolio_returns)
        sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns)
        sortino_ratio = self._calculate_sortino_ratio(portfolio_returns)
        max_drawdown = self._calculate_max_drawdown(portfolio_returns)
        correlation_matrix = self._calculate_correlation_matrix(portfolio, market_data_service)
        stress_test_loss = self._perform_stress_test(portfolio, market_data_service)
        concentration_risk = self._calculate_concentration_risk(portfolio)
        
        return RiskMetrics(
            volatility=volatility,
            value_at_risk=var_95,
            conditional_value_at_risk=cvar_95,
            beta=beta,
            alpha=alpha,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            correlation_matrix=correlation_matrix,
            stress_test_loss=stress_test_loss,
            concentration_risk=concentration_risk
        )
    
    def _calculate_portfolio_returns(self, portfolio: Portfolio, market_data_service: MarketDataPort) -> List[float]:
        """
        Calculate historical portfolio returns.
        """
        # In a real implementation, this would fetch historical prices for all positions
        # and calculate portfolio returns based on position weights over time
        # For now, generating mock returns
        np.random.seed(42)  # For reproducible results
        returns = np.random.normal(0.0005, 0.02, self.lookback_period).tolist()  # Daily returns ~ 12% annual
        return returns
    
    def _calculate_volatility(self, returns: List[float]) -> float:
        """
        Calculate portfolio volatility (annualized standard deviation).
        """
        if not returns:
            return 0.0
        
        # Calculate standard deviation of returns
        std_dev = np.std(returns)
        
        # Annualize volatility (assuming 252 trading days per year)
        annualized_vol = std_dev * np.sqrt(252)
        
        return float(annualized_vol)
    
    def _calculate_value_at_risk(self, returns: List[float], confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR) using historical method.
        """
        if not returns:
            return 0.0
        
        # Sort returns
        sorted_returns = sorted(returns)
        
        # Calculate the index for the confidence level
        var_index = int((1 - confidence_level) * len(sorted_returns))
        
        # Get the VaR value
        var = sorted_returns[var_index] if var_index < len(sorted_returns) else sorted_returns[0]
        
        # Convert to portfolio value terms
        portfolio_value = 100000.0  # Placeholder - would use actual portfolio value
        var_value = abs(var) * portfolio_value
        
        return var_value
    
    def _calculate_conditional_var(self, returns: List[float], confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
        """
        if not returns:
            return 0.0
        
        # Sort returns
        sorted_returns = sorted(returns)
        
        # Calculate the index for the confidence level
        var_index = int((1 - confidence_level) * len(sorted_returns))
        
        # Get the worst returns beyond the VaR threshold
        tail_returns = sorted_returns[:var_index]
        
        if not tail_returns:
            return 0.0
        
        # Calculate expected shortfall (average of tail returns)
        cvar = np.mean(tail_returns)
        
        # Convert to portfolio value terms
        portfolio_value = 100000.0  # Placeholder - would use actual portfolio value
        cvar_value = abs(cvar) * portfolio_value
        
        return cvar_value
    
    def _calculate_beta(self, portfolio_returns: List[float]) -> float:
        """
        Calculate portfolio beta relative to market.
        """
        if not portfolio_returns:
            return 0.0
        
        # For simplicity, we'll use mock market returns
        # In reality, this would compare against market index returns
        market_returns = np.random.normal(0.0003, 0.015, len(portfolio_returns)).tolist()
        
        if len(portfolio_returns) < 2 or len(market_returns) < 2:
            return 1.0
        
        # Calculate covariance and variance
        cov_matrix = np.cov(portfolio_returns, market_returns)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 0.0
        
        return float(beta)
    
    def _calculate_alpha(self, portfolio_returns: List[float]) -> float:
        """
        Calculate portfolio alpha relative to market.
        """
        # Alpha = Portfolio Return - Risk-Free Rate - Beta * (Market Return - Risk-Free Rate)
        # Simplified calculation
        portfolio_return = np.mean(portfolio_returns) * 252  # Annualize
        risk_free_rate = 0.02  # 2% risk-free rate
        
        alpha = portfolio_return - risk_free_rate
        return float(alpha)
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio.
        """
        if not returns:
            return 0.0
        
        excess_returns = [(r * 252) - risk_free_rate for r in returns]  # Annualize returns
        mean_excess_return = np.mean(excess_returns)
        std_excess_return = np.std(excess_returns)
        
        if std_excess_return == 0:
            return 0.0
        
        return float(mean_excess_return / std_excess_return)
    
    def _calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino ratio (uses downside deviation instead of total volatility).
        """
        if not returns:
            return 0.0
        
        # Calculate excess returns
        excess_returns = [(r * 252) - risk_free_rate for r in returns]
        mean_excess_return = np.mean(excess_returns)
        
        # Calculate downside deviation
        negative_returns = [r for r in excess_returns if r < 0]
        if negative_returns:
            downside_deviation = np.std(negative_returns)
        else:
            downside_deviation = 0.0
        
        if downside_deviation == 0:
            return 0.0
        
        return float(mean_excess_return / downside_deviation)
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """
        Calculate maximum drawdown.
        """
        if not returns:
            return 0.0
        
        # Calculate cumulative returns
        cumulative_returns = [0]
        for r in returns:
            cumulative_returns.append(cumulative_returns[-1] + r)
        
        # Calculate drawdowns
        running_max = [cumulative_returns[0]]
        drawdowns = [0]
        
        for i in range(1, len(cumulative_returns)):
            if cumulative_returns[i] > running_max[-1]:
                running_max.append(cumulative_returns[i])
            else:
                running_max.append(running_max[-1])
            
            drawdown = (running_max[-1] - cumulative_returns[i]) / (running_max[-1] + 1e-8)  # Avoid division by zero
            drawdowns.append(drawdown)
        
        max_drawdown = max(drawdowns) if drawdowns else 0.0
        return float(max_drawdown)
    
    def _calculate_correlation_matrix(self, portfolio: Portfolio, market_data_service: MarketDataPort) -> Optional[np.array]:
        """
        Calculate correlation matrix between portfolio components.
        """
        # This would calculate correlations between all positions
        # For now, returning a mock correlation matrix
        n_positions = len(portfolio.positions)
        if n_positions == 0:
            return None
        
        # Create a mock correlation matrix
        correlation_matrix = np.eye(n_positions)  # Identity matrix as placeholder
        for i in range(n_positions):
            for j in range(i+1, n_positions):
                # Add some correlation between different assets
                corr = np.random.uniform(0.1, 0.6)
                correlation_matrix[i, j] = corr
                correlation_matrix[j, i] = corr
        
        return correlation_matrix
    
    def _perform_stress_test(self, portfolio: Portfolio, market_data_service: MarketDataPort) -> float:
        """
        Perform stress testing on the portfolio.
        """
        # Apply different scenarios and measure impact
        max_loss = 0.0
        
        for scenario in self.stress_scenarios:
            # Calculate potential loss under this scenario
            scenario_loss = self._apply_scenario(portfolio, scenario['shock'])
            if scenario_loss > max_loss:
                max_loss = scenario_loss
        
        return max_loss
    
    def _apply_scenario(self, portfolio: Portfolio, market_shock: float) -> float:
        """
        Apply a market shock scenario to the portfolio.
        """
        # Simplified: apply shock to each position proportionally
        total_loss = 0.0
        for position in portfolio.positions:
            position_impact = float(position.market_value.amount) * market_shock
            total_loss += abs(position_impact)
        
        return total_loss
    
    def _calculate_concentration_risk(self, portfolio: Portfolio) -> float:
        """
        Calculate concentration risk in the portfolio.
        """
        if not portfolio.positions or float(portfolio.total_value.amount) == 0:
            return 0.0
        
        # Calculate position weights and concentration risk
        position_weights = []
        for position in portfolio.positions:
            weight = float(position.market_value.amount) / float(portfolio.total_value.amount)
            position_weights.append(weight)
        
        # Concentration risk measured by Herfindahl-Hirschman Index (HHI)
        hhi = sum(w*w for w in position_weights)
        
        # Normalize to 0-1 scale (HHI normally ranges from 1/n to 1)
        n = len(position_weights)
        if n > 1:
            normalized_hhi = (hhi - 1/n) / (1 - 1/n)
        else:
            normalized_hhi = 1.0  # Fully concentrated portfolio
        
        return float(normalized_hhi)
    
    def get_dynamic_stop_losses(self, portfolio: Portfolio, risk_metrics: RiskMetrics) -> Dict[str, float]:
        """
        Calculate dynamic stop-loss levels based on risk metrics.
        """
        stop_losses = {}
        
        for position in portfolio.positions:
            # Calculate stop-loss based on volatility and risk tolerance
            current_price = float(position.current_price.amount)
            
            # Base stop-loss percentage based on portfolio volatility
            base_stop_pct = risk_metrics.volatility * 0.5  # Use half the volatility
            
            # Adjust based on user's risk tolerance
            if position.user.risk_tolerance == RiskTolerance.CONSERVATIVE:
                stop_pct = min(base_stop_pct, 0.08)  # Max 8%
            elif position.user.risk_tolerance == RiskTolerance.MODERATE:
                stop_pct = min(base_stop_pct, 0.12)  # Max 12%
            else:  # AGGRESSIVE
                stop_pct = min(base_stop_pct, 0.20)  # Max 20%
            
            stop_price = current_price * (1 - stop_pct)
            stop_losses[str(position.symbol)] = stop_price
        
        return stop_losses
    
    def generate_risk_report(self, portfolio: Portfolio, risk_metrics: RiskMetrics) -> str:
        """
        Generate a comprehensive risk report.
        """
        report = f"""
        ADVANCED RISK REPORT
        ===================
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Portfolio Value: ${portfolio.total_value.amount:,.2f}
        
        RISK METRICS:
        - Volatility (Annualized): {risk_metrics.volatility:.2%}
        - Value at Risk (95%): ${risk_metrics.value_at_risk:,.2f}
        - Conditional VaR (95%): ${risk_metrics.conditional_value_at_risk:,.2f}
        - Beta: {risk_metrics.beta:.2f}
        - Alpha: {risk_metrics.alpha:.2%}
        - Sharpe Ratio: {risk_metrics.sharpe_ratio:.2f}
        - Sortino Ratio: {risk_metrics.sortino_ratio:.2f}
        - Max Drawdown: {risk_metrics.max_drawdown:.2%}
        - Concentration Risk: {risk_metrics.concentration_risk:.2%}
        - Stress Test Max Loss: ${risk_metrics.stress_test_loss:,.2f}
        
        RECOMMENDATIONS:
        """
        
        # Risk level assessment
        risk_level = self._assess_risk_level(risk_metrics)
        report += f"- Overall Risk Level: {risk_level}\n"
        
        # Specific recommendations based on metrics
        if risk_metrics.volatility > 0.25:
            report += "- High volatility detected: Consider diversification\n"
        if risk_metrics.max_drawdown > 0.20:
            report += "- Significant drawdown risk: Review stop-loss strategies\n"
        if risk_metrics.concentration_risk > 0.5:
            report += "- High concentration risk: Diversify holdings\n"
        if risk_metrics.beta > 1.2:
            report += "- Portfolio is more volatile than market: Consider hedging\n"
        if risk_metrics.sharpe_ratio < 0.5:
            report += "- Poor risk-adjusted returns: Review strategy\n"
        
        return report
    
    def _assess_risk_level(self, risk_metrics: RiskMetrics) -> str:
        """
        Assess overall portfolio risk level.
        """
        # Weighted risk score (0-10 scale)
        score = 0
        
        # Higher volatility = higher risk
        score += min(risk_metrics.volatility * 20, 3)  # Max 3 points
        
        # Higher VaR = higher risk
        score += min(risk_metrics.value_at_risk / 10000, 2)  # Max 2 points (assuming $100k portfolio)
        
        # Higher max drawdown = higher risk
        score += min(risk_metrics.max_drawdown * 10, 2)  # Max 2 points
        
        # Higher concentration = higher risk
        score += risk_metrics.concentration_risk * 3  # Max 3 points
        
        # Lower Sharpe ratio = higher risk
        if risk_metrics.sharpe_ratio < 0.5:
            score += 2 - (risk_metrics.sharpe_ratio * 4)  # Max 2 points
        
        # Risk level based on score
        if score < 3:
            return "LOW"
        elif score < 6:
            return "MODERATE"
        elif score < 8:
            return "HIGH"
        else:
            return "VERY HIGH"


class DynamicRiskController:
    """
    Controller for implementing dynamic risk controls based on market conditions.
    """
    
    def __init__(self, advanced_risk_manager: AdvancedRiskManager):
        self.risk_manager = advanced_risk_manager
        self.market_regime = 'normal'  # 'bull', 'bear', 'high_volatility', 'low_volatility', 'normal'
        self.position_size_limits = {
            'bull': 0.15,  # 15% max per position
            'bear': 0.05,  # 5% max per position
            'high_volatility': 0.07,  # 7% max per position
            'low_volatility': 0.12,  # 12% max per position
            'normal': 0.10  # 10% max per position
        }
    
    def adjust_risk_controls(self, market_data_service: MarketDataPort) -> Dict[str, Any]:
        """
        Adjust risk controls based on current market conditions.
        """
        # Detect market regime (simplified implementation)
        self._detect_market_regime(market_data_service)
        
        # Get current position size limit based on regime
        position_limit = self.position_size_limits[self.market_regime]
        
        # Adjust risk parameters
        risk_adjustments = {
            'position_size_limit': position_limit,
            'volatility_threshold': self._get_volatility_threshold(),
            'correlation_threshold': 0.7,  # Don't add assets with >70% correlation
            'stop_loss_multiplier': self._get_stop_loss_multiplier(),
            'leverage_limit': self._get_leverage_limit()
        }
        
        return risk_adjustments
    
    def _detect_market_regime(self, market_data_service: MarketDataPort):
        """
        Detect current market regime based on various indicators.
        """
        # This would analyze market indicators to determine regime
        # For simplicity, we'll use mock logic
        import random
        
        regime_options = ['bull', 'bear', 'high_volatility', 'low_volatility', 'normal']
        self.market_regime = random.choice(regime_options)
    
    def _get_volatility_threshold(self) -> float:
        """
        Get volatility threshold based on market regime.
        """
        threshold_map = {
            'bull': 0.30,  # High threshold in bull market
            'bear': 0.15,  # Low threshold in bear market
            'high_volatility': 0.20,
            'low_volatility': 0.40,
            'normal': 0.25
        }
        return threshold_map.get(self.market_regime, 0.25)
    
    def _get_stop_loss_multiplier(self) -> float:
        """
        Get stop-loss multiplier based on market regime.
        """
        multiplier_map = {
            'bull': 1.0,  # Normal stop-losses
            'bear': 0.8,  # Tighter stop-losses
            'high_volatility': 1.2,  # Wider stop-losses
            'low_volatility': 0.9,  # Slightly tighter
            'normal': 1.0
        }
        return multiplier_map.get(self.market_regime, 1.0)
    
    def _get_leverage_limit(self) -> float:
        """
        Get leverage limit based on market regime.
        """
        leverage_map = {
            'bull': 2.0,  # Higher leverage allowed
            'bear': 0.5,  # Lower leverage allowed
            'high_volatility': 0.75,
            'low_volatility': 1.5,
            'normal': 1.0
        }
        return leverage_map.get(self.market_regime, 1.0)


class TailRiskHedgeService:
    """
    Service for implementing tail risk hedging strategies.
    """
    
    def __init__(self, market_data_service: MarketDataPort):
        self.market_data_service = market_data_service
        self.hedge_strategies = {
            'put_protection': self._put_protection_hedge,
            'inverse_etf': self._inverse_etf_hedge,
            'volatility_etf': self._volatility_etf_hedge
        }
    
    def recommend_hedging_strategy(self, portfolio: Portfolio, risk_metrics: RiskMetrics) -> Dict[str, Any]:
        """
        Recommend a hedging strategy based on portfolio risk profile.
        """
        recommendations = []
        
        # Check if portfolio needs hedging
        if risk_metrics.max_drawdown > 0.15 or risk_metrics.volatility > 0.25:
            # Recommend put protection for equity-heavy portfolios
            equity_positions = [p for p in portfolio.positions if self._is_equity(p.symbol)]
            if len(equity_positions) > len(portfolio.positions) * 0.5:  # More than 50% equity
                put_hedge = self.hedge_strategies['put_protection'](portfolio, risk_metrics)
                recommendations.append(put_hedge)
        
        # Check for high correlation risk
        if risk_metrics.concentration_risk > 0.6:
            # Recommend inverse ETF hedge
            inverse_hedge = self.hedge_strategies['inverse_etf'](portfolio, risk_metrics)
            recommendations.append(inverse_hedge)
        
        # Check for high volatility environment
        if risk_metrics.volatility > 0.30:
            # Recommend volatility hedge
            vol_hedge = self.hedge_strategies['volatility_etf'](portfolio, risk_metrics)
            recommendations.append(vol_hedge)
        
        return {
            'recommended_hedges': recommendations,
            'implementation_cost': sum(r.get('cost', 0) for r in recommendations),
            'risk_reduction': self._estimate_risk_reduction(recommendations, risk_metrics)
        }
    
    def _is_equity(self, symbol: Symbol) -> bool:
        """
        Determine if a symbol represents an equity instrument.
        """
        # Simplified check - in reality, this would look up security type
        return True
    
    def _put_protection_hedge(self, portfolio: Portfolio, risk_metrics: RiskMetrics) -> Dict[str, Any]:
        """
        Recommend put protection strategy.
        """
        # Calculate how many puts to buy based on portfolio value and risk
        portfolio_value = float(portfolio.total_value.amount)
        put_size = min(portfolio_value * 0.10, 100000)  # Max $100k in puts
        
        return {
            'strategy': 'put_protection',
            'instrument': 'SPY puts ATM',
            'size': put_size,
            'cost': put_size * 0.03,  # 3% premium
            'protection_level': '20%',
            'estimated_benefit': 'Reduces portfolio loss by up to 20% in crash'
        }
    
    def _inverse_etf_hedge(self, portfolio: Portfolio, risk_metrics: RiskMetrics) -> Dict[str, Any]:
        """
        Recommend inverse ETF strategy.
        """
        portfolio_value = float(portfolio.total_value.amount)
        hedge_size = portfolio_value * 0.15  # 15% hedge
        
        return {
            'strategy': 'inverse_etf',
            'instrument': 'SH, SDS, etc.',
            'size': hedge_size,
            'cost': hedge_size * 0.01,  # 1% annual fee
            'protection_level': 'Market directional',
            'estimated_benefit': 'Benefits in broad market decline'
        }
    
    def _volatility_etf_hedge(self, portfolio: Portfolio, risk_metrics: RiskMetrics) -> Dict[str, Any]:
        """
        Recommend volatility ETF strategy (e.g., VXX for long volatility).
        """
        portfolio_value = float(portfolio.total_value.amount)
        hedge_size = portfolio_value * 0.05  # 5% allocation to vol
        
        return {
            'strategy': 'volatility_etf',
            'instrument': 'VXX, UVXY for long vol',
            'size': hedge_size,
            'cost': hedge_size * 0.10,  # Higher cost due to contango
            'protection_level': 'Volatility spike',
            'estimated_benefit': 'Benefits during volatility spikes'
        }
    
    def _estimate_risk_reduction(self, recommendations: List[Dict], original_risk: RiskMetrics) -> Dict[str, float]:
        """
        Estimate risk reduction from hedging strategies.
        """
        reductions = {
            'volatility_reduction': 0.0,
            'var_reduction': 0.0,
            'max_drawdown_reduction': 0.0
        }
        
        for rec in recommendations:
            if rec['strategy'] == 'put_protection':
                reductions['volatility_reduction'] += 0.05
                reductions['max_drawdown_reduction'] += 0.10
            elif rec['strategy'] == 'inverse_etf':
                reductions['volatility_reduction'] += 0.03
                reductions['max_drawdown_reduction'] += 0.05
            elif rec['strategy'] == 'volatility_etf':
                reductions['max_drawdown_reduction'] += 0.03
        
        return reductions


# Initialize the advanced risk manager
advanced_risk_manager = AdvancedRiskManager(None)  # Notification service would be injected in real implementation