"""
Portfolio Optimization Algorithms

This module implements advanced portfolio optimization techniques
including Modern Portfolio Theory, Black-Litterman, and AI-enhanced methods.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import cvxpy as cp

from src.domain.entities.trading import Position, Portfolio
from src.domain.entities.user import User, InvestmentGoal
from src.domain.value_objects import Money, Symbol, Price
from src.domain.ports import MarketDataPort
from src.domain.services.trading import PortfolioOptimizationDomainService


@dataclass
class PortfolioWeights:
    """Data class to hold portfolio weights."""
    symbol: str
    weight: float
    expected_return: float
    risk_contribution: float


class ModernPortfolioTheoryOptimizer:
    """
    Implementation of Modern Portfolio Theory (MPT) for portfolio optimization.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def calculate_efficient_frontier(
        self, 
        returns: pd.DataFrame, 
        target_returns: Optional[List[float]] = None
    ) -> List[Dict[str, any]]:
        """
        Calculate the efficient frontier for given asset returns.
        """
        if target_returns is None:
            target_returns = np.linspace(
                returns.mean().min(), 
                returns.mean().max(), 
                num=50
            ).tolist()
        
        efficient_portfolios = []
        
        for target_return in target_returns:
            weights = self._optimize_for_target_return(returns, target_return)
            if weights is not None:
                portfolio_return, portfolio_risk = self._calculate_portfolio_metrics(
                    returns, weights
                )
                efficient_portfolios.append({
                    'weights': weights,
                    'expected_return': portfolio_return,
                    'risk': portfolio_risk,
                    'sharpe_ratio': (portfolio_return - self.risk_free_rate) / portfolio_risk
                })
        
        return efficient_portfolios
    
    def _optimize_for_target_return(self, returns: pd.DataFrame, target_return: float) -> Optional[np.array]:
        """
        Find portfolio weights that minimize risk for a given target return.
        """
        try:
            n_assets = len(returns.columns)
            mean_returns = returns.mean().values
            cov_matrix = returns.cov().values
            
            # Define optimization variables
            weights = cp.Variable(n_assets)
            
            # Define constraints
            constraints = [
                cp.sum(weights) == 1,  # Weights sum to 1
                mean_returns @ weights >= target_return,  # Target return constraint
                weights >= 0  # Long-only constraint
            ]
            
            # Define objective (minimize variance)
            objective = cp.Minimize(cp.quad_form(weights, cov_matrix))
            
            # Create and solve problem
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if problem.status not in ["infeasible", "unbounded"]:
                return weights.value
            else:
                return None
        except:
            return None
    
    def _calculate_portfolio_metrics(self, returns: pd.DataFrame, weights: np.array) -> Tuple[float, float]:
        """
        Calculate expected return and risk for portfolio.
        """
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values
        
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        return portfolio_return, portfolio_risk
    
    def maximize_sharpe_ratio(self, returns: pd.DataFrame) -> Dict[str, any]:
        """
        Find portfolio that maximizes Sharpe ratio.
        """
        try:
            n_assets = len(returns.columns)
            mean_returns = returns.mean().values
            cov_matrix = returns.cov().values
            
            # Define optimization variables
            weights = cp.Variable(n_assets)
            k = cp.Variable(1)
            
            # Transform variables: x = w/z, z = 1/k
            x = cp.Variable(n_assets)
            
            # Define constraints
            constraints = [
                cp.sum((mean_returns - self.risk_free_rate) * x) == 1,
                cp.quad_form(x, cov_matrix) <= k,
                x >= 0  # Long-only constraint
            ]
            
            # Define objective (minimize k, which maximizes Sharpe ratio)
            objective = cp.Minimize(k)
            
            # Create and solve problem
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if problem.status not in ["infeasible", "unbounded"] and x.value is not None:
                weights_optimal = x.value / np.sum(x.value * (mean_returns - self.risk_free_rate))
                
                portfolio_return, portfolio_risk = self._calculate_portfolio_metrics(
                    returns, weights_optimal
                )
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
                
                return {
                    'weights': weights_optimal,
                    'expected_return': portfolio_return,
                    'risk': portfolio_risk,
                    'sharpe_ratio': sharpe_ratio
                }
            else:
                # Fallback: equal weighting
                equal_weights = np.ones(n_assets) / n_assets
                portfolio_return, portfolio_risk = self._calculate_portfolio_metrics(
                    returns, equal_weights
                )
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
                
                return {
                    'weights': equal_weights,
                    'expected_return': portfolio_return,
                    'risk': portfolio_risk,
                    'sharpe_ratio': sharpe_ratio
                }
        except:
            # Fallback: equal weighting
            n_assets = len(returns.columns)
            equal_weights = np.ones(n_assets) / n_assets
            portfolio_return, portfolio_risk = self._calculate_portfolio_metrics(
                returns, equal_weights
            )
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
            
            return {
                'weights': equal_weights,
                'expected_return': portfolio_return,
                'risk': portfolio_risk,
                'sharpe_ratio': sharpe_ratio
            }


class BlackLittermanOptimizer:
    """
    Black-Litterman model for portfolio optimization.
    """
    
    def __init__(self, delta: float = 2.5, tau: float = 0.05):
        self.delta = delta  # Risk aversion coefficient
        self.tau = tau      # Uncertainty parameter
    
    def optimize(
        self,
        market_caps: List[float],
        historical_returns: pd.DataFrame,
        views: Dict[str, float],  # {symbol: expected_return}
        confidences: Dict[str, float]  # {symbol: confidence_level}
    ) -> Dict[str, any]:
        """
        Optimize portfolio using Black-Litterman model.
        """
        # Calculate market equilibrium weights
        market_weights = market_caps / np.sum(market_caps)
        
        # Calculate covariance matrix
        cov_matrix = historical_returns.cov().values
        n_assets = len(cov_matrix)
        
        # Calculate implied equilibrium returns
        implied_returns = self.delta * cov_matrix @ market_weights
        
        # Prepare view matrices
        if views and confidences:
            view_symbols = list(views.keys())
            view_returns = np.array([views[sym] for sym in view_symbols])
            
            # Create pick matrix (for now, assume each view is about one asset)
            P = np.zeros((len(view_symbols), n_assets))
            for i, symbol in enumerate(view_symbols):
                if symbol in historical_returns.columns:
                    col_idx = list(historical_returns.columns).index(symbol)
                    P[i, col_idx] = 1
            
            # Confidence matrix
            conf_values = [confidences[sym] for sym in view_symbols]
            conf_matrix = np.diag(conf_values)
            
            # Calculate Black-Litterman expected returns
            try:
                tau_cov = self.tau * cov_matrix
                middle_term = P @ tau_cov @ P.T + conf_matrix
                omega = np.linalg.inv(middle_term)
                
                bl_returns = implied_returns + tau_cov @ P.T @ omega @ (
                    view_returns - P @ implied_returns
                )
                
                # Use MPT to find optimal weights with BL returns
                mpt_optimizer = ModernPortfolioTheoryOptimizer()
                # This is a simplification - in practice, you'd continue with BL math
                # For now, we'll use the returns to calculate optimal weights
                weights = self._calculate_equilibrium_weights(bl_returns, cov_matrix)
                
                portfolio_return, portfolio_risk = self._calculate_portfolio_metrics(
                    historical_returns, weights
                )
                
                return {
                    'weights': weights,
                    'expected_return': portfolio_return,
                    'risk': portfolio_risk,
                    'sharpe_ratio': (portfolio_return - 0.02) / portfolio_risk if portfolio_risk > 0 else 0
                }
            except:
                # Fallback to market weights
                portfolio_return, portfolio_risk = self._calculate_portfolio_metrics(
                    historical_returns, market_weights
                )
                
                return {
                    'weights': market_weights,
                    'expected_return': portfolio_return,
                    'risk': portfolio_risk,
                    'sharpe_ratio': (portfolio_return - 0.02) / portfolio_risk if portfolio_risk > 0 else 0
                }
        else:
            # Fallback to market equilibrium
            portfolio_return, portfolio_risk = self._calculate_portfolio_metrics(
                historical_returns, market_weights
            )
            
            return {
                'weights': market_weights,
                'expected_return': portfolio_return,
                'risk': portfolio_risk,
                'sharpe_ratio': (portfolio_return - 0.02) / portfolio_risk if portfolio_risk > 0 else 0
            }
    
    def _calculate_equilibrium_weights(self, expected_returns: np.array, cov_matrix: np.array) -> np.array:
        """
        Calculate equilibrium portfolio weights.
        """
        try:
            weights = (np.linalg.inv(cov_matrix) @ expected_returns) 
            weights = weights / np.sum(weights)
            return weights
        except:
            # Fallback to inverse volatility weighting
            volatilities = np.sqrt(np.diag(cov_matrix))
            weights = 1.0 / volatilities
            weights = weights / np.sum(weights)
            return weights
    
    def _calculate_portfolio_metrics(self, returns: pd.DataFrame, weights: np.array) -> Tuple[float, float]:
        """
        Calculate expected return and risk for portfolio.
        """
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values
        
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        return portfolio_return, portfolio_risk


class AIEnhancedOptimizer:
    """
    AI-enhanced portfolio optimization using machine learning predictions.
    """
    
    def __init__(self, market_data_service):
        self.market_data_service = market_data_service
        self.price_predictor = None  # Would be an instance of LSTMPricePredictor
    
    def optimize_with_ai_predictions(
        self,
        symbols: List[Symbol],
        user: User,
        current_portfolio: Portfolio
    ) -> Dict[str, any]:
        """
        Optimize portfolio using AI price predictions and risk management.
        """
        # Get historical data for all symbols
        historical_data = {}
        current_prices = {}
        
        for symbol in symbols:
            # Get recent historical data (last 252 trading days ~ 1 year)
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=365)  # Extra days to account for weekends/holidays
            
            try:
                # Get historical prices from market data service
                prices = self.market_data_service.get_historical_prices(symbol, start_date, end_date)
                historical_data[str(symbol)] = [float(p.amount) for p in prices]
                
                # Get current price
                current_price = self.market_data_service.get_current_price(symbol)
                if current_price:
                    current_prices[str(symbol)] = float(current_price.amount)
            except:
                # If we can't get data, skip this symbol
                continue
        
        if not historical_data:
            return self._fallback_optimization(symbols, current_portfolio)
        
        # Create returns dataframe
        returns_data = {}
        for symbol, prices in historical_data.items():
            if len(prices) > 1:
                returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
                returns_data[symbol] = returns
        
        if not returns_data:
            return self._fallback_optimization(symbols, current_portfolio)
        
        # Convert to DataFrame
        returns_df = pd.DataFrame(returns_data)
        
        # Apply different optimization based on user's investment goal
        if user.investment_goal == InvestmentGoal.CAPITAL_PRESERVATION:
            return self._conservative_optimization(returns_df, symbols, user, current_portfolio)
        elif user.investment_goal == InvestmentGoal.BALANCED_GROWTH:
            return self._balanced_optimization(returns_df, symbols, user, current_portfolio)
        elif user.investment_goal == InvestmentGoal.MAXIMUM_RETURNS:
            return self._aggressive_optimization(returns_df, symbols, user, current_portfolio)
        else:
            return self._balanced_optimization(returns_df, symbols, user, current_portfolio)
    
    def _conservative_optimization(
        self,
        returns_df: pd.DataFrame,
        symbols: List[Symbol],
        user: User,
        current_portfolio: Portfolio
    ) -> Dict[str, any]:
        """
        Conservative optimization - minimize risk while ensuring positive returns.
        """
        mpt_optimizer = ModernPortfolioTheoryOptimizer()
        
        # Find portfolio with minimum risk that meets return threshold
        target_return = 0.03  # 3% annual minimum
        min_risk_portfolio = mpt_optimizer._optimize_for_target_return(returns_df, target_return)
        
        if min_risk_portfolio is None:
            # If can't find portfolio with target return, maximize Sharpe ratio
            return mpt_optimizer.maximize_sharpe_ratio(returns_df)
        
        return {
            'weights': min_risk_portfolio,
            'symbols': list(returns_df.columns),
            'expected_return': float(returns_df.mean().values @ min_risk_portfolio),
            'risk': float(np.sqrt(min_risk_portfolio.T @ returns_df.cov().values @ min_risk_portfolio)),
            'sharpe_ratio': 0  # Placeholder
        }
    
    def _balanced_optimization(
        self,
        returns_df: pd.DataFrame,
        symbols: List[Symbol],
        user: User,
        current_portfolio: Portfolio
    ) -> Dict[str, any]:
        """
        Balanced optimization - balance risk and return.
        """
        mpt_optimizer = ModernPortfolioTheoryOptimizer()
        return mpt_optimizer.maximize_sharpe_ratio(returns_df)
    
    def _aggressive_optimization(
        self,
        returns_df: pd.DataFrame,
        symbols: List[Symbol],
        user: User,
        current_portfolio: Portfolio
    ) -> Dict[str, any]:
        """
        Aggressive optimization - maximize returns accepting higher risk.
        """
        # For aggressive optimization, we might want to look at higher moments
        # or use different risk measures, but for now we'll use MPT with different parameters
        mpt_optimizer = ModernPortfolioTheoryOptimizer()
        
        # In an aggressive portfolio, we might accept higher risk for higher returns
        # For now, we'll just use the Sharpe ratio maximizer
        return mpt_optimizer.maximize_sharpe_ratio(returns_df)
    
    def _fallback_optimization(
        self,
        symbols: List[Symbol],
        current_portfolio: Portfolio
    ) -> Dict[str, any]:
        """
        Fallback optimization if data is insufficient.
        """
        n_assets = len(symbols)
        if n_assets == 0:
            return {
                'weights': np.array([]),
                'symbols': [],
                'expected_return': 0.0,
                'risk': 0.0,
                'sharpe_ratio': 0.0
            }
        
        # Equal weight allocation
        weights = np.ones(n_assets) / n_assets
        symbols_list = [str(sym) for sym in symbols]
        
        return {
            'weights': weights,
            'symbols': symbols_list,
            'expected_return': 0.05,  # Assumed 5% annual return
            'risk': 0.15,  # Assumed 15% volatility
            'sharpe_ratio': 0.2  # Assumed Sharpe ratio
        }


class DefaultPortfolioOptimizationService(PortfolioOptimizationDomainService):
    """
    Default implementation of portfolio optimization domain service.
    """
    
    def __init__(self, market_data_service, ai_model_service):
        self.mpt_optimizer = ModernPortfolioTheoryOptimizer()
        self.black_litterman_optimizer = BlackLittermanOptimizer()
        self.ai_enhanced_optimizer = AIEnhancedOptimizer(market_data_service)
        self.market_data_service = market_data_service
        self.ai_model_service = ai_model_service
    
    def rebalance_portfolio(self, portfolio: Portfolio, user: User) -> List[object]:
        """
        Generate rebalancing orders for the portfolio based on target allocation.
        """
        # This would return a list of orders to rebalance the portfolio
        # For now, returning an empty list as a placeholder
        return []
    
    def optimize_allocation(self, portfolio: Portfolio, user: User) -> Dict[str, float]:
        """
        Optimize the portfolio allocation based on the user's risk tolerance and goals.
        """
        # Get symbols for assets in portfolio and watchlist
        symbols = set()
        
        # Add symbols from current positions
        for position in portfolio.positions:
            symbols.add(position.symbol)
        
        # In a real implementation, we'd also get symbols from user's watchlist
        # For now, we'll just use the current positions
        
        if not symbols:
            return {}
        
        try:
            # Use AI-enhanced optimization
            result = self.ai_enhanced_optimizer.optimize_with_ai_predictions(
                list(symbols), user, portfolio
            )
            
            # Convert to the expected format
            allocation = {}
            if 'weights' in result and 'symbols' in result:
                for i, symbol in enumerate(result['symbols']):
                    if i < len(result['weights']):
                        allocation[symbol] = float(result['weights'][i])
            
            return allocation
        except Exception as e:
            print(f"Error in portfolio optimization: {e}")
            # Return empty allocation if optimization fails
            return {}
    
    def calculate_risk_metrics(self, portfolio: Portfolio) -> Dict[str, float]:
        """
        Calculate risk metrics for the portfolio.
        """
        # Calculate basic risk metrics
        total_value = float(portfolio.total_value.amount)
        
        if total_value == 0:
            return {
                'volatility': 0,
                'value_at_risk': 0,
                'beta': 0,
                'sharpe_ratio': 0
            }
        
        # Calculate position weights
        position_weights = {}
        for position in portfolio.positions:
            position_value = float(position.market_value.amount)
            weight = position_value / total_value
            position_weights[str(position.symbol)] = weight
        
        # Placeholder calculations - in real implementation,
        # these would be calculated based on historical correlations and volatilities
        return {
            'volatility': 0.15,  # Placeholder: 15% annualized
            'value_at_risk': total_value * 0.05,  # Placeholder: 5% VaR
            'beta': 1.0,  # Placeholder: Market beta
            'sharpe_ratio': 0.8  # Placeholder Sharpe ratio
        }
    
    def calculate_expected_return(self, portfolio: Portfolio, allocation: Dict[str, float]) -> float:
        """
        Calculate expected return for the portfolio based on allocation.
        """
        # This would use predictions from AI models
        # For now, returning a placeholder value
        return 0.08  # 8% expected annual return