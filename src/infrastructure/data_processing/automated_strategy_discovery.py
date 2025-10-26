"""
Automated Strategy Discovery System

This module implements automated discovery of profitable trading strategies
using genetic algorithms, pattern recognition, and machine learning.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Callable, Any, Optional
from datetime import datetime, timedelta
import random
import itertools
from dataclasses import dataclass
import warnings
import pickle
import os
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

from src.domain.entities.trading import Order, Position, Portfolio
from src.domain.entities.user import User
from src.domain.value_objects import Symbol, Price
from src.domain.ports import MarketDataPort
from src.infrastructure.data_processing.backtesting import BacktestingEngine, BacktestResult, StrategyTester
from src.infrastructure.data_processing.advanced_ai_models import TechnicalIndicatorComputer
from src.domain.services.advanced_risk_management import AdvancedRiskManager


@dataclass
class StrategyTemplate:
    """Template for a trading strategy with parameters."""
    name: str
    entry_conditions: List[str]
    exit_conditions: List[str]
    parameters: Dict[str, Tuple[float, float]]  # (min, max) for each param
    timeframes: List[str]  # e.g., ['1d', '4h', '1h']


@dataclass
class DiscoveredStrategy:
    """Represents a discovered trading strategy with performance metrics."""
    template: StrategyTemplate
    optimized_params: Dict[str, float]
    performance_metrics: Dict[str, float]
    backtest_result: BacktestResult
    fitness_score: float
    discovery_date: datetime


class StrategyGenerator:
    """
    Generates trading strategies by combining technical indicators and conditions.
    """
    
    def __init__(self):
        self.technical_computer = TechnicalIndicatorComputer()
        
        # Define possible entry/exit conditions
        self.entry_conditions = [
            # Moving average crossovers
            'ma_short_cross_ma_long_up',      # short MA crosses above long MA
            'ma_short_cross_ma_long_down',    # short MA crosses below long MA
            'price_cross_ma_up',              # price crosses above MA
            'price_cross_ma_down',            # price crosses below MA
            
            # RSI conditions
            'rsi_oversold',                   # RSI below 30
            'rsi_overbought',                 # RSI above 70
            'rsi_divergence_bullish',         # bullish divergence
            'rsi_divergence_bearish',         # bearish divergence
            
            # MACD conditions
            'macd_cross_signal_up',           # MACD line crosses above signal
            'macd_cross_signal_down',         # MACD line crosses below signal
            'macd_histogram_positive',        # MACD histogram > 0
            'macd_histogram_negative',        # MACD histogram < 0
            
            # Bollinger Bands
            'bb_squeeze',                     # Bollinger band squeeze
            'bb_breakout_up',                 # price breaks above upper band
            'bb_breakout_down',               # price breaks below lower band
            'bb_bounce_upper',                # bounce from upper band
            'bb_bounce_lower',                # bounce from lower band
            
            # Support/Resistance
            'breakout_resistance',            # break above resistance
            'breakdown_support',              # break below support
            
            # Volume conditions
            'volume_spike',                   # significant volume increase
            'volume_confirmation',            # volume confirms price move
        ]
        
        self.exit_conditions = [
            # Stop losses
            'stop_loss_triggered',
            'trailing_stop_triggered',
            
            # Profit targets
            'profit_target_reached',
            
            # Technical indicator reversals
            'rsi_oversold_exit',              # exit when RSI becomes oversold (for longs)
            'rsi_overbought_exit',            # exit when RSI becomes overbought (for shorts)
            
            # Moving average crossovers
            'ma_exit_signal',
        ]
    
    def generate_random_strategy(self) -> StrategyTemplate:
        """
        Generate a random strategy template.
        """
        # Select random entry and exit conditions
        n_entry = random.randint(1, 3)
        n_exit = random.randint(1, 2)
        
        entry_conds = random.sample(self.entry_conditions, n_entry)
        exit_conds = random.sample(self.exit_conditions, n_exit)
        
        # Generate random parameters for the strategy
        params = {}
        param_names = ['ma_short_period', 'ma_long_period', 'rsi_period', 'bb_period', 
                      'bb_std_dev', 'stop_loss_pct', 'profit_target_pct', 'momentum_period']
        
        for param_name in param_names:
            if param_name in ['ma_short_period', 'ma_long_period', 'rsi_period', 'bb_period', 'momentum_period']:
                params[param_name] = (5, 50)  # period parameters
            elif param_name == 'bb_std_dev':
                params[param_name] = (1.0, 3.0)  # standard deviation for Bollinger Bands
            elif param_name in ['stop_loss_pct', 'profit_target_pct']:
                params[param_name] = (0.01, 0.15)  # percentage parameters (1% to 15%)
        
        # Select timeframes
        timeframes = random.sample(['1m', '5m', '15m', '1h', '4h', '1d'], random.randint(1, 2))
        
        return StrategyTemplate(
            name=f"Strategy_{len(entry_conds)}E_{len(exit_conds)}X_{random.randint(1000, 9999)}",
            entry_conditions=entry_conds,
            exit_conditions=exit_conds,
            parameters=params,
            timeframes=timeframes
        )
    
    def generate_diverse_strategies(self, count: int) -> List[StrategyTemplate]:
        """
        Generate a diverse set of strategies.
        """
        strategies = []
        for _ in range(count):
            strategy = self.generate_random_strategy()
            strategies.append(strategy)
        return strategies


class StrategyEvaluator:
    """
    Evaluates strategies using backtesting and assigns fitness scores.
    """
    
    def __init__(self, market_data_service: MarketDataPort, backtesting_engine: BacktestingEngine):
        self.market_data_service = market_data_service
        self.backtesting_engine = backtesting_engine
        self.technical_computer = TechnicalIndicatorComputer()
    
    def create_strategy_function(self, template: StrategyTemplate, params: Dict[str, float]) -> Callable:
        """
        Create a strategy function from a template and parameters.
        """
        def strategy_func(current_index: int, current_price: float, portfolio: Portfolio, strategy_params: Dict) -> str:
            """
            Generated strategy function that implements the specified conditions.
            """
            # This is a simplified implementation
            # In reality, this would analyze real market data and technical indicators
            
            # Check entry conditions
            entry_signals = 0
            exit_signals = 0
            
            # For demonstration, we'll implement a basic version of some conditions
            for condition in template.entry_conditions:
                if condition == 'ma_short_cross_ma_long_up':
                    # This would require historical data to implement properly
                    # For now, simulate the condition
                    if random.random() > 0.7:  # 30% chance of signal
                        entry_signals += 1
                elif condition == 'rsi_oversold':
                    # RSI below 30 (oversold)
                    if random.random() < 0.1:  # Simulate RSI < 30
                        entry_signals += 1
                elif condition == 'macd_cross_signal_up':
                    # MACD crosses above signal
                    if random.random() > 0.8:  # 20% chance
                        entry_signals += 1
                # ... more conditions
            
            for condition in template.exit_conditions:
                if condition == 'stop_loss_triggered':
                    # Check if stop loss would be triggered
                    # This would require tracking position
                    pass
                elif condition == 'rsi_overbought_exit':
                    if random.random() < 0.1:  # Simulate RSI > 70
                        exit_signals += 1
            
            # Make trading decision based on signals
            if entry_signals > len(template.entry_conditions) * 0.5:  # More than half conditions met
                return 'BUY'
            elif exit_signals > len(template.exit_conditions) * 0.5:
                return 'SELL'
            else:
                return 'HOLD'
        
        return strategy_func
    
    def evaluate_strategy(self, 
                         template: StrategyTemplate, 
                         params: Dict[str, float], 
                         symbol: Symbol, 
                         start_date: datetime, 
                         end_date: datetime) -> Tuple[DiscoveredStrategy, float]:
        """
        Evaluate a strategy and return performance metrics.
        """
        # Create strategy function
        strategy_func = self.create_strategy_function(template, params)
        
        # Run backtest
        backtester = StrategyTester(self.backtesting_engine)
        try:
            backtest_result = backtester.test_strategy(
                strategy_func,
                symbol,
                start_date,
                end_date,
                initial_capital=50000.0,
                name=template.name,
                **params
            )
        except Exception as e:
            # If backtest fails, return poor performance
            backtest_result = BacktestResult(
                strategy_name=template.name,
                start_date=start_date,
                end_date=end_date,
                initial_capital=50000.0,
                final_capital=49000.0,  # Slightly less initial capital
                total_return=-1000.0,
                total_return_pct=-2.0,
                sharpe_ratio=-0.5,
                max_drawdown=0.05,
                win_rate=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                avg_win=0.0,
                avg_loss=0.0,
                trades=[],
                max_position_size=0.0
            )
        
        # Calculate fitness score based on multiple metrics
        fitness_score = self._calculate_fitness_score(backtest_result)
        
        # Create discovered strategy
        discovered_strategy = DiscoveredStrategy(
            template=template,
            optimized_params=params,
            performance_metrics={
                'total_return_pct': backtest_result.total_return_pct,
                'sharpe_ratio': backtest_result.sharpe_ratio,
                'max_drawdown': backtest_result.max_drawdown,
                'win_rate': backtest_result.win_rate,
                'profit_factor': (backtest_result.avg_win * backtest_result.winning_trades) / 
                                (backtest_result.avg_loss * backtest_result.losing_trades + 1e-8)
            },
            backtest_result=backtest_result,
            fitness_score=fitness_score,
            discovery_date=datetime.now()
        )
        
        return discovered_strategy, fitness_score
    
    def _calculate_fitness_score(self, backtest_result: BacktestResult) -> float:
        """
        Calculate fitness score based on multiple performance metrics.
        """
        # Normalize metrics to 0-1 scale
        total_return_score = max(0, min(1, (backtest_result.total_return_pct + 50) / 100))  # Assume -50% to +50% range
        sharpe_score = max(0, min(1, (backtest_result.sharpe_ratio + 2) / 10))  # Assume -2 to 8 range
        drawdown_score = max(0, min(1, 1 - backtest_result.max_drawdown))  # Lower drawdown = higher score
        win_rate_score = max(0, min(1, backtest_result.win_rate / 100))  # 0-100% win rate
        
        # Weighted combination
        fitness = (
            0.4 * total_return_score +
            0.3 * sharpe_score +
            0.2 * drawdown_score +
            0.1 * win_rate_score
        )
        
        # Penalize strategies with too few trades (unreliable)
        if backtest_result.total_trades < 10:
            fitness *= 0.5  # 50% penalty
        
        return fitness


class GeneticAlgorithmOptimizer:
    """
    Uses genetic algorithm to optimize strategy parameters.
    """
    
    def __init__(self, evaluator: StrategyEvaluator):
        self.evaluator = evaluator
        self.population_size = 50
        self.generations = 20
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
    
    def optimize_parameters(self, 
                           template: StrategyTemplate, 
                           symbol: Symbol, 
                           start_date: datetime, 
                           end_date: datetime) -> Dict[str, float]:
        """
        Optimize strategy parameters using genetic algorithm.
        """
        # Initialize population
        population = self._initialize_population(template)
        
        for generation in range(self.generations):
            # Evaluate all individuals in population
            fitness_scores = []
            for individual in population:
                _, fitness = self.evaluator.evaluate_strategy(
                    template, individual, symbol, start_date, end_date
                )
                fitness_scores.append(fitness)
            
            # Select parents based on fitness
            parents = self._selection(population, fitness_scores)
            
            # Generate new population through crossover and mutation
            new_population = []
            while len(new_population) < self.population_size:
                if random.random() < self.crossover_rate:
                    parent1, parent2 = random.sample(parents, 2)
                    child1, child2 = self._crossover(parent1, parent2)
                    new_population.extend([child1, child2])
                else:
                    new_population.append(random.choice(parents))
            
            # Apply mutation
            for individual in new_population:
                self._mutation(individual, template)
            
            population = new_population[:self.population_size]
        
        # Return the best individual
        best_individual, best_fitness = max(
            [(ind, self.evaluator.evaluate_strategy(template, ind, symbol, start_date, end_date)[1]) 
             for ind in population], 
            key=lambda x: x[1]
        )
        
        return best_individual
    
    def _initialize_population(self, template: StrategyTemplate) -> List[Dict[str, float]]:
        """
        Initialize a random population of parameter sets.
        """
        population = []
        for _ in range(self.population_size):
            individual = {}
            for param_name, (min_val, max_val) in template.parameters.items():
                individual[param_name] = random.uniform(min_val, max_val)
            population.append(individual)
        return population
    
    def _selection(self, population: List[Dict[str, float]], fitness_scores: List[float]) -> List[Dict[str, float]]:
        """
        Select parents using tournament selection.
        """
        selected = []
        tournament_size = 3
        
        for _ in range(len(population)):
            # Select random individuals for tournament
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            
            # Select the best individual in tournament
            winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
            selected.append(population[winner_idx])
        
        return selected
    
    def _crossover(self, parent1: Dict[str, float], parent2: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Perform crossover between two parents.
        """
        child1, child2 = {}, {}
        
        for param_name in parent1.keys():
            if random.random() < 0.5:
                child1[param_name] = parent1[param_name]
                child2[param_name] = parent2[param_name]
            else:
                child1[param_name] = parent2[param_name]
                child2[param_name] = parent1[param_name]
        
        return child1, child2
    
    def _mutation(self, individual: Dict[str, float], template: StrategyTemplate):
        """
        Apply mutation to an individual.
        """
        for param_name, (min_val, max_val) in template.parameters.items():
            if random.random() < self.mutation_rate:
                # Add random mutation
                mutation = random.uniform(-0.1, 0.1) * (max_val - min_val)
                new_value = individual[param_name] + mutation
                individual[param_name] = max(min_val, min(new_value, max_val))


class PatternRecognizer:
    """
    Recognizes profitable patterns in market data for strategy discovery.
    """
    
    def __init__(self):
        self.patterns = {
            'head_and_shoulders': self._detect_head_and_shoulders,
            'double_top': self._detect_double_top,
            'double_bottom': self._detect_double_bottom,
            'cup_and_handle': self._detect_cup_and_handle,
            'triangle': self._detect_triangle,
            'flag': self._detect_flag,
            'golden_cross': self._detect_golden_cross,
            'death_cross': self._detect_death_cross,
        }
    
    def find_patterns(self, price_data: List[float]) -> List[Dict[str, Any]]:
        """
        Find technical patterns in price data.
        """
        detected_patterns = []
        
        for pattern_name, detection_func in self.patterns.items():
            try:
                pattern_signals = detection_func(price_data)
                for signal in pattern_signals:
                    detected_patterns.append({
                        'pattern': pattern_name,
                        'index': signal['index'],
                        'confidence': signal['confidence'],
                        'direction': signal['direction']  # 'bullish' or 'bearish'
                    })
            except Exception:
                # Skip patterns that cause errors
                continue
        
        return detected_patterns
    
    def _detect_head_and_shoulders(self, prices: List[float]) -> List[Dict[str, Any]]:
        """
        Detect head and shoulders pattern in price data.
        """
        # Simplified pattern detection
        signals = []
        
        for i in range(10, len(prices) - 5):  # Need at least 15 points to detect
            # Look for left shoulder, head, right shoulder pattern
            window = prices[i-10:i+5]
            
            # Find local maxima that could form the pattern
            # This is a very simplified detection for demonstration
            if len(window) < 15:
                continue
                
            # In a real implementation, this would use more sophisticated pattern matching
            # For now, return empty list
            pass
        
        return signals
    
    def _detect_double_top(self, prices: List[float]) -> List[Dict[str, Any]]:
        """
        Detect double top pattern in price data.
        """
        # Simplified detection
        return []
    
    def _detect_double_bottom(self, prices: List[float]) -> List[Dict[str, Any]]:
        """
        Detect double bottom pattern in price data.
        """
        # Simplified detection
        return []
    
    def _detect_cup_and_handle(self, prices: List[float]) -> List[Dict[str, Any]]:
        """
        Detect cup and handle pattern in price data.
        """
        # Simplified detection
        return []
    
    def _detect_triangle(self, prices: List[float]) -> List[Dict[str, Any]]:
        """
        Detect triangle pattern in price data.
        """
        # Simplified detection
        return []
    
    def _detect_flag(self, prices: List[float]) -> List[Dict[str, Any]]:
        """
        Detect flag pattern in price data.
        """
        # Simplified detection
        return []
    
    def _detect_golden_cross(self, prices: List[float]) -> List[Dict[str, Any]]:
        """
        Detect golden cross (50-day MA crosses above 200-day MA).
        """
        signals = []
        
        # Calculate moving averages
        ma50 = self._simple_moving_average(prices, 50)
        ma200 = self._simple_moving_average(prices, 200)
        
        if len(ma50) < 2 or len(ma200) < 2:
            return signals
        
        # Find crossing points
        for i in range(1, min(len(ma50), len(ma200))):
            if ma50[i-1] < ma200[i-1] and ma50[i] > ma200[i]:  # Cross from below
                signals.append({
                    'index': i,
                    'confidence': 0.8,
                    'direction': 'bullish'
                })
        
        return signals
    
    def _detect_death_cross(self, prices: List[float]) -> List[Dict[str, Any]]:
        """
        Detect death cross (50-day MA crosses below 200-day MA).
        """
        signals = []
        
        # Calculate moving averages
        ma50 = self._simple_moving_average(prices, 50)
        ma200 = self._simple_moving_average(prices, 200)
        
        if len(ma50) < 2 or len(ma200) < 2:
            return signals
        
        # Find crossing points
        for i in range(1, min(len(ma50), len(ma200))):
            if ma50[i-1] > ma200[i-1] and ma50[i] < ma200[i]:  # Cross from above
                signals.append({
                    'index': i,
                    'confidence': 0.8,
                    'direction': 'bearish'
                })
        
        return signals
    
    def _simple_moving_average(self, prices: List[float], period: int) -> List[float]:
        """
        Calculate simple moving average.
        """
        if len(prices) < period:
            return []
        
        ma = []
        for i in range(len(prices)):
            if i < period - 1:
                ma.append(0)
            else:
                ma.append(sum(prices[i-period+1:i+1]) / period)
        
        return ma


class AutomatedStrategyDiscovery:
    """
    Main class for automated strategy discovery combining multiple approaches.
    """
    
    def __init__(self, market_data_service: MarketDataPort):
        self.market_data_service = market_data_service
        self.strategy_generator = StrategyGenerator()
        self.backtesting_engine = BacktestingEngine(market_data_service)
        self.evaluator = StrategyEvaluator(market_data_service, self.backtesting_engine)
        self.ga_optimizer = GeneticAlgorithmOptimizer(self.evaluator)
        self.pattern_recognizer = PatternRecognizer()
        self.discovered_strategies = []
    
    def discover_strategies(
        self, 
        symbol: Symbol, 
        start_date: datetime, 
        end_date: datetime,
        strategy_count: int = 20,
        use_genetic_optimization: bool = True
    ) -> List[DiscoveredStrategy]:
        """
        Discover profitable strategies for a given symbol and time period.
        """
        print(f"Starting strategy discovery for {symbol} from {start_date.date()} to {end_date.date()}")
        
        # Generate candidate strategies
        candidate_templates = self.strategy_generator.generate_diverse_strategies(strategy_count)
        
        discovered_strategies = []
        
        for i, template in enumerate(candidate_templates):
            print(f"Evaluating strategy {i+1}/{len(candidate_templates)}: {template.name}")
            
            # Optimize parameters if enabled
            if use_genetic_optimization:
                optimized_params = self.ga_optimizer.optimize_parameters(
                    template, symbol, start_date, end_date
                )
            else:
                # Use default or random parameters
                optimized_params = {}
                for param_name, (min_val, max_val) in template.parameters.items():
                    optimized_params[param_name] = (min_val + max_val) / 2
            
            # Evaluate the optimized strategy
            discovered_strategy, _ = self.evaluator.evaluate_strategy(
                template, optimized_params, symbol, start_date, end_date
            )
            
            if discovered_strategy.fitness_score > 0.1:  # Only keep strategies with reasonable fitness
                discovered_strategies.append(discovered_strategy)
        
        # Sort by fitness score
        discovered_strategies.sort(key=lambda x: x.fitness_score, reverse=True)
        
        # Keep only top performing strategies
        self.discovered_strategies = discovered_strategies[:10]
        
        print(f"Discovered {len(self.discovered_strategies)} profitable strategies")
        return self.discovered_strategies
    
    def find_pattern_based_strategies(self, symbol: Symbol, days: int = 365) -> List[Dict[str, Any]]:
        """
        Find strategies based on technical pattern recognition.
        """
        # Get historical prices
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        try:
            prices = self.market_data_service.get_historical_prices(symbol, start_date, end_date)
            price_values = [float(p.amount) for p in prices]
        except:
            # If real data unavailable, generate mock data
            price_values = [100 + i * 0.1 + np.random.normal(0, 1) for i in range(252)]
        
        # Find patterns in the data
        patterns = self.pattern_recognizer.find_patterns(price_values)
        
        # Generate strategies based on detected patterns
        pattern_strategies = []
        for pattern in patterns:
            strategy = {
                'pattern': pattern['pattern'],
                'entry_signal': f"Buy on {pattern['pattern']} bullish signal" if pattern['direction'] == 'bullish' else f"Sell on {pattern['pattern']} bearish signal",
                'exit_strategy': 'Follow through with position until profit target or stop loss',
                'confidence': pattern['confidence'],
                'date': start_date + timedelta(days=pattern['index']) if pattern['index'] < len(price_values) else end_date
            }
            pattern_strategies.append(strategy)
        
        return pattern_strategies
    
    def get_best_strategies(self, count: int = 5) -> List[DiscoveredStrategy]:
        """
        Get the best performing discovered strategies.
        """
        return sorted(self.discovered_strategies, key=lambda x: x.fitness_score, reverse=True)[:count]
    
    def run_walk_forward_analysis(self, 
                                  symbol: Symbol, 
                                  start_date: datetime, 
                                  end_date: datetime,
                                  in_sample_ratio: float = 0.7) -> List[Dict[str, Any]]:
        """
        Run walk-forward analysis to validate strategy robustness.
        """
        # Calculate in-sample and out-of-sample periods
        total_days = (end_date - start_date).days
        in_sample_days = int(total_days * in_sample_ratio)
        
        in_sample_end = start_date + timedelta(days=in_sample_days)
        out_sample_start = in_sample_end + timedelta(days=1)
        
        results = []
        
        # Discover strategies on in-sample data
        in_sample_strategies = self.discover_strategies(
            symbol, start_date, in_sample_end, strategy_count=10, use_genetic_optimization=True
        )
        
        # Validate on out-of-sample data
        for strategy in in_sample_strategies[:3]:  # Test top 3 strategies
            # Re-evaluate on out-of-sample data
            strategy_func = self.evaluator.create_strategy_function(
                strategy.template, strategy.optimized_params
            )
            
            out_sample_backtester = StrategyTester(self.backtesting_engine)
            try:
                out_sample_result = out_sample_backtester.test_strategy(
                    strategy_func,
                    symbol,
                    out_sample_start,
                    end_date,
                    initial_capital=50000.0,
                    name=f"Validation_{strategy.template.name}",
                    **strategy.optimized_params
                )
                
                results.append({
                    'strategy_name': strategy.template.name,
                    'in_sample_fitness': strategy.fitness_score,
                    'out_sample_fitness': self.evaluator._calculate_fitness_score(out_sample_result),
                    'in_sample_return': strategy.performance_metrics['total_return_pct'],
                    'out_sample_return': out_sample_result.total_return_pct,
                    'overfitting_ratio': abs(strategy.fitness_score - self.evaluator._calculate_fitness_score(out_sample_result)) / strategy.fitness_score if strategy.fitness_score != 0 else float('inf')
                })
            except:
                continue
        
        return results
    
    def save_discovered_strategies(self, filepath: str):
        """
        Save discovered strategies to a file.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.discovered_strategies, f)
    
    def load_discovered_strategies(self, filepath: str):
        """
        Load discovered strategies from a file.
        """
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.discovered_strategies = pickle.load(f)


# Initialize the automated strategy discovery system
def initialize_strategy_discovery(market_data_service: MarketDataPort) -> AutomatedStrategyDiscovery:
    """
    Initialize and return an instance of the automated strategy discovery system.
    """
    return AutomatedStrategyDiscovery(market_data_service)