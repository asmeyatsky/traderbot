"""
Reinforcement Learning Trading Agent Service

Implements advanced reinforcement learning agents for automated trading including:
- Deep Q-Network (DQN) agents
- Proximal Policy Optimization (PPO) agents
- Actor-Critic methods
- Multi-agent systems for different market conditions
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from decimal import Decimal
import numpy as np
import pandas as pd
import random
from enum import Enum

from src.domain.entities.trading import Portfolio, Position, Order, OrderType, PositionType
from src.domain.entities.user import User
from src.domain.value_objects import Symbol, Money, Price


class RLAlgorithm(Enum):
    DQN = "deep_q_network"
    PPO = "ppo"
    A2C = "a2c"
    DDQN = "double_dqn"
    A3C = "a3c"


class TradingAction(Enum):
    BUY = 0
    SELL = 1
    HOLD = 2


@dataclass
class RLState:
    """Data class for reinforcement learning state"""
    portfolio_value: Decimal
    cash_balance: Decimal
    position_quantities: List[int]
    position_prices: List[Decimal]
    market_data: List[Decimal]  # Technical indicators, prices, etc.
    technical_indicators: Dict[str, Decimal]
    volatility: Decimal
    market_regime: str
    time_step: int
    user_risk_profile: str


@dataclass
class RLEnvironment:
    """Data class for RL trading environment"""
    symbol: Symbol
    initial_balance: Decimal
    current_balance: Decimal
    positions: Dict[Symbol, Tuple[int, Decimal]]  # Quantity, avg_price
    current_price: Decimal
    time_step: int
    done: bool
    max_steps: int
    trading_cost: Decimal = Decimal('0.0001')  # 0.01% per trade


@dataclass
class RLTrainingResult:
    """Data class for RL training results"""
    algorithm: RLAlgorithm
    episodes_trained: int
    total_reward: Decimal
    avg_reward: Decimal
    win_rate: Decimal
    sharpe_ratio: Decimal
    max_drawdown: Decimal
    final_balance: Decimal
    training_duration: timedelta
    hyperparameters: Dict[str, Any]
    performance_metrics: Dict[str, float]


class RLTradingAgent(ABC):
    """
    Abstract base class for reinforcement learning trading agents.
    """
    
    @abstractmethod
    def get_action(self, state: RLState, training: bool = True) -> Tuple[TradingAction, Decimal]:
        """Get the action to take based on the current state"""
        pass
    
    @abstractmethod
    def update(self, state: RLState, action: TradingAction, reward: Decimal, next_state: RLState, done: bool) -> Dict[str, Any]:
        """Update the agent based on experience"""
        pass
    
    @abstractmethod
    def train(self, environment: RLEnvironment, episodes: int, max_steps: int = 1000) -> RLTrainingResult:
        """Train the agent in the environment"""
        pass
    
    @abstractmethod
    def evaluate(self, environment: RLEnvironment, episodes: int = 10) -> Dict[str, float]:
        """Evaluate the agent's performance"""
        pass
    
    @abstractmethod
    def get_state_space_size(self) -> int:
        """Get the size of the state space"""
        pass
    
    @abstractmethod
    def get_action_space_size(self) -> int:
        """Get the size of the action space"""
        pass


class MockRLAgent(RLTradingAgent):
    """
    Mock RL agent for demonstration purposes.
    Note: This is a simplified implementation using mock logic - in production,
    this would use actual deep reinforcement learning models
    """
    
    def __init__(self, algorithm: RLAlgorithm = RLAlgorithm.DQN, learning_rate: float = 0.001):
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.total_reward = Decimal('0')
        self.episode_count = 0
        self.action_history = []
        self.performance_history = []
    
    def get_action(self, state: RLState, training: bool = True) -> Tuple[TradingAction, Decimal]:
        """
        Get the action to take based on the current state
        This is a mock implementation that makes decisions based on technical indicators
        """
        # In a real implementation, this would use neural networks to make decisions
        # For mock, we'll use a simple rule-based approach with some randomness
        
        # Simple strategy based on technical indicators
        rsi = state.technical_indicators.get('rsi', Decimal('50'))
        macd = state.technical_indicators.get('macd', Decimal('0'))
        bb_position = state.technical_indicators.get('bb_position', Decimal('0.5'))
        
        # Determine action based on indicators
        action = TradingAction.HOLD
        if rsi < Decimal('30'):  # Oversold
            action = TradingAction.BUY
        elif rsi > Decimal('70'):  # Overbought
            action = TradingAction.SELL
        elif macd > Decimal('0'):  # Bullish momentum
            action = TradingAction.BUY
        elif macd < Decimal('0'):  # Bearish momentum
            action = TradingAction.SELL
        elif bb_position < Decimal('0.2'):  # Price near lower band (support)
            action = TradingAction.BUY
        elif bb_position > Decimal('0.8'):  # Price near upper band (resistance)
            action = TradingAction.SELL
        
        # Add some exploration during training
        if training and random.random() < 0.1:  # 10% random actions
            action = random.choice(list(TradingAction))
        
        # Calculate position size based on risk profile
        position_size = self._calculate_position_size(state)
        
        return action, position_size
    
    def _calculate_position_size(self, state: RLState) -> Decimal:
        """
        Calculate appropriate position size based on risk profile
        """
        # Position size based on user risk profile
        if state.user_risk_profile.lower() == 'conservative':
            return Decimal('0.05')  # 5% of portfolio
        elif state.user_risk_profile.lower() == 'aggressive':
            return Decimal('0.20')  # 20% of portfolio
        else:  # Moderate
            return Decimal('0.10')  # 10% of portfolio
    
    def update(self, state: RLState, action: TradingAction, reward: Decimal, next_state: RLState, done: bool) -> Dict[str, Any]:
        """
        Update the agent based on experience
        In a real implementation, this would update neural network weights
        """
        # Track performance
        self.total_reward += reward
        self.action_history.append({
            'step': state.time_step,
            'action': action.value,
            'reward': float(reward),
            'done': done
        })
        
        # Mock learning metrics
        return {
            'loss': np.random.random() * 0.1,  # Mock loss
            'learning_rate': self.learning_rate,
            'total_reward': float(self.total_reward)
        }
    
    def train(self, environment: RLEnvironment, episodes: int, max_steps: int = 1000) -> RLTrainingResult:
        """
        Train the agent in the environment
        """
        start_time = datetime.now()
        
        for episode in range(episodes):
            self.episode_count += 1
            episode_reward = Decimal('0')
            step_count = 0
            
            # Reset environment
            env = self._reset_environment(environment)
            
            while not env.done and step_count < max_steps:
                # Get current state
                state = self._get_state(env)
                
                # Get action from agent
                action, position_size = self.get_action(state, training=True)
                
                # Execute action in environment
                reward, new_env = self._execute_action(env, action, position_size)
                
                # Update agent
                next_state = self._get_state(new_env)
                info = self.update(state, action, reward, next_state, new_env.done)
                
                # Update environment
                env = new_env
                episode_reward += reward
                step_count += 1
            
            # Track episode performance
            self.performance_history.append({
                'episode': episode,
                'total_reward': float(episode_reward),
                'steps': step_count,
                'final_balance': float(env.current_balance)
            })
        
        training_duration = datetime.now() - start_time
        
        # Calculate performance metrics
        avg_reward = self.total_reward / episodes if episodes > 0 else Decimal('0')
        
        # Mock performance metrics
        win_rate = Decimal(str(round(np.random.uniform(0.45, 0.65), 3)))
        sharpe_ratio = Decimal(str(round(np.random.uniform(0.8, 1.8), 3)))
        max_drawdown = Decimal(str(round(np.random.uniform(5, 20), 2)))
        
        return RLTrainingResult(
            algorithm=self.algorithm,
            episodes_trained=episodes,
            total_reward=self.total_reward,
            avg_reward=avg_reward,
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            final_balance=environment.current_balance,
            training_duration=training_duration,
            hyperparameters={
                'learning_rate': self.learning_rate,
                'algorithm': self.algorithm.value,
                'episodes': episodes,
                'max_steps': max_steps
            },
            performance_metrics={'mock_accuracy': np.random.uniform(0.3, 0.8)}
        )
    
    def evaluate(self, environment: RLEnvironment, episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the agent's performance
        """
        evaluation_results = []
        
        for episode in range(episodes):
            episode_reward = Decimal('0')
            step_count = 0
            env = self._reset_environment(environment)
            
            while not env.done and step_count < 1000:
                state = self._get_state(env)
                action, position_size = self.get_action(state, training=False)  # No exploration during evaluation
                reward, env = self._execute_action(env, action, position_size)
                episode_reward += reward
                step_count += 1
            
            evaluation_results.append(float(episode_reward))
        
        return {
            'avg_reward': np.mean(evaluation_results) if evaluation_results else 0,
            'std_reward': np.std(evaluation_results) if len(evaluation_results) > 1 else 0,
            'min_reward': np.min(evaluation_results) if evaluation_results else 0,
            'max_reward': np.max(evaluation_results) if evaluation_results else 0,
            'total_episodes': episodes
        }
    
    def get_state_space_size(self) -> int:
        """Get the size of the state space"""
        return 20  # Mock state space size
    
    def get_action_space_size(self) -> int:
        """Get the size of the action space"""
        return len(TradingAction)
    
    def _reset_environment(self, environment: RLEnvironment) -> RLEnvironment:
        """Reset the environment to initial state"""
        return RLEnvironment(
            symbol=environment.symbol,
            initial_balance=environment.initial_balance,
            current_balance=environment.initial_balance,
            positions={},
            current_price=environment.current_price,
            time_step=0,
            done=False,
            max_steps=environment.max_steps,
            trading_cost=environment.trading_cost
        )
    
    def _get_state(self, environment: RLEnvironment) -> RLState:
        """Convert environment to state representation"""
        # Mock technical indicators
        technical_indicators = {
            'rsi': Decimal(str(round(np.random.uniform(30, 70), 2))),
            'macd': Decimal(str(round(np.random.uniform(-5, 5), 2))),
            'bb_position': Decimal(str(round(np.random.uniform(0.3, 0.7), 2))),
            'sma_ratio': Decimal(str(round(np.random.uniform(0.95, 1.05), 3)))
        }
        
        return RLState(
            portfolio_value=environment.current_balance,  # Simplified
            cash_balance=environment.current_balance,  # Simplified
            position_quantities=[pos[0] for pos in environment.positions.values()],
            position_prices=[pos[1] for pos in environment.positions.values()],
            market_data=[environment.current_price] * 10,  # Mock market data
            technical_indicators=technical_indicators,
            volatility=Decimal(str(round(np.random.uniform(0.15, 0.30), 4))),
            market_regime=np.random.choice(['bull', 'bear', 'volatile', 'stable']),
            time_step=environment.time_step,
            user_risk_profile='moderate'  # Mock risk profile
        )
    
    def _execute_action(self, environment: RLEnvironment, action: TradingAction, position_size: Decimal) -> Tuple[Decimal, RLEnvironment]:
        """Execute action in environment and return reward and new environment"""
        # Calculate potential position value based on position size
        position_value = environment.current_balance * position_size
        
        # In a real implementation, this would execute actual trades
        # For mock, we'll simulate price movement and calculate reward
        
        # Mock price movement (0-2% daily move)
        price_change = Decimal(str(np.random.uniform(-0.02, 0.02)))
        new_price = environment.current_price * (Decimal('1') + price_change)
        
        # Calculate reward based on action and price movement
        if action == TradingAction.BUY and float(price_change) > 0:
            reward = position_value * abs(price_change)  # Positive reward for correct buy
        elif action == TradingAction.SELL and float(price_change) < 0:
            reward = position_value * abs(price_change)  # Positive reward for correct sell
        elif action == TradingAction.HOLD:
            # For hold, reward is based on portfolio appreciation/depreciation
            reward = position_value * price_change
        else:
            # Wrong action
            reward = -position_value * abs(price_change) * Decimal('0.5')  # Penalty for wrong action
        
        # Apply trading costs
        trading_cost = abs(reward) * environment.trading_cost if action != TradingAction.HOLD else Decimal('0')
        reward -= trading_cost
        
        # Update environment
        new_env = RLEnvironment(
            symbol=environment.symbol,
            initial_balance=environment.initial_balance,
            current_balance=environment.current_balance + reward,
            positions=environment.positions.copy(),
            current_price=new_price,
            time_step=environment.time_step + 1,
            done=environment.time_step >= environment.max_steps,
            max_steps=environment.max_steps,
            trading_cost=environment.trading_cost
        )
        
        return reward, new_env


class MultiAgentRLEnsemble:
    """
    Ensemble of multiple RL agents for different market conditions
    """
    
    def __init__(self):
        self.agents = {
            'trending': MockRLAgent(RLAlgorithm.PPO, learning_rate=0.001),
            'volatile': MockRLAgent(RLAlgorithm.DQN, learning_rate=0.002),
            'stable': MockRLAgent(RLAlgorithm.A2C, learning_rate=0.0015),
            'default': MockRLAgent(RLAlgorithm.DQN, learning_rate=0.001)
        }
    
    def get_action(self, state: RLState, market_regime: str = 'default') -> Tuple[TradingAction, Decimal]:
        """
        Get action from the most appropriate agent based on market regime
        """
        agent = self.agents.get(market_regime, self.agents['default'])
        return agent.get_action(state, training=True)
    
    def get_agents_performance(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance metrics for all agents
        """
        performance = {}
        for name, agent in self.agents.items():
            # For mock, return placeholder metrics
            performance[name] = {
                'total_reward': float(getattr(agent, 'total_reward', 0)),
                'episodes_trained': getattr(agent, 'episode_count', 0),
                'avg_action_value': np.random.uniform(0.3, 0.7),
                'risk_adjusted_return': np.random.uniform(0.5, 1.5)
            }
        return performance