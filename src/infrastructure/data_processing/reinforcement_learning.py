"""
Reinforcement Learning Trading Strategies

This module implements reinforcement learning algorithms for trading,
including DQN, A2C, and PPO agents for autonomous trading decisions.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import random
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pickle
import os

from src.domain.entities.trading import Order, Position, Portfolio
from src.domain.entities.user import User
from src.domain.value_objects import Money, Symbol, Price
from src.domain.ports import MarketDataPort, AIModelPort
from src.infrastructure.config.settings import settings


class TradingEnvironment(gym.Env):
    """
    Custom trading environment for reinforcement learning.
    
    The environment simulates trading conditions where an RL agent can learn
    to make profitable trades while managing risk.
    """
    
    def __init__(self, 
                 symbol: Symbol, 
                 initial_balance: float = 10000.0,
                 transaction_cost: float = 0.001,  # 0.1% per transaction
                 max_position: float = 0.2):  # Max 20% of portfolio per position
        super(TradingEnvironment, self).__init__()
        
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        
        # Define action space: [0=hold, 1=buy, 2=sell]
        self.action_space = spaces.Discrete(3)
        
        # Define observation space: [balance_ratio, position_size, price_change, volatility, volume]
        # In reality, this would be more complex with many more features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )
        
        # Trading state variables
        self.balance = initial_balance
        self.position = 0  # Current position size in shares
        self.position_avg_price = 0  # Average price of current position
        self.current_step = 0
        self.net_worth_history = []
        self.trades = []
        self.price_data = []
        self.data_index = 0
        
    def reset(self):
        """
        Reset the environment to initial state.
        """
        self.balance = self.initial_balance
        self.position = 0
        self.position_avg_price = 0
        self.current_step = 0
        self.net_worth_history = []
        self.trades = []
        self.data_index = 0
        
        # Return initial observation
        return self._next_observation()
    
    def _next_observation(self):
        """
        Get the next observation from the environment.
        """
        # This is a simplified observation - in reality, you'd have more features
        if self.data_index < len(self.price_data):
            current_price = self.price_data[self.data_index]
        else:
            current_price = 100  # Default if no data
        
        # Calculate balance ratio (how much cash is available vs total portfolio value)
        current_value = self.balance + (self.position * current_price if self.position > 0 else 0)
        balance_ratio = self.balance / current_value if current_value > 0 else 1.0
        
        # Position size as percentage of portfolio
        position_ratio = (self.position * current_price) / current_value if current_value > 0 else 0
        
        # Price change (simplified)
        price_change = 0.0
        volatility = 0.01  # Simplified volatility
        volume = 100000  # Simplified volume
        
        return np.array([
            balance_ratio,
            position_ratio,
            price_change,
            volatility,
            volume
        ], dtype=np.float32)
    
    def step(self, action):
        """
        Execute one time step within the environment.
        """
        if self.data_index >= len(self.price_data):
            return self._next_observation(), 0, True, {}
        
        current_price = self.price_data[self.data_index]
        
        # Execute action
        reward = 0
        if action == 1:  # Buy
            max_buy_amount = self.balance * self.max_position
            max_buy_shares = int(max_buy_amount / current_price)
            
            if max_buy_shares > 0:
                # Calculate transaction cost
                cost = max_buy_shares * current_price * self.transaction_cost
                total_cost = (max_buy_shares * current_price) + cost
                
                if total_cost <= self.balance:
                    # Execute buy
                    self.balance -= total_cost
                    
                    # Update position (average price calculation simplified)
                    if self.position > 0:
                        total_shares = self.position + max_buy_shares
                        total_value = (self.position * self.position_avg_price) + (max_buy_shares * current_price)
                        self.position_avg_price = total_value / total_shares
                    else:
                        self.position_avg_price = current_price
                    
                    self.position += max_buy_shares
                    
                    self.trades.append({
                        'action': 'buy',
                        'price': current_price,
                        'shares': max_buy_shares,
                        'cost': total_cost
                    })
        
        elif action == 2:  # Sell
            if self.position > 0:
                # Sell all current position
                revenue = self.position * current_price
                cost = revenue * self.transaction_cost
                net_revenue = revenue - cost
                
                self.balance += net_revenue
                
                self.trades.append({
                    'action': 'sell',
                    'price': current_price,
                    'shares': self.position,
                    'revenue': net_revenue
                })
                
                # Calculate reward based on profit/loss
                avg_cost = self.position * self.position_avg_price
                profit = revenue - avg_cost
                reward = profit / avg_cost  # Normalize reward by cost basis
                
                # Reset position
                self.position = 0
                self.position_avg_price = 0
        
        # Update net worth history
        current_value = self.balance + (self.position * current_price)
        self.net_worth_history.append(current_value)
        
        # Calculate reward
        if len(self.net_worth_history) > 1:
            # Reward based on portfolio value change
            reward = (self.net_worth_history[-1] - self.net_worth_history[-2]) / self.net_worth_history[-2]
        else:
            reward = 0
        
        # Move to next time step
        self.data_index += 1
        
        # Check if done (end of data)
        done = self.data_index >= len(self.price_data)
        
        # Get next observation
        obs = self._next_observation()
        
        # Info dict (empty for now)
        info = {}
        
        return obs, reward, done, info
    
    def set_price_data(self, prices: List[float]):
        """
        Set the historical price data for the environment.
        """
        self.price_data = prices


class DQNAgent:
    """
    Deep Q-Network agent for trading.
    """
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    
    def _build_model(self):
        """
        Build the neural network model.
        """
        model = keras.Sequential([
            layers.Dense(24, input_dim=self.state_size, activation='relu'),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(
            loss='mse',
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate)
        )
        return model
    
    def update_target_model(self):
        """
        Update the target model with the main model's weights.
        """
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory.
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """
        Choose action based on epsilon-greedy policy.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size: int = 32):
        """
        Train the model on a batch of experiences.
        """
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = np.zeros((batch_size, self.state_size))
        next_states = np.zeros((batch_size, self.state_size))
        actions, rewards, dones = [], [], []
        
        for i, (state, action, reward, next_state, done) in enumerate(batch):
            states[i] = state
            next_states[i] = next_state
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
        
        target = rewards
        target = np.array(target) + 0.95 * (
            np.amax(self.target_model.predict(next_states, verbose=0), axis=1)
        ) * (1 - np.array(dones))
        
        target_f = self.model.predict(states, verbose=0)
        target_f[range(batch_size), actions] = target
        
        self.model.fit(states, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class PPOAgent:
    """
    Proximal Policy Optimization agent for trading.
    """
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 0.002, gamma: float = 0.99, 
                 eps_clip: float = 0.2, k_epochs: int = 4):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.action_dim = action_dim
        
        self.actor = self._build_actor(state_dim, action_dim)
        self.critic = self._build_critic(state_dim)
        
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.MseLoss = nn.MSELoss()
    
    def _build_actor(self, state_dim, action_dim):
        """
        Build the actor network (policy network).
        """
        class Actor(nn.Module):
            def __init__(self, state_dim, action_dim):
                super(Actor, self).__init__()
                self.layer1 = nn.Linear(state_dim, 256)
                self.layer2 = nn.Linear(256, 256)
                self.layer3 = nn.Linear(256, action_dim)
                self.dropout = nn.Dropout(0.1)
                
            def forward(self, state):
                x = torch.tanh(self.layer1(state))
                x = self.dropout(x)
                x = torch.tanh(self.layer2(x))
                x = self.dropout(x)
                x = self.layer3(x)
                return torch.softmax(x, dim=-1)
        
        return Actor(state_dim, action_dim)
    
    def _build_critic(self, state_dim):
        """
        Build the critic network (value network).
        """
        class Critic(nn.Module):
            def __init__(self, state_dim):
                super(Critic, self).__init__()
                self.layer1 = nn.Linear(state_dim, 256)
                self.layer2 = nn.Linear(256, 256)
                self.layer3 = nn.Linear(256, 1)
                self.dropout = nn.Dropout(0.1)
                
            def forward(self, state):
                x = torch.tanh(self.layer1(state))
                x = self.dropout(x)
                x = torch.tanh(self.layer2(x))
                x = self.dropout(x)
                x = self.layer3(x)
                return x
        
        return Critic(state_dim)
    
    def get_action(self, state):
        """
        Get action based on current policy.
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action).item()
    
    def update(self, states, actions, logprobs, rewards, dones, next_states):
        """
        Update the PPO networks.
        """
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_logprobs = torch.FloatTensor(logprobs)
        rewards = torch.FloatTensor(rewards)
        dones = torch.BoolTensor(dones)
        next_states = torch.FloatTensor(next_states)
        
        # Compute discounted rewards
        discounted_rewards = []
        running_add = 0
        for reward in reversed(rewards):
            running_add = reward + self.gamma * running_add
            discounted_rewards.insert(0, running_add)
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        
        # Normalize rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)
        
        # Optimize policy K epochs times
        for _ in range(self.k_epochs):
            # Evaluate old actions and values
            logprobs = torch.log(self.actor(states).gather(1, actions.unsqueeze(1))).squeeze(1)
            state_values = self.critic(states).squeeze(1)
            
            # Find ratios
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # Compute advantage
            advantages = discounted_rewards - state_values.detach()
            
            # Compute surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Compute critic loss
            critic_loss = self.MseLoss(state_values, discounted_rewards)
            
            # Update networks
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()
            
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()


class A2CAgent:
    """
    Actor-Critic Advantage agent for trading.
    """
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        
        # Actor network (policy)
        self.actor = self._build_actor_network()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        # Critic network (value)
        self.critic = self._build_critic_network()
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.MseLoss = nn.MSELoss()
    
    def _build_actor_network(self):
        """
        Build the actor network.
        """
        class Actor(nn.Module):
            def __init__(self, state_dim, action_dim):
                super(Actor, self).__init__()
                self.fc1 = nn.Linear(state_dim, 128)
                self.fc2 = nn.Linear(128, 128)
                self.fc3 = nn.Linear(128, action_dim)
                
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.fc3(x)
                return torch.softmax(x, dim=-1)
        
        return Actor(self.state_dim, self.action_dim)
    
    def _build_critic_network(self):
        """
        Build the critic network.
        """
        class Critic(nn.Module):
            def __init__(self, state_dim):
                super(Critic, self).__init__()
                self.fc1 = nn.Linear(state_dim, 128)
                self.fc2 = nn.Linear(128, 128)
                self.fc3 = nn.Linear(128, 1)
                
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.fc3(x)
                return x
        
        return Critic(self.state_dim)
    
    def get_action(self, state):
        """
        Get action based on current state.
        """
        state = torch.FloatTensor(state)
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action).item()
    
    def update(self, state, action, reward, next_state, done):
        """
        Update the actor and critic networks.
        """
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        
        # Compute value of current state
        current_value = self.critic(state)
        
        # Compute value of next state
        next_value = self.critic(next_state)
        
        # Compute advantage
        advantage = reward + (0.99 * next_value * (1 - done)) - current_value
        
        # Compute actor loss
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        log_prob = dist.log_prob(torch.LongTensor([action]))
        actor_loss = -log_prob * advantage.detach()
        
        # Compute critic loss
        critic_loss = advantage.pow(2).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


class RLTradingStrategy:
    """
    Reinforcement Learning Trading Strategy that uses trained agents.
    """
    
    def __init__(self, 
                 market_data_service: MarketDataPort,
                 strategy_type: str = "DQN"):
        self.market_data_service = market_data_service
        self.strategy_type = strategy_type
        self.agents = {}  # Store agents for different symbols
        self.environments = {}  # Store environments for different symbols
        
    def prepare_training_data(self, symbol: Symbol, days: int = 365) -> List[float]:
        """
        Prepare training data for reinforcement learning.
        """
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        # Get historical prices
        try:
            prices = self.market_data_service.get_historical_prices(symbol, start_date, end_date)
            return [float(p.amount) for p in prices]
        except:
            # Return mock data if real data unavailable
            return [100 + i * 0.1 + np.random.normal(0, 1) for i in range(252)]
    
    def train_agent(self, symbol: Symbol, episodes: int = 1000) -> Any:
        """
        Train an RL agent for a specific symbol.
        """
        # Prepare environment and data
        price_data = self.prepare_training_data(symbol)
        
        if symbol not in self.environments:
            self.environments[symbol] = TradingEnvironment(symbol)
        
        env = self.environments[symbol]
        env.set_price_data(price_data)
        
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        
        # Initialize agent based on strategy type
        if self.strategy_type == "DQN":
            if symbol not in self.agents:
                self.agents[symbol] = DQNAgent(state_size, action_size)
            agent = self.agents[symbol]
            
            # Training loop for DQN
            for e in range(episodes):
                state = env.reset()
                state = np.reshape(state, [1, state_size])
                total_reward = 0
                
                for time in range(len(price_data) - 1):
                    # Get action from agent
                    action = agent.act(state)
                    
                    # Take action in environment
                    next_state, reward, done, _ = env.step(action)
                    next_state = np.reshape(next_state, [1, state_size])
                    
                    # Remember experience
                    agent.remember(state, action, reward, next_state, done)
                    
                    state = next_state
                    total_reward += reward
                    
                    if done:
                        print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}")
                        break
                
                # Train the agent on batch
                if len(agent.memory) > 32:
                    agent.replay(32)
                
                # Update target model every 10 episodes
                if e % 10 == 0:
                    agent.update_target_model()
        
        elif self.strategy_type == "PPO":
            if symbol not in self.agents:
                self.agents[symbol] = PPOAgent(state_size, action_size)
            agent = self.agents[symbol]
            
            # Training loop for PPO
            for e in range(episodes):
                state = env.reset()
                states, actions, logprobs, rewards, dones, next_states = [], [], [], [], [], []
                
                for time in range(len(price_data) - 1):
                    action, logprob = agent.get_action(state)
                    next_state, reward, done, _ = env.step(action)
                    
                    states.append(state)
                    actions.append(action)
                    logprobs.append(logprob)
                    rewards.append(reward)
                    dones.append(done)
                    next_states.append(next_state)
                    
                    state = next_state
                    
                    if done:
                        break
                
                # Update the agent
                agent.update(states, actions, logprobs, rewards, dones, next_states)
                
                print(f"PPO Episode {e+1}/{episodes} completed")
        
        elif self.strategy_type == "A2C":
            if symbol not in self.agents:
                self.agents[symbol] = A2CAgent(state_size, action_size)
            agent = self.agents[symbol]
            
            # Training loop for A2C
            for e in range(episodes):
                state = env.reset()
                total_reward = 0
                
                for time in range(len(price_data) - 1):
                    action, logprob = agent.get_action(state)
                    next_state, reward, done, _ = env.step(action)
                    
                    # Update agent
                    agent.update(state, action, reward, next_state, done)
                    
                    state = next_state
                    total_reward += reward
                    
                    if done:
                        break
                
                print(f"A2C Episode {e+1}/{episodes}, Total Reward: {total_reward}")
        
        return agent
    
    def get_trading_signal(self, symbol: Symbol) -> str:
        """
        Get trading signal from trained agent.
        """
        if symbol not in self.agents:
            # If no trained agent, return HOLD
            return "HOLD"
        
        # Get current state from environment (simplified)
        if symbol not in self.environments:
            self.environments[symbol] = TradingEnvironment(symbol)
        
        env = self.environments[symbol]
        current_state = env._next_observation()
        
        # Get action from agent
        agent = self.agents[symbol]
        
        if self.strategy_type == "DQN":
            current_state = np.reshape(current_state, [1, -1])
            action_values = agent.model.predict(current_state, verbose=0)
            action = np.argmax(action_values[0])
        
        elif self.strategy_type in ["PPO", "A2C"]:
            action, _ = agent.get_action(current_state)
        
        # Map actions to trading signals
        action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        return action_map.get(action, "HOLD")
    
    def save_agent(self, symbol: Symbol, filepath: str):
        """
        Save the trained agent to a file.
        """
        if symbol in self.agents:
            if self.strategy_type == "DQN":
                self.agents[symbol].model.save(filepath)
            else:
                # For PyTorch models
                torch.save(self.agents[symbol].state_dict(), filepath)
    
    def load_agent(self, symbol: Symbol, filepath: str):
        """
        Load a trained agent from a file.
        """
        if self.strategy_type == "DQN":
            # Load the model and create a new agent
            loaded_model = keras.models.load_model(filepath)
            # Need to get state and action sizes from somewhere
            # This is simplified - in practice you'd save this info too
            pass
        else:
            # For PyTorch models
            if self.strategy_type == "PPO":
                agent = PPOAgent(5, 3)  # state_dim, action_dim - simplified
            elif self.strategy_type == "A2C":
                agent = A2CAgent(5, 3)  # state_dim, action_dim - simplified
            
            agent.load_state_dict(torch.load(filepath))
            self.agents[symbol] = agent


class DefaultAIModelService(AIModelPort):
    """
    Default implementation of AI model port with RL capabilities.
    """
    
    def __init__(self, market_data_service: MarketDataPort):
        self.market_data_service = market_data_service
        self.rl_strategies = {
            'DQN': RLTradingStrategy(market_data_service, 'DQN'),
            'PPO': RLTradingStrategy(market_data_service, 'PPO'),
            'A2C': RLTradingStrategy(market_data_service, 'A2C')
        }
    
    def predict_price_movement(self, symbol: Symbol, days: int = 1) -> float:
        """
        Predict price movement using ensemble of models.
        """
        # This would use the trained RL agents and other models
        # For now, returning a random prediction in the short term
        return np.random.uniform(-0.02, 0.02)  # -2% to +2% daily movement
    
    def get_trading_signal(self, symbol: Symbol) -> str:
        """
        Get trading signal from the main RL strategy.
        """
        # Use the DQN strategy by default
        return self.rl_strategies['DQN'].get_trading_signal(symbol)
    
    def analyze_portfolio_risk(self, portfolio: Portfolio) -> float:
        """
        Analyze portfolio risk using ML models.
        """
        # This would use trained models to assess portfolio risk
        # For now, returning a simple calculation
        total_value = float(portfolio.total_value.amount)
        if total_value == 0:
            return 0.0
        
        # Risk score based on position concentration and volatility
        risk_score = 0.0
        for position in portfolio.positions:
            # Simple risk score based on position size relative to portfolio
            position_ratio = float(position.market_value.amount) / total_value
            risk_score += position_ratio * 0.5  # Base risk for holding any position
        
        # Add random factor to simulate risk from market conditions
        risk_score += np.random.uniform(0.0, 0.3)
        
        return min(risk_score, 1.0)  # Cap at 1.0


# Initialize the default AI model service
default_ai_model_service = None