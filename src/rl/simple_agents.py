"""
Simplified RL agents without PyTorch dependencies.
These agents use basic Q-learning and policy gradient methods.
"""

import numpy as np
import random
from collections import defaultdict, deque
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class SimpleDQNAgent:
    """Simplified DQN agent using tabular Q-learning."""
    
    def __init__(self, state_size: int, action_size: int, lr: float = 0.1, epsilon: float = 0.1):
        """
        Initialize simplified DQN agent.
        
        Args:
            state_size (int): Size of state vector
            action_size (int): Number of possible actions
            lr (float): Learning rate
            epsilon (float): Exploration rate
        """
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = 0.95  # Discount factor
        
        # Use tabular Q-learning with state discretization
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        self.memory = deque(maxlen=2000)
        self.training_count = 0
        
        logger.info(f"SimpleDQNAgent initialized with {action_size} actions")
    
    def _discretize_state(self, state: np.ndarray) -> str:
        """Convert continuous state to discrete key for Q-table."""
        # Round to 1 decimal place and create string key
        discrete_state = np.round(state, 1)
        return str(discrete_state.tolist())
    
    def act(self, state: np.ndarray) -> int:
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state (np.ndarray): Current state
            
        Returns:
            int: Selected action
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        state_key = self._discretize_state(state)
        q_values = self.q_table[state_key]
        return int(np.argmax(q_values))
    
    def step(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> bool:
        """
        Learn from experience.
        
        Args:
            state (np.ndarray): Previous state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.ndarray): New state
            done (bool): Whether episode is finished
            
        Returns:
            bool: True if learning was successful
        """
        try:
            # Store experience
            self.memory.append((state, action, reward, next_state, done))
            
            # Q-learning update
            state_key = self._discretize_state(state)
            next_state_key = self._discretize_state(next_state)
            
            # Current Q-value
            current_q = self.q_table[state_key][action]
            
            # Target Q-value
            if done:
                target_q = reward
            else:
                next_q_values = self.q_table[next_state_key]
                target_q = reward + self.gamma * np.max(next_q_values)
            
            # Update Q-value
            self.q_table[state_key][action] += self.lr * (target_q - current_q)
            
            self.training_count += 1
            
            # Decay epsilon
            self.epsilon = max(0.01, self.epsilon * 0.995)
            
            logger.debug(f"DQN training step {self.training_count}, reward: {reward:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"DQN training error: {e}")
            return False
    
    def get_training_info(self) -> Dict:
        """Get training information."""
        return {
            'training_count': self.training_count,
            'epsilon': self.epsilon,
            'q_table_size': len(self.q_table),
            'memory_size': len(self.memory)
        }


class SimplePPOAgent:
    """Simplified PPO agent using basic policy gradient."""
    
    def __init__(self, state_size: int, action_size: int, lr: float = 0.01):
        """
        Initialize simplified PPO agent.
        
        Args:
            state_size (int): Size of state vector
            action_size (int): Number of possible actions
            lr (float): Learning rate
        """
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        
        # Simple policy network (linear weights)
        self.policy_weights = np.random.normal(0, 0.1, (state_size, action_size))
        self.value_weights = np.random.normal(0, 0.1, state_size)
        
        # Training data storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.training_count = 0
        
        logger.info(f"SimplePPOAgent initialized with {action_size} actions")
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def act(self, state: np.ndarray) -> int:
        """
        Choose action using policy network.
        
        Args:
            state (np.ndarray): Current state
            
        Returns:
            int: Selected action
        """
        # Compute action probabilities
        logits = np.dot(state, self.policy_weights)
        probs = self._softmax(logits)
        
        # Sample action
        action = np.random.choice(self.action_size, p=probs)
        return int(action)
    
    def update(self, states: List[np.ndarray], actions: List[int], 
               rewards: List[float], next_states: List[np.ndarray], 
               dones: List[bool]) -> bool:
        """
        Update policy using collected experiences.
        
        Args:
            states: List of states
            actions: List of actions
            rewards: List of rewards
            next_states: List of next states
            dones: List of done flags
            
        Returns:
            bool: True if update was successful
        """
        try:
            if len(states) == 0:
                return False
            
            # Convert to numpy arrays
            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)
            
            # Compute returns (simple Monte Carlo)
            returns = []
            G = 0
            for reward in reversed(rewards):
                G = reward + 0.99 * G
                returns.insert(0, G)
            returns = np.array(returns)
            
            # Normalize returns
            if len(returns) > 1:
                returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
            
            # Policy gradient update
            for i in range(len(states)):
                state = states[i]
                action = actions[i]
                return_val = returns[i]
                
                # Compute current policy
                logits = np.dot(state, self.policy_weights)
                probs = self._softmax(logits)
                
                # Policy gradient
                grad = np.zeros_like(probs)
                grad[action] = return_val / (probs[action] + 1e-8)
                
                # Update weights
                policy_grad = np.outer(state, grad)
                self.policy_weights += self.lr * policy_grad
                
                # Simple value function update
                value = np.dot(state, self.value_weights)
                value_error = return_val - value
                self.value_weights += self.lr * value_error * state
            
            self.training_count += 1
            logger.debug(f"PPO training step {self.training_count}")
            return True
            
        except Exception as e:
            logger.error(f"PPO training error: {e}")
            return False
    
    def get_training_info(self) -> Dict:
        """Get training information."""
        return {
            'training_count': self.training_count,
            'policy_norm': np.linalg.norm(self.policy_weights),
            'value_norm': np.linalg.norm(self.value_weights)
        }
