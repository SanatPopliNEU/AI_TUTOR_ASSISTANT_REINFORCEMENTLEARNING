"""
Deep Q-Network (DQN) implementation for tutorial action selection.

This module implements a DQN agent that learns to select optimal tutoring actions
based on student state and performance metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import logging

logger = logging.getLogger(__name__)

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DQNNetwork(nn.Module):
    """Neural network for DQN agent."""
    
    def __init__(self, state_size, action_size, hidden_size=256):
        """
        Initialize DQN network.
        
        Args:
            state_size (int): Dimension of state space
            action_size (int): Number of possible actions
            hidden_size (int): Size of hidden layers
        """
        super(DQNNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, state):
        """Forward pass through the network."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, buffer_size=10000):
        """
        Initialize replay buffer.
        
        Args:
            buffer_size (int): Maximum size of buffer
        """
        self.buffer = deque(maxlen=buffer_size)
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """Sample a batch of experiences."""
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float()
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return buffer size."""
        return len(self.buffer)


class DQNAgent:
    """DQN agent for tutorial action selection."""
    
    def __init__(self, state_size, action_size, config=None):
        """
        Initialize DQN agent.
        
        Args:
            state_size (int): Dimension of state space
            action_size (int): Number of possible actions
            config (dict): Configuration parameters
        """
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters - ensure proper type conversion
        self.lr = float(config.get('learning_rate', 1e-3)) if config else 1e-3
        self.gamma = float(config.get('gamma', 0.99)) if config else 0.99
        self.epsilon = float(config.get('epsilon_start', 1.0)) if config else 1.0
        self.epsilon_min = float(config.get('epsilon_min', 0.01)) if config else 0.01
        self.epsilon_decay = float(config.get('epsilon_decay', 0.995)) if config else 0.995
        self.batch_size = int(config.get('batch_size', 32)) if config else 32
        self.update_freq = int(config.get('update_freq', 4)) if config else 4
        self.target_update_freq = int(config.get('target_update_freq', 100)) if config else 100
        
        # Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Training variables
        self.step_count = 0
        self.training_scores = []
        
        logger.info(f"DQN Agent initialized with {self.device}")
    
    def act(self, state, training=True):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state (np.array): Current state
            training (bool): Whether in training mode
            
        Returns:
            int: Selected action
        """
        if training and random.random() <= self.epsilon:
            return random.choice(range(self.action_size))
        
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.cpu().data.numpy().argmax()
    
    def step(self, state, action, reward, next_state, done):
        """
        Save experience and learn if enough samples are available.
        
        Args:
            state (np.array): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.array): Next state
            done (bool): Whether episode is finished
        """
        # Save experience
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        # Learn every update_freq steps
        self.step_count += 1
        if self.step_count % self.update_freq == 0:
            if len(self.replay_buffer) > self.batch_size:
                self.learn()
        
        # Update target network
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()
    
    def learn(self):
        """Update Q-network using experience replay."""
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions)
        
        # Next Q values from target network
        next_q_values = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filepath):
        """Save model weights."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, filepath)
        logger.info(f"DQN model saved to {filepath}")
    
    def load(self, filepath):
        """Load model weights."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        logger.info(f"DQN model loaded from {filepath}")
    
    def get_metrics(self):
        """Get training metrics."""
        return {
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'buffer_size': len(self.replay_buffer),
            'training_scores': self.training_scores
        }
