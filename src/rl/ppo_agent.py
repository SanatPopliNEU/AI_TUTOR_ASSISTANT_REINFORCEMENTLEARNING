"""
Proximal Policy Optimization (PPO) implementation for tutorial strategy learning.

This module implements a PPO agent that learns optimal teaching policies
through continuous interaction with students.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import logging
from collections import namedtuple

logger = logging.getLogger(__name__)

# Experience tuple for PPO
PPOExperience = namedtuple('PPOExperience', 
                          ['state', 'action', 'log_prob', 'reward', 'value', 'done'])


class PPONetwork(nn.Module):
    """Actor-Critic network for PPO agent."""
    
    def __init__(self, state_size, action_size, hidden_size=256):
        """
        Initialize PPO network.
        
        Args:
            state_size (int): Dimension of state space
            action_size (int): Number of possible actions
            hidden_size (int): Size of hidden layers
        """
        super(PPONetwork, self).__init__()
        
        # Shared layers
        self.shared_fc1 = nn.Linear(state_size, hidden_size)
        self.shared_fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Actor head (policy)
        self.actor_fc = nn.Linear(hidden_size, hidden_size)
        self.actor_out = nn.Linear(hidden_size, action_size)
        
        # Critic head (value function)
        self.critic_fc = nn.Linear(hidden_size, hidden_size)
        self.critic_out = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.fill_(0.0)
    
    def forward(self, state):
        """Forward pass through the network."""
        # Shared layers
        x = F.relu(self.shared_fc1(state))
        x = F.relu(self.shared_fc2(x))
        
        # Actor (policy) output
        actor_x = F.relu(self.actor_fc(x))
        action_logits = self.actor_out(actor_x)
        
        # Critic (value) output
        critic_x = F.relu(self.critic_fc(x))
        value = self.critic_out(critic_x)
        
        return action_logits, value
    
    def get_action_and_value(self, state):
        """Get action and value for given state."""
        action_logits, value = self.forward(state)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, value.squeeze()
    
    def evaluate_actions(self, states, actions):
        """Evaluate actions for given states."""
        action_logits, values = self.forward(states)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values.squeeze(), entropy


class PPOBuffer:
    """Buffer for storing PPO experiences."""
    
    def __init__(self, buffer_size=2048):
        """
        Initialize PPO buffer.
        
        Args:
            buffer_size (int): Maximum size of buffer
        """
        self.buffer_size = buffer_size
        self.clear()
    
    def clear(self):
        """Clear the buffer."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.advantages = []
        self.returns = []
    
    def add(self, state, action, log_prob, reward, value, done):
        """Add experience to buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_advantages(self, gamma=0.99, lambda_gae=0.95):
        """Compute advantages using GAE (Generalized Advantage Estimation)."""
        advantages = []
        returns = []
        
        # Convert to numpy arrays
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        
        # Compute advantages and returns
        advantage = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
                next_done = 1
            else:
                next_value = values[t + 1]
                next_done = dones[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - next_done) - values[t]
            advantage = delta + gamma * lambda_gae * (1 - next_done) * advantage
            advantages.insert(0, advantage)
            returns.insert(0, advantage + values[t])
        
        self.advantages = advantages
        self.returns = returns
    
    def get_batch_data(self):
        """Get all data as tensors."""
        return {
            'states': torch.FloatTensor(self.states),
            'actions': torch.LongTensor(self.actions),
            'log_probs': torch.FloatTensor(self.log_probs),
            'advantages': torch.FloatTensor(self.advantages),
            'returns': torch.FloatTensor(self.returns),
            'values': torch.FloatTensor(self.values)
        }
    
    def __len__(self):
        """Return buffer size."""
        return len(self.states)


class PPOAgent:
    """PPO agent for tutorial strategy learning."""
    
    def __init__(self, state_size, action_size, config=None):
        """
        Initialize PPO agent.
        
        Args:
            state_size (int): Dimension of state space
            action_size (int): Number of possible actions
            config (dict): Configuration parameters
        """
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters - ensure proper type conversion
        self.lr = float(config.get('learning_rate', 3e-4)) if config else 3e-4
        self.gamma = float(config.get('gamma', 0.99)) if config else 0.99
        self.lambda_gae = float(config.get('lambda_gae', 0.95)) if config else 0.95
        self.clip_epsilon = float(config.get('clip_epsilon', 0.2)) if config else 0.2
        self.value_coef = float(config.get('value_coef', 0.5)) if config else 0.5
        self.entropy_coef = float(config.get('entropy_coef', 0.01)) if config else 0.01
        self.max_grad_norm = float(config.get('max_grad_norm', 0.5)) if config else 0.5
        self.ppo_epochs = int(config.get('ppo_epochs', 4)) if config else 4
        self.batch_size = int(config.get('batch_size', 64)) if config else 64
        self.buffer_size = int(config.get('buffer_size', 2048)) if config else 2048
        
        # Network and optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = PPONetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
        
        # Experience buffer
        self.buffer = PPOBuffer(self.buffer_size)
        
        # Training variables
        self.episode_count = 0
        self.training_rewards = []
        self.policy_losses = []
        self.value_losses = []
        
        logger.info(f"PPO Agent initialized with {self.device}")
    
    def act(self, state, training=True):
        """
        Select action using current policy.
        
        Args:
            state (np.array): Current state
            training (bool): Whether in training mode
            
        Returns:
            tuple: (action, log_prob, value) if training, else action only
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if training:
                action, log_prob, value = self.network.get_action_and_value(state_tensor)
                return action, log_prob.item(), value.item()
            else:
                action_logits, _ = self.network(state_tensor)
                action_probs = F.softmax(action_logits, dim=-1)
                action = action_probs.argmax().item()
                return action
    
    def store_experience(self, state, action, log_prob, reward, value, done):
        """Store experience in buffer."""
        self.buffer.add(state, action, log_prob, reward, value, done)
    
    def update(self):
        """Update policy using PPO algorithm."""
        if len(self.buffer) < self.buffer_size:
            return
        
        # Compute advantages
        self.buffer.compute_advantages(self.gamma, self.lambda_gae)
        
        # Get batch data
        batch_data = self.buffer.get_batch_data()
        
        # Move to device
        for key in batch_data:
            batch_data[key] = batch_data[key].to(self.device)
        
        # Normalize advantages
        advantages = batch_data['advantages']
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.ppo_epochs):
            # Get current policy outputs
            log_probs, values, entropy = self.network.evaluate_actions(
                batch_data['states'], batch_data['actions']
            )
            
            # Compute ratios
            ratios = torch.exp(log_probs - batch_data['log_probs'])
            
            # Policy loss with clipping
            policy_loss1 = ratios * advantages
            policy_loss2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, batch_data['returns'])
            
            # Entropy loss
            entropy_loss = -entropy.mean()
            
            # Total loss
            total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Optimize
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Store losses
            self.policy_losses.append(policy_loss.item())
            self.value_losses.append(value_loss.item())
        
        # Clear buffer
        self.buffer.clear()
        
        logger.debug(f"PPO update completed. Policy loss: {policy_loss.item():.4f}, "
                    f"Value loss: {value_loss.item():.4f}")
    
    def save(self, filepath):
        """Save model weights."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_count': self.episode_count,
            'training_rewards': self.training_rewards
        }, filepath)
        logger.info(f"PPO model saved to {filepath}")
    
    def load(self, filepath):
        """Load model weights."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_count = checkpoint['episode_count']
        self.training_rewards = checkpoint['training_rewards']
        logger.info(f"PPO model loaded from {filepath}")
    
    def get_metrics(self):
        """Get training metrics."""
        return {
            'episode_count': self.episode_count,
            'avg_policy_loss': np.mean(self.policy_losses[-100:]) if self.policy_losses else 0,
            'avg_value_loss': np.mean(self.value_losses[-100:]) if self.value_losses else 0,
            'training_rewards': self.training_rewards,
            'buffer_size': len(self.buffer)
        }
