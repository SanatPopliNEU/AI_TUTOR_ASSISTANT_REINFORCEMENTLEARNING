"""
Simplified strategy agent without PyTorch dependencies.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple
from rl.simple_agents import SimplePPOAgent
from environment.tutoring_environment import ActionType, DifficultyLevel

logger = logging.getLogger(__name__)


class SimpleStrategyAgent:
    """Simplified agent for strategic decisions without PyTorch."""
    
    def __init__(self, state_size: int, config: Dict = None):
        """
        Initialize simplified strategy agent.
        
        Args:
            state_size (int): Size of state vector
            config (Dict): Configuration parameters
        """
        self.state_size = state_size
        self.config = config or {}
        
        # Strategy-specific actions
        self.strategy_actions = [
            ActionType.INCREASE_DIFFICULTY,
            ActionType.DECREASE_DIFFICULTY,
            ActionType.ENCOURAGE,
            ActionType.TAKE_BREAK
        ]
        
        # Initialize PPO agent
        self.ppo_agent = SimplePPOAgent(
            state_size=state_size,
            action_size=len(self.strategy_actions),
            lr=self.config.get('learning_rate', 0.01)
        )
        
        # Batch collection for PPO
        self.batch_states = []
        self.batch_actions = []
        self.batch_rewards = []
        self.batch_next_states = []
        self.batch_dones = []
        
        self.batch_size = self.config.get('batch_size', 10)
        
        logger.info(f"SimpleStrategyAgent initialized with {len(self.strategy_actions)} actions")
    
    def get_action(self, state: np.ndarray) -> ActionType:
        """
        Get strategic action.
        
        Args:
            state (np.ndarray): Current state
            
        Returns:
            ActionType: Selected action
        """
        action_idx = self.ppo_agent.act(state)
        return self.strategy_actions[action_idx]
    
    def update_from_experience(self, state: np.ndarray, action: ActionType, 
                             reward: float, next_state: np.ndarray, done: bool = False) -> bool:
        """
        Update from experience (batch-based for PPO).
        
        Args:
            state (np.ndarray): Previous state
            action (ActionType): Action taken
            reward (float): Reward received
            next_state (np.ndarray): New state
            done (bool): Whether episode is finished
            
        Returns:
            bool: True if update was successful
        """
        try:
            # Convert action to index
            if action not in self.strategy_actions:
                logger.warning(f"Action {action} not in strategy actions")
                return False
            
            action_idx = self.strategy_actions.index(action)
            
            # Add to batch
            self.batch_states.append(state)
            self.batch_actions.append(action_idx)
            self.batch_rewards.append(reward)
            self.batch_next_states.append(next_state)
            self.batch_dones.append(done)
            
            # Update when batch is full or episode is done
            if len(self.batch_states) >= self.batch_size or done:
                success = self.ppo_agent.update(
                    self.batch_states,
                    self.batch_actions,
                    self.batch_rewards,
                    self.batch_next_states,
                    self.batch_dones
                )
                
                # Clear batch
                self.batch_states = []
                self.batch_actions = []
                self.batch_rewards = []
                self.batch_next_states = []
                self.batch_dones = []
                
                return success
            
            return True  # Successfully added to batch
                
        except Exception as e:
            logger.error(f"Strategy agent update error: {e}")
            return False
    
    def get_training_info(self) -> Dict:
        """Get training information."""
        return {
            'agent_type': 'SimpleStrategyAgent',
            'batch_size': len(self.batch_states),
            **self.ppo_agent.get_training_info()
        }
