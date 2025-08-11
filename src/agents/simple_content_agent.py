"""
Simplified content agent without PyTorch dependencies.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple
from rl.simple_agents import SimpleDQNAgent
from environment.tutoring_environment import ActionType, DifficultyLevel

logger = logging.getLogger(__name__)


class SimpleContentAgent:
    """Simplified agent for content delivery without PyTorch."""
    
    def __init__(self, state_size: int, config: Dict = None):
        """
        Initialize simplified content agent.
        
        Args:
            state_size (int): Size of state vector
            config (Dict): Configuration parameters
        """
        self.state_size = state_size
        self.config = config or {}
        
        # Content-specific actions
        self.content_actions = [
            ActionType.ASK_QUESTION,
            ActionType.PROVIDE_HINT,
            ActionType.EXPLAIN_CONCEPT,
            ActionType.REVIEW_PREVIOUS
        ]
        
        # Initialize DQN agent
        self.dqn_agent = SimpleDQNAgent(
            state_size=state_size,
            action_size=len(self.content_actions),
            lr=self.config.get('learning_rate', 0.1),
            epsilon=self.config.get('epsilon', 0.1)
        )
        
        logger.info(f"SimpleContentAgent initialized with {len(self.content_actions)} actions")
    
    def get_action(self, state: np.ndarray) -> ActionType:
        """
        Get content delivery action.
        
        Args:
            state (np.ndarray): Current state
            
        Returns:
            ActionType: Selected action
        """
        action_idx = self.dqn_agent.act(state)
        return self.content_actions[action_idx]
    
    def update_from_feedback(self, state: np.ndarray, action: ActionType, 
                           reward: float, next_state: np.ndarray, done: bool = False) -> bool:
        """
        Update from student feedback.
        
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
            if action in self.content_actions:
                action_idx = self.content_actions.index(action)
                return self.dqn_agent.step(state, action_idx, reward, next_state, done)
            else:
                logger.warning(f"Action {action} not in content actions")
                return False
                
        except Exception as e:
            logger.error(f"Content agent update error: {e}")
            return False
    
    def get_training_info(self) -> Dict:
        """Get training information."""
        return {
            'agent_type': 'SimpleContentAgent',
            **self.dqn_agent.get_training_info()
        }
