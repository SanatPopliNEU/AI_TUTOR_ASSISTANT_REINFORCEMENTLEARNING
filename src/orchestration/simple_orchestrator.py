"""
Simplified Tutorial Orchestrator without PyTorch dependencies.
"""

import numpy as np
import logging
import time
from typing import Dict, List, Any, Tuple
from pathlib import Path

from agents.simple_content_agent import SimpleContentAgent
from agents.simple_strategy_agent import SimpleStrategyAgent
from environment.tutoring_environment import TutoringEnvironment, ActionType

logger = logging.getLogger(__name__)


class SimpleTutorialOrchestrator:
    """Simplified orchestrator without PyTorch dependencies."""
    
    def __init__(self, environment: TutoringEnvironment, config: Dict = None):
        """
        Initialize simplified tutorial orchestrator.
        
        Args:
            environment (TutoringEnvironment): The tutoring environment
            config (Dict): Configuration parameters
        """
        self.environment = environment
        self.config = config or {}
        
        # Initialize simplified agents
        state_size = environment.state_size
        
        self.content_agent = SimpleContentAgent(
            state_size=state_size,
            config=self.config.get('content_agent', {})
        )
        
        self.strategy_agent = SimpleStrategyAgent(
            state_size=state_size,
            config=self.config.get('strategy_agent', {})
        )
        
        # Coordination parameters
        self.coordination_mode = self.config.get('coordination_mode', 'competitive')
        self.content_weight = self.config.get('content_weight', 0.6)
        self.strategy_weight = self.config.get('strategy_weight', 0.4)
        
        # Training tracking
        self.total_training_updates = 0
        self.content_updates = 0
        self.strategy_updates = 0
        self._last_acting_agent = None
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        
        logger.info("SimpleTutorialOrchestrator initialized")
    
    def get_next_action(self, state: np.ndarray) -> ActionType:
        """
        Get next tutoring action from agents.
        
        Args:
            state (np.ndarray): Current state
            
        Returns:
            ActionType: Selected action
        """
        try:
            if self.coordination_mode == 'competitive':
                # Agents compete based on weights
                if np.random.random() < self.content_weight:
                    action = self.content_agent.get_action(state)
                    self._last_acting_agent = 'content'
                else:
                    action = self.strategy_agent.get_action(state)
                    self._last_acting_agent = 'strategy'
            
            elif self.coordination_mode == 'collaborative':
                # Simple collaboration: alternate based on state
                engagement = state[7] if len(state) > 7 else 0.5  # Engagement index
                
                if engagement < 0.4:
                    # Low engagement - use strategy agent
                    action = self.strategy_agent.get_action(state)
                    self._last_acting_agent = 'strategy'
                else:
                    # Good engagement - use content agent
                    action = self.content_agent.get_action(state)
                    self._last_acting_agent = 'content'
            
            else:  # alternating
                # Simple alternation
                if self._last_acting_agent == 'content':
                    action = self.strategy_agent.get_action(state)
                    self._last_acting_agent = 'strategy'
                else:
                    action = self.content_agent.get_action(state)
                    self._last_acting_agent = 'content'
            
            logger.debug(f"Action selected: {action} by {self._last_acting_agent} agent")
            return action
            
        except Exception as e:
            logger.error(f"Error getting next action: {e}")
            # Fallback to asking a question
            return ActionType.ASK_QUESTION
    
    def update_from_experience(self, state: np.ndarray, action: ActionType, 
                             reward: float, next_state: np.ndarray, done: bool = False) -> bool:
        """
        Update agents from experience.
        
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
            success = False
            
            # DEBUG: Log the action and its type
            logger.info(f"🔍 UPDATE DEBUG: action={action}, type={type(action)}")
            
            # Update the appropriate agent based on action type
            content_actions = [ActionType.ASK_QUESTION, ActionType.PROVIDE_HINT, 
                             ActionType.EXPLAIN_CONCEPT, ActionType.REVIEW_PREVIOUS]
            
            logger.info(f"🔍 Content actions: {content_actions}")
            logger.info(f"🔍 Action in content actions: {action in content_actions}")
            
            if action in content_actions:
                # Update content agent using update_from_feedback
                logger.info("🎯 Updating CONTENT agent")
                success = self.content_agent.update_from_feedback(
                    state, action, reward, next_state, done
                )
                if success:
                    self.content_updates += 1
                    logger.info(f"✅ Content agent updated (total: {self.content_updates})")
                else:
                    logger.warning("❌ Content agent update failed")
            
            else:
                # Update strategy agent using update_from_experience
                logger.info("🎯 Updating STRATEGY agent")
                success = self.strategy_agent.update_from_experience(
                    state, action, reward, next_state, done
                )
                if success:
                    self.strategy_updates += 1
                    logger.info(f"✅ Strategy agent updated (total: {self.strategy_updates})")
                else:
                    logger.warning("❌ Strategy agent update failed")
            
            if success:
                self.total_training_updates += 1
                logger.info(f"🚀 TOTAL TRAINING UPDATES: {self.total_training_updates}")
            else:
                logger.warning("⚠️ No training update occurred")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error updating from experience: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def run_episode(self, max_steps: int = 50) -> Dict:
        """
        Run a complete tutoring episode.
        
        Args:
            max_steps (int): Maximum steps per episode
            
        Returns:
            Dict: Episode results
        """
        try:
            state = self.environment.reset()
            total_reward = 0
            steps = 0
            
            for step in range(max_steps):
                # Get action
                action = self.get_next_action(state)
                
                # Execute action
                next_state, reward, done, info = self.environment.step(action.value)
                
                # Update agents
                self.update_from_experience(state, action, reward, next_state, done)
                
                # Update tracking
                total_reward += reward
                steps += 1
                state = next_state
                
                if done:
                    break
            
            # Store episode results
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            
            results = {
                'total_reward': total_reward,
                'steps': steps,
                'training_updates': self.total_training_updates,
                'content_updates': self.content_updates,
                'strategy_updates': self.strategy_updates
            }
            
            logger.info(f"Episode completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error running episode: {e}")
            return {'error': str(e)}
    
    def get_training_stats(self) -> Dict:
        """Get comprehensive training statistics."""
        content_info = self.content_agent.get_training_info()
        strategy_info = self.strategy_agent.get_training_info()
        
        return {
            'orchestrator': {
                'total_updates': self.total_training_updates,
                'content_updates': self.content_updates,
                'strategy_updates': self.strategy_updates,
                'coordination_mode': self.coordination_mode,
                'last_acting_agent': self._last_acting_agent,
                'episodes_run': len(self.episode_rewards)
            },
            'content_agent': content_info,
            'strategy_agent': strategy_info,
            'performance': {
                'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
                'avg_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
                'total_episodes': len(self.episode_rewards)
            }
        }
