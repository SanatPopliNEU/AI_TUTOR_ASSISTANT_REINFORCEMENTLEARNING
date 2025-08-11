"""
Tutorial Content Agent - Specialized agent for content delivery and question selection.

This agent uses reinforcement learning to optimize question sequencing,
difficulty progression, and content delivery strategies.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple
from rl.dqn_agent import DQNAgent
from environment.tutoring_environment import ActionType, DifficultyLevel

logger = logging.getLogger(__name__)


class TutorialContentAgent:
    """Agent responsible for content delivery and question selection."""
    
    def __init__(self, state_size: int, config: Dict = None):
        """
        Initialize tutorial content agent.
        
        Args:
            state_size (int): Size of state space
            config (Dict): Configuration parameters
        """
        self.config = config or {}
        
        # Define action space for content delivery
        self.content_actions = [
            ActionType.ASK_QUESTION,
            ActionType.PROVIDE_HINT,
            ActionType.EXPLAIN_CONCEPT,
            ActionType.REVIEW_PREVIOUS
        ]
        
        # Initialize DQN agent for content decisions
        # Enhanced state size accounts for additional content-specific features
        enhanced_state_size = state_size + 6  # 6 additional features from _enhance_state_for_content
        self.dqn_agent = DQNAgent(
            state_size=enhanced_state_size,
            action_size=len(self.content_actions),
            config=config.get('dqn', {})
        )
        
        # Agent memory and metrics
        self.decision_history = []
        self.performance_metrics = {
            'questions_asked': 0,
            'hints_provided': 0,
            'concepts_explained': 0,
            'reviews_conducted': 0,
            'success_rate': 0.0
        }
        
        logger.info("Tutorial Content Agent initialized")
    
    def select_content_action(self, state: np.ndarray, student_metrics: Dict, training: bool = True) -> int:
        """
        Select content delivery action based on current state.
        
        Args:
            state (np.ndarray): Current environment state
            student_metrics (Dict): Student performance metrics
            training (bool): Whether in training mode
            
        Returns:
            int: Selected action index
        """
        # Get enhanced state with content-specific features
        enhanced_state = self._enhance_state_for_content(state, student_metrics)
        
        # Select action using DQN
        action_idx = self.dqn_agent.act(enhanced_state, training=training)
        selected_action = self.content_actions[action_idx]
        
        # Update metrics
        self._update_action_metrics(selected_action)
        
        # Log decision
        self.decision_history.append({
            'action': selected_action.name,
            'engagement': student_metrics.get('engagement', 0),
            'motivation': student_metrics.get('motivation', 0),
            'knowledge_level': np.mean(list(student_metrics.get('knowledge_levels', {}).values()))
        })
        
        logger.debug(f"Content agent selected action: {selected_action.name}")
        
        return selected_action.value
    
    def _enhance_state_for_content(self, base_state: np.ndarray, student_metrics: Dict) -> np.ndarray:
        """
        Enhance state with content-specific features.
        
        Args:
            base_state (np.ndarray): Base environment state
            student_metrics (Dict): Student metrics
            
        Returns:
            np.ndarray: Enhanced state vector
        """
        # Content-specific features
        content_features = []
        
        # Recent performance patterns
        recent_decisions = self.decision_history[-5:] if len(self.decision_history) >= 5 else self.decision_history
        
        # Action frequency in recent history
        if recent_decisions:
            question_freq = sum(1 for d in recent_decisions if 'QUESTION' in d['action']) / len(recent_decisions)
            hint_freq = sum(1 for d in recent_decisions if 'HINT' in d['action']) / len(recent_decisions)
            explain_freq = sum(1 for d in recent_decisions if 'EXPLAIN' in d['action']) / len(recent_decisions)
            review_freq = sum(1 for d in recent_decisions if 'REVIEW' in d['action']) / len(recent_decisions)
        else:
            question_freq = hint_freq = explain_freq = review_freq = 0.0
        
        content_features.extend([question_freq, hint_freq, explain_freq, review_freq])
        
        # Learning progression indicators
        knowledge_variance = np.var(list(student_metrics.get('knowledge_levels', {}).values()))
        avg_knowledge = np.mean(list(student_metrics.get('knowledge_levels', {}).values()))
        
        content_features.extend([knowledge_variance, avg_knowledge])
        
        # Combine with base state
        enhanced_state = np.concatenate([base_state, content_features])
        
        return enhanced_state
    
    def _update_action_metrics(self, action: ActionType):
        """Update performance metrics based on action."""
        if action == ActionType.ASK_QUESTION:
            self.performance_metrics['questions_asked'] += 1
        elif action == ActionType.PROVIDE_HINT:
            self.performance_metrics['hints_provided'] += 1
        elif action == ActionType.EXPLAIN_CONCEPT:
            self.performance_metrics['concepts_explained'] += 1
        elif action == ActionType.REVIEW_PREVIOUS:
            self.performance_metrics['reviews_conducted'] += 1
    
    def update_from_feedback(self, state: np.ndarray, action: int, reward: float, 
                           next_state: np.ndarray, done: bool, student_metrics: Dict):
        """
        Update agent based on environment feedback.
        
        Args:
            state (np.ndarray): Previous state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.ndarray): New state
            done (bool): Whether episode ended
            student_metrics (Dict): Student performance metrics
        """
        # Enhance states for content-specific learning
        enhanced_state = self._enhance_state_for_content(state, student_metrics)
        enhanced_next_state = self._enhance_state_for_content(next_state, student_metrics)
        
        # Map environment action back to content action index
        content_action_idx = None
        for idx, content_action in enumerate(self.content_actions):
            if content_action.value == action:
                content_action_idx = idx
                break
        
        if content_action_idx is not None:
            # Update DQN agent
            self.dqn_agent.step(enhanced_state, content_action_idx, reward, 
                              enhanced_next_state, done)
    
    def get_q_value(self, state: np.ndarray, action: int) -> float:
        """
        Get Q-value for a specific state-action pair.
        
        Args:
            state (np.ndarray): State
            action (int): Action (environment action, not DQN action index)
            
        Returns:
            float: Q-value for the state-action pair
        """
        try:
            # Find the content action index corresponding to the environment action
            content_action_idx = None
            for idx, content_action in enumerate(self.content_actions):
                if content_action.value == action:
                    content_action_idx = idx
                    break
            
            if content_action_idx is not None:
                # Get Q-values from DQN agent
                import torch
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_values = self.dqn_agent.qnetwork_local(state_tensor)
                    return float(q_values[0][content_action_idx])
            else:
                # Return a default Q-value if action not found
                return 0.0
        except Exception:
            # Return default Q-value if there's any error
            return 0.0
    
    def calculate_content_effectiveness(self) -> float:
        """
        Calculate effectiveness of content delivery strategy.
        
        Returns:
            float: Effectiveness score (0-1)
        """
        if not self.decision_history:
            return 0.0
        
        # Analyze recent decisions
        recent_decisions = self.decision_history[-10:]
        
        # Effectiveness based on engagement maintenance
        avg_engagement = np.mean([d['engagement'] for d in recent_decisions])
        
        # Effectiveness based on knowledge growth
        if len(recent_decisions) > 1:
            knowledge_growth = (recent_decisions[-1]['knowledge_level'] - 
                              recent_decisions[0]['knowledge_level'])
            knowledge_effectiveness = max(0, knowledge_growth * 10)  # Scale up
        else:
            knowledge_effectiveness = 0.0
        
        # Combine metrics
        overall_effectiveness = (avg_engagement + knowledge_effectiveness) / 2
        return min(1.0, max(0.0, overall_effectiveness))
    
    def get_content_recommendations(self, student_metrics: Dict) -> List[str]:
        """
        Get content recommendations based on current student state.
        
        Args:
            student_metrics (Dict): Current student metrics
            
        Returns:
            List[str]: Content recommendations
        """
        recommendations = []
        
        engagement = student_metrics.get('engagement', 0.5)
        motivation = student_metrics.get('motivation', 0.5)
        avg_knowledge = np.mean(list(student_metrics.get('knowledge_levels', {}).values()))
        
        # Engagement-based recommendations
        if engagement < 0.3:
            recommendations.append("Consider taking a break or providing encouragement")
            recommendations.append("Switch to easier, more engaging content")
        elif engagement > 0.8:
            recommendations.append("Student is highly engaged - good time for challenging content")
        
        # Knowledge-based recommendations
        if avg_knowledge < 0.4:
            recommendations.append("Focus on foundational concepts and explanations")
            recommendations.append("Provide more hints and scaffolding")
        elif avg_knowledge > 0.7:
            recommendations.append("Introduce advanced topics and complex problems")
        
        # Motivation-based recommendations
        if motivation < 0.4:
            recommendations.append("Provide positive reinforcement and encouragement")
            recommendations.append("Review recent successes")
        
        return recommendations
    
    def save(self, filepath: str):
        """Save agent state."""
        self.dqn_agent.save(filepath)
        logger.info(f"Content agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent state."""
        self.dqn_agent.load(filepath)
        logger.info(f"Content agent loaded from {filepath}")
    
    def get_metrics(self) -> Dict:
        """Get agent metrics."""
        metrics = self.performance_metrics.copy()
        metrics.update({
            'content_effectiveness': self.calculate_content_effectiveness(),
            'decision_count': len(self.decision_history),
            'dqn_metrics': self.dqn_agent.get_metrics()
        })
        return metrics
