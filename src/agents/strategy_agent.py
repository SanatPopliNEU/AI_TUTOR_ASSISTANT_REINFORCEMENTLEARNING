"""
Tutorial Strategy Agent - Specialized agent for adaptive teaching strategy.

This agent uses PPO to learn optimal high-level teaching strategies,
including difficulty adjustment, pacing, and motivation management.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple
from rl.ppo_agent import PPOAgent
from environment.tutoring_environment import ActionType

logger = logging.getLogger(__name__)


class TutorialStrategyAgent:
    """Agent responsible for high-level teaching strategy and adaptation."""
    
    def __init__(self, state_size: int, config: Dict = None):
        """
        Initialize tutorial strategy agent.
        
        Args:
            state_size (int): Size of state space
            config (Dict): Configuration parameters
        """
        self.config = config or {}
        
        # Define action space for strategic decisions
        self.strategy_actions = [
            ActionType.INCREASE_DIFFICULTY,
            ActionType.DECREASE_DIFFICULTY,
            ActionType.ENCOURAGE,
            ActionType.TAKE_BREAK
        ]
        
        # Initialize PPO agent for strategy decisions
        self.ppo_agent = PPOAgent(
            state_size=state_size + 8,  # Enhanced state with strategy features
            action_size=len(self.strategy_actions),
            config=config.get('ppo', {})
        )
        
        # Strategy memory and adaptation tracking
        self.adaptation_history = []
        self.strategy_metrics = {
            'difficulty_adjustments': 0,
            'encouragements_given': 0,
            'breaks_suggested': 0,
            'adaptation_effectiveness': 0.0,
            'student_progress_rate': 0.0
        }
        
        # Learning analytics
        self.session_analytics = {
            'initial_knowledge': 0.0,
            'current_knowledge': 0.0,
            'initial_motivation': 0.0,
            'current_motivation': 0.0,
            'total_adaptations': 0,
            'successful_adaptations': 0
        }
        
        logger.info("Tutorial Strategy Agent initialized")
    
    def select_strategy_action(self, state: np.ndarray, student_metrics: Dict, 
                             content_effectiveness: float, training: bool = True) -> Tuple[int, float, float]:
        """
        Select strategic teaching action based on current state.
        
        Args:
            state (np.ndarray): Current environment state
            student_metrics (Dict): Student performance metrics
            content_effectiveness (float): Effectiveness of content delivery
            training (bool): Whether in training mode
            
        Returns:
            Tuple[int, float, float]: (action, log_prob, value) for training, action only for inference
        """
        # Enhance state with strategy-specific features
        enhanced_state = self._enhance_state_for_strategy(
            state, student_metrics, content_effectiveness
        )
        
        # Select action using PPO
        if training:
            action_idx, log_prob, value = self.ppo_agent.act(enhanced_state, training=True)
            selected_action = self.strategy_actions[action_idx]
            
            # Store experience for later update
            self.current_experience = {
                'state': enhanced_state,
                'action': action_idx,
                'log_prob': log_prob,
                'value': value
            }
            
            return selected_action.value, log_prob, value
        else:
            action_idx = self.ppo_agent.act(enhanced_state, training=False)
            selected_action = self.strategy_actions[action_idx]
            return selected_action.value
    
    def _enhance_state_for_strategy(self, base_state: np.ndarray, student_metrics: Dict, 
                                  content_effectiveness: float) -> np.ndarray:
        """
        Enhance state with strategy-specific features.
        
        Args:
            base_state (np.ndarray): Base environment state
            student_metrics (Dict): Student metrics
            content_effectiveness (float): Content delivery effectiveness
            
        Returns:
            np.ndarray: Enhanced state vector
        """
        strategy_features = []
        
        # Adaptation trend analysis
        recent_adaptations = self.adaptation_history[-5:] if len(self.adaptation_history) >= 5 else self.adaptation_history
        
        if recent_adaptations:
            # Trend in student engagement
            engagement_trend = self._calculate_trend([a['engagement_after'] for a in recent_adaptations])
            
            # Trend in motivation
            motivation_trend = self._calculate_trend([a['motivation_after'] for a in recent_adaptations])
            
            # Recent adaptation effectiveness
            recent_effectiveness = np.mean([a['effectiveness'] for a in recent_adaptations])
            
            # Time since last adaptation
            time_since_adaptation = len(self.adaptation_history) - len(recent_adaptations)
        else:
            engagement_trend = 0.0
            motivation_trend = 0.0
            recent_effectiveness = 0.0
            time_since_adaptation = 0.0
        
        strategy_features.extend([
            engagement_trend,
            motivation_trend,
            recent_effectiveness,
            time_since_adaptation / 10.0,  # Normalized
            content_effectiveness
        ])
        
        # Student learning patterns
        knowledge_levels = list(student_metrics.get('knowledge_levels', {}).values())
        if knowledge_levels:
            knowledge_std = np.std(knowledge_levels)  # Learning consistency
            knowledge_max = np.max(knowledge_levels)  # Peak knowledge
            knowledge_min = np.min(knowledge_levels)  # Weakest area
        else:
            knowledge_std = knowledge_max = knowledge_min = 0.0
        
        strategy_features.extend([knowledge_std, knowledge_max, knowledge_min])
        
        # Combine with base state
        enhanced_state = np.concatenate([base_state, strategy_features])
        
        return enhanced_state
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in a series of values."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        return np.clip(slope, -1.0, 1.0)  # Normalize
    
    def update_from_feedback(self, reward: float, next_state: np.ndarray, done: bool, 
                           student_metrics: Dict, content_effectiveness: float):
        """
        Update strategy agent based on feedback.
        
        Args:
            reward (float): Reward received
            next_state (np.ndarray): New state
            done (bool): Whether episode ended
            student_metrics (Dict): Student metrics
            content_effectiveness (float): Content effectiveness
        """
        if hasattr(self, 'current_experience'):
            # Enhance next state
            enhanced_next_state = self._enhance_state_for_strategy(
                next_state, student_metrics, content_effectiveness
            )
            
            # Store experience in PPO buffer
            self.ppo_agent.store_experience(
                self.current_experience['state'],
                self.current_experience['action'],
                self.current_experience['log_prob'],
                reward,
                self.current_experience['value'],
                done
            )
            
            # Record adaptation
            self._record_adaptation(reward, student_metrics)
            
            # Update policy if buffer is full
            if done or len(self.ppo_agent.buffer) >= self.ppo_agent.buffer_size:
                self.ppo_agent.update()
    
    def _record_adaptation(self, reward: float, student_metrics: Dict):
        """Record adaptation attempt and its effectiveness."""
        adaptation_record = {
            'engagement_after': student_metrics.get('engagement', 0),
            'motivation_after': student_metrics.get('motivation', 0),
            'knowledge_after': np.mean(list(student_metrics.get('knowledge_levels', {}).values())),
            'reward': reward,
            'effectiveness': max(0, reward / 10.0)  # Normalize reward to effectiveness
        }
        
        self.adaptation_history.append(adaptation_record)
        
        # Update session analytics
        self.session_analytics['total_adaptations'] += 1
        if reward > 0:
            self.session_analytics['successful_adaptations'] += 1
        
        # Update strategy metrics
        self.strategy_metrics['adaptation_effectiveness'] = (
            self.session_analytics['successful_adaptations'] / 
            max(1, self.session_analytics['total_adaptations'])
        )
    
    def analyze_student_progress(self, initial_metrics: Dict, current_metrics: Dict) -> Dict:
        """
        Analyze student progress throughout the session.
        
        Args:
            initial_metrics (Dict): Initial student state
            current_metrics (Dict): Current student state
            
        Returns:
            Dict: Progress analysis
        """
        # Knowledge progress
        initial_knowledge = np.mean(list(initial_metrics.get('knowledge_levels', {}).values()))
        current_knowledge = np.mean(list(current_metrics.get('knowledge_levels', {}).values()))
        knowledge_growth = current_knowledge - initial_knowledge
        
        # Motivation change
        motivation_change = (current_metrics.get('motivation', 0) - 
                           initial_metrics.get('motivation', 0))
        
        # Engagement sustainability
        current_engagement = current_metrics.get('engagement', 0)
        
        # Calculate progress rate
        session_length = current_metrics.get('session_length', 1)
        progress_rate = knowledge_growth / max(1, session_length / 10)  # Per 10 steps
        
        analysis = {
            'knowledge_growth': knowledge_growth,
            'motivation_change': motivation_change,
            'current_engagement': current_engagement,
            'progress_rate': progress_rate,
            'learning_efficiency': knowledge_growth / max(1, current_metrics.get('total_questions', 1)),
            'adaptation_success_rate': self.strategy_metrics['adaptation_effectiveness']
        }
        
        # Update session analytics
        self.session_analytics.update({
            'current_knowledge': current_knowledge,
            'current_motivation': current_metrics.get('motivation', 0),
            'student_progress_rate': progress_rate
        })
        
        return analysis
    
    def get_strategic_recommendations(self, student_metrics: Dict, progress_analysis: Dict) -> List[str]:
        """
        Generate strategic recommendations for the tutoring session.
        
        Args:
            student_metrics (Dict): Current student metrics
            progress_analysis (Dict): Progress analysis results
            
        Returns:
            List[str]: Strategic recommendations
        """
        recommendations = []
        
        # Progress-based recommendations
        if progress_analysis['knowledge_growth'] < 0.1:
            recommendations.append("Consider adjusting teaching approach - limited knowledge growth")
            recommendations.append("May need more foundational review or different explanation style")
        
        if progress_analysis['progress_rate'] < 0.01:
            recommendations.append("Learning pace is slow - consider simplifying or adding more scaffolding")
        
        # Engagement-based strategies
        engagement = student_metrics.get('engagement', 0.5)
        if engagement < 0.4:
            recommendations.append("Low engagement detected - recommend break or motivational intervention")
        elif engagement > 0.8:
            recommendations.append("High engagement - good opportunity to introduce challenging content")
        
        # Motivation management
        if progress_analysis['motivation_change'] < -0.2:
            recommendations.append("Motivation declining - focus on encouragement and achievable goals")
        
        # Adaptation effectiveness
        if self.strategy_metrics['adaptation_effectiveness'] < 0.5:
            recommendations.append("Recent adaptations not effective - review strategy approach")
        
        return recommendations
    
    def save(self, filepath: str):
        """Save agent state."""
        self.ppo_agent.save(filepath)
        logger.info(f"Strategy agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent state."""
        self.ppo_agent.load(filepath)
        logger.info(f"Strategy agent loaded from {filepath}")
    
    def get_metrics(self) -> Dict:
        """Get comprehensive strategy metrics."""
        metrics = self.strategy_metrics.copy()
        metrics.update({
            'session_analytics': self.session_analytics,
            'adaptation_count': len(self.adaptation_history),
            'ppo_metrics': self.ppo_agent.get_metrics()
        })
        
        # Calculate additional insights
        if self.adaptation_history:
            recent_adaptations = self.adaptation_history[-10:]
            metrics['recent_adaptation_effectiveness'] = np.mean([a['effectiveness'] for a in recent_adaptations])
            metrics['engagement_trend'] = self._calculate_trend([a['engagement_after'] for a in recent_adaptations])
            metrics['motivation_trend'] = self._calculate_trend([a['motivation_after'] for a in recent_adaptations])
        
        return metrics
