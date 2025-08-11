"""
Tutorial Orchestrator - Coordinates multiple RL agents for adaptive tutoring.

This module implements the main orchestration system that coordinates
content delivery and strategic adaptation agents to provide optimal
tutoring experiences.
"""

import numpy as np
import logging
import time
from typing import Dict, List, Any, Tuple
from pathlib import Path

from agents.content_agent import TutorialContentAgent
from agents.strategy_agent import TutorialStrategyAgent
from environment.tutoring_environment import TutoringEnvironment, ActionType

logger = logging.getLogger(__name__)


class TutorialOrchestrator:
    """Main orchestrator for the adaptive tutorial system."""
    
    def __init__(self, environment: TutoringEnvironment, config: Dict = None):
        """
        Initialize tutorial orchestrator.
        
        Args:
            environment (TutoringEnvironment): Tutoring environment
            config (Dict): Configuration parameters
        """
        self.env = environment
        self.config = config or {}
        
        # Initialize specialized agents
        state_size = environment.state_size
        
        self.content_agent = TutorialContentAgent(
            state_size=state_size,
            config=config
        )
        
        self.strategy_agent = TutorialStrategyAgent(
            state_size=state_size,
            config=config
        )
        
        # Orchestration parameters
        self.strategy_action_frequency = self.config.get('strategy_frequency', 5)  # Every 5 steps
        self.coordination_mode = self.config.get('coordination_mode', 'hierarchical')
        
        # Training and evaluation metrics
        self.training_history = []
        self.evaluation_results = []
        self.session_metrics = {}
        
        # Session state
        self.current_episode = 0
        self.step_count = 0
        self.last_strategy_action = 0
        self.initial_student_state = None
        
        logger.info("Tutorial Orchestrator initialized with multi-agent coordination")
    
    def train(self, episodes: int = 1000, save_frequency: int = 100):
        """
        Train the coordinated agent system.
        
        Args:
            episodes (int): Number of training episodes
            save_frequency (int): How often to save models
        """
        logger.info(f"Starting training for {episodes} episodes")
        
        for episode in range(episodes):
            self.current_episode = episode
            episode_reward = self._run_training_episode()
            
            # Store training metrics
            self.training_history.append({
                'episode': episode,
                'reward': episode_reward,
                'content_metrics': self.content_agent.get_metrics(),
                'strategy_metrics': self.strategy_agent.get_metrics()
            })
            
            # Log progress
            if episode % 50 == 0:
                avg_reward = np.mean([h['reward'] for h in self.training_history[-50:]])
                logger.info(f"Episode {episode}: Average reward = {avg_reward:.2f}")
            
            # Save models periodically
            if episode % save_frequency == 0 and episode > 0:
                self.save_models(f"models/checkpoint_episode_{episode}")
        
        logger.info("Training completed")
    
    def _run_training_episode(self) -> float:
        """
        Run a single training episode.
        
        Returns:
            float: Total episode reward
        """
        state = self.env.reset()
        self.initial_student_state = self.env.get_student_metrics()
        
        total_reward = 0.0
        done = False
        self.step_count = 0
        self.last_strategy_action = 0
        
        while not done:
            # Decide which agent should act
            action, agent_type = self._coordinate_agents(state)
            
            # Execute action
            next_state, reward, done, info = self.env.step(action)
            
            # Get current student metrics
            student_metrics = self.env.get_student_metrics()
            content_effectiveness = self.content_agent.calculate_content_effectiveness()
            
            # Update appropriate agent
            if agent_type == 'content':
                self.content_agent.update_from_feedback(
                    state, action, reward, next_state, done, student_metrics
                )
            elif agent_type == 'strategy':
                self.strategy_agent.update_from_feedback(
                    reward, next_state, done, student_metrics, content_effectiveness
                )
            
            # Update state and metrics
            state = next_state
            total_reward += reward
            self.step_count += 1
        
        # Episode completed - analyze performance
        final_metrics = self.env.get_student_metrics()
        progress_analysis = self.strategy_agent.analyze_student_progress(
            self.initial_student_state, final_metrics
        )
        
        # Store session metrics
        self.session_metrics = {
            'total_reward': total_reward,
            'episode_length': self.step_count,
            'progress_analysis': progress_analysis,
            'final_engagement': final_metrics.get('engagement', 0),
            'knowledge_growth': progress_analysis.get('knowledge_growth', 0)
        }
        
        return total_reward
    
    def update_from_experience(self, state: np.ndarray, action: int, reward: float, 
                             student_metrics: Dict) -> None:
        """
        Update agents based on human tutoring experience.
        
        Args:
            state (np.ndarray): Current environment state
            action (int): Action that was taken
            reward (float): Reward received
            student_metrics (Dict): Student performance metrics
        """
        try:
            # Get next state after the action
            next_state = self.env._get_state()
            done = False  # Human tutoring sessions don't really "end"
            
            # Determine which agent should be updated based on action type
            # Content actions: ASK_QUESTION(0), PROVIDE_HINT(1), EXPLAIN_CONCEPT(2), REVIEW_PREVIOUS(3)
            # Strategy actions: INCREASE_DIFFICULTY(4), DECREASE_DIFFICULTY(5), PROVIDE_ENCOURAGEMENT(6), SUGGEST_BREAK(7)
            
            if action in [0, 1, 2, 3]:  # Content agent actions
                # Update content agent (DQN)
                self.content_agent.update_from_feedback(
                    state, action, reward, next_state, done, student_metrics
                )
                logger.debug(f"Updated content agent with action {action}, reward: {reward:.3f}")
                
            elif action in [4, 5, 6, 7]:  # Strategy agent actions
                # Update strategy agent (PPO)  
                content_effectiveness = self.content_agent.calculate_content_effectiveness()
                self.strategy_agent.update_from_feedback(
                    reward, next_state, done, student_metrics, content_effectiveness
                )
                logger.debug(f"Updated strategy agent with action {action}, reward: {reward:.3f}")
                
            else:
                logger.warning(f"Unknown action {action}, defaulting to content agent update")
                # Default to content agent for unknown actions
                # Map unknown action to ASK_QUESTION (0) for safety
                self.content_agent.update_from_feedback(
                    state, 0, reward, next_state, done, student_metrics
                )
            
            # Update step count for coordination
            self.step_count += 1
            
        except Exception as e:
            logger.error(f"Error in update_from_experience: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise e
    
    def _coordinate_agents(self, state: np.ndarray) -> Tuple[int, str]:
        """
        Coordinate agents to select optimal action.
        
        Args:
            state (np.ndarray): Current state
            
        Returns:
            Tuple[int, str]: (action, agent_type)
        """
        student_metrics = self.env.get_student_metrics()
        content_effectiveness = self.content_agent.calculate_content_effectiveness()
        
        if self.coordination_mode == 'hierarchical':
            action, agent_type = self._hierarchical_coordination(state, student_metrics, content_effectiveness)
        elif self.coordination_mode == 'competitive':
            action, agent_type = self._competitive_coordination(state, student_metrics, content_effectiveness)
        else:  # collaborative
            action, agent_type = self._collaborative_coordination(state, student_metrics, content_effectiveness)
        
        # Store which agent acted for RL updates
        self._last_acting_agent = agent_type
        
        # Ensure action is within valid bounds [0, 3]
        action = max(0, min(int(action), 3))
        
        return action, agent_type
    
    def _hierarchical_coordination(self, state: np.ndarray, student_metrics: Dict, 
                                 content_effectiveness: float) -> Tuple[int, str]:
        """Hierarchical coordination: Strategy agent decides when to intervene."""
        # Strategy agent decides if strategic intervention is needed
        should_intervene = (
            self.step_count - self.last_strategy_action >= self.strategy_action_frequency or
            student_metrics.get('engagement', 1.0) < 0.4 or
            student_metrics.get('motivation', 1.0) < 0.3 or
            content_effectiveness < 0.3
        )
        
        if should_intervene:
            self.last_strategy_action = self.step_count
            action, log_prob, value = self.strategy_agent.select_strategy_action(
                state, student_metrics, content_effectiveness, training=True
            )
            return action, 'strategy'
        else:
            action = self.content_agent.select_content_action(
                state, student_metrics, training=True
            )
            return action, 'content'
    
    def _competitive_coordination(self, state: np.ndarray, student_metrics: Dict, 
                                content_effectiveness: float) -> Tuple[int, str]:
        """Competitive coordination: Both agents propose actions, best one wins."""
        # Get content action proposal with Q-value
        content_action = self.content_agent.select_content_action(
            state, student_metrics, training=True
        )
        
        # Get content Q-value for the selected action
        content_q_value = self.content_agent.get_q_value(state, content_action)
        
        # Get strategy action proposal with value
        strategy_action, _, strategy_value = self.strategy_agent.select_strategy_action(
            state, student_metrics, content_effectiveness, training=True
        )
        
        # Normalize values for fair comparison (0-1 range)
        content_score = min(1.0, max(0.0, (content_q_value + 10) / 20))  # Normalize Q-value
        strategy_score = min(1.0, max(0.0, strategy_value))
        
        # Add context-based bonuses
        engagement = student_metrics.get('engagement', 0.5)
        motivation = student_metrics.get('motivation', 0.5)
        
        # Make competition more balanced by adding base scores
        content_base_score = 0.3  # Base competence
        strategy_base_score = 0.4  # Slightly favor strategy for balance
        
        # Scale the agent-specific scores
        content_score = content_base_score + (content_score * 0.4)
        strategy_score = strategy_base_score + (strategy_score * 0.4)
        
        # Strategy bonus when student needs motivation/engagement
        if engagement < 0.5 or motivation < 0.5:
            strategy_score += 0.3
        
        # Content bonus when student is engaged and motivated
        if engagement > 0.7 and motivation > 0.7:
            content_score += 0.2
        
        # Add some randomness for variety (±0.1)
        import random
        content_score += random.uniform(-0.1, 0.1)
        strategy_score += random.uniform(-0.1, 0.1)
        
        # Debug logging for competitive coordination
        logger.debug(f"Competitive coordination - Content Q: {content_q_value:.3f}, Strategy Value: {strategy_value:.3f}")
        logger.debug(f"Final scores - Content: {content_score:.3f}, Strategy: {strategy_score:.3f}")
        logger.debug(f"Student metrics - Engagement: {engagement:.3f}, Motivation: {motivation:.3f}")
        
        # Select the agent with higher score
        if strategy_score > content_score:
            logger.debug(f"Strategy agent wins with score {strategy_score:.3f} vs {content_score:.3f}")
            return strategy_action, 'strategy'
        else:
            logger.debug(f"Content agent wins with score {content_score:.3f} vs {strategy_score:.3f}")
            return content_action, 'content'
    
    def _collaborative_coordination(self, state: np.ndarray, student_metrics: Dict, 
                                  content_effectiveness: float) -> Tuple[int, str]:
        """Collaborative coordination: Weighted combination of agent recommendations."""
        engagement = student_metrics.get('engagement', 0.5)
        motivation = student_metrics.get('motivation', 0.5)
        
        # Calculate dynamic weights based on student needs
        avg_wellbeing = (engagement + motivation) / 2
        
        # Strategy weight increases when student wellbeing is low
        strategy_weight = 1.0 - avg_wellbeing
        content_weight = avg_wellbeing
        
        # Add step-based alternation to ensure both agents get used
        step_bias = 0.2 if self.step_count % 4 < 2 else -0.2
        strategy_weight += step_bias
        content_weight -= step_bias
        
        # Ensure weights stay in valid range
        strategy_weight = max(0.1, min(0.9, strategy_weight))  # Minimum 10% chance for strategy
        content_weight = max(0.1, min(0.9, content_weight))    # Minimum 10% chance for content
        
        # Normalize weights to sum to 1
        total_weight = strategy_weight + content_weight
        strategy_weight /= total_weight
        content_weight /= total_weight
        
        # Additional factors
        if content_effectiveness < 0.4:
            strategy_weight += 0.3  # Strategy agent should help
        
        if engagement > 0.8 and motivation > 0.8:
            content_weight += 0.2  # Content agent can continue
        
        # Select agent based on weights
        import random
        random_value = random.random()
        
        # Debug logging for collaborative coordination
        logger.debug(f"Collaborative coordination - Strategy weight: {strategy_weight:.3f}, Content weight: {content_weight:.3f}")
        logger.debug(f"Random value: {random_value:.3f}, Student wellbeing: {avg_wellbeing:.3f}")
        
        if random_value < strategy_weight:
            logger.debug(f"Strategy agent selected in collaborative mode")
            action, _, _ = self.strategy_agent.select_strategy_action(
                state, student_metrics, content_effectiveness, training=True
            )
            return action, 'strategy'
        else:
            logger.debug(f"Content agent selected in collaborative mode")
            action = self.content_agent.select_content_action(
                state, student_metrics, training=True
            )
            return action, 'content'
    
    def evaluate(self, num_episodes: int = 100) -> Dict:
        """
        Evaluate trained agents.
        
        Args:
            num_episodes (int): Number of evaluation episodes
            
        Returns:
            Dict: Evaluation results
        """
        logger.info(f"Evaluating agents for {num_episodes} episodes")
        
        evaluation_rewards = []
        engagement_scores = []
        knowledge_growth_scores = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            initial_state = self.env.get_student_metrics()
            
            total_reward = 0.0
            done = False
            
            while not done:
                # Select action (no training updates)
                action, _ = self._coordinate_agents(state)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
            
            # Collect metrics
            final_state = self.env.get_student_metrics()
            
            evaluation_rewards.append(total_reward)
            engagement_scores.append(final_state.get('engagement', 0))
            
            # Calculate knowledge growth
            initial_knowledge = np.mean(list(initial_state.get('knowledge_levels', {}).values()))
            final_knowledge = np.mean(list(final_state.get('knowledge_levels', {}).values()))
            knowledge_growth_scores.append(final_knowledge - initial_knowledge)
        
        results = {
            'average_reward': np.mean(evaluation_rewards),
            'reward_std': np.std(evaluation_rewards),
            'average_engagement': np.mean(engagement_scores),
            'average_knowledge_growth': np.mean(knowledge_growth_scores),
            'success_rate': sum(1 for r in evaluation_rewards if r > 0) / len(evaluation_rewards)
        }
        
        self.evaluation_results.append(results)
        logger.info(f"Evaluation results: {results}")
        
        return results
    
    def demonstrate(self, num_interactions: int = 20, interactive: bool = True):
        """
        Run interactive demonstration of the tutorial system.
        
        Args:
            num_interactions (int): Number of interactions to demonstrate
            interactive (bool): Whether to pause for user input
        """
        logger.info("Starting tutorial system demonstration")
        
        state = self.env.reset()
        initial_metrics = self.env.get_student_metrics()
        
        print("=== Adaptive Tutorial System Demonstration ===")
        print(f"Student Profile: {self.env.student_profile_type}")
        print(f"Initial Knowledge Level: {np.mean(list(initial_metrics.get('knowledge_levels', {}).values())):.2f}")
        print(f"Initial Motivation: {initial_metrics.get('motivation', 0):.2f}")
        print(f"Initial Engagement: {initial_metrics.get('engagement', 0):.2f}")
        print("-" * 50)
        
        for step in range(num_interactions):
            if interactive:
                input("Press Enter to continue...")
            
            # Get action from coordinated agents
            action, agent_type = self._coordinate_agents(state)
            action_name = ActionType(action).name
            
            print(f"\nStep {step + 1}:")
            print(f"Agent Type: {agent_type.title()}")
            print(f"Action: {action_name}")
            
            # Execute action
            next_state, reward, done, info = self.env.step(action)
            
            # Show results
            current_metrics = self.env.get_student_metrics()
            print(f"Reward: {reward:.2f}")
            print(f"Engagement: {current_metrics.get('engagement', 0):.2f}")
            print(f"Motivation: {current_metrics.get('motivation', 0):.2f}")
            
            if 'correct' in info:
                print(f"Answer Correct: {info['correct']}")
            
            # Show recommendations
            if step % 5 == 0:
                content_recs = self.content_agent.get_content_recommendations(current_metrics)
                strategy_recs = self.strategy_agent.get_strategic_recommendations(
                    current_metrics, 
                    self.strategy_agent.analyze_student_progress(initial_metrics, current_metrics)
                )
                
                if content_recs or strategy_recs:
                    print("\nSystem Recommendations:")
                    for rec in content_recs:
                        print(f"  Content: {rec}")
                    for rec in strategy_recs:
                        print(f"  Strategy: {rec}")
            
            state = next_state
            
            if done:
                break
        
        # Final analysis
        final_metrics = self.env.get_student_metrics()
        progress = self.strategy_agent.analyze_student_progress(initial_metrics, final_metrics)
        
        print("\n" + "=" * 50)
        print("DEMONSTRATION SUMMARY")
        print("=" * 50)
        print(f"Knowledge Growth: {progress['knowledge_growth']:.3f}")
        print(f"Motivation Change: {progress['motivation_change']:.3f}")
        print(f"Final Engagement: {progress['current_engagement']:.3f}")
        print(f"Learning Efficiency: {progress['learning_efficiency']:.3f}")
        print(f"Content Agent Effectiveness: {self.content_agent.calculate_content_effectiveness():.3f}")
        print(f"Strategy Adaptation Success Rate: {progress['adaptation_success_rate']:.3f}")
    
    def save_models(self, base_path: str):
        """
        Save all agent models.
        
        Args:
            base_path (str): Base path for saving models
        """
        Path(base_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.content_agent.save(f"{base_path}_content.pth")
        self.strategy_agent.save(f"{base_path}_strategy.pth")
        
        logger.info(f"Models saved to {base_path}")
    
    def load_models(self, base_path: str):
        """
        Load all agent models.
        
        Args:
            base_path (str): Base path for loading models
        """
        self.content_agent.load(f"{base_path}_content.pth")
        self.strategy_agent.load(f"{base_path}_strategy.pth")
        
        logger.info(f"Models loaded from {base_path}")
    
    def get_comprehensive_metrics(self) -> Dict:
        """Get comprehensive system metrics."""
        return {
            'orchestrator_metrics': {
                'current_episode': self.current_episode,
                'step_count': self.step_count,
                'coordination_mode': self.coordination_mode,
                'session_metrics': self.session_metrics
            },
            'content_agent_metrics': self.content_agent.get_metrics(),
            'strategy_agent_metrics': self.strategy_agent.get_metrics(),
            'training_history': self.training_history[-10:],  # Last 10 episodes
            'evaluation_results': self.evaluation_results
        }
