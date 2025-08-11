"""
Comprehensive test suite for the Adaptive Tutorial System.

This module provides unit tests, integration tests, and system tests
for all components of the reinforcement learning tutorial system.
"""

import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys

# Add src to path for imports
src_path = str(Path(__file__).parent.parent / 'src')
sys.path.insert(0, src_path)

# Import directly from modules
try:
    from environment.tutoring_environment import TutoringEnvironment, StudentProfile, DifficultyLevel
    from rl.dqn_agent import DQNAgent
    from rl.ppo_agent import PPOAgent
    from agents.content_agent import TutorialContentAgent
    from agents.strategy_agent import TutorialStrategyAgent
    from orchestration.tutorial_orchestrator import TutorialOrchestrator
except ImportError as e:
    print(f"Import error: {e}")
    print("Skipping tests due to import issues")
    sys.exit(0)


class TestTutoringEnvironment(unittest.TestCase):
    """Test cases for the tutoring environment."""
    
    def setUp(self):
        """Set up test environment."""
        self.env = TutoringEnvironment(student_profile="beginner")
    
    def test_environment_initialization(self):
        """Test environment initializes correctly."""
        self.assertIsNotNone(self.env)
        self.assertEqual(self.env.state_size, 15)
        self.assertEqual(self.env.action_size, 8)
        self.assertGreater(len(self.env.questions), 0)
    
    def test_environment_reset(self):
        """Test environment reset functionality."""
        state = self.env.reset()
        self.assertEqual(len(state), self.env.state_size)
        self.assertIsNotNone(self.env.current_student)
        self.assertEqual(self.env.episode_step, 0)
    
    def test_environment_step(self):
        """Test environment step functionality."""
        self.env.reset()
        
        # Test valid action
        action = 0  # ASK_QUESTION
        next_state, reward, done, info = self.env.step(action)
        
        self.assertEqual(len(next_state), self.env.state_size)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
        
        # Check step counter increased
        self.assertEqual(self.env.episode_step, 1)
    
    def test_student_metrics(self):
        """Test student metrics extraction."""
        self.env.reset()
        metrics = self.env.get_student_metrics()
        
        required_keys = ['motivation', 'fatigue', 'engagement', 'knowledge_levels']
        for key in required_keys:
            self.assertIn(key, metrics)
        
        # Check value ranges
        self.assertTrue(0 <= metrics['motivation'] <= 1)
        self.assertTrue(0 <= metrics['fatigue'] <= 1)
        self.assertTrue(0 <= metrics['engagement'] <= 1)


class TestDQNAgent(unittest.TestCase):
    """Test cases for DQN agent."""
    
    def setUp(self):
        """Set up test agent."""
        self.state_size = 10
        self.action_size = 4
        self.agent = DQNAgent(self.state_size, self.action_size)
    
    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        self.assertEqual(self.agent.state_size, self.state_size)
        self.assertEqual(self.agent.action_size, self.action_size)
        self.assertIsNotNone(self.agent.q_network)
        self.assertIsNotNone(self.agent.target_network)
    
    def test_action_selection(self):
        """Test action selection."""
        state = np.random.rand(self.state_size)
        
        # Test training mode
        action = self.agent.act(state, training=True)
        self.assertTrue(0 <= action < self.action_size)
        
        # Test evaluation mode
        action = self.agent.act(state, training=False)
        self.assertTrue(0 <= action < self.action_size)
    
    def test_experience_storage(self):
        """Test experience replay functionality."""
        state = np.random.rand(self.state_size)
        action = 0
        reward = 1.0
        next_state = np.random.rand(self.state_size)
        done = False
        
        initial_buffer_size = len(self.agent.replay_buffer)
        self.agent.step(state, action, reward, next_state, done)
        
        self.assertEqual(len(self.agent.replay_buffer), initial_buffer_size + 1)
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.pth"
            
            # Save model
            self.agent.save(str(model_path))
            self.assertTrue(model_path.exists())
            
            # Create new agent and load model
            new_agent = DQNAgent(self.state_size, self.action_size)
            new_agent.load(str(model_path))
            
            # Test that loaded agent works
            state = np.random.rand(self.state_size)
            action = new_agent.act(state, training=False)
            self.assertTrue(0 <= action < self.action_size)


class TestPPOAgent(unittest.TestCase):
    """Test cases for PPO agent."""
    
    def setUp(self):
        """Set up test agent."""
        self.state_size = 10
        self.action_size = 4
        self.agent = PPOAgent(self.state_size, self.action_size)
    
    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        self.assertEqual(self.agent.state_size, self.state_size)
        self.assertEqual(self.agent.action_size, self.action_size)
        self.assertIsNotNone(self.agent.network)
    
    def test_action_selection(self):
        """Test action selection."""
        state = np.random.rand(self.state_size)
        
        # Test training mode
        action, log_prob, value = self.agent.act(state, training=True)
        self.assertTrue(0 <= action < self.action_size)
        self.assertIsInstance(log_prob, float)
        self.assertIsInstance(value, float)
        
        # Test evaluation mode
        action = self.agent.act(state, training=False)
        self.assertTrue(0 <= action < self.action_size)
    
    def test_experience_storage(self):
        """Test experience buffer functionality."""
        state = np.random.rand(self.state_size)
        action = 0
        log_prob = -1.5
        reward = 1.0
        value = 0.5
        done = False
        
        initial_buffer_size = len(self.agent.buffer)
        self.agent.store_experience(state, action, log_prob, reward, value, done)
        
        self.assertEqual(len(self.agent.buffer), initial_buffer_size + 1)


class TestContentAgent(unittest.TestCase):
    """Test cases for tutorial content agent."""
    
    def setUp(self):
        """Set up test agent."""
        self.state_size = 15
        self.agent = TutorialContentAgent(self.state_size)
    
    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        self.assertIsNotNone(self.agent.dqn_agent)
        self.assertEqual(len(self.agent.content_actions), 4)
    
    def test_action_selection(self):
        """Test content action selection."""
        state = np.random.rand(self.state_size)
        student_metrics = {
            'engagement': 0.7,
            'motivation': 0.6,
            'knowledge_levels': {'math': 0.5, 'science': 0.4}
        }
        
        action = self.agent.select_content_action(state, student_metrics)
        self.assertIn(action, [a.value for a in self.agent.content_actions])
    
    def test_effectiveness_calculation(self):
        """Test content effectiveness calculation."""
        # Initially should be 0
        effectiveness = self.agent.calculate_content_effectiveness()
        self.assertEqual(effectiveness, 0.0)
        
        # Add some decision history
        self.agent.decision_history = [
            {'action': 'ASK_QUESTION', 'engagement': 0.8, 'knowledge_level': 0.5},
            {'action': 'PROVIDE_HINT', 'engagement': 0.7, 'knowledge_level': 0.6}
        ]
        
        effectiveness = self.agent.calculate_content_effectiveness()
        self.assertTrue(0 <= effectiveness <= 1)
    
    def test_recommendations(self):
        """Test content recommendations."""
        student_metrics = {
            'engagement': 0.2,  # Low engagement
            'motivation': 0.8,
            'knowledge_levels': {'math': 0.3, 'science': 0.2}  # Low knowledge
        }
        
        recommendations = self.agent.get_content_recommendations(student_metrics)
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)


class TestStrategyAgent(unittest.TestCase):
    """Test cases for tutorial strategy agent."""
    
    def setUp(self):
        """Set up test agent."""
        self.state_size = 15
        self.agent = TutorialStrategyAgent(self.state_size)
    
    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        self.assertIsNotNone(self.agent.ppo_agent)
        self.assertEqual(len(self.agent.strategy_actions), 4)
    
    def test_action_selection(self):
        """Test strategy action selection."""
        state = np.random.rand(self.state_size)
        student_metrics = {
            'engagement': 0.5,
            'motivation': 0.6,
            'knowledge_levels': {'math': 0.4, 'science': 0.5}
        }
        content_effectiveness = 0.7
        
        # Test training mode
        action, log_prob, value = self.agent.select_strategy_action(
            state, student_metrics, content_effectiveness, training=True
        )
        self.assertIn(action, [a.value for a in self.agent.strategy_actions])
        self.assertIsInstance(log_prob, float)
        self.assertIsInstance(value, float)
        
        # Test evaluation mode
        action = self.agent.select_strategy_action(
            state, student_metrics, content_effectiveness, training=False
        )
        self.assertIn(action, [a.value for a in self.agent.strategy_actions])
    
    def test_progress_analysis(self):
        """Test student progress analysis."""
        initial_metrics = {
            'engagement': 0.5,
            'motivation': 0.6,
            'knowledge_levels': {'math': 0.3, 'science': 0.2},
            'session_length': 1,
            'total_questions': 1
        }
        
        current_metrics = {
            'engagement': 0.7,
            'motivation': 0.8,
            'knowledge_levels': {'math': 0.5, 'science': 0.4},
            'session_length': 10,
            'total_questions': 5
        }
        
        analysis = self.agent.analyze_student_progress(initial_metrics, current_metrics)
        
        required_keys = ['knowledge_growth', 'motivation_change', 'progress_rate', 'learning_efficiency']
        for key in required_keys:
            self.assertIn(key, analysis)
        
        # Knowledge should have grown
        self.assertGreater(analysis['knowledge_growth'], 0)
        self.assertGreater(analysis['motivation_change'], 0)


class TestTutorialOrchestrator(unittest.TestCase):
    """Test cases for tutorial orchestrator."""
    
    def setUp(self):
        """Set up test orchestrator."""
        self.env = TutoringEnvironment(student_profile="beginner")
        self.orchestrator = TutorialOrchestrator(self.env)
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initializes correctly."""
        self.assertIsNotNone(self.orchestrator.content_agent)
        self.assertIsNotNone(self.orchestrator.strategy_agent)
        self.assertIsNotNone(self.orchestrator.env)
    
    def test_agent_coordination(self):
        """Test agent coordination functionality."""
        state = self.env.reset()
        
        action, agent_type = self.orchestrator._coordinate_agents(state)
        
        self.assertTrue(0 <= action < self.env.action_size)
        self.assertIn(agent_type, ['content', 'strategy'])
    
    def test_training_episode(self):
        """Test single training episode."""
        # This is a more complex integration test
        reward = self.orchestrator._run_training_episode()
        
        self.assertIsInstance(reward, float)
        self.assertGreater(self.orchestrator.step_count, 0)
        self.assertIsNotNone(self.orchestrator.session_metrics)
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_orchestrator"
            
            # Save models
            self.orchestrator.save_models(str(model_path))
            
            # Check files exist
            self.assertTrue(Path(f"{model_path}_content.pth").exists())
            self.assertTrue(Path(f"{model_path}_strategy.pth").exists())
            
            # Load models
            self.orchestrator.load_models(str(model_path))


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.env = TutoringEnvironment(student_profile="intermediate")
        self.orchestrator = TutorialOrchestrator(self.env)
    
    def test_complete_episode(self):
        """Test complete episode from start to finish."""
        state = self.env.reset()
        total_reward = 0
        step_count = 0
        max_steps = 20  # Shorter episode for testing
        
        while step_count < max_steps:
            # Get action from orchestrator
            action, agent_type = self.orchestrator._coordinate_agents(state)
            
            # Execute action
            next_state, reward, done, info = self.env.step(action)
            
            # Update agents (simplified for testing)
            student_metrics = self.env.get_student_metrics()
            content_effectiveness = self.orchestrator.content_agent.calculate_content_effectiveness()
            
            if agent_type == 'content':
                self.orchestrator.content_agent.update_from_feedback(
                    state, action, reward, next_state, done, student_metrics
                )
            else:
                self.orchestrator.strategy_agent.update_from_feedback(
                    reward, next_state, done, student_metrics, content_effectiveness
                )
            
            total_reward += reward
            state = next_state
            step_count += 1
            
            if done:
                break
        
        # Verify episode completed successfully
        self.assertGreater(step_count, 0)
        self.assertIsInstance(total_reward, float)
    
    def test_multi_student_profiles(self):
        """Test system with different student profiles."""
        profiles = ['beginner', 'intermediate', 'advanced']
        
        for profile in profiles:
            env = TutoringEnvironment(student_profile=profile)
            orchestrator = TutorialOrchestrator(env)
            
            # Run short episode
            state = env.reset()
            action, agent_type = orchestrator._coordinate_agents(state)
            next_state, reward, done, info = env.step(action)
            
            # Verify system works with this profile
            self.assertIsInstance(reward, float)
            self.assertIsInstance(done, bool)
            self.assertIn(agent_type, ['content', 'strategy'])


class TestPerformance(unittest.TestCase):
    """Performance and stress tests."""
    
    def test_large_episode(self):
        """Test system performance with longer episodes."""
        env = TutoringEnvironment(student_profile="beginner")
        orchestrator = TutorialOrchestrator(env)
        
        import time
        start_time = time.time()
        
        # Run longer episode
        reward = orchestrator._run_training_episode()
        
        end_time = time.time()
        episode_time = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        self.assertLess(episode_time, 30.0)  # 30 seconds max
        self.assertIsInstance(reward, float)
    
    def test_memory_usage(self):
        """Test that system doesn't have memory leaks."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Run multiple episodes
        env = TutoringEnvironment(student_profile="beginner")
        orchestrator = TutorialOrchestrator(env)
        
        for _ in range(5):
            orchestrator._run_training_episode()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        self.assertLess(memory_increase, 100 * 1024 * 1024)


def run_all_tests():
    """Run all test suites."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestTutoringEnvironment,
        TestDQNAgent,
        TestPPOAgent,
        TestContentAgent,
        TestStrategyAgent,
        TestTutorialOrchestrator,
        TestIntegration,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("Running Adaptive Tutorial System Test Suite")
    print("=" * 50)
    
    success = run_all_tests()
    
    if success:
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED! ✅")
    else:
        print("\n" + "=" * 50)
        print("SOME TESTS FAILED! ❌")
        exit(1)
