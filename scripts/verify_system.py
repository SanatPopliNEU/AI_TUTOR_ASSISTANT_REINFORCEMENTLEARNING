"""
System Verification and Testing Guide
=====================================

This script provides comprehensive methods to verify that the Adaptive Tutorial
Agent System is working correctly.
"""

import sys
import os
from pathlib import Path
import importlib.util
import subprocess
import time

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'src'))

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("🔍 CHECKING DEPENDENCIES")
    print("=" * 30)
    
    required_packages = [
        'torch', 'numpy', 'matplotlib', 'seaborn', 'pandas', 
        'scipy', 'gymnasium', 'pyyaml', 'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies are installed!")
        return True

def check_imports():
    """Check if core system modules can be imported."""
    print("\n🔍 CHECKING CORE IMPORTS")
    print("=" * 30)
    
    modules_to_test = [
        ('environment.tutoring_environment', 'TutoringEnvironment'),
        ('rl.dqn_agent', 'DQNAgent'),
        ('rl.ppo_agent', 'PPOAgent'),
        ('agents.content_agent', 'TutorialContentAgent'),
        ('agents.strategy_agent', 'TutorialStrategyAgent'),
        ('orchestration.tutorial_orchestrator', 'TutorialOrchestrator')
    ]
    
    failed_imports = []
    
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {module_name}.{class_name}")
        except Exception as e:
            print(f"❌ {module_name}.{class_name} - {str(e)}")
            failed_imports.append(f"{module_name}.{class_name}")
    
    if failed_imports:
        print(f"\n⚠️ Failed imports: {failed_imports}")
        return False
    else:
        print("\n✅ All core modules import successfully!")
        return True

def test_environment():
    """Test the tutoring environment functionality."""
    print("\n🔍 TESTING TUTORING ENVIRONMENT")
    print("=" * 35)
    
    try:
        from environment.tutoring_environment import TutoringEnvironment
        
        # Test environment creation
        env = TutoringEnvironment(student_profile="beginner")
        print("✅ Environment created successfully")
        
        # Test environment reset
        state = env.reset()
        print(f"✅ Environment reset - State shape: {state.shape}")
        
        # Test environment step
        action = 0  # Ask question
        next_state, reward, done, info = env.step(action)
        print(f"✅ Environment step - Reward: {reward:.2f}, Done: {done}")
        
        # Test student metrics
        metrics = env.get_student_metrics()
        print(f"✅ Student metrics - Engagement: {metrics['engagement']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

def test_rl_agents():
    """Test the reinforcement learning agents."""
    print("\n🔍 TESTING RL AGENTS")
    print("=" * 25)
    
    try:
        from rl.dqn_agent import DQNAgent
        from rl.ppo_agent import PPOAgent
        
        state_size = 15
        action_size = 4
        
        # Test DQN Agent
        dqn_agent = DQNAgent(state_size, action_size)
        test_state = np.random.rand(state_size)
        action = dqn_agent.act(test_state, training=False)
        print(f"✅ DQN Agent - Action selected: {action}")
        
        # Test PPO Agent  
        ppo_agent = PPOAgent(state_size, action_size)
        action = ppo_agent.act(test_state, training=False)
        print(f"✅ PPO Agent - Action selected: {action}")
        
        return True
        
    except Exception as e:
        print(f"❌ RL Agents test failed: {e}")
        return False

def test_tutorial_agents():
    """Test the specialized tutorial agents."""
    print("\n🔍 TESTING TUTORIAL AGENTS")
    print("=" * 30)
    
    try:
        from agents.content_agent import TutorialContentAgent
        from agents.strategy_agent import TutorialStrategyAgent
        
        state_size = 15
        
        # Test Content Agent
        content_agent = TutorialContentAgent(state_size)
        test_state = np.random.rand(state_size)
        student_metrics = {
            'engagement': 0.7,
            'motivation': 0.6,
            'knowledge_levels': {'math': 0.5, 'science': 0.4}
        }
        
        action = content_agent.select_content_action(test_state, student_metrics, training=False)
        print(f"✅ Content Agent - Action selected: {action}")
        
        # Test Strategy Agent
        strategy_agent = TutorialStrategyAgent(state_size)
        action = strategy_agent.select_strategy_action(
            test_state, student_metrics, 0.7, training=False
        )
        print(f"✅ Strategy Agent - Action selected: {action}")
        
        return True
        
    except Exception as e:
        print(f"❌ Tutorial Agents test failed: {e}")
        return False

def test_orchestrator():
    """Test the tutorial orchestrator."""
    print("\n🔍 TESTING TUTORIAL ORCHESTRATOR")
    print("=" * 35)
    
    try:
        from environment.tutoring_environment import TutoringEnvironment
        from orchestration.tutorial_orchestrator import TutorialOrchestrator
        
        # Create environment and orchestrator
        env = TutoringEnvironment(student_profile="beginner")
        orchestrator = TutorialOrchestrator(env)
        
        print("✅ Orchestrator created successfully")
        
        # Test agent coordination
        state = env.reset()
        action, agent_type = orchestrator._coordinate_agents(state)
        print(f"✅ Agent coordination - Agent: {agent_type}, Action: {action}")
        
        return True
        
    except Exception as e:
        print(f"❌ Orchestrator test failed: {e}")
        return False

def run_quick_training():
    """Run a quick training session to test learning."""
    print("\n🔍 TESTING QUICK TRAINING")
    print("=" * 30)
    
    try:
        from environment.tutoring_environment import TutoringEnvironment
        from orchestration.tutorial_orchestrator import TutorialOrchestrator
        
        env = TutoringEnvironment(student_profile="intermediate")
        orchestrator = TutorialOrchestrator(env)
        
        print("Running 5 training episodes...")
        
        # Run quick training
        for episode in range(5):
            reward = orchestrator._run_training_episode()
            print(f"Episode {episode + 1}: Reward = {reward:.2f}")
        
        print("✅ Quick training completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Quick training failed: {e}")
        return False

def test_visualization():
    """Test the visualization tools."""
    print("\n🔍 TESTING VISUALIZATION")
    print("=" * 30)
    
    try:
        from tools.visualizer import TutorialVisualizer
        import tempfile
        
        # Create visualizer with temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = TutorialVisualizer(temp_dir)
            
            # Test with sample data
            training_history = [
                {'episode': i, 'reward': 5 + i * 0.1, 'content_metrics': {'content_effectiveness': 0.5 + i * 0.01}}
                for i in range(20)
            ]
            
            visualizer.plot_training_progress(training_history, "Test Training")
            print("✅ Visualization test completed!")
            
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False

def run_integration_test():
    """Run a complete integration test."""
    print("\n🔍 RUNNING INTEGRATION TEST")
    print("=" * 35)
    
    try:
        from environment.tutoring_environment import TutoringEnvironment
        from orchestration.tutorial_orchestrator import TutorialOrchestrator
        
        print("Creating system components...")
        env = TutoringEnvironment(student_profile="beginner")
        orchestrator = TutorialOrchestrator(env)
        
        print("Running complete episode...")
        state = env.reset()
        initial_metrics = env.get_student_metrics()
        
        total_reward = 0
        step_count = 0
        max_steps = 10
        
        while step_count < max_steps:
            # Get action from orchestrator
            action, agent_type = orchestrator._coordinate_agents(state)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Update metrics
            current_metrics = env.get_student_metrics()
            content_effectiveness = orchestrator.content_agent.calculate_content_effectiveness()
            
            # Update agents
            if agent_type == 'content':
                orchestrator.content_agent.update_from_feedback(
                    state, action, reward, next_state, done, current_metrics
                )
            else:
                orchestrator.strategy_agent.update_from_feedback(
                    reward, next_state, done, current_metrics, content_effectiveness
                )
            
            total_reward += reward
            state = next_state
            step_count += 1
            
            if done:
                break
        
        final_metrics = env.get_student_metrics()
        
        print(f"✅ Integration test completed!")
        print(f"   Steps: {step_count}")
        print(f"   Total Reward: {total_reward:.2f}")
        print(f"   Final Engagement: {final_metrics['engagement']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def run_full_system_check():
    """Run complete system verification."""
    print("🚀 ADAPTIVE TUTORIAL SYSTEM VERIFICATION")
    print("=" * 50)
    print("This will check if your system is working correctly.\n")
    
    # Import numpy for tests
    global np
    import numpy as np
    
    tests = [
        ("Dependencies", check_dependencies),
        ("Core Imports", check_imports),
        ("Environment", test_environment),
        ("RL Agents", test_rl_agents),
        ("Tutorial Agents", test_tutorial_agents),
        ("Orchestrator", test_orchestrator),
        ("Quick Training", run_quick_training),
        ("Visualization", test_visualization),
        ("Integration Test", run_integration_test)
    ]
    
    passed_tests = 0
    failed_tests = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed_tests += 1
            else:
                failed_tests.append(test_name)
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            failed_tests.append(test_name)
    
    # Final summary
    print("\n" + "=" * 50)
    print("🏁 VERIFICATION SUMMARY")
    print("=" * 50)
    
    print(f"✅ Passed: {passed_tests}/{len(tests)} tests")
    
    if failed_tests:
        print(f"❌ Failed: {failed_tests}")
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Make sure you've run: python setup.py")
        print("2. Check that all dependencies are installed")
        print("3. Verify Python version is 3.8+")
        print("4. Check file paths and project structure")
        return False
    else:
        print("🎉 ALL TESTS PASSED!")
        print("\n✨ Your Adaptive Tutorial System is working perfectly!")
        print("\n🚀 Next steps:")
        print("   • Run demo: python src/main.py --mode demo")
        print("   • Start training: python src/main.py --mode train")
        print("   • Run experiments: python src/main.py --mode experiment")
        return True

if __name__ == "__main__":
    success = run_full_system_check()
    sys.exit(0 if success else 1)
