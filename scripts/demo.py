"""
Demonstration script for the Adaptive Tutorial System.

This script provides various demonstration modes to showcase
the capabilities of the reinforcement learning tutorial system.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from environment.tutoring_environment import TutoringEnvironment
from orchestration.tutorial_orchestrator import TutorialOrchestrator
from tools.visualizer import TutorialVisualizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demonstrate_basic_functionality():
    """Demonstrate basic system functionality."""
    print("🎓 ADAPTIVE TUTORIAL SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    # Create environment and orchestrator
    env = TutoringEnvironment(student_profile="beginner")
    orchestrator = TutorialOrchestrator(env)
    
    print("✅ System initialized successfully")
    print(f"   - Student Profile: {env.student_profile_type}")
    print(f"   - State Space Size: {env.state_size}")
    print(f"   - Action Space Size: {env.action_size}")
    print(f"   - Question Bank Size: {len(env.questions)}")
    
    # Reset environment
    state = env.reset()
    initial_metrics = env.get_student_metrics()
    
    print("\n📊 INITIAL STUDENT STATE")
    print("-" * 30)
    print(f"   Motivation: {initial_metrics['motivation']:.2f}")
    print(f"   Engagement: {initial_metrics['engagement']:.2f}")
    print(f"   Average Knowledge: {np.mean(list(initial_metrics['knowledge_levels'].values())):.2f}")
    
    # Run demonstration episode
    print("\n🎯 RUNNING DEMONSTRATION EPISODE")
    print("-" * 40)
    
    total_reward = 0
    step_count = 0
    max_steps = 15
    
    while step_count < max_steps:
        print(f"\nStep {step_count + 1}:")
        
        # Get action from orchestrator
        action, agent_type = orchestrator._coordinate_agents(state)
        action_name = env.env.action_type(action).name if hasattr(env, 'env') else f"Action_{action}"
        
        print(f"  🤖 Agent: {agent_type.title()}")
        print(f"  🎬 Action: {action_name}")
        
        # Execute action
        next_state, reward, done, info = env.step(action)
        
        print(f"  🏆 Reward: {reward:.2f}")
        
        # Show student response
        if 'correct' in info:
            result = "✅ Correct" if info['correct'] else "❌ Incorrect"
            confidence = info.get('confidence', 0)
            print(f"  📝 Student Response: {result} (Confidence: {confidence:.2f})")
        
        # Update metrics
        current_metrics = env.get_student_metrics()
        print(f"  📈 Engagement: {current_metrics['engagement']:.2f}")
        print(f"  💪 Motivation: {current_metrics['motivation']:.2f}")
        
        total_reward += reward
        state = next_state
        step_count += 1
        
        if done:
            print("  🏁 Episode completed")
            break
        
        # Pause for dramatic effect
        time.sleep(1)
    
    # Final results
    final_metrics = env.get_student_metrics()
    knowledge_growth = (np.mean(list(final_metrics['knowledge_levels'].values())) - 
                       np.mean(list(initial_metrics['knowledge_levels'].values())))
    
    print("\n📊 FINAL RESULTS")
    print("-" * 25)
    print(f"   Total Reward: {total_reward:.2f}")
    print(f"   Steps Completed: {step_count}")
    print(f"   Knowledge Growth: {knowledge_growth:.3f}")
    print(f"   Final Engagement: {final_metrics['engagement']:.2f}")
    print(f"   Final Motivation: {final_metrics['motivation']:.2f}")
    
    return {
        'total_reward': total_reward,
        'knowledge_growth': knowledge_growth,
        'final_engagement': final_metrics['engagement'],
        'step_count': step_count
    }


def demonstrate_learning_progression():
    """Demonstrate how the system learns and improves over time."""
    print("\n🧠 LEARNING PROGRESSION DEMONSTRATION")
    print("=" * 50)
    
    env = TutoringEnvironment(student_profile="intermediate")
    orchestrator = TutorialOrchestrator(env)
    
    print("Training the system to learn optimal teaching strategies...")
    
    # Quick training session
    training_episodes = 100
    episode_rewards = []
    
    for episode in range(training_episodes):
        reward = orchestrator._run_training_episode()
        episode_rewards.append(reward)
        
        if episode % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards)
            print(f"Episode {episode}: Average reward = {avg_reward:.2f}")
    
    print("✅ Training completed!")
    
    # Show learning curve
    if len(episode_rewards) > 10:
        initial_performance = np.mean(episode_rewards[:10])
        final_performance = np.mean(episode_rewards[-10:])
        improvement = final_performance - initial_performance
        
        print(f"\n📈 LEARNING ANALYSIS")
        print("-" * 25)
        print(f"   Initial Performance: {initial_performance:.2f}")
        print(f"   Final Performance: {final_performance:.2f}")
        print(f"   Improvement: {improvement:.2f}")
        
        if improvement > 0:
            print("   🎉 System learned to improve its teaching strategy!")
        else:
            print("   🤔 System maintained consistent performance")
    
    return episode_rewards


def demonstrate_student_profiles():
    """Demonstrate system adaptation to different student profiles."""
    print("\n👥 STUDENT PROFILE ADAPTATION DEMONSTRATION")
    print("=" * 55)
    
    profiles = ['beginner', 'intermediate', 'advanced']
    profile_results = {}
    
    for profile in profiles:
        print(f"\n🎯 Testing with {profile.title()} Student")
        print("-" * 35)
        
        env = TutoringEnvironment(student_profile=profile)
        orchestrator = TutorialOrchestrator(env)
        
        # Run evaluation
        results = orchestrator.evaluate(num_episodes=10)
        profile_results[profile] = results
        
        print(f"   Average Reward: {results['average_reward']:.2f}")
        print(f"   Average Engagement: {results['average_engagement']:.2f}")
        print(f"   Knowledge Growth: {results['average_knowledge_growth']:.3f}")
        print(f"   Success Rate: {results['success_rate']:.2f}")
    
    # Compare profiles
    print(f"\n📊 PROFILE COMPARISON")
    print("-" * 25)
    
    best_engagement = max(profiles, key=lambda p: profile_results[p]['average_engagement'])
    best_knowledge = max(profiles, key=lambda p: profile_results[p]['average_knowledge_growth'])
    
    print(f"   Best Engagement: {best_engagement.title()} students")
    print(f"   Best Knowledge Growth: {best_knowledge.title()} students")
    
    return profile_results


def demonstrate_coordination_strategies():
    """Demonstrate different agent coordination strategies."""
    print("\n🤝 COORDINATION STRATEGY DEMONSTRATION")
    print("=" * 50)
    
    strategies = ['hierarchical', 'competitive', 'collaborative']
    strategy_results = {}
    
    for strategy in strategies:
        print(f"\n🎯 Testing {strategy.title()} Coordination")
        print("-" * 35)
        
        env = TutoringEnvironment(student_profile="intermediate")
        config = {'coordination_mode': strategy}
        orchestrator = TutorialOrchestrator(env, config)
        
        # Quick evaluation
        results = orchestrator.evaluate(num_episodes=5)
        strategy_results[strategy] = results
        
        print(f"   Average Reward: {results['average_reward']:.2f}")
        print(f"   Success Rate: {results['success_rate']:.2f}")
        print(f"   Average Engagement: {results['average_engagement']:.2f}")
    
    # Find best strategy
    best_strategy = max(strategies, key=lambda s: strategy_results[s]['average_reward'])
    print(f"\n🏆 BEST COORDINATION STRATEGY: {best_strategy.title()}")
    
    return strategy_results


def create_demonstration_visualizations():
    """Create visualizations for demonstration results."""
    print("\n📊 CREATING DEMONSTRATION VISUALIZATIONS")
    print("=" * 45)
    
    visualizer = TutorialVisualizer("demo_visualizations")
    
    # Generate sample data for visualization
    print("   Generating sample training data...")
    env = TutoringEnvironment(student_profile="intermediate")
    orchestrator = TutorialOrchestrator(env)
    
    # Quick training to get data
    orchestrator.train(episodes=50, save_frequency=50)
    
    # Create visualizations
    print("   Creating training progress visualization...")
    visualizer.plot_training_progress(
        orchestrator.training_history, 
        "Demo Training Progress"
    )
    
    # Create student analytics visualization
    print("   Creating student analytics visualization...")
    # Generate sample session data
    session_data = []
    for i in range(20):
        session_data.append({
            'engagement': 0.5 + 0.3 * np.sin(i * 0.3) + np.random.normal(0, 0.1),
            'motivation': 0.6 + 0.2 * np.cos(i * 0.2) + np.random.normal(0, 0.1),
            'knowledge_levels': {
                'math': 0.3 + i * 0.02,
                'science': 0.2 + i * 0.03
            }
        })
    
    visualizer.plot_student_analytics(session_data, "Demo Student Learning Analytics")
    
    print("   ✅ Visualizations created in 'demo_visualizations' folder")


def run_comprehensive_demonstration():
    """Run comprehensive demonstration of all system capabilities."""
    print("🚀 COMPREHENSIVE ADAPTIVE TUTORIAL SYSTEM DEMONSTRATION")
    print("=" * 65)
    print("This demonstration showcases the key capabilities of our")
    print("reinforcement learning-enhanced tutorial system.\n")
    
    try:
        # 1. Basic functionality
        basic_results = demonstrate_basic_functionality()
        
        # 2. Learning progression
        learning_results = demonstrate_learning_progression()
        
        # 3. Student profile adaptation
        profile_results = demonstrate_student_profiles()
        
        # 4. Coordination strategies
        coordination_results = demonstrate_coordination_strategies()
        
        # 5. Create visualizations
        create_demonstration_visualizations()
        
        # Summary
        print("\n🎉 DEMONSTRATION SUMMARY")
        print("=" * 30)
        print("✅ Basic functionality: Working")
        print("✅ Learning progression: Demonstrated")
        print("✅ Student adaptation: Multiple profiles tested")
        print("✅ Coordination strategies: Compared")
        print("✅ Visualizations: Generated")
        
        print(f"\n📊 KEY METRICS")
        print("-" * 15)
        print(f"   Basic Demo Reward: {basic_results['total_reward']:.2f}")
        print(f"   Knowledge Growth: {basic_results['knowledge_growth']:.3f}")
        print(f"   Final Engagement: {basic_results['final_engagement']:.2f}")
        
        if learning_results:
            improvement = np.mean(learning_results[-10:]) - np.mean(learning_results[:10])
            print(f"   Learning Improvement: {improvement:.2f}")
        
        print("\n🎓 The Adaptive Tutorial System successfully demonstrates:")
        print("   • Multi-agent reinforcement learning coordination")
        print("   • Personalized teaching strategy adaptation") 
        print("   • Real-time student engagement optimization")
        print("   • Comprehensive learning analytics")
        
        return True
        
    except Exception as e:
        print(f"\n❌ DEMONSTRATION FAILED: {e}")
        logger.error(f"Demonstration error: {e}")
        return False


if __name__ == "__main__":
    print("Starting Adaptive Tutorial System Demonstration...")
    
    success = run_comprehensive_demonstration()
    
    if success:
        print("\n✨ Demonstration completed successfully!")
    else:
        print("\n💥 Demonstration encountered errors.")
        sys.exit(1)
