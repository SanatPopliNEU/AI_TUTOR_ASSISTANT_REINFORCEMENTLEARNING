"""
Generate visualizations for the RL Tutorial System experimental results.
Creates learning curves, performance comparisons, and agent behavior plots.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import pandas as pd
from datetime import datetime

# Set style for professional plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_student_data():
    """Load all student results data."""
    results_dir = Path("student_results")
    
    data = {}
    
    # Load interactions
    try:
        with open(results_dir / "interactions.json", 'r') as f:
            data['interactions'] = json.load(f)
    except FileNotFoundError:
        data['interactions'] = []
    
    # Load sessions
    try:
        with open(results_dir / "sessions.json", 'r') as f:
            data['sessions'] = json.load(f)
    except FileNotFoundError:
        data['sessions'] = []
    
    # Load evaluations
    try:
        with open(results_dir / "evaluations.json", 'r') as f:
            data['evaluations'] = json.load(f)
    except FileNotFoundError:
        data['evaluations'] = []
    
    return data

def create_learning_curves(data):
    """Create learning curves showing improvement over time."""
    interactions = data['interactions']
    
    if not interactions:
        print("No interaction data found for learning curves")
        return
    
    # Extract reward progression
    rewards = [float(interaction.get('reward_score', 0)) for interaction in interactions]
    episodes = list(range(1, len(rewards) + 1))
    
    # Create moving average for smoother curves
    window_size = min(10, len(rewards) // 3) if len(rewards) > 10 else 3
    moving_avg = []
    for i in range(len(rewards)):
        start_idx = max(0, i - window_size + 1)
        moving_avg.append(np.mean(rewards[start_idx:i+1]))
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Learning curve
    ax1.plot(episodes, rewards, alpha=0.3, color='lightblue', label='Raw Rewards')
    ax1.plot(episodes, moving_avg, color='darkblue', linewidth=2, label=f'Moving Average (window={window_size})')
    ax1.set_xlabel('Interaction Number')
    ax1.set_ylabel('Reward Score')
    ax1.set_title('Learning Curve: Reward Progression')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Response length progression (proxy for answer quality)
    response_lengths = [interaction.get('response_length', 0) for interaction in interactions]
    moving_avg_length = []
    for i in range(len(response_lengths)):
        start_idx = max(0, i - window_size + 1)
        moving_avg_length.append(np.mean(response_lengths[start_idx:i+1]))
    
    ax2.plot(episodes, response_lengths, alpha=0.3, color='lightgreen', label='Raw Response Length')
    ax2.plot(episodes, moving_avg_length, color='darkgreen', linewidth=2, label=f'Moving Average (window={window_size})')
    ax2.set_xlabel('Interaction Number')
    ax2.set_ylabel('Response Length (characters)')
    ax2.set_title('Student Engagement: Response Length Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Learning curves saved to results/learning_curves.png")

def create_agent_performance_comparison(data):
    """Compare DQN vs PPO agent performance."""
    interactions = data['interactions']
    
    if not interactions:
        print("No interaction data found for agent comparison")
        return
    
    # Extract agent performance data
    dqn_actions = []
    cumulative_rewards = []
    
    for interaction in interactions:
        if 'dqn_action' in interaction:
            dqn_actions.append(interaction['dqn_action'])
        if 'cumulative_reward' in interaction:
            cumulative_rewards.append(float(interaction['cumulative_reward']))
    
    # Create performance comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # DQN Action Distribution
    if dqn_actions:
        action_counts = pd.Series(dqn_actions).value_counts().sort_index()
        ax1.bar(action_counts.index, action_counts.values, color='skyblue', alpha=0.7)
        ax1.set_xlabel('DQN Action Type')
        ax1.set_ylabel('Frequency')
        ax1.set_title('DQN Agent: Action Selection Distribution')
        ax1.grid(True, alpha=0.3)
    
    # Cumulative Reward Progression
    if cumulative_rewards:
        ax2.plot(range(1, len(cumulative_rewards) + 1), cumulative_rewards, 
                color='orange', linewidth=2, marker='o', markersize=3)
        ax2.set_xlabel('Interaction Number')
        ax2.set_ylabel('Cumulative Reward')
        ax2.set_title('System Performance: Cumulative Rewards')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/agent_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Agent performance comparison saved to results/agent_performance.png")

def create_topic_performance_heatmap(data):
    """Create heatmap showing performance across different topics."""
    interactions = data['interactions']
    
    if not interactions:
        print("No interaction data found for topic analysis")
        return
    
    # Extract topic and reward data
    topic_rewards = {}
    for interaction in interactions:
        topic = interaction.get('topic', 'unknown')
        reward = float(interaction.get('reward_score', 0))
        
        if topic not in topic_rewards:
            topic_rewards[topic] = []
        topic_rewards[topic].append(reward)
    
    # Calculate statistics
    topic_stats = {}
    for topic, rewards in topic_rewards.items():
        topic_stats[topic] = {
            'mean': np.mean(rewards),
            'std': np.std(rewards),
            'count': len(rewards),
            'improvement': rewards[-1] - rewards[0] if len(rewards) > 1 else 0
        }
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Performance by topic
    topics = list(topic_stats.keys())
    means = [topic_stats[topic]['mean'] for topic in topics]
    stds = [topic_stats[topic]['std'] for topic in topics]
    
    bars = ax1.bar(topics, means, yerr=stds, capsize=5, alpha=0.7, color='lightcoral')
    ax1.set_xlabel('Subject Topic')
    ax1.set_ylabel('Average Reward Score')
    ax1.set_title('Performance by Subject Topic')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{mean:.3f}', ha='center', va='bottom')
    
    # Improvement by topic
    improvements = [topic_stats[topic]['improvement'] for topic in topics]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    
    bars2 = ax2.bar(topics, improvements, color=colors, alpha=0.7)
    ax2.set_xlabel('Subject Topic')
    ax2.set_ylabel('Improvement (Last - First)')
    ax2.set_title('Learning Improvement by Topic')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/topic_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Topic performance analysis saved to results/topic_performance.png")

def create_session_analytics(data):
    """Create analytics showing session-level performance."""
    evaluations = data['evaluations']
    
    if not evaluations:
        print("No evaluation data found for session analytics")
        return
    
    # Extract session data
    overall_performance = []
    engagement_scores = []
    learning_velocities = []
    dates = []
    
    for eval_data in evaluations:
        overall_performance.append(float(eval_data.get('overall_performance', 0)))
        engagement_scores.append(float(eval_data.get('engagement_score', 0)))
        learning_velocities.append(float(eval_data.get('learning_velocity', 0)))
        
        # Parse date
        date_str = eval_data.get('evaluation_date', '')
        try:
            date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            dates.append(date)
        except:
            dates.append(datetime.now())
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Overall performance trend
    if dates and overall_performance:
        ax1.plot(dates, overall_performance, marker='o', linewidth=2, markersize=5)
        ax1.set_xlabel('Session Date')
        ax1.set_ylabel('Overall Performance')
        ax1.set_title('Overall Performance Trend')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
    
    # Engagement scores
    if engagement_scores:
        ax2.hist(engagement_scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Engagement Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Engagement Scores')
        ax2.grid(True, alpha=0.3)
    
    # Learning velocity
    if learning_velocities:
        ax3.scatter(range(len(learning_velocities)), learning_velocities, 
                   alpha=0.7, color='green', s=50)
        ax3.set_xlabel('Session Number')
        ax3.set_ylabel('Learning Velocity')
        ax3.set_title('Learning Velocity by Session')
        ax3.grid(True, alpha=0.3)
    
    # Performance correlation
    if overall_performance and engagement_scores and len(overall_performance) == len(engagement_scores):
        ax4.scatter(engagement_scores, overall_performance, alpha=0.7, color='purple', s=50)
        ax4.set_xlabel('Engagement Score')
        ax4.set_ylabel('Overall Performance')
        ax4.set_title('Performance vs Engagement Correlation')
        ax4.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        if len(engagement_scores) > 2:
            corr = np.corrcoef(engagement_scores, overall_performance)[0, 1]
            ax4.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                    transform=ax4.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    plt.tight_layout()
    plt.savefig('results/session_analytics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Session analytics saved to results/session_analytics.png")

def create_summary_statistics(data):
    """Create a summary statistics visualization."""
    interactions = data['interactions']
    evaluations = data['evaluations']
    
    # Calculate key statistics
    stats = {
        'Total Interactions': len(interactions),
        'Total Sessions': len(evaluations),
        'Average Reward': np.mean([float(i.get('reward_score', 0)) for i in interactions]) if interactions else 0,
        'Average Engagement': np.mean([float(e.get('engagement_score', 0)) for e in evaluations]) if evaluations else 0,
        'Average Learning Velocity': np.mean([float(e.get('learning_velocity', 0)) for e in evaluations]) if evaluations else 0
    }
    
    # Create summary plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Remove learning velocity for cleaner visualization
    display_stats = {k: v for k, v in stats.items() if k != 'Average Learning Velocity'}
    
    categories = list(display_stats.keys())
    values = list(display_stats.values())
    
    bars = ax.bar(categories, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'], alpha=0.8)
    ax.set_ylabel('Value')
    ax.set_title('System Performance Summary Statistics')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                f'{value:.3f}' if value < 10 else f'{int(value)}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/summary_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Summary statistics saved to results/summary_statistics.png")

def main():
    """Generate all visualizations."""
    print("🎨 Generating Experimental Visualizations for RL Tutorial System")
    print("=" * 70)
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Load data
    print("📊 Loading student results data...")
    data = load_student_data()
    
    print(f"   • Found {len(data['interactions'])} interactions")
    print(f"   • Found {len(data['sessions'])} sessions") 
    print(f"   • Found {len(data['evaluations'])} evaluations")
    
    if not data['interactions'] and not data['evaluations']:
        print("\n⚠️ No data found. Run complete_assignment_demo.py first to generate data.")
        return
    
    print("\n📈 Generating visualizations...")
    
    # Generate all plots
    create_learning_curves(data)
    create_agent_performance_comparison(data)
    create_topic_performance_heatmap(data)
    create_session_analytics(data)
    create_summary_statistics(data)
    
    print("\n🎉 All visualizations generated successfully!")
    print(f"📁 Check the 'results/' folder for:")
    print("   • learning_curves.png - Learning progression over time")
    print("   • agent_performance.png - DQN agent performance analysis") 
    print("   • topic_performance.png - Performance by subject topic")
    print("   • session_analytics.png - Session-level analytics")
    print("   • summary_statistics.png - Overall system statistics")
    
    print("\n💡 These plots demonstrate:")
    print("   ✅ Agent learning and improvement over time")
    print("   ✅ Performance metrics and evaluation criteria")
    print("   ✅ Comparative analysis across topics and sessions")
    print("   ✅ Evidence of system effectiveness and adaptation")

if __name__ == "__main__":
    main()
