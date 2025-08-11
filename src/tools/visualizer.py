"""
Visualization utilities for the adaptive tutorial system.

This module provides comprehensive visualization tools for analyzing
agent performance, learning curves, and system effectiveness.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TutorialVisualizer:
    """Comprehensive visualization toolkit for tutorial system analysis."""
    
    def __init__(self, output_dir: str = "visualizations"):
        """
        Initialize visualizer.
        
        Args:
            output_dir (str): Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        logger.info(f"Visualizer initialized, output directory: {self.output_dir}")
    
    def plot_training_progress(self, training_history: List[Dict], title: str = "Training Progress"):
        """
        Plot training progress over episodes.
        
        Args:
            training_history (List[Dict]): Training history data
            title (str): Plot title
        """
        if not training_history:
            logger.warning("No training history data provided")
            return
        
        episodes = [h['episode'] for h in training_history]
        rewards = [h['reward'] for h in training_history]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Raw rewards
        axes[0, 0].plot(episodes, rewards, alpha=0.7, linewidth=1)
        
        # Smoothed rewards (moving average)
        window_size = min(50, len(rewards) // 10)
        if window_size > 1:
            smoothed_rewards = self._moving_average(rewards, window_size)
            axes[0, 0].plot(episodes[window_size-1:], smoothed_rewards, 
                           linewidth=2, label=f'Moving Avg ({window_size})')
        
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Content agent metrics
        if training_history[0].get('content_metrics'):
            content_effectiveness = [h['content_metrics'].get('content_effectiveness', 0) 
                                   for h in training_history]
            axes[0, 1].plot(episodes, content_effectiveness, color='green')
            axes[0, 1].set_title('Content Agent Effectiveness')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Effectiveness')
            axes[0, 1].grid(True)
        
        # Strategy agent metrics
        if training_history[0].get('strategy_metrics'):
            adaptation_rates = [h['strategy_metrics'].get('adaptation_effectiveness', 0) 
                              for h in training_history]
            axes[1, 0].plot(episodes, adaptation_rates, color='orange')
            axes[1, 0].set_title('Strategy Agent Adaptation Rate')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Adaptation Success Rate')
            axes[1, 0].grid(True)
        
        # Cumulative reward
        cumulative_rewards = np.cumsum(rewards)
        axes[1, 1].plot(episodes, cumulative_rewards, color='purple')
        axes[1, 1].set_title('Cumulative Reward')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Cumulative Reward')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{title.lower().replace(" ", "_")}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_student_analytics(self, session_data: List[Dict], title: str = "Student Learning Analytics"):
        """
        Plot student learning analytics throughout session.
        
        Args:
            session_data (List[Dict]): Session data with student metrics
            title (str): Plot title
        """
        if not session_data:
            logger.warning("No session data provided")
            return
        
        steps = list(range(len(session_data)))
        engagement = [d.get('engagement', 0) for d in session_data]
        motivation = [d.get('motivation', 0) for d in session_data]
        
        # Extract knowledge levels if available
        knowledge_data = []
        for d in session_data:
            if 'knowledge_levels' in d:
                avg_knowledge = np.mean(list(d['knowledge_levels'].values()))
                knowledge_data.append(avg_knowledge)
            else:
                knowledge_data.append(0)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Engagement over time
        axes[0, 0].plot(steps, engagement, color='blue', linewidth=2, label='Engagement')
        axes[0, 0].fill_between(steps, engagement, alpha=0.3, color='blue')
        axes[0, 0].set_title('Student Engagement')
        axes[0, 0].set_xlabel('Session Step')
        axes[0, 0].set_ylabel('Engagement Level')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(True)
        
        # Motivation over time
        axes[0, 1].plot(steps, motivation, color='red', linewidth=2, label='Motivation')
        axes[0, 1].fill_between(steps, motivation, alpha=0.3, color='red')
        axes[0, 1].set_title('Student Motivation')
        axes[0, 1].set_xlabel('Session Step')
        axes[0, 1].set_ylabel('Motivation Level')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True)
        
        # Knowledge progression
        axes[1, 0].plot(steps, knowledge_data, color='green', linewidth=2, label='Knowledge')
        axes[1, 0].fill_between(steps, knowledge_data, alpha=0.3, color='green')
        axes[1, 0].set_title('Knowledge Progression')
        axes[1, 0].set_xlabel('Session Step')
        axes[1, 0].set_ylabel('Average Knowledge Level')
        axes[1, 0].grid(True)
        
        # Combined view
        axes[1, 1].plot(steps, engagement, label='Engagement', linewidth=2)
        axes[1, 1].plot(steps, motivation, label='Motivation', linewidth=2)
        axes[1, 1].plot(steps, knowledge_data, label='Knowledge', linewidth=2)
        axes[1, 1].set_title('Combined Learning Metrics')
        axes[1, 1].set_xlabel('Session Step')
        axes[1, 1].set_ylabel('Level')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{title.lower().replace(" ", "_")}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_action_distribution(self, action_history: List[Dict], title: str = "Action Distribution Analysis"):
        """
        Plot distribution of actions taken by agents.
        
        Args:
            action_history (List[Dict]): History of actions taken
            title (str): Plot title
        """
        if not action_history:
            logger.warning("No action history provided")
            return
        
        # Count action frequencies
        action_counts = {}
        agent_action_counts = {'content': {}, 'strategy': {}}
        
        for action_data in action_history:
            action = action_data.get('action', 'unknown')
            agent_type = action_data.get('agent_type', 'unknown')
            
            # Overall action counts
            action_counts[action] = action_counts.get(action, 0) + 1
            
            # Agent-specific action counts
            if agent_type in agent_action_counts:
                agent_action_counts[agent_type][action] = agent_action_counts[agent_type].get(action, 0) + 1
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Overall action distribution
        actions = list(action_counts.keys())
        counts = list(action_counts.values())
        
        axes[0, 0].pie(counts, labels=actions, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Overall Action Distribution')
        
        # Action frequency over time
        action_timeline = [a.get('action', 'unknown') for a in action_history]
        unique_actions = list(set(action_timeline))
        
        for action in unique_actions[:5]:  # Limit to top 5 actions
            action_freq = self._calculate_action_frequency(action_timeline, action, window_size=10)
            axes[0, 1].plot(action_freq, label=action, linewidth=2)
        
        axes[0, 1].set_title('Action Frequency Over Time')
        axes[0, 1].set_xlabel('Time Window')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Content agent actions
        if agent_action_counts['content']:
            content_actions = list(agent_action_counts['content'].keys())
            content_counts = list(agent_action_counts['content'].values())
            axes[1, 0].bar(content_actions, content_counts, color='lightblue')
            axes[1, 0].set_title('Content Agent Actions')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Strategy agent actions
        if agent_action_counts['strategy']:
            strategy_actions = list(agent_action_counts['strategy'].keys())
            strategy_counts = list(agent_action_counts['strategy'].values())
            axes[1, 1].bar(strategy_actions, strategy_counts, color='lightcoral')
            axes[1, 1].set_title('Strategy Agent Actions')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{title.lower().replace(" ", "_")}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_comparative_analysis(self, experiment_results: Dict, title: str = "Comparative Analysis"):
        """
        Plot comparative analysis across different experiments.
        
        Args:
            experiment_results (Dict): Results from multiple experiments
            title (str): Plot title
        """
        if not experiment_results:
            logger.warning("No experiment results provided")
            return
        
        # Extract comparison data
        exp_names = list(experiment_results.keys())
        avg_rewards = [r['performance_metrics']['average_reward']['mean'] 
                      for r in experiment_results.values()]
        avg_engagements = [r['performance_metrics']['average_engagement']['mean'] 
                          for r in experiment_results.values()]
        knowledge_growths = [r['performance_metrics']['knowledge_growth']['mean'] 
                           for r in experiment_results.values()]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Average rewards comparison
        bars1 = axes[0, 0].bar(exp_names, avg_rewards, color='skyblue')
        axes[0, 0].set_title('Average Reward by Experiment')
        axes[0, 0].set_ylabel('Average Reward')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, avg_rewards):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.2f}', ha='center', va='bottom')
        
        # Engagement comparison
        bars2 = axes[0, 1].bar(exp_names, avg_engagements, color='lightgreen')
        axes[0, 1].set_title('Average Engagement by Experiment')
        axes[0, 1].set_ylabel('Average Engagement')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars2, avg_engagements):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.2f}', ha='center', va='bottom')
        
        # Knowledge growth comparison
        bars3 = axes[1, 0].bar(exp_names, knowledge_growths, color='coral')
        axes[1, 0].set_title('Knowledge Growth by Experiment')
        axes[1, 0].set_ylabel('Knowledge Growth')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars3, knowledge_growths):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Scatter plot: Reward vs Engagement
        axes[1, 1].scatter(avg_rewards, avg_engagements, s=100, alpha=0.7)
        
        # Add experiment labels
        for i, name in enumerate(exp_names):
            axes[1, 1].annotate(name, (avg_rewards[i], avg_engagements[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        axes[1, 1].set_title('Reward vs Engagement')
        axes[1, 1].set_xlabel('Average Reward')
        axes[1, 1].set_ylabel('Average Engagement')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{title.lower().replace(" ", "_")}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_dashboard(self, comprehensive_data: Dict):
        """
        Create a comprehensive dashboard visualization.
        
        Args:
            comprehensive_data (Dict): All available data for visualization
        """
        fig = plt.figure(figsize=(20, 15))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1])
        
        # Training progress (top row, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        if 'training_history' in comprehensive_data:
            training_data = comprehensive_data['training_history']
            episodes = [h['episode'] for h in training_data]
            rewards = [h['reward'] for h in training_data]
            ax1.plot(episodes, rewards, linewidth=2)
            ax1.set_title('Training Progress', fontsize=14)
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.grid(True)
        
        # Performance metrics (top row, right side)
        ax2 = fig.add_subplot(gs[0, 2:])
        if 'performance_metrics' in comprehensive_data:
            metrics = comprehensive_data['performance_metrics']
            metric_names = list(metrics.keys())
            metric_values = [metrics[m]['mean'] for m in metric_names]
            bars = ax2.bar(metric_names, metric_values, color=['blue', 'green', 'red', 'orange'])
            ax2.set_title('Performance Metrics', fontsize=14)
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, metric_values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.2f}', ha='center', va='bottom')
        
        # Student analytics (middle row, left)
        ax3 = fig.add_subplot(gs[1, :2])
        if 'student_analytics' in comprehensive_data:
            analytics = comprehensive_data['student_analytics']
            steps = range(len(analytics))
            engagement = [a.get('engagement', 0) for a in analytics]
            motivation = [a.get('motivation', 0) for a in analytics]
            
            ax3.plot(steps, engagement, label='Engagement', linewidth=2)
            ax3.plot(steps, motivation, label='Motivation', linewidth=2)
            ax3.set_title('Student Metrics Over Time', fontsize=14)
            ax3.set_xlabel('Session Step')
            ax3.set_ylabel('Level')
            ax3.legend()
            ax3.grid(True)
        
        # Agent effectiveness (middle row, right)
        ax4 = fig.add_subplot(gs[1, 2:])
        if 'agent_metrics' in comprehensive_data:
            agent_data = comprehensive_data['agent_metrics']
            content_eff = agent_data.get('content_effectiveness', 0)
            strategy_eff = agent_data.get('strategy_effectiveness', 0)
            
            agents = ['Content Agent', 'Strategy Agent']
            effectiveness = [content_eff, strategy_eff]
            
            bars = ax4.bar(agents, effectiveness, color=['lightblue', 'lightcoral'])
            ax4.set_title('Agent Effectiveness', fontsize=14)
            ax4.set_ylabel('Effectiveness Score')
            ax4.set_ylim(0, 1)
            
            for bar, value in zip(bars, effectiveness):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # Action distribution (bottom row, left)
        ax5 = fig.add_subplot(gs[2, :2])
        if 'action_distribution' in comprehensive_data:
            actions = list(comprehensive_data['action_distribution'].keys())
            counts = list(comprehensive_data['action_distribution'].values())
            
            ax5.pie(counts, labels=actions, autopct='%1.1f%%', startangle=90)
            ax5.set_title('Action Distribution', fontsize=14)
        
        # Learning efficiency (bottom row, right)
        ax6 = fig.add_subplot(gs[2, 2:])
        if 'learning_efficiency' in comprehensive_data:
            efficiency_data = comprehensive_data['learning_efficiency']
            categories = list(efficiency_data.keys())
            values = list(efficiency_data.values())
            
            bars = ax6.barh(categories, values, color='gold')
            ax6.set_title('Learning Efficiency Metrics', fontsize=14)
            ax6.set_xlabel('Score')
            
            for bar, value in zip(bars, values):
                ax6.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{value:.3f}', ha='left', va='center')
        
        plt.suptitle('Adaptive Tutorial System - Comprehensive Dashboard', fontsize=18)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comprehensive_dashboard.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def _moving_average(self, data: List[float], window_size: int) -> List[float]:
        """Calculate moving average of data."""
        return [np.mean(data[i-window_size+1:i+1]) for i in range(window_size-1, len(data))]
    
    def _calculate_action_frequency(self, action_timeline: List[str], 
                                  target_action: str, window_size: int = 10) -> List[float]:
        """Calculate frequency of specific action over time windows."""
        frequencies = []
        for i in range(window_size, len(action_timeline) + 1):
            window = action_timeline[i-window_size:i]
            freq = window.count(target_action) / window_size
            frequencies.append(freq)
        return frequencies
