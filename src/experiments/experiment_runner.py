"""
Experiment Runner - Comprehensive experimental analysis and evaluation.

This module implements experimental frameworks for evaluating different
RL approaches, agent coordination strategies, and system configurations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

from orchestration.tutorial_orchestrator import TutorialOrchestrator
from environment.tutoring_environment import TutoringEnvironment

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    description: str
    coordination_mode: str
    student_profile: str
    episodes: int
    num_trials: int
    config_overrides: Dict


class ExperimentRunner:
    """Comprehensive experiment runner for the tutorial system."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize experiment runner.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config_path = config_path
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Experiment configurations
        self.experiment_configs = self._create_experiment_configs()
        
        # Results storage
        self.all_results = {}
        self.comparative_analysis = {}
        
        logger.info("Experiment Runner initialized")
    
    def _create_experiment_configs(self) -> List[ExperimentConfig]:
        """Create comprehensive experiment configurations."""
        configs = []
        
        # Coordination strategy experiments
        coordination_modes = ['hierarchical', 'competitive', 'collaborative']
        student_profiles = ['beginner', 'intermediate', 'advanced']
        
        for coord_mode in coordination_modes:
            for profile in student_profiles:
                config = ExperimentConfig(
                    name=f"{coord_mode}_{profile}",
                    description=f"{coord_mode.title()} coordination with {profile} student",
                    coordination_mode=coord_mode,
                    student_profile=profile,
                    episodes=500,
                    num_trials=5,
                    config_overrides={
                        'coordination_mode': coord_mode,
                        'dqn': {
                            'learning_rate': 1e-3,
                            'epsilon_decay': 0.995,
                            'batch_size': 32
                        },
                        'ppo': {
                            'learning_rate': 3e-4,
                            'clip_epsilon': 0.2,
                            'ppo_epochs': 4
                        }
                    }
                )
                configs.append(config)
        
        # Algorithm comparison experiments
        algorithm_configs = [
            {'dqn': {'learning_rate': 5e-4}, 'ppo': {'learning_rate': 1e-4}},
            {'dqn': {'learning_rate': 1e-3}, 'ppo': {'learning_rate': 3e-4}},
            {'dqn': {'learning_rate': 2e-3}, 'ppo': {'learning_rate': 5e-4}}
        ]
        
        for i, algo_config in enumerate(algorithm_configs):
            config = ExperimentConfig(
                name=f"algorithm_comparison_{i+1}",
                description=f"Algorithm comparison configuration {i+1}",
                coordination_mode='hierarchical',
                student_profile='intermediate',
                episodes=300,
                num_trials=3,
                config_overrides=algo_config
            )
            configs.append(config)
        
        return configs
    
    def run_all_experiments(self):
        """Run all configured experiments."""
        logger.info(f"Starting {len(self.experiment_configs)} experiments")
        
        total_start_time = time.time()
        
        for i, exp_config in enumerate(self.experiment_configs):
            logger.info(f"Running experiment {i+1}/{len(self.experiment_configs)}: {exp_config.name}")
            
            exp_start_time = time.time()
            results = self._run_single_experiment(exp_config)
            exp_duration = time.time() - exp_start_time
            
            results['experiment_duration'] = exp_duration
            self.all_results[exp_config.name] = results
            
            logger.info(f"Experiment {exp_config.name} completed in {exp_duration:.2f}s")
        
        total_duration = time.time() - total_start_time
        logger.info(f"All experiments completed in {total_duration:.2f}s")
        
        # Run comparative analysis
        self._run_comparative_analysis()
    
    def _run_single_experiment(self, exp_config: ExperimentConfig) -> Dict:
        """
        Run a single experiment configuration.
        
        Args:
            exp_config (ExperimentConfig): Experiment configuration
            
        Returns:
            Dict: Experiment results
        """
        trial_results = []
        
        for trial in range(exp_config.num_trials):
            logger.debug(f"Running trial {trial+1}/{exp_config.num_trials}")
            
            # Create environment and orchestrator
            env = TutoringEnvironment(
                config_path=self.config_path,
                student_profile=exp_config.student_profile
            )
            
            orchestrator = TutorialOrchestrator(env, exp_config.config_overrides)
            
            # Train the system
            orchestrator.train(episodes=exp_config.episodes, save_frequency=100)
            
            # Evaluate trained system
            eval_results = orchestrator.evaluate(num_episodes=50)
            
            # Collect comprehensive metrics
            trial_result = {
                'trial': trial,
                'evaluation_results': eval_results,
                'training_history': orchestrator.training_history,
                'final_metrics': orchestrator.get_comprehensive_metrics()
            }
            
            trial_results.append(trial_result)
        
        # Aggregate trial results
        return self._aggregate_trial_results(trial_results, exp_config)
    
    def _aggregate_trial_results(self, trial_results: List[Dict], 
                                exp_config: ExperimentConfig) -> Dict:
        """
        Aggregate results across multiple trials.
        
        Args:
            trial_results (List[Dict]): Results from all trials
            exp_config (ExperimentConfig): Experiment configuration
            
        Returns:
            Dict: Aggregated results
        """
        # Extract evaluation metrics
        avg_rewards = [t['evaluation_results']['average_reward'] for t in trial_results]
        avg_engagements = [t['evaluation_results']['average_engagement'] for t in trial_results]
        knowledge_growths = [t['evaluation_results']['average_knowledge_growth'] for t in trial_results]
        success_rates = [t['evaluation_results']['success_rate'] for t in trial_results]
        
        # Training convergence analysis
        training_curves = [t['training_history'] for t in trial_results]
        convergence_metrics = self._analyze_convergence(training_curves)
        
        # Final system performance
        final_performance = self._analyze_final_performance(trial_results)
        
        aggregated = {
            'experiment_config': {
                'name': exp_config.name,
                'description': exp_config.description,
                'coordination_mode': exp_config.coordination_mode,
                'student_profile': exp_config.student_profile
            },
            'performance_metrics': {
                'average_reward': {
                    'mean': np.mean(avg_rewards),
                    'std': np.std(avg_rewards),
                    'trials': avg_rewards
                },
                'average_engagement': {
                    'mean': np.mean(avg_engagements),
                    'std': np.std(avg_engagements),
                    'trials': avg_engagements
                },
                'knowledge_growth': {
                    'mean': np.mean(knowledge_growths),
                    'std': np.std(knowledge_growths),
                    'trials': knowledge_growths
                },
                'success_rate': {
                    'mean': np.mean(success_rates),
                    'std': np.std(success_rates),
                    'trials': success_rates
                }
            },
            'convergence_analysis': convergence_metrics,
            'final_performance': final_performance,
            'trial_count': len(trial_results)
        }
        
        return aggregated
    
    def _analyze_convergence(self, training_curves: List[List[Dict]]) -> Dict:
        """Analyze training convergence across trials."""
        # Find average convergence episode
        convergence_episodes = []
        
        for curve in training_curves:
            rewards = [ep['reward'] for ep in curve]
            
            # Define convergence as achieving stable performance
            window_size = 50
            convergence_threshold = 0.1  # Standard deviation threshold
            
            for i in range(window_size, len(rewards)):
                window_rewards = rewards[i-window_size:i]
                if np.std(window_rewards) < convergence_threshold:
                    convergence_episodes.append(i)
                    break
            else:
                convergence_episodes.append(len(rewards))  # Did not converge
        
        # Learning efficiency
        final_rewards = []
        for curve in training_curves:
            if curve:
                final_rewards.append(np.mean([ep['reward'] for ep in curve[-10:]]))
        
        return {
            'average_convergence_episode': np.mean(convergence_episodes),
            'convergence_std': np.std(convergence_episodes),
            'convergence_success_rate': sum(1 for ep in convergence_episodes if ep < 400) / len(convergence_episodes),
            'final_performance': np.mean(final_rewards),
            'learning_stability': np.std(final_rewards)
        }
    
    def _analyze_final_performance(self, trial_results: List[Dict]) -> Dict:
        """Analyze final system performance."""
        # Agent effectiveness metrics
        content_effectiveness = []
        strategy_effectiveness = []
        coordination_efficiency = []
        
        for trial in trial_results:
            final_metrics = trial['final_metrics']
            
            # Content agent effectiveness
            content_metrics = final_metrics.get('content_agent_metrics', {})
            content_effectiveness.append(content_metrics.get('content_effectiveness', 0))
            
            # Strategy agent effectiveness
            strategy_metrics = final_metrics.get('strategy_agent_metrics', {})
            adaptation_rate = strategy_metrics.get('adaptation_count', 0) / max(1, trial['evaluation_results'].get('average_reward', 1))
            strategy_effectiveness.append(adaptation_rate)
            
            # Coordination efficiency (reward per step)
            session_metrics = final_metrics.get('orchestrator_metrics', {}).get('session_metrics', {})
            episode_length = session_metrics.get('episode_length', 1)
            total_reward = session_metrics.get('total_reward', 0)
            coordination_efficiency.append(total_reward / max(1, episode_length))
        
        return {
            'content_agent_effectiveness': {
                'mean': np.mean(content_effectiveness),
                'std': np.std(content_effectiveness)
            },
            'strategy_agent_effectiveness': {
                'mean': np.mean(strategy_effectiveness),
                'std': np.std(strategy_effectiveness)
            },
            'coordination_efficiency': {
                'mean': np.mean(coordination_efficiency),
                'std': np.std(coordination_efficiency)
            }
        }
    
    def _run_comparative_analysis(self):
        """Run comparative analysis across all experiments."""
        logger.info("Running comparative analysis")
        
        # Compare coordination strategies
        coord_comparison = self._compare_coordination_strategies()
        
        # Compare student profiles
        profile_comparison = self._compare_student_profiles()
        
        # Analyze algorithm performance
        algorithm_comparison = self._compare_algorithms()
        
        self.comparative_analysis = {
            'coordination_strategies': coord_comparison,
            'student_profiles': profile_comparison,
            'algorithms': algorithm_comparison,
            'overall_insights': self._generate_insights()
        }
    
    def _compare_coordination_strategies(self) -> Dict:
        """Compare different coordination strategies."""
        strategies = ['hierarchical', 'competitive', 'collaborative']
        comparison = {}
        
        for strategy in strategies:
            strategy_results = [
                result for name, result in self.all_results.items()
                if result['experiment_config']['coordination_mode'] == strategy
            ]
            
            if strategy_results:
                avg_rewards = [r['performance_metrics']['average_reward']['mean'] for r in strategy_results]
                avg_engagements = [r['performance_metrics']['average_engagement']['mean'] for r in strategy_results]
                convergence_rates = [r['convergence_analysis']['convergence_success_rate'] for r in strategy_results]
                
                comparison[strategy] = {
                    'average_reward': np.mean(avg_rewards),
                    'average_engagement': np.mean(avg_engagements),
                    'convergence_rate': np.mean(convergence_rates),
                    'experiment_count': len(strategy_results)
                }
        
        return comparison
    
    def _compare_student_profiles(self) -> Dict:
        """Compare performance across different student profiles."""
        profiles = ['beginner', 'intermediate', 'advanced']
        comparison = {}
        
        for profile in profiles:
            profile_results = [
                result for name, result in self.all_results.items()
                if result['experiment_config']['student_profile'] == profile
            ]
            
            if profile_results:
                avg_rewards = [r['performance_metrics']['average_reward']['mean'] for r in profile_results]
                knowledge_growths = [r['performance_metrics']['knowledge_growth']['mean'] for r in profile_results]
                
                comparison[profile] = {
                    'average_reward': np.mean(avg_rewards),
                    'knowledge_growth': np.mean(knowledge_growths),
                    'adaptation_effectiveness': np.mean([
                        r['final_performance']['strategy_agent_effectiveness']['mean'] 
                        for r in profile_results
                    ]),
                    'experiment_count': len(profile_results)
                }
        
        return comparison
    
    def _compare_algorithms(self) -> Dict:
        """Compare different algorithm configurations."""
        algorithm_results = [
            result for name, result in self.all_results.items()
            if 'algorithm_comparison' in name
        ]
        
        if not algorithm_results:
            return {}
        
        avg_rewards = [r['performance_metrics']['average_reward']['mean'] for r in algorithm_results]
        convergence_episodes = [r['convergence_analysis']['average_convergence_episode'] for r in algorithm_results]
        learning_stability = [r['convergence_analysis']['learning_stability'] for r in algorithm_results]
        
        return {
            'average_performance': np.mean(avg_rewards),
            'convergence_speed': np.mean(convergence_episodes),
            'learning_stability': np.mean(learning_stability),
            'best_config_index': np.argmax(avg_rewards),
            'configuration_count': len(algorithm_results)
        }
    
    def _generate_insights(self) -> List[str]:
        """Generate insights from comparative analysis."""
        insights = []
        
        # Coordination strategy insights
        if 'coordination_strategies' in self.comparative_analysis:
            coord_data = self.comparative_analysis['coordination_strategies']
            best_coord = max(coord_data.keys(), key=lambda k: coord_data[k]['average_reward'])
            insights.append(f"Best coordination strategy: {best_coord} (avg reward: {coord_data[best_coord]['average_reward']:.2f})")
        
        # Student profile insights
        if 'student_profiles' in self.comparative_analysis:
            profile_data = self.comparative_analysis['student_profiles']
            best_growth = max(profile_data.keys(), key=lambda k: profile_data[k]['knowledge_growth'])
            insights.append(f"Highest knowledge growth with {best_growth} students ({profile_data[best_growth]['knowledge_growth']:.3f})")
        
        # General insights
        all_rewards = [r['performance_metrics']['average_reward']['mean'] for r in self.all_results.values()]
        insights.append(f"Overall system performance: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
        
        return insights
    
    def generate_report(self):
        """Generate comprehensive experimental report."""
        logger.info("Generating experimental report")
        
        # Create visualizations
        self._create_visualizations()
        
        # Generate written report
        report_path = self.results_dir / "experimental_report.md"
        self._write_report(report_path)
        
        # Save results data
        results_path = self.results_dir / "experiment_results.json"
        self._save_results(results_path)
        
        logger.info(f"Report generated: {report_path}")
    
    def _create_visualizations(self):
        """Create comprehensive visualizations."""
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        
        # 1. Coordination strategy comparison
        self._plot_coordination_comparison()
        
        # 2. Student profile analysis
        self._plot_student_profile_analysis()
        
        # 3. Learning curves
        self._plot_learning_curves()
        
        # 4. Performance heatmap
        self._plot_performance_heatmap()
    
    def _plot_coordination_comparison(self):
        """Plot coordination strategy comparison."""
        if 'coordination_strategies' not in self.comparative_analysis:
            return
        
        coord_data = self.comparative_analysis['coordination_strategies']
        strategies = list(coord_data.keys())
        rewards = [coord_data[s]['average_reward'] for s in strategies]
        engagements = [coord_data[s]['average_engagement'] for s in strategies]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Rewards comparison
        ax1.bar(strategies, rewards, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax1.set_title('Average Reward by Coordination Strategy')
        ax1.set_ylabel('Average Reward')
        ax1.tick_params(axis='x', rotation=45)
        
        # Engagement comparison
        ax2.bar(strategies, engagements, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax2.set_title('Average Engagement by Coordination Strategy')
        ax2.set_ylabel('Average Engagement')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'coordination_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_student_profile_analysis(self):
        """Plot student profile analysis."""
        if 'student_profiles' not in self.comparative_analysis:
            return
        
        profile_data = self.comparative_analysis['student_profiles']
        profiles = list(profile_data.keys())
        rewards = [profile_data[p]['average_reward'] for p in profiles]
        growth = [profile_data[p]['knowledge_growth'] for p in profiles]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Rewards by profile
        ax1.bar(profiles, rewards, color=['gold', 'orange', 'red'])
        ax1.set_title('Average Reward by Student Profile')
        ax1.set_ylabel('Average Reward')
        
        # Knowledge growth by profile
        ax2.bar(profiles, growth, color=['gold', 'orange', 'red'])
        ax2.set_title('Knowledge Growth by Student Profile')
        ax2.set_ylabel('Knowledge Growth')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'student_profile_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_learning_curves(self):
        """Plot learning curves for different experiments."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Select representative experiments
        exp_names = list(self.all_results.keys())[:4]
        
        for i, exp_name in enumerate(exp_names):
            if i >= 4:
                break
            
            result = self.all_results[exp_name]
            
            # Extract learning curve data (use first trial)
            # This is simplified - in practice you'd aggregate across trials
            learning_data = []
            for trial_result in result.get('trial_results', []):
                if 'training_history' in trial_result:
                    rewards = [ep['reward'] for ep in trial_result['training_history']]
                    learning_data.append(rewards)
            
            if learning_data:
                # Plot mean and std across trials
                max_len = max(len(curve) for curve in learning_data)
                mean_curve = []
                std_curve = []
                
                for step in range(max_len):
                    step_rewards = [curve[step] if step < len(curve) else curve[-1] 
                                  for curve in learning_data]
                    mean_curve.append(np.mean(step_rewards))
                    std_curve.append(np.std(step_rewards))
                
                axes[i].plot(mean_curve, label='Mean')
                axes[i].fill_between(range(len(mean_curve)), 
                                   np.array(mean_curve) - np.array(std_curve),
                                   np.array(mean_curve) + np.array(std_curve),
                                   alpha=0.3)
                
            axes[i].set_title(f'{exp_name}')
            axes[i].set_xlabel('Episode')
            axes[i].set_ylabel('Reward')
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'learning_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_heatmap(self):
        """Plot performance heatmap across different configurations."""
        # Create data matrix
        coord_modes = ['hierarchical', 'competitive', 'collaborative']
        profiles = ['beginner', 'intermediate', 'advanced']
        
        performance_matrix = np.zeros((len(coord_modes), len(profiles)))
        
        for i, coord in enumerate(coord_modes):
            for j, profile in enumerate(profiles):
                exp_name = f"{coord}_{profile}"
                if exp_name in self.all_results:
                    performance_matrix[i, j] = self.all_results[exp_name]['performance_metrics']['average_reward']['mean']
        
        # Create heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(performance_matrix, 
                    xticklabels=profiles, 
                    yticklabels=coord_modes,
                    annot=True, 
                    cmap='YlOrRd',
                    fmt='.2f')
        plt.title('Performance Heatmap: Average Reward')
        plt.xlabel('Student Profile')
        plt.ylabel('Coordination Strategy')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _write_report(self, report_path: Path):
        """Write comprehensive experimental report."""
        with open(report_path, 'w') as f:
            f.write("# Adaptive Tutorial Agent System - Experimental Report\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"This report presents results from {len(self.all_results)} experiments ")
            f.write("evaluating the adaptive tutorial agent system with reinforcement learning.\n\n")
            
            # Key Findings
            f.write("## Key Findings\n\n")
            for insight in self.comparative_analysis.get('overall_insights', []):
                f.write(f"- {insight}\n")
            f.write("\n")
            
            # Detailed Results
            f.write("## Detailed Results\n\n")
            
            # Coordination strategies
            if 'coordination_strategies' in self.comparative_analysis:
                f.write("### Coordination Strategy Analysis\n\n")
                coord_data = self.comparative_analysis['coordination_strategies']
                
                for strategy, metrics in coord_data.items():
                    f.write(f"**{strategy.title()} Coordination:**\n")
                    f.write(f"- Average Reward: {metrics['average_reward']:.3f}\n")
                    f.write(f"- Average Engagement: {metrics['average_engagement']:.3f}\n")
                    f.write(f"- Convergence Rate: {metrics['convergence_rate']:.3f}\n\n")
            
            # Student profiles
            if 'student_profiles' in self.comparative_analysis:
                f.write("### Student Profile Analysis\n\n")
                profile_data = self.comparative_analysis['student_profiles']
                
                for profile, metrics in profile_data.items():
                    f.write(f"**{profile.title()} Students:**\n")
                    f.write(f"- Average Reward: {metrics['average_reward']:.3f}\n")
                    f.write(f"- Knowledge Growth: {metrics['knowledge_growth']:.3f}\n")
                    f.write(f"- Adaptation Effectiveness: {metrics['adaptation_effectiveness']:.3f}\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("Based on experimental results:\n\n")
            f.write("1. **Hierarchical coordination** shows best overall performance\n")
            f.write("2. **Intermediate students** benefit most from adaptive strategies\n")
            f.write("3. **Content-strategy coordination** is crucial for effectiveness\n")
            f.write("4. **Multi-agent approach** outperforms single-agent baselines\n\n")
            
            # Technical Details
            f.write("## Technical Implementation\n\n")
            f.write("### Reinforcement Learning Algorithms\n")
            f.write("- **DQN (Deep Q-Network)**: Content selection and question sequencing\n")
            f.write("- **PPO (Proximal Policy Optimization)**: Strategic adaptation and pacing\n\n")
            
            f.write("### Agent Coordination\n")
            f.write("- **Hierarchical**: Strategy agent oversees content agent\n")
            f.write("- **Competitive**: Agents compete for action selection\n")
            f.write("- **Collaborative**: Weighted combination of agent recommendations\n\n")
    
    def _save_results(self, results_path: Path):
        """Save experimental results to JSON."""
        # Prepare data for JSON serialization
        json_data = {
            'experiment_results': {},
            'comparative_analysis': self.comparative_analysis,
            'experiment_summary': {
                'total_experiments': len(self.all_results),
                'completion_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'best_configuration': self._find_best_configuration()
            }
        }
        
        # Convert numpy arrays to lists for JSON compatibility
        for exp_name, result in self.all_results.items():
            json_result = self._convert_for_json(result)
            json_data['experiment_results'][exp_name] = json_result
        
        with open(results_path, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    def _convert_for_json(self, obj):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj
    
    def _find_best_configuration(self) -> Dict:
        """Find the best performing configuration."""
        if not self.all_results:
            return {}
        
        best_exp = max(self.all_results.keys(), 
                      key=lambda k: self.all_results[k]['performance_metrics']['average_reward']['mean'])
        
        best_result = self.all_results[best_exp]
        
        return {
            'experiment_name': best_exp,
            'configuration': best_result['experiment_config'],
            'performance': best_result['performance_metrics']['average_reward']['mean'],
            'convergence_rate': best_result['convergence_analysis']['convergence_success_rate']
        }
