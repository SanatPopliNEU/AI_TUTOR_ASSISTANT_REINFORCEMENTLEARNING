"""
Main entry point for the Adaptive Tutorial Agent System.

This module orchestrates the reinforcement learning-enhanced tutoring system,
providing interfaces for training, evaluation, and demonstration modes.
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path
from typing import Dict

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from orchestration.tutorial_orchestrator import TutorialOrchestrator
from environment.tutoring_environment import TutoringEnvironment
from experiments.experiment_runner import ExperimentRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/tutorial_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from file."""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Default configuration
        return {
            'dqn': {
                'learning_rate': 1e-3,
                'epsilon_start': 1.0,
                'epsilon_end': 0.01,
                'epsilon_decay': 0.995,
                'batch_size': 32,
                'memory_size': 10000,
                'target_update': 100
            },
            'ppo': {
                'learning_rate': 3e-4,
                'clip_epsilon': 0.2,
                'batch_size': 2048,
                'n_epochs': 4,
                'gamma': 0.99,
                'gae_lambda': 0.95
            },
            'coordination': {
                'mode': 'hierarchical',
                'intervention_threshold': 0.4,
                'effectiveness_threshold': 0.3,
                'check_interval': 5
            }
        }


def main():
    """Main function to run the tutorial system."""
    parser = argparse.ArgumentParser(description='Adaptive Tutorial Agent System')
    parser.add_argument('--mode', choices=['train', 'eval', 'demo', 'experiment'], 
                       default='demo', help='Mode to run the system')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Configuration file path')
    parser.add_argument('--student-profile', type=str, default='beginner',
                       choices=['beginner', 'intermediate', 'advanced'],
                       help='Student profile for demonstration')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--save-model', type=str, default='models/tutorial_agent.pth',
                       help='Path to save trained model')
    
    args = parser.parse_args()
    
    # Create necessary directories
    Path('logs').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    
    logger.info(f"Starting Adaptive Tutorial Agent System in {args.mode} mode")
    
    try:
        if args.mode == 'train':
            run_training(args)
        elif args.mode == 'eval':
            run_evaluation(args)
        elif args.mode == 'demo':
            run_demonstration(args)
        elif args.mode == 'experiment':
            run_experiments(args)
    except Exception as e:
        logger.error(f"Error running system: {e}")
        raise


def run_training(args):
    """Run training mode to train the RL agents."""
    logger.info("Initializing training environment...")
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize environment and orchestrator
    env = TutoringEnvironment(config_path=args.config)
    orchestrator = TutorialOrchestrator(env, config)
    
    # Train the agents
    logger.info(f"Starting training for {args.episodes} episodes...")
    orchestrator.train(episodes=args.episodes)
    
    # Save trained models
    orchestrator.save_models(args.save_model)
    logger.info(f"Models saved to {args.save_model}")


def run_evaluation(args):
    """Run evaluation mode to test trained agents."""
    logger.info("Running evaluation...")
    
    # Load configuration
    config = load_config(args.config)
    
    env = TutoringEnvironment(config_path=args.config)
    orchestrator = TutorialOrchestrator(env, config)
    
    # Load trained models
    orchestrator.load_models(args.save_model)
    
    # Run evaluation
    results = orchestrator.evaluate(num_episodes=100)
    logger.info(f"Evaluation results: {results}")


def run_demonstration(args):
    """Run interactive demonstration mode."""
    logger.info(f"Starting demonstration with {args.student_profile} student profile...")
    
    # Load configuration
    config = load_config(args.config)
    
    env = TutoringEnvironment(config_path=args.config, student_profile=args.student_profile)
    orchestrator = TutorialOrchestrator(env, config)
    
    # Load pre-trained models if available
    try:
        orchestrator.load_models(args.save_model)
        logger.info("Loaded pre-trained models")
    except FileNotFoundError:
        logger.warning("No pre-trained models found, using random policy")
    
    # Run interactive demonstration
    orchestrator.demonstrate()


def run_experiments(args):
    """Run comprehensive experiments and analysis."""
    logger.info("Running experimental analysis...")
    
    experiment_runner = ExperimentRunner(config_path=args.config)
    experiment_runner.run_all_experiments()
    experiment_runner.generate_report()


if __name__ == "__main__":
    main()
