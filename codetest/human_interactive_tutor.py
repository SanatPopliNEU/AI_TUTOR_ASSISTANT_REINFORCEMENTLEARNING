"""
Human Interactive Tutorial System - Main Entry Point

This script provides real human interaction with the AI tutoring system,
replacing simulated students with actual human input and responses.

Author: Sanat Popli
Date: August 2025
"""

import argparse
import sys
import time
import numpy as np
from pathlib import Path
import os

# Add src to path for imports
script_dir = Path(__file__).parent.absolute()
src_path = script_dir / "src"
sys.path.insert(0, str(src_path))

# Also add the script directory itself
sys.path.insert(0, str(script_dir))

try:
    from environment.human_tutoring_environment import HumanTutoringEnvironment
    from orchestration.tutorial_orchestrator import TutorialOrchestrator
    from environment.tutoring_environment import ActionType
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {script_dir}")
    print(f"Src path: {src_path}")
    print(f"Python path: {sys.path}")
    sys.exit(1)

def run_human_session(student_name: str = None, coordination_mode: str = "hierarchical"):
    """
    Run an interactive tutoring session with a real human student.
    
    Args:
        student_name (str): Name of the human student
        coordination_mode (str): Coordination strategy to use
    """
    if not student_name:
        print("🎓 Welcome to the AI Tutorial System!")
        student_name = input("What's your name? ").strip() or "Student"
    
    print(f"\n👋 Hello {student_name}! Let's start your personalized learning session.")
    print(f"🤖 AI Coordination Mode: {coordination_mode.title()}")
    print("\nThis system uses advanced AI to adapt to your learning style in real-time.")
    print("The AI will ask questions, provide hints, and adjust difficulty based on your responses.")
    print("\n" + "="*60)
    
    # Initialize environment and orchestrator
    env = HumanTutoringEnvironment(student_name=student_name)
    
    # Configure the orchestrator with coordination mode
    config = {
        'coordination_mode': coordination_mode,
        'strategy_frequency': 5,
        'learning_rate': 0.001,
        'epsilon_decay': 0.995
    }
    
    orchestrator = TutorialOrchestrator(env, config=config)
    
    # Load any existing models
    try:
        orchestrator.load_models("models/")
        print("✅ AI models loaded successfully!")
    except:
        print("ℹ️ Training AI from scratch (no pre-trained models found)")
    
    state = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    consecutive_reviews = 0  # Track consecutive review actions
    
    print(f"\n🚀 Starting your learning session...")
    time.sleep(2)
    
    try:
        while not done and step_count < 30:  # Max 30 interactions
            step_count += 1
            
            # Get AI decision
            print(f"\n🧠 AI Tutor (Step {step_count}): Analyzing your learning state...")
            time.sleep(1)
            
            # AI selects action based on student state
            action, agent_type = orchestrator._coordinate_agents(state)
            
            # Force transition away from excessive review actions
            if action == ActionType.REVIEW_PREVIOUS.value:
                consecutive_reviews += 1
                if consecutive_reviews >= 3:  # After 3 reviews, force a question
                    print("🎯 AI Tutor: Time to test your knowledge!")
                    action = ActionType.ASK_QUESTION.value
                    agent_type = "content"
                    consecutive_reviews = 0
            else:
                consecutive_reviews = 0
            
            # Execute action
            state, reward, done, info = env.step(action)
            total_reward += reward
            
            # Show AI reasoning
            print(f"🤖 Decision: {ActionType(action).name} (by {agent_type.title()} Agent)")
            print(f"📊 Current Engagement: {info['engagement']:.1%}")
            print(f"💪 Motivation Level: {info['motivation']:.1%}")
            
            # Check if student wants to continue
            if step_count % 5 == 0:
                print(f"\n⏸️ Quick check-in after {step_count} interactions...")
                continue_session = input("Would you like to continue learning? (y/n): ").strip().lower()
                if continue_session in ['n', 'no', 'quit', 'exit']:
                    print("👋 Thanks for learning with us today!")
                    done = True
            
            # Safety check for engagement
            if info['engagement'] < 0.2:
                print("\n😴 I notice you might be getting tired.")
                take_break = input("Would you like to take a break? (y/n): ").strip().lower()
                if take_break in ['y', 'yes']:
                    env.human_student.take_break()
    
    except KeyboardInterrupt:
        print("\n\n👋 Session ended by user. Thanks for learning with us!")
    
    # Final summary
    print(f"\n🎯 Session Complete!")
    print(f"Total Interactions: {step_count}")
    print(f"AI Performance Score: {total_reward:.1f}")
    
    env.print_session_summary()
    
    # Save progress (optional)
    save_progress = input("\nWould you like to save your progress? (y/n): ").strip().lower()
    if save_progress in ['y', 'yes']:
        save_student_progress(student_name, env.get_final_summary())
        print("💾 Progress saved successfully!")

def save_student_progress(student_name: str, summary: dict):
    """Save student progress to file."""
    import json
    from datetime import datetime
    
    progress_dir = Path("student_progress")
    progress_dir.mkdir(exist_ok=True)
    
    filename = progress_dir / f"{student_name.replace(' ', '_')}_progress.json"
    
    # Load existing progress
    if filename.exists():
        with open(filename, 'r') as f:
            all_progress = json.load(f)
    else:
        all_progress = {"student_name": student_name, "sessions": []}
    
    # Add current session
    summary["session_date"] = datetime.now().isoformat()
    all_progress["sessions"].append(summary)
    
    # Save updated progress
    with open(filename, 'w') as f:
        json.dump(all_progress, f, indent=2)

def run_human_training_session(episodes: int = 10):
    """
    Run a training session where the AI learns from human interactions.
    
    Args:
        episodes (int): Number of training episodes
    """
    print(f"🏋️ AI Training Mode: Learning from human interactions")
    print(f"Episodes planned: {episodes}")
    
    student_name = input("What's your name for this training session? ").strip() or "Trainer"
    
    env = HumanTutoringEnvironment(student_name=student_name)
    orchestrator = TutorialOrchestrator(env, coordination_mode="hierarchical")
    
    for episode in range(episodes):
        print(f"\n📚 Training Episode {episode + 1}/{episodes}")
        
        state = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done and steps < 10:  # Shorter episodes for training
            action, agent_type = orchestrator.select_action(
                state, 
                env.human_student.get_session_summary(), 
                training=True
            )
            
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            
            # Train the agents
            orchestrator.train_step(state, action, reward, next_state, done, agent_type)
            
            state = next_state
            steps += 1
        
        print(f"Episode {episode + 1} completed: {steps} steps, reward: {episode_reward:.1f}")
        
        # Ask if user wants to continue
        if episode < episodes - 1:
            continue_training = input("Continue training? (y/n): ").strip().lower()
            if continue_training in ['n', 'no']:
                break
    
    # Save trained models
    save_models = input("Save the trained AI models? (y/n): ").strip().lower()
    if save_models in ['y', 'yes']:
        orchestrator.save_models("models/")
        print("🤖 AI models saved successfully!")

def quick_demo():
    """Run a quick demonstration of the human interaction system."""
    print("🚀 Quick Demo: Human Interactive AI Tutor")
    print("This is a shortened demo to show how the system works.")
    print("="*50)
    
    # Use default name for demo
    env = HumanTutoringEnvironment(student_name="Demo User")
    state = env.reset()
    
    # Simulate a few interactions
    actions = [0, 1, 2, 0, 3]  # Ask, Hint, Explain, Ask, Review
    action_names = ["Ask Question", "Provide Hint", "Explain Concept", "Ask Question", "Review"]
    
    for i, (action, name) in enumerate(zip(actions, action_names)):
        print(f"\n🤖 AI Action {i+1}: {name}")
        
        state, reward, done, info = env.step(action)
        
        print(f"📊 Reward: {reward:.1f}")
        print(f"💡 Engagement: {info['engagement']:.1%}")
        
        if done:
            break
    
    env.print_session_summary()

def main():
    """Main entry point for human interactive tutoring."""
    parser = argparse.ArgumentParser(description="Human Interactive AI Tutoring System")
    parser.add_argument("--mode", choices=["session", "train", "demo"], default="session",
                        help="Mode to run: session (interactive), train (AI learning), demo (quick demo)")
    parser.add_argument("--name", type=str, help="Student name")
    parser.add_argument("--coordination", choices=["hierarchical", "competitive", "collaborative"], 
                        default="hierarchical", help="AI coordination strategy")
    parser.add_argument("--episodes", type=int, default=10, help="Training episodes (for train mode)")
    
    args = parser.parse_args()
    
    if args.mode == "session":
        run_human_session(args.name, args.coordination)
    elif args.mode == "train":
        run_human_training_session(args.episodes)
    elif args.mode == "demo":
        quick_demo()
    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()
