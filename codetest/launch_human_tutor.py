"""
Easy Launcher for Human Interactive Tutoring System

This script provides a simple menu-driven interface to launch
the human interactive tutoring system.

Author: Sanat Popli
Date: August 2025
"""

import os
import sys
from pathlib import Path

def print_banner():
    """Print welcome banner."""
    print("=" * 70)
    print("🎓 HUMAN INTERACTIVE AI TUTORING SYSTEM")
    print("   Adaptive Learning with Real-Time AI Coordination")
    print("=" * 70)
    print("📚 Features:")
    print("   • Real-time adaptation to your learning style")
    print("   • Multi-agent AI coordination (DQN + PPO)")
    print("   • Interactive questions across multiple subjects")
    print("   • Personalized hints and explanations")
    print("   • Progress tracking and analytics")
    print("=" * 70)

def get_student_info():
    """Get student information."""
    print("\n👤 Student Information:")
    name = input("Enter your name: ").strip()
    if not name:
        name = "Student"
    
    print(f"\nHello {name}! 👋")
    return name

def select_mode():
    """Let user select the tutoring mode."""
    print("\n🎯 Select Learning Mode:")
    print("1. 📖 Interactive Learning Session (Recommended)")
    print("2. 🏋️ Training Mode (Help AI Learn)")
    print("3. 🚀 Quick Demo")
    print("4. ❌ Exit")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        if choice in ['1', '2', '3', '4']:
            return int(choice)
        print("⚠️ Please enter 1, 2, 3, or 4")

def select_ai_strategy():
    """Let user select AI coordination strategy."""
    print("\n🤖 Select AI Teaching Strategy:")
    print("1. 🏗️ Hierarchical (Recommended) - Strategic oversight with content focus")
    print("2. ⚔️ Competitive - Both AI agents compete for best teaching approach")
    print("3. 🤝 Collaborative - AI agents work together cooperatively")
    
    strategies = {
        '1': 'hierarchical',
        '2': 'competitive', 
        '3': 'collaborative'
    }
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        if choice in strategies:
            return strategies[choice]
        print("⚠️ Please enter 1, 2, or 3")

def run_interactive_session(name, strategy):
    """Run interactive learning session."""
    print(f"\n🚀 Starting Interactive Session for {name}")
    print(f"🧠 AI Strategy: {strategy.title()}")
    print("\nPress Ctrl+C anytime to exit gracefully")
    print("=" * 50)
    
    # Run the human interactive tutor
    command = f'python human_interactive_tutor.py --mode session --name "{name}" --coordination {strategy}'
    os.system(command)

def run_training_mode(name):
    """Run training mode."""
    print(f"\n🏋️ Starting AI Training Mode with {name}")
    print("In this mode, you help the AI learn by providing feedback")
    
    episodes = input("\nHow many training episodes? (default 5): ").strip()
    if not episodes.isdigit():
        episodes = "5"
    
    print(f"\nStarting {episodes} training episodes...")
    print("=" * 50)
    
    command = f'python human_interactive_tutor.py --mode train --name "{name}" --episodes {episodes}'
    os.system(command)

def run_demo():
    """Run quick demo."""
    print("\n🚀 Starting Quick Demo")
    print("This will show you how the system works with sample interactions")
    print("=" * 50)
    
    command = 'python human_interactive_tutor.py --mode demo'
    os.system(command)

def check_system():
    """Check if the system is properly set up."""
    required_files = [
        "human_interactive_tutor.py",
        "src/environment/human_tutoring_environment.py",
        "src/orchestration/tutorial_orchestrator.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ System Check Failed!")
        print("Missing files:")
        for file in missing_files:
            print(f"   • {file}")
        print("\nPlease ensure all system files are present.")
        return False
    
    print("✅ System Check Passed!")
    return True

def main():
    """Main launcher function."""
    print_banner()
    
    # Check system
    if not check_system():
        input("\nPress Enter to exit...")
        return
    
    while True:
        try:
            # Get user choices
            mode = select_mode()
            
            if mode == 4:  # Exit
                print("\n👋 Thanks for using the AI Tutoring System!")
                print("Keep learning and growing! 🌱")
                break
            
            elif mode == 3:  # Demo
                run_demo()
                
            else:  # Interactive session or training
                name = get_student_info()
                
                if mode == 1:  # Interactive session
                    strategy = select_ai_strategy()
                    run_interactive_session(name, strategy)
                    
                elif mode == 2:  # Training mode
                    run_training_mode(name)
            
            # Ask if user wants to continue
            print("\n" + "=" * 50)
            continue_choice = input("Would you like to run another session? (y/n): ").strip().lower()
            if continue_choice not in ['y', 'yes']:
                print("\n👋 Thanks for using the AI Tutoring System!")
                break
                
        except KeyboardInterrupt:
            print("\n\n👋 Session ended by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please try again or contact support.")
            continue

if __name__ == "__main__":
    main()
