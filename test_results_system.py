"""
Quick test to demonstrate the student results storage system
"""

import json
import os
from student_results_manager import StudentResultsManager

def test_results_system():
    """Test the results storage system"""
    
    print("🧪 TESTING STUDENT RESULTS STORAGE SYSTEM")
    print("=" * 50)
    
    # Initialize results manager
    results_manager = StudentResultsManager()
    
    # Check if we have any stored results
    if os.path.exists("student_results"):
        print(f"\n📁 Found results directory with these files:")
        for file in os.listdir("student_results"):
            print(f"   • {file}")
            
        # Try to load and display some results
        try:
            evaluations_file = "student_results/student_evaluations.json"
            if os.path.exists(evaluations_file):
                with open(evaluations_file, 'r') as f:
                    evaluations = json.load(f)
                    
                print(f"\n📊 Found {len(evaluations)} student evaluations:")
                for eval_data in evaluations[-3:]:  # Show last 3
                    print(f"\n   👤 Student: {eval_data['student_name']} (ID: {eval_data['student_id']})")
                    print(f"   📅 Date: {eval_data['evaluation_date'][:19]}")
                    print(f"   📈 Performance: {eval_data['overall_performance']:.2f}")
                    print(f"   🎯 Total Interactions: {eval_data['total_interactions']}")
                    print(f"   ✅ Correct Responses: {eval_data['correct_responses']}")
                    print(f"   💬 Detailed Responses: {eval_data['detailed_responses']}")
                    print(f"   🔥 Engagement: {eval_data['engagement_score']:.2f}")
                    
        except Exception as e:
            print(f"   ⚠️ Error reading evaluations: {e}")
            
        # Check sessions
        try:
            sessions_file = "student_results/learning_sessions.json"
            if os.path.exists(sessions_file):
                with open(sessions_file, 'r') as f:
                    sessions = json.load(f)
                    
                print(f"\n📚 Found {len(sessions)} learning sessions:")
                for session in sessions[-2:]:  # Show last 2
                    print(f"\n   🔗 Session: {session['session_id']}")
                    print(f"   👤 Student: {session['student_name']}")
                    print(f"   🎯 Total Interactions: {session['total_interactions']}")
                    print(f"   📈 Final Reward: {session['cumulative_reward']:.2f}")
                    print(f"   🤖 Agent Updates: DQN={session.get('dqn_updates', 0)}, PPO={session.get('ppo_updates', 0)}")
                    
        except Exception as e:
            print(f"   ⚠️ Error reading sessions: {e}")
            
        # Check interactions
        try:
            interactions_file = "student_results/student_interactions.json"
            if os.path.exists(interactions_file):
                with open(interactions_file, 'r') as f:
                    interactions = json.load(f)
                    
                print(f"\n💬 Found {len(interactions)} individual interactions")
                if interactions:
                    latest = interactions[-1]
                    print(f"   📝 Latest: {latest['topic']} ({latest['difficulty']}) - Reward: {latest['reward']:.2f}")
                    
        except Exception as e:
            print(f"   ⚠️ Error reading interactions: {e}")
            
    else:
        print("\n📂 No results directory found. Run the main demo first!")
        print("   Command: python complete_assignment_demo.py")
        
    print(f"\n✅ Results system test complete!")
    print(f"💡 All student data is automatically saved to the 'student_results' folder")
    print(f"📊 You can analyze this data later for research or progress tracking")

if __name__ == "__main__":
    test_results_system()
