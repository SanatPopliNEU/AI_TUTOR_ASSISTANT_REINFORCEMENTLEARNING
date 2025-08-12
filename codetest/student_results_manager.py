"""
Student Results Manager - Comprehensive Storage and Analytics System
Stores all student interactions, evaluations, and progress data
Supports JSON export, detailed analytics, and historical tracking
"""

import json
import os
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import statistics

@dataclass
class InteractionRecord:
    """Individual question-answer interaction record"""
    timestamp: str
    session_id: str
    question_text: str
    topic: str
    difficulty: str
    student_response: str
    response_length: int
    reward_score: float
    feedback: str
    dqn_action: int
    ppo_topic_selection: str
    cumulative_reward: float
    session_number: int

@dataclass
class SessionSummary:
    """Complete learning session summary"""
    session_id: str
    student_id: str
    start_time: str
    end_time: str
    duration_minutes: float
    total_interactions: int
    topics_covered: List[str]
    difficulties_attempted: List[str]
    average_reward: float
    total_reward: float
    improvement_trend: str
    engagement_level: str
    agent_coordination_mode: str

@dataclass
class StudentEvaluation:
    """Comprehensive student evaluation data"""
    student_id: str
    student_name: str
    evaluation_date: str
    overall_performance: float
    topic_performance: Dict[str, float]
    difficulty_performance: Dict[str, float]
    learning_velocity: float
    engagement_score: float
    total_sessions: int
    total_interactions: int
    correct_responses: int
    detailed_responses: int
    accuracy_rate: float
    detailed_response_rate: float
    preferred_topics: List[str]
    improvement_areas: List[str]
    strength_areas: List[str]
    learning_style: str

class StudentResultsManager:
    """Manages all student results storage and analytics"""
    
    def __init__(self, storage_directory: str = "student_results"):
        self.storage_dir = storage_directory
        self.ensure_storage_directory()
        
        # File paths
        self.interactions_file = os.path.join(self.storage_dir, "interactions.json")
        self.sessions_file = os.path.join(self.storage_dir, "sessions.json")
        self.evaluations_file = os.path.join(self.storage_dir, "evaluations.json")
        self.analytics_file = os.path.join(self.storage_dir, "analytics_summary.json")
        
        # Initialize storage files
        self.initialize_storage_files()
    
    def ensure_storage_directory(self):
        """Create storage directory if it doesn't exist"""
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
            print(f"📁 Created storage directory: {self.storage_dir}")
    
    def initialize_storage_files(self):
        """Initialize JSON storage files if they don't exist"""
        files_to_init = [
            self.interactions_file,
            self.sessions_file, 
            self.evaluations_file,
            self.analytics_file
        ]
        
        for file_path in files_to_init:
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    json.dump([], f, indent=2)
    
    def save_interaction(self, student_id: str, session_id: str, question_data: Dict[str, Any], 
                        response_data: Dict[str, Any], agent_data: Dict[str, Any]):
        """Save individual question-answer interaction"""
        
        interaction = InteractionRecord(
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            question_text=question_data.get('question', ''),
            topic=question_data.get('topic', ''),
            difficulty=question_data.get('difficulty', ''),
            student_response=response_data.get('response', ''),
            response_length=len(response_data.get('response', '')),
            reward_score=response_data.get('reward', 0.0),
            feedback=response_data.get('feedback', ''),
            dqn_action=agent_data.get('dqn_action', 0),
            ppo_topic_selection=agent_data.get('ppo_topic', ''),
            cumulative_reward=agent_data.get('cumulative_reward', 0.0),
            session_number=agent_data.get('session_number', 1)
        )
        
        # Load existing interactions
        interactions = self.load_json_data(self.interactions_file)
        
        # Add new interaction
        interactions.append(asdict(interaction))
        
        # Save updated interactions
        self.save_json_data(self.interactions_file, interactions)
        
        print(f"💾 Saved interaction for student {student_id} in session {session_id}")
    
    def save_session_summary(self, session_data: Dict[str, Any]):
        """Save complete session summary"""
        
        session = SessionSummary(
            session_id=session_data.get('session_id', ''),
            student_id=session_data.get('student_id', ''),
            start_time=session_data.get('start_time', ''),
            end_time=datetime.now().isoformat(),
            duration_minutes=session_data.get('duration_minutes', 0.0),
            total_interactions=session_data.get('total_interactions', 0),
            topics_covered=session_data.get('topics_covered', []),
            difficulties_attempted=session_data.get('difficulties_attempted', []),
            average_reward=session_data.get('average_reward', 0.0),
            total_reward=session_data.get('total_reward', 0.0),
            improvement_trend=session_data.get('improvement_trend', 'stable'),
            engagement_level=session_data.get('engagement_level', 'medium'),
            agent_coordination_mode=session_data.get('coordination_mode', 'collaborative')
        )
        
        # Load existing sessions
        sessions = self.load_json_data(self.sessions_file)
        
        # Add new session
        sessions.append(asdict(session))
        
        # Save updated sessions
        self.save_json_data(self.sessions_file, sessions)
        
        print(f"📊 Saved session summary: {session.session_id}")
    
    def save_student_evaluation(self, student_profile: Any, analytics_data: Dict[str, Any]):
        """Save comprehensive student evaluation"""
        
        evaluation = StudentEvaluation(
            student_id=student_profile.student_id,
            student_name=student_profile.name,
            evaluation_date=datetime.now().isoformat(),
            overall_performance=student_profile.overall_performance,
            topic_performance=dict(student_profile.topic_performance),
            difficulty_performance=dict(student_profile.difficulty_performance),
            learning_velocity=student_profile.learning_velocity,
            engagement_score=student_profile.engagement_score,
            total_sessions=student_profile.session_count,
            total_interactions=student_profile.total_questions_answered,
            correct_responses=student_profile.correct_responses,
            detailed_responses=student_profile.detailed_responses,
            accuracy_rate=analytics_data.get('accuracy_rate', 0.0),
            detailed_response_rate=analytics_data.get('detailed_response_rate', 0.0),
            preferred_topics=student_profile.preferred_topics,
            improvement_areas=student_profile.improvement_areas,
            strength_areas=analytics_data.get('strength_areas', []),
            learning_style=student_profile.learning_style
        )
        
        # Load existing evaluations
        evaluations = self.load_json_data(self.evaluations_file)
        
        # Add new evaluation
        evaluations.append(asdict(evaluation))
        
        # Save updated evaluations
        self.save_json_data(self.evaluations_file, evaluations)
        
        print(f"📋 Saved evaluation for student {student_profile.name} ({student_profile.student_id})")
    
    def generate_student_report(self, student_id: str) -> Dict[str, Any]:
        """Generate comprehensive report for a specific student"""
        
        # Load all data
        interactions = self.load_json_data(self.interactions_file)
        sessions = self.load_json_data(self.sessions_file)
        evaluations = self.load_json_data(self.evaluations_file)
        
        # Filter data for specific student
        student_interactions = [i for i in interactions if i.get('session_id', '').startswith(student_id)]
        student_sessions = [s for s in sessions if s.get('student_id') == student_id]
        student_evaluations = [e for e in evaluations if e.get('student_id') == student_id]
        
        if not student_interactions:
            return {"error": f"No data found for student {student_id}"}
        
        # Calculate analytics
        total_interactions = len(student_interactions)
        total_sessions = len(student_sessions)
        
        if total_interactions > 0:
            avg_reward = statistics.mean([i.get('reward_score', 0) for i in student_interactions])
            total_reward = sum([i.get('reward_score', 0) for i in student_interactions])
            
            # Topic analysis
            topic_counts = {}
            for interaction in student_interactions:
                topic = interaction.get('topic', 'unknown')
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
            
            # Difficulty analysis
            difficulty_rewards = {'easy': [], 'medium': [], 'hard': []}
            for interaction in student_interactions:
                diff = interaction.get('difficulty', 'easy')
                reward = interaction.get('reward_score', 0)
                if diff in difficulty_rewards:
                    difficulty_rewards[diff].append(reward)
            
            # Learning progression
            rewards_over_time = [i.get('reward_score', 0) for i in student_interactions]
            improvement_trend = "improving" if len(rewards_over_time) > 5 and \
                              statistics.mean(rewards_over_time[-5:]) > statistics.mean(rewards_over_time[:5]) else "stable"
        else:
            avg_reward = total_reward = 0
            topic_counts = {}
            difficulty_rewards = {}
            improvement_trend = "insufficient_data"
        
        # Generate report
        report = {
            "student_id": student_id,
            "report_generated": datetime.now().isoformat(),
            "summary": {
                "total_interactions": total_interactions,
                "total_sessions": total_sessions,
                "average_reward": round(avg_reward, 3),
                "total_reward": round(total_reward, 2),
                "improvement_trend": improvement_trend
            },
            "topic_analysis": topic_counts,
            "difficulty_performance": {
                diff: {
                    "attempts": len(rewards),
                    "average_score": round(statistics.mean(rewards), 3) if rewards else 0,
                    "best_score": round(max(rewards), 3) if rewards else 0
                } for diff, rewards in difficulty_rewards.items()
            },
            "recent_interactions": student_interactions[-10:],  # Last 10 interactions
            "latest_evaluation": student_evaluations[-1] if student_evaluations else None,
            "learning_progression": rewards_over_time[-20:] if len(rewards_over_time) > 20 else rewards_over_time
        }
        
        return report
    
    def export_to_csv(self, student_id: Optional[str] = None):
        """Export data to CSV files for analysis"""
        
        interactions = self.load_json_data(self.interactions_file)
        sessions = self.load_json_data(self.sessions_file)
        evaluations = self.load_json_data(self.evaluations_file)
        
        # Filter by student if specified
        if student_id:
            interactions = [i for i in interactions if i.get('session_id', '').startswith(student_id)]
            sessions = [s for s in sessions if s.get('student_id') == student_id]
            evaluations = [e for e in evaluations if e.get('student_id') == student_id]
            suffix = f"_{student_id}"
        else:
            suffix = "_all"
        
        # Export interactions
        if interactions:
            interactions_csv = os.path.join(self.storage_dir, f"interactions{suffix}.csv")
            with open(interactions_csv, 'w', newline='', encoding='utf-8') as f:
                if interactions:
                    writer = csv.DictWriter(f, fieldnames=interactions[0].keys())
                    writer.writeheader()
                    writer.writerows(interactions)
            print(f"📊 Exported interactions to {interactions_csv}")
        
        # Export sessions
        if sessions:
            sessions_csv = os.path.join(self.storage_dir, f"sessions{suffix}.csv")
            with open(sessions_csv, 'w', newline='', encoding='utf-8') as f:
                if sessions:
                    writer = csv.DictWriter(f, fieldnames=sessions[0].keys())
                    writer.writeheader()
                    writer.writerows(sessions)
            print(f"📊 Exported sessions to {sessions_csv}")
        
        # Export evaluations
        if evaluations:
            evaluations_csv = os.path.join(self.storage_dir, f"evaluations{suffix}.csv")
            with open(evaluations_csv, 'w', newline='', encoding='utf-8') as f:
                if evaluations:
                    # Flatten nested dictionaries for CSV
                    flattened_evaluations = []
                    for eval_data in evaluations:
                        flat_eval = eval_data.copy()
                        # Convert dict fields to strings
                        for key in ['topic_performance', 'difficulty_performance']:
                            if key in flat_eval and isinstance(flat_eval[key], dict):
                                flat_eval[key] = json.dumps(flat_eval[key])
                        # Convert list fields to strings
                        for key in ['preferred_topics', 'improvement_areas', 'strength_areas']:
                            if key in flat_eval and isinstance(flat_eval[key], list):
                                flat_eval[key] = ', '.join(flat_eval[key])
                        flattened_evaluations.append(flat_eval)
                    
                    writer = csv.DictWriter(f, fieldnames=flattened_evaluations[0].keys())
                    writer.writeheader()
                    writer.writerows(flattened_evaluations)
            print(f"📊 Exported evaluations to {evaluations_csv}")
    
    def load_json_data(self, file_path: str) -> List[Dict]:
        """Load data from JSON file"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def save_json_data(self, file_path: str, data: List[Dict]):
        """Save data to JSON file"""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Generate overall analytics across all students"""
        
        interactions = self.load_json_data(self.interactions_file)
        sessions = self.load_json_data(self.sessions_file)
        evaluations = self.load_json_data(self.evaluations_file)
        
        if not interactions:
            return {"message": "No interaction data available"}
        
        # Overall statistics
        total_interactions = len(interactions)
        total_sessions = len(sessions)
        unique_students = len(set(s.get('student_id', '') for s in sessions))
        
        # Performance analytics
        all_rewards = [i.get('reward_score', 0) for i in interactions]
        avg_performance = statistics.mean(all_rewards) if all_rewards else 0
        
        # Topic popularity
        topic_counts = {}
        for interaction in interactions:
            topic = interaction.get('topic', 'unknown')
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Generate summary
        summary = {
            "generated_at": datetime.now().isoformat(),
            "overall_stats": {
                "total_interactions": total_interactions,
                "total_sessions": total_sessions,
                "unique_students": unique_students,
                "average_performance": round(avg_performance, 3)
            },
            "topic_popularity": dict(sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)),
            "recent_activity": {
                "last_24_hours": len([i for i in interactions 
                                    if datetime.fromisoformat(i.get('timestamp', '2020-01-01')) > 
                                    datetime.now() - timedelta(days=1)]),
                "last_week": len([i for i in interactions 
                                if datetime.fromisoformat(i.get('timestamp', '2020-01-01')) > 
                                datetime.now() - timedelta(days=7)])
            }
        }
        
        # Save analytics summary
        self.save_json_data(self.analytics_file, [summary])
        
        return summary
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Remove data older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Clean interactions
        interactions = self.load_json_data(self.interactions_file)
        recent_interactions = [
            i for i in interactions 
            if datetime.fromisoformat(i.get('timestamp', '2020-01-01')) > cutoff_date
        ]
        
        if len(recent_interactions) < len(interactions):
            self.save_json_data(self.interactions_file, recent_interactions)
            removed_count = len(interactions) - len(recent_interactions)
            print(f"🗑️ Cleaned up {removed_count} old interaction records")
        
        # Clean sessions
        sessions = self.load_json_data(self.sessions_file)
        recent_sessions = [
            s for s in sessions 
            if datetime.fromisoformat(s.get('end_time', '2020-01-01')) > cutoff_date
        ]
        
        if len(recent_sessions) < len(sessions):
            self.save_json_data(self.sessions_file, recent_sessions)
            removed_count = len(sessions) - len(recent_sessions)
            print(f"🗑️ Cleaned up {removed_count} old session records")
    
    def record_interaction(self, interaction_data: Dict[str, Any]):
        """Record a single interaction (FastAPI web interface compatibility)"""
        try:
            # Load existing interactions
            interactions = self.load_json_data(self.interactions_file)
            
            # Add new interaction
            interactions.append(interaction_data)
            
            # Save updated interactions
            self.save_json_data(self.interactions_file, interactions)
            
            print(f"💾 Recorded interaction for session {interaction_data.get('session_id', 'unknown')}")
        except Exception as e:
            print(f"⚠️ Error recording interaction: {e}")
    
    def get_all_interactions(self) -> List[Dict[str, Any]]:
        """Get all recorded interactions (FastAPI web interface compatibility)"""
        return self.load_json_data(self.interactions_file)
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get all recorded sessions (FastAPI web interface compatibility)"""
        return self.load_json_data(self.sessions_file)

# Example usage and testing
if __name__ == "__main__":
    # Initialize results manager
    results_manager = StudentResultsManager()
    
    print("📊 Student Results Manager initialized")
    print(f"📁 Storage directory: {results_manager.storage_dir}")
    print("✅ Ready to track student learning progress!")
