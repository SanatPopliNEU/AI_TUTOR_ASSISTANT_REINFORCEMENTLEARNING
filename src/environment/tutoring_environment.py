"""
Tutoring Environment for Reinforcement Learning.

This module implements a realistic tutoring environment that simulates
student interactions, learning progression, and feedback mechanisms.
"""

import numpy as np
import random
import yaml
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class DifficultyLevel(Enum):
    """Difficulty levels for questions."""
    EASY = 1
    MEDIUM = 2
    HARD = 3


class QuestionType(Enum):
    """Types of questions."""
    MULTIPLE_CHOICE = 1
    TRUE_FALSE = 2
    SHORT_ANSWER = 3
    PROBLEM_SOLVING = 4


class ActionType(Enum):
    """Types of tutoring actions."""
    ASK_QUESTION = 0
    PROVIDE_HINT = 1
    EXPLAIN_CONCEPT = 2
    REVIEW_PREVIOUS = 3
    INCREASE_DIFFICULTY = 4
    DECREASE_DIFFICULTY = 5
    ENCOURAGE = 6
    TAKE_BREAK = 7


@dataclass
class Question:
    """Represents a tutorial question."""
    id: int
    content: str
    difficulty: DifficultyLevel
    question_type: QuestionType
    topic: str
    correct_answer: str
    hints: List[str]
    explanation: str


@dataclass
class StudentProfile:
    """Represents a student's learning profile."""
    learning_rate: float  # How quickly student learns (0-1)
    attention_span: float  # How long student can focus (0-1)
    difficulty_preference: float  # Preferred difficulty (0-1)
    motivation: float  # Current motivation level (0-1)
    knowledge_level: Dict[str, float]  # Knowledge in different topics (0-1)
    learning_style: str  # Visual, auditory, kinesthetic
    mistake_tendency: float  # Likelihood of making mistakes (0-1)


class StudentSimulator:
    """Simulates student behavior and responses."""
    
    def __init__(self, profile: StudentProfile):
        """
        Initialize student simulator.
        
        Args:
            profile (StudentProfile): Student learning profile
        """
        self.profile = profile
        self.current_motivation = profile.motivation
        self.fatigue = 0.0
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.session_length = 0
        self.total_questions_answered = 0
        
    def answer_question(self, question: Question, received_hint: bool = False) -> Tuple[bool, float]:
        """
        Simulate student answering a question.
        
        Args:
            question (Question): The question being asked
            received_hint (bool): Whether student received a hint
            
        Returns:
            Tuple[bool, float]: (is_correct, confidence_level)
        """
        # Get student's knowledge in this topic
        topic_knowledge = self.profile.knowledge_level.get(question.topic, 0.3)
        
        # Calculate difficulty factor
        difficulty_factor = self._calculate_difficulty_factor(question.difficulty)
        
        # Calculate probability of correct answer
        base_prob = topic_knowledge * (1 - self.profile.mistake_tendency)
        
        # Adjust for hint
        if received_hint:
            base_prob *= 1.3  # Hints improve success rate
        
        # Adjust for fatigue and motivation
        motivation_factor = (self.current_motivation + 1) / 2  # Scale to 0.5-1
        fatigue_factor = max(0.3, 1 - self.fatigue)
        
        final_prob = base_prob * difficulty_factor * motivation_factor * fatigue_factor
        final_prob = np.clip(final_prob, 0.05, 0.95)  # Keep in reasonable bounds
        
        # Determine if answer is correct
        is_correct = random.random() < final_prob
        
        # Calculate confidence (higher for easier questions, correct answers)
        confidence = final_prob * (1.2 if is_correct else 0.7)
        confidence = np.clip(confidence, 0.1, 1.0)
        
        # Update student state
        self._update_after_question(question, is_correct)
        
        return is_correct, confidence
    
    def _calculate_difficulty_factor(self, difficulty: DifficultyLevel) -> float:
        """Calculate how difficulty affects success probability."""
        if difficulty == DifficultyLevel.EASY:
            return 1.2
        elif difficulty == DifficultyLevel.MEDIUM:
            return 1.0
        else:  # HARD
            return 0.7
    
    def _update_after_question(self, question: Question, is_correct: bool):
        """Update student state after answering a question."""
        self.total_questions_answered += 1
        self.session_length += 1
        
        # Update consecutive counters
        if is_correct:
            self.consecutive_successes += 1
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            self.consecutive_successes = 0
        
        # Update motivation based on performance
        if is_correct:
            self.current_motivation = min(1.0, self.current_motivation + 0.05)
        else:
            self.current_motivation = max(0.1, self.current_motivation - 0.1)
        
        # Increase fatigue
        self.fatigue = min(1.0, self.fatigue + 0.02)
        
        # Update knowledge (learning occurs)
        if is_correct:
            topic = question.topic
            current_knowledge = self.profile.knowledge_level.get(topic, 0.3)
            learning_increment = self.profile.learning_rate * 0.05
            self.profile.knowledge_level[topic] = min(1.0, current_knowledge + learning_increment)
    
    def get_engagement_level(self) -> float:
        """Calculate current engagement level."""
        # Based on motivation, fatigue, and recent performance
        motivation_component = self.current_motivation
        fatigue_component = 1 - self.fatigue
        
        # Recent performance component
        if self.consecutive_failures > 3:
            performance_component = 0.3
        elif self.consecutive_successes > 2:
            performance_component = 1.0
        else:
            performance_component = 0.7
        
        engagement = (motivation_component + fatigue_component + performance_component) / 3
        return np.clip(engagement, 0.1, 1.0)
    
    def rest(self):
        """Student takes a break."""
        self.fatigue = max(0.0, self.fatigue - 0.3)
        self.current_motivation = min(1.0, self.current_motivation + 0.1)


class TutoringEnvironment:
    """Main tutoring environment for RL agents."""
    
    def __init__(self, config_path: str = None, student_profile: str = "beginner"):
        """
        Initialize tutoring environment.
        
        Args:
            config_path (str): Path to configuration file
            student_profile (str): Type of student profile to use
        """
        self.config = self._load_config(config_path)
        self.student_profile_type = student_profile
        
        # Initialize question bank and student
        self.questions = self._create_question_bank()
        self.topics = list(set(q.topic for q in self.questions))
        
        # State and action spaces
        self.state_size = 15  # Current state representation size
        self.action_size = len(ActionType)
        
        # Environment state
        self.current_student = None
        self.current_question = None
        self.session_history = []
        self.episode_step = 0
        self.max_episode_steps = 50
        
        logger.info(f"Tutoring environment initialized with {len(self.questions)} questions")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'max_episode_steps': 50,
                'question_bank_size': 100,
                'topics': ['mathematics', 'science', 'programming', 'language'],
                'reward_weights': {
                    'correct_answer': 10.0,
                    'incorrect_answer': -5.0,
                    'engagement_bonus': 5.0,
                    'efficiency_bonus': 3.0,
                    'knowledge_growth': 15.0
                }
            }
    
    def _create_question_bank(self) -> List[Question]:
        """Create a bank of tutorial questions."""
        questions = []
        topics = self.config.get('topics', ['mathematics', 'science', 'programming', 'language'])
        
        question_id = 0
        for topic in topics:
            for difficulty in DifficultyLevel:
                for qtype in QuestionType:
                    for i in range(5):  # 5 questions per combination
                        question = Question(
                            id=question_id,
                            content=f"{topic.title()} {difficulty.name.lower()} {qtype.name.lower()} question {i+1}",
                            difficulty=difficulty,
                            question_type=qtype,
                            topic=topic,
                            correct_answer=f"Answer {question_id}",
                            hints=[f"Hint 1 for Q{question_id}", f"Hint 2 for Q{question_id}"],
                            explanation=f"Explanation for question {question_id} in {topic}"
                        )
                        questions.append(question)
                        question_id += 1
        
        return questions
    
    def _create_student_profile(self, profile_type: str) -> StudentProfile:
        """Create student profile based on type."""
        if profile_type == "beginner":
            return StudentProfile(
                learning_rate=0.7,
                attention_span=0.6,
                difficulty_preference=0.3,
                motivation=0.8,
                knowledge_level={topic: 0.2 for topic in self.topics},
                learning_style="visual",
                mistake_tendency=0.4
            )
        elif profile_type == "intermediate":
            return StudentProfile(
                learning_rate=0.5,
                attention_span=0.8,
                difficulty_preference=0.6,
                motivation=0.7,
                knowledge_level={topic: 0.5 for topic in self.topics},
                learning_style="auditory",
                mistake_tendency=0.2
            )
        else:  # advanced
            return StudentProfile(
                learning_rate=0.3,
                attention_span=0.9,
                difficulty_preference=0.8,
                motivation=0.6,
                knowledge_level={topic: 0.8 for topic in self.topics},
                learning_style="kinesthetic",
                mistake_tendency=0.1
            )
    
    def reset(self) -> np.ndarray:
        """
        Reset environment for new episode.
        
        Returns:
            np.ndarray: Initial state
        """
        # Create new student
        self.current_student = StudentSimulator(
            self._create_student_profile(self.student_profile_type)
        )
        
        # Reset episode variables
        self.current_question = None
        self.session_history = []
        self.episode_step = 0
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action in environment.
        
        Args:
            action (int): Action to take
            
        Returns:
            Tuple[np.ndarray, float, bool, Dict]: (next_state, reward, done, info)
        """
        self.episode_step += 1
        action_type = ActionType(action)
        
        # Execute action and get reward
        reward, info = self._execute_action(action_type)
        
        # Check if episode is done
        done = (self.episode_step >= self.max_episode_steps or 
                self.current_student.get_engagement_level() < 0.2)
        
        # Get next state
        next_state = self._get_state()
        
        # Update session history
        self.session_history.append({
            'step': self.episode_step,
            'action': action_type.name,
            'reward': reward,
            'engagement': self.current_student.get_engagement_level(),
            'motivation': self.current_student.current_motivation
        })
        
        return next_state, reward, done, info
    
    def _execute_action(self, action: ActionType) -> Tuple[float, Dict]:
        """Execute specific action and return reward."""
        reward = 0.0
        info = {'action_executed': action.name}
        
        if action == ActionType.ASK_QUESTION:
            reward, question_info = self._ask_question()
            info.update(question_info)
        
        elif action == ActionType.PROVIDE_HINT:
            reward, hint_info = self._provide_hint()
            info.update(hint_info)
        
        elif action == ActionType.EXPLAIN_CONCEPT:
            reward = self._explain_concept()
            info['concept_explained'] = True
        
        elif action == ActionType.REVIEW_PREVIOUS:
            reward = self._review_previous()
            info['review_conducted'] = True
        
        elif action == ActionType.INCREASE_DIFFICULTY:
            reward = self._adjust_difficulty(increase=True)
            info['difficulty_increased'] = True
        
        elif action == ActionType.DECREASE_DIFFICULTY:
            reward = self._adjust_difficulty(increase=False)
            info['difficulty_decreased'] = True
        
        elif action == ActionType.ENCOURAGE:
            reward = self._encourage_student()
            info['encouragement_given'] = True
        
        elif action == ActionType.TAKE_BREAK:
            reward = self._take_break()
            info['break_taken'] = True
        
        return reward, info
    
    def _ask_question(self) -> Tuple[float, Dict]:
        """Ask a question to the student."""
        # Select appropriate question
        question = self._select_question()
        self.current_question = question
        
        # Student answers
        is_correct, confidence = self.current_student.answer_question(question)
        
        # Calculate reward
        reward_weights = self.config['reward_weights']
        if is_correct:
            reward = reward_weights['correct_answer']
            # Bonus for high confidence correct answers
            reward += confidence * 5
        else:
            reward = reward_weights['incorrect_answer']
        
        # Engagement bonus
        engagement = self.current_student.get_engagement_level()
        reward += engagement * reward_weights['engagement_bonus']
        
        return reward, {
            'question_asked': True,
            'correct': is_correct,
            'confidence': confidence,
            'engagement': engagement,
            'question_topic': question.topic,
            'question_difficulty': question.difficulty.name
        }
    
    def _provide_hint(self) -> Tuple[float, Dict]:
        """Provide hint for current question."""
        if self.current_question is None:
            return -2.0, {'error': 'No current question to hint'}
        
        # Hint effectiveness depends on student engagement
        engagement = self.current_student.get_engagement_level()
        reward = 2.0 * engagement
        
        return reward, {'hint_provided': True, 'effectiveness': engagement}
    
    def _explain_concept(self) -> float:
        """Explain a concept to the student."""
        # Explanation helps with motivation and knowledge
        self.current_student.current_motivation = min(1.0, 
            self.current_student.current_motivation + 0.1)
        
        # Knowledge improvement
        if self.current_question:
            topic = self.current_question.topic
            current_knowledge = self.current_student.profile.knowledge_level.get(topic, 0.3)
            self.current_student.profile.knowledge_level[topic] = min(1.0, 
                current_knowledge + 0.03)
        
        return 3.0
    
    def _review_previous(self) -> float:
        """Review previous material."""
        # Helps with retention and confidence
        if len(self.session_history) > 0:
            self.current_student.current_motivation = min(1.0,
                self.current_student.current_motivation + 0.05)
            return 2.0
        return 0.0
    
    def _adjust_difficulty(self, increase: bool) -> float:
        """Adjust question difficulty."""
        engagement = self.current_student.get_engagement_level()
        
        if increase and engagement > 0.7:
            return 3.0  # Good to increase when engaged
        elif not increase and engagement < 0.4:
            return 3.0  # Good to decrease when struggling
        else:
            return -1.0  # Poor timing for adjustment
    
    def _encourage_student(self) -> float:
        """Provide encouragement."""
        # Encouragement helps with motivation
        self.current_student.current_motivation = min(1.0,
            self.current_student.current_motivation + 0.15)
        return 2.0
    
    def _take_break(self) -> float:
        """Allow student to take a break."""
        fatigue_before = self.current_student.fatigue
        self.current_student.rest()
        
        # Reward based on how much the break was needed
        return (fatigue_before * 5.0)
    
    def _select_question(self) -> Question:
        """Select appropriate question based on student state."""
        engagement = self.current_student.get_engagement_level()
        
        # Filter questions by appropriateness
        suitable_questions = []
        
        for question in self.questions:
            topic_knowledge = self.current_student.profile.knowledge_level.get(
                question.topic, 0.3)
            
            # Select based on knowledge level and engagement
            if engagement > 0.7:
                # High engagement - can handle challenging questions
                if (question.difficulty == DifficultyLevel.MEDIUM and topic_knowledge < 0.7) or \
                   (question.difficulty == DifficultyLevel.HARD and topic_knowledge > 0.5):
                    suitable_questions.append(question)
            else:
                # Low engagement - stick to easier questions
                if question.difficulty == DifficultyLevel.EASY or \
                   (question.difficulty == DifficultyLevel.MEDIUM and topic_knowledge > 0.6):
                    suitable_questions.append(question)
        
        if not suitable_questions:
            suitable_questions = [q for q in self.questions 
                                if q.difficulty == DifficultyLevel.EASY]
        
        return random.choice(suitable_questions)
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state representation.
        
        Returns:
            np.ndarray: Current state vector
        """
        if self.current_student is None:
            return np.zeros(self.state_size)
        
        # State components
        state = []
        
        # Student profile features (5 features)
        state.extend([
            self.current_student.profile.learning_rate,
            self.current_student.profile.attention_span,
            self.current_student.profile.difficulty_preference,
            self.current_student.current_motivation,
            self.current_student.fatigue
        ])
        
        # Performance features (4 features)
        state.extend([
            self.current_student.consecutive_successes / 10.0,  # Normalized
            self.current_student.consecutive_failures / 10.0,   # Normalized
            self.current_student.get_engagement_level(),
            self.current_student.session_length / self.max_episode_steps
        ])
        
        # Knowledge levels (4 features - average per topic)
        avg_knowledge = []
        for topic in self.topics[:4]:  # Take first 4 topics
            knowledge = self.current_student.profile.knowledge_level.get(topic, 0.3)
            avg_knowledge.append(knowledge)
        state.extend(avg_knowledge)
        
        # Current question context (2 features)
        if self.current_question:
            state.extend([
                self.current_question.difficulty.value / 3.0,  # Normalized
                self.current_question.question_type.value / 4.0  # Normalized
            ])
        else:
            state.extend([0.0, 0.0])
        
        return np.array(state, dtype=np.float32)
    
    def get_student_metrics(self) -> Dict:
        """Get current student metrics."""
        if self.current_student is None:
            return {}
        
        return {
            'motivation': self.current_student.current_motivation,
            'fatigue': self.current_student.fatigue,
            'engagement': self.current_student.get_engagement_level(),
            'consecutive_successes': self.current_student.consecutive_successes,
            'consecutive_failures': self.current_student.consecutive_failures,
            'total_questions': self.current_student.total_questions_answered,
            'knowledge_levels': dict(self.current_student.profile.knowledge_level),
            'session_length': self.current_student.session_length
        }
    
    def render(self, mode='human'):
        """Render current state (for debugging)."""
        if mode == 'human':
            metrics = self.get_student_metrics()
            print(f"Step: {self.episode_step}")
            print(f"Engagement: {metrics.get('engagement', 0):.2f}")
            print(f"Motivation: {metrics.get('motivation', 0):.2f}")
            print(f"Fatigue: {metrics.get('fatigue', 0):.2f}")
            if self.current_question:
                print(f"Current Question: {self.current_question.topic} - {self.current_question.difficulty.name}")
            print("-" * 40)
