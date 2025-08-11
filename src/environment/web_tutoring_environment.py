"""
Web-compatible tutoring environment that doesn't block on user input.
Supports async question/answer flow for web interfaces.
"""

import json
import time
from typing import Dict, Tuple, Optional, Any, List
import numpy as np
from src.environment.human_tutoring_environment import HumanTutoringEnvironment, HumanStudent
from src.environment.tutoring_environment import Question, QuestionType, ActionType
import logging

logger = logging.getLogger(__name__)

class WebStudent(HumanStudent):
    """Modified student class for web-based interaction."""
    
    def __init__(self, name: str = "WebStudent"):
        super().__init__(name)
        self.pending_question = None
        self.pending_answer = None
        # Initialize knowledge tracking
        self.knowledge_state = {}
        
    def set_pending_question(self, question: Question):
        """Store a question that's waiting for an answer."""
        self.pending_question = question
        self.pending_answer = None
        
    def submit_answer(self, answer: str) -> Tuple[bool, float]:
        """Submit an answer to the pending question."""
        if self.pending_question is None:
            raise ValueError("No pending question to answer")
            
        start_time = time.time()
        
        # Check if answer is correct
        is_correct = self._check_answer(answer, self.pending_question)
        
        # Calculate confidence (simulated since we can't measure response time accurately in web)
        response_time = 5.0  # Default reasonable response time
        confidence = self._calculate_confidence(response_time, is_correct, False)
        
        # Update student state
        self._update_after_question(self.pending_question, is_correct, response_time)
        
        # Store the answer
        self.pending_answer = answer
        
        # Clear pending question
        question = self.pending_question
        self.pending_question = None
        
        logger.info(f"Student answered question: {is_correct} (confidence: {confidence:.2f})")
        
        return is_correct, confidence
        
    def get_question_data(self) -> Optional[Dict[str, Any]]:
        """Get the current pending question data for web display."""
        if self.pending_question is None:
            return None
            
        question = self.pending_question
        
        # Generate question text and options
        question_data = {
            'id': f"q_{int(time.time())}",
            'topic': question.topic.title(),
            'difficulty': question.difficulty.name,
            'type': question.question_type.name,
            'question_number': self.questions_answered + 1
        }
        
        if question.question_type == QuestionType.MULTIPLE_CHOICE:
            question_data.update(self._get_multiple_choice_data(question))
        elif question.question_type == QuestionType.TRUE_FALSE:
            question_data.update(self._get_true_false_data(question))
        else:
            question_data.update(self._get_open_question_data(question))
            
        return question_data
        
    def _get_multiple_choice_data(self, question: Question) -> Dict[str, Any]:
        """Get multiple choice question data."""
        question_text = self._generate_realistic_question(question)
        
        # Generate options (correct answer + distractors)
        options = [question.correct_answer]
        if hasattr(question, 'distractors') and question.distractors:
            options.extend(question.distractors[:3])
        else:
            # Generate some default distractors based on topic
            options.extend(self._generate_distractors(question)[:3])
            
        # Shuffle options
        import random
        random.shuffle(options)
        
        return {
            'question_text': question_text,
            'options': options,
            'input_type': 'radio'
        }
        
    def _get_true_false_data(self, question: Question) -> Dict[str, Any]:
        """Get true/false question data."""
        question_text = self._generate_realistic_question(question)
        
        return {
            'question_text': question_text,
            'options': ['True', 'False'],
            'input_type': 'radio'
        }
        
    def _get_open_question_data(self, question: Question) -> Dict[str, Any]:
        """Get open-ended question data."""
        question_text = self._generate_realistic_question(question)
        
        return {
            'question_text': question_text,
            'input_type': 'text',
            'placeholder': 'Enter your answer...'
        }
        
    def _generate_distractors(self, question: Question) -> List[str]:
        """Generate distractor options for multiple choice questions."""
        topic = question.topic
        correct = question.correct_answer
        
        if topic == 'mathematics':
            # Generate mathematical distractors
            try:
                correct_num = float(correct)
                distractors = [
                    str(correct_num + 1),
                    str(correct_num - 1),
                    str(correct_num * 2),
                    str(int(correct_num + 5)) if correct_num > 5 else str(int(correct_num + 2))
                ]
            except:
                distractors = ["42", "0", "1", "100"]
        elif topic == 'programming':
            distractors = ["TypeError", "ValueError", "SyntaxError", "NameError"]
        else:
            distractors = ["Option A", "Option B", "Option C", "Option D"]
            
        return distractors[:3]  # Return only 3 distractors

class WebTutoringEnvironment(HumanTutoringEnvironment):
    """Web-compatible tutoring environment for non-blocking operation."""
    
    def __init__(self, student_name: str = "WebStudent", **kwargs):
        # Initialize with web student
        super().__init__(student_name, **kwargs)
        self.web_student = None
        
    def reset(self, **kwargs) -> np.ndarray:
        """Reset environment with web student."""
        # Create web student instead of human student
        self.web_student = WebStudent(self.student_name)
        self.human_student = self.web_student  # Keep compatibility
        
        # Initialize session
        self.session_questions = []
        self.episode_step = 0
        
        logger.info(f"Starting new web tutoring session for {self.student_name}")
        return self._get_state()
        
    def get_current_question(self) -> Optional[Dict[str, Any]]:
        """Get the current pending question for web display."""
        if self.web_student is None:
            return None
        return self.web_student.get_question_data()
        
    def submit_answer(self, answer: str) -> Tuple[bool, float, Dict[str, Any]]:
        """Submit an answer to the current question."""
        if self.web_student is None or self.web_student.pending_question is None:
            raise ValueError("No pending question")
            
        is_correct, confidence = self.web_student.submit_answer(answer)
        
        # Get feedback info
        feedback = self._get_feedback_info(is_correct, self.web_student.pending_answer or answer)
        
        return is_correct, confidence, feedback
        
    def _ask_human_question(self) -> Tuple[float, Dict]:
        """Ask a question to the web student (non-blocking)."""
        question = self._select_question()
        self.current_question = question
        
        # Set pending question (doesn't block)
        self.web_student.set_pending_question(question)
        
        # Return immediately with question ready status
        reward = 0.0  # No reward yet, waiting for answer
        
        return reward, {
            'question_asked': True,
            'question_ready': True,
            'waiting_for_answer': True,
            'question_topic': question.topic,
            'question_difficulty': question.difficulty.name
        }
        
    def process_answer(self, answer: str) -> Tuple[float, Dict]:
        """Process the submitted answer and calculate reward."""
        if self.web_student is None or self.web_student.pending_question is None:
            return -5.0, {'error': 'No pending question'}
            
        is_correct, confidence, feedback = self.submit_answer(answer)
        
        # Calculate reward based on educational effectiveness
        reward_weights = self.config['reward_weights']
        if is_correct:
            reward = reward_weights['correct_answer']
            reward += confidence * 5  # Confidence bonus
        else:
            reward = reward_weights['incorrect_answer']
            
        # Engagement bonus
        engagement = self.web_student.get_engagement_level()
        reward += engagement * reward_weights['engagement_bonus']
        
        return reward, {
            'question_answered': True,
            'correct': is_correct,
            'confidence': confidence,
            'engagement': engagement,
            'question_topic': self.current_question.topic if self.current_question else 'unknown',
            'question_difficulty': self.current_question.difficulty.name if self.current_question else 'unknown'
        }
        
    def _get_feedback_info(self, is_correct: bool, answer: str) -> Dict[str, Any]:
        """Generate feedback information for the web interface."""
        feedback = {
            'correct': is_correct,
            'user_answer': answer,
            'correct_answer': self.current_question.correct_answer if self.current_question else 'Unknown'
        }
        
        if is_correct:
            feedback['message'] = "✅ Correct! Well done!"
            feedback['type'] = 'success'
        else:
            feedback['message'] = f"❌ Incorrect. The correct answer is: {feedback['correct_answer']}"
            feedback['type'] = 'error'
            
        # Add explanation if available
        if self.current_question and hasattr(self.current_question, 'explanation'):
            feedback['explanation'] = self.current_question.explanation
            
        return feedback
        
    def has_pending_question(self) -> bool:
        """Check if there's a question waiting for an answer."""
        return (self.web_student is not None and 
                self.web_student.pending_question is not None)
                
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics."""
        if self.web_student is None:
            return {}
            
        return {
            'questions_answered': self.web_student.questions_answered,
            'correct_answers': self.web_student.correct_answers,
            'accuracy': (self.web_student.correct_answers / max(1, self.web_student.questions_answered)),
            'engagement': self.web_student.get_engagement_level(),
            'motivation': self.web_student.current_motivation,
            'episode_step': self.episode_step,
            'session_summary': self.web_student.get_session_summary()
        }
