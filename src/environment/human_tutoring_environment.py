"""
Human Interactive Tutoring Environment - Real Human Student Interaction

This module extends the tutoring environment to support real human students
instead of simulated ones, enabling actual interactive learning sessions.

Author: Sanat Popli
Date: August 2025
"""

import time
import random
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from pathlib import Path

from .tutoring_environment import (
    DifficultyLevel, QuestionType, ActionType, Question, 
    TutoringEnvironment, StudentProfile
)

logger = logging.getLogger(__name__)

class HumanStudent:
    """Real human student interface for interactive learning."""
    
    def __init__(self, name: str = "Student"):
        """
        Initialize human student interface.
        
        Args:
            name (str): Student's name for personalization
        """
        self.name = name
        self.session_start_time = time.time()
        self.questions_answered = 0
        self.correct_answers = 0
        self.hints_used = 0
        self.current_motivation = 0.8  # Start optimistic
        self.fatigue = 0.0
        
        # Track learning progress
        self.knowledge_levels = {
            'mathematics': 0.5,
            'science': 0.5, 
            'programming': 0.5,
            'language': 0.5
        }
        
        # Performance tracking
        self.answer_history = []
        self.response_times = []
        self.recent_questions = []  # Track recently asked questions to avoid repetition
        
        print(f"\n🎓 Welcome to the Adaptive Tutorial System, {self.name}!")
        print("This AI tutor will adapt to your learning style in real-time.")
        print("=" * 60)
    
    def answer_question(self, question: Question, received_hint: bool = False) -> Tuple[bool, float]:
        """
        Present question to human student and get their response.
        
        Args:
            question (Question): The question to ask
            received_hint (bool): Whether student received a hint
            
        Returns:
            Tuple[bool, float]: (is_correct, confidence_level)
        """
        print(f"\n📝 QUESTION #{self.questions_answered + 1}")
        print(f"Topic: {question.topic.title()}")
        print(f"Difficulty: {question.difficulty.name}")
        print(f"Type: {question.question_type.name.replace('_', ' ').title()}")
        print("-" * 50)
        
        # Display the actual question
        if question.question_type == QuestionType.MULTIPLE_CHOICE:
            self._display_multiple_choice(question)
        elif question.question_type == QuestionType.TRUE_FALSE:
            self._display_true_false(question)
        else:
            self._display_open_question(question)
        
        # Get student's answer
        start_time = time.time()
        user_answer = self._get_user_input()
        response_time = time.time() - start_time
        
        # Check if answer is correct
        is_correct = self._check_answer(user_answer, question)
        
        # Calculate confidence based on response time and correctness
        confidence = self._calculate_confidence(response_time, is_correct, received_hint)
        
        # Update student state
        self._update_after_question(question, is_correct, response_time)
        
        # Provide feedback
        self._give_feedback(is_correct, question)
        
        return is_correct, confidence
    
    def _display_multiple_choice(self, question: Question):
        """Display a multiple choice question with realistic options."""
        print(f"Question: {self._generate_realistic_question(question)}")
        print("\nChoose the correct answer:")
        
        options = self._generate_mc_options(question)
        for i, option in enumerate(options, 1):
            print(f"  {i}. {option}")
        print(f"  5. I don't know")
    
    def _display_true_false(self, question: Question):
        """Display a true/false question."""
        print(f"Question: {self._generate_realistic_question(question)}")
        print("\nIs this statement TRUE or FALSE?")
        print("  1. True")
        print("  2. False")
        print("  3. I'm not sure")
    
    def _display_open_question(self, question: Question):
        """Display an open-ended question."""
        print(f"Question: {self._generate_realistic_question(question)}")
        print("\nPlease provide your answer:")
    
    def _generate_realistic_question(self, question: Question) -> str:
        """Generate realistic questions based on topic and difficulty."""
        topic = question.topic
        difficulty = question.difficulty
        
        # Mathematics questions
        if topic == 'mathematics':
            if difficulty == DifficultyLevel.EASY:
                questions = [
                    "What is 15 + 27?",
                    "If you have 8 apples and give away 3, how many do you have left?",
                    "What is 6 × 4?",
                    "Convert 1/2 to a decimal.",
                    "What is 9 - 4?",
                    "Calculate 12 ÷ 3.",
                    "What is 5 × 7?",
                    "If a pizza has 8 slices and you eat 2, how many are left?",
                    "What is 20 + 15?",
                    "Calculate 100 - 25."
                ]
            elif difficulty == DifficultyLevel.MEDIUM:
                questions = [
                    "Solve for x: 2x + 5 = 17",
                    "What is the area of a rectangle with length 8 and width 6?",
                    "If f(x) = 2x + 3, what is f(5)?",
                    "Calculate 25% of 80.",
                    "What is the circumference of a circle with radius 5?",
                    "Solve: 3x - 7 = 14",
                    "Find the volume of a cube with side length 4.",
                    "What is 15% of 200?",
                    "Calculate the perimeter of a square with side 6.",
                    "If y = 3x + 2, what is y when x = 4?"
                ]
            else:  # HARD
                questions = [
                    "Find the derivative of f(x) = x³ + 2x² - 5x + 1",
                    "Solve the system: 2x + 3y = 12, x - y = 1",
                    "What is the integral of sin(x) dx?",
                    "Find the limit as x approaches 0 of (sin(x))/x",
                    "Factor completely: x³ - 8",
                    "Find the slope of the tangent line to y = x² at x = 3",
                    "Solve: log₂(x + 1) = 3",
                    "What is the sum of the geometric series 1 + 1/2 + 1/4 + ...?",
                    "Find the roots of x² - 4x + 3 = 0",
                    "Calculate ∫(2x + 1)dx from 0 to 2"
                ]
        
        # Science questions
        elif topic == 'science':
            if difficulty == DifficultyLevel.EASY:
                questions = [
                    "What gas do plants absorb from the atmosphere?",
                    "How many legs does a spider have?",
                    "What is the chemical symbol for water?",
                    "Which planet is closest to the Sun?",
                    "What do we call animals that eat only plants?",
                    "How many bones are in the human body?",
                    "What is the fastest land animal?",
                    "Which gas makes up most of Earth's atmosphere?",
                    "What is the center of an atom called?",
                    "How many chambers does a human heart have?"
                ]
            elif difficulty == DifficultyLevel.MEDIUM:
                questions = [
                    "What is the powerhouse of the cell?",
                    "Explain Newton's first law of motion.",
                    "What is photosynthesis?",
                    "Name three states of matter.",
                    "What is the pH of pure water?",
                    "Explain the difference between speed and velocity.",
                    "What is the chemical formula for table salt?",
                    "Name the four forces of nature.",
                    "What is the difference between mass and weight?",
                    "Explain what causes the seasons on Earth."
                ]
            else:  # HARD
                questions = [
                    "Explain the process of cellular respiration.",
                    "What is quantum entanglement?",
                    "Describe the structure of DNA.",
                    "How does CRISPR gene editing work?",
                    "Explain the theory of relativity.",
                    "What is the Heisenberg uncertainty principle?",
                    "Describe the process of protein synthesis.",
                    "What is dark matter and dark energy?",
                    "Explain how vaccines work at the molecular level.",
                    "What is the difference between mitosis and meiosis?"
                ]
        
        # Programming questions  
        elif topic == 'programming':
            if difficulty == DifficultyLevel.EASY:
                questions = [
                    "What does 'print()' do in Python?",
                    "How do you create a variable in Python?",
                    "What symbol is used for comments in Python?",
                    "What does HTML stand for?",
                    "How do you start a comment in JavaScript?",
                    "What is a string in programming?",
                    "What does CSS stand for?",
                    "How do you create a list in Python?",
                    "What is the file extension for Python files?",
                    "What does IDE stand for?"
                ]
            elif difficulty == DifficultyLevel.MEDIUM:
                questions = [
                    "Write a Python function that returns the square of a number.",
                    "What is the difference between a list and a tuple?",
                    "How do you handle exceptions in Python?",
                    "Explain what a for loop does.",
                    "What is the difference between == and = in Python?",
                    "How do you import a module in Python?",
                    "What is object-oriented programming?",
                    "Explain the concept of variables and data types.",
                    "What is the difference between a function and a method?",
                    "How do you create a dictionary in Python?"
                ]
            else:  # HARD
                questions = [
                    "Implement a binary search algorithm.",
                    "Explain the concept of recursion with an example.",
                    "What are decorators in Python?",
                    "Describe the Model-View-Controller pattern.",
                    "Explain big O notation and time complexity.",
                    "What is the difference between SQL and NoSQL databases?",
                    "Implement a sorting algorithm of your choice.",
                    "Explain how garbage collection works.",
                    "What are design patterns? Name three examples.",
                    "Describe RESTful API principles."
                ]
        
        # Language questions
        else:  # language
            if difficulty == DifficultyLevel.EASY:
                questions = [
                    "What is the past tense of 'run'?",
                    "Complete: 'I ___ going to the store.' (am/is/are)",
                    "What does 'Hello' mean?",
                    "How do you say 'thank you' in English?",
                    "What is the plural of 'child'?",
                    "Is 'dog' a noun or a verb?",
                    "What comes after 'A, B, C'?",
                    "Complete: 'The cat ___ on the mat.' (sit/sits)",
                    "What is the opposite of 'hot'?",
                    "How many letters are in the English alphabet?"
                ]
            elif difficulty == DifficultyLevel.MEDIUM:
                questions = [
                    "What is the difference between 'your' and 'you're'?",
                    "Identify the noun in: 'The quick brown fox jumps.'",
                    "What is a synonym for 'happy'?",
                    "Correct this sentence: 'Me and John went to the park.'",
                    "What is the comparative form of 'good'?",
                    "Identify the verb in: 'She sings beautifully.'",
                    "What is an antonym for 'difficult'?",
                    "Use 'there', 'their', or 'they're': '___ going to the movies.'",
                    "What is the superlative form of 'bad'?",
                    "Complete: 'If I ___ rich, I would travel.' (was/were)"
                ]
            else:  # HARD
                questions = [
                    "Explain the difference between active and passive voice.",
                    "What is a metaphor? Give an example.",
                    "Parse this sentence: 'The student who studied hard passed the exam.'",
                    "What is the subjunctive mood in English?",
                    "Explain the difference between a simile and a metaphor.",
                    "What is alliteration? Provide an example.",
                    "Identify the literary device: 'The wind whispered secrets.'",
                    "What is irony? Give three types with examples.",
                    "Explain the concept of syntax in language.",
                    "What is a dangling modifier? How do you fix it?"
                ]
        
        # Filter out recently asked questions to avoid immediate repetition
        available_questions = [q for q in questions if q not in self.recent_questions]
        
        # If all questions have been used recently, reset the recent questions list
        if not available_questions:
            self.recent_questions = []
            available_questions = questions
        
        # Select a random question from available options
        selected_question = random.choice(available_questions)
        
        # Add to recent questions (keep only last 5 to allow eventual repetition)
        self.recent_questions.append(selected_question)
        if len(self.recent_questions) > 5:
            self.recent_questions.pop(0)
        
        return selected_question
    
    def _generate_mc_options(self, question: Question) -> List[str]:
        """Generate realistic multiple choice options."""
        topic = question.topic
        
        if topic == 'mathematics':
            return ["42", "35", "29", "51"]
        elif topic == 'science':
            return ["Carbon dioxide", "Oxygen", "Nitrogen", "Hydrogen"]
        elif topic == 'programming':
            return ["Displays output", "Creates variables", "Loops code", "Imports libraries"]
        else:  # language
            return ["ran", "runned", "running", "runs"]
    
    def _get_user_input(self) -> str:
        """Get input from the human user."""
        try:
            print("\n💬 Your answer:", end=" ")
            answer = input().strip()
            return answer
        except KeyboardInterrupt:
            print("\n\n👋 Session ended by user. Goodbye!")
            exit()
        except Exception as e:
            print(f"\nError getting input: {e}")
            return ""
    
    def _check_answer(self, user_answer: str, question: Question) -> bool:
        """
        Check if the user's answer is correct.
        Improved logic to handle specific questions properly.
        """
        if not user_answer:
            return False
        
        user_answer = user_answer.lower().strip()
        
        # Get the actual question text from the stored content
        question_text = question.content.lower() if hasattr(question, 'content') and question.content else self._generate_realistic_question(question).lower()
        
        # Mathematics questions
        if question.topic == 'mathematics':
            if question.difficulty == DifficultyLevel.EASY:
                # Easy math questions and their answers
                if "8 apples" in question_text and "give away 3" in question_text:
                    return '5' in user_answer
                elif "6 × 4" in question_text or "6 * 4" in question_text:
                    return '24' in user_answer
                elif "1/2 to a decimal" in question_text:
                    return '0.5' in user_answer or '0.50' in user_answer
                else:
                    return any(ans in user_answer for ans in ['5', '24', '0.5', '42'])
                    
            elif question.difficulty == DifficultyLevel.MEDIUM:
                # Medium math questions and their answers
                if "2x + 5 = 17" in question_text:
                    return '6' in user_answer or 'x = 6' in user_answer or 'x=6' in user_answer
                elif "area of a rectangle" in question_text and "length 8" in question_text and "width 6" in question_text:
                    return '48' in user_answer
                elif "f(x) = 2x + 3" in question_text and "f(5)" in question_text:
                    return '13' in user_answer
                elif "25% of 80" in question_text:
                    return '20' in user_answer
                else:
                    return any(ans in user_answer for ans in ['6', '48', '13', '20'])
                    
            else:  # HARD
                # Hard math - more flexible checking
                if "derivative" in question_text:
                    return any(term in user_answer for term in ['3x²', '4x', '-5', '3x^2'])
                elif "system" in question_text and "2x + 3y = 12" in question_text:
                    return ('3' in user_answer and '2' in user_answer) or 'x=3' in user_answer
                elif "integral of sin(x)" in question_text:
                    return '-cos' in user_answer or 'cos' in user_answer
                elif "limit" in question_text and "sin(x)/x" in question_text:
                    return '1' in user_answer
                else:
                    return len(user_answer) > 5  # Give credit for attempting hard problems
        
        # Science questions
        elif question.topic == 'science':
            if question.difficulty == DifficultyLevel.EASY:
                if "gas do plants absorb" in question_text:
                    return any(term in user_answer for term in ['carbon dioxide', 'co2', 'co₂'])
                elif "legs does a spider have" in question_text:
                    return '8' in user_answer or 'eight' in user_answer
                elif "chemical symbol for water" in question_text:
                    return 'h2o' in user_answer or 'h₂o' in user_answer
                elif "planet is closest to the sun" in question_text:
                    return 'mercury' in user_answer
                else:
                    return any(term in user_answer for term in ['carbon dioxide', 'co2', '8', 'eight', 'h2o', 'mercury'])
                    
            elif question.difficulty == DifficultyLevel.MEDIUM:
                if "powerhouse of the cell" in question_text:
                    return 'mitochondria' in user_answer or 'mitochondrion' in user_answer
                elif "newton's first law" in question_text:
                    return any(term in user_answer for term in ['inertia', 'motion', 'rest', 'force', 'newton'])
                elif "photosynthesis" in question_text:
                    return any(term in user_answer for term in ['light', 'energy', 'glucose', 'chlorophyll', 'plant', 'sun'])
                elif "three states of matter" in question_text:
                    # Check if user mentioned at least 2 of the 3 states
                    states_mentioned = sum([
                        'solid' in user_answer,
                        'liquid' in user_answer, 
                        'gas' in user_answer,
                        'plasma' in user_answer
                    ])
                    return states_mentioned >= 2
                else:
                    return any(term in user_answer for term in ['mitochondria', 'newton', 'inertia', 'photosynthesis', 'light', 'solid', 'liquid', 'gas'])
                    
            else:  # HARD
                return any(term in user_answer for term in ['cellular', 'respiration', 'quantum', 'dna', 'crispr', 'gene']) or len(user_answer) > 10
        
        # Programming questions
        elif question.topic == 'programming':
            if question.difficulty == DifficultyLevel.EASY:
                return any(term in user_answer for term in ['print', 'console.log', 'cout', 'display', 'output'])
            else:
                return any(term in user_answer for term in ['print', 'display', 'output', 'show', 'html', 'function', 'method']) or len(user_answer) > 5
        
        # Language questions
        else:  # language
            if "past tense" in question_text and "run" in question_text:
                return 'ran' in user_answer
            elif "i __ a student" in question_text:
                return 'am' in user_answer
            else:
                return any(term in user_answer for term in ['ran', 'am', 'hello', 'thank you']) or len(user_answer) > 3
        
        # Default: give some credit for reasonable effort
        return len(user_answer) > 3
    
    def _calculate_confidence(self, response_time: float, is_correct: bool, received_hint: bool) -> float:
        """Calculate confidence based on response time and correctness."""
        base_confidence = 0.8 if is_correct else 0.3
        
        # Adjust for response time (quick answers suggest confidence)
        if response_time < 5:
            time_bonus = 0.2
        elif response_time < 15:
            time_bonus = 0.0
        else:
            time_bonus = -0.2
        
        # Adjust for hints
        hint_penalty = -0.1 if received_hint else 0.0
        
        confidence = base_confidence + time_bonus + hint_penalty
        return np.clip(confidence, 0.1, 1.0)
    
    def _update_after_question(self, question: Question, is_correct: bool, response_time: float):
        """Update student state after answering a question."""
        self.questions_answered += 1
        self.response_times.append(response_time)
        
        if is_correct:
            self.correct_answers += 1
            
            # **IMPROVED KNOWLEDGE BOOST BASED ON DIFFICULTY**
            current_knowledge = self.knowledge_levels.get(question.topic, 0.5)
            
            # Bigger knowledge boost for harder questions answered correctly
            if question.difficulty == DifficultyLevel.EASY:
                knowledge_boost = 0.08  # Modest boost for easy questions
            elif question.difficulty == DifficultyLevel.MEDIUM:
                knowledge_boost = 0.12  # Good boost for medium questions
            else:  # HARD
                knowledge_boost = 0.18  # Significant boost for hard questions
            
            # Scale boost based on current knowledge (faster progress when starting)
            if current_knowledge < 0.3:
                knowledge_boost *= 1.5  # Faster initial learning
            elif current_knowledge > 0.7:
                knowledge_boost *= 0.7  # Slower progress at advanced levels
            
            self.knowledge_levels[question.topic] = min(1.0, current_knowledge + knowledge_boost)
            
            # Motivation boost - bigger for harder questions
            motivation_boost = 0.03 + (0.02 * (question.difficulty.value - 1))  # 0.03, 0.05, 0.07
            self.current_motivation = min(1.0, self.current_motivation + motivation_boost)
            
        else:
            # **SLIGHT KNOWLEDGE DECREASE FOR WRONG ANSWERS**
            current_knowledge = self.knowledge_levels.get(question.topic, 0.5)
            knowledge_decrease = 0.02  # Small decrease to indicate gap
            self.knowledge_levels[question.topic] = max(0.1, current_knowledge - knowledge_decrease)
            
            # Motivation decrease - less harsh for harder questions
            motivation_decrease = 0.08 - (0.02 * (question.difficulty.value - 1))  # 0.08, 0.06, 0.04
            self.current_motivation = max(0.2, self.current_motivation - motivation_decrease)
        
        # Increase fatigue over time
        self.fatigue = min(1.0, self.fatigue + 0.05)
        
        # Record answer
        self.answer_history.append({
            'question_id': question.id,
            'topic': question.topic,
            'difficulty': question.difficulty.name,
            'correct': is_correct,
            'response_time': response_time,
            'timestamp': time.time()
        })
    
    def _give_feedback(self, is_correct: bool, question: Question):
        """Provide immediate feedback to the student."""
        if is_correct:
            feedback = random.choice([
                "✅ Excellent! That's correct!",
                "🎉 Great job! You got it right!",
                "👏 Perfect! Well done!",
                "⭐ Fantastic! Correct answer!"
            ])
            print(f"\n{feedback}")
        else:
            feedback = random.choice([
                "❌ Not quite right, but good effort!",
                "🤔 That's not correct, but keep trying!",
                "💭 Close, but not the right answer.",
                "📚 Incorrect, but you're learning!"
            ])
            print(f"\n{feedback}")
            
            # Offer explanation
            if question.explanation:
                print(f"💡 Explanation: {question.explanation}")
    
    def get_engagement_level(self) -> float:
        """Calculate current engagement level based on performance and state."""
        # Base engagement on motivation and inverse of fatigue
        motivation_component = self.current_motivation
        fatigue_component = 1.0 - self.fatigue
        
        # Performance component
        if self.questions_answered > 0:
            accuracy = self.correct_answers / self.questions_answered
            performance_component = accuracy
        else:
            performance_component = 0.5
        
        # Time component (longer sessions may reduce engagement)
        session_time = time.time() - self.session_start_time
        time_component = max(0.3, 1.0 - (session_time / 1800))  # 30 minutes max
        
        engagement = (motivation_component + fatigue_component + performance_component + time_component) / 4
        return np.clip(engagement, 0.1, 1.0)
    
    def get_session_summary(self) -> Dict:
        """Get summary of the current learning session."""
        if self.questions_answered == 0:
            accuracy = 0.0
            avg_response_time = 0.0
        else:
            accuracy = self.correct_answers / self.questions_answered
            avg_response_time = sum(self.response_times) / len(self.response_times)
        
        session_time = time.time() - self.session_start_time
        
        return {
            'student_name': self.name,
            'questions_answered': self.questions_answered,
            'correct_answers': self.correct_answers,
            'accuracy': accuracy,
            'avg_response_time': avg_response_time,
            'session_duration': session_time,
            'current_engagement': self.get_engagement_level(),
            'current_motivation': self.current_motivation,
            'knowledge_levels': self.knowledge_levels.copy(),
            'hints_used': self.hints_used
        }
    
    def take_break(self):
        """Allow student to take a break."""
        print("\n☕ Taking a short break...")
        print("Press Enter when you're ready to continue...")
        input()
        
        # Reduce fatigue
        self.fatigue = max(0.0, self.fatigue - 0.3)
        self.current_motivation = min(1.0, self.current_motivation + 0.1)
        
        print("✨ Break complete! Let's continue learning.")


class HumanTutoringEnvironment(TutoringEnvironment):
    """Extended tutoring environment for real human interaction."""
    
    def __init__(self, config_path: str = None, student_name: str = "Student"):
        """
        Initialize human tutoring environment.
        
        Args:
            config_path (str): Path to configuration file
            student_name (str): Name of the human student
        """
        super().__init__(config_path, "human")  # Use human profile type
        self.student_name = student_name
        self.human_student = None
        
        print(f"\n🤖 AI Tutor System initialized for {student_name}")
        print("The system will adapt to your learning style in real-time!")
    
    def reset(self) -> np.ndarray:
        """Reset environment for a new session with human student."""
        self.current_student = None
        self.human_student = HumanStudent(self.student_name)
        self.current_question = None
        self.session_history = []
        self.episode_step = 0
        
        logger.info(f"Starting new human tutoring session for {self.student_name}")
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action in environment with human student.
        
        Args:
            action (int): Action to take
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if self.human_student is None:
            raise ValueError("Environment not reset. Call reset() first.")
        
        self.episode_step += 1
        action_type = ActionType(action)
        
        # Execute action and get reward
        reward, action_info = self._execute_action(action_type)
        
        # Check if session should end
        engagement = self.human_student.get_engagement_level()
        done = (self.episode_step >= self.max_episode_steps or 
                engagement < 0.2 or
                self._should_end_session())
        
        # Compile info
        info = {
            'action_type': action_type.name,
            'step': self.episode_step,
            'engagement': engagement,
            'motivation': self.human_student.current_motivation,
            'session_summary': self.human_student.get_session_summary(),
            **action_info
        }
        
        return self._get_state(), reward, done, info
    
    def _execute_action(self, action: ActionType) -> Tuple[float, Dict]:
        """Execute the given action with human interaction."""
        if action == ActionType.ASK_QUESTION:
            return self._ask_human_question()
        elif action == ActionType.PROVIDE_HINT:
            return self._provide_human_hint()
        elif action == ActionType.EXPLAIN_CONCEPT:
            return self._explain_to_human()
        elif action == ActionType.REVIEW_PREVIOUS:
            return self._review_with_human()
        else:
            return 0.0, {'error': 'Unknown action'}
    
    def _ask_human_question(self) -> Tuple[float, Dict]:
        """Ask a question to the human student."""
        question = self._select_question()
        self.current_question = question
        
        # Student answers the question
        is_correct, confidence = self.human_student.answer_question(question)
        
        # Calculate reward based on educational effectiveness
        reward_weights = self.config['reward_weights']
        if is_correct:
            reward = reward_weights['correct_answer']
            reward += confidence * 5  # Confidence bonus
        else:
            reward = reward_weights['incorrect_answer']
        
        # Engagement bonus
        engagement = self.human_student.get_engagement_level()
        reward += engagement * reward_weights['engagement_bonus']
        
        return reward, {
            'question_asked': True,
            'correct': is_correct,
            'confidence': confidence,
            'engagement': engagement,
            'question_topic': question.topic,
            'question_difficulty': question.difficulty.name
        }
    
    def _provide_human_hint(self) -> Tuple[float, Dict]:
        """Provide a hint to the human student."""
        if self.current_question is None:
            print("\n💭 No current question to provide hints for.")
            return -2.0, {'error': 'No current question'}
        
        print(f"\n💡 HINT: {random.choice(self.current_question.hints)}")
        self.human_student.hints_used += 1
        
        engagement = self.human_student.get_engagement_level()
        reward = 2.0 * engagement
        
        return reward, {'hint_provided': True, 'effectiveness': engagement}
    
    def _explain_to_human(self) -> Tuple[float, Dict]:
        """Provide explanation to the human student."""
        if self.current_question:
            print(f"\n📖 EXPLANATION:")
            print(f"   {self.current_question.explanation}")
            topic = self.current_question.topic
        else:
            print(f"\n📖 GENERAL EXPLANATION:")
            print(f"   Let me explain some key concepts to help you understand better.")
            topic = random.choice(list(self.human_student.knowledge_levels.keys()))
        
        # Boost motivation and knowledge slightly
        self.human_student.current_motivation = min(1.0, self.human_student.current_motivation + 0.1)
        current_knowledge = self.human_student.knowledge_levels.get(topic, 0.5)
        self.human_student.knowledge_levels[topic] = min(1.0, current_knowledge + 0.05)
        
        engagement = self.human_student.get_engagement_level()
        return 3.0 * engagement, {'explanation_given': True, 'topic': topic}
    
    def _review_with_human(self) -> Tuple[float, Dict]:
        """Review previous material with human student."""
        print(f"\n📚 REVIEW SESSION:")
        
        if self.human_student.answer_history:
            # Review recent incorrect answers
            recent_incorrect = [ans for ans in self.human_student.answer_history[-5:] if not ans['correct']]
            
            if recent_incorrect:
                print("Let's review some concepts you found challenging:")
                for ans in recent_incorrect[-2:]:  # Review last 2 incorrect
                    print(f"   • {ans['topic'].title()} ({ans['difficulty']} level)")
            else:
                print("Great job! You've been doing well. Let's reinforce what you've learned.")
        else:
            print("Let's review some fundamental concepts before we continue.")
        
        # Reduce fatigue and boost confidence
        self.human_student.fatigue = max(0.0, self.human_student.fatigue - 0.2)
        self.human_student.current_motivation = min(1.0, self.human_student.current_motivation + 0.15)
        
        engagement = self.human_student.get_engagement_level()
        return 4.0 * engagement, {'review_conducted': True}
    
    def _should_end_session(self) -> bool:
        """Check if the session should end based on human factors."""
        # Ask user if they want to continue
        if self.episode_step % 10 == 0 and self.episode_step > 0:
            print(f"\n🤔 You've answered {self.human_student.questions_answered} questions.")
            print("Would you like to continue? (y/n): ", end="")
            try:
                response = input().strip().lower()
                return response in ['n', 'no', 'quit', 'exit']
            except:
                return False
        
        return False
    
    def _select_question(self) -> Question:
        """Select appropriate question based on human student state with intelligent difficulty progression."""
        if self.human_student is None:
            # Fallback to default question selection
            return self.questions[0] if self.questions else None
            
        engagement = self.human_student.get_engagement_level()
        
        # Select topic with variety (not always the lowest)
        topics = ['mathematics', 'science', 'programming', 'language']
        
        # Weight topics inversely by knowledge level (favor weaker topics but add randomness)
        topic_weights = []
        for topic in topics:
            knowledge = self.human_student.knowledge_levels.get(topic, 0.5)
            # Higher weight for lower knowledge, but add randomness
            weight = (1.0 - knowledge) + random.uniform(0.2, 0.8)
            topic_weights.append(weight)
        
        # Select topic based on weighted random choice
        topic = random.choices(topics, weights=topic_weights)[0]
        
        # Add some variety by occasionally picking a random topic
        if random.random() < 0.3:  # 30% chance for random topic
            topic = random.choice(topics)
        
        # **IMPROVED DIFFICULTY SELECTION BASED ON KNOWLEDGE AND PERFORMANCE**
        current_knowledge = self.human_student.knowledge_levels.get(topic, 0.5)
        recent_accuracy = self._calculate_recent_accuracy(topic)
        
        # Start with base difficulty weights
        difficulty_weights = {
            DifficultyLevel.EASY: 0.5,
            DifficultyLevel.MEDIUM: 0.3,
            DifficultyLevel.HARD: 0.2
        }
        
        # **KNOWLEDGE-BASED DIFFICULTY PROGRESSION**
        if current_knowledge >= 0.8:  # High knowledge - prefer harder questions
            difficulty_weights[DifficultyLevel.EASY] = 0.1
            difficulty_weights[DifficultyLevel.MEDIUM] = 0.4
            difficulty_weights[DifficultyLevel.HARD] = 0.5
        elif current_knowledge >= 0.6:  # Medium-high knowledge - mostly medium/hard
            difficulty_weights[DifficultyLevel.EASY] = 0.2
            difficulty_weights[DifficultyLevel.MEDIUM] = 0.5
            difficulty_weights[DifficultyLevel.HARD] = 0.3
        elif current_knowledge >= 0.4:  # Medium knowledge - mixed difficulty
            difficulty_weights[DifficultyLevel.EASY] = 0.3
            difficulty_weights[DifficultyLevel.MEDIUM] = 0.5
            difficulty_weights[DifficultyLevel.HARD] = 0.2
        else:  # Low knowledge - mostly easy with some medium
            difficulty_weights[DifficultyLevel.EASY] = 0.6
            difficulty_weights[DifficultyLevel.MEDIUM] = 0.3
            difficulty_weights[DifficultyLevel.HARD] = 0.1
        
        # **PERFORMANCE-BASED ADJUSTMENTS**
        if recent_accuracy >= 0.8:  # High recent accuracy - challenge more
            difficulty_weights[DifficultyLevel.MEDIUM] += 0.2
            difficulty_weights[DifficultyLevel.HARD] += 0.2
            difficulty_weights[DifficultyLevel.EASY] = max(0.1, difficulty_weights[DifficultyLevel.EASY] - 0.4)
        elif recent_accuracy <= 0.4:  # Low recent accuracy - ease up
            difficulty_weights[DifficultyLevel.EASY] += 0.3
            difficulty_weights[DifficultyLevel.MEDIUM] = max(0.1, difficulty_weights[DifficultyLevel.MEDIUM] - 0.2)
            difficulty_weights[DifficultyLevel.HARD] = max(0.05, difficulty_weights[DifficultyLevel.HARD] - 0.1)
        
        # **ENGAGEMENT-BASED FINE-TUNING** (secondary to knowledge/performance)
        if engagement < 0.4:  # Low engagement - slightly easier
            difficulty_weights[DifficultyLevel.EASY] += 0.1
            difficulty_weights[DifficultyLevel.HARD] = max(0.05, difficulty_weights[DifficultyLevel.HARD] - 0.1)
        elif engagement > 0.8:  # High engagement - can handle slightly more challenge
            difficulty_weights[DifficultyLevel.HARD] += 0.1
            difficulty_weights[DifficultyLevel.EASY] = max(0.1, difficulty_weights[DifficultyLevel.EASY] - 0.1)
        
        # Normalize weights to sum to 1.0
        total_weight = sum(difficulty_weights.values())
        difficulty_weights = {k: v/total_weight for k, v in difficulty_weights.items()}
        
        # Select difficulty based on weights
        difficulty = random.choices(
            list(difficulty_weights.keys()),
            weights=list(difficulty_weights.values())
        )[0]
        
        # Create a Question object with more variety
        # Generate topic-specific, contextual hints
        hints = []
        if topic == 'mathematics':
            if difficulty == DifficultyLevel.EASY:
                hints = [
                    f"Remember basic arithmetic operations: addition (+), subtraction (-), multiplication (×), division (÷)",
                    f"Take your time and work through the {topic} step by step",
                    f"Try using your fingers or drawing the problem if it helps"
                ]
            elif difficulty == DifficultyLevel.MEDIUM:
                hints = [
                    f"Break down the {topic} problem into smaller steps",
                    f"Remember the order of operations: PEMDAS (Parentheses, Exponents, Multiplication/Division, Addition/Subtraction)",
                    f"Consider what mathematical concepts you know about {topic}"
                ]
            else:  # HARD
                hints = [
                    f"Advanced {topic}: Think about underlying mathematical principles",
                    f"Consider using calculus concepts like derivatives or integrals",
                    f"Remember mathematical theorems and formulas you've learned"
                ]
        elif topic == 'science':
            if difficulty == DifficultyLevel.EASY:
                hints = [
                    f"Think about basic {topic} facts you learned in elementary school",
                    f"Consider what you observe in everyday life related to {topic}",
                    f"Remember simple scientific concepts about nature and living things"
                ]
            elif difficulty == DifficultyLevel.MEDIUM:
                hints = [
                    f"Think about {topic} processes and how they work",
                    f"Consider the relationship between different scientific concepts",
                    f"Remember scientific methods and principles you've studied"
                ]
            else:  # HARD
                hints = [
                    f"Advanced {topic}: Think about complex scientific theories and principles",
                    f"Consider molecular or cellular level processes in {topic}",
                    f"Think about advanced scientific concepts and their applications"
                ]
        elif topic == 'programming':
            if difficulty == DifficultyLevel.EASY:
                hints = [
                    f"Think about basic {topic} concepts like variables, functions, and syntax",
                    f"Consider common programming terms and what they mean",
                    f"Remember simple programming operations you've learned"
                ]
            elif difficulty == DifficultyLevel.MEDIUM:
                hints = [
                    f"Think about {topic} logic and how different concepts work together",
                    f"Consider programming structures like loops, conditions, and data types",
                    f"Remember programming best practices and common patterns"
                ]
            else:  # HARD
                hints = [
                    f"Advanced {topic}: Think about algorithms, data structures, and design patterns",
                    f"Consider computational complexity and efficiency in {topic}",
                    f"Think about advanced programming concepts and architectures"
                ]
        else:  # language
            if difficulty == DifficultyLevel.EASY:
                hints = [
                    f"Think about basic {topic} rules like grammar, spelling, and simple vocabulary",
                    f"Consider common words and phrases you use every day",
                    f"Remember basic language structure and simple sentences"
                ]
            elif difficulty == DifficultyLevel.MEDIUM:
                hints = [
                    f"Think about {topic} concepts like grammar rules, word relationships, and sentence structure",
                    f"Consider different parts of speech and how they work together",
                    f"Remember language patterns and common expressions"
                ]
            else:  # HARD
                hints = [
                    f"Advanced {topic}: Think about literary devices, complex grammar, and sophisticated vocabulary",
                    f"Consider literary analysis, rhetorical techniques, and advanced language concepts",
                    f"Think about writing styles, literary movements, and complex language structures"
                ]
        
        question = Question(
            id=random.randint(1000, 9999),
            content="",  # Will be filled by _generate_realistic_question
            topic=topic,
            difficulty=difficulty,
            question_type=random.choice([QuestionType.SHORT_ANSWER, QuestionType.MULTIPLE_CHOICE, QuestionType.TRUE_FALSE]),
            correct_answer="",  # Will be determined during interaction
            hints=hints,
            explanation=f"This {topic} question tests your understanding of {difficulty.name.lower()} level concepts."
        )
        
        # Generate the actual question text
        question.content = self.human_student._generate_realistic_question(question)
        
        return question
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation for RL agents."""
        if self.human_student is None:
            return np.zeros(self.state_size)
        
        # Build state vector from human student data
        state = []
        
        # Student profile approximation
        summary = self.human_student.get_session_summary()
        state.extend([
            summary['accuracy'],  # Learning rate approximation
            self.human_student.get_engagement_level(),  # Attention span
            0.5,  # Difficulty preference (neutral)
            self.human_student.current_motivation,
            self.human_student.fatigue
        ])
        
        # Knowledge levels (4 subjects)
        for topic in ['mathematics', 'science', 'programming', 'language']:
            state.append(self.human_student.knowledge_levels.get(topic, 0.5))
        
        # Performance metrics
        state.extend([
            min(summary['questions_answered'] / 10, 1.0),  # Session progress
            summary['accuracy'],  # Success rate
            0.0 if summary['questions_answered'] == 0 else min(summary['avg_response_time'] / 30, 1.0),  # Response time
            min(self.episode_step / self.max_episode_steps, 1.0),  # Episode progress
            self.human_student.get_engagement_level(),  # Current engagement
            min(summary['session_duration'] / 1800, 1.0)  # Session duration (normalized to 30 min)
        ])
        
        return np.array(state, dtype=np.float32)
    
    def get_final_summary(self) -> Dict:
        """Get comprehensive session summary."""
        if self.human_student is None:
            return {}
        
        summary = self.human_student.get_session_summary()
        summary.update({
            'total_steps': self.episode_step,
            'questions_per_minute': summary['questions_answered'] / (summary['session_duration'] / 60) if summary['session_duration'] > 0 else 0,
            'learning_efficiency': summary['accuracy'] * summary['questions_answered'],
            'engagement_trend': 'improving' if self.human_student.current_motivation > 0.6 else 'needs_attention'
        })
        
        return summary
    
    def print_session_summary(self):
        """Print a nice summary of the learning session."""
        summary = self.get_final_summary()
        
        print("\n" + "="*60)
        print("🎓 LEARNING SESSION SUMMARY")
        print("="*60)
        print(f"Student: {summary['student_name']}")
        print(f"Questions Answered: {summary['questions_answered']}")
        print(f"Correct Answers: {summary['correct_answers']}")
        print(f"Accuracy: {summary['accuracy']:.1%}")
        print(f"Session Duration: {summary['session_duration']/60:.1f} minutes")
        print(f"Final Engagement: {summary['current_engagement']:.1%}")
        print(f"Hints Used: {summary['hints_used']}")
        
        print(f"\n📊 KNOWLEDGE LEVELS:")
        for topic, level in summary['knowledge_levels'].items():
            bar = "█" * int(level * 10) + "░" * (10 - int(level * 10))
            print(f"  {topic.title():12} [{bar}] {level:.1%}")
        
        print(f"\n🏆 ACHIEVEMENT LEVEL: ", end="")
        if summary['accuracy'] >= 0.8:
            print("🌟 Excellent!")
        elif summary['accuracy'] >= 0.6:
            print("👍 Good work!")
        elif summary['accuracy'] >= 0.4:
            print("📈 Making progress!")
        else:
            print("💪 Keep practicing!")
        
        print("="*60)
    
    def _calculate_recent_accuracy(self, topic=None, lookback=5):
        """Calculate accuracy for recent answers, optionally filtered by topic."""
        if not self.human_student or not self.human_student.answer_history:
            return 0.5  # Default neutral accuracy
        
        # Get recent answers (last 'lookback' answers)
        recent_answers = self.human_student.answer_history[-lookback:]
        
        # Filter by topic if specified
        if topic:
            recent_answers = [ans for ans in recent_answers if ans.get('topic') == topic]
        
        # If no recent answers for this topic, return neutral
        if not recent_answers:
            return 0.5
        
        # Calculate accuracy
        correct_count = sum(1 for ans in recent_answers if ans.get('correct', False))
        return correct_count / len(recent_answers)


# Demo function for testing
def run_human_demo():
    """Run a demo session with human interaction."""
    print("Starting Human Interactive Tutoring Demo...")
    
    student_name = input("Enter your name: ").strip() or "Student"
    
    env = HumanTutoringEnvironment(student_name=student_name)
    state = env.reset()
    
    done = False
    step_count = 0
    
    while not done and step_count < 20:  # Limit for demo
        # Simple action selection (in real system, AI agents would choose)
        print(f"\n🤖 AI Tutor is thinking...")
        time.sleep(1)
        
        # Choose action based on simple heuristics
        engagement = env.human_student.get_engagement_level()
        
        if step_count == 0 or engagement > 0.7:
            action = 0  # ASK_QUESTION
        elif engagement < 0.4:
            action = 3  # REVIEW_PREVIOUS
        elif env.current_question and random.random() < 0.3:
            action = 1  # PROVIDE_HINT
        else:
            action = 2  # EXPLAIN_CONCEPT
        
        state, reward, done, info = env.step(action)
        step_count += 1
        
        print(f"🔄 Step {step_count}, Action: {ActionType(action).name}, Reward: {reward:.1f}")
    
    env.print_session_summary()


if __name__ == "__main__":
    run_human_demo()
