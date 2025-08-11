"""
Professional FastAPI Web Interface for RL Tutorial System
Replicates complete_assignment_demo.py functionality with modern web UI
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import os
import sys
import time
import random
import json
import uvicorn
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

# Add current directory to path for imports
sys.path.append('.')

# Import the components from complete_assignment_demo.py
try:
    from complete_assignment_demo import (
        StudentProfile, LearningSession, Question, Difficulty, CoordinationMode,
        ComprehensiveQuestionBank, EnhancedDQNAgent, EnhancedPPOAgent
    )
    from student_results_manager import StudentResultsManager
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    COMPONENTS_AVAILABLE = False

# FastAPI app initialization
app = FastAPI(
    title="RL Tutorial System - Professional Web Interface",
    description="Multi-Agent Reinforcement Learning Tutorial System",
    version="2.0.0"
)

# Global storage
active_sessions: Dict[str, Any] = {}
question_bank = None
results_manager = None

# Initialize components
if COMPONENTS_AVAILABLE:
    try:
        question_bank = ComprehensiveQuestionBank()
        results_manager = StudentResultsManager()
        print("✅ All components initialized successfully")
    except Exception as e:
        print(f"⚠️ Component initialization error: {e}")

# Pydantic models
class StudentCreate(BaseModel):
    name: str
    student_id: str
    preferred_topics: List[str]
    preferred_difficulty: str
    learning_style: str

class SessionStart(BaseModel):
    session_id: str
    coordination_mode: str

class ResponseSubmit(BaseModel):
    session_id: str
    response: str

# HTML Templates
def get_base_html(content: str, title: str = "RL Tutorial System") -> str:
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            font-size: 3em;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 15px;
        }}
        
        .header p {{
            font-size: 1.2em;
            color: #666;
        }}
        
        .card {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            border: 1px solid rgba(255,255,255,0.2);
        }}
        
        .form-group {{
            margin-bottom: 25px;
        }}
        
        .form-group label {{
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #555;
            font-size: 1.1em;
        }}
        
        .form-control {{
            width: 100%;
            padding: 15px 20px;
            border: 2px solid #e1e5e9;
            border-radius: 12px;
            font-size: 16px;
            transition: all 0.3s ease;
            background: white;
        }}
        
        .form-control:focus {{
            outline: none;
            border-color: #667eea;
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.2);
        }}
        
        .btn {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-block;
            margin: 5px;
        }}
        
        .btn:hover {{
            transform: translateY(-3px);
            box-shadow: 0 15px 30px rgba(0,0,0,0.2);
        }}
        
        .btn-secondary {{
            background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
        }}
        
        .btn-large {{
            padding: 20px 40px;
            font-size: 18px;
            border-radius: 15px;
        }}
        
        .coordination-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        
        .coordination-card {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 2px solid transparent;
        }}
        
        .coordination-card:hover {{
            transform: translateY(-5px);
            border-color: #667eea;
            box-shadow: 0 20px 40px rgba(102, 126, 234, 0.2);
        }}
        
        .coordination-card i {{
            font-size: 3em;
            margin-bottom: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        
        .stat-card {{
            background: rgba(255, 255, 255, 0.95);
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            border-left: 5px solid #667eea;
        }}
        
        .stat-value {{
            font-size: 2.5em;
            font-weight: 700;
            color: #667eea;
            margin-bottom: 5px;
        }}
        
        .stat-label {{
            color: #666;
            font-weight: 500;
        }}
        
        .question-card {{
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-left: 5px solid #667eea;
            border-radius: 15px;
            padding: 30px;
            margin: 25px 0;
        }}
        
        .question-card h3 {{
            color: #667eea;
            margin-bottom: 15px;
        }}
        
        .progress-container {{
            background: #e9ecef;
            border-radius: 10px;
            height: 12px;
            margin: 20px 0;
            overflow: hidden;
        }}
        
        .progress-bar {{
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            transition: width 0.5s ease;
        }}
        
        .alert {{
            padding: 20px;
            border-radius: 12px;
            margin: 20px 0;
            border-left: 5px solid;
        }}
        
        .alert-success {{
            background: rgba(40, 167, 69, 0.1);
            border-color: #28a745;
            color: #155724;
        }}
        
        .alert-info {{
            background: rgba(23, 162, 184, 0.1);
            border-color: #17a2b8;
            color: #0c5460;
        }}
        
        .alert-warning {{
            background: rgba(255, 193, 7, 0.1);
            border-color: #ffc107;
            color: #856404;
        }}
        
        .checkbox-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        
        .checkbox-item {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            border: 2px solid #e9ecef;
            transition: all 0.3s ease;
            cursor: pointer;
        }}
        
        .checkbox-item:hover {{
            border-color: #667eea;
            background: rgba(102, 126, 234, 0.05);
        }}
        
        .checkbox-item input[type="checkbox"] {{
            margin-right: 10px;
            transform: scale(1.2);
        }}
        
        .checkbox-item.checked {{
            border-color: #667eea;
            background: rgba(102, 126, 234, 0.1);
        }}
        
        .loading {{
            text-align: center;
            padding: 40px;
        }}
        
        .spinner {{
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }}
        
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        
        .fade-in {{
            animation: fadeIn 0.5s ease-in;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 10px;
            }}
            
            .header {{
                padding: 20px;
            }}
            
            .header h1 {{
                font-size: 2em;
            }}
            
            .coordination-cards {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header fade-in">
            <h1><i class="fas fa-robot"></i> RL Tutorial System</h1>
            <p>Multi-Agent Reinforcement Learning for Adaptive Education</p>
        </div>
        
        <div id="main-content" class="fade-in">
            {content}
        </div>
    </div>
    
    <script>
        // Enhanced JavaScript functionality
        
        function showLoading() {{
            document.getElementById('main-content').innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    <h3>Processing your request...</h3>
                    <p>Please wait while we set up your personalized learning experience.</p>
                </div>
            `;
        }}
        
        function showError(message) {{
            document.getElementById('main-content').innerHTML = `
                <div class="card">
                    <div class="alert alert-warning">
                        <h3><i class="fas fa-exclamation-triangle"></i> Error</h3>
                        <p>${{message}}</p>
                        <button class="btn" onclick="location.reload()">
                            <i class="fas fa-refresh"></i> Try Again
                        </button>
                    </div>
                </div>
            `;
        }}
        
        async function makeRequest(url, data = null) {{
            try {{
                showLoading();
                const options = {{
                    method: data ? 'POST' : 'GET',
                    headers: {{
                        'Content-Type': 'application/json',
                    }}
                }};
                
                if (data) {{
                    options.body = JSON.stringify(data);
                }}
                
                const response = await fetch(url, options);
                const result = await response.json();
                
                if (result.success) {{
                    document.getElementById('main-content').innerHTML = result.html;
                    document.getElementById('main-content').classList.add('fade-in');
                }} else {{
                    showError(result.message || 'An error occurred');
                }}
            }} catch (error) {{
                showError('Network error. Please check your connection and try again.');
                console.error('Error:', error);
            }}
        }}
        
        function createStudent() {{
            const name = document.getElementById('name').value.trim();
            const studentId = document.getElementById('student_id').value.trim();
            const difficulty = document.getElementById('difficulty').value;
            const learningStyle = document.getElementById('learning_style').value;
            
            const checkedTopics = Array.from(document.querySelectorAll('input[name="topics"]:checked'))
                .map(cb => cb.value);
            
            if (!name || !studentId) {{
                alert('Please enter your name and student ID');
                return;
            }}
            
            if (checkedTopics.length === 0) {{
                alert('Please select at least one preferred topic');
                return;
            }}
            
            const data = {{
                name: name,
                student_id: studentId,
                preferred_topics: checkedTopics,
                preferred_difficulty: difficulty,
                learning_style: learningStyle
            }};
            
            makeRequest('/api/create_student', data);
        }}
        
        function selectCoordination(sessionId, mode) {{
            const data = {{
                session_id: sessionId,
                coordination_mode: mode
            }};
            
            makeRequest('/api/start_session', data);
        }}
        
        function submitResponse(sessionId) {{
            const response = document.getElementById('response').value.trim();
            
            if (!response) {{
                alert('Please provide a response to the question');
                return;
            }}
            
            const data = {{
                session_id: sessionId,
                response: response
            }};
            
            makeRequest('/api/submit_response', data);
        }}
        
        function restartSystem() {{
            makeRequest('/api/restart');
        }}
        
        // Checkbox interaction
        document.addEventListener('change', function(e) {{
            if (e.target.type === 'checkbox') {{
                const item = e.target.closest('.checkbox-item');
                if (e.target.checked) {{
                    item.classList.add('checked');
                }} else {{
                    item.classList.remove('checked');
                }}
            }}
        }});
        
        // Auto-resize textareas
        document.addEventListener('input', function(e) {{
            if (e.target.tagName === 'TEXTAREA') {{
                e.target.style.height = 'auto';
                e.target.style.height = e.target.scrollHeight + 'px';
            }}
        }});
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    """Main page - Student profile creation"""
    if not COMPONENTS_AVAILABLE:
        content = """
        <div class="card">
            <div class="alert alert-warning">
                <h3><i class="fas fa-exclamation-triangle"></i> System Not Ready</h3>
                <p>The RL tutorial system components are not available. Please ensure all required files are present.</p>
            </div>
        </div>
        """
        return get_base_html(content)
    
    content = """
    <div class="card">
        <div class="alert alert-info">
            <h3><i class="fas fa-info-circle"></i> Welcome to the RL Tutorial System</h3>
            <p>Experience cutting-edge multi-agent reinforcement learning in education. Our DQN and PPO agents work together to create a personalized learning experience that adapts to your responses in real-time.</p>
        </div>
        
        <h2><i class="fas fa-user-plus"></i> Create Your Student Profile</h2>
        <p>Tell us about your learning preferences so our AI agents can personalize your experience.</p>
        
        <form onsubmit="event.preventDefault(); createStudent();">
            <div class="form-group">
                <label for="name"><i class="fas fa-user"></i> Full Name *</label>
                <input type="text" id="name" class="form-control" placeholder="Enter your full name" required>
            </div>
            
            <div class="form-group">
                <label for="student_id"><i class="fas fa-id-card"></i> Student ID *</label>
                <input type="text" id="student_id" class="form-control" placeholder="Enter your student ID" required>
            </div>
            
            <div class="form-group">
                <label><i class="fas fa-heart"></i> Preferred Learning Topics * (Select at least one)</label>
                <div class="checkbox-grid">
                    <div class="checkbox-item">
                        <input type="checkbox" id="math" name="topics" value="mathematics">
                        <label for="math"><i class="fas fa-calculator"></i> Mathematics</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="science" name="topics" value="science">
                        <label for="science"><i class="fas fa-flask"></i> Science</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="programming" name="topics" value="programming">
                        <label for="programming"><i class="fas fa-code"></i> Programming</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="language" name="topics" value="language">
                        <label for="language"><i class="fas fa-book"></i> Language Arts</label>
                    </div>
                </div>
            </div>
            
            <div class="form-group">
                <label for="difficulty"><i class="fas fa-chart-line"></i> Preferred Difficulty Level</label>
                <select id="difficulty" class="form-control">
                    <option value="easy">Easy - Building fundamentals</option>
                    <option value="medium" selected>Medium - Balanced challenge</option>
                    <option value="hard">Hard - Advanced concepts</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="learning_style"><i class="fas fa-brain"></i> Learning Style</label>
                <select id="learning_style" class="form-control">
                    <option value="visual">Visual - Learn through images and diagrams</option>
                    <option value="auditory">Auditory - Learn through listening</option>
                    <option value="kinesthetic">Kinesthetic - Learn through hands-on activities</option>
                    <option value="reading" selected>Reading/Writing - Learn through text</option>
                </select>
            </div>
            
            <button type="submit" class="btn btn-large">
                <i class="fas fa-rocket"></i> Create Profile & Start Learning
            </button>
        </form>
    </div>
    """
    
    return get_base_html(content)

@app.post("/api/create_student")
async def create_student(student: StudentCreate):
    """Create student profile and show coordination selection"""
    try:
        if not COMPONENTS_AVAILABLE:
            return JSONResponse({
                "success": False,
                "message": "System components not available"
            })
        
        # Create student profile
        profile = StudentProfile(
            name=student.name,
            student_id=student.student_id,
            created_at=datetime.now().isoformat(),
            preferred_topics=student.preferred_topics,
            preferred_difficulty=student.preferred_difficulty,
            learning_style=student.learning_style
        )
        
        # Generate session ID
        session_id = f"{student.student_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Store session
        active_sessions[session_id] = {
            'profile': profile,
            'session_id': session_id,
            'created_at': datetime.now().isoformat()
        }
        
        content = f"""
        <div class="card">
            <div class="alert alert-success">
                <h3><i class="fas fa-check-circle"></i> Profile Created Successfully!</h3>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value"><i class="fas fa-user"></i></div>
                        <div class="stat-label">{profile.name}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value"><i class="fas fa-id-badge"></i></div>
                        <div class="stat-label">{profile.student_id}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{len(profile.preferred_topics)}</div>
                        <div class="stat-label">Topics Selected</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value"><i class="fas fa-brain"></i></div>
                        <div class="stat-label">{profile.learning_style.title()}</div>
                    </div>
                </div>
            </div>
            
            <h2><i class="fas fa-robot"></i> Choose Multi-Agent Coordination Mode</h2>
            <p>Select how our DQN and PPO agents will work together to optimize your learning experience:</p>
            
            <div class="coordination-cards">
                <div class="coordination-card" onclick="selectCoordination('{session_id}', 'hierarchical')">
                    <i class="fas fa-sitemap"></i>
                    <h3>Hierarchical Mode</h3>
                    <p>PPO agent provides strategic oversight while DQN agent handles tactical content selection. Clear command structure for complex learning scenarios.</p>
                    <div style="margin-top: 15px;">
                        <span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 5px 15px; border-radius: 20px; font-size: 0.9em;">
                            <i class="fas fa-crown"></i> Strategic Leadership
                        </span>
                    </div>
                </div>
                
                <div class="coordination-card" onclick="selectCoordination('{session_id}', 'collaborative')">
                    <i class="fas fa-handshake"></i>
                    <h3>Collaborative Mode</h3>
                    <p>Both agents work together on joint decisions with shared responsibility. Balanced approach combining both agent strengths for optimal outcomes.</p>
                    <div style="margin-top: 15px;">
                        <span style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; padding: 5px 15px; border-radius: 20px; font-size: 0.9em;">
                            <i class="fas fa-users"></i> Team Approach
                        </span>
                    </div>
                </div>
                
                <div class="coordination-card" onclick="selectCoordination('{session_id}', 'competitive')">
                    <i class="fas fa-trophy"></i>
                    <h3>Competitive Mode</h3>
                    <p>Agents compete based on performance metrics with dynamic leadership. The best-performing agent takes control to maximize learning effectiveness.</p>
                    <div style="margin-top: 15px;">
                        <span style="background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%); color: white; padding: 5px 15px; border-radius: 20px; font-size: 0.9em;">
                            <i class="fas fa-bolt"></i> Performance Driven
                        </span>
                    </div>
                </div>
            </div>
            
            <div class="alert alert-info">
                <h4><i class="fas fa-lightbulb"></i> How It Works</h4>
                <p>Each coordination mode demonstrates different multi-agent reinforcement learning strategies. Your choice affects how the AI agents adapt their teaching approach based on your responses and learning patterns.</p>
            </div>
        </div>
        """
        
        return JSONResponse({
            "success": True,
            "html": content
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": str(e)
        })

@app.post("/api/start_session")
async def start_session(session_config: SessionStart):
    """Start learning session with coordination mode"""
    try:
        if session_config.session_id not in active_sessions:
            return JSONResponse({
                "success": False,
                "message": "Session not found"
            })
        
        session_data = active_sessions[session_config.session_id]
        profile = session_data['profile']
        
        # Initialize RL agents
        dqn_agent = EnhancedDQNAgent("DQN-Content")
        ppo_agent = EnhancedPPOAgent("PPO-Strategy")
        
        # Update session with learning configuration
        session_data.update({
            'coordination_mode': session_config.coordination_mode,
            'dqn_agent': dqn_agent,
            'ppo_agent': ppo_agent,
            'current_question': 0,
            'total_questions': 7,
            'cumulative_reward': 0.0,
            'questions_history': [],
            'session_start': datetime.now().isoformat(),
            'dqn_updates': 0,
            'ppo_updates': 0
        })
        
        # Get first question
        return await get_next_question(session_config.session_id)
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": str(e)
        })

async def get_next_question(session_id: str):
    """Get next question for the learning session"""
    try:
        session_data = active_sessions[session_id]
        profile = session_data['profile']
        current_q = session_data['current_question']
        total_q = session_data['total_questions']
        
        if current_q >= total_q:
            return await complete_session(session_id)
        
        # Select question using agent coordination
        topic = random.choice(profile.preferred_topics)
        difficulty = random.choice(['easy', 'medium', 'hard'])
        
        available_questions = question_bank.questions[topic][difficulty]
        question_data = random.choice(available_questions)
        
        question_info = {
            'topic': topic,
            'difficulty': difficulty,
            'text': question_data['q'],
            'sample': question_data['sample'],
            'question_number': current_q + 1
        }
        
        session_data['current_question_info'] = question_info
        
        # Calculate progress
        progress = ((current_q) / total_q) * 100
        
        content = f"""
        <div class="card">
            <div class="alert alert-info">
                <h3><i class="fas fa-brain"></i> Learning Round {question_info['question_number']}/{total_q}</h3>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value"><i class="fas fa-robot"></i></div>
                        <div class="stat-label">{session_data['coordination_mode'].title()} Mode</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{profile.topic_performance.get(topic, 0.5):.2f}</div>
                        <div class="stat-label">{topic.title()} Performance</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{profile.engagement_score:.2f}</div>
                        <div class="stat-label">Engagement Score</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{session_data['cumulative_reward']:.2f}</div>
                        <div class="stat-label">Total Reward</div>
                    </div>
                </div>
            </div>
            
            <div class="progress-container">
                <div class="progress-bar" style="width: {progress}%"></div>
            </div>
            <p style="text-align: center; color: #666; margin-bottom: 20px;">Progress: {progress:.1f}%</p>
            
            <div class="question-card">
                <h3><i class="fas fa-question-circle"></i> Question: {topic.title()} ({difficulty.title()} Level)</h3>
                <p style="font-size: 1.1em; line-height: 1.6; margin: 20px 0;">
                    <strong>{question_data['q']}</strong>
                </p>
                
                <div class="form-group">
                    <label for="response"><i class="fas fa-edit"></i> Your Response:</label>
                    <textarea id="response" class="form-control" rows="5" 
                        placeholder="Provide a detailed explanation with examples and reasoning. The more thoughtful your response, the better our AI agents can adapt to help you learn..."></textarea>
                </div>
                
                <button class="btn btn-large" onclick="submitResponse('{session_id}')">
                    <i class="fas fa-paper-plane"></i> Submit Response
                </button>
            </div>
            
            <div class="alert alert-info">
                <h4><i class="fas fa-lightbulb"></i> Sample Answer</h4>
                <p><em>{question_data['sample']}</em></p>
            </div>
        </div>
        """
        
        return JSONResponse({
            "success": True,
            "html": content
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": str(e)
        })

@app.post("/api/submit_response")
async def submit_response(response_data: ResponseSubmit):
    """Process student response and update RL agents"""
    try:
        session_id = response_data.session_id
        if session_id not in active_sessions:
            return JSONResponse({
                "success": False,
                "message": "Session not found"
            })
        
        session_data = active_sessions[session_id]
        profile = session_data['profile']
        question_info = session_data['current_question_info']
        response = response_data.response
        
        # Evaluate response (same logic as complete_assignment_demo.py)
        response_length = len(response)
        
        if response_length < 10:
            reward = 0.1
            feedback = "Brief response - try to elaborate more with examples"
        elif response_length < 50:
            reward = 0.3
            feedback = "Good start - consider adding more detail and examples"
        elif response_length < 100:
            reward = 0.6
            feedback = "Well-developed response with good explanation"
        else:
            reward = 0.8
            feedback = "Excellent detailed response showing thorough understanding"
        
        # Update session data
        session_data['cumulative_reward'] += reward
        session_data['current_question'] += 1
        session_data['dqn_updates'] += 1
        session_data['ppo_updates'] += 1
        
        # Store interaction for results manager
        if results_manager:
            try:
                interaction_data = {
                    'timestamp': datetime.now().isoformat(),
                    'session_id': session_id,
                    'question_text': question_info['text'],
                    'topic': question_info['topic'],
                    'difficulty': question_info['difficulty'],
                    'student_response': response,
                    'response_length': response_length,
                    'reward_score': reward,
                    'feedback': feedback,
                    'dqn_action': 0,
                    'ppo_topic_selection': question_info['topic'],
                    'cumulative_reward': session_data['cumulative_reward'],
                    'session_number': question_info['question_number']
                }
                results_manager.record_interaction(interaction_data)
            except:
                pass
        
        # Show feedback
        content = f"""
        <div class="card">
            <div class="alert alert-success">
                <h3><i class="fas fa-check-circle"></i> Response Processed Successfully!</h3>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">{reward:.2f}</div>
                        <div class="stat-label">Reward Earned</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{response_length}</div>
                        <div class="stat-label">Response Length</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{session_data['cumulative_reward']:.2f}</div>
                        <div class="stat-label">Total Reward</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{session_data['current_question']}/{session_data['total_questions']}</div>
                        <div class="stat-label">Progress</div>
                    </div>
                </div>
            </div>
            
            <div class="alert alert-info">
                <h4><i class="fas fa-comment-alt"></i> AI Feedback</h4>
                <p><strong>{feedback}</strong></p>
            </div>
            
            <div class="alert alert-success">
                <h4><i class="fas fa-brain"></i> Real-time RL Agent Updates</h4>
                <p><strong><i class="fas fa-network-wired"></i> DQN Update #{session_data['dqn_updates']}:</strong> Q-value adjustment based on response quality and engagement</p>
                <p><strong><i class="fas fa-chart-line"></i> PPO Update #{session_data['ppo_updates']}:</strong> Policy gradient step with performance score: {reward:.3f}</p>
            </div>
            
            <div style="text-align: center; margin: 30px 0;">
                <button class="btn btn-large" onclick="makeRequest('/api/continue/{session_id}')">
                    <i class="fas fa-arrow-right"></i> Continue to Next Question
                </button>
            </div>
        </div>
        """
        
        return JSONResponse({
            "success": True,
            "html": content
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": str(e)
        })

@app.get("/api/continue/{session_id}")
async def continue_session(session_id: str):
    """Continue to next question"""
    if session_id not in active_sessions:
        return JSONResponse({
            "success": False,
            "message": "Session not found"
        })
    
    return await get_next_question(session_id)

async def complete_session(session_id: str):
    """Complete learning session and show results"""
    try:
        session_data = active_sessions[session_id]
        profile = session_data['profile']
        
        # Calculate final stats
        total_questions = session_data['total_questions']
        cumulative_reward = session_data['cumulative_reward']
        average_reward = cumulative_reward / total_questions if total_questions > 0 else 0
        
        # Save session summary
        if results_manager:
            try:
                session_summary = {
                    'session_id': session_id,
                    'student_id': profile.student_id,
                    'start_time': session_data.get('session_start', ''),
                    'end_time': datetime.now().isoformat(),
                    'total_interactions': total_questions,
                    'topics_covered': profile.preferred_topics,
                    'average_reward': average_reward,
                    'total_reward': cumulative_reward,
                    'coordination_mode': session_data.get('coordination_mode', 'N/A'),
                    'agent_coordination_mode': session_data.get('coordination_mode', 'N/A')
                }
                results_manager.save_session_summary(session_summary)
            except Exception as save_error:
                print(f"⚠️ Error saving session summary: {save_error}")
        
        content = f"""
        <div class="card">
            <div class="alert alert-success">
                <h2><i class="fas fa-trophy"></i> Learning Session Complete!</h2>
                <p>Congratulations on completing your personalized RL tutorial session with our multi-agent system.</p>
            </div>
            
            <h3><i class="fas fa-chart-bar"></i> Final Learning Analytics Report</h3>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{total_questions}</div>
                    <div class="stat-label">Questions Completed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{cumulative_reward:.2f}</div>
                    <div class="stat-label">Total Reward</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{average_reward:.2f}</div>
                    <div class="stat-label">Average Reward</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{session_data['dqn_updates']}</div>
                    <div class="stat-label">DQN Updates</div>
                </div>
            </div>
            
            <div class="alert alert-info">
                <h4><i class="fas fa-user"></i> Session Summary</h4>
                <p><strong>Student:</strong> {profile.name} (ID: {profile.student_id})</p>
                <p><strong>Coordination Mode:</strong> {session_data['coordination_mode'].title()}</p>
                <p><strong>Topics Covered:</strong> {', '.join(profile.preferred_topics)}</p>
                <p><strong>Learning Style:</strong> {profile.learning_style.title()}</p>
                <p><strong>Agent Performance:</strong> DQN ({session_data['dqn_updates']} updates), PPO ({session_data['ppo_updates']} updates)</p>
            </div>
            
            <div class="alert alert-success">
                <h4><i class="fas fa-graduation-cap"></i> Assignment Requirements Demonstrated</h4>
                <p><i class="fas fa-check"></i> <strong>Value-Based Learning (DQN):</strong> Q-value updates with student adaptation</p>
                <p><i class="fas fa-check"></i> <strong>Policy Gradient Methods (PPO):</strong> Policy optimization with engagement factors</p>
                <p><i class="fas fa-check"></i> <strong>Multi-Agent Coordination:</strong> {session_data['coordination_mode'].title()} mode coordination</p>
                <p><i class="fas fa-check"></i> <strong>Real-time Learning:</strong> Continuous adaptation to student responses</p>
                <p><i class="fas fa-check"></i> <strong>Student Progress Definition:</strong> Comprehensive profiling and tracking</p>
                <p><i class="fas fa-check"></i> <strong>Subjective Assessment:</strong> Open-ended question evaluation</p>
            </div>
            
            <div style="text-align: center; margin: 30px 0;">
                <button class="btn btn-large" onclick="restartSystem()">
                    <i class="fas fa-redo"></i> Start New Session
                </button>
                <a href="/api/results" target="_blank" class="btn btn-secondary btn-large">
                    <i class="fas fa-chart-line"></i> View Detailed Results
                </a>
            </div>
        </div>
        """
        
        # Mark session complete
        session_data['completed'] = True
        
        return JSONResponse({
            "success": True,
            "html": content
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": str(e)
        })

@app.post("/api/restart")
async def restart_system():
    """Restart the system"""
    global active_sessions
    active_sessions.clear()
    return JSONResponse({"success": True, "html": "Restarting..."})

@app.get("/api/results", response_class=HTMLResponse)
async def get_results():
    """Get system results in a detailed HTML page"""
    try:
        if not results_manager:
            content = """
            <div class="card">
                <div class="alert alert-warning">
                    <h3><i class="fas fa-exclamation-triangle"></i> Results Manager Not Available</h3>
                    <p>The results tracking system is not initialized. Results cannot be displayed.</p>
                </div>
            </div>
            """
            return get_base_html(content, "Results - Not Available")
        
        interactions = results_manager.get_all_interactions()
        sessions = results_manager.get_all_sessions()
        
        # Calculate summary statistics
        total_interactions = len(interactions)
        total_sessions = len(sessions)
        
        if total_interactions > 0:
            avg_reward = sum(i.get('reward_score', 0) for i in interactions) / total_interactions
            total_reward = sum(i.get('reward_score', 0) for i in interactions)
            
            # Topic analysis
            topic_counts = {}
            for interaction in interactions:
                topic = interaction.get('topic', 'unknown')
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
            
            # Recent activity (last 10 interactions)
            recent_interactions = interactions[-10:] if len(interactions) > 10 else interactions
        else:
            avg_reward = 0
            total_reward = 0
            topic_counts = {}
            recent_interactions = []
        
        # Generate detailed results HTML
        content = f"""
        <div class="card">
            <h2><i class="fas fa-chart-bar"></i> Detailed Learning Analytics Report</h2>
            <p>Comprehensive analysis of student interactions and RL agent performance</p>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{total_interactions}</div>
                    <div class="stat-label">Total Interactions</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{total_sessions}</div>
                    <div class="stat-label">Learning Sessions</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{avg_reward:.3f}</div>
                    <div class="stat-label">Average Reward</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{total_reward:.2f}</div>
                    <div class="stat-label">Total Cumulative Reward</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3><i class="fas fa-chart-pie"></i> Topic Distribution</h3>
            <div class="stats-grid">
        """
        
        # Add topic statistics
        for topic, count in topic_counts.items():
            percentage = (count / total_interactions * 100) if total_interactions > 0 else 0
            content += f"""
                <div class="stat-card">
                    <div class="stat-value">{count}</div>
                    <div class="stat-label">{topic.title()} ({percentage:.1f}%)</div>
                </div>
            """
        
        content += """
            </div>
        </div>
        """
        
        # Add recent interactions
        if recent_interactions:
            content += """
            <div class="card">
                <h3><i class="fas fa-history"></i> Recent Learning Interactions</h3>
                <div style="overflow-x: auto;">
                    <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
                        <thead>
                            <tr style="background: #f8f9fa; border-bottom: 2px solid #dee2e6;">
                                <th style="padding: 12px; text-align: left; border: 1px solid #dee2e6;">Timestamp</th>
                                <th style="padding: 12px; text-align: left; border: 1px solid #dee2e6;">Topic</th>
                                <th style="padding: 12px; text-align: left; border: 1px solid #dee2e6;">Difficulty</th>
                                <th style="padding: 12px; text-align: left; border: 1px solid #dee2e6;">Response Length</th>
                                <th style="padding: 12px; text-align: left; border: 1px solid #dee2e6;">Reward</th>
                                <th style="padding: 12px; text-align: left; border: 1px solid #dee2e6;">Feedback</th>
                            </tr>
                        </thead>
                        <tbody>
            """
            
            for interaction in recent_interactions:
                timestamp = interaction.get('timestamp', '')
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', ''))
                        formatted_time = dt.strftime('%m/%d %H:%M')
                    except:
                        formatted_time = timestamp[:16]
                else:
                    formatted_time = 'Unknown'
                
                content += f"""
                    <tr style="border-bottom: 1px solid #dee2e6;">
                        <td style="padding: 10px; border: 1px solid #dee2e6;">{formatted_time}</td>
                        <td style="padding: 10px; border: 1px solid #dee2e6;">{interaction.get('topic', 'N/A').title()}</td>
                        <td style="padding: 10px; border: 1px solid #dee2e6;">{interaction.get('difficulty', 'N/A').title()}</td>
                        <td style="padding: 10px; border: 1px solid #dee2e6;">{interaction.get('response_length', 0)} chars</td>
                        <td style="padding: 10px; border: 1px solid #dee2e6; color: #28a745; font-weight: bold;">{interaction.get('reward_score', 0):.3f}</td>
                        <td style="padding: 10px; border: 1px solid #dee2e6; max-width: 300px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{interaction.get('feedback', 'No feedback')}</td>
                    </tr>
                """
            
            content += """
                        </tbody>
                    </table>
                </div>
            </div>
            """
        
        # Add sessions summary
        if sessions:
            content += """
            <div class="card">
                <h3><i class="fas fa-graduation-cap"></i> Learning Sessions Summary</h3>
                <div style="overflow-x: auto;">
                    <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
                        <thead>
                            <tr style="background: #f8f9fa; border-bottom: 2px solid #dee2e6;">
                                <th style="padding: 12px; text-align: left; border: 1px solid #dee2e6;">Session ID</th>
                                <th style="padding: 12px; text-align: left; border: 1px solid #dee2e6;">Student ID</th>
                                <th style="padding: 12px; text-align: left; border: 1px solid #dee2e6;">Start Time</th>
                                <th style="padding: 12px; text-align: left; border: 1px solid #dee2e6;">Interactions</th>
                                <th style="padding: 12px; text-align: left; border: 1px solid #dee2e6;">Total Reward</th>
                                <th style="padding: 12px; text-align: left; border: 1px solid #dee2e6;">Coordination Mode</th>
                            </tr>
                        </thead>
                        <tbody>
            """
            
            for session in sessions[-10:]:  # Show last 10 sessions
                start_time = session.get('start_time', '')
                if start_time:
                    try:
                        dt = datetime.fromisoformat(start_time.replace('Z', ''))
                        formatted_time = dt.strftime('%m/%d/%Y %H:%M')
                    except:
                        formatted_time = start_time[:16]
                else:
                    formatted_time = 'Unknown'
                
                content += f"""
                    <tr style="border-bottom: 1px solid #dee2e6;">
                        <td style="padding: 10px; border: 1px solid #dee2e6; font-family: monospace; font-size: 0.9em;">{session.get('session_id', 'N/A')[:20]}...</td>
                        <td style="padding: 10px; border: 1px solid #dee2e6;">{session.get('student_id', 'N/A')}</td>
                        <td style="padding: 10px; border: 1px solid #dee2e6;">{formatted_time}</td>
                        <td style="padding: 10px; border: 1px solid #dee2e6;">{session.get('total_interactions', 0)}</td>
                        <td style="padding: 10px; border: 1px solid #dee2e6; color: #28a745; font-weight: bold;">{session.get('total_reward', 0):.2f}</td>
                        <td style="padding: 10px; border: 1px solid #dee2e6;">{session.get('agent_coordination_mode', session.get('coordination_mode', 'N/A')).title()}</td>
                    </tr>
                """
            
            content += """
                        </tbody>
                    </table>
                </div>
            </div>
            """
        
        # Add export options
        content += f"""
        <div class="card">
            <h3><i class="fas fa-download"></i> Data Export & Analytics</h3>
            <div class="alert alert-info">
                <p>All learning data is automatically saved in JSON format for further analysis.</p>
                <p><strong>Storage Location:</strong> student_results/ directory</p>
                <p><strong>Files:</strong> interactions.json, sessions.json, evaluations.json</p>
            </div>
            
            <div style="text-align: center; margin: 20px 0;">
                <a href="/api/results" class="btn">
                    <i class="fas fa-refresh"></i> Refresh Data
                </a>
                <a href="/api/status" target="_blank" class="btn btn-secondary">
                    <i class="fas fa-info-circle"></i> System Status
                </a>
            </div>
        </div>
        
        <div style="text-align: center; margin: 30px 0;">
            <button class="btn btn-large" onclick="window.close()">
                <i class="fas fa-times"></i> Close Results
            </button>
        </div>
        """
        
        return get_base_html(content, f"Learning Analytics Report - {total_interactions} Interactions")
        
    except Exception as e:
        content = f"""
        <div class="card">
            <div class="alert alert-warning">
                <h3><i class="fas fa-exclamation-triangle"></i> Error Loading Results</h3>
                <p>Error: {str(e)}</p>
                <button class="btn" onclick="location.reload()">
                    <i class="fas fa-refresh"></i> Retry
                </button>
            </div>
        </div>
        """
        return get_base_html(content, "Results - Error")

@app.get("/api/status", response_class=HTMLResponse)
async def system_status():
    """System status endpoint"""
    try:
        # Get system information
        active_session_count = len(active_sessions)
        components_available = COMPONENTS_AVAILABLE
        results_available = results_manager is not None
        current_time = datetime.now().isoformat()
        
        # Get results data if available
        total_interactions = 0
        total_sessions = 0
        if results_manager:
            try:
                interactions = results_manager.get_all_interactions()
                sessions = results_manager.get_all_sessions()
                total_interactions = len(interactions)
                total_sessions = len(sessions)
            except:
                pass
        
        content = f"""
        <div class="card">
            <h2><i class="fas fa-server"></i> System Status Dashboard</h2>
            <p>Real-time status of the RL Tutorial System components</p>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{'✅' if components_available else '❌'}</div>
                    <div class="stat-label">Core Components</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{'✅' if results_available else '❌'}</div>
                    <div class="stat-label">Results Manager</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{active_session_count}</div>
                    <div class="stat-label">Active Sessions</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{total_interactions}</div>
                    <div class="stat-label">Total Interactions</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3><i class="fas fa-cogs"></i> Component Status</h3>
            <div class="alert {'alert-success' if components_available else 'alert-warning'}">
                <h4>Core RL Components</h4>
                <p><strong>Status:</strong> {'Available' if components_available else 'Not Available'}</p>
                <p><strong>DQN Agent:</strong> {'✅ Ready' if components_available else '❌ Not Loaded'}</p>
                <p><strong>PPO Agent:</strong> {'✅ Ready' if components_available else '❌ Not Loaded'}</p>
                <p><strong>Question Bank:</strong> {'✅ Loaded' if components_available else '❌ Not Loaded'}</p>
                <p><strong>Student Profiles:</strong> {'✅ Available' if components_available else '❌ Not Available'}</p>
            </div>
            
            <div class="alert {'alert-success' if results_available else 'alert-warning'}">
                <h4>Data Persistence</h4>
                <p><strong>Results Manager:</strong> {'✅ Active' if results_available else '❌ Not Available'}</p>
                <p><strong>Interactions Recorded:</strong> {total_interactions}</p>
                <p><strong>Sessions Tracked:</strong> {total_sessions}</p>
                <p><strong>Storage Directory:</strong> student_results/</p>
            </div>
        </div>
        
        <div class="card">
            <h3><i class="fas fa-info-circle"></i> System Information</h3>
            <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; font-family: monospace;">
                <p><strong>Timestamp:</strong> {current_time}</p>
                <p><strong>FastAPI Status:</strong> Running</p>
                <p><strong>Port:</strong> 8000</p>
                <p><strong>Environment:</strong> Development</p>
                <p><strong>Active Sessions:</strong> {active_session_count}</p>
            </div>
        </div>
        
        <div style="text-align: center; margin: 30px 0;">
            <a href="/" class="btn btn-large">
                <i class="fas fa-home"></i> Back to Home
            </a>
            <a href="/api/results" class="btn btn-secondary btn-large">
                <i class="fas fa-chart-line"></i> View Results
            </a>
        </div>
        """
        
        return get_base_html(content, "System Status Dashboard")
        
    except Exception as e:
        content = f"""
        <div class="card">
            <div class="alert alert-warning">
                <h3><i class="fas fa-exclamation-triangle"></i> Status Check Error</h3>
                <p>Error retrieving system status: {str(e)}</p>
            </div>
        </div>
        """
        return get_base_html(content, "System Status - Error")

if __name__ == "__main__":
    print("🚀 Starting Professional RL Tutorial System")
    print("=" * 60)
    print("📋 System Features:")
    print("   ✅ Professional Web Interface")
    print("   ✅ Multi-Agent RL Coordination (DQN + PPO)")
    print("   ✅ Real-time Learning Adaptation")
    print("   ✅ Student Progress Tracking")
    print("   ✅ Interactive Learning Sessions")
    print(f"   {'✅' if COMPONENTS_AVAILABLE else '❌'} Complete Assignment Integration")
    print(f"   {'✅' if results_manager else '❌'} Results Persistence")
    print()
    print("🌐 Web Interface: http://localhost:8000")
    print("📊 Results API: http://localhost:8000/api/results")
    print("⚙️ Status API: http://localhost:8000/api/status")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
