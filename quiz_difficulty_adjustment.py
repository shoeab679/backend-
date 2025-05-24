import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os
import datetime
import json
import random

class QuizDifficultyAdjustment:
    """
    Quiz Difficulty Adjustment System for EduSaarthi
    
    This class implements an adaptive algorithm that dynamically adjusts
    quiz difficulty based on student performance, learning pace, and
    historical data to provide an optimal learning challenge.
    """
    
    def __init__(self, data_path=None, model_path=None):
        """
        Initialize the quiz difficulty adjustment system
        
        Parameters:
        -----------
        data_path : str
            Path to load historical quiz data (optional)
        model_path : str
            Path to load pre-trained models (optional)
        """
        self.user_performance = {}
        self.question_bank = {}
        self.difficulty_levels = ['easy', 'medium', 'hard', 'expert']
        self.difficulty_scores = {'easy': 1, 'medium': 2, 'hard': 3, 'expert': 4}
        self.models = {}
        
        # Load models if path provided
        if model_path and os.path.exists(model_path):
            self.load_models(model_path)
        
        # Load data if path provided
        if data_path and os.path.exists(data_path):
            self.load_data(data_path)
    
    def load_data(self, data_path):
        """Load historical quiz data"""
        try:
            # Load user performance data
            performance_path = os.path.join(data_path, 'user_performance.json')
            if os.path.exists(performance_path):
                with open(performance_path, 'r') as f:
                    self.user_performance = json.load(f)
            
            # Load question bank
            question_bank_path = os.path.join(data_path, 'question_bank.json')
            if os.path.exists(question_bank_path):
                with open(question_bank_path, 'r') as f:
                    self.question_bank = json.load(f)
                    
            print(f"Data loaded successfully from {data_path}")
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def load_models(self, model_path):
        """Load pre-trained difficulty adjustment models"""
        try:
            # Load subject-specific models
            subjects = ['mathematics', 'science', 'english', 'hindi', 'social_science']
            
            for subject in subjects:
                model_file = os.path.join(model_path, f"{subject}_difficulty_model.pkl")
                if os.path.exists(model_file):
                    self.models[subject] = joblib.load(model_file)
            
            print(f"Models loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def save_models(self, model_path):
        """Save trained difficulty adjustment models"""
        try:
            os.makedirs(model_path, exist_ok=True)
            
            for subject, model in self.models.items():
                model_file = os.path.join(model_path, f"{subject}_difficulty_model.pkl")
                joblib.dump(model, model_file)
                
            print(f"Models saved successfully to {model_path}")
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def save_data(self, data_path):
        """Save quiz data"""
        try:
            os.makedirs(data_path, exist_ok=True)
            
            # Save user performance data
            performance_path = os.path.join(data_path, 'user_performance.json')
            with open(performance_path, 'w') as f:
                json.dump(self.user_performance, f, indent=2)
            
            # Save question bank
            question_bank_path = os.path.join(data_path, 'question_bank.json')
            with open(question_bank_path, 'w') as f:
                json.dump(self.question_bank, f, indent=2)
                
            print(f"Data saved successfully to {data_path}")
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def add_question(self, question_id, subject, topic, difficulty, question_data):
        """
        Add a question to the question bank
        
        Parameters:
        -----------
        question_id : str
            Unique identifier for the question
        subject : str
            Subject category
        topic : str
            Topic within the subject
        difficulty : str
            Difficulty level ('easy', 'medium', 'hard', 'expert')
        question_data : dict
            Question content and metadata
        
        Returns:
        --------
        bool
            Success status
        """
        if difficulty not in self.difficulty_levels:
            return False
        
        # Initialize subject if not exists
        if subject not in self.question_bank:
            self.question_bank[subject] = {}
        
        # Initialize topic if not exists
        if topic not in self.question_bank[subject]:
            self.question_bank[subject][topic] = {}
        
        # Initialize difficulty if not exists
        if difficulty not in self.question_bank[subject][topic]:
            self.question_bank[subject][topic][difficulty] = {}
        
        # Add question
        self.question_bank[subject][topic][difficulty][question_id] = question_data
        
        return True
    
    def record_question_attempt(self, user_id, question_id, correct, time_taken=None):
        """
        Record a user's attempt at a question
        
        Parameters:
        -----------
        user_id : str
            Unique identifier for the user
        question_id : str
            Unique identifier for the question
        correct : bool
            Whether the answer was correct
        time_taken : float
            Time taken to answer in seconds (optional)
        
        Returns:
        --------
        bool
            Success status
        """
        # Find question in question bank
        question_info = None
        question_subject = None
        question_topic = None
        question_difficulty = None
        
        for subject, topics in self.question_bank.items():
            for topic, difficulties in topics.items():
                for difficulty, questions in difficulties.items():
                    if question_id in questions:
                        question_info = questions[question_id]
                        question_subject = subject
                        question_topic = topic
                        question_difficulty = difficulty
                        break
                if question_info:
                    break
            if question_info:
                break
        
        if not question_info:
            return False
        
        # Initialize user if not exists
        if user_id not in self.user_performance:
            self.user_performance[user_id] = {}
        
        # Initialize subject if not exists
        if question_subject not in self.user_performance[user_id]:
            self.user_performance[user_id][question_subject] = {}
        
        # Initialize topic if not exists
        if question_topic not in self.user_performance[user_id][question_subject]:
            self.user_performance[user_id][question_subject][question_topic] = {
                'attempts': [],
                'correct_count': 0,
                'total_count': 0,
                'streak': 0,
                'current_difficulty': 'easy',
                'difficulty_history': []
            }
        
        # Update performance data
        user_topic_data = self.user_performance[user_id][question_subject][question_topic]
        user_topic_data['total_count'] += 1
        
        if correct:
            user_topic_data['correct_count'] += 1
            user_topic_data['streak'] += 1
        else:
            user_topic_data['streak'] = 0
        
        # Record attempt
        attempt = {
            'question_id': question_id,
            'difficulty': question_difficulty,
            'correct': correct,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        if time_taken is not None:
            attempt['time_taken'] = time_taken
        
        user_topic_data['attempts'].append(attempt)
        
        # Update difficulty level if needed
        self._update_difficulty_level(user_id, question_subject, question_topic)
        
        return True
    
    def _update_difficulty_level(self, user_id, subject, topic):
        """Update difficulty level based on performance"""
        if user_id not in self.user_performance:
            return
        
        if subject not in self.user_performance[user_id]:
            return
        
        if topic not in self.user_performance[user_id][subject]:
            return
        
        user_topic_data = self.user_performance[user_id][subject][topic]
        
        # Need at least 5 attempts to adjust difficulty
        if user_topic_data['total_count'] < 5:
            return
        
        # Calculate success rate
        success_rate = user_topic_data['correct_count'] / user_topic_data['total_count']
        
        # Get current difficulty
        current_difficulty = user_topic_data['current_difficulty']
        current_difficulty_index = self.difficulty_levels.index(current_difficulty)
        
        # Determine if difficulty should change
        new_difficulty_index = current_difficulty_index
        
        # If success rate is high and on a streak, increase difficulty
        if success_rate > 0.7 and user_topic_data['streak'] >= 3:
            new_difficulty_index = min(current_difficulty_index + 1, len(self.difficulty_levels) - 1)
        
        # If success rate is low, decrease difficulty
        elif success_rate < 0.4:
            new_difficulty_index = max(current_difficulty_index - 1, 0)
        
        # Update difficulty if changed
        if new_difficulty_index != current_difficulty_index:
            new_difficulty = self.difficulty_levels[new_difficulty_index]
            user_topic_data['current_difficulty'] = new_difficulty
            
            # Record difficulty change
            user_topic_data['difficulty_history'].append({
                'from': current_difficulty,
                'to': new_difficulty,
                'timestamp': datetime.datetime.now().isoformat(),
                'reason': 'success_rate' if success_rate < 0.4 or success_rate > 0.7 else 'streak'
            })
    
    def get_user_difficulty(self, user_id, subject, topic):
        """
        Get current difficulty level for a user on a specific topic
        
        Parameters:
        -----------
        user_id : str
            Unique identifier for the user
        subject : str
            Subject category
        topic : str
            Topic within the subject
        
        Returns:
        --------
        str
            Difficulty level
        """
        if user_id not in self.user_performance:
            return 'easy'  # Default for new users
        
        if subject not in self.user_performance[user_id]:
            return 'easy'
        
        if topic not in self.user_performance[user_id][subject]:
            return 'easy'
        
        return self.user_performance[user_id][subject][topic]['current_difficulty']
    
    def select_questions(self, user_id, subject, topic, count=5):
        """
        Select questions for a quiz based on user's current difficulty level
        
        Parameters:
        -----------
        user_id : str
            Unique identifier for the user
        subject : str
            Subject category
        topic : str
            Topic within the subject
        count : int
            Number of questions to select
        
        Returns:
        --------
        list
            List of selected question IDs
        """
        # Get user's current difficulty level
        difficulty = self.get_user_difficulty(user_id, subject, topic)
        
        # Check if subject and topic exist in question bank
        if subject not in self.question_bank:
            return []
        
        if topic not in self.question_bank[subject]:
            return []
        
        # Get questions at the appropriate difficulty level
        questions = []
        
        # Try to get questions at the exact difficulty level
        if difficulty in self.question_bank[subject][topic]:
            questions = list(self.question_bank[subject][topic][difficulty].keys())
        
        # If not enough questions, include questions from adjacent difficulty levels
        if len(questions) < count:
            difficulty_index = self.difficulty_levels.index(difficulty)
            
            # Try easier questions first
            if difficulty_index > 0:
                easier_difficulty = self.difficulty_levels[difficulty_index - 1]
                if easier_difficulty in self.question_bank[subject][topic]:
                    questions.extend(list(self.question_bank[subject][topic][easier_difficulty].keys()))
            
            # Then try harder questions
            if difficulty_index < len(self.difficulty_levels) - 1:
                harder_difficulty = self.difficulty_levels[difficulty_index + 1]
                if harder_difficulty in self.question_bank[subject][topic]:
                    questions.extend(list(self.question_bank[subject][topic][harder_difficulty].keys()))
        
        # Get previously attempted questions
        attempted_questions = set()
        if user_id in self.user_performance and subject in self.user_performance[user_id] and topic in self.user_performance[user_id][subject]:
            for attempt in self.user_performance[user_id][subject][topic]['attempts']:
                attempted_questions.add(attempt['question_id'])
        
        # Prioritize questions that haven't been attempted yet
        unattempted = [q for q in questions if q not in attempted_questions]
        
        # If we have enough unattempted questions, use those
        if len(unattempted) >= count:
            random.shuffle(unattempted)
            return unattempted[:count]
        
        # Otherwise, include some previously attempted questions
        random.shuffle(questions)
        selected = unattempted
        
        for q in questions:
            if q not in selected:
                selected.append(q)
                if len(selected) >= count:
                    break
        
        return selected[:count]
    
    def train_difficulty_model(self, subject):
        """
        Train a machine learning model to predict optimal difficulty level
        
        Parameters:
        -----------
        subject : str
            Subject to train model for
        
        Returns:
        --------
        bool
            Success status
        """
        # Collect training data
        X = []  # Features
        y = []  # Target difficulty levels
        
        for user_id, user_data in self.user_performance.items():
            if subject in user_data:
                for topic, topic_data in user_data[subject].items():
                    # Need sufficient data
                    if topic_data['total_count'] >= 10:
                        # Extract features
                        success_rate = topic_data['correct_count'] / topic_data['total_count']
                        
                        # Calculate average time per question if available
                        avg_time = 0
                        time_count = 0
                        for attempt in topic_data['attempts']:
                            if 'time_taken' in attempt:
                                avg_time += attempt['time_taken']
                                time_count += 1
                        
                        if time_count > 0:
                            avg_time /= time_count
                        
                        # Calculate recent performance (last 5 attempts)
                        recent_attempts = topic_data['attempts'][-5:]
                        recent_correct = sum(1 for a in recent_attempts if a['correct'])
                        recent_success_rate = recent_correct / len(recent_attempts) if recent_attempts else 0
                        
                        # Create feature vector
                        features = [
                            success_rate,
                            recent_success_rate,
                            topic_data['streak'],
                            avg_time,
                            topic_data['total_count']
                        ]
                        
                        X.append(features)
                        
                        # Target is the current difficulty level
                        current_difficulty = topic_data['current_difficulty']
                        y.append(self.difficulty_scores[current_difficulty])
        
        # If not enough data, return
        if len(X) < 10:
            print(f"Not enough data to train model for {subject}")
            return False
        
        # Train logistic regression model
        X = np.array(X)
        y = np.array(y)
        
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        model.fit(X, y)
        
        # Save model
        self.models[subject] = model
        
        return True
    
    def predict_optimal_difficulty(self, user_id, subject, topic):
        """
        Predict optimal difficulty level using trained model
        
        Parameters:
        -----------
        user_id : str
            Unique identifier for the user
        subject : str
            Subject category
        topic : str
            Topic within the subject
        
        Returns:
        --------
        str
            Predicted optimal difficulty level
        """
        # If no model for this subject, use rule-based approach
        if subject not in self.models:
            return self.get_user_difficulty(user_id, subject, topic)
        
        # If user has no data for this topic, return default
        if user_id not in self.user_performance:
            return 'easy'
        
        if subject not in self.user_performance[user_id]:
            return 'easy'
        
        if topic not in self.user_performance[user_id][subject]:
            return 'easy'
        
        topic_data = self.user_performance[user_id][subject][topic]
        
        # Extract features
        success_rate = topic_data['correct_count'] / topic_data['total_count'] if topic_data['total_count'] > 0 else 0
        
        # Calculate average time per question if available
        avg_time = 0
        time_count = 0
        for attempt in topic_data['attempts']:
            if 'time_taken' in attempt:
                avg_time += attempt['time_taken']
                time_count += 1
        
        if time_count > 0:
            avg_time /= time_count
        
        # Calculate recent performance (last 5 attempts)
        recent_attempts = topic_data['attempts'][-5:] if topic_data['attempts'] else []
        recent_correct = sum(1 for a in recent_attempts if a['correct'])
        recent_success_rate = recent_correct / len(recent_attempts) if recent_attempts else 0
        
        # Create feature vector
        features = [
            success_rate,
            recent_success_rate,
            topic_data['streak'],
            avg_time,
            topic_data['total_count']
        ]
        
        # Make prediction
        features = np.array([features])
        difficulty_score = self.models[subject].predict(features)[0]
        
        # Convert score to difficulty level
        for level, score in self.difficulty_scores.items():
            if score == difficulty_score:
                return level
        
        # Fallback
        return 'medium'
    
    def generate_adaptive_quiz(self, user_id, subject, topic, count=5):
        """
        Generate an adaptive quiz for a user
        
        Parameters:
        -----------
        user_id : str
            Unique identifier for the user
        subject : str
            Subject category
        topic : str
            Topic within the subject
        count : int
            Number of questions in the quiz
        
        Returns:
        --------
        dict
            Quiz data with questions
        """
        # Select questions
        question_ids = self.select_questions(user_id, subject, topic, count)
        
        # If no questions available, return empty quiz
        if not question_ids:
            return {
                'user_id': user_id,
                'subject': subject,
                'topic': topic,
                'difficulty': 'easy',
                'questions': [],
                'created_at': datetime.datetime.now().isoformat()
            }
        
        # Get difficulty level
        difficulty = self.get_user_difficulty(user_id, subject, topic)
        
        # Collect questions
        questions = []
        for question_id in question_ids:
            # Find question in question bank
            question_data = None
            question_difficulty = None
            
            for diff, qs in self.question_bank[subject][topic].items():
                if question_id in qs:
                    question_data = qs[question_id]
                    question_difficulty = diff
                    break
            
            if question_data:
                # Add question to quiz
                questions.append({
                    'id': question_id,
                    'difficulty': question_difficulty,
                    'data': question_data
                })
        
        # Create quiz
        quiz = {
            'user_id': user_id,
            'subject': subject,
            'topic': topic,
            'difficulty': difficulty,
            'questions': questions,
            'created_at': datetime.datetime.now().isoformat()
        }
        
        return quiz
    
    def analyze_quiz_performance(self, user_id, quiz_results):
        """
        Analyze quiz performance and update user data
        
        Parameters:
        -----------
        user_id : str
            Unique identifier for the user
        quiz_results : dict
            Results of the quiz
        
        Returns:
        --------
        dict
            Performance analysis
        """
        subject = quiz_results.get('subject')
        topic = quiz_results.get('topic')
        questions = quiz_results.get('questions', [])
        
        if not subject or not topic or not questions:
            return {'error': 'Invalid quiz results'}
        
        # Record each question attempt
        for question in questions:
            question_id = question.get('id')
            correct = question.get('correct', False)
            time_taken = question.get('time_taken')
            
            if question_id:
                self.record_question_attempt(user_id, question_id, correct, time_taken)
        
        # Get updated difficulty
        new_difficulty = self.get_user_difficulty(user_id, subject, topic)
        
        # Calculate performance metrics
        total_questions = len(questions)
        correct_answers = sum(1 for q in questions if q.get('correct', False))
        accuracy = correct_answers / total_questions if total_questions > 0 else 0
        
        # Calculate average time if available
        times = [q.get('time_taken') for q in questions if 'time_taken' in q]
        avg_time = sum(times) / len(times) if times else None
        
        # Get user's performance data
        user_data = None
        if user_id in self.user_performance and subject in self.user_performance[user_id] and topic in self.user_performance[user_id][topic]:
            user_data = self.user_performance[user_id][subject][topic]
        
        # Generate analysis
        analysis = {
            'user_id': user_id,
            'subject': subject,
            'topic': topic,
            'accuracy': accuracy,
            'correct_answers': correct_answers,
            'total_questions': total_questions,
            'average_time': avg_time,
            'previous_difficulty': quiz_results.get('difficulty'),
            'new_difficulty': new_difficulty,
            'difficulty_changed': quiz_results.get('difficulty') != new_difficulty,
            'streak': user_data['streak'] if user_data else 0,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        return analysis
