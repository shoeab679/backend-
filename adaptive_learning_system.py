import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import os

class AdaptiveLearningSystem:
    """
    Adaptive Learning System for EduSaarthi
    
    This class implements the core adaptive learning algorithms that personalize
    content and learning paths based on student performance, learning style,
    and engagement patterns.
    """
    
    def __init__(self, data_path=None, model_path=None):
        """
        Initialize the adaptive learning system
        
        Parameters:
        -----------
        data_path : str
            Path to load historical learning data (optional)
        model_path : str
            Path to load pre-trained models (optional)
        """
        self.user_profiles = {}
        self.content_profiles = {}
        self.learning_paths = {}
        self.difficulty_levels = ['beginner', 'intermediate', 'advanced']
        
        # Load models if paths provided
        if model_path and os.path.exists(model_path):
            self.load_models(model_path)
        
        # Load data if path provided
        if data_path and os.path.exists(data_path):
            self.load_data(data_path)
    
    def load_data(self, data_path):
        """Load historical learning data"""
        try:
            # Load user profiles
            user_profiles_path = os.path.join(data_path, 'user_profiles.pkl')
            if os.path.exists(user_profiles_path):
                self.user_profiles = joblib.load(user_profiles_path)
            
            # Load content profiles
            content_profiles_path = os.path.join(data_path, 'content_profiles.pkl')
            if os.path.exists(content_profiles_path):
                self.content_profiles = joblib.load(content_profiles_path)
            
            # Load learning paths
            learning_paths_path = os.path.join(data_path, 'learning_paths.pkl')
            if os.path.exists(learning_paths_path):
                self.learning_paths = joblib.load(learning_paths_path)
                
            print(f"Data loaded successfully from {data_path}")
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def load_models(self, model_path):
        """Load pre-trained models"""
        try:
            # Load clustering model
            clustering_model_path = os.path.join(model_path, 'user_clustering_model.pkl')
            if os.path.exists(clustering_model_path):
                self.clustering_model = joblib.load(clustering_model_path)
            
            # Load recommendation model
            recommendation_model_path = os.path.join(model_path, 'recommendation_model.pkl')
            if os.path.exists(recommendation_model_path):
                self.recommendation_model = joblib.load(recommendation_model_path)
                
            print(f"Models loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def save_models(self, model_path):
        """Save trained models"""
        try:
            os.makedirs(model_path, exist_ok=True)
            
            # Save clustering model
            if hasattr(self, 'clustering_model'):
                clustering_model_path = os.path.join(model_path, 'user_clustering_model.pkl')
                joblib.dump(self.clustering_model, clustering_model_path)
            
            # Save recommendation model
            if hasattr(self, 'recommendation_model'):
                recommendation_model_path = os.path.join(model_path, 'recommendation_model.pkl')
                joblib.dump(self.recommendation_model, recommendation_model_path)
                
            print(f"Models saved successfully to {model_path}")
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def create_user_profile(self, user_id, initial_data=None):
        """
        Create a new user profile or update existing one
        
        Parameters:
        -----------
        user_id : str
            Unique identifier for the user
        initial_data : dict
            Initial data about the user (optional)
        """
        # Initialize default profile
        default_profile = {
            'user_id': user_id,
            'learning_style': None,  # Will be determined after sufficient data
            'knowledge_levels': {},  # Subject/topic -> level mapping
            'strengths': [],         # List of strong subjects/topics
            'weaknesses': [],        # List of weak subjects/topics
            'engagement_patterns': {
                'time_of_day': [],   # When user typically studies
                'session_duration': [],  # How long user typically studies
                'days_active': []    # Which days user is active
            },
            'performance_history': {},  # Subject/topic -> [scores]
            'quiz_attempts': {},     # Quiz ID -> attempts data
            'content_interactions': {},  # Content ID -> interaction data
            'recommendations_history': [],  # Previous recommendations
            'cluster': None          # Learning style cluster (determined later)
        }
        
        # If user already exists, update profile
        if user_id in self.user_profiles:
            if initial_data:
                for key, value in initial_data.items():
                    if key in self.user_profiles[user_id]:
                        if isinstance(self.user_profiles[user_id][key], dict) and isinstance(value, dict):
                            self.user_profiles[user_id][key].update(value)
                        else:
                            self.user_profiles[user_id][key] = value
            return self.user_profiles[user_id]
        
        # Create new profile
        profile = default_profile
        if initial_data:
            for key, value in initial_data.items():
                if key in profile:
                    profile[key] = value
        
        self.user_profiles[user_id] = profile
        return profile
    
    def update_user_performance(self, user_id, subject, topic, score, time_spent=None, attempts=None):
        """
        Update user performance data
        
        Parameters:
        -----------
        user_id : str
            Unique identifier for the user
        subject : str
            Subject identifier
        topic : str
            Topic identifier
        score : float
            Score achieved (0-100)
        time_spent : float
            Time spent in minutes (optional)
        attempts : int
            Number of attempts (optional)
        """
        # Ensure user profile exists
        if user_id not in self.user_profiles:
            self.create_user_profile(user_id)
        
        # Initialize subject if not exists
        if subject not in self.user_profiles[user_id]['performance_history']:
            self.user_profiles[user_id]['performance_history'][subject] = {}
        
        # Initialize topic if not exists
        if topic not in self.user_profiles[user_id]['performance_history'][subject]:
            self.user_profiles[user_id]['performance_history'][subject][topic] = {
                'scores': [],
                'times': [],
                'attempts': [],
                'timestamps': []
            }
        
        # Update performance data
        self.user_profiles[user_id]['performance_history'][subject][topic]['scores'].append(score)
        if time_spent:
            self.user_profiles[user_id]['performance_history'][subject][topic]['times'].append(time_spent)
        if attempts:
            self.user_profiles[user_id]['performance_history'][subject][topic]['attempts'].append(attempts)
        
        # Add timestamp
        import datetime
        self.user_profiles[user_id]['performance_history'][subject][topic]['timestamps'].append(
            datetime.datetime.now().isoformat()
        )
        
        # Update knowledge level for this topic
        self._update_knowledge_level(user_id, subject, topic)
        
        # Update strengths and weaknesses
        self._update_strengths_weaknesses(user_id)
        
        return self.user_profiles[user_id]
    
    def _update_knowledge_level(self, user_id, subject, topic):
        """Update knowledge level based on performance history"""
        if user_id not in self.user_profiles:
            return
        
        # Get performance history for this topic
        if subject in self.user_profiles[user_id]['performance_history'] and \
           topic in self.user_profiles[user_id]['performance_history'][subject]:
            
            scores = self.user_profiles[user_id]['performance_history'][subject][topic]['scores']
            
            # Need at least 2 data points to determine level
            if len(scores) < 2:
                level = 'beginner'  # Default for new users
            else:
                avg_score = sum(scores[-3:]) / min(len(scores), 3)  # Average of last 3 attempts
                
                if avg_score >= 80:
                    level = 'advanced'
                elif avg_score >= 60:
                    level = 'intermediate'
                else:
                    level = 'beginner'
            
            # Initialize knowledge_levels for subject if not exists
            if subject not in self.user_profiles[user_id]['knowledge_levels']:
                self.user_profiles[user_id]['knowledge_levels'][subject] = {}
            
            # Update knowledge level
            self.user_profiles[user_id]['knowledge_levels'][subject][topic] = level
    
    def _update_strengths_weaknesses(self, user_id):
        """Update user strengths and weaknesses based on performance"""
        if user_id not in self.user_profiles:
            return
        
        strengths = []
        weaknesses = []
        
        # Analyze each subject and topic
        for subject, topics in self.user_profiles[user_id]['performance_history'].items():
            for topic, data in topics.items():
                if len(data['scores']) >= 2:  # Need at least 2 data points
                    avg_score = sum(data['scores']) / len(data['scores'])
                    
                    if avg_score >= 75:
                        strengths.append({'subject': subject, 'topic': topic, 'score': avg_score})
                    elif avg_score <= 50:
                        weaknesses.append({'subject': subject, 'topic': topic, 'score': avg_score})
        
        # Sort by score
        strengths.sort(key=lambda x: x['score'], reverse=True)
        weaknesses.sort(key=lambda x: x['score'])
        
        # Keep top 5 strengths and weaknesses
        self.user_profiles[user_id]['strengths'] = strengths[:5]
        self.user_profiles[user_id]['weaknesses'] = weaknesses[:5]
    
    def cluster_users(self, min_data_points=10):
        """
        Cluster users based on learning patterns and performance
        
        Parameters:
        -----------
        min_data_points : int
            Minimum number of data points required for clustering
        """
        # Extract features for clustering
        user_features = []
        user_ids = []
        
        for user_id, profile in self.user_profiles.items():
            # Skip users with insufficient data
            total_interactions = sum(len(topic_data['scores']) 
                                    for subject_data in profile['performance_history'].values() 
                                    for topic_data in subject_data.values())
            
            if total_interactions < min_data_points:
                continue
            
            # Extract features
            features = self._extract_user_features(profile)
            if features:
                user_features.append(features)
                user_ids.append(user_id)
        
        # If not enough users for clustering
        if len(user_features) < 3:
            print("Not enough users with sufficient data for clustering")
            return
        
        # Convert to numpy array
        X = np.array(user_features)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine optimal number of clusters
        best_k = 3  # Default
        best_score = -1
        
        for k in range(2, min(6, len(user_features))):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            if len(set(cluster_labels)) < k:  # Skip if some clusters are empty
                continue
                
            score = silhouette_score(X_scaled, cluster_labels)
            if score > best_score:
                best_score = score
                best_k = k
        
        # Final clustering with optimal k
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Save clustering model
        self.clustering_model = {
            'kmeans': kmeans,
            'scaler': scaler
        }
        
        # Update user profiles with cluster assignments
        for i, user_id in enumerate(user_ids):
            self.user_profiles[user_id]['cluster'] = int(cluster_labels[i])
        
        # Analyze clusters to determine learning styles
        self._analyze_learning_styles(X_scaled, cluster_labels, user_ids)
        
        return {
            'num_clusters': best_k,
            'silhouette_score': best_score,
            'cluster_distribution': np.bincount(cluster_labels).tolist()
        }
    
    def _extract_user_features(self, profile):
        """Extract features from user profile for clustering"""
        features = []
        
        # Calculate average score across all subjects/topics
        all_scores = []
        for subject_data in profile['performance_history'].values():
            for topic_data in subject_data.values():
                all_scores.extend(topic_data['scores'])
        
        if not all_scores:
            return None
        
        # Average score
        avg_score = sum(all_scores) / len(all_scores)
        features.append(avg_score)
        
        # Score variance (learning consistency)
        if len(all_scores) > 1:
            score_variance = np.var(all_scores)
        else:
            score_variance = 0
        features.append(score_variance)
        
        # Average attempts per topic
        all_attempts = []
        for subject_data in profile['performance_history'].values():
            for topic_data in subject_data.values():
                if 'attempts' in topic_data and topic_data['attempts']:
                    all_attempts.extend(topic_data['attempts'])
        
        if all_attempts:
            avg_attempts = sum(all_attempts) / len(all_attempts)
        else:
            avg_attempts = 1  # Default
        features.append(avg_attempts)
        
        # Average time spent per topic
        all_times = []
        for subject_data in profile['performance_history'].values():
            for topic_data in subject_data.values():
                if 'times' in topic_data and topic_data['times']:
                    all_times.extend(topic_data['times'])
        
        if all_times:
            avg_time = sum(all_times) / len(all_times)
        else:
            avg_time = 0  # Default
        features.append(avg_time)
        
        # Number of subjects/topics attempted
        num_topics = sum(len(subject_data) for subject_data in profile['performance_history'].values())
        features.append(num_topics)
        
        # Breadth vs depth: ratio of topics to average attempts
        if avg_attempts > 0:
            breadth_depth_ratio = num_topics / avg_attempts
        else:
            breadth_depth_ratio = 0
        features.append(breadth_depth_ratio)
        
        return features
    
    def _analyze_learning_styles(self, X_scaled, cluster_labels, user_ids):
        """Analyze clusters to determine learning styles"""
        # Calculate cluster centers
        cluster_centers = {}
        for i in range(max(cluster_labels) + 1):
            cluster_points = X_scaled[cluster_labels == i]
            if len(cluster_points) > 0:
                cluster_centers[i] = np.mean(cluster_points, axis=0)
        
        # Define learning styles based on cluster characteristics
        learning_styles = {
            'visual_learner': 'Learns best through visual aids, diagrams, and charts',
            'auditory_learner': 'Learns best through listening and verbal explanations',
            'reading_writing_learner': 'Learns best through reading and writing activities',
            'kinesthetic_learner': 'Learns best through hands-on activities and practice',
            'logical_learner': 'Learns best through logical reasoning and systematic approaches',
            'social_learner': 'Learns best through group activities and discussions',
            'solitary_learner': 'Learns best through independent study and self-reflection'
        }
        
        # Assign learning styles to clusters based on characteristics
        # This is a simplified approach; in a real system, this would be more sophisticated
        cluster_styles = {}
        
        for cluster_id, center in cluster_centers.items():
            # Analyze center to determine dominant characteristics
            # For this demo, we'll assign styles based on cluster ID
            if cluster_id == 0:
                style = 'visual_learner'
            elif cluster_id == 1:
                style = 'reading_writing_learner'
            elif cluster_id == 2:
                style = 'logical_learner'
            elif cluster_id == 3:
                style = 'kinesthetic_learner'
            elif cluster_id == 4:
                style = 'auditory_learner'
            else:
                style = 'social_learner'
            
            cluster_styles[cluster_id] = style
        
        # Update user profiles with learning styles
        for i, user_id in enumerate(user_ids):
            cluster = int(cluster_labels[i])
            if cluster in cluster_styles:
                self.user_profiles[user_id]['learning_style'] = cluster_styles[cluster]
    
    def predict_user_cluster(self, user_id):
        """
        Predict cluster for a user
        
        Parameters:
        -----------
        user_id : str
            Unique identifier for the user
        
        Returns:
        --------
        int or None
            Predicted cluster ID or None if prediction not possible
        """
        if user_id not in self.user_profiles:
            return None
        
        # If user already has a cluster assigned
        if self.user_profiles[user_id]['cluster'] is not None:
            return self.user_profiles[user_id]['cluster']
        
        # If no clustering model available
        if not hasattr(self, 'clustering_model'):
            return None
        
        # Extract features
        features = self._extract_user_features(self.user_profiles[user_id])
        if not features:
            return None
        
        # Scale features
        X = np.array([features])
        X_scaled = self.clustering_model['scaler'].transform(X)
        
        # Predict cluster
        cluster = self.clustering_model['kmeans'].predict(X_scaled)[0]
        
        # Update user profile
        self.user_profiles[user_id]['cluster'] = int(cluster)
        
        return int(cluster)
    
    def recommend_content(self, user_id, subject=None, topic=None, count=5):
        """
        Recommend content for a user
        
        Parameters:
        -----------
        user_id : str
            Unique identifier for the user
        subject : str
            Subject to filter recommendations (optional)
        topic : str
            Topic to filter recommendations (optional)
        count : int
            Number of recommendations to return
        
        Returns:
        --------
        list
            List of recommended content IDs
        """
        if user_id not in self.user_profiles:
            return []
        
        # Get user profile
        profile = self.user_profiles[user_id]
        
        # Determine appropriate difficulty level
        if subject and topic:
            difficulty = self._get_appropriate_difficulty(user_id, subject, topic)
        else:
            difficulty = 'beginner'  # Default
        
        # Get learning style
        learning_style = profile.get('learning_style')
        
        # Filter content based on criteria
        filtered_content = []
        for content_id, content in self.content_profiles.items():
            # Filter by subject if specified
            if subject and content.get('subject') != subject:
                continue
            
            # Filter by topic if specified
            if topic and content.get('topic') != topic:
                continue
            
            # Check if content matches user's difficulty level
            if content.get('difficulty') == difficulty:
                # Calculate relevance score
                relevance = self._calculate_content_relevance(content, profile, learning_style)
                filtered_content.append((content_id, relevance))
        
        # Sort by relevance
        filtered_content.sort(key=lambda x: x[1], reverse=True)
        
        # Get top recommendations
        recommendations = [content_id for content_id, _ in filtered_content[:count]]
        
        # Update recommendations history
        profile['recommendations_history'].append({
            'timestamp': datetime.datetime.now().isoformat(),
            'recommendations': recommendations,
            'subject': subject,
            'topic': topic
        })
        
        return recommendations
    
    def _get_appropriate_difficulty(self, user_id, subject, topic):
        """Determine appropriate difficulty level for a user on a specific topic"""
        if user_id not in self.user_profiles:
            return 'beginner'
        
        # Check if user has knowledge level for this topic
        if subject in self.user_profiles[user_id]['knowledge_levels'] and \
           topic in self.user_profiles[user_id]['knowledge_levels'][subject]:
            return self.user_profiles[user_id]['knowledge_levels'][subject][topic]
        
        return 'beginner'  # Default
    
    def _calculate_content_relevance(self, content, profile, learning_style):
        """Calculate relevance score for content based on user profile"""
        relevance = 1.0  # Base relevance
        
        # Adjust based on learning style match
        if learning_style and content.get('learning_style') == learning_style:
            relevance *= 1.5
        
        # Adjust based on strengths/weaknesses
        subject = content.get('subject')
        topic = content.get('topic')
        
        # Check if content is in user's weaknesses (prioritize)
        for weakness in profile['weaknesses']:
            if weakness['subject'] == subject and weakness['topic'] == topic:
                relevance *= 2.0
                break
        
        # Adjust based on previous interactions
        if subject in profile['performance_history'] and topic in profile['performance_history'][subject]:
            # If user has struggled with this topic, prioritize it
            scores = profile['performance_history'][subject][topic]['scores']
            if scores and sum(scores) / len(scores) < 60:
                relevance *= 1.5
        
        return relevance
    
    def generate_learning_path(self, user_id, subject, goal=None):
        """
        Generate personalized learning path for a user
        
        Parameters:
        -----------
        user_id : str
            Unique identifier for the user
        subject : str
            Subject for the learning path
        goal : str
            Specific goal or target (optional)
        
        Returns:
        --------
        dict
            Learning path with ordered topics and content
        """
        if user_id not in self.user_profiles:
            return None
        
        # Get user profile
        profile = self.user_profiles[user_id]
        
        # Get all topics for this subject
        topics = set()
        for content_id, content in self.content_profiles.items():
            if content.get('subject') == subject:
                topics.add(content.get('topic'))
        
        # Sort topics based on dependencies and user knowledge
        ordered_topics = self._order_topics(topics, subject, profile)
        
        # Generate path with content for each topic
        path = {
            'user_id': user_id,
            'subject': subject,
            'goal': goal,
            'created_at': datetime.datetime.now().isoformat(),
            'topics': []
        }
        
        for topic in ordered_topics:
            # Determine appropriate difficulty
            difficulty = self._get_appropriate_difficulty(user_id, subject, topic)
            
            # Get content for this topic
            topic_content = []
            for content_id, content in self.content_profiles.items():
                if content.get('subject') == subject and content.get('topic') == topic:
                    if content.get('difficulty') == difficulty:
                        topic_content.append(content_id)
            
            # Add topic to path
            path['topics'].append({
                'topic': topic,
                'difficulty': difficulty,
                'content': topic_content,
                'completed': False
            })
        
        # Save learning path
        path_id = f"{user_id}_{subject}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.learning_paths[path_id] = path
        
        return path
    
    def _order_topics(self, topics, subject, profile):
        """Order topics based on dependencies and user knowledge"""
        # This is a simplified implementation
        # In a real system, this would consider topic dependencies and prerequisites
        
        # Convert to list
        topics_list = list(topics)
        
        # Get user knowledge levels for these topics
        knowledge = {}
        for topic in topics_list:
            if subject in profile['knowledge_levels'] and topic in profile['knowledge_levels'][subject]:
                level = profile['knowledge_levels'][subject][topic]
                knowledge[topic] = self.difficulty_levels.index(level)
            else:
                knowledge[topic] = 0  # Beginner
        
        # Sort topics by knowledge level (ascending)
        topics_list.sort(key=lambda x: knowledge.get(x, 0))
        
        return topics_list
    
    def update_learning_path_progress(self, path_id, topic, completed=True):
        """
        Update progress on a learning path
        
        Parameters:
        -----------
        path_id : str
            Unique identifier for the learning path
        topic : str
            Topic to update
        completed : bool
            Whether the topic is completed
        
        Returns:
        --------
        dict
            Updated learning path
        """
        if path_id not in self.learning_paths:
            return None
        
        path = self.learning_paths[path_id]
        
        # Update topic completion status
        for topic_data in path['topics']:
            if topic_data['topic'] == topic:
                topic_data['completed'] = completed
                break
        
        return path
    
    def get_next_content(self, user_id, path_id):
        """
        Get next content item in a learning path
        
        Parameters:
        -----------
        user_id : str
            Unique identifier for the user
        path_id : str
            Unique identifier for the learning path
        
        Returns:
        --------
        str
            Content ID for next item
        """
        if path_id not in self.learning_paths:
            return None
        
        path = self.learning_paths[path_id]
        
        # Find first uncompleted topic
        for topic_data in path['topics']:
            if not topic_data['completed'] and topic_data['content']:
                # Return first content item for this topic
                return topic_data['content'][0]
        
        return None  # All topics completed
