import numpy as np
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import json
import os
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class AiTutor:
    """
    AI Tutor Implementation for EduSaarthi
    
    This class implements the NLP-based AI tutor that can answer questions,
    explain concepts, and provide personalized learning assistance in
    both English and Hindi.
    """
    
    def __init__(self, knowledge_base_path=None, model_path=None):
        """
        Initialize the AI tutor
        
        Parameters:
        -----------
        knowledge_base_path : str
            Path to knowledge base files (optional)
        model_path : str
            Path to pre-trained models (optional)
        """
        self.knowledge_base = {}
        self.sessions = {}
        self.vectorizer = TfidfVectorizer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words_english = set(stopwords.words('english'))
        
        # Hindi stopwords (common Hindi stopwords)
        self.stop_words_hindi = {
            'का', 'के', 'की', 'है', 'में', 'से', 'हैं', 'को', 'पर', 'इस', 'होता', 'कि', 'जो', 'कर', 'मे', 
            'गया', 'करने', 'किया', 'लिये', 'अपने', 'ने', 'बनी', 'नहीं', 'तो', 'ही', 'या', 'एवं', 'दिया', 
            'हो', 'इसका', 'था', 'द्वारा', 'हुआ', 'तक', 'साथ', 'करना', 'वाले', 'बाद', 'लिए', 'आप', 'कुछ', 
            'सकते', 'किसी', 'ये', 'इसके', 'सबसे', 'इसमें', 'थे', 'दो', 'होने', 'वह', 'वे', 'करते', 'बहुत', 
            'कहा', 'वर्ग', 'कई', 'करें', 'होती', 'अपनी', 'उनके', 'थी', 'यदि', 'हुई', 'जा', 'ना', 'इसे', 'कहते', 
            'जब', 'होते', 'कोई', 'हुए', 'व', 'न', 'अभी', 'जैसे', 'सभी', 'करता', 'उनकी', 'तरह', 'उस', 'आदि', 
            'कुल', 'एस', 'रहा', 'इसकी', 'सकता', 'रहे', 'उनका', 'इसी', 'रखें', 'अपना', 'पे', 'उसके'
        }
        
        # Load knowledge base if path provided
        if knowledge_base_path and os.path.exists(knowledge_base_path):
            self.load_knowledge_base(knowledge_base_path)
        
        # Initialize default knowledge base if none loaded
        if not self.knowledge_base:
            self._initialize_default_knowledge_base()
    
    def load_knowledge_base(self, knowledge_base_path):
        """Load knowledge base from files"""
        try:
            # Load subject-specific knowledge bases
            subjects = ['mathematics', 'science', 'english', 'hindi', 'social_science']
            
            for subject in subjects:
                subject_path = os.path.join(knowledge_base_path, f"{subject}_kb.json")
                if os.path.exists(subject_path):
                    with open(subject_path, 'r', encoding='utf-8') as f:
                        self.knowledge_base[subject] = json.load(f)
            
            # Load general knowledge base
            general_path = os.path.join(knowledge_base_path, "general_kb.json")
            if os.path.exists(general_path):
                with open(general_path, 'r', encoding='utf-8') as f:
                    self.knowledge_base['general'] = json.load(f)
            
            print(f"Knowledge base loaded successfully from {knowledge_base_path}")
            
            # Prepare vectorizer with all content
            self._prepare_vectorizer()
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
    
    def _initialize_default_knowledge_base(self):
        """Initialize a minimal default knowledge base"""
        self.knowledge_base = {
            'mathematics': {
                'concepts': {
                    'algebra': {
                        'title': 'Algebra',
                        'description': 'Algebra is a branch of mathematics dealing with symbols and the rules for manipulating these symbols.',
                        'content': 'In algebra, we use letters (like x or y) to represent numbers whose values are not yet known. These letters are called variables. Algebra helps us solve equations and understand relationships between quantities.',
                        'examples': [
                            'Solving for x in the equation 2x + 3 = 7',
                            'Finding the value of y in y = mx + c'
                        ],
                        'hindi_content': 'बीजगणित गणित की एक शाखा है जो प्रतीकों और इन प्रतीकों के हेरफेर के नियमों से संबंधित है। बीजगणित में, हम अज्ञात मानों के लिए अक्षरों (जैसे x या y) का उपयोग करते हैं। इन अक्षरों को चर कहा जाता है। बीजगणित हमें समीकरणों को हल करने और मात्राओं के बीच संबंधों को समझने में मदद करता है।'
                    },
                    'geometry': {
                        'title': 'Geometry',
                        'description': 'Geometry is a branch of mathematics that studies the sizes, shapes, positions, and dimensions of things.',
                        'content': 'Geometry deals with questions of shape, size, relative position of figures, and the properties of space. It is one of the oldest branches of mathematics, having arisen in response to practical needs in surveying, construction, astronomy, and various crafts.',
                        'examples': [
                            'Calculating the area of a triangle',
                            'Finding the volume of a sphere',
                            'Determining the angles in a polygon'
                        ],
                        'hindi_content': 'ज्यामिति गणित की एक शाखा है जो वस्तुओं के आकार, आकृति, स्थिति और आयामों का अध्ययन करती है। ज्यामिति आकृति, आकार, आंकड़ों की सापेक्ष स्थिति और स्थान के गुणों से संबंधित प्रश्नों से निपटती है। यह गणित की सबसे पुरानी शाखाओं में से एक है।'
                    }
                },
                'faqs': [
                    {
                        'question': 'What is the Pythagorean theorem?',
                        'answer': 'The Pythagorean theorem states that in a right-angled triangle, the square of the length of the hypotenuse (the side opposite the right angle) is equal to the sum of the squares of the other two sides. It is expressed as a² + b² = c², where c is the length of the hypotenuse, and a and b are the lengths of the other two sides.',
                        'hindi_question': 'पाइथागोरस प्रमेय क्या है?',
                        'hindi_answer': 'पाइथागोरस प्रमेय कहता है कि एक समकोण त्रिभुज में, कर्ण (समकोण के विपरीत भुजा) की लंबाई का वर्ग अन्य दो भुजाओं के वर्गों के योग के बराबर होता है। इसे a² + b² = c² के रूप में व्यक्त किया जाता है, जहां c कर्ण की लंबाई है, और a और b अन्य दो भुजाओं की लंबाई हैं।'
                    },
                    {
                        'question': 'How do you solve a quadratic equation?',
                        'answer': 'A quadratic equation can be solved using the quadratic formula: x = (-b ± √(b² - 4ac)) / 2a, where the equation is in the form ax² + bx + c = 0. You can also solve it by factoring, completing the square, or graphing.',
                        'hindi_question': 'द्विघात समीकरण को कैसे हल करें?',
                        'hindi_answer': 'द्विघात समीकरण को द्विघात सूत्र का उपयोग करके हल किया जा सकता है: x = (-b ± √(b² - 4ac)) / 2a, जहां समीकरण ax² + bx + c = 0 के रूप में है। आप इसे गुणनखंड, वर्ग पूरा करके, या ग्राफिंग द्वारा भी हल कर सकते हैं।'
                    }
                ]
            },
            'science': {
                'concepts': {
                    'photosynthesis': {
                        'title': 'Photosynthesis',
                        'description': 'Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with carbon dioxide and water.',
                        'content': 'Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that can later be released to fuel the organism's activities. This chemical energy is stored in carbohydrate molecules, such as sugars, which are synthesized from carbon dioxide and water.',
                        'examples': [
                            'Plants converting sunlight to glucose',
                            'Algae producing oxygen through photosynthesis'
                        ],
                        'hindi_content': 'प्रकाश संश्लेषण वह प्रक्रिया है जिसके द्वारा हरे पौधे और कुछ अन्य जीव सूर्य के प्रकाश का उपयोग कार्बन डाइऑक्साइड और पानी के साथ भोजन संश्लेषित करने के लिए करते हैं। यह रासायनिक ऊर्जा कार्बोहाइड्रेट अणुओं में संग्रहीत होती है, जैसे शर्करा, जो कार्बन डाइऑक्साइड और पानी से संश्लेषित होती हैं।'
                    }
                },
                'faqs': [
                    {
                        'question': 'What are the states of matter?',
                        'answer': 'The four common states of matter are solid, liquid, gas, and plasma. In solids, particles are closely packed and have fixed positions. In liquids, particles are close together but can move around. In gases, particles are far apart and move freely. Plasma is similar to gas but contains charged particles (ions and electrons).',
                        'hindi_question': 'पदार्थ की अवस्थाएं क्या हैं?',
                        'hindi_answer': 'पदार्थ की चार सामान्य अवस्थाएं ठोस, तरल, गैस और प्लाज्मा हैं। ठोस में, कण निकटता से पैक होते हैं और निश्चित स्थिति में होते हैं। तरल में, कण एक साथ निकट होते हैं लेकिन चारों ओर घूम सकते हैं। गैसों में, कण दूर-दूर होते हैं और स्वतंत्र रूप से घूमते हैं। प्लाज्मा गैस के समान है लेकिन इसमें आवेशित कण (आयन और इलेक्ट्रॉन) होते हैं।'
                    }
                ]
            },
            'general': {
                'greeting_responses': [
                    'Hello! How can I help you with your studies today?',
                    'Hi there! What would you like to learn about?',
                    'Greetings! I\'m your AI tutor. What subject are you studying?',
                    'Welcome to EduSaarthi! How can I assist with your learning?'
                ],
                'hindi_greeting_responses': [
                    'नमस्ते! आज मैं आपकी पढ़ाई में कैसे मदद कर सकता हूँ?',
                    'नमस्कार! आप क्या सीखना चाहेंगे?',
                    'प्रणाम! मैं आपका AI ट्यूटर हूँ। आप किस विषय का अध्ययन कर रहे हैं?',
                    'EduSaarthi में आपका स्वागत है! मैं आपके सीखने में कैसे सहायता कर सकता हूँ?'
                ],
                'fallback_responses': [
                    'I'm not sure about that. Could you ask me something related to your school subjects?',
                    'I don't have information on that topic yet. Would you like to ask about mathematics, science, or another school subject?',
                    'I'm still learning about that. Can you try asking about a different topic?',
                    'That's a good question, but it's outside my current knowledge. Let's focus on your school subjects.'
                ],
                'hindi_fallback_responses': [
                    'मुझे इसके बारे में पक्का नहीं पता। क्या आप मुझसे अपने स्कूल विषयों से संबंधित कुछ पूछ सकते हैं?',
                    'मुझे अभी तक उस विषय पर जानकारी नहीं है। क्या आप गणित, विज्ञान या किसी अन्य स्कूल विषय के बारे में पूछना चाहेंगे?',
                    'मैं अभी भी उसके बारे में सीख रहा हूँ। क्या आप किसी अलग विषय के बारे में पूछने का प्रयास कर सकते हैं?',
                    'यह एक अच्छा प्रश्न है, लेकिन यह मेरे वर्तमान ज्ञान से बाहर है। आइए अपने स्कूल विषयों पर ध्यान केंद्रित करें।'
                ]
            }
        }
        
        # Prepare vectorizer with default content
        self._prepare_vectorizer()
    
    def _prepare_vectorizer(self):
        """Prepare TF-IDF vectorizer with knowledge base content"""
        # Collect all text content from knowledge base
        all_texts = []
        
        # Process concepts
        for subject, data in self.knowledge_base.items():
            if subject == 'general':
                continue
                
            if 'concepts' in data:
                for concept_id, concept in data['concepts'].items():
                    all_texts.append(concept.get('content', ''))
            
            if 'faqs' in data:
                for faq in data['faqs']:
                    all_texts.append(faq.get('question', '') + ' ' + faq.get('answer', ''))
        
        # Fit vectorizer if we have content
        if all_texts:
            self.vectorizer.fit(all_texts)
    
    def create_session(self, user_id, language='english', subject=None):
        """
        Create a new tutoring session
        
        Parameters:
        -----------
        user_id : str
            Unique identifier for the user
        language : str
            Preferred language ('english' or 'hindi')
        subject : str
            Subject focus for the session (optional)
        
        Returns:
        --------
        str
            Session ID
        """
        session_id = f"{user_id}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        self.sessions[session_id] = {
            'user_id': user_id,
            'language': language,
            'subject': subject,
            'messages': [],
            'created_at': datetime.datetime.now().isoformat(),
            'last_interaction_at': datetime.datetime.now().isoformat()
        }
        
        # Add greeting message
        greeting = self._get_greeting(language)
        self.sessions[session_id]['messages'].append({
            'role': 'assistant',
            'content': greeting,
            'timestamp': datetime.datetime.now().isoformat(),
            'content_type': 'text'
        })
        
        return session_id
    
    def _get_greeting(self, language):
        """Get appropriate greeting based on language"""
        if language == 'hindi':
            greetings = self.knowledge_base.get('general', {}).get('hindi_greeting_responses', [])
            if not greetings:
                return "नमस्ते! मैं आपका AI ट्यूटर हूँ। आप क्या सीखना चाहेंगे?"
        else:
            greetings = self.knowledge_base.get('general', {}).get('greeting_responses', [])
            if not greetings:
                return "Hello! I'm your AI tutor. What would you like to learn about?"
        
        import random
        return random.choice(greetings)
    
    def get_session(self, session_id):
        """
        Get session data
        
        Parameters:
        -----------
        session_id : str
            Session identifier
        
        Returns:
        --------
        dict
            Session data
        """
        return self.sessions.get(session_id)
    
    def process_message(self, session_id, message, content_type='text'):
        """
        Process a user message and generate a response
        
        Parameters:
        -----------
        session_id : str
            Session identifier
        message : str
            User message
        content_type : str
            Type of content ('text', 'voice', etc.)
        
        Returns:
        --------
        dict
            Response message
        """
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        language = session.get('language', 'english')
        
        # Add user message to session
        session['messages'].append({
            'role': 'user',
            'content': message,
            'timestamp': datetime.datetime.now().isoformat(),
            'content_type': content_type
        })
        
        # Update last interaction time
        session['last_interaction_at'] = datetime.datetime.now().isoformat()
        
        # Generate response
        response_content = self._generate_response(message, session)
        
        # Add assistant response to session
        response = {
            'role': 'assistant',
            'content': response_content,
            'timestamp': datetime.datetime.now().isoformat(),
            'content_type': 'text'
        }
        
        session['messages'].append(response)
        
        return response
    
    def _generate_response(self, message, session):
        """Generate response based on user message and session context"""
        language = session.get('language', 'english')
        subject = session.get('subject')
        
        # Detect language if not specified
        if self._is_hindi_text(message):
            detected_language = 'hindi'
        else:
            detected_language = 'english'
        
        # Check if it's a greeting
        if self._is_greeting(message, detected_language):
            return self._get_greeting(language)
        
        # Check if it's a subject-specific question
        detected_subject = self._detect_subject(message)
        if detected_subject:
            subject = detected_subject
        
        # Try to find a direct match in FAQs
        faq_response = self._find_faq_match(message, language, subject)
        if faq_response:
            return faq_response
        
        # Try to find a concept match
        concept_response = self._find_concept_match(message, language, subject)
        if concept_response:
            return concept_response
        
        # Generate a response based on previous conversation context
        context_response = self._generate_context_response(message, session)
        if context_response:
            return context_response
        
        # Fallback response
        return self._get_fallback_response(language)
    
    def _is_hindi_text(self, text):
        """Check if text contains Hindi characters"""
        # Hindi Unicode range (approximate)
        hindi_pattern = re.compile(r'[\u0900-\u097F\u0A00-\u0A7F]+')
        return bool(hindi_pattern.search(text))
    
    def _is_greeting(self, message, language):
        """Check if message is a greeting"""
        english_greetings = {'hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening'}
        hindi_greetings = {'नमस्ते', 'नमस्कार', 'प्रणाम', 'हेलो', 'हाय', 'सुप्रभात', 'शुभ दोपहर', 'शुभ संध्या'}
        
        message_lower = message.lower()
        
        if language == 'hindi':
            return any(greeting in message_lower for greeting in hindi_greetings)
        else:
            return any(greeting in message_lower for greeting in english_greetings)
    
    def _detect_subject(self, message):
        """Detect subject from message"""
        message_lower = message.lower()
        
        subject_keywords = {
            'mathematics': ['math', 'mathematics', 'algebra', 'geometry', 'calculus', 'arithmetic', 'equation', 'गणित', 'बीजगणित', 'ज्यामिति'],
            'science': ['science', 'physics', 'chemistry', 'biology', 'experiment', 'विज्ञान', 'भौतिकी', 'रसायन', 'जीव विज्ञान'],
            'english': ['english', 'grammar', 'literature', 'writing', 'reading', 'अंग्रेजी', 'व्याकरण', 'साहित्य'],
            'hindi': ['hindi', 'हिंदी', 'हिन्दी', 'हिंदी व्याकरण'],
            'social_science': ['social', 'history', 'geography', 'civics', 'economics', 'सामाजिक विज्ञान', 'इतिहास', 'भूगोल']
        }
        
        for subject, keywords in subject_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return subject
        
        return None
    
    def _find_faq_match(self, message, language, subject=None):
        """Find matching FAQ for the message"""
        best_match = None
        best_score = 0.7  # Threshold for considering a match
        
        # Preprocess the message
        if language == 'hindi':
            processed_message = self._preprocess_text_hindi(message)
            question_key = 'hindi_question'
            answer_key = 'hindi_answer'
        else:
            processed_message = self._preprocess_text_english(message)
            question_key = 'question'
            answer_key = 'answer'
        
        # Convert message to vector
        if not processed_message:
            return None
            
        try:
            message_vector = self.vectorizer.transform([processed_message])
        except:
            # If vectorizer fails (e.g., new terms), use simpler matching
            for subj, data in self.knowledge_base.items():
                if subject and subj != subject and subj != 'general':
                    continue
                    
                if 'faqs' in data:
                    for faq in data['faqs']:
                        if question_key in faq and self._simple_match(message, faq[question_key]):
                            return faq[answer_key]
            return None
        
        # Search through all FAQs
        for subj, data in self.knowledge_base.items():
            if subject and subj != subject and subj != 'general':
                continue
                
            if 'faqs' in data:
                for faq in data['faqs']:
                    if question_key in faq:
                        question = faq[question_key]
                        processed_question = self._preprocess_text_english(question) if language == 'english' else self._preprocess_text_hindi(question)
                        
                        try:
                            question_vector = self.vectorizer.transform([processed_question])
                            similarity = cosine_similarity(message_vector, question_vector)[0][0]
                            
                            if similarity > best_score:
                                best_score = similarity
                                best_match = faq[answer_key]
                        except:
                            # Fall back to simple matching if vectorization fails
                            if self._simple_match(message, question):
                                best_match = faq[answer_key]
                                best_score = 0.8
        
        return best_match
    
    def _simple_match(self, message, text):
        """Simple keyword matching when vectorization fails"""
        message_words = set(message.lower().split())
        text_words = set(text.lower().split())
        common_words = message_words.intersection(text_words)
        
        # If more than 50% of the words in the shorter text match
        shorter_length = min(len(message_words), len(text_words))
        if shorter_length > 0 and len(common_words) / shorter_length > 0.5:
            return True
        return False
    
    def _find_concept_match(self, message, language, subject=None):
        """Find matching concept for the message"""
        best_match = None
        best_score = 0.6  # Threshold for considering a match
        
        # Preprocess the message
        if language == 'hindi':
            processed_message = self._preprocess_text_hindi(message)
            content_key = 'hindi_content'
        else:
            processed_message = self._preprocess_text_english(message)
            content_key = 'content'
        
        # Convert message to vector
        if not processed_message:
            return None
            
        try:
            message_vector = self.vectorizer.transform([processed_message])
        except:
            # If vectorizer fails, use simpler matching
            return None
        
        # Search through all concepts
        for subj, data in self.knowledge_base.items():
            if subject and subj != subject:
                continue
                
            if 'concepts' in data:
                for concept_id, concept in data['concepts'].items():
                    if content_key in concept:
                        content = concept[content_key]
                        title = concept.get('title', '')
                        
                        # Check if message contains concept title
                        if title.lower() in message.lower():
                            return content
                        
                        processed_content = self._preprocess_text_english(content) if language == 'english' else self._preprocess_text_hindi(content)
                        
                        try:
                            content_vector = self.vectorizer.transform([processed_content])
                            similarity = cosine_similarity(message_vector, content_vector)[0][0]
                            
                            if similarity > best_score:
                                best_score = similarity
                                best_match = content
                        except:
                            pass
        
        return best_match
    
    def _generate_context_response(self, message, session):
        """Generate response based on conversation context"""
        # This is a simplified implementation
        # In a real system, this would use more sophisticated NLP and context understanding
        
        # For now, we'll just check if the message contains certain keywords
        message_lower = message.lower()
        language = session.get('language', 'english')
        
        # Check for explanation requests
        explanation_keywords_english = ['explain', 'how', 'what is', 'tell me about', 'describe']
        explanation_keywords_hindi = ['समझाओ', 'कैसे', 'क्या है', 'बताओ', 'वर्णन करो']
        
        explanation_keywords = explanation_keywords_english if language == 'english' else explanation_keywords_hindi
        
        for keyword in explanation_keywords:
            if keyword in message_lower:
                # Extract the topic to explain
                topic = message_lower.split(keyword, 1)[1].strip()
                if topic:
                    # Try to find information about this topic
                    for subject, data in self.knowledge_base.items():
                        if 'concepts' in data:
                            for concept_id, concept in data['concepts'].items():
                                if concept_id.lower() in topic or concept.get('title', '').lower() in topic:
                                    content_key = 'content' if language == 'english' else 'hindi_content'
                                    if content_key in concept:
                                        return concept[content_key]
        
        # No context-based response generated
        return None
    
    def _get_fallback_response(self, language):
        """Get appropriate fallback response based on language"""
        if language == 'hindi':
            fallbacks = self.knowledge_base.get('general', {}).get('hindi_fallback_responses', [])
            if not fallbacks:
                return "मुझे इस प्रश्न का उत्तर नहीं पता। क्या आप कोई अन्य प्रश्न पूछना चाहेंगे?"
        else:
            fallbacks = self.knowledge_base.get('general', {}).get('fallback_responses', [])
            if not fallbacks:
                return "I don't know the answer to that question. Would you like to ask something else?"
        
        import random
        return random.choice(fallbacks)
    
    def _preprocess_text_english(self, text):
        """Preprocess English text for better matching"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [word for word in tokens if word not in self.stop_words_english]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        return ' '.join(tokens)
    
    def _preprocess_text_hindi(self, text):
        """Preprocess Hindi text for better matching"""
        # Convert to lowercase (for any English characters)
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [word for word in tokens if word not in self.stop_words_hindi]
        
        return ' '.join(tokens)
    
    def summarize_content(self, content, language='english', level='medium'):
        """
        Summarize content based on difficulty level
        
        Parameters:
        -----------
        content : str
            Content to summarize
        language : str
            Language of content ('english' or 'hindi')
        level : str
            Difficulty level ('beginner', 'medium', 'advanced')
        
        Returns:
        --------
        str
            Summarized content
        """
        # This is a simplified implementation
        # In a real system, this would use more sophisticated NLP techniques
        
        if not content:
            return ""
        
        # Split into sentences
        sentences = sent_tokenize(content)
        
        if not sentences:
            return content
        
        # For beginner level, return fewer sentences
        if level == 'beginner':
            if len(sentences) <= 2:
                return content
            return ' '.join(sentences[:2])
        
        # For medium level, return about half the sentences
        elif level == 'medium':
            if len(sentences) <= 4:
                return content
            return ' '.join(sentences[:len(sentences)//2])
        
        # For advanced level, return most sentences
        else:
            return content
    
    def save_knowledge_base(self, path):
        """Save knowledge base to files"""
        try:
            os.makedirs(path, exist_ok=True)
            
            # Save each subject to a separate file
            for subject, data in self.knowledge_base.items():
                file_path = os.path.join(path, f"{subject}_kb.json")
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"Knowledge base saved successfully to {path}")
            return True
        except Exception as e:
            print(f"Error saving knowledge base: {e}")
            return False
    
    def add_to_knowledge_base(self, subject, content_type, content_data):
        """
        Add new content to knowledge base
        
        Parameters:
        -----------
        subject : str
            Subject category
        content_type : str
            Type of content ('concept' or 'faq')
        content_data : dict
            Content data
        
        Returns:
        --------
        bool
            Success status
        """
        try:
            # Initialize subject if not exists
            if subject not in self.knowledge_base:
                self.knowledge_base[subject] = {'concepts': {}, 'faqs': []}
            
            # Add concept
            if content_type == 'concept':
                if 'id' not in content_data:
                    return False
                
                concept_id = content_data['id']
                if 'concepts' not in self.knowledge_base[subject]:
                    self.knowledge_base[subject]['concepts'] = {}
                
                self.knowledge_base[subject]['concepts'][concept_id] = content_data
            
            # Add FAQ
            elif content_type == 'faq':
                if 'question' not in content_data or 'answer' not in content_data:
                    return False
                
                if 'faqs' not in self.knowledge_base[subject]:
                    self.knowledge_base[subject]['faqs'] = []
                
                self.knowledge_base[subject]['faqs'].append(content_data)
            
            else:
                return False
            
            # Re-prepare vectorizer with updated content
            self._prepare_vectorizer()
            
            return True
        except Exception as e:
            print(f"Error adding to knowledge base: {e}")
            return False
