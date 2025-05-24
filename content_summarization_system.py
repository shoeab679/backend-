import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
import string
import json
import os
import datetime
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class ContentSummarizationSystem:
    """
    Content Summarization System for EduSaarthi
    
    This class implements advanced content summarization and rephrasing capabilities
    that can adapt content to different learning levels (beginner, medium, advanced)
    in both English and Hindi.
    """
    
    def __init__(self, models_path=None):
        """
        Initialize the content summarization system
        
        Parameters:
        -----------
        models_path : str
            Path to pre-trained models (optional)
        """
        self.stop_words_english = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
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
        
        # Initialize models
        self.english_model = None
        self.hindi_model = None
        self.english_tokenizer = None
        self.hindi_tokenizer = None
        
        self._initialize_models(models_path)
        
        # History of summarizations
        self.summarization_history = {}
    
    def _initialize_models(self, models_path=None):
        """Initialize summarization models"""
        try:
            # For English, we'll use T5 model
            if models_path and os.path.exists(os.path.join(models_path, "english_summarization")):
                self.english_tokenizer = T5Tokenizer.from_pretrained(os.path.join(models_path, "english_summarization"))
                self.english_model = T5ForConditionalGeneration.from_pretrained(os.path.join(models_path, "english_summarization")).to(self.device)
            else:
                # Use pre-trained model from Hugging Face
                self.english_tokenizer = T5Tokenizer.from_pretrained("t5-small")
                self.english_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(self.device)
            
            # For Hindi, we'll use a model fine-tuned for Hindi
            if models_path and os.path.exists(os.path.join(models_path, "hindi_summarization")):
                self.hindi_tokenizer = T5Tokenizer.from_pretrained(os.path.join(models_path, "hindi_summarization"))
                self.hindi_model = T5ForConditionalGeneration.from_pretrained(os.path.join(models_path, "hindi_summarization")).to(self.device)
            else:
                # Use pre-trained model from Hugging Face that supports Hindi
                self.hindi_tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
                self.hindi_model = T5ForConditionalGeneration.from_pretrained("google/mt5-small").to(self.device)
            
            print("Summarization models initialized successfully")
        except Exception as e:
            print(f"Error initializing summarization models: {e}")
            print("Falling back to extractive summarization methods")
    
    def summarize_text(self, text, language='english', level='medium', max_length=None, min_length=None):
        """
        Summarize text content based on difficulty level
        
        Parameters:
        -----------
        text : str
            Text content to summarize
        language : str
            Language of the text ('english' or 'hindi')
        level : str
            Difficulty level ('beginner', 'medium', 'advanced')
        max_length : int
            Maximum length of summary in words (optional)
        min_length : int
            Minimum length of summary in words (optional)
        
        Returns:
        --------
        dict
            Summarization result with text and metadata
        """
        if not text:
            return {
                "summary": "",
                "original_length": 0,
                "summary_length": 0,
                "language": language,
                "level": level,
                "timestamp": datetime.datetime.now().isoformat(),
                "success": False,
                "error": "Empty text provided"
            }
        
        # Detect language if not specified
        if language not in ['english', 'hindi']:
            language = self._detect_language(text)
        
        # Set length parameters based on level if not specified
        if not max_length:
            if level == 'beginner':
                max_length = len(text.split()) // 4  # 25% of original for beginners
            elif level == 'medium':
                max_length = len(text.split()) // 2  # 50% of original for medium
            else:  # advanced
                max_length = int(len(text.split()) * 0.75)  # 75% of original for advanced
        
        if not min_length:
            min_length = max(30, max_length // 2)  # At least 30 words or half of max_length
        
        # Choose summarization method based on available models
        if language == 'hindi':
            if self.hindi_model and self.hindi_tokenizer:
                summary = self._abstractive_summarize_hindi(text, level, max_length, min_length)
            else:
                summary = self._extractive_summarize(text, level, language)
        else:  # english
            if self.english_model and self.english_tokenizer:
                summary = self._abstractive_summarize_english(text, level, max_length, min_length)
            else:
                summary = self._extractive_summarize(text, level, language)
        
        # Create result
        result = {
            "summary": summary,
            "original_length": len(text.split()),
            "summary_length": len(summary.split()),
            "language": language,
            "level": level,
            "timestamp": datetime.datetime.now().isoformat(),
            "success": True
        }
        
        # Add to history
        self._add_to_history(result)
        
        return result
    
    def _abstractive_summarize_english(self, text, level, max_length, min_length):
        """Summarize English text using T5 model"""
        try:
            # Prepare input text
            input_text = "summarize: " + text
            
            # Tokenize input
            inputs = self.english_tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
            
            # Adjust generation parameters based on level
            if level == 'beginner':
                num_beams = 2
                length_penalty = 1.5  # Favor shorter summaries
                repetition_penalty = 2.0  # Strongly avoid repetition
            elif level == 'medium':
                num_beams = 4
                length_penalty = 1.0  # Balanced
                repetition_penalty = 1.5  # Moderate repetition avoidance
            else:  # advanced
                num_beams = 6
                length_penalty = 0.8  # Allow longer summaries
                repetition_penalty = 1.2  # Less repetition avoidance
            
            # Generate summary
            outputs = self.english_model.generate(
                inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                early_stopping=True
            )
            
            # Decode summary
            summary = self.english_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return summary
        except Exception as e:
            print(f"Error in abstractive summarization: {e}")
            # Fall back to extractive summarization
            return self._extractive_summarize(text, level, 'english')
    
    def _abstractive_summarize_hindi(self, text, level, max_length, min_length):
        """Summarize Hindi text using mT5 model"""
        try:
            # Prepare input text
            input_text = "सारांश: " + text  # "summarize: " in Hindi
            
            # Tokenize input
            inputs = self.hindi_tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
            
            # Adjust generation parameters based on level
            if level == 'beginner':
                num_beams = 2
                length_penalty = 1.5  # Favor shorter summaries
                repetition_penalty = 2.0  # Strongly avoid repetition
            elif level == 'medium':
                num_beams = 4
                length_penalty = 1.0  # Balanced
                repetition_penalty = 1.5  # Moderate repetition avoidance
            else:  # advanced
                num_beams = 6
                length_penalty = 0.8  # Allow longer summaries
                repetition_penalty = 1.2  # Less repetition avoidance
            
            # Generate summary
            outputs = self.hindi_model.generate(
                inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                early_stopping=True
            )
            
            # Decode summary
            summary = self.hindi_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return summary
        except Exception as e:
            print(f"Error in Hindi abstractive summarization: {e}")
            # Fall back to extractive summarization
            return self._extractive_summarize(text, level, 'hindi')
    
    def _extractive_summarize(self, text, level, language):
        """Extractive summarization as fallback method"""
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        if not sentences:
            return text
        
        # Calculate sentence scores
        sentence_scores = self._score_sentences(sentences, language)
        
        # Determine number of sentences to include based on level
        if level == 'beginner':
            num_sentences = max(1, len(sentences) // 4)  # 25% of sentences
        elif level == 'medium':
            num_sentences = max(2, len(sentences) // 2)  # 50% of sentences
        else:  # advanced
            num_sentences = max(3, int(len(sentences) * 0.75))  # 75% of sentences
        
        # Get top sentences
        top_sentences = sorted(range(len(sentences)), key=lambda i: sentence_scores[i], reverse=True)[:num_sentences]
        
        # Sort by original order
        top_sentences.sort()
        
        # Combine sentences
        summary = ' '.join([sentences[i] for i in top_sentences])
        
        return summary
    
    def _score_sentences(self, sentences, language):
        """Score sentences based on importance"""
        # Preprocess sentences
        if language == 'hindi':
            preprocessed_sentences = [self._preprocess_text_hindi(sentence) for sentence in sentences]
            stop_words = self.stop_words_hindi
        else:
            preprocessed_sentences = [self._preprocess_text_english(sentence) for sentence in sentences]
            stop_words = self.stop_words_english
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words=list(stop_words))
        
        # Get TF-IDF matrix
        try:
            tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
            
            # Calculate sentence scores
            sentence_scores = []
            
            for i in range(len(sentences)):
                score = np.sum(tfidf_matrix[i].toarray())
                sentence_scores.append(score)
            
            return sentence_scores
        except:
            # If vectorization fails, use simpler scoring
            return [len(sentence.split()) for sentence in sentences]
    
    def _preprocess_text_english(self, text):
        """Preprocess English text"""
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
        """Preprocess Hindi text"""
        # Convert to lowercase (for any English characters)
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [word for word in tokens if word not in self.stop_words_hindi]
        
        return ' '.join(tokens)
    
    def _detect_language(self, text):
        """Detect language of the text (English or Hindi)"""
        # Hindi Unicode range (approximate)
        hindi_pattern = re.compile(r'[\u0900-\u097F\u0A00-\u0A7F]+')
        hindi_chars = hindi_pattern.findall(text)
        
        # If text contains Hindi characters, classify as Hindi
        if hindi_chars:
            return 'hindi'
        else:
            return 'english'
    
    def _add_to_history(self, result):
        """Add summarization result to history"""
        user_id = result.get('user_id', 'anonymous')
        
        if user_id not in self.summarization_history:
            self.summarization_history[user_id] = []
        
        self.summarization_history[user_id].append(result)
    
    def get_user_history(self, user_id):
        """
        Get summarization history for a user
        
        Parameters:
        -----------
        user_id : str
            User identifier
        
        Returns:
        --------
        list
            List of summarization results
        """
        return self.summarization_history.get(user_id, [])
    
    def save_history(self, file_path):
        """
        Save summarization history to file
        
        Parameters:
        -----------
        file_path : str
            Path to save history
        
        Returns:
        --------
        bool
            Success status
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.summarization_history, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving history: {e}")
            return False
    
    def load_history(self, file_path):
        """
        Load summarization history from file
        
        Parameters:
        -----------
        file_path : str
            Path to load history from
        
        Returns:
        --------
        bool
            Success status
        """
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    self.summarization_history = json.load(f)
                return True
            return False
        except Exception as e:
            print(f"Error loading history: {e}")
            return False
    
    def rephrase_content(self, text, language='english', level='medium'):
        """
        Rephrase content for different learning levels
        
        Parameters:
        -----------
        text : str
            Text content to rephrase
        language : str
            Language of the text ('english' or 'hindi')
        level : str
            Difficulty level ('beginner', 'medium', 'advanced')
        
        Returns:
        --------
        dict
            Rephrasing result with text and metadata
        """
        if not text:
            return {
                "rephrased": "",
                "original_length": 0,
                "rephrased_length": 0,
                "language": language,
                "level": level,
                "timestamp": datetime.datetime.now().isoformat(),
                "success": False,
                "error": "Empty text provided"
            }
        
        # Detect language if not specified
        if language not in ['english', 'hindi']:
            language = self._detect_language(text)
        
        # Choose rephrasing method based on available models
        if language == 'hindi':
            if self.hindi_model and self.hindi_tokenizer:
                rephrased = self._abstractive_rephrase_hindi(text, level)
            else:
                rephrased = self._rule_based_rephrase(text, level, language)
        else:  # english
            if self.english_model and self.english_tokenizer:
                rephrased = self._abstractive_rephrase_english(text, level)
            else:
                rephrased = self._rule_based_rephrase(text, level, language)
        
        # Create result
        result = {
            "rephrased": rephrased,
            "original_length": len(text.split()),
            "rephrased_length": len(rephrased.split()),
            "language": language,
            "level": level,
            "timestamp": datetime.datetime.now().isoformat(),
            "success": True
        }
        
        return result
    
    def _abstractive_rephrase_english(self, text, level):
        """Rephrase English text using T5 model"""
        try:
            # Prepare input text based on level
            if level == 'beginner':
                input_text = "simplify: " + text
            elif level == 'medium':
                input_text = "rephrase: " + text
            else:  # advanced
                input_text = "elaborate: " + text
            
            # Tokenize input
            inputs = self.english_tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
            
            # Generate rephrased text
            outputs = self.english_model.generate(
                inputs,
                max_length=1024,
                num_beams=4,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                early_stopping=True
            )
            
            # Decode rephrased text
            rephrased = self.english_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return rephrased
        except Exception as e:
            print(f"Error in abstractive rephrasing: {e}")
            # Fall back to rule-based rephrasing
            return self._rule_based_rephrase(text, level, 'english')
    
    def _abstractive_rephrase_hindi(self, text, level):
        """Rephrase Hindi text using mT5 model"""
        try:
            # Prepare input text based on level
            if level == 'beginner':
                input_text = "सरल करें: " + text  # "simplify: " in Hindi
            elif level == 'medium':
                input_text = "पुनर्कथन: " + text  # "rephrase: " in Hindi
            else:  # advanced
                input_text = "विस्तार से बताएं: " + text  # "elaborate: " in Hindi
            
            # Tokenize input
            inputs = self.hindi_tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
            
            # Generate rephrased text
            outputs = self.hindi_model.generate(
                inputs,
                max_length=1024,
                num_beams=4,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                early_stopping=True
            )
            
            # Decode rephrased text
            rephrased = self.hindi_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return rephrased
        except Exception as e:
            print(f"Error in Hindi abstractive rephrasing: {e}")
            # Fall back to rule-based rephrasing
            return self._rule_based_rephrase(text, level, 'hindi')
    
    def _rule_based_rephrase(self, text, level, language):
        """Rule-based rephrasing as fallback method"""
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        if not sentences:
            return text
        
        # Process sentences based on level
        if level == 'beginner':
            # For beginners, simplify by keeping shorter sentences and removing complex ones
            processed_sentences = []
            for sentence in sentences:
                words = sentence.split()
                if len(words) < 15:  # Keep only shorter sentences
                    processed_sentences.append(sentence)
                else:
                    # Simplify longer sentences by keeping first part
                    processed_sentences.append(' '.join(words[:12]) + '.')
            
            # If we removed too many sentences, add some back
            if len(processed_sentences) < len(sentences) // 2:
                for sentence in sentences:
                    if sentence not in processed_sentences and len(processed_sentences) < len(sentences) // 2:
                        processed_sentences.append(sentence)
        
        elif level == 'medium':
            # For medium level, keep sentences as is
            processed_sentences = sentences
        
        else:  # advanced
            # For advanced level, keep all sentences and potentially add explanatory notes
            processed_sentences = []
            for sentence in sentences:
                processed_sentences.append(sentence)
                
                # For longer, potentially complex sentences, add explanatory note
                words = sentence.split()
                if len(words) > 20 and random.random() < 0.3:  # 30% chance to add note
                    if language == 'hindi':
                        processed_sentences.append("यह महत्वपूर्ण है क्योंकि यह विषय के मूल सिद्धांतों से संबंधित है।")  # "This is important as it relates to core principles of the subject."
                    else:
                        processed_sentences.append("This is important as it relates to core principles of the subject.")
        
        # Combine processed sentences
        rephrased = ' '.join(processed_sentences)
        
        return rephrased
    
    def generate_explanation(self, concept, language='english', level='medium'):
        """
        Generate explanation for a concept at different learning levels
        
        Parameters:
        -----------
        concept : str
            Concept to explain
        language : str
            Language for explanation ('english' or 'hindi')
        level : str
            Difficulty level ('beginner', 'medium', 'advanced')
        
        Returns:
        --------
        dict
            Explanation result with text and metadata
        """
        if not concept:
            return {
                "explanation": "",
                "language": language,
                "level": level,
                "timestamp": datetime.datetime.now().isoformat(),
                "success": False,
                "error": "Empty concept provided"
            }
        
        # Detect language if not specified
        if language not in ['english', 'hindi']:
            language = self._detect_language(concept)
        
        # Choose explanation method based on available models
        if language == 'hindi':
            if self.hindi_model and self.hindi_tokenizer:
                explanation = self._generate_explanation_with_model(concept, level, 'hindi')
            else:
                explanation = self._generate_simple_explanation(concept, level, language)
        else:  # english
            if self.english_model and self.english_tokenizer:
                explanation = self._generate_explanation_with_model(concept, level, 'english')
            else:
                explanation = self._generate_simple_explanation(concept, level, language)
        
        # Create result
        result = {
            "explanation": explanation,
            "language": language,
            "level": level,
            "timestamp": datetime.datetime.now().isoformat(),
            "success": True
        }
        
        return result
    
    def _generate_explanation_with_model(self, concept, level, language):
        """Generate explanation using language model"""
        try:
            # Prepare input text based on level and language
            if language == 'hindi':
                if level == 'beginner':
                    input_text = f"एक बच्चे को {concept} समझाएं"  # "Explain {concept} to a child"
                elif level == 'medium':
                    input_text = f"{concept} की व्याख्या करें"  # "Explain {concept}"
                else:  # advanced
                    input_text = f"{concept} पर विस्तृत व्याख्या दें"  # "Provide detailed explanation on {concept}"
                
                # Tokenize input
                inputs = self.hindi_tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
                
                # Generate explanation
                outputs = self.hindi_model.generate(
                    inputs,
                    max_length=300,
                    num_beams=4,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    repetition_penalty=1.2,
                    early_stopping=True
                )
                
                # Decode explanation
                explanation = self.hindi_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            else:  # english
                if level == 'beginner':
                    input_text = f"Explain {concept} to a child"
                elif level == 'medium':
                    input_text = f"Explain {concept}"
                else:  # advanced
                    input_text = f"Provide detailed explanation on {concept}"
                
                # Tokenize input
                inputs = self.english_tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
                
                # Generate explanation
                outputs = self.english_model.generate(
                    inputs,
                    max_length=300,
                    num_beams=4,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    repetition_penalty=1.2,
                    early_stopping=True
                )
                
                # Decode explanation
                explanation = self.english_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return explanation
        except Exception as e:
            print(f"Error generating explanation with model: {e}")
            # Fall back to simple explanation
            return self._generate_simple_explanation(concept, level, language)
    
    def _generate_simple_explanation(self, concept, level, language):
        """Generate simple explanation as fallback method"""
        if language == 'hindi':
            if level == 'beginner':
                return f"{concept} एक महत्वपूर्ण विषय है जो आपको नई चीजें सिखाता है। इसे समझने के लिए, आप इसे रोजमर्रा की जिंदगी से जोड़ सकते हैं।"
            elif level == 'medium':
                return f"{concept} एक महत्वपूर्ण अवधारणा है जो विषय की समझ के लिए आवश्यक है। यह विभिन्न सिद्धांतों और अनुप्रयोगों से जुड़ा हुआ है।"
            else:  # advanced
                return f"{concept} एक जटिल अवधारणा है जिसमें कई परतें और पहलू हैं। इसकी गहरी समझ के लिए, आपको इसके सिद्धांतिक आधार और व्यावहारिक अनुप्रयोगों दोनों का अध्ययन करना चाहिए।"
        else:  # english
            if level == 'beginner':
                return f"{concept} is an important topic that teaches you new things. To understand it, you can relate it to everyday life."
            elif level == 'medium':
                return f"{concept} is an important concept necessary for understanding the subject. It is connected to various theories and applications."
            else:  # advanced
                return f"{concept} is a complex concept with many layers and aspects. For a deep understanding, you should study both its theoretical foundations and practical applications."
