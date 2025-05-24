import speech_recognition as sr
import os
import json
import datetime
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import soundfile as sf
import tempfile

class SpeechToTextSystem:
    """
    Speech-to-Text System for EduSaarthi
    
    This class implements speech recognition capabilities for both English and Hindi,
    allowing students to interact with the platform using voice commands and
    to transcribe spoken answers.
    """
    
    def __init__(self, models_path=None):
        """
        Initialize the speech-to-text system
        
        Parameters:
        -----------
        models_path : str
            Path to pre-trained models (optional)
        """
        self.recognizer = sr.Recognizer()
        self.english_model = None
        self.hindi_model = None
        self.english_processor = None
        self.hindi_processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize models
        self._initialize_models(models_path)
        
        # History of transcriptions
        self.transcription_history = {}
    
    def _initialize_models(self, models_path=None):
        """Initialize speech recognition models"""
        try:
            # For English, we'll use the wav2vec2 model fine-tuned on English
            if models_path and os.path.exists(os.path.join(models_path, "english")):
                self.english_processor = Wav2Vec2Processor.from_pretrained(os.path.join(models_path, "english"))
                self.english_model = Wav2Vec2ForCTC.from_pretrained(os.path.join(models_path, "english")).to(self.device)
            else:
                # Use pre-trained model from Hugging Face
                self.english_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
                self.english_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(self.device)
            
            # For Hindi, we'll use a model fine-tuned on Hindi
            if models_path and os.path.exists(os.path.join(models_path, "hindi")):
                self.hindi_processor = Wav2Vec2Processor.from_pretrained(os.path.join(models_path, "hindi"))
                self.hindi_model = Wav2Vec2ForCTC.from_pretrained(os.path.join(models_path, "hindi")).to(self.device)
            else:
                # Use pre-trained model from Hugging Face that supports Hindi
                self.hindi_processor = Wav2Vec2Processor.from_pretrained("ai4bharat/indicwav2vec-hindi")
                self.hindi_model = Wav2Vec2ForCTC.from_pretrained("ai4bharat/indicwav2vec-hindi").to(self.device)
            
            print("Speech recognition models initialized successfully")
        except Exception as e:
            print(f"Error initializing speech recognition models: {e}")
            # Fall back to using Google's speech recognition API
            print("Falling back to Google Speech Recognition API")
    
    def transcribe_audio_file(self, audio_file_path, language='auto'):
        """
        Transcribe speech from an audio file
        
        Parameters:
        -----------
        audio_file_path : str
            Path to the audio file
        language : str
            Language of the audio ('english', 'hindi', or 'auto' for auto-detection)
        
        Returns:
        --------
        dict
            Transcription result with text and metadata
        """
        if not os.path.exists(audio_file_path):
            return {"error": "Audio file not found"}
        
        try:
            # Load audio file
            with sr.AudioFile(audio_file_path) as source:
                audio_data = self.recognizer.record(source)
            
            # Determine language if auto
            if language == 'auto':
                language = self._detect_language(audio_file_path)
            
            # Transcribe based on language
            if language == 'hindi':
                text = self._transcribe_hindi(audio_file_path)
            else:  # Default to English
                text = self._transcribe_english(audio_file_path)
            
            # Create result
            result = {
                "text": text,
                "language": language,
                "timestamp": datetime.datetime.now().isoformat(),
                "audio_file": audio_file_path,
                "success": True
            }
            
            # Add to history
            self._add_to_history(result)
            
            return result
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return {
                "error": str(e),
                "success": False,
                "audio_file": audio_file_path,
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    def _transcribe_english(self, audio_file_path):
        """Transcribe English audio using wav2vec2 model"""
        try:
            if self.english_model and self.english_processor:
                # Load audio
                speech_array, sampling_rate = librosa.load(audio_file_path, sr=16000)
                
                # Process through model
                inputs = self.english_processor(speech_array, sampling_rate=16000, return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    logits = self.english_model(inputs.input_values, attention_mask=inputs.attention_mask).logits
                
                # Get predicted ids and convert to text
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.english_processor.batch_decode(predicted_ids)[0]
                
                return transcription
            else:
                # Fall back to Google's API
                with sr.AudioFile(audio_file_path) as source:
                    audio_data = self.recognizer.record(source)
                return self.recognizer.recognize_google(audio_data, language="en-US")
        except Exception as e:
            print(f"Error in English transcription: {e}")
            # Try Google's API as a fallback
            try:
                with sr.AudioFile(audio_file_path) as source:
                    audio_data = self.recognizer.record(source)
                return self.recognizer.recognize_google(audio_data, language="en-US")
            except:
                return "Could not transcribe audio"
    
    def _transcribe_hindi(self, audio_file_path):
        """Transcribe Hindi audio using wav2vec2 model"""
        try:
            if self.hindi_model and self.hindi_processor:
                # Load audio
                speech_array, sampling_rate = librosa.load(audio_file_path, sr=16000)
                
                # Process through model
                inputs = self.hindi_processor(speech_array, sampling_rate=16000, return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    logits = self.hindi_model(inputs.input_values, attention_mask=inputs.attention_mask).logits
                
                # Get predicted ids and convert to text
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.hindi_processor.batch_decode(predicted_ids)[0]
                
                return transcription
            else:
                # Fall back to Google's API
                with sr.AudioFile(audio_file_path) as source:
                    audio_data = self.recognizer.record(source)
                return self.recognizer.recognize_google(audio_data, language="hi-IN")
        except Exception as e:
            print(f"Error in Hindi transcription: {e}")
            # Try Google's API as a fallback
            try:
                with sr.AudioFile(audio_file_path) as source:
                    audio_data = self.recognizer.record(source)
                return self.recognizer.recognize_google(audio_data, language="hi-IN")
            except:
                return "ऑडियो को ट्रांसक्राइब नहीं किया जा सका"
    
    def _detect_language(self, audio_file_path):
        """Detect language of the audio (English or Hindi)"""
        try:
            # This is a simplified approach - in a real system, we would use a language identification model
            # For now, we'll try to transcribe with both models and see which one gives a better confidence score
            
            # Try with Google's API first
            with sr.AudioFile(audio_file_path) as source:
                audio_data = self.recognizer.record(source)
            
            try:
                # Try English
                result_en = self.recognizer.recognize_google(audio_data, language="en-US", show_all=True)
                confidence_en = result_en['alternative'][0]['confidence'] if 'alternative' in result_en else 0
            except:
                confidence_en = 0
            
            try:
                # Try Hindi
                result_hi = self.recognizer.recognize_google(audio_data, language="hi-IN", show_all=True)
                confidence_hi = result_hi['alternative'][0]['confidence'] if 'alternative' in result_hi else 0
            except:
                confidence_hi = 0
            
            # Return the language with higher confidence
            return 'hindi' if confidence_hi > confidence_en else 'english'
        except Exception as e:
            print(f"Error detecting language: {e}")
            return 'english'  # Default to English
    
    def record_and_transcribe(self, duration=5, language='auto'):
        """
        Record audio from microphone and transcribe
        
        Parameters:
        -----------
        duration : int
            Duration to record in seconds
        language : str
            Language of the audio ('english', 'hindi', or 'auto' for auto-detection)
        
        Returns:
        --------
        dict
            Transcription result with text and metadata
        """
        try:
            # Create a temporary file to save the recording
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_file_path = temp_file.name
            temp_file.close()
            
            # Record audio
            with sr.Microphone() as source:
                print(f"Recording for {duration} seconds...")
                self.recognizer.adjust_for_ambient_noise(source)
                audio_data = self.recognizer.listen(source, timeout=duration)
            
            # Save audio to file
            with open(temp_file_path, "wb") as f:
                f.write(audio_data.get_wav_data())
            
            # Transcribe the saved audio
            result = self.transcribe_audio_file(temp_file_path, language)
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            return result
        except Exception as e:
            print(f"Error recording and transcribing: {e}")
            return {
                "error": str(e),
                "success": False,
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    def _add_to_history(self, result):
        """Add transcription result to history"""
        user_id = result.get('user_id', 'anonymous')
        
        if user_id not in self.transcription_history:
            self.transcription_history[user_id] = []
        
        self.transcription_history[user_id].append(result)
    
    def get_user_history(self, user_id):
        """
        Get transcription history for a user
        
        Parameters:
        -----------
        user_id : str
            User identifier
        
        Returns:
        --------
        list
            List of transcription results
        """
        return self.transcription_history.get(user_id, [])
    
    def save_history(self, file_path):
        """
        Save transcription history to file
        
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
                json.dump(self.transcription_history, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving history: {e}")
            return False
    
    def load_history(self, file_path):
        """
        Load transcription history from file
        
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
                    self.transcription_history = json.load(f)
                return True
            return False
        except Exception as e:
            print(f"Error loading history: {e}")
            return False
    
    def transcribe_for_quiz(self, audio_file_path, expected_answer, language='auto', similarity_threshold=0.7):
        """
        Transcribe audio and check if it matches an expected answer
        
        Parameters:
        -----------
        audio_file_path : str
            Path to the audio file
        expected_answer : str
            Expected answer to compare against
        language : str
            Language of the audio ('english', 'hindi', or 'auto' for auto-detection)
        similarity_threshold : float
            Threshold for considering answers similar (0-1)
        
        Returns:
        --------
        dict
            Result with transcription and match information
        """
        # Transcribe the audio
        transcription = self.transcribe_audio_file(audio_file_path, language)
        
        if not transcription.get('success', False):
            return transcription
        
        # Compare with expected answer
        text = transcription['text']
        similarity = self._calculate_similarity(text, expected_answer)
        is_match = similarity >= similarity_threshold
        
        # Add match information to result
        transcription['expected_answer'] = expected_answer
        transcription['similarity'] = similarity
        transcription['is_match'] = is_match
        
        return transcription
    
    def _calculate_similarity(self, text1, text2):
        """Calculate similarity between two texts"""
        # This is a simplified approach - in a real system, we would use more sophisticated NLP techniques
        # Convert to lowercase and remove punctuation
        import re
        import string
        
        def clean_text(text):
            text = text.lower()
            text = re.sub(f'[{string.punctuation}]', '', text)
            return text
        
        text1 = clean_text(text1)
        text2 = clean_text(text2)
        
        # Split into words
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0
        
        return intersection / union
