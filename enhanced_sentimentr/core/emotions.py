"""
Emotion analysis module for enhanced sentiment analysis
"""
import time
import logging
from typing import Dict, List, Optional
import re

from .models import EmotionType, AnalysisConfig


logger = logging.getLogger(__name__)


class EmotionAnalyzer:
    """
    Analyzes emotions in text using rule-based and lexicon approaches
    """
    
    def __init__(self):
        """Initialize emotion analyzer with emotion lexicons"""
        self.emotion_lexicons = self._load_emotion_lexicons()
        self.emotion_patterns = self._load_emotion_patterns()
    
    def _load_emotion_lexicons(self) -> Dict[EmotionType, List[str]]:
        """Load emotion-specific word lexicons"""
        return {
            EmotionType.JOY: [
                'happy', 'joy', 'joyful', 'delighted', 'cheerful', 'glad', 'pleased',
                'excited', 'thrilled', 'elated', 'ecstatic', 'euphoric', 'blissful',
                'content', 'satisfied', 'grateful', 'thankful', 'optimistic', 'hopeful',
                'love', 'adore', 'amazing', 'awesome', 'wonderful', 'fantastic', 'great'
            ],
            EmotionType.SADNESS: [
                'sad', 'sadness', 'sorrow', 'grief', 'melancholy', 'depressed', 'gloomy',
                'miserable', 'heartbroken', 'devastated', 'disappointed', 'dejected',
                'despair', 'hopeless', 'lonely', 'isolated', 'crying', 'tears', 'weep',
                'mourn', 'regret', 'sorry', 'tragic', 'terrible', 'awful', 'horrible'
            ],
            EmotionType.ANGER: [
                'angry', 'anger', 'rage', 'furious', 'mad', 'irritated', 'annoyed',
                'frustrated', 'livid', 'outraged', 'enraged', 'hostile', 'aggressive',
                'resentful', 'bitter', 'hate', 'hatred', 'disgusted', 'revolted',
                'pissed', 'damn', 'hell', 'stupid', 'idiot', 'terrible', 'worst'
            ],
            EmotionType.FEAR: [
                'fear', 'afraid', 'scared', 'terrified', 'frightened', 'anxious',
                'worried', 'nervous', 'panic', 'alarmed', 'concerned', 'apprehensive',
                'uneasy', 'intimidated', 'threatened', 'vulnerable', 'insecure',
                'paranoid', 'phobia', 'dread', 'horror', 'nightmare', 'creepy', 'spooky'
            ],
            EmotionType.SURPRISE: [
                'surprised', 'surprise', 'amazed', 'astonished', 'shocked', 'stunned',
                'startled', 'bewildered', 'confused', 'perplexed', 'unexpected',
                'sudden', 'wow', 'whoa', 'omg', 'unbelievable', 'incredible', 'wow'
            ],
            EmotionType.DISGUST: [
                'disgusted', 'disgust', 'revolted', 'repulsed', 'nauseated', 'sick',
                'gross', 'yuck', 'ew', 'nasty', 'vile', 'repugnant', 'offensive',
                'appalling', 'hideous', 'ugly', 'foul', 'rotten', 'contaminated'
            ],
            EmotionType.TRUST: [
                'trust', 'confident', 'reliable', 'dependable', 'faithful', 'loyal',
                'honest', 'sincere', 'authentic', 'genuine', 'credible', 'believable',
                'secure', 'safe', 'protected', 'assured', 'certain', 'convinced'
            ],
            EmotionType.ANTICIPATION: [
                'anticipation', 'expectation', 'hope', 'excitement', 'eager', 'looking forward',
                'await', 'prepare', 'ready', 'upcoming', 'future', 'plan', 'goal',
                'dream', 'wish', 'desire', 'want', 'need', 'curious', 'interested'
            ]
        }
    
    def _load_emotion_patterns(self) -> Dict[EmotionType, List[str]]:
        """Load regex patterns for detecting emotions"""
        return {
            EmotionType.JOY: [
                r'\b(haha|lol|lmao|rofl)\b',
                r'[😀😃😄😁😆😊☺️😂🤣😌🥰😍🤩]',
                r':\)|:D|XD|\^_\^'
            ],
            EmotionType.SADNESS: [
                r'[😢😭😞😔😟😕🙁☹️😣😖😫😩]',
                r':\(|;\(|T_T|ToT',
                r'\b(cry|crying|tears)\b'
            ],
            EmotionType.ANGER: [
                r'[😠😡🤬😤💢]',
                r'>:\(|>:O|\*grr\*',
                r'\b(damn|hell|wtf|angry)\b'
            ],
            EmotionType.FEAR: [
                r'[😨😰😱🫣😳]',
                r'O_O|o_o|@_@',
                r'\b(scared|afraid|help)\b'
            ],
            EmotionType.SURPRISE: [
                r'[😲😯😮😦😧🤯]',
                r'O_O|:O|\*gasp\*',
                r'\b(wow|omg|whoa|what)\b'
            ],
            EmotionType.DISGUST: [
                r'[🤢🤮😷🤧😬]',
                r':\||eww|yuck',
                r'\b(gross|disgusting|yuck)\b'
            ]
        }
    
    async def analyze_emotions(
        self, 
        text: str, 
        config: AnalysisConfig
    ) -> Dict[EmotionType, float]:
        """
        Analyze emotions in text
        
        Args:
            text: Text to analyze
            config: Analysis configuration
            
        Returns:
            Dictionary mapping emotion types to scores (0.0-1.0)
        """
        try:
            text_lower = text.lower()
            emotion_scores = {}
            
            # Initialize all emotions to 0
            for emotion in EmotionType:
                emotion_scores[emotion] = 0.0
            
            # Word-based emotion detection
            word_scores = self._analyze_emotion_words(text_lower)
            
            # Pattern-based emotion detection
            pattern_scores = self._analyze_emotion_patterns(text)
            
            # Combine scores
            for emotion in EmotionType:
                combined_score = (word_scores.get(emotion, 0.0) + 
                                pattern_scores.get(emotion, 0.0))
                
                # Normalize to 0-1 range
                emotion_scores[emotion] = min(1.0, combined_score)
            
            return emotion_scores
            
        except Exception as e:
            logger.error(f"Emotion analysis failed: {str(e)}")
            # Return neutral emotions
            return {emotion: 0.0 for emotion in EmotionType}
    
    def _analyze_emotion_words(self, text_lower: str) -> Dict[EmotionType, float]:
        """Analyze emotions based on word lexicons"""
        word_scores = {}
        words = text_lower.split()
        total_words = len(words)
        
        if total_words == 0:
            return {emotion: 0.0 for emotion in EmotionType}
        
        for emotion, word_list in self.emotion_lexicons.items():
            matches = sum(1 for word in words if word in word_list)
            # Score based on proportion of emotion words
            word_scores[emotion] = matches / total_words
        
        return word_scores
    
    def _analyze_emotion_patterns(self, text: str) -> Dict[EmotionType, float]:
        """Analyze emotions based on patterns (emojis, emoticons, etc.)"""
        pattern_scores = {}
        
        for emotion, patterns in self.emotion_patterns.items():
            total_matches = 0
            
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                total_matches += matches
            
            # Normalize by text length (characters)
            text_length = max(len(text), 1)
            pattern_scores[emotion] = min(1.0, total_matches / (text_length / 100))
        
        return pattern_scores
    
    def get_dominant_emotion(self, emotion_scores: Dict[EmotionType, float]) -> Optional[EmotionType]:
        """Get the most dominant emotion"""
        if not emotion_scores:
            return None
        
        max_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        
        # Only return if score is above threshold
        if max_emotion[1] > 0.1:
            return max_emotion[0]
        
        return None
    
    def get_emotion_summary(self, emotion_scores: Dict[EmotionType, float]) -> str:
        """Get a human-readable summary of emotions"""
        if not emotion_scores:
            return "No emotions detected"
        
        # Filter emotions with significant scores
        significant_emotions = {
            emotion: score for emotion, score in emotion_scores.items() 
            if score > 0.1
        }
        
        if not significant_emotions:
            return "Neutral emotional tone"
        
        # Sort by score
        sorted_emotions = sorted(
            significant_emotions.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Build summary
        parts = []
        for emotion, score in sorted_emotions[:3]:  # Top 3 emotions
            intensity = "strong" if score > 0.5 else "moderate" if score > 0.3 else "mild"
            parts.append(f"{intensity} {emotion.value}")
        
        return f"Detected emotions: {', '.join(parts)}"
    
    async def batch_analyze_emotions(
        self, 
        texts: List[str], 
        config: AnalysisConfig
    ) -> List[Dict[EmotionType, float]]:
        """
        Analyze emotions for multiple texts
        
        Args:
            texts: List of texts to analyze
            config: Analysis configuration
            
        Returns:
            List of emotion score dictionaries
        """
        results = []
        
        for text in texts:
            emotion_scores = await self.analyze_emotions(text, config)
            results.append(emotion_scores)
        
        return results
