"""
Core data models for enhanced sentiment analysis
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from enum import Enum
import json
from datetime import datetime


class SentimentMethod(str, Enum):
    """Available sentiment analysis methods"""
    RULE_BASED = "rule_based"
    GEMINI = "gemini"
    HYBRID = "hybrid"
    ENSEMBLE = "ensemble"


class EmotionType(str, Enum):
    """Basic emotion types"""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"


@dataclass
class SentimentResult:
    """
    Comprehensive sentiment analysis result
    """
    polarity: float  # -1.0 to +1.0
    confidence: float  # 0.0 to 1.0
    method: SentimentMethod
    
    # Optional detailed analysis
    emotions: Optional[Dict[EmotionType, float]] = None
    aspects: Optional[Dict[str, float]] = None
    subjectivity: Optional[float] = None
    intensity: Optional[float] = None
    
    # Explanations and metadata
    explanation: Optional[str] = None
    detected_features: Optional[List[str]] = None
    processing_time: Optional[float] = None
    
    # Raw scores from different methods
    rule_based_score: Optional[float] = None
    gemini_score: Optional[float] = None
    
    # Text analysis metadata
    word_count: Optional[int] = None
    sentence_count: Optional[int] = None
    language: Optional[str] = None
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "polarity": self.polarity,
            "confidence": self.confidence,
            "method": self.method.value,
            "timestamp": self.timestamp.isoformat(),
        }
        
        if self.emotions:
            result["emotions"] = {k.value: v for k, v in self.emotions.items()}
        if self.aspects:
            result["aspects"] = self.aspects
        if self.subjectivity is not None:
            result["subjectivity"] = self.subjectivity
        if self.intensity is not None:
            result["intensity"] = self.intensity
        if self.explanation:
            result["explanation"] = self.explanation
        if self.detected_features:
            result["detected_features"] = self.detected_features
        if self.processing_time is not None:
            result["processing_time"] = self.processing_time
        if self.rule_based_score is not None:
            result["rule_based_score"] = self.rule_based_score
        if self.gemini_score is not None:
            result["gemini_score"] = self.gemini_score
        if self.word_count is not None:
            result["word_count"] = self.word_count
        if self.sentence_count is not None:
            result["sentence_count"] = self.sentence_count
        if self.language:
            result["language"] = self.language
            
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SentimentResult":
        """Create from dictionary"""
        # Convert emotions back to enum
        emotions = None
        if "emotions" in data and data["emotions"]:
            emotions = {EmotionType(k): v for k, v in data["emotions"].items()}
        
        # Parse timestamp
        timestamp = datetime.now()
        if "timestamp" in data:
            timestamp = datetime.fromisoformat(data["timestamp"])
        
        return cls(
            polarity=data["polarity"],
            confidence=data["confidence"],
            method=SentimentMethod(data["method"]),
            emotions=emotions,
            aspects=data.get("aspects"),
            subjectivity=data.get("subjectivity"),
            intensity=data.get("intensity"),
            explanation=data.get("explanation"),
            detected_features=data.get("detected_features"),
            processing_time=data.get("processing_time"),
            rule_based_score=data.get("rule_based_score"),
            gemini_score=data.get("gemini_score"),
            word_count=data.get("word_count"),
            sentence_count=data.get("sentence_count"),
            language=data.get("language"),
            timestamp=timestamp,
        )


@dataclass
class BatchAnalysisResult:
    """Result for batch sentiment analysis"""
    results: List[SentimentResult]
    total_texts: int
    successful_analyses: int
    failed_analyses: int
    average_processing_time: float
    total_processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "total_texts": self.total_texts,
            "successful_analyses": self.successful_analyses,
            "failed_analyses": self.failed_analyses,
            "average_processing_time": self.average_processing_time,
            "total_processing_time": self.total_processing_time,
        }


@dataclass
class AnalysisConfig:
    """Configuration for sentiment analysis"""
    method: SentimentMethod = SentimentMethod.HYBRID
    include_emotions: bool = False
    include_aspects: bool = False
    include_explanation: bool = False
    language: str = "en"
    
    # Gemini-specific settings
    gemini_model: str = "gemini-pro"
    gemini_temperature: float = 0.1
    gemini_max_tokens: int = 1000
    
    # Rule-based settings
    rule_based_verbose: bool = False
    rule_based_subjectivity: bool = False
    
    # Hybrid settings
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        "rule_based": 0.4,
        "gemini": 0.6
    })
    
    # Performance settings
    cache_enabled: bool = True
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method.value,
            "include_emotions": self.include_emotions,
            "include_aspects": self.include_aspects,
            "include_explanation": self.include_explanation,
            "language": self.language,
            "gemini_model": self.gemini_model,
            "gemini_temperature": self.gemini_temperature,
            "gemini_max_tokens": self.gemini_max_tokens,
            "rule_based_verbose": self.rule_based_verbose,
            "rule_based_subjectivity": self.rule_based_subjectivity,
            "ensemble_weights": self.ensemble_weights,
            "cache_enabled": self.cache_enabled,
            "timeout_seconds": self.timeout_seconds,
            "retry_attempts": self.retry_attempts,
        }
