"""
Main hybrid sentiment analyzer combining rule-based and Gemini approaches
"""
import asyncio
import time
import logging
from typing import Dict, List, Optional, Union, Any
import os

from .models import (
    SentimentResult, SentimentMethod, AnalysisConfig, 
    BatchAnalysisResult, EmotionType
)
from .rule_based import EnhancedRuleBasedAnalyzer
from .gemini_client import GeminiClient
from .emotions import EmotionAnalyzer
from .aspects import AspectAnalyzer


logger = logging.getLogger(__name__)


class HybridSentimentAnalyzer:
    """
    Advanced sentiment analyzer combining rule-based and Gemini approaches
    """
    
    def __init__(self, gemini_api_key: Optional[str] = None):
        """
        Initialize hybrid sentiment analyzer
        
        Args:
            gemini_api_key: Google API key for Gemini. If None, will try to get from environment
        """
        self.rule_based_analyzer = EnhancedRuleBasedAnalyzer()
        self.emotion_analyzer = EmotionAnalyzer()
        self.aspect_analyzer = AspectAnalyzer()
        
        # Initialize Gemini client if API key is provided
        self.gemini_client = None
        if gemini_api_key:
            self.gemini_client = GeminiClient(gemini_api_key)
        elif os.getenv('GEMINI_API_KEY'):
            gemini_key = os.getenv('GEMINI_API_KEY')
            if gemini_key:
                self.gemini_client = GeminiClient(gemini_key)
        else:
            logger.warning("No Gemini API key provided. Gemini analysis will be unavailable.")
    
    async def analyze(
        self, 
        text: str, 
        config: Optional[AnalysisConfig] = None
    ) -> SentimentResult:
        """
        Analyze sentiment of text using specified method
        
        Args:
            text: Text to analyze
            config: Analysis configuration. If None, uses default hybrid config
            
        Returns:
            SentimentResult with comprehensive analysis
        """
        if config is None:
            config = AnalysisConfig()
        
        start_time = time.time()
        
        try:
            if config.method == SentimentMethod.RULE_BASED:
                return await self._analyze_rule_based(text, config)
            
            elif config.method == SentimentMethod.GEMINI:
                if not self.gemini_client:
                    logger.warning("Gemini client not available, falling back to rule-based")
                    return await self._analyze_rule_based(text, config)
                return await self._analyze_gemini(text, config)
            
            elif config.method == SentimentMethod.HYBRID:
                return await self._analyze_hybrid(text, config)
            
            elif config.method == SentimentMethod.ENSEMBLE:
                return await self._analyze_ensemble(text, config)
            
            else:
                raise ValueError(f"Unknown analysis method: {config.method}")
                
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return SentimentResult(
                polarity=0.0,
                confidence=0.0,
                method=config.method,
                explanation=f"Analysis failed: {str(e)}",
                processing_time=time.time() - start_time
            )
    
    async def _analyze_rule_based(self, text: str, config: AnalysisConfig) -> SentimentResult:
        """Perform rule-based sentiment analysis"""
        result = await self.rule_based_analyzer.analyze_sentiment(text, config)
        
        # Add additional analysis if requested
        if config.include_emotions:
            result.emotions = await self.emotion_analyzer.analyze_emotions(text, config)
        
        if config.include_aspects:
            result.aspects = await self.aspect_analyzer.analyze_aspects(text, config)
        
        return result
    
    async def _analyze_gemini(self, text: str, config: AnalysisConfig) -> SentimentResult:
        """Perform Gemini-based sentiment analysis"""
        if not self.gemini_client:
            raise ValueError("Gemini client not available")
        
        return await self.gemini_client.analyze_sentiment(text, config)
    
    async def _analyze_hybrid(self, text: str, config: AnalysisConfig) -> SentimentResult:
        """Perform hybrid analysis combining rule-based and Gemini"""
        # Run both analyses concurrently
        tasks = [
            self.rule_based_analyzer.analyze_sentiment(text, config)
        ]
        
        if self.gemini_client:
            tasks.append(self.gemini_client.analyze_sentiment(text, config))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        rule_based_result = results[0] if not isinstance(results[0], Exception) else None
        gemini_result = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else None
        
        # Combine results
        if rule_based_result and gemini_result and isinstance(rule_based_result, SentimentResult) and isinstance(gemini_result, SentimentResult):
            combined_result = self._combine_results(rule_based_result, gemini_result, config)
        elif rule_based_result and isinstance(rule_based_result, SentimentResult):
            combined_result = rule_based_result
            combined_result.method = SentimentMethod.HYBRID
            logger.warning("Gemini analysis failed, using rule-based only")
        else:
            # Both failed, return neutral result
            combined_result = SentimentResult(
                polarity=0.0,
                confidence=0.0,
                method=SentimentMethod.HYBRID,
                explanation="Both analyses failed"
            )
        
        # Add additional analysis if requested
        if config.include_emotions and not combined_result.emotions:
            combined_result.emotions = await self.emotion_analyzer.analyze_emotions(text, config)
        
        if config.include_aspects and not combined_result.aspects:
            combined_result.aspects = await self.aspect_analyzer.analyze_aspects(text, config)
        
        return combined_result
    
    async def _analyze_ensemble(self, text: str, config: AnalysisConfig) -> SentimentResult:
        """Perform ensemble analysis using multiple methods"""
        # For now, ensemble is the same as hybrid
        # In the future, this could include additional models
        return await self._analyze_hybrid(text, config)
    
    def _combine_results(
        self, 
        rule_based_result: SentimentResult, 
        gemini_result: SentimentResult, 
        config: AnalysisConfig
    ) -> SentimentResult:
        """Combine results from rule-based and Gemini analyses"""
        
        # Get ensemble weights
        rule_weight = config.ensemble_weights.get("rule_based", 0.4)
        gemini_weight = config.ensemble_weights.get("gemini", 0.6)
        
        # Normalize weights
        total_weight = rule_weight + gemini_weight
        rule_weight = rule_weight / total_weight
        gemini_weight = gemini_weight / total_weight
        
        # Combine polarities
        combined_polarity = (
            rule_based_result.polarity * rule_weight + 
            gemini_result.polarity * gemini_weight
        )
        
        # Combine confidences (take weighted average)
        combined_confidence = (
            rule_based_result.confidence * rule_weight + 
            gemini_result.confidence * gemini_weight
        )
        
        # Combine subjectivity if available
        combined_subjectivity = None
        if rule_based_result.subjectivity is not None and gemini_result.subjectivity is not None:
            combined_subjectivity = (
                rule_based_result.subjectivity * rule_weight + 
                gemini_result.subjectivity * gemini_weight
            )
        elif rule_based_result.subjectivity is not None:
            combined_subjectivity = rule_based_result.subjectivity
        elif gemini_result.subjectivity is not None:
            combined_subjectivity = gemini_result.subjectivity
        
        # Combine intensity if available
        combined_intensity = None
        if rule_based_result.intensity is not None and gemini_result.intensity is not None:
            combined_intensity = (
                rule_based_result.intensity * rule_weight + 
                gemini_result.intensity * gemini_weight
            )
        elif rule_based_result.intensity is not None:
            combined_intensity = rule_based_result.intensity
        elif gemini_result.intensity is not None:
            combined_intensity = gemini_result.intensity
        
        # Combine emotions if available
        combined_emotions = None
        if gemini_result.emotions:
            combined_emotions = gemini_result.emotions
        elif rule_based_result.emotions:
            combined_emotions = rule_based_result.emotions
        
        # Combine aspects if available
        combined_aspects = None
        if gemini_result.aspects:
            combined_aspects = gemini_result.aspects
        elif rule_based_result.aspects:
            combined_aspects = rule_based_result.aspects
        
        # Build explanation
        explanation = None
        if config.include_explanation:
            explanation = self._build_hybrid_explanation(
                rule_based_result, gemini_result, combined_polarity, config
            )
        
        # Combine detected features
        combined_features = []
        if rule_based_result.detected_features:
            combined_features.extend(rule_based_result.detected_features)
        if gemini_result.detected_features:
            combined_features.extend(gemini_result.detected_features)
        
        combined_features = list(set(combined_features)) if combined_features else None
        
        return SentimentResult(
            polarity=combined_polarity,
            confidence=combined_confidence,
            method=SentimentMethod.HYBRID,
            emotions=combined_emotions,
            aspects=combined_aspects,
            subjectivity=combined_subjectivity,
            intensity=combined_intensity,
            explanation=explanation,
            detected_features=combined_features,
            processing_time=(rule_based_result.processing_time or 0) + (gemini_result.processing_time or 0),
            rule_based_score=rule_based_result.polarity,
            gemini_score=gemini_result.polarity,
            word_count=rule_based_result.word_count or gemini_result.word_count,
            sentence_count=rule_based_result.sentence_count or gemini_result.sentence_count,
            language=config.language
        )
    
    def _build_hybrid_explanation(
        self, 
        rule_based_result: SentimentResult, 
        gemini_result: SentimentResult, 
        combined_polarity: float,
        config: AnalysisConfig
    ) -> str:
        """Build explanation for hybrid analysis"""
        
        parts = []
        
        # Overall sentiment
        if combined_polarity > 0.1:
            parts.append(f"Overall sentiment: Positive ({combined_polarity:.3f})")
        elif combined_polarity < -0.1:
            parts.append(f"Overall sentiment: Negative ({combined_polarity:.3f})")
        else:
            parts.append(f"Overall sentiment: Neutral ({combined_polarity:.3f})")
        
        # Component scores
        parts.append(f"Rule-based score: {rule_based_result.polarity:.3f}")
        parts.append(f"Gemini score: {gemini_result.polarity:.3f}")
        
        # Weights
        rule_weight = config.ensemble_weights.get("rule_based", 0.4)
        gemini_weight = config.ensemble_weights.get("gemini", 0.6)
        parts.append(f"Weights: Rule-based {rule_weight:.1f}, Gemini {gemini_weight:.1f}")
        
        # Individual explanations
        if rule_based_result.explanation:
            parts.append(f"Rule-based analysis: {rule_based_result.explanation}")
        
        if gemini_result.explanation:
            parts.append(f"Gemini analysis: {gemini_result.explanation}")
        
        return ". ".join(parts)
    
    async def batch_analyze(
        self, 
        texts: List[str], 
        config: Optional[AnalysisConfig] = None,
        batch_size: int = 10
    ) -> BatchAnalysisResult:
        """
        Analyze multiple texts in batches
        
        Args:
            texts: List of texts to analyze
            config: Analysis configuration
            batch_size: Number of texts to process concurrently
            
        Returns:
            BatchAnalysisResult with all results and statistics
        """
        if config is None:
            config = AnalysisConfig()
        
        start_time = time.time()
        results = []
        successful = 0
        failed = 0
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [self.analyze(text, config) for text in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle results and exceptions
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Failed to analyze text: {str(result)}")
                    failed += 1
                    # Create error result
                    error_result = SentimentResult(
                        polarity=0.0,
                        confidence=0.0,
                        method=config.method,
                        explanation=f"Analysis failed: {str(result)}"
                    )
                    results.append(error_result)
                else:
                    successful += 1
                    results.append(result)
        
        total_time = time.time() - start_time
        avg_time = total_time / len(texts) if texts else 0
        
        return BatchAnalysisResult(
            results=results,
            total_texts=len(texts),
            successful_analyses=successful,
            failed_analyses=failed,
            average_processing_time=avg_time,
            total_processing_time=total_time
        )
    
    def set_gemini_api_key(self, api_key: str) -> None:
        """Set or update Gemini API key"""
        self.gemini_client = GeminiClient(api_key)
        logger.info("Gemini client initialized with new API key")
    
    def is_gemini_available(self) -> bool:
        """Check if Gemini client is available"""
        return self.gemini_client is not None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics from all components"""
        stats = {}
        
        if self.gemini_client:
            stats["gemini"] = self.gemini_client.get_cache_stats()
        
        return stats
    
    def clear_caches(self) -> None:
        """Clear all caches"""
        if self.gemini_client:
            self.gemini_client.clear_cache()
        
        logger.info("All caches cleared")
