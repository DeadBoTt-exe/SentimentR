"""
Enhanced rule-based sentiment analyzer based on original sentimentr
"""
import time
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .models import SentimentResult, SentimentMethod, AnalysisConfig
from .sentiment_wrapper import get_sentiment_score


logger = logging.getLogger(__name__)


class EnhancedRuleBasedAnalyzer:
    """
    Enhanced version of the original rule-based sentiment analyzer
    """
    
    def __init__(self):
        """Initialize the enhanced rule-based analyzer"""
        # Note: Original Sentiment class has static methods
        pass
        
    async def analyze_sentiment(
        self, 
        text: str, 
        config: AnalysisConfig
    ) -> SentimentResult:
        """
        Analyze sentiment using enhanced rule-based approach
        
        Args:
            text: Text to analyze
            config: Analysis configuration
            
        Returns:
            SentimentResult with rule-based analysis
        """
        start_time = time.time()
        
        try:
            # Use original analyzer with subjectivity if requested
            if config.rule_based_subjectivity:
                raw_result = get_sentiment_score(
                    text, 
                    subjectivity=True, 
                    verbose=config.rule_based_verbose
                )
                
                if isinstance(raw_result, dict):
                    polarity = raw_result.get('polarity', 0.0)
                    pos_portion = raw_result.get('pos portion', 0.0)
                    neg_portion = raw_result.get('neg portion', 0.0)
                    neutral_portion = raw_result.get('neutral portion', 1.0)
                    
                    # Calculate confidence based on portion distribution
                    confidence = max(pos_portion, neg_portion)
                    subjectivity = 1.0 - neutral_portion
                else:
                    polarity = raw_result
                    confidence = abs(polarity) if polarity != 0 else 0.1
                    subjectivity = None
            else:
                polarity = get_sentiment_score(
                    text, 
                    subjectivity=False,
                    verbose=config.rule_based_verbose
                )
                confidence = abs(polarity) if polarity != 0 else 0.1
                subjectivity = None
            
            # Calculate intensity (absolute value of polarity)
            intensity = abs(polarity)
            
            # Extract detected features if verbose was enabled
            detected_features = self._extract_features(text) if config.rule_based_verbose else None
            
            # Build explanation if requested
            explanation = None
            if config.include_explanation:
                explanation = self._build_explanation(text, polarity, detected_features)
            
            processing_time = time.time() - start_time
            
            return SentimentResult(
                polarity=polarity,
                confidence=confidence,
                method=SentimentMethod.RULE_BASED,
                subjectivity=subjectivity,
                intensity=intensity,
                explanation=explanation,
                detected_features=detected_features,
                processing_time=processing_time,
                rule_based_score=polarity,
                word_count=len(text.split()),
                sentence_count=len([s for s in text.split('.') if s.strip()]),
                language=config.language
            )
            
        except Exception as e:
            logger.error(f"Rule-based analysis failed: {str(e)}")
            return SentimentResult(
                polarity=0.0,
                confidence=0.0,
                method=SentimentMethod.RULE_BASED,
                explanation=f"Analysis failed: {str(e)}",
                processing_time=time.time() - start_time
            )
    
    def _extract_features(self, text: str) -> List[str]:
        """
        Extract linguistic features detected in the text
        """
        features = []
        text_lower = text.lower()
        
        # Check for various features based on original sentimentr rules
        if any(char.isupper() for char in text):
            features.append("emphatic_uppercasing")
        
        if '!' in text:
            exclamation_count = text.count('!')
            features.append(f"exclamations_{exclamation_count}")
        
        # Check for lengthening
        import re
        if re.search(r'([a-zA-Z])\1{2,}', text):
            features.append("emphatic_lengthening")
        
        # Check for emoticons
        emoticon_patterns = [':)', ':(', ':D', ':/']
        for pattern in emoticon_patterns:
            if pattern in text:
                features.append(f"emoticon_{pattern}")
        
        # Check for contrasting connectors
        connectors = ['but', 'however', 'although', 'nevertheless']
        for connector in connectors:
            if connector in text_lower:
                features.append(f"contrasting_connector_{connector}")
        
        # Check for negators
        negators = ['not', "n't", 'never', 'no']
        for negator in negators:
            if negator in text_lower:
                features.append(f"negator_{negator}")
        
        # Check for intensifiers
        intensifiers = ['very', 'extremely', 'absolutely', 'really']
        for intensifier in intensifiers:
            if intensifier in text_lower:
                features.append(f"intensifier_{intensifier}")
        
        return features
    
    def _build_explanation(
        self, 
        text: str, 
        polarity: float, 
        features: Optional[List[str]]
    ) -> str:
        """
        Build a human-readable explanation of the sentiment analysis
        """
        explanation_parts = []
        
        # Basic sentiment
        if polarity > 0.1:
            explanation_parts.append(f"The text shows positive sentiment (score: {polarity:.3f})")
        elif polarity < -0.1:
            explanation_parts.append(f"The text shows negative sentiment (score: {polarity:.3f})")
        else:
            explanation_parts.append(f"The text appears neutral (score: {polarity:.3f})")
        
        # Feature-based explanations
        if features:
            feature_explanations = []
            
            for feature in features:
                if feature == "emphatic_uppercasing":
                    feature_explanations.append("emphatic uppercasing detected (boosts sentiment)")
                elif feature.startswith("exclamations_"):
                    count = feature.split('_')[1]
                    feature_explanations.append(f"{count} exclamation mark(s) detected (amplifies sentiment)")
                elif feature == "emphatic_lengthening":
                    feature_explanations.append("emphatic lengthening detected (boosts sentiment)")
                elif feature.startswith("emoticon_"):
                    emoticon = feature.split('_')[1]
                    feature_explanations.append(f"emoticon '{emoticon}' detected")
                elif feature.startswith("contrasting_connector_"):
                    connector = feature.split('_')[2]
                    feature_explanations.append(f"contrasting connector '{connector}' creates mixed sentiment")
                elif feature.startswith("negator_"):
                    negator = feature.split('_')[1]
                    feature_explanations.append(f"negator '{negator}' reverses sentiment")
                elif feature.startswith("intensifier_"):
                    intensifier = feature.split('_')[1]
                    feature_explanations.append(f"intensifier '{intensifier}' amplifies sentiment")
            
            if feature_explanations:
                explanation_parts.append("Key features: " + "; ".join(feature_explanations))
        
        return ". ".join(explanation_parts) + "."
    
    async def batch_analyze(
        self, 
        texts: List[str], 
        config: AnalysisConfig
    ) -> List[SentimentResult]:
        """
        Analyze multiple texts using rule-based approach
        
        Args:
            texts: List of texts to analyze
            config: Analysis configuration
            
        Returns:
            List of SentimentResult objects
        """
        results = []
        
        for text in texts:
            result = await self.analyze_sentiment(text, config)
            results.append(result)
        
        return results
