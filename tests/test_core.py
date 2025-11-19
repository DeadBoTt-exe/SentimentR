"""
Tests for core sentiment analysis functionality
"""
import pytest
import asyncio
from enhanced_sentimentr.core.analyzer import HybridSentimentAnalyzer
from enhanced_sentimentr.core.models import SentimentResult, SentimentMethod, AnalysisConfig


class TestBasicSentimentAnalysis:
    """Test basic sentiment analysis functionality"""
    
    @pytest.mark.asyncio
    async def test_positive_sentiment(self, analyzer, sample_texts):
        """Test detection of positive sentiment"""
        config = AnalysisConfig(method=SentimentMethod.RULE_BASED)
        
        for text in sample_texts["positive"]:
            result = await analyzer.analyze(text, config)
            
            assert isinstance(result, SentimentResult)
            assert result.polarity > 0, f"Expected positive sentiment for: {text}"
            assert 0 <= result.confidence <= 1
            assert result.method == SentimentMethod.RULE_BASED
    
    @pytest.mark.asyncio
    async def test_negative_sentiment(self, analyzer, sample_texts):
        """Test detection of negative sentiment"""
        config = AnalysisConfig(method=SentimentMethod.RULE_BASED)
        
        for text in sample_texts["negative"]:
            result = await analyzer.analyze(text, config)
            
            assert isinstance(result, SentimentResult)
            assert result.polarity < 0, f"Expected negative sentiment for: {text}"
            assert 0 <= result.confidence <= 1
            assert result.method == SentimentMethod.RULE_BASED
    
    @pytest.mark.asyncio
    async def test_neutral_sentiment(self, analyzer, sample_texts):
        """Test detection of neutral sentiment"""
        config = AnalysisConfig(method=SentimentMethod.RULE_BASED)
        
        for text in sample_texts["neutral"]:
            result = await analyzer.analyze(text, config)
            
            assert isinstance(result, SentimentResult)
            # Neutral texts should have polarity close to 0
            assert abs(result.polarity) <= 0.2, f"Expected neutral sentiment for: {text}"
            assert 0 <= result.confidence <= 1
            assert result.method == SentimentMethod.RULE_BASED
    
    @pytest.mark.asyncio
    async def test_polarity_range(self, analyzer):
        """Test that polarity is within expected range"""
        texts = [
            "I absolutely love this amazing product!",
            "This is completely terrible and awful.",
            "The weather is okay today."
        ]
        
        config = AnalysisConfig(method=SentimentMethod.RULE_BASED)
        
        for text in texts:
            result = await analyzer.analyze(text, config)
            assert -1.0 <= result.polarity <= 1.0, f"Polarity out of range for: {text}"
    
    @pytest.mark.asyncio
    async def test_confidence_range(self, analyzer):
        """Test that confidence is within expected range"""
        texts = [
            "This is great!",
            "This is bad.",
            "Hello world."
        ]
        
        config = AnalysisConfig(method=SentimentMethod.RULE_BASED)
        
        for text in texts:
            result = await analyzer.analyze(text, config)
            assert 0.0 <= result.confidence <= 1.0, f"Confidence out of range for: {text}"


class TestAnalysisConfiguration:
    """Test different analysis configurations"""
    
    @pytest.mark.asyncio
    async def test_emotion_analysis(self, analyzer):
        """Test emotion analysis inclusion"""
        text = "I'm so happy and excited about this!"
        config = AnalysisConfig(
            method=SentimentMethod.RULE_BASED,
            include_emotions=True
        )
        
        result = await analyzer.analyze(text, config)
        
        assert result.emotions is not None
        assert isinstance(result.emotions, dict)
        # Should have some joy/happiness
        if result.emotions:
            assert any(score > 0 for score in result.emotions.values())
    
    @pytest.mark.asyncio
    async def test_aspect_analysis(self, analyzer):
        """Test aspect analysis inclusion"""
        text = "The food was delicious but the service was slow."
        config = AnalysisConfig(
            method=SentimentMethod.RULE_BASED,
            include_aspects=True
        )
        
        result = await analyzer.analyze(text, config)
        
        assert result.aspects is not None
        # Should detect food and service aspects
        if result.aspects:
            assert isinstance(result.aspects, dict)
    
    @pytest.mark.asyncio
    async def test_explanation_inclusion(self, analyzer):
        """Test explanation inclusion"""
        text = "This product is amazing!"
        config = AnalysisConfig(
            method=SentimentMethod.RULE_BASED,
            include_explanation=True
        )
        
        result = await analyzer.analyze(text, config)
        
        assert result.explanation is not None
        assert isinstance(result.explanation, str)
        assert len(result.explanation) > 0


class TestBatchAnalysis:
    """Test batch analysis functionality"""
    
    @pytest.mark.asyncio
    async def test_batch_analysis_basic(self, analyzer, sample_texts):
        """Test basic batch analysis"""
        texts = sample_texts["positive"][:3] + sample_texts["negative"][:3]
        config = AnalysisConfig(method=SentimentMethod.RULE_BASED)
        
        batch_result = await analyzer.batch_analyze(texts, config)
        
        assert batch_result.total_texts == len(texts)
        assert batch_result.successful_analyses <= len(texts)
        assert len(batch_result.results) == len(texts)
        
        # Check individual results
        for result in batch_result.results:
            assert isinstance(result, SentimentResult)
            assert -1.0 <= result.polarity <= 1.0
            assert 0.0 <= result.confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_batch_analysis_timing(self, analyzer, sample_texts):
        """Test batch analysis timing"""
        texts = sample_texts["positive"][:5]
        config = AnalysisConfig(method=SentimentMethod.RULE_BASED)
        
        batch_result = await analyzer.batch_analyze(texts, config)
        
        assert batch_result.total_processing_time > 0
        assert batch_result.average_processing_time > 0
        assert batch_result.average_processing_time <= batch_result.total_processing_time


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.mark.asyncio
    async def test_empty_text(self, analyzer):
        """Test handling of empty text"""
        config = AnalysisConfig(method=SentimentMethod.RULE_BASED)
        
        result = await analyzer.analyze("", config)
        
        # Should not crash and should return a valid result
        assert isinstance(result, SentimentResult)
        assert result.polarity == 0.0  # Empty text should be neutral
    
    @pytest.mark.asyncio
    async def test_very_long_text(self, analyzer):
        """Test handling of very long text"""
        long_text = "This is great! " * 1000  # Very long text
        config = AnalysisConfig(method=SentimentMethod.RULE_BASED)
        
        result = await analyzer.analyze(long_text, config)
        
        # Should not crash and should detect positive sentiment
        assert isinstance(result, SentimentResult)
        assert result.polarity > 0
    
    @pytest.mark.asyncio
    async def test_special_characters(self, analyzer):
        """Test handling of special characters"""
        texts = [
            "This is great!!! 😊😊😊",
            "Bad product :-(",
            "Neutral text with numbers 123 and symbols @#$%"
        ]
        
        config = AnalysisConfig(method=SentimentMethod.RULE_BASED)
        
        for text in texts:
            result = await analyzer.analyze(text, config)
            assert isinstance(result, SentimentResult)
            # Should not crash
    
    @pytest.mark.asyncio
    async def test_unicode_text(self, analyzer):
        """Test handling of unicode text"""
        texts = [
            "C'est fantastique!",  # French
            "¡Esto es genial!",    # Spanish
            "这很棒！",             # Chinese
            "これは素晴らしい！"      # Japanese
        ]
        
        config = AnalysisConfig(method=SentimentMethod.RULE_BASED)
        
        for text in texts:
            result = await analyzer.analyze(text, config)
            assert isinstance(result, SentimentResult)
            # Should not crash


class TestMethodComparison:
    """Test comparison between different analysis methods"""
    
    @pytest.mark.asyncio
    async def test_rule_based_vs_hybrid(self, analyzer):
        """Compare rule-based and hybrid methods"""
        text = "This product is absolutely amazing!"
        
        rule_config = AnalysisConfig(method=SentimentMethod.RULE_BASED)
        hybrid_config = AnalysisConfig(method=SentimentMethod.HYBRID)
        
        rule_result = await analyzer.analyze(text, rule_config)
        hybrid_result = await analyzer.analyze(text, hybrid_config)
        
        # Both should detect positive sentiment
        assert rule_result.polarity > 0
        assert hybrid_result.polarity > 0
        
        # Methods should be correctly set
        assert rule_result.method == SentimentMethod.RULE_BASED
        assert hybrid_result.method == SentimentMethod.HYBRID
    
    @pytest.mark.asyncio
    async def test_consistency_across_methods(self, analyzer, sample_texts):
        """Test consistency across different methods"""
        positive_text = sample_texts["positive"][0]
        negative_text = sample_texts["negative"][0]
        
        methods = [SentimentMethod.RULE_BASED, SentimentMethod.HYBRID]
        
        for method in methods:
            config = AnalysisConfig(method=method)
            
            pos_result = await analyzer.analyze(positive_text, config)
            neg_result = await analyzer.analyze(negative_text, config)
            
            # Should consistently detect positive and negative
            assert pos_result.polarity > neg_result.polarity


if __name__ == "__main__":
    pytest.main([__file__])
