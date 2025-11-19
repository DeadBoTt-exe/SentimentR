"""
Gemini API client for sentiment analysis
"""
import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any
import google.generativeai as genai
from asyncio_throttle.throttler import Throttler

from .models import SentimentResult, SentimentMethod, EmotionType, AnalysisConfig


logger = logging.getLogger(__name__)


class GeminiClient:
    """
    Async Gemini client for sentiment analysis with rate limiting and caching
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        """
        Initialize Gemini client
        
        Args:
            api_key: Google API key for Gemini
            model_name: Gemini model to use (default: gemini-1.5-flash)
        """
        genai.configure(api_key=api_key)
        try:
            self.model = genai.GenerativeModel(model_name)
        except Exception as e:
            # Fallback to older model names
            try:
                self.model = genai.GenerativeModel("gemini-pro")
            except Exception:
                try:
                    self.model = genai.GenerativeModel("gemini-1.0-pro")
                except Exception:
                    raise ValueError(f"Could not initialize Gemini model. Error: {e}")
        
        self.throttler = Throttler(rate_limit=60, period=60)  # 60 requests per minute
        self._cache: Dict[str, Any] = {}
        
    async def analyze_sentiment(
        self, 
        text: str, 
        config: AnalysisConfig
    ) -> SentimentResult:
        """
        Analyze sentiment using Gemini
        
        Args:
            text: Text to analyze
            config: Analysis configuration
            
        Returns:
            SentimentResult with Gemini analysis
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(text, config)
            if config.cache_enabled and cache_key in self._cache:
                logger.debug(f"Cache hit for text: {text[:50]}...")
                cached_result = self._cache[cache_key]
                cached_result.processing_time = time.time() - start_time
                return cached_result
            
            # Rate limiting
            async with self.throttler:
                prompt = self._build_prompt(text, config)
                
                # Generate response
                response = await self._generate_with_retry(prompt, config)
                
                # Parse response
                result = self._parse_response(text, response, config)
                result.processing_time = time.time() - start_time
                
                # Cache result
                if config.cache_enabled:
                    self._cache[cache_key] = result
                
                return result
                
        except Exception as e:
            logger.error(f"Gemini analysis failed: {str(e)}")
            # Return neutral result as fallback
            return SentimentResult(
                polarity=0.0,
                confidence=0.0,
                method=SentimentMethod.GEMINI,
                explanation=f"Analysis failed: {str(e)}",
                processing_time=time.time() - start_time
            )
    
    def _build_prompt(self, text: str, config: AnalysisConfig) -> str:
        """Build prompt for Gemini based on configuration"""
        
        base_prompt = f"""
Analyze the sentiment of the following text with high precision:

Text: "{text}"

Please provide a detailed sentiment analysis in JSON format with the following structure:
{{
    "polarity": <float between -1.0 and 1.0>,
    "confidence": <float between 0.0 and 1.0>,
    "subjectivity": <float between 0.0 and 1.0>,
    "intensity": <float between 0.0 and 1.0>
"""
        
        if config.include_emotions:
            base_prompt += """,
    "emotions": {
        "joy": <float 0.0-1.0>,
        "sadness": <float 0.0-1.0>,
        "anger": <float 0.0-1.0>,
        "fear": <float 0.0-1.0>,
        "surprise": <float 0.0-1.0>,
        "disgust": <float 0.0-1.0>,
        "trust": <float 0.0-1.0>,
        "anticipation": <float 0.0-1.0>
    }"""
        
        if config.include_aspects:
            base_prompt += """,
    "aspects": {
        <detected aspects and their sentiment scores as key-value pairs>
    }"""
        
        if config.include_explanation:
            base_prompt += """,
    "explanation": "<detailed explanation of the sentiment analysis>"
            """
        
        base_prompt += """
}

Important guidelines:
- Polarity: -1.0 (very negative) to +1.0 (very positive), 0.0 is neutral
- Confidence: How certain you are about the analysis (0.0-1.0)
- Consider context, sarcasm, irony, and cultural nuances
- Handle informal language, slang, emojis, and emoticons appropriately
- Subjectivity: 0.0 (objective) to 1.0 (subjective)
- Intensity: How strong the sentiment is regardless of polarity
- Provide only valid JSON response, no additional text
"""
        
        return base_prompt
    
    async def _generate_with_retry(self, prompt: str, config: AnalysisConfig) -> str:
        """Generate response with retry logic"""
        
        for attempt in range(config.retry_attempts):
            try:
                # Use asyncio.wait_for for timeout
                response = await asyncio.wait_for(
                    self._generate_async(prompt, config),
                    timeout=config.timeout_seconds
                )
                return response
                
            except asyncio.TimeoutError:
                logger.warning(f"Gemini request timeout on attempt {attempt + 1}")
                if attempt == config.retry_attempts - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
            except Exception as e:
                logger.warning(f"Gemini request failed on attempt {attempt + 1}: {str(e)}")
                if attempt == config.retry_attempts - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
        
        raise Exception("All retry attempts failed")
    
    async def _generate_async(self, prompt: str, config: AnalysisConfig) -> str:
        """Async wrapper for Gemini generation"""
        loop = asyncio.get_event_loop()
        
        def _generate_sync():
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": config.gemini_temperature,
                    "max_output_tokens": config.gemini_max_tokens,
                }
            )
            return response.text
        
        return await loop.run_in_executor(None, _generate_sync)
    
    def _parse_response(self, text: str, response: str, config: AnalysisConfig) -> SentimentResult:
        """Parse Gemini response into SentimentResult"""
        
        try:
            # Clean response and parse JSON
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            
            data = json.loads(cleaned_response)
            
            # Extract basic sentiment data
            polarity = float(data.get("polarity", 0.0))
            confidence = float(data.get("confidence", 0.0))
            subjectivity = data.get("subjectivity")
            intensity = data.get("intensity")
            explanation = data.get("explanation")
            
            # Parse emotions if included
            emotions = None
            if config.include_emotions and "emotions" in data:
                emotions = {}
                for emotion_name, score in data["emotions"].items():
                    try:
                        emotion_type = EmotionType(emotion_name.lower())
                        emotions[emotion_type] = float(score)
                    except (ValueError, TypeError):
                        continue
            
            # Parse aspects if included
            aspects = None
            if config.include_aspects and "aspects" in data:
                aspects = {k: float(v) for k, v in data["aspects"].items() 
                        if isinstance(v, (int, float))}
            
            return SentimentResult(
                polarity=polarity,
                confidence=confidence,
                method=SentimentMethod.GEMINI,
                emotions=emotions,
                aspects=aspects,
                subjectivity=subjectivity,
                intensity=intensity,
                explanation=explanation,
                gemini_score=polarity,
                word_count=len(text.split()),
                sentence_count=len([s for s in text.split('.') if s.strip()]),
                language=config.language
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse Gemini response: {str(e)}")
            logger.debug(f"Raw response: {response}")
            
            # Fallback: try to extract polarity with regex
            import re
            polarity_match = re.search(r'"polarity":\s*(-?\d+\.?\d*)', response)
            polarity = float(polarity_match.group(1)) if polarity_match else 0.0
            
            return SentimentResult(
                polarity=polarity,
                confidence=0.5,
                method=SentimentMethod.GEMINI,
                explanation=f"Partial analysis due to parsing error: {str(e)}",
                gemini_score=polarity
            )
    
    def _get_cache_key(self, text: str, config: AnalysisConfig) -> str:
        """Generate cache key for text and config"""
        import hashlib
        
        # Create a key based on text and relevant config parameters
        config_str = f"{config.method.value}_{config.include_emotions}_{config.include_aspects}_{config.include_explanation}_{config.gemini_temperature}"
        combined = f"{text}_{config_str}"
        
        return hashlib.md5(combined.encode()).hexdigest()
    
    async def batch_analyze(
        self, 
        texts: List[str], 
        config: AnalysisConfig,
        batch_size: int = 10
    ) -> List[SentimentResult]:
        """
        Analyze multiple texts in batches
        
        Args:
            texts: List of texts to analyze
            config: Analysis configuration
            batch_size: Number of texts to process concurrently
            
        Returns:
            List of SentimentResult objects
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [self.analyze_sentiment(text, config) for text in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to analyze text {i+j}: {str(result)}")
                    # Create error result
                    error_result = SentimentResult(
                        polarity=0.0,
                        confidence=0.0,
                        method=SentimentMethod.GEMINI,
                        explanation=f"Analysis failed: {str(result)}"
                    )
                    results.append(error_result)
                else:
                    results.append(result)
        
        return results
    
    def clear_cache(self) -> None:
        """Clear the response cache"""
        self._cache.clear()
        logger.info("Gemini cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            "cache_size": len(self._cache),
            "memory_usage_mb": sum(len(str(v)) for v in self._cache.values()) / (1024 * 1024)
        }
