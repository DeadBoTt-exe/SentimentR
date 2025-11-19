"""
Test configuration and fixtures
"""
import pytest
import asyncio
import os
from typing import Generator

from enhanced_sentimentr.core.analyzer import HybridSentimentAnalyzer
from enhanced_sentimentr.core.models import AnalysisConfig, SentimentMethod


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def analyzer() -> HybridSentimentAnalyzer:
    """Create a sentiment analyzer instance for testing"""
    # Don't use real API key in tests unless explicitly set
    api_key = os.getenv('GEMINI_API_KEY_TEST')
    return HybridSentimentAnalyzer(api_key)


@pytest.fixture
def sample_texts():
    """Sample texts for testing"""
    return {
        "positive": [
            "I love this product! It's amazing.",
            "This is the best day ever!",
            "Fantastic work, keep it up!",
            "I'm so happy and grateful.",
            "Excellent quality and great service."
        ],
        "negative": [
            "This is terrible and I hate it.",
            "Worst experience ever.",
            "I'm so disappointed and angry.",
            "This product is awful and broken.",
            "Terrible service, never coming back."
        ],
        "neutral": [
            "The weather is cloudy today.",
            "I went to the store.",
            "The meeting is at 3 PM.",
            "Please send the report.",
            "The file is in the folder."
        ],
        "mixed": [
            "The food was great but the service was terrible.",
            "I love the design but hate the price.",
            "Good product but poor packaging.",
            "Nice features but too complicated to use.",
            "Great idea but bad execution."
        ]
    }


@pytest.fixture
def test_configs():
    """Different configuration options for testing"""
    return {
        "rule_based": AnalysisConfig(method=SentimentMethod.RULE_BASED),
        "hybrid": AnalysisConfig(method=SentimentMethod.HYBRID),
        "with_emotions": AnalysisConfig(
            method=SentimentMethod.RULE_BASED,
            include_emotions=True
        ),
        "with_aspects": AnalysisConfig(
            method=SentimentMethod.RULE_BASED,
            include_aspects=True
        ),
        "full_analysis": AnalysisConfig(
            method=SentimentMethod.RULE_BASED,
            include_emotions=True,
            include_aspects=True,
            include_explanation=True
        )
    }
