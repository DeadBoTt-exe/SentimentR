#!/usr/bin/env python3
"""
Enhanced SentimentR Demo Script

This script demonstrates all the major features of Enhanced SentimentR v2.0
including rule-based analysis, hybrid analysis, emotions, aspects, and more.
"""
import asyncio
import json
import time
from typing import List, Dict
import sys
import os

# Add the package to path for demo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from enhanced_sentimentr.core.analyzer import HybridSentimentAnalyzer
    from enhanced_sentimentr.core.models import AnalysisConfig, SentimentMethod, EmotionType
    from enhanced_sentimentr.core.emotions import EmotionAnalyzer
    from enhanced_sentimentr.core.aspects import AspectAnalyzer
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please make sure enhanced_sentimentr is properly installed")
    sys.exit(1)


def print_header(title: str):
    """Print a styled header"""
    print("\n" + "="*60)
    print(f"🎭 {title}")
    print("="*60)


def print_result(text: str, result, show_details: bool = True):
    """Print analysis result in a nice format"""
    polarity = result.polarity
    
    # Sentiment emoji and label
    if polarity > 0.5:
        emoji = "😊"
        label = "Very Positive"
        color = "\033[92m"  # Green
    elif polarity > 0.1:
        emoji = "🙂"
        label = "Positive"
        color = "\033[92m"  # Green
    elif polarity < -0.5:
        emoji = "😞"
        label = "Very Negative"
        color = "\033[91m"  # Red
    elif polarity < -0.1:
        emoji = "😐"
        label = "Negative"
        color = "\033[91m"  # Red
    else:
        emoji = "😶"
        label = "Neutral"
        color = "\033[93m"  # Yellow
    
    reset_color = "\033[0m"
    
    print(f"\n📝 Text: {text[:80]}{'...' if len(text) > 80 else ''}")
    print(f"{emoji} Sentiment: {color}{label}{reset_color} (Score: {polarity:.3f})")
    print(f"🎯 Confidence: {result.confidence:.1%}")
    print(f"⚙️ Method: {result.method.value}")
    
    if show_details:
        if result.subjectivity is not None:
            print(f"📊 Subjectivity: {result.subjectivity:.1%}")
        
        if result.intensity is not None:
            print(f"💪 Intensity: {result.intensity:.1%}")
        
        if result.processing_time:
            print(f"⏱️ Time: {result.processing_time:.3f}s")
        
        if result.emotions:
            print("😊 Emotions:")
            for emotion, score in result.emotions.items():
                if score > 0.1:
                    print(f"   • {emotion.value.title()}: {score:.1%}")
        
        if result.aspects:
            print("🎯 Aspects:")
            for aspect, score in result.aspects.items():
                sentiment = "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral"
                print(f"   • {aspect}: {sentiment} ({score:.3f})")
        
        if result.explanation:
            print(f"💡 Explanation: {result.explanation}")


async def demo_basic_analysis():
    """Demonstrate basic sentiment analysis"""
    print_header("Basic Sentiment Analysis")
    
    # Initialize analyzer
    analyzer = HybridSentimentAnalyzer()
    
    # Test texts
    test_texts = [
        "I absolutely love this product! It's amazing and works perfectly!",
        "This is the worst experience I've ever had. Completely disappointed.",
        "The weather is cloudy today and it might rain.",
        "The food was delicious but the service was terrible.",
        "OMG this is sooo good!!! 😍😍😍 Best purchase ever!!!"
    ]
    
    print("Testing different sentiment analysis methods...\n")
    
    for text in test_texts:
        print(f"🔍 Analyzing: '{text[:50]}...'")
        
        # Rule-based analysis
        config = AnalysisConfig(method=SentimentMethod.RULE_BASED)
        result = await analyzer.analyze(text, config)
        print(f"   Rule-based: {result.polarity:.3f} (confidence: {result.confidence:.1%})")
        
        # Hybrid analysis
        config = AnalysisConfig(method=SentimentMethod.HYBRID)
        result = await analyzer.analyze(text, config)
        print(f"   Hybrid: {result.polarity:.3f} (confidence: {result.confidence:.1%})")
        
        print()


async def demo_emotion_analysis():
    """Demonstrate emotion analysis"""
    print_header("Emotion Analysis")
    
    analyzer = HybridSentimentAnalyzer()
    
    emotion_texts = [
        "I'm so excited and happy about this opportunity!",
        "I'm really worried and scared about the future.",
        "This makes me so angry and frustrated!",
        "I feel sad and disappointed about what happened.",
        "What a wonderful surprise! I can't believe this happened!",
        "This is disgusting and revolting."
    ]
    
    config = AnalysisConfig(
        method=SentimentMethod.RULE_BASED,
        include_emotions=True
    )
    
    for text in emotion_texts:
        result = await analyzer.analyze(text, config)
        print_result(text, result, show_details=False)
        
        if result.emotions:
            print("   Top emotions:")
            sorted_emotions = sorted(result.emotions.items(), key=lambda x: x[1], reverse=True)
            for emotion, score in sorted_emotions[:3]:
                if score > 0.1:
                    print(f"      • {emotion.value.title()}: {score:.1%}")
        print()


async def demo_aspect_analysis():
    """Demonstrate aspect-based sentiment analysis"""
    print_header("Aspect-Based Sentiment Analysis")
    
    analyzer = HybridSentimentAnalyzer()
    
    aspect_texts = [
        "The food at this restaurant was amazing but the service was really slow.",
        "Great product quality but the price is too expensive.",
        "The user interface is beautiful but the app crashes frequently.",
        "Fast delivery and excellent customer support, highly recommended!",
        "The location is perfect and the staff is friendly, but the rooms are tiny."
    ]
    
    config = AnalysisConfig(
        method=SentimentMethod.RULE_BASED,
        include_aspects=True
    )
    
    for text in aspect_texts:
        result = await analyzer.analyze(text, config)
        print_result(text, result, show_details=False)
        
        if result.aspects:
            print("   Aspect sentiments:")
            for aspect, score in result.aspects.items():
                sentiment_label = "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral"
                color = "\033[92m" if score > 0.1 else "\033[91m" if score < -0.1 else "\033[93m"
                print(f"      • {aspect}: {color}{sentiment_label}{'\033[0m'} ({score:.3f})")
        print()


async def demo_advanced_analysis():
    """Demonstrate advanced analysis with all features"""
    print_header("Advanced Analysis (All Features)")
    
    analyzer = HybridSentimentAnalyzer()
    
    complex_text = """
    I had mixed feelings about this new smartphone. The design is absolutely gorgeous 
    and the camera quality is outstanding - I love taking photos with it! However, 
    the battery life is disappointing and it gets really hot during heavy use. 
    The customer service was helpful when I called, but the price is definitely 
    too high for what you get. Overall, it's a decent phone but not worth the premium cost.
    """
    
    config = AnalysisConfig(
        method=SentimentMethod.RULE_BASED,
        include_emotions=True,
        include_aspects=True,
        include_explanation=True
    )
    
    print("🔍 Performing comprehensive analysis...")
    start_time = time.time()
    
    result = await analyzer.analyze(complex_text, config)
    
    print_result(complex_text, result, show_details=True)
    
    print(f"\n⏱️ Total analysis time: {time.time() - start_time:.3f}s")


async def demo_batch_analysis():
    """Demonstrate batch processing"""
    print_header("Batch Processing")
    
    analyzer = HybridSentimentAnalyzer()
    
    # Simulate a batch of customer reviews
    reviews = [
        "Excellent product, highly recommend!",
        "Terrible quality, waste of money.",
        "It's okay, nothing special.",
        "Amazing customer service and fast shipping!",
        "Product arrived damaged, very disappointed.",
        "Love it! Works exactly as described.",
        "Poor build quality, broke after one week.",
        "Good value for money, satisfied with purchase.",
        "Outstanding quality and beautiful design!",
        "Worst purchase ever, requesting refund."
    ]
    
    config = AnalysisConfig(method=SentimentMethod.RULE_BASED)
    
    print(f"📊 Processing {len(reviews)} reviews...")
    
    start_time = time.time()
    batch_result = await analyzer.batch_analyze(reviews, config, batch_size=5)
    total_time = time.time() - start_time
    
    print(f"\n✅ Batch analysis complete!")
    print(f"• Total texts: {batch_result.total_texts}")
    print(f"• Successful: {batch_result.successful_analyses}")
    print(f"• Failed: {batch_result.failed_analyses}")
    print(f"• Average time per text: {batch_result.average_processing_time:.3f}s")
    print(f"• Total processing time: {total_time:.3f}s")
    
    # Show sentiment distribution
    polarities = [r.polarity for r in batch_result.results]
    positive_count = sum(1 for p in polarities if p > 0.1)
    negative_count = sum(1 for p in polarities if p < -0.1)
    neutral_count = len(polarities) - positive_count - negative_count
    
    print(f"\n📈 Sentiment Distribution:")
    print(f"   • Positive: {positive_count} ({positive_count/len(polarities):.1%})")
    print(f"   • Negative: {negative_count} ({negative_count/len(polarities):.1%})")
    print(f"   • Neutral: {neutral_count} ({neutral_count/len(polarities):.1%})")
    
    # Show top positive and negative reviews
    sorted_results = sorted(zip(reviews, batch_result.results), key=lambda x: x[1].polarity)
    
    print(f"\n😞 Most Negative: '{sorted_results[0][0]}' (Score: {sorted_results[0][1].polarity:.3f})")
    print(f"😊 Most Positive: '{sorted_results[-1][0]}' (Score: {sorted_results[-1][1].polarity:.3f})")


async def demo_error_handling():
    """Demonstrate error handling and edge cases"""
    print_header("Error Handling & Edge Cases")
    
    analyzer = HybridSentimentAnalyzer()
    config = AnalysisConfig(method=SentimentMethod.RULE_BASED)
    
    edge_cases = [
        "",  # Empty text
        "a",  # Single character
        "😊😊😊",  # Only emojis
        "123 456 789",  # Only numbers
        "!@#$%^&*()",  # Only special characters
        "This is a very long text " * 100,  # Very long text
        "Café naïve résumé",  # Unicode characters
        "I can't believe this isn't working!!!",  # Multiple punctuation
    ]
    
    for text in edge_cases:
        try:
            result = await analyzer.analyze(text, config)
            display_text = text[:30] + "..." if len(text) > 30 else text
            if not text:
                display_text = "(empty string)"
            print(f"✅ '{display_text}' -> {result.polarity:.3f}")
        except Exception as e:
            print(f"❌ Error analyzing '{text[:30]}...': {str(e)}")


async def demo_performance_comparison():
    """Compare performance of different methods"""
    print_header("Performance Comparison")
    
    analyzer = HybridSentimentAnalyzer()
    
    # Test text
    test_text = "This product is absolutely amazing and I love it!"
    iterations = 50
    
    methods = [
        (SentimentMethod.RULE_BASED, "Rule-based"),
        (SentimentMethod.HYBRID, "Hybrid"),
    ]
    
    print(f"🏃 Performance test with {iterations} iterations each:\n")
    
    for method, name in methods:
        config = AnalysisConfig(method=method)
        
        start_time = time.time()
        for _ in range(iterations):
            await analyzer.analyze(test_text, config)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / iterations
        throughput = iterations / total_time
        
        print(f"⚙️ {name}:")
        print(f"   • Total time: {total_time:.3f}s")
        print(f"   • Average per text: {avg_time:.4f}s")
        print(f"   • Throughput: {throughput:.1f} texts/second\n")


def demo_configuration():
    """Show configuration options"""
    print_header("Configuration Options")
    
    print("🔧 Available Analysis Methods:")
    for method in SentimentMethod:
        print(f"   • {method.value}")
    
    print("\n😊 Available Emotions:")
    for emotion in EmotionType:
        print(f"   • {emotion.value}")
    
    print("\n⚙️ Configuration Example:")
    config = AnalysisConfig(
        method=SentimentMethod.HYBRID,
        include_emotions=True,
        include_aspects=True,
        include_explanation=True,
        gemini_temperature=0.1,
        ensemble_weights={"rule_based": 0.4, "gemini": 0.6}
    )
    
    print(json.dumps(config.to_dict(), indent=2))


async def main():
    """Run all demos"""
    print("🎭 Enhanced SentimentR v2.0 - Comprehensive Demo")
    print("=" * 60)
    print("This demo showcases all the major features of Enhanced SentimentR")
    print("including sentiment analysis, emotion detection, aspect analysis, and more!")
    
    try:
        # Basic functionality
        await demo_basic_analysis()
        
        # Advanced features
        await demo_emotion_analysis()
        await demo_aspect_analysis()
        await demo_advanced_analysis()
        
        # Batch processing
        await demo_batch_analysis()
        
        # Edge cases
        await demo_error_handling()
        
        # Performance
        await demo_performance_comparison()
        
        # Configuration
        demo_configuration()
        
        print_header("Demo Complete!")
        print("✅ All demos completed successfully!")
        print("\n🚀 Next steps:")
        print("   • Try the CLI: python -m enhanced_sentimentr.cli analyze 'Your text here'")
        print("   • Start the API: python -m enhanced_sentimentr.api.main")
        print("   • Launch web interface: streamlit run enhanced_sentimentr/web/streamlit_app.py")
        print("   • Read the docs: README_enhanced.md")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
