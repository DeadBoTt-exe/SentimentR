#!/usr/bin/env python3
"""
Quick test script to verify Enhanced SentimentR installation and basic functionality
"""
import sys
import os
import asyncio

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from enhanced_sentimentr.core.analyzer import HybridSentimentAnalyzer
        print("   ✅ HybridSentimentAnalyzer")
        
        from enhanced_sentimentr.core.models import SentimentResult, AnalysisConfig
        print("   ✅ Models")
        
        from enhanced_sentimentr.core.emotions import EmotionAnalyzer
        print("   ✅ EmotionAnalyzer")
        
        from enhanced_sentimentr.core.aspects import AspectAnalyzer
        print("   ✅ AspectAnalyzer")
        
        from enhanced_sentimentr.core.rule_based import EnhancedRuleBasedAnalyzer
        print("   ✅ EnhancedRuleBasedAnalyzer")
        
        print("   ✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False


async def test_basic_analysis():
    """Test basic sentiment analysis"""
    print("\n🔍 Testing basic analysis...")
    
    try:
        from enhanced_sentimentr.core.analyzer import HybridSentimentAnalyzer
        from enhanced_sentimentr.core.models import AnalysisConfig, SentimentMethod
        
        analyzer = HybridSentimentAnalyzer()
        config = AnalysisConfig(method=SentimentMethod.RULE_BASED)
        
        # Test positive sentiment
        result = await analyzer.analyze("I love this product!", config)
        print(f"   ✅ Positive text: {result.polarity:.3f} (expected: > 0)")
        
        # Test negative sentiment
        result = await analyzer.analyze("This is terrible!", config)
        print(f"   ✅ Negative text: {result.polarity:.3f} (expected: < 0)")
        
        # Test neutral sentiment
        result = await analyzer.analyze("The weather is cloudy.", config)
        print(f"   ✅ Neutral text: {result.polarity:.3f} (expected: ~0)")
        
        print("   ✅ Basic analysis working!")
        return True
        
    except Exception as e:
        print(f"   ❌ Analysis error: {e}")
        return False


async def test_emotion_analysis():
    """Test emotion analysis"""
    print("\n🔍 Testing emotion analysis...")
    
    try:
        from enhanced_sentimentr.core.analyzer import HybridSentimentAnalyzer
        from enhanced_sentimentr.core.models import AnalysisConfig, SentimentMethod
        
        analyzer = HybridSentimentAnalyzer()
        config = AnalysisConfig(
            method=SentimentMethod.RULE_BASED,
            include_emotions=True
        )
        
        result = await analyzer.analyze("I'm so happy and excited!", config)
        
        if result.emotions:
            print(f"   ✅ Emotions detected: {len(result.emotions)} emotions")
            for emotion, score in result.emotions.items():
                if score > 0.1:
                    print(f"      • {emotion.value}: {score:.2f}")
        else:
            print("   ⚠️ No emotions detected")
        
        print("   ✅ Emotion analysis working!")
        return True
        
    except Exception as e:
        print(f"   ❌ Emotion analysis error: {e}")
        return False


async def test_aspect_analysis():
    """Test aspect analysis"""
    print("\n🔍 Testing aspect analysis...")
    
    try:
        from enhanced_sentimentr.core.analyzer import HybridSentimentAnalyzer
        from enhanced_sentimentr.core.models import AnalysisConfig, SentimentMethod
        
        analyzer = HybridSentimentAnalyzer()
        config = AnalysisConfig(
            method=SentimentMethod.RULE_BASED,
            include_aspects=True
        )
        
        result = await analyzer.analyze("Great product but terrible service!", config)
        
        if result.aspects:
            print(f"   ✅ Aspects detected: {len(result.aspects)} aspects")
            for aspect, score in result.aspects.items():
                print(f"      • {aspect}: {score:.2f}")
        else:
            print("   ⚠️ No aspects detected")
        
        print("   ✅ Aspect analysis working!")
        return True
        
    except Exception as e:
        print(f"   ❌ Aspect analysis error: {e}")
        return False


def test_lexicon_files():
    """Test if lexicon files are accessible"""
    print("\n🔍 Testing lexicon files...")
    
    try:
        from enhanced_sentimentr.utils.data_loader import DataLoader
        
        loader = DataLoader()
        
        # Test loading different lexicons
        pos_words = loader.load_word_list("pos_lexicon.txt")
        print(f"   ✅ Positive words: {len(pos_words)} loaded")
        
        neg_words = loader.load_word_list("neg_lexicon.txt")
        print(f"   ✅ Negative words: {len(neg_words)} loaded")
        
        intensifiers = loader.load_word_list("intensifiers.txt")
        print(f"   ✅ Intensifiers: {len(intensifiers)} loaded")
        
        print("   ✅ All lexicon files accessible!")
        return True
        
    except Exception as e:
        print(f"   ❌ Lexicon file error: {e}")
        return False


async def main():
    """Run all tests"""
    print("🧪 Enhanced SentimentR - Quick Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Basic Analysis", test_basic_analysis),
        ("Emotion Analysis", test_emotion_analysis),
        ("Aspect Analysis", test_aspect_analysis),
        ("Lexicon Files", test_lexicon_files),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔬 Running {test_name}...")
        try:
            if asyncio.iscoroutinefunction(test_func):
                success = await test_func()
            else:
                success = test_func()
            
            if success:
                passed += 1
                print(f"   ✅ {test_name} PASSED")
            else:
                print(f"   ❌ {test_name} FAILED")
                
        except Exception as e:
            print(f"   ❌ {test_name} CRASHED: {e}")
    
    print("\n" + "=" * 50)
    print(f"🧪 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced SentimentR is ready to use!")
        print("\n🚀 Try running the demo:")
        print("   python demo.py")
    else:
        print("⚠️ Some tests failed. Please check the installation.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
