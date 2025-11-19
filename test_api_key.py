#!/usr/bin/env python3
"""
Test script to verify Gemini API key is working
"""
import asyncio
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_api_key():
    """Test if Gemini API key is working"""
    print("🧪 Testing Gemini API Key...")
    print("-" * 40)
    
    # Check if API key is set
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("❌ No GEMINI_API_KEY environment variable found!")
        print("\n💡 To set it:")
        print("   export GEMINI_API_KEY='your-api-key-here'")
        print("   OR run: python setup_api_key.py")
        return False
    
    print(f"✅ API key found: {api_key[:8]}...{api_key[-4:]}")
    
    try:
        from enhanced_sentimentr.core.analyzer import HybridSentimentAnalyzer
        from enhanced_sentimentr.core.models import AnalysisConfig, SentimentMethod
        
        print("✅ Enhanced SentimentR imported successfully")
        
        # Test analyzer initialization
        analyzer = HybridSentimentAnalyzer()
        
        if not analyzer.gemini_client:
            print("❌ Gemini client not initialized!")
            print("   Check if your API key is valid")
            return False
        
        print("✅ Gemini client initialized")
        
        # Test actual analysis
        print("\n🔍 Testing sentiment analysis...")
        
        test_cases = [
            ("I love this product!", "positive"),
            ("This is terrible", "negative"),
            ("The weather is cloudy", "neutral")
        ]
        
        for text, expected in test_cases:
            try:
                config = AnalysisConfig(method=SentimentMethod.GEMINI)
                result = await analyzer.analyze(text, config)
                
                if result.method == SentimentMethod.GEMINI:
                    print(f"✅ '{text}' → {result.polarity:.3f} ({expected})")
                else:
                    print(f"⚠️  '{text}' → fell back to rule-based")
                    
            except Exception as e:
                print(f"❌ Error analyzing '{text}': {e}")
                return False
        
        print("\n🎉 API key is working perfectly!")
        print("\n🚀 You can now use:")
        print("   • python -m enhanced_sentimentr.cli analyze 'text' --method hybrid")
        print("   • streamlit run enhanced_sentimentr/web/streamlit_app.py")
        print("   • python demo.py")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Run: pip install google-generativeai")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

async def main():
    """Main test function"""
    print("🔑 Enhanced SentimentR - API Key Test")
    print("=" * 45)
    
    success = await test_api_key()
    
    if not success:
        print("\n💡 Need help setting up your API key?")
        print("   Run: python setup_api_key.py")
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n❌ Test cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
