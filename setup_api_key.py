#!/usr/bin/env python3
"""
Interactive API key setup for Enhanced SentimentR
"""
import os
import sys
import asyncio
from pathlib import Path

def print_header():
    """Print welcome header"""
    print("🔑 Enhanced SentimentR - API Key Setup")
    print("=" * 50)
    print()

def get_api_key_input():
    """Get API key from user input"""
    print("📝 Please enter your Google Gemini API key:")
    print("   (You can get one from: https://makersuite.google.com/app/apikey)")
    print()
    
    api_key = input("🔑 Gemini API Key: ").strip()
    
    if not api_key:
        print("❌ No API key provided!")
        return None
    
    if len(api_key) < 20:
        print("⚠️  Warning: This seems like a short API key. Are you sure it's correct?")
        confirm = input("Continue anyway? (y/N): ").strip().lower()
        if confirm != 'y':
            return None
    
    return api_key

def save_to_env_file(api_key):
    """Save API key to .env file"""
    try:
        env_file = Path(".env")
        
        # Read existing content if file exists
        existing_content = ""
        if env_file.exists():
            with open(env_file, 'r') as f:
                lines = f.readlines()
            
            # Filter out existing GEMINI_API_KEY lines
            filtered_lines = [line for line in lines if not line.startswith('GEMINI_API_KEY=')]
            existing_content = ''.join(filtered_lines)
        
        # Write new content
        with open(env_file, 'w') as f:
            f.write(existing_content)
            if existing_content and not existing_content.endswith('\n'):
                f.write('\n')
            f.write(f'GEMINI_API_KEY={api_key}\n')
        
        print(f"✅ API key saved to {env_file.absolute()}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving to .env file: {e}")
        return False

def set_environment_variable(api_key):
    """Set environment variable for current session"""
    os.environ['GEMINI_API_KEY'] = api_key
    print("✅ API key set for current session")

def show_usage_examples():
    """Show how to use the API key"""
    print("\n🎯 Usage Examples:")
    print()
    print("1. Command Line:")
    print("   python -m enhanced_sentimentr.cli analyze 'I love this!' --method hybrid")
    print()
    print("2. Python Code:")
    print("   from enhanced_sentimentr import HybridSentimentAnalyzer")
    print("   analyzer = HybridSentimentAnalyzer()  # Will use env variable")
    print("   result = await analyzer.analyze('Great product!')")
    print()
    print("3. Test your setup:")
    print("   python test_api_key.py")

async def test_api_key():
    """Test if the API key works"""
    print("\n🧪 Testing API key...")
    
    try:
        # Import here to avoid circular imports
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from enhanced_sentimentr.core.analyzer import HybridSentimentAnalyzer
        from enhanced_sentimentr.core.models import AnalysisConfig, SentimentMethod
        
        analyzer = HybridSentimentAnalyzer()
        
        if not analyzer.gemini_client:
            print("❌ Gemini client not initialized. API key may be missing or invalid.")
            return False
        
        # Test with a simple analysis
        config = AnalysisConfig(method=SentimentMethod.GEMINI)
        result = await analyzer.analyze("This is a test message", config)
        
        if result.method == SentimentMethod.GEMINI:
            print("✅ API key is working! Gemini analysis successful.")
            print(f"   Test result: {result.polarity:.3f} sentiment")
            return True
        else:
            print("⚠️  Gemini analysis failed, but system is working with rule-based fallback")
            return False
            
    except Exception as e:
        print(f"❌ Error testing API key: {e}")
        return False

def show_shell_commands():
    """Show shell commands to set environment variable"""
    print("\n🐚 To set permanently in your shell:")
    print()
    print("For Bash/Zsh (Linux/Mac):")
    print("   echo 'export GEMINI_API_KEY=\"your-key-here\"' >> ~/.bashrc")
    print("   source ~/.bashrc")
    print()
    print("For Fish shell:")
    print("   set -Ux GEMINI_API_KEY your-key-here")
    print()
    print("For Windows Command Prompt:")
    print("   setx GEMINI_API_KEY \"your-key-here\"")

async def main():
    """Main setup flow"""
    print_header()
    
    # Check if API key is already set
    existing_key = os.environ.get('GEMINI_API_KEY')
    if existing_key:
        print(f"✅ API key is already set in environment: {existing_key[:8]}...")
        test_anyway = input("\n🧪 Test the existing API key? (Y/n): ").strip().lower()
        if test_anyway != 'n':
            success = await test_api_key()
            if success:
                print("\n🎉 Your API key is working perfectly!")
                show_usage_examples()
                return
    
    # Get new API key
    api_key = get_api_key_input()
    if not api_key:
        print("❌ Setup cancelled.")
        return
    
    print("\n📁 How would you like to save the API key?")
    print("1. Environment variable for current session only")
    print("2. Save to .env file (recommended)")
    print("3. Both")
    print("4. Show me shell commands to set it permanently")
    
    choice = input("\nChoose option (1-4): ").strip()
    
    if choice == "1":
        set_environment_variable(api_key)
    elif choice == "2":
        if save_to_env_file(api_key):
            set_environment_variable(api_key)  # Also set for current session
    elif choice == "3":
        set_environment_variable(api_key)
        save_to_env_file(api_key)
    elif choice == "4":
        show_shell_commands()
        print(f"\nReplace 'your-key-here' with: {api_key}")
        return
    else:
        print("❌ Invalid choice. Setting for current session only.")
        set_environment_variable(api_key)
    
    # Test the API key
    test_key = input("\n🧪 Test the API key now? (Y/n): ").strip().lower()
    if test_key != 'n':
        success = await test_api_key()
        if success:
            print("\n🎉 Setup complete! Your API key is working.")
        else:
            print("\n⚠️  Setup complete, but API key test failed.")
            print("   You can still use rule-based analysis without Gemini.")
    
    show_usage_examples()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user.")
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)
