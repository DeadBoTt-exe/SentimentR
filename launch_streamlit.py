#!/usr/bin/env python3
"""
Streamlit launcher for Enhanced SentimentR
This script ensures proper path setup before launching the Streamlit app
"""
import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch Streamlit with proper path setup"""
    # Get the project root directory
    current_dir = Path(__file__).parent.absolute()
    project_root = current_dir
    
    # Add project root to Python path
    env = os.environ.copy()
    pythonpath = env.get('PYTHONPATH', '')
    if pythonpath:
        env['PYTHONPATH'] = f"{project_root}:{pythonpath}"
    else:
        env['PYTHONPATH'] = str(project_root)
    
    # Set API key if available from .env file
    env_file = project_root / '.env'
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    env[key.strip()] = value.strip()
    
    # Launch streamlit
    streamlit_app = project_root / 'enhanced_sentimentr' / 'web' / 'streamlit_app.py'
    
    if not streamlit_app.exists():
        print(f"❌ Streamlit app not found at: {streamlit_app}")
        return 1
    
    print("🚀 Launching Enhanced SentimentR Web Interface...")
    print(f"📁 Project root: {project_root}")
    print(f"🎯 App path: {streamlit_app}")
    
    # Check if API key is available
    if env.get('GEMINI_API_KEY'):
        print(f"✅ Gemini API key found: {env['GEMINI_API_KEY'][:8]}...")
    else:
        print("⚠️  No Gemini API key found - will use rule-based analysis only")
        print("   To enable AI features, set: export GEMINI_API_KEY='your-key'")
    
    print("\n" + "="*60)
    
    try:
        # Run streamlit with proper environment
        result = subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', str(streamlit_app)
        ], env=env, cwd=project_root)
        
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
