#!/usr/bin/env python3
"""
Direct Gemini API test
"""

import os
import subprocess

def test_gemini_api():
    """Test Gemini API directly"""
    print("🧪 Testing Gemini API Connection...")
    print("=" * 50)

    try:
        # Test with subprocess to bypass import issues
        test_code = '''
import os
import sys
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

if api_key:
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content("Hello! Please respond with 'API connection successful'")
        print(f"✅ Gemini API Response: {response.text}")
    except Exception as e:
        print(f"❌ Gemini API Error: {e}")
        sys.exit(1)
else:
    print("❌ GEMINI_API_KEY not found")
    sys.exit(1)
'''

        result = subprocess.run(['python3', '-c', test_code],
                          capture_output=True, text=True, cwd='.')

        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print(result.stderr)
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def show_next_steps():
    """Show next steps for user"""
    print("\n🚀 Next Steps:")
    print("1. ✅ Gemini API is configured and working")
    print("2. ✅ Your AI trading bot is ready to use")
    print("3. 📝 Start paper trading:")
    print("   python main.py --mode paper --strategy hybrid")
    print("4. 📊 Launch dashboard:")
    print("   python dashboard.py")
    print("5. 🧪 Run backtesting:")
    print("   python backtester.py")

if __name__ == "__main__":
    success = test_gemini_api()

    if success:
        show_next_steps()
        print(f"\n🎉 SUCCESS! Your AI trading bot is ready!")
    else:
        print(f"\n⚠️  Setup failed. Please check your configuration.")