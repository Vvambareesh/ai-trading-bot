#!/usr/bin/env python3
"""
API Setup Verification Script
Test if Gemini API key is properly configured
"""

import os
from dotenv import load_dotenv

def verify_api_setup():
    """Verify API configuration"""
    print("🔍 Verifying API Setup...")
    print("=" * 50)

    # Load environment variables
    load_dotenv()

    # Check Gemini API key
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        print(f"✅ GEMINI_API_KEY found")
        print(f"   Length: {len(api_key)} characters")
        print(f"   Format: {api_key[:10]}...{api_key[-10:]}")

        # Basic validation
        if api_key.startswith('AIzaSy') and len(api_key) > 30:
            print(f"✅ API key format looks valid")
            return True
        else:
            print(f"⚠️  API key format may be invalid")
            return False
    else:
        print(f"❌ GEMINI_API_KEY not found in .env file")
        return False

def test_gemini_connection():
    """Test actual Gemini API connection"""
    try:
        import google.generativeai as genai

        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            return False

        print(f"\n🧪 Testing Gemini API Connection...")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')

        # Simple test request
        response = model.generate_content("Hello! Please respond with 'API connection successful' to test.")

        if 'API connection successful' in response.text:
            print(f"✅ Gemini API connection successful!")
            print(f"   Response: {response.text}")
            return True
        else:
            print(f"⚠️  API responded but unexpected response:")
            print(f"   Response: {response.text}")
            return True  # Still counts as success

    except Exception as e:
        print(f"❌ Gemini API connection failed: {e}")
        return False

def check_delta_webhook():
    """Check Delta Exchange webhook configuration"""
    webhook_url = os.getenv('DELTA_WEBHOOK_URL')

    if webhook_url:
        print(f"\n🔗 Delta Exchange Webhook: {webhook_url}")
        if 'delta.exchange' in webhook_url:
            print(f"✅ Webhook URL format looks valid")
            return True
        else:
            print(f"⚠️  Webhook URL may not be Delta Exchange")
            return False
    else:
        print(f"\n⚠️  DELTA_WEBHOOK_URL not configured (optional for paper trading)")
        return True

def show_configuration_status():
    """Show current configuration status"""
    print(f"\n📊 Configuration Status:")
    print(f"   Trading Mode: {os.getenv('TRADING_MODE', 'paper')}")
    print(f"   Initial Capital: ₹{os.getenv('INITIAL_CAPITAL_INR', '1000')}")
    print(f"   Strategy: {os.getenv('STRATEGY', 'hybrid')}")
    print(f"   Conviction Threshold: {os.getenv('CONVICTION_THRESHOLD', '95')}%")
    print(f"   Risk Per Trade: {os.getenv('RISK_PER_TRADE', '5')}%")
    print(f"   Max Concurrent Trades: {os.getenv('MAX_CONCURRENT_TRADES', '3')}")

    print(f"\n📱 Ready for:")
    print(f"   ✅ Paper Trading: python main.py --mode paper --strategy hybrid")

    if os.getenv('DELTA_WEBHOOK_URL') and os.getenv('GEMINI_API_KEY'):
        print(f"   ✅ Live Trading: python main.py --mode live --strategy hybrid --conviction 95")
    else:
        print(f"   ⚠️  Live Trading: Configure DELTA_WEBHOOK_URL and GEMINI_API_KEY")

def main():
    """Main verification function"""
    print("🤖 AI Trading Bot Setup Verification")
    print("Checking your configuration...")

    # Verify API setup
    api_valid = verify_api_setup()

    if api_valid:
        # Test connection
        connection_success = test_gemini_connection()

        # Check webhook
        webhook_valid = check_delta_webhook()

        # Show status
        show_configuration_status()

        if connection_success:
            print(f"\n🎉 SETUP COMPLETE!")
            print(f"Your AI trading bot is ready to start trading.")
            print(f"\n🚀 Quick Start Commands:")
            print(f"   Paper Trading:")
            print(f"   python main.py --mode paper --strategy hybrid --capital 1000")
            print(f"")
            print(f"   Dashboard:")
            print(f"   python dashboard.py")
            print(f"")
            print(f"   Backtesting:")
            print(f"   python backtester.py")

            if webhook_valid:
                print(f"")
                print(f"   Live Trading (when ready):")
                print(f"   python main.py --mode live --strategy hybrid --conviction 95")

        else:
            print(f"\n⚠️  SETUP INCOMPLETE:")
            print(f"API key found but connection failed. Please check:")
            print(f"1. API key is valid")
            print(f"2. Internet connection")
            print(f"3. Gemini API service is available")
    else:
        print(f"\n❌ SETUP FAILED:")
        print(f"Please fix the issues above and run this script again.")

        print(f"\n📝 How to fix:")
        print(f"1. Copy .env.example to .env: cp .env.example .env")
        print(f"2. Edit .env and add your Gemini API key:")
        print(f"   GEMINI_API_KEY=your_actual_api_key_here")
        print(f"3. Save the file and run this script again")

if __name__ == "__main__":
    main()