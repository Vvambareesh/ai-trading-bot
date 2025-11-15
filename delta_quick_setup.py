#!/usr/bin/env python3
"""
Quick Delta Exchange setup - generates templates and shows current configuration
"""

import os
from dotenv import load_dotenv
import json

def show_current_config():
    """Show current Delta configuration in .env"""
    print("🔗 Current Delta Exchange Configuration:")
    print("=" * 50)

    load_dotenv()

    delta_key = os.getenv('DELTA_API_KEY')
    delta_secret = os.getenv('DELTA_API_SECRET')
    webhook_url = os.getenv('DELTA_WEBHOOK_URL')

    print(f"DELTA_API_KEY: {'✅ Configured' if delta_key else '❌ Not configured'}")
    print(f"DELTA_API_SECRET: {'✅ Configured' if delta_secret else '❌ Not configured'}")
    print(f"DELTA_WEBHOOK_URL: {'✅ Configured' if webhook_url else '❌ Not configured'}")

    if delta_key and delta_secret and webhook_url:
        print(f"\n✅ Delta API is fully configured!")
        print(f"   Webhook: {webhook_url}")
        return True
    else:
        print(f"\n⚠️  Delta API needs configuration")
        return False

def generate_delta_config():
    """Generate Delta configuration templates"""
    print("\n📋 Generating Delta Configuration Templates...")

    # API configuration example
    api_config = {
        "description": "Delta Exchange API Configuration",
        "steps": [
            "1. Login to delta.exchange",
            "2. Go to Account Settings → API Keys",
            "3. Click 'Generate New API Key'",
            "4. Set permissions: Read Balance, Place Orders, Cancel Orders",
            "5. Copy API Key and Secret",
            "6. Add to .env file"
        ],
        "env_template": {
            "DELTA_API_KEY": "your_delta_api_key_here",
            "DELTA_API_SECRET": "your_delta_api_secret_here",
            "DELTA_WEBHOOK_URL": "https://delta.exchange/webhook/your_webhook_id"
        }
    }

    # TradingView webhook format
    webhook_format = {
        "description": "TradingView Webhook Format for Delta Exchange",
        "url": "https://delta.exchange/webhook/YOUR_WEBHOOK_ID",
        "message_template": {
            "symbol": "{{ticker}}",
            "side": "{{strategy.order.action}}",
            "qty": "{{strategy.order.contracts}}",
            "type": "market",
            "trigger_time": "{{timenow}}"
        },
        "examples": {
            "buy_signal": {
                "symbol": "BTCUSD",
                "side": "buy",
                "qty": 0.01,
                "type": "market",
                "trigger_time": "2024-01-15T15:30:00Z"
            },
            "sell_signal": {
                "symbol": "ETHUSD",
                "side": "sell",
                "qty": 0.3,
                "type": "market",
                "trigger_time": "2024-01-15T15:30:00Z"
            }
        }
    }

    # Symbol mapping
    symbol_mapping = {
        "tradingview_to_delta": {
            "BINANCE:BTCUSDT": "BTCUSD",
            "BINANCE:ETHUSDT": "ETHUSD",
            "BINANCE:ADAUSDT": "ADAUSD",
            "BINANCE:DOGEUSDT": "DOGEUSD",
            "BINANCE:SHIBUSDT": "SHIBUSD"
        },
        "delta_symbol_format": "CRYPTOUSD"
    }

    # Save configurations
    with open('delta_api_config.json', 'w') as f:
        json.dump(api_config, f, indent=2)

    with open('delta_webhook_format.json', 'w') as f:
        json.dump(webhook_format, f, indent=2)

    with open('delta_symbol_mapping.json', 'w') as f:
        json.dump(symbol_mapping, f, indent=2)

    print("✅ Configuration templates created:")
    print("   - delta_api_config.json")
    print("   - delta_webhook_format.json")
    print("   - delta_symbol_mapping.json")

def create_tradingview_setup():
    """Create TradingView setup instructions"""
    print("\n📱 TradingView Setup Instructions:")
    print("=" * 50)

    setup_instructions = """
# TradingView Setup for Delta Exchange

## Step 1: Create Strategy
1. Open TradingView Chart
2. Go to Pine Editor
3. Create or edit your strategy
4. Add alert conditions for buy/sell signals

## Step 2: Set Up Alerts
1. Click "Alert" icon on your chart
2. Configure alert settings:
   - Condition: Your strategy alert condition
   - Webhook URL: https://delta.exchange/webhook/YOUR_WEBHOOK_ID
   - Message: Use the template below
   - Expiry: Good until cancelled

## Step 3: Message Template
Copy this exact format for your TradingView alert message:

{
    "symbol": "{{ticker}}",
    "side": "{{strategy.order.action}}",
    "qty": "{{strategy.order.contracts}}",
    "type": "market",
    "trigger_time": "{{timenow}}"
}

## Step 4: Test Alert
1. Trigger your strategy condition manually
2. Check Delta Exchange for order execution
3. Verify webhook is working properly

## Step 5: Configure for AI Bot
1. Ensure webhook matches your .env DELTA_WEBHOOK_URL
2. Start AI bot in live mode: python main.py --mode live
3. Monitor dashboard: python dashboard.py
"""

    with open('TRADINGVIEW_SETUP.txt', 'w') as f:
        f.write(setup_instructions)

    print("✅ TradingView setup saved to: TRADINGVIEW_SETUP.txt")

def main():
    """Main setup function"""
    print("🔗 Delta Exchange Quick Setup")
    print("=" * 50)

    # Check current configuration
    is_configured = show_current_config()

    if not is_configured:
        print(f"\n📋 Generating Setup Resources...")
        generate_delta_config()
        create_tradingview_setup()

    print(f"\n🎯 Quick Actions:")
    print(f"1. Get Delta API credentials from delta.exchange")
    print(f"2. Add to .env file:")
    print(f"   DELTA_API_KEY=your_key_here")
    print(f"   DELTA_API_SECRET=your_secret_here")
    print(f"   DELTA_WEBHOOK_URL=https://delta.exchange/webhook/your_id")
    print(f"")
    print(f"3. Start paper trading first:")
    print(f"   python main.py --mode paper --strategy hybrid")
    print(f"")
    print(f"4. Review generated configuration files:")
    print(f"   - delta_api_config.json")
    print(f"   - delta_webhook_format.json")
    print(f"   - TRADINGVIEW_SETUP.txt")

    print(f"\n📊 Configuration Files Created:")
    print(f"✅ API Configuration Template")
    print(f"✅ Webhook Format Guide")
    print(f"✅ TradingView Setup Instructions")
    print(f"✅ Symbol Mapping Reference")

if __name__ == "__main__":
    main()