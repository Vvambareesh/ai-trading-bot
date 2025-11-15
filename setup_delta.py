#!/usr/bin/env python3
"""
Delta Exchange API Setup Helper
Guide to get your Delta API credentials and configure them
"""

import os
import json

def show_delta_setup_guide():
    """Show step-by-step guide for Delta Exchange API setup"""
    print("🔗 Delta Exchange API Setup Guide")
    print("=" * 50)

    print("\n📋 Step 1: Create Delta Exchange Account")
    print("   1. Go to https://delta.exchange")
    print("   2. Sign up or log in to your account")
    print("   3. Complete 2FA setup (highly recommended)")
    print("   4. Verify your email address")

    print("\n🔑 Step 2: Get API Credentials")
    print("   1. Go to Account Settings")
    print("   2. Navigate to 'API Keys' section")
    print("   3. Click 'Generate New API Key'")
    print("   4. Give your API key a name (e.g., 'AI Trading Bot')")
    print("   5. Select permissions:")
    print("      ✅ Read Account Balance")
    print("      ✅ Place Orders")
    print("      ✅ Cancel Orders")
    print("      ✅ View Order History")
    print("   6. Copy the API Key and Secret")

    print("\n🪝 Step 3: Configure TradingView Webhook")
    print("   1. Go to Algo dropdown → TradingBot page")
    print("   2. Click 'Create New Webhook'")
    print("   3. Set webhook name (e.g., 'TradingView Alerts')")
    print("   4. Copy the Webhook URL")

    print("\n📝 Step 4: Update Your .env File")
    print("   Your API credentials need to be added to your .env file:")
    print("")
    print("   DELTA_API_KEY=your_delta_api_key_here")
    print("   DELTA_API_SECRET=your_delta_api_secret_here")
    print("   DELTA_WEBHOOK_URL=https://delta.exchange/webhook/your_webhook_id")
    print("")

def create_delta_config_template():
    """Create a Delta configuration template"""
    config = {
        "api_credentials": {
            "api_key": "your_delta_api_key_here",
            "api_secret": "your_delta_api_secret_here",
            "environment": "production"
        },
        "webhook_config": {
            "url": "https://delta.exchange/webhook/your_webhook_id",
            "name": "AI Trading Bot Webhook"
        },
        "trading_view_message": {
            "symbol": "{{ticker}}",
            "side": "{{strategy.order.action}}",
            "qty": "{{strategy.order.contracts}}",
            "type": "market",
            "trigger_time": "{{timenow}}"
        },
        "tradingview_alert_setup": {
            "webhook_url": "https://delta.exchange/webhook/your_webhook_id",
            "message": {
                "symbol": "{{ticker}}",
                "side": "{{strategy.order.action}}",
                "qty": "{{strategy.order.contracts}}",
                "type": "market",
                "trigger_time": "{{timenow}}"
            },
            "conditions": [
                "Your strategy condition here"
            ]
        }
    }

    # Save configuration template
    with open('delta_config_template.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("💾 Delta configuration template saved to: delta_config_template.json")

def test_delta_webhook_format():
    """Generate TradingView webhook message format"""
    webhook_format = {
        "description": "TradingView webhook format for Delta Exchange",
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

    with open('delta_webhook_format.json', 'w') as f:
        json.dump(webhook_format, f, indent=2)

    print("💾 Webhook format saved to: delta_webhook_format.json")

def update_env_with_delta(delta_api_key=None, delta_api_secret=None, webhook_url=None):
    """Update .env file with Delta credentials"""
    env_file = '.env'

    if not os.path.exists(env_file):
        print("❌ .env file not found. Creating new one...")
        with open(env_file, 'w') as f:
            f.write("# AI Trading Bot Configuration\n")
            f.write("# Delta Exchange API Configuration\n")

    # Read existing .env content
    with open(env_file, 'r') as f:
        lines = f.readlines()

    # Update or add Delta configurations
    updated_lines = []
    delta_configs = {
        'DELTA_API_KEY': delta_api_key,
        'DELTA_API_SECRET': delta_api_secret,
        'DELTA_WEBHOOK_URL': webhook_url
    }

    delta_keys_found = set()

    for line in lines:
        # Check if line contains Delta config
        line_stripped = line.strip()
        if line_stripped.startswith('#') or '=' not in line_stripped:
            updated_lines.append(line)
            continue

        # Extract key
        key = line_stripped.split('=')[0].strip()

        if key in delta_configs and delta_configs[key]:
            # Update existing Delta config
            updated_lines.append(f"{key}={delta_configs[key]}\n")
            delta_keys_found.add(key)
        elif key not in delta_configs:
            # Keep other existing configs
            updated_lines.append(line)

    # Add any missing Delta configs
    for key, value in delta_configs.items():
        if value and key not in delta_keys_found:
            updated_lines.append(f"{key}={value}\n")

    # Write back to .env
    with open(env_file, 'w') as f:
        f.writelines(updated_lines)

    print("✅ .env file updated with Delta API configuration")

def main():
    """Main setup function"""
    print("🔗 Delta Exchange Setup Assistant")
    print("=" * 50)

    choice = input("""
Choose an option:
1. Show complete setup guide
2. Generate configuration templates
3. Update .env with credentials
4. Exit

Enter your choice (1-4): """).strip()

    if choice == '1':
        show_delta_setup_guide()
    elif choice == '2':
        print("📋 Generating configuration templates...")
        create_delta_config_template()
        test_delta_webhook_format()
        print("\n✅ Templates generated. Files created:")
        print("   - delta_config_template.json")
        print("   - delta_webhook_format.json")
    elif choice == '3':
        print("📝 Updating .env with Delta credentials...")

        delta_api_key = input("Enter your Delta API Key: ").strip()
        delta_api_secret = input("Enter your Delta API Secret: ").strip()
        webhook_url = input("Enter your Delta Webhook URL: ").strip()

        if delta_api_key and delta_api_secret and webhook_url:
            update_env_with_delta(delta_api_key, delta_api_secret, webhook_url)
        else:
            print("❌ All credentials are required. Please provide:")
            print("   - Delta API Key")
            print("   - Delta API Secret")
            print("   - Delta Webhook URL")
    elif choice == '4':
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    main()