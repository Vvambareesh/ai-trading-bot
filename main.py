import os
import time
import requests
import google.generativeai as genai
from datetime import datetime

# Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
DELTA_WEBHOOK_URL = os.getenv('DELTA_WEBHOOK_URL')
SYMBOL = 'SHIBUSD'
CHECK_INTERVAL = 300  # Check every 5 minutes

# Initialize Gemini AI
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

print(f"🤖 AI Trading Bot Started for {SYMBOL}")
print(f"⏰ Check interval: {CHECK_INTERVAL} seconds")
print("="*50)

def get_market_data():
    """Fetch current SHIBUSD price from CoinGecko"""
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=shiba-inu&vs_currencies=usd&include_24hr_change=true"
        response = requests.get(url)
        data = response.json()
        price = data['shiba-inu']['usd']
        change_24h = data['shiba-inu']['usd_24h_change']
        return price, change_24h
    except Exception as e:
        print(f"❌ Error fetching market data: {e}")
        return None, None

def analyze_with_ai(price, change_24h):
    """Use Gemini AI to analyze market conditions"""
    try:
        prompt = f"""
        You are a cryptocurrency trading advisor. Analyze SHIBA INU (SHIBUSD) current market conditions:
        
        Current Price: ${price}
        24h Change: {change_24h:.2f}%
        
        Based on technical analysis principles:
        - If price is rising with strong momentum (>5% gain): Recommend BUY
        - If price is falling with strong momentum (<-5% loss): Recommend SELL
        - Otherwise: Recommend HOLD
        
        Respond with ONLY ONE WORD: BUY, SELL, or HOLD
        """
        
        response = model.generate_content(prompt)
        signal = response.text.strip().upper()
        
        if 'BUY' in signal:
            return 'BUY'
        elif 'SELL' in signal:
            return 'SELL'
        else:
            return 'HOLD'
    except Exception as e:
        print(f"❌ AI Analysis Error: {e}")
        return 'HOLD'

def send_signal_to_delta(signal):
    """Send trading signal to Delta Exchange via webhook"""
    try:
        if signal == 'HOLD':
            return False
            
        payload = {
            "symbol": SYMBOL,
            "action": signal,
            "timestamp": datetime.now().isoformat()
        }
        
        response = requests.post(DELTA_WEBHOOK_URL, json=payload)
        
        if response.status_code == 200:
            print(f"✅ {signal} signal sent to Delta Exchange")
            return True
        else:
            print(f"⚠️ Failed to send signal. Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error sending signal to Delta: {e}")
        return False

def main_loop():
    """Main trading bot loop"""
    while True:
        try:
            print(f"\n⏰ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
            
            # Get market data
            price, change_24h = get_market_data()
            
            if price is None:
                print("⚠️ Skipping this cycle due to data fetch error")
                time.sleep(CHECK_INTERVAL)
                continue
            
            print(f"💰 SHIBUSD Price: ${price}")
            print(f"📊 24h Change: {change_24h:+.2f}%")
            
            # Analyze with AI
            signal = analyze_with_ai(price, change_24h)
            print(f"🤖 AI Signal: {signal}")
            
            # Send signal to Delta Exchange
            if signal != 'HOLD':
                send_signal_to_delta(signal)
            else:
                print("⏸️ HOLD - No trade executed")
            
            print(f"⏳ Next check in {CHECK_INTERVAL} seconds")
            time.sleep(CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            time.sleep(60)  # Wait 1 minute before retrying

if __name__ == "__main__":
    # Verify configuration
    if not GEMINI_API_KEY:
        print("❌ ERROR: GEMINI_API_KEY environment variable not set!")
        exit(1)
    
    if not DELTA_WEBHOOK_URL:
        print("❌ ERROR: DELTA_WEBHOOK_URL environment variable not set!")
        exit(1)
    
    main_loop()
