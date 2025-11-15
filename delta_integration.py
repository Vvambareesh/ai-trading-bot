import os
import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import hashlib
import hmac
import base64

@dataclass
class DeltaOrder:
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market', 'limit', 'stop'
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = 'GTC'  # Good Till Cancelled
    client_order_id: Optional[str] = None

@dataclass
class OrderResponse:
    success: bool
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: datetime = None

class DeltaIntegration:
    """
    Enhanced Delta Exchange integration with TradingView automation support
    Provides webhook-based trading with comprehensive error handling and retry logic
    """

    def __init__(self, webhook_url: str, api_key: str = None, api_secret: str = None):
        self.webhook_url = webhook_url
        self.api_key = api_key or os.getenv('DELTA_API_KEY')
        self.api_secret = api_secret or os.getenv('DELTA_API_SECRET')
        self.base_url = "https://api.delta.exchange"

        # Trading parameters
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        self.timeout = 30  # seconds

        # Symbol mapping for Delta Exchange
        self.symbol_map = {
            'BTC': 'BTCUSD',
            'ETH': 'ETHUSD',
            'ADA': 'ADAUSD',
            'DOGE': 'DOGEUSD',
            'SHIB': 'SHIBUSD'
        }

        # Order tracking
        self.pending_orders = {}
        self.order_history = []

        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'AITradingBot/1.0'
        })

        print(f"🔗 Delta Exchange Integration Initialized")
        print(f"   Webhook URL: {self.webhook_url}")
        print(f"   Base API URL: {self.base_url}")

    def generate_signature(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """
        Generate HMAC signature for API authentication
        """
        if not self.api_secret:
            return ""

        # Create message to sign
        message = f"{timestamp}{method.upper()}{path}{body}"

        # Generate signature
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()

        # Base64 encode
        signature_b64 = base64.b64encode(signature).decode('utf-8')

        return signature_b64

    def send_order(self, symbol: str, action: str, quantity: float, price: float,
                   order_type: str = 'market') -> OrderResponse:
        """
        Send order to Delta Exchange via webhook
        """
        try:
            # Map symbol to Delta format
            delta_symbol = self.symbol_map.get(symbol.upper(), f"{symbol.upper()}USD")
            delta_side = action.lower()

            # Create webhook payload
            webhook_payload = {
                "symbol": delta_symbol,
                "side": delta_side,
                "qty": quantity,
                "type": order_type,
                "trigger_time": datetime.now().isoformat(),
                "client_order_id": f"AI_{int(time.time())}_{symbol}"
            }

            # Add price for non-market orders
            if order_type != 'market' and price:
                webhook_payload["price"] = price

            # Send to webhook
            response = self.session.post(
                self.webhook_url,
                json=webhook_payload,
                timeout=self.timeout
            )

            # Process response
            if response.status_code == 200:
                return OrderResponse(
                    success=True,
                    client_order_id=webhook_payload["client_order_id"],
                    timestamp=datetime.now()
                )
            else:
                return OrderResponse(
                    success=False,
                    error_message=f"HTTP {response.status_code}: {response.text}",
                    client_order_id=webhook_payload["client_order_id"],
                    timestamp=datetime.now()
                )

        except requests.exceptions.Timeout:
            return OrderResponse(
                success=False,
                error_message="Request timeout",
                timestamp=datetime.now()
            )
        except requests.exceptions.ConnectionError:
            return OrderResponse(
                success=False,
                error_message="Connection error",
                timestamp=datetime.now()
            )
        except Exception as e:
            return OrderResponse(
                success=False,
                error_message=f"Unexpected error: {str(e)}",
                timestamp=datetime.now()
            )

    def send_order_with_retry(self, symbol: str, action: str, quantity: float,
                           price: float, order_type: str = 'market') -> OrderResponse:
        """
        Send order with retry logic
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self.send_order(symbol, action, quantity, price, order_type)

                if response.success:
                    # Track successful order
                    self.order_history.append({
                        'symbol': symbol,
                        'action': action,
                        'quantity': quantity,
                        'price': price,
                        'order_type': order_type,
                        'client_order_id': response.client_order_id,
                        'timestamp': response.timestamp,
                        'attempt': attempt + 1
                    })
                    return response
                else:
                    last_error = response.error_message
                    if attempt < self.max_retries - 1:
                        print(f"⚠️ Order attempt {attempt + 1} failed: {last_error}")
                        print(f"   Retrying in {self.retry_delay} seconds...")
                        time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff

            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries - 1:
                    print(f"⚠️ Order attempt {attempt + 1} error: {last_error}")
                    time.sleep(self.retry_delay * (2 ** attempt))

        # All attempts failed
        return OrderResponse(
            success=False,
            error_message=f"All {self.max_retries} attempts failed. Last error: {last_error}",
            timestamp=datetime.now()
        )

    def close_position(self, symbol: str, current_price: float) -> bool:
        """
        Close position for given symbol
        """
        try:
            # For now, we'll use a simple market order
            # In production, you would track actual position size
            quantity = 0.01  # Default small quantity for demo

            response = self.send_order_with_retry(
                symbol=symbol,
                action='sell',  # Assuming long position, would need to check actual position
                quantity=quantity,
                price=current_price,
                order_type='market'
            )

            if response.success:
                print(f"✅ Closed {symbol} position at {current_price}")
                return True
            else:
                print(f"❌ Failed to close {symbol} position: {response.error_message}")
                return False

        except Exception as e:
            print(f"❌ Error closing {symbol} position: {e}")
            return False

    def get_account_balance(self) -> Optional[Dict]:
        """
        Get account balance from Delta Exchange
        Note: This requires API authentication, not just webhook
        """
        if not self.api_key:
            print("⚠️ API key not configured - cannot fetch balance")
            return None

        try:
            timestamp = str(int(time.time()))
            path = "/v2/account/balances"
            signature = self.generate_signature(timestamp, 'GET', path)

            headers = {
                'api-key': self.api_key,
                'signature': signature,
                'timestamp': timestamp
            }

            response = self.session.get(
                f"{self.base_url}{path}",
                headers=headers,
                timeout=self.timeout
            )

            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Failed to get balance: {response.status_code} {response.text}")
                return None

        except Exception as e:
            print(f"❌ Error getting balance: {e}")
            return None

    def get_open_orders(self) -> List[Dict]:
        """
        Get open orders from Delta Exchange
        Note: This requires API authentication
        """
        if not self.api_key:
            print("⚠️ API key not configured - cannot fetch orders")
            return []

        try:
            timestamp = str(int(time.time()))
            path = "/v2/orders"
            signature = self.generate_signature(timestamp, 'GET', path)

            headers = {
                'api-key': self.api_key,
                'signature': signature,
                'timestamp': timestamp
            }

            response = self.session.get(
                f"{self.base_url}{path}",
                headers=headers,
                timeout=self.timeout
            )

            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Failed to get orders: {response.status_code} {response.text}")
                return []

        except Exception as e:
            print(f"❌ Error getting orders: {e}")
            return []

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel specific order
        Note: This requires API authentication
        """
        if not self.api_key:
            print("⚠️ API key not configured - cannot cancel order")
            return False

        try:
            timestamp = str(int(time.time()))
            path = f"/v2/orders/{order_id}"
            signature = self.generate_signature(timestamp, 'DELETE', path)

            headers = {
                'api-key': self.api_key,
                'signature': signature,
                'timestamp': timestamp
            }

            response = self.session.delete(
                f"{self.base_url}{path}",
                headers=headers,
                timeout=self.timeout
            )

            if response.status_code == 200:
                print(f"✅ Order {order_id} cancelled successfully")
                return True
            else:
                print(f"❌ Failed to cancel order {order_id}: {response.status_code} {response.text}")
                return False

        except Exception as e:
            print(f"❌ Error cancelling order {order_id}: {e}")
            return False

    def test_webhook(self) -> bool:
        """
        Test webhook connectivity
        """
        try:
            # Send a small test order (or just a ping if Delta supports it)
            test_payload = {
                "test": True,
                "timestamp": datetime.now().isoformat(),
                "source": "AITradingBot"
            }

            response = self.session.post(
                self.webhook_url,
                json=test_payload,
                timeout=10
            )

            if response.status_code == 200:
                print("✅ Delta Exchange webhook test successful")
                return True
            else:
                print(f"❌ Webhook test failed: {response.status_code} {response.text}")
                return False

        except Exception as e:
            print(f"❌ Webhook test error: {e}")
            return False

    def setup_tradingview_webhook(self) -> Dict:
        """
        Generate TradingView webhook setup instructions
        """
        webhook_info = {
            "webhook_url": self.webhook_url,
            "message_format": {
                "symbol": "{{ticker}}",
                "side": "{{strategy.order.action}}",
                "qty": "{{strategy.order.contracts}}",
                "type": "market",
                "trigger_time": "{{timenow}}",
                "client_order_id": "TV_{{timenow}}"
            },
            "alert_settings": {
                "webhook_enabled": True,
                "message": json.dumps({
                    "symbol": "{{ticker}}",
                    "side": "{{strategy.order.action}}",
                    "qty": "{{strategy.order.contracts}}",
                    "type": "market",
                    "trigger_time": "{{timenow}}",
                    "client_order_id": "TV_{{timenow}}"
                }),
                "conditions": [
                    "Strategy: Your Trading Strategy Name"
                ]
            },
            "symbol_mapping": {
                "BINANCE:BTCUSDT": "BTCUSD",
                "BINANCE:ETHUSDT": "ETHUSD",
                "BINANCE:ADAUSDT": "ADAUSD",
                "BINANCE:DOGEUSDT": "DOGEUSD",
                "BINANCE:SHIBUSDT": "SHIBUSD"
            }
        }

        return webhook_info

    def get_order_statistics(self) -> Dict:
        """
        Get statistics about order history
        """
        if not self.order_history:
            return {
                'total_orders': 0,
                'successful_orders': 0,
                'failed_orders': 0,
                'success_rate': 0,
                'avg_execution_time': 0,
                'symbols_traded': [],
                'order_types': {}
            }

        total_orders = len(self.order_history)
        successful_orders = len([o for o in self.order_history if 'error' not in o])
        failed_orders = total_orders - successful_orders
        success_rate = (successful_orders / total_orders * 100) if total_orders > 0 else 0

        symbols_traded = list(set([o['symbol'] for o in self.order_history]))
        order_types = {}
        for order in self.order_history:
            order_type = order.get('order_type', 'unknown')
            order_types[order_type] = order_types.get(order_type, 0) + 1

        return {
            'total_orders': total_orders,
            'successful_orders': successful_orders,
            'failed_orders': failed_orders,
            'success_rate': success_rate,
            'symbols_traded': symbols_traded,
            'order_types': order_types,
            'recent_orders': self.order_history[-10:]  # Last 10 orders
        }

    def save_order_history(self, filename: str = None):
        """
        Save order history to file
        """
        if filename is None:
            filename = f"delta_orders_{datetime.now().strftime('%Y%m%d')}.json"

        try:
            with open(filename, 'w') as f:
                json.dump(self.order_history, f, indent=2, default=str)

            print(f"💾 Order history saved to {filename}")

        except Exception as e:
            print(f"❌ Error saving order history: {e}")

    def create_tradingview_alert_json(self) -> str:
        """
        Create JSON template for TradingView alerts
        """
        alert_template = {
            "description": "AI Trading Bot Alert for Delta Exchange",
            "name": "AI Trading Bot Alert",
            "condition": "Your strategy condition here",
            "options": {
                "close_position": True,
                "send_to_webhook": True
            },
            "text": json.dumps({
                "symbol": "{{ticker}}",
                "side": "{{strategy.order.action}}",
                "qty": "{{strategy.order.contracts}}",
                "price": "{{close}}",
                "type": "market",
                "trigger_time": "{{timenow}}",
                "source": "TradingView",
                "client_order_id": "TV_{{timenow}}"
            }),
            "webhook": {
                "url": self.webhook_url,
                "method": "POST",
                "headers": {
                    "Content-Type": "application/json"
                }
            }
        }

        return json.dumps(alert_template, indent=2)

if __name__ == "__main__":
    # Example usage
    print("🔗 Testing Delta Exchange Integration...")

    # Test with webhook URL (would use environment variable in production)
    webhook_url = os.getenv('DELTA_WEBHOOK_URL', 'https://example.com/webhook')

    if webhook_url != 'https://example.com/webhook':
        delta = DeltaIntegration(webhook_url)

        # Test webhook connectivity
        print("\n📡 Testing webhook connectivity...")
        webhook_test = delta.test_webhook()

        # Test sending an order (paper trade)
        if webhook_test:
            print("\n📊 Sending test order...")
            order_response = delta.send_order_with_retry(
                symbol='BTC',
                action='buy',
                quantity=0.001,
                price=45000,
                order_type='limit'
            )

            if order_response.success:
                print(f"✅ Test order sent successfully")
                print(f"   Client Order ID: {order_response.client_order_id}")
            else:
                print(f"❌ Test order failed: {order_response.error_message}")

        # Get order statistics
        stats = delta.get_order_statistics()
        print(f"\n📈 Order Statistics:")
        print(f"   Total Orders: {stats['total_orders']}")
        print(f"   Success Rate: {stats['success_rate']:.1f}%")
        print(f"   Symbols Traded: {', '.join(stats['symbols_traded'])}")

        # Generate TradingView setup
        print(f"\n📱 TradingView Webhook Setup:")
        tv_info = delta.setup_tradingview_webhook()
        print(f"   Webhook URL: {tv_info['webhook_url']}")
        print(f"   Message Format: {json.dumps(tv_info['message_format'], indent=6)}")

        # Save setup instructions
        with open('tradingview_setup.json', 'w') as f:
            json.dump(tv_info, f, indent=2)
        print("   📄 TradingView setup saved to tradingview_setup.json")

    else:
        print("⚠️ Please set DELTA_WEBHOOK_URL environment variable to test Delta integration")
        print("   Example: export DELTA_WEBHOOK_URL='https://delta.exchange/webhook/your-webhook-id'")