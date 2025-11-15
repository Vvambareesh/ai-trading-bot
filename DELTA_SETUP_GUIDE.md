# 🔗 Delta Exchange Setup Guide

## 📋 What You Need to Provide

To set up Delta Exchange integration with your AI trading bot, you need:

1. **Delta Exchange Account** (with 2FA enabled)
2. **Delta API Key and Secret** (from account settings)
3. **Webhook URL** (from Delta Exchange TradingBot page)
4. **TradingView Account** (optional, for charting and automation)

---

## 🚀 Quick Setup Steps

### Step 1: Get Delta Exchange API Credentials

1. **Sign up/login** to [Delta Exchange](https://delta.exchange)
2. **Enable 2FA** in Security Settings (highly recommended)
3. **Go to** Account Settings → API Keys
4. **Click "Generate New API Key"**
5. **Fill in the form:**
   - **Name**: AI Trading Bot
   - **Permissions**:
     - ✅ Read Account Balance
     - ✅ Place Orders
     - ✅ Cancel Orders
     - ✅ Read Order History
   - **IP Whitelist**: Add your server IP (optional but recommended)
6. **Copy** the API Key and Secret
7. **Store securely** - never commit to version control

### Step 2: Create Delta Webhook

1. **Go to** Algo dropdown → TradingBot
2. **Click "Create New Webhook"**
3. **Configure webhook:**
   - **Name**: AI Trading Bot Webhook
   - **Description**: Automated trading signals from AI bot
   - **Symbol Mapping**: Map TradingView symbols to Delta symbols
4. **Copy the Webhook URL** (looks like: `https://delta.exchange/webhook/xxxxx`)
5. **Note the Webhook ID** for TradingView setup

### Step 3: Update Your .env File

```bash
# Copy the template
cp .env.example .env

# Edit and add your Delta credentials
nano .env  # or use your preferred editor
```

Add these lines to your `.env`:

```bash
# Delta Exchange API Configuration
DELTA_API_KEY=your_actual_delta_api_key_here
DELTA_API_SECRET=your_actual_delta_api_secret_here
DELTA_WEBHOOK_URL=https://delta.exchange/webhook/your_webhook_id
DELTA_ENVIRONMENT=production

# Trading configuration (keep existing)
TRADING_MODE=paper  # Start with paper trading
STRATEGY=hybrid
INITIAL_CAPITAL_INR=1000
```

### Step 4: Verify Setup

```bash
# Run the setup helper
python setup_delta.py

# Choose option 3 to update credentials
# Enter your Delta API key, secret, and webhook URL
```

---

## 📱 TradingView Integration (Optional but Recommended)

### Step 1: Create TradingView Strategy

1. **Open TradingView** and go to Pine Editor
2. **Create or edit a strategy** with your preferred indicators
3. **Add strategy.alertcondition()** for buy/sell signals

Example Pine Script alert conditions:
```pine
// Buy signal
strategy.entry("Long", strategy.long, when = sma_cross_above and rsi_oversold)

// Sell signal
strategy.close("Long", when = sma_cross_below or rsi_overbought)
```

### Step 2: Set Up TradingView Alert

1. **Go to** Chart → Alert Dialog
2. **Configure alert:**
   - **Condition**: Your strategy alert
   - **Webhook URL**: `https://delta.exchange/webhook/your_webhook_id`
   - **Message Template**:
     ```
     {
       "symbol": "{{ticker}}",
       "side": "{{strategy.order.action}}",
       "qty": "{{strategy.order.contracts}}",
       "type": "market",
       "trigger_time": "{{timenow}}"
     }
     ```
   - **Expiry**: Keep as "Good until cancelled"

3. **Test** the alert with a paper trade first

---

## 🔧 Advanced Configuration

### Symbol Mapping

Ensure TradingView symbols match Delta Exchange format:

| TradingView | Delta Exchange |
|-------------|---------------|
| BINANCE:BTCUSDT | BTCUSD |
| BINANCE:ETHUSDT | ETHUSD |
| BINANCE:ADAUSDT | ADAUSD |
| BINANCE:DOGEUSDT | DOGEUSD |
| BINANCE:SHIBUSDT | SHIBUSD |

### Order Types Supported

Your bot supports:
- **Market Orders**: Immediate execution
- **Limit Orders**: Specific price execution
- **Stop Loss Orders**: Automatic risk protection
- **Take Profit Orders**: Automatic profit taking

### Risk Management

The bot implements:
- **Maximum 5% risk per trade** (₹50 for ₹1000 capital)
- **2% stop loss** protection
- **6% take profit** targets (3:1 risk:reward)
- **Maximum 3 concurrent positions**
- **10% daily loss limit**

---

## 🧪 Testing Your Setup

### Paper Trading Test

```bash
# Start with paper mode first
python main.py --mode paper --strategy hybrid --capital 1000

# Monitor with dashboard
python dashboard.py  # in another terminal
```

### Live Trading Test

```bash
# After successful paper trading (2+ weeks)
python main.py --mode live --strategy hybrid --conviction 95

# Monitor performance closely
```

### Delta Webhook Test

```bash
# Test webhook connectivity
python delta_integration.py

# Verify message format
python -c "
from delta_integration import DeltaIntegration
delta = DeltaIntegration('https://delta.exchange/webhook/test')
print(delta.create_tradingview_alert_json())
"
```

---

## 🔒 Security Best Practices

### Account Security
1. **Enable 2FA** on Delta Exchange account
2. **Use unique, strong passwords**
3. **Regular API key rotation** (every 30-90 days)
4. **IP whitelisting** for API access
5. **Monitor account activity** regularly

### API Key Management
1. **Never commit** API keys to version control
2. **Use environment variables** (not hardcoded keys)
3. **Separate keys** for paper vs live trading
4. **Revoke unused API keys** immediately
5. **Keep secure backup** of credentials

### Webhook Security
1. **HTTPS only** for webhook URLs
2. **Validate webhook payloads** in your bot
3. **Rate limiting** on webhook endpoints
4. **Monitor webhook logs** for suspicious activity

---

## 📊 Performance Monitoring

### Key Metrics to Track
1. **Win Rate**: Target 60%+ for live trading
2. **Profit Factor**: Target 1.5+ (total profit/total loss)
3. **Sharpe Ratio**: Target 1.0+ for risk-adjusted returns
4. **Maximum Drawdown**: Keep below 20%
5. **Conviction Score**: Only trade with 95%+ for live

### Dashboard Features
Your trading dashboard provides:
- **Real-time portfolio value** tracking
- **Active positions** monitoring
- **P&L visualization** over time
- **Risk metrics** (drawdown, volatility)
- **Strategy performance** comparison
- **Trade history** with detailed logs

---

## 🚨 Troubleshooting

### Common Issues

#### API Connection Problems
```bash
# Check API credentials
python -c "
from dotenv import load_dotenv
load_dotenv()
print('API Key:', os.getenv('DELTA_API_KEY'))
print('API Secret:', os.getenv('DELTA_API_SECRET'))
print('Webhook URL:', os.getenv('DELTA_WEBHOOK_URL'))
"

# Test connection
python delta_integration.py
```

#### Webhook Issues
- **Check webhook URL** is correct and accessible
- **Verify JSON format** matches Delta requirements
- **Check TradingView alert** configuration
- **Monitor webhook logs** for errors

#### Trading Issues
- **Verify sufficient account balance**
- **Check position limits** (max 3 concurrent)
- **Confirm symbol mapping** is correct
- **Review API permissions**

### Getting Help

1. **Check Delta Exchange API documentation**
2. **Review your trading logs** in dashboard
3. **Test with small amounts** first
4. **Monitor system logs** for errors

---

## 📞 Support Resources

### Delta Exchange Support
- **Documentation**: [Delta Exchange API Docs](https://delta.exchange/docs/api)
- **Support**: support@delta.exchange
- **Status Page**: [Delta Status](https://status.delta.exchange)

### Trading Bot Support
- **Repository Issues**: Create GitHub issues
- **Community**: Join trading community discussions
- **Documentation**: Review README.md and code comments

---

## 🎯 Best Practices

### Before Going Live
1. **Paper trade minimum 2 weeks** with your strategy
2. **Achieve consistent profits** in paper trading
3. **Test all strategies** before selecting your primary
4. **Validate risk management** works correctly
5. **Ensure dashboard monitoring** is functioning

### Live Trading
1. **Start with small position sizes** (1-2% risk per trade)
2. **Monitor performance daily** with dashboard
3. **Adjust conviction threshold** based on strategy performance
4. **Keep detailed trade logs** for analysis
5. **Be prepared to stop** if drawdown exceeds limits

### Ongoing Management
1. **Regular strategy review** (weekly/monthly)
2. **Performance optimization** based on results
3. **Risk parameter adjustment** as needed
4. **API key rotation** every 60-90 days
5. **Backup trading data** regularly

---

## 🎉 Ready to Start!

Once you've completed these steps:

1. ✅ API credentials configured
2. ✅ Webhook set up and tested
3. ✅ Paper trading validated
4. ✅ Risk management confirmed
5. ✅ Dashboard monitoring ready

You're ready to start automated cryptocurrency trading with your AI bot!

### Final Commands
```bash
# Paper Trading
python main.py --mode paper --strategy hybrid --capital 1000

# Live Trading (when ready)
python main.py --mode live --strategy hybrid --conviction 95

# Dashboard Monitoring
python dashboard.py
```

**🚀 Happy trading! May your AI strategies bring consistent profits!**

---

*Last updated: 2024-11-15*