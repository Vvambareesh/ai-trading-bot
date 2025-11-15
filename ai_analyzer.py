import os
import json
import pandas as pd
import numpy as np
import google.generativeai as genai
import talib
import requests
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import re

@dataclass
class AISignal:
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    conviction: float  # 0-100 conviction score
    confidence_intervals: Dict[str, Tuple[float, float]]  # Price targets with confidence
    risk_assessment: str  # 'LOW', 'MEDIUM', 'HIGH'
    market_sentiment: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    reasoning: str
    technical_analysis: Dict
    fundamental_factors: Dict
    ai_confidence: float  # AI model's confidence in its own analysis
    timestamp: datetime

class AIAnalyzer:
    """
    Advanced AI-powered market analysis with conviction scoring
    Integrates technical indicators, market sentiment, and multi-timeframe analysis
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable must be set")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-pro')

        # Analysis parameters
        self.timeframes = ['1h', '4h', '1d', '1w']  # Multi-timeframe analysis
        self.supported_symbols = ['BTC', 'ETH', 'ADA', 'DOGE', 'SHIB']

        # Conviction scoring thresholds
        self.conviction_thresholds = {
            'very_low': (0, 30),
            'low': (30, 60),
            'medium': (60, 80),
            'high': (80, 95),
            'very_high': (95, 100)
        }

        # Risk levels
        self.risk_levels = {
            'LOW': (0, 0.02),    # <2% volatility
            'MEDIUM': (0.02, 0.05),  # 2-5% volatility
            'HIGH': (0.05, 1.0)   # >5% volatility
        }

    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive technical indicators
        """
        if len(data) < 50:  # Need minimum data for indicators
            return {}

        indicators = {}

        # Trend indicators
        indicators['sma_10'] = talib.SMA(data['close'], timeperiod=10)[-1] if len(data) >= 10 else None
        indicators['sma_20'] = talib.SMA(data['close'], timeperiod=20)[-1] if len(data) >= 20 else None
        indicators['sma_50'] = talib.SMA(data['close'], timeperiod=50)[-1] if len(data) >= 50 else None
        indicators['ema_12'] = talib.EMA(data['close'], timeperiod=12)[-1] if len(data) >= 12 else None
        indicators['ema_26'] = talib.EMA(data['close'], timeperiod=26)[-1] if len(data) >= 26 else None

        # Momentum indicators
        indicators['rsi_14'] = talib.RSI(data['close'], timeperiod=14)[-1] if len(data) >= 14 else None
        indicators['rsi_30'] = talib.RSI(data['close'], timeperiod=30)[-1] if len(data) >= 30 else None

        # MACD
        if len(data) >= 35:  # 26 + 9 for signal line
            macd, macd_signal, macd_hist = talib.MACD(data['close'])
            indicators['macd'] = macd[-1]
            indicators['macd_signal'] = macd_signal[-1]
            indicators['macd_histogram'] = macd_hist[-1]

        # Volatility indicators
        indicators['bollinger_upper'], indicators['bollinger_middle'], indicators['bollinger_lower'] = talib.BBANDS(
            data['close'], timeperiod=20
        )[:, -1] if len(data) >= 20 else (None, None, None)

        indicators['atr_14'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)[-1] if len(data) >= 14 else None

        # Volume indicators
        if 'volume' in data.columns:
            indicators['volume_sma_20'] = talib.SMA(data['volume'], timeperiod=20)[-1] if len(data) >= 20 else None
            indicators['ad'] = talib.AD(data['high'], data['low'], data['close'], data['volume'])[-1] if len(data) >= 14 else None
            indicators['obv'] = talib.OBV(data['close'], data['volume'])[-1] if len(data) >= 14 else None

        # Pattern recognition
        indicators['doji'] = talib.CDLDOJI(data['open'], data['high'], data['low'], data['close'])[-1] if len(data) >= 14 else None
        indicators['hammer'] = talib.CDLHAMMER(data['open'], data['high'], data['low'], data['close'])[-1] if len(data) >= 14 else None
        indicators['engulfing'] = talib.CDLENGULFING(data['open'], data['high'], data['low'], data['close'])[-1] if len(data) >= 14 else None

        return indicators

    def calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate current volatility"""
        if len(data) < 20:
            return 0.0

        returns = data['close'].pct_change().dropna()
        volatility = returns.rolling(window=20).std().iloc[-1]

        return volatility

    def assess_risk_level(self, data: pd.DataFrame) -> str:
        """Assess risk level based on volatility and other factors"""
        volatility = self.calculate_volatility(data)

        for risk_level, (min_vol, max_vol) in self.risk_levels.items():
            if min_vol <= volatility < max_vol:
                return risk_level

        return 'HIGH'

    def generate_price_targets(self, current_price: float, indicators: Dict, action: str) -> Dict[str, Tuple[float, float]]:
        """
        Generate price targets with confidence intervals
        Returns: {'short_term': (lower, upper), 'medium_term': (lower, upper), 'long_term': (lower, upper)}
        """
        targets = {}

        # Use ATR for volatility-based targets
        atr = indicators.get('atr_14', current_price * 0.02)  # Default 2% if no ATR

        # Bollinger Bands for targets
        bb_upper = indicators.get('bollinger_upper', current_price * 1.02)
        bb_lower = indicators.get('bollinger_lower', current_price * 0.98)

        if action == 'BUY':
            # Conservative targets for BUY
            targets['short_term'] = (
                current_price * 0.98,  # Stop loss: 2% below
                current_price + atr * 2  # Take profit: 2 ATR above
            )
            targets['medium_term'] = (
                current_price * 0.95,  # Stop loss: 5% below
                bb_upper  # Target: upper Bollinger Band
            )
            targets['long_term'] = (
                current_price * 0.90,  # Stop loss: 10% below
                current_price * 1.15   # Target: 15% above
            )

        elif action == 'SELL':
            # Conservative targets for SELL
            targets['short_term'] = (
                current_price - atr * 2,  # Take profit: 2 ATR below
                current_price * 1.02      # Stop loss: 2% above
            )
            targets['medium_term'] = (
                bb_lower,               # Target: lower Bollinger Band
                current_price * 1.05    # Stop loss: 5% above
            )
            targets['long_term'] = (
                current_price * 0.85,   # Target: 15% below
                current_price * 1.10    # Stop loss: 10% above
            )

        else:  # HOLD
            # Neutral targets
            targets['short_term'] = (
                current_price * 0.98,
                current_price * 1.02
            )
            targets['medium_term'] = (
                current_price * 0.95,
                current_price * 1.05
            )
            targets['long_term'] = (
                current_price * 0.90,
                current_price * 1.10
            )

        return targets

    def analyze_market_sentiment(self, symbol: str, data: pd.DataFrame) -> str:
        """
        Analyze market sentiment based on technical indicators
        """
        if len(data) < 20:
            return 'NEUTRAL'

        indicators = self.calculate_technical_indicators(data)

        # Get key indicators
        rsi = indicators.get('rsi_14', 50)
        macd_hist = indicators.get('macd_histogram', 0)
        sma_20 = indicators.get('sma_20')
        current_price = data['close'].iloc[-1]

        # Calculate sentiment score
        sentiment_score = 0

        # RSI contribution
        if rsi > 70:
            sentiment_score -= 2  # Overbought
        elif rsi < 30:
            sentiment_score += 2  # Oversold
        elif rsi > 50:
            sentiment_score += 1  # Bullish
        else:
            sentiment_score -= 1  # Bearish

        # MACD contribution
        if macd_hist > 0:
            sentiment_score += 1  # Bullish
        else:
            sentiment_score -= 1  # Bearish

        # SMA contribution
        if sma_20 and current_price > sma_20:
            sentiment_score += 1  # Above average
        elif sma_20 and current_price < sma_20:
            sentiment_score -= 1  # Below average

        # Trend contribution (simple price momentum)
        if len(data) >= 10:
            short_trend = (data['close'].iloc[-1] - data['close'].iloc[-10]) / data['close'].iloc[-10]
            if short_trend > 0.05:  # 5% up in 10 periods
                sentiment_score += 2
            elif short_trend < -0.05:  # 5% down in 10 periods
                sentiment_score -= 2
            elif short_trend > 0:
                sentiment_score += 1
            else:
                sentiment_score -= 1

        # Determine sentiment
        if sentiment_score >= 3:
            return 'BULLISH'
        elif sentiment_score <= -3:
            return 'BEARISH'
        else:
            return 'NEUTRAL'

    def create_ai_prompt(self, symbol: str, data: pd.DataFrame, indicators: Dict, sentiment: str) -> str:
        """
        Create comprehensive prompt for AI analysis
        """
        current_price = data['close'].iloc[-1]
        price_change_24h = ((data['close'].iloc[-1] - data['close'].iloc[-2]) / data['close'].iloc[-2]) * 100 if len(data) >= 2 else 0
        volatility = self.calculate_volatility(data) * 100

        # Format indicators for prompt
        indicator_text = f"""
Current Price: ${current_price:.6f}
24h Change: {price_change_24h:+.2f}%
Volatility: {volatility:.2f}%

Key Technical Indicators:
- RSI (14): {indicators.get('rsi_14', 'N/A')}
- MACD Histogram: {indicators.get('macd_histogram', 'N/A')}
- SMA 20: ${indicators.get('sma_20', 'N/A')}
- Bollinger Bands: Upper=${indicators.get('bollinger_upper', 'N/A')}, Lower=${indicators.get('bollinger_lower', 'N/A')}
- ATR (14): {indicators.get('atr_14', 'N/A')}

Market Sentiment: {sentiment}

Recent Price Action (Last 10 periods):
{data['close'].tail(10).tolist()}
"""

        prompt = f"""
You are an expert cryptocurrency trading analyst with deep knowledge of technical analysis, market psychology, and risk management.

Analyze {symbol} (cryptocurrency) trading conditions based on the following data:

{indicator_text}

Provide a comprehensive analysis covering:

1. **Technical Analysis**: Evaluate trend strength, momentum, and key technical patterns
2. **Market Psychology**: Assess current market sentiment and potential crowd behavior
3. **Risk Assessment**: Evaluate volatility and potential downside risk
4. **Trading Recommendation**: Provide BUY, SELL, or HOLD recommendation
5. **Conviction Score**: Rate your confidence from 0-100% (0=lowest, 100=highest)

**Scoring Guidelines:**
- 95-100%: Very high conviction, multiple strong confirmations
- 80-95%: High conviction, strong technical confirmation
- 60-80%: Medium conviction, decent technical signals
- 30-60%: Low conviction, weak or conflicting signals
- 0-30%: Very low conviction, avoid trading

**Response Format:**
Provide your analysis in JSON format:
{{
    "action": "BUY/SELL/HOLD",
    "conviction": 0-100,
    "reasoning": "Detailed explanation of your analysis",
    "key_factors": ["factor1", "factor2", "factor3"],
    "risk_factors": ["risk1", "risk2"],
    "technical_strength": "WEAK/MODERATE/STRONG",
    "ai_confidence": 0-100
}}
"""

        return prompt

    def parse_ai_response(self, response_text: str) -> Dict:
        """Parse AI response and extract JSON"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                # Fallback parsing if no JSON found
                return {
                    'action': 'HOLD',
                    'conviction': 50,
                    'reasoning': response_text[:200],
                    'key_factors': [],
                    'risk_factors': ['AI response parsing failed'],
                    'technical_strength': 'MODERATE',
                    'ai_confidence': 30
                }
        except json.JSONDecodeError as e:
            print(f"Error parsing AI response: {e}")
            return {
                'action': 'HOLD',
                'conviction': 30,
                'reasoning': 'Failed to parse AI response',
                'key_factors': [],
                'risk_factors': ['JSON parsing error'],
                'technical_strength': 'WEAK',
                'ai_confidence': 20
            }

    def analyze_symbol(self, symbol: str, data: pd.DataFrame) -> Optional[AISignal]:
        """
        Perform comprehensive AI analysis for a cryptocurrency symbol
        """
        try:
            if len(data) < 50:
                print(f"❌ Insufficient data for {symbol} analysis")
                return None

            print(f"🧠 Analyzing {symbol} with AI...")

            # Calculate technical indicators
            indicators = self.calculate_technical_indicators(data)

            # Assess risk level
            risk_assessment = self.assess_risk_level(data)

            # Analyze market sentiment
            market_sentiment = self.analyze_market_sentiment(symbol, data)

            # Create AI prompt
            prompt = self.create_ai_prompt(symbol, data, indicators, market_sentiment)

            # Get AI analysis
            response = self.model.generate_content(prompt)
            ai_analysis = self.parse_ai_response(response.text)

            # Validate AI response
            action = ai_analysis.get('action', 'HOLD').upper()
            if action not in ['BUY', 'SELL', 'HOLD']:
                action = 'HOLD'

            conviction = min(100, max(0, float(ai_analysis.get('conviction', 50))))
            ai_confidence = min(100, max(0, float(ai_analysis.get('ai_confidence', 50))))

            # Generate price targets
            current_price = data['close'].iloc[-1]
            price_targets = self.generate_price_targets(current_price, indicators, action)

            # Combine conviction with AI confidence
            final_conviction = (conviction * 0.7) + (ai_confidence * 0.3)

            # Get fundamental factors (basic for now)
            fundamental_factors = {
                'market_cap_rank': self.get_market_cap_rank(symbol),
                'trading_volume_24h': data['volume'].iloc[-1] if 'volume' in data.columns else None,
                'price_change_7d': self.calculate_price_change(data, 7),
                'price_change_30d': self.calculate_price_change(data, 30)
            }

            # Create AI signal
            signal = AISignal(
                symbol=symbol,
                action=action,
                conviction=final_conviction,
                confidence_intervals=price_targets,
                risk_assessment=risk_assessment,
                market_sentiment=market_sentiment,
                reasoning=ai_analysis.get('reasoning', 'AI analysis'),
                technical_analysis=indicators,
                fundamental_factors=fundamental_factors,
                ai_confidence=ai_confidence,
                timestamp=datetime.now()
            )

            print(f"✅ AI Analysis for {symbol}:")
            print(f"   Action: {action}")
            print(f"   Conviction: {final_conviction:.1f}%")
            print(f"   Risk: {risk_assessment}")
            print(f"   Sentiment: {market_sentiment}")
            print(f"   Reasoning: {ai_analysis.get('reasoning', 'N/A')[:100]}...")

            return signal

        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
            return None

    def get_market_cap_rank(self, symbol: str) -> int:
        """Get market cap rank for symbol (mock implementation)"""
        # In production, this would fetch from CoinGecko/CoinMarketCap
        ranks = {'BTC': 1, 'ETH': 2, 'ADA': 8, 'DOGE': 10, 'SHIB': 16}
        return ranks.get(symbol, 999)

    def calculate_price_change(self, data: pd.DataFrame, periods: int) -> float:
        """Calculate price change over given periods"""
        if len(data) < periods:
            return 0.0

        old_price = data['close'].iloc[-periods-1] if len(data) > periods else data['close'].iloc[0]
        current_price = data['close'].iloc[-1]

        return ((current_price - old_price) / old_price) * 100 if old_price > 0 else 0.0

    def get_conviction_category(self, conviction: float) -> str:
        """Get conviction category based on score"""
        for category, (min_val, max_val) in self.conviction_thresholds.items():
            if min_val <= conviction < max_val:
                return category.replace('_', ' ').title()
        return 'Very High'

    def batch_analyze(self, symbols: List[str], data_dict: Dict[str, pd.DataFrame]) -> Dict[str, AISignal]:
        """Analyze multiple symbols in batch"""
        results = {}

        for symbol in symbols:
            if symbol in data_dict:
                signal = self.analyze_symbol(symbol, data_dict[symbol])
                if signal:
                    results[symbol] = signal

        return results

    def generate_analysis_report(self, signals: Dict[str, AISignal]) -> Dict:
        """Generate comprehensive analysis report"""
        if not signals:
            return {}

        report = {
            'timestamp': datetime.now().isoformat(),
            'total_symbols_analyzed': len(signals),
            'summary': {
                'buy_signals': len([s for s in signals.values() if s.action == 'BUY']),
                'sell_signals': len([s for s in signals.values() if s.action == 'SELL']),
                'hold_signals': len([s for s in signals.values() if s.action == 'HOLD']),
                'avg_conviction': np.mean([s.conviction for s in signals.values()]),
                'high_conviction_trades': len([s for s in signals.values() if s.conviction >= 95])
            },
            'risk_distribution': {
                'low_risk': len([s for s in signals.values() if s.risk_assessment == 'LOW']),
                'medium_risk': len([s for s in signals.values() if s.risk_assessment == 'MEDIUM']),
                'high_risk': len([s for s in signals.values() if s.risk_assessment == 'HIGH'])
            },
            'sentiment_distribution': {
                'bullish': len([s for s in signals.values() if s.market_sentiment == 'BULLISH']),
                'bearish': len([s for s in signals.values() if s.market_sentiment == 'BEARISH']),
                'neutral': len([s for s in signals.values() if s.market_sentiment == 'NEUTRAL'])
            },
            'top_opportunities': []
        }

        # Find top opportunities (high conviction BUY or SELL signals)
        high_conviction_signals = sorted(
            [s for s in signals.values() if s.action in ['BUY', 'SELL'] and s.conviction >= 80],
            key=lambda x: x.conviction,
            reverse=True
        )

        for signal in high_conviction_signals[:3]:  # Top 3 opportunities
            report['top_opportunities'].append({
                'symbol': signal.symbol,
                'action': signal.action,
                'conviction': signal.conviction,
                'risk': signal.risk_assessment,
                'sentiment': signal.market_sentiment,
                'reasoning': signal.reasoning[:200] + '...' if len(signal.reasoning) > 200 else signal.reasoning
            })

        return report

if __name__ == "__main__":
    # Example usage
    print("🤖 Testing AI Analyzer...")

    # Test with mock data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    prices = 45000 + np.cumsum(np.random.randn(100) * 100)
    volumes = 1000 + np.random.randn(100) * 100

    test_data = pd.DataFrame({
        'open': np.roll(prices, 1),
        'high': prices * (1 + np.abs(np.random.randn(100) * 0.01)),
        'low': prices * (1 - np.abs(np.random.randn(100) * 0.01)),
        'close': prices,
        'volume': volumes
    }, index=dates)

    # Test AI analysis (requires actual API key)
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            analyzer = AIAnalyzer(api_key)
            signal = analyzer.analyze_symbol('BTC', test_data)

            if signal:
                print(f"\n📊 AI Signal Generated:")
                print(f"   Symbol: {signal.symbol}")
                print(f"   Action: {signal.action}")
                print(f"   Conviction: {signal.conviction:.1f}%")
                print(f"   Category: {analyzer.get_conviction_category(signal.conviction)}")
                print(f"   Risk: {signal.risk_assessment}")
                print(f"   Sentiment: {signal.market_sentiment}")
                print(f"   AI Confidence: {signal.ai_confidence:.1f}%")

                # Generate report
                signals = {'BTC': signal}
                report = analyzer.generate_analysis_report(signals)
                print(f"\n📋 Analysis Report:")
                for key, value in report['summary'].items():
                    print(f"   {key}: {value}")

        else:
            print("⚠️ GEMINI_API_KEY not found. Skipping AI analysis test.")

    except Exception as e:
        print(f"❌ Error testing AI analyzer: {e}")