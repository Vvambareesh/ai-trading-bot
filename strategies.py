import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class TradingSignal:
    symbol: str
    strategy: str
    action: str  # 'BUY', 'SELL', or 'HOLD'
    price: float
    conviction: float  # 0-100 conviction score
    reasoning: str
    indicators: Dict
    timestamp: datetime

class BaseStrategy:
    """Base class for all trading strategies"""

    def __init__(self, name: str):
        self.name = name
        self.last_signals = {}

    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Optional[TradingSignal]:
        """Generate trading signal for given symbol and data"""
        raise NotImplementedError("Subclasses must implement generate_signal")

class SMAStrategy(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy
    Uses 10-day and 20-day moving averages for entry/exit signals
    """

    def __init__(self, fast_period: int = 10, slow_period: int = 20):
        super().__init__("SMA_Crossover")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Optional[TradingSignal]:
        if len(data) < self.slow_period:
            return None

        # Calculate SMAs
        data['sma_fast'] = talib.SMA(data['close'], timeperiod=self.fast_period)
        data['sma_slow'] = talib.SMA(data['close'], timeperiod=self.slow_period)

        # Get latest values
        current_price = data['close'].iloc[-1]
        sma_fast = data['sma_fast'].iloc[-1]
        sma_slow = data['sma_slow'].iloc[-1]
        prev_sma_fast = data['sma_fast'].iloc[-2]
        prev_sma_slow = data['sma_slow'].iloc[-2]

        # Crossover detection
        if sma_fast > sma_slow and prev_sma_fast <= prev_sma_slow:
            # Golden cross - BUY signal
            conviction = min(75, abs(sma_fast - sma_slow) / sma_slow * 1000)
            return TradingSignal(
                symbol=symbol,
                strategy=self.name,
                action='BUY',
                price=current_price,
                conviction=conviction,
                reasoning=f"Golden cross: {self.fast_period}-day SMA ({sma_fast:.4f}) crossed above {self.slow_period}-day SMA ({sma_slow:.4f})",
                indicators={
                    'sma_fast': sma_fast,
                    'sma_slow': sma_slow,
                    'crossover_type': 'golden'
                },
                timestamp=datetime.now()
            )

        elif sma_fast < sma_slow and prev_sma_fast >= prev_sma_slow:
            # Death cross - SELL signal
            conviction = min(75, abs(sma_fast - sma_slow) / sma_slow * 1000)
            return TradingSignal(
                symbol=symbol,
                strategy=self.name,
                action='SELL',
                price=current_price,
                conviction=conviction,
                reasoning=f"Death cross: {self.fast_period}-day SMA ({sma_fast:.4f}) crossed below {self.slow_period}-day SMA ({sma_slow:.4f})",
                indicators={
                    'sma_fast': sma_fast,
                    'sma_slow': sma_slow,
                    'crossover_type': 'death'
                },
                timestamp=datetime.now()
            )

        else:
            # HOLD signal
            return TradingSignal(
                symbol=symbol,
                strategy=self.name,
                action='HOLD',
                price=current_price,
                conviction=50,
                reasoning="No crossover detected",
                indicators={
                    'sma_fast': sma_fast,
                    'sma_slow': sma_slow,
                    'crossover_type': 'none'
                },
                timestamp=datetime.now()
            )

class RSIStrategy(BaseStrategy):
    """
    RSI Overbought/Oversold Strategy
    Uses 30/70 levels for entry/exit signals
    """

    def __init__(self, rsi_period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__("RSI_Mean_Reversion")
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Optional[TradingSignal]:
        if len(data) < self.rsi_period + 1:
            return None

        # Calculate RSI
        data['rsi'] = talib.RSI(data['close'], timeperiod=self.rsi_period)

        current_price = data['close'].iloc[-1]
        current_rsi = data['rsi'].iloc[-1]
        prev_rsi = data['rsi'].iloc[-2]

        # RSI signals
        if current_rsi < self.oversold and prev_rsi >= self.oversold:
            # Oversold - BUY signal
            conviction = min(80, (self.oversold - current_rsi) / 10 * 100)
            return TradingSignal(
                symbol=symbol,
                strategy=self.name,
                action='BUY',
                price=current_price,
                conviction=conviction,
                reasoning=f"RSI oversold at {current_rsi:.2f} (below {self.oversold})",
                indicators={
                    'rsi': current_rsi,
                    'oversold_level': self.oversold,
                    'condition': 'oversold'
                },
                timestamp=datetime.now()
            )

        elif current_rsi > self.overbought and prev_rsi <= self.overbought:
            # Overbought - SELL signal
            conviction = min(80, (current_rsi - self.overbought) / 10 * 100)
            return TradingSignal(
                symbol=symbol,
                strategy=self.name,
                action='SELL',
                price=current_price,
                conviction=conviction,
                reasoning=f"RSI overbought at {current_rsi:.2f} (above {self.overbought})",
                indicators={
                    'rsi': current_rsi,
                    'overbought_level': self.overbought,
                    'condition': 'overbought'
                },
                timestamp=datetime.now()
            )

        else:
            # HOLD signal
            action = 'HOLD'
            if current_rsi < 40:
                reasoning = f"RSI approaching oversold ({current_rsi:.2f})"
            elif current_rsi > 60:
                reasoning = f"RSI approaching overbought ({current_rsi:.2f})"
            else:
                reasoning = f"RSI neutral ({current_rsi:.2f})"

            return TradingSignal(
                symbol=symbol,
                strategy=self.name,
                action=action,
                price=current_price,
                conviction=50,
                reasoning=reasoning,
                indicators={
                    'rsi': current_rsi,
                    'condition': 'neutral'
                },
                timestamp=datetime.now()
            )

class MACDStrategy(BaseStrategy):
    """
    MACD Trend Following Strategy
    Uses MACD line crossover with signal line for trend signals
    """

    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__("MACD_Trend")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Optional[TradingSignal]:
        if len(data) < self.slow_period + self.signal_period:
            return None

        # Calculate MACD
        macd, macd_signal, macd_hist = talib.MACD(
            data['close'],
            fastperiod=self.fast_period,
            slowperiod=self.slow_period,
            signalperiod=self.signal_period
        )

        current_price = data['close'].iloc[-1]
        current_macd = macd[-1]
        current_signal = macd_signal[-1]
        current_hist = macd_hist[-1]
        prev_hist = macd_hist[-2]

        # MACD histogram signals
        if current_hist > 0 and prev_hist <= 0:
            # MACD crossed above signal - BUY signal
            conviction = min(85, abs(current_hist) * 1000)
            return TradingSignal(
                symbol=symbol,
                strategy=self.name,
                action='BUY',
                price=current_price,
                conviction=conviction,
                reasoning=f"MACD bullish crossover: Histogram turned positive ({current_hist:.6f})",
                indicators={
                    'macd': current_macd,
                    'signal': current_signal,
                    'histogram': current_hist,
                    'trend': 'bullish'
                },
                timestamp=datetime.now()
            )

        elif current_hist < 0 and prev_hist >= 0:
            # MACD crossed below signal - SELL signal
            conviction = min(85, abs(current_hist) * 1000)
            return TradingSignal(
                symbol=symbol,
                strategy=self.name,
                action='SELL',
                price=current_price,
                conviction=conviction,
                reasoning=f"MACD bearish crossover: Histogram turned negative ({current_hist:.6f})",
                indicators={
                    'macd': current_macd,
                    'signal': current_signal,
                    'histogram': current_hist,
                    'trend': 'bearish'
                },
                timestamp=datetime.now()
            )

        else:
            # HOLD signal with trend direction
            trend = 'bullish' if current_hist > 0 else 'bearish'
            conviction = 60 if abs(current_hist) > 0.0001 else 50

            return TradingSignal(
                symbol=symbol,
                strategy=self.name,
                action='HOLD',
                price=current_price,
                conviction=conviction,
                reasoning=f"MACD trend is {trend} (histogram: {current_hist:.6f})",
                indicators={
                    'macd': current_macd,
                    'signal': current_signal,
                    'histogram': current_hist,
                    'trend': trend
                },
                timestamp=datetime.now()
            )

class VolumeStrategy(BaseStrategy):
    """
    Volume-based Breakout Strategy
    Uses volume spikes with price breakouts for signals
    """

    def __init__(self, volume_period: int = 20, volume_multiplier: float = 1.5):
        super().__init__("Volume_Breakout")
        self.volume_period = volume_period
        self.volume_multiplier = volume_multiplier

    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Optional[TradingSignal]:
        if len(data) < self.volume_period + 1:
            return None

        # Calculate volume moving average
        data['volume_sma'] = talib.SMA(data['volume'], timeperiod=self.volume_period)
        data['price_change'] = data['close'].pct_change()

        current_price = data['close'].iloc[-1]
        current_volume = data['volume'].iloc[-1]
        avg_volume = data['volume_sma'].iloc[-1]
        price_change = data['price_change'].iloc[-1]

        # Volume spike detection
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        if volume_ratio >= self.volume_multiplier:
            if price_change > 0.02:  # 2% price increase
                conviction = min(90, volume_ratio * 30 + abs(price_change) * 1000)
                return TradingSignal(
                    symbol=symbol,
                    strategy=self.name,
                    action='BUY',
                    price=current_price,
                    conviction=conviction,
                    reasoning=f"Volume breakout: Volume {volume_ratio:.1f}x average, Price up {price_change*100:.2f}%",
                    indicators={
                        'volume_ratio': volume_ratio,
                        'price_change_pct': price_change * 100,
                        'breakout_type': 'bullish'
                    },
                    timestamp=datetime.now()
                )

            elif price_change < -0.02:  # 2% price decrease
                conviction = min(90, volume_ratio * 30 + abs(price_change) * 1000)
                return TradingSignal(
                    symbol=symbol,
                    strategy=self.name,
                    action='SELL',
                    price=current_price,
                    conviction=conviction,
                    reasoning=f"Volume breakdown: Volume {volume_ratio:.1f}x average, Price down {price_change*100:.2f}%",
                    indicators={
                        'volume_ratio': volume_ratio,
                        'price_change_pct': price_change * 100,
                        'breakout_type': 'bearish'
                    },
                    timestamp=datetime.now()
                )

        # HOLD signal
        return TradingSignal(
            symbol=symbol,
            strategy=self.name,
            action='HOLD',
            price=current_price,
            conviction=50,
            reasoning=f"No volume breakout detected (Volume ratio: {volume_ratio:.1f}x)",
            indicators={
                'volume_ratio': volume_ratio,
                'price_change_pct': price_change * 100,
                'breakout_type': 'none'
            },
            timestamp=datetime.now()
        )

class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands Mean Reversion Strategy
    Uses price interaction with Bollinger Bands for signals
    """

    def __init__(self, period: int = 20, std_dev: float = 2):
        super().__init__("Bollinger_Bands")
        self.period = period
        self.std_dev = std_dev

    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Optional[TradingSignal]:
        if len(data) < self.period:
            return None

        # Calculate Bollinger Bands
        upper_band, middle_band, lower_band = talib.BBANDS(
            data['close'],
            timeperiod=self.period,
            nbdevup=self.std_dev,
            nbdevdn=self.std_dev
        )

        current_price = data['close'].iloc[-1]
        current_upper = upper_band[-1]
        current_middle = middle_band[-1]
        current_lower = lower_band[-1]

        # Calculate position within bands
        bb_position = (current_price - current_lower) / (current_upper - current_lower)

        if current_price <= current_lower:
            # Price at or below lower band - BUY signal
            conviction = min(80, (current_lower - current_price) / current_lower * 1000 + 50)
            return TradingSignal(
                symbol=symbol,
                strategy=self.name,
                action='BUY',
                price=current_price,
                conviction=conviction,
                reasoning=f"Price at lower Bollinger Band (${current_price:.6f}) - Oversold condition",
                indicators={
                    'upper_band': current_upper,
                    'middle_band': current_middle,
                    'lower_band': current_lower,
                    'bb_position': bb_position,
                    'condition': 'oversold'
                },
                timestamp=datetime.now()
            )

        elif current_price >= current_upper:
            # Price at or above upper band - SELL signal
            conviction = min(80, (current_price - current_upper) / current_upper * 1000 + 50)
            return TradingSignal(
                symbol=symbol,
                strategy=self.name,
                action='SELL',
                price=current_price,
                conviction=conviction,
                reasoning=f"Price at upper Bollinger Band (${current_price:.6f}) - Overbought condition",
                indicators={
                    'upper_band': current_upper,
                    'middle_band': current_middle,
                    'lower_band': current_lower,
                    'bb_position': bb_position,
                    'condition': 'overbought'
                },
                timestamp=datetime.now()
            )

        else:
            # HOLD signal
            action = 'HOLD'
            if bb_position < 0.3:
                reasoning = f"Price in lower half of Bollinger Bands (position: {bb_position:.2f})"
            elif bb_position > 0.7:
                reasoning = f"Price in upper half of Bollinger Bands (position: {bb_position:.2f})"
            else:
                reasoning = f"Price in middle of Bollinger Bands (position: {bb_position:.2f})"

            return TradingSignal(
                symbol=symbol,
                strategy=self.name,
                action=action,
                price=current_price,
                conviction=50,
                reasoning=reasoning,
                indicators={
                    'upper_band': current_upper,
                    'middle_band': current_middle,
                    'lower_band': current_lower,
                    'bb_position': bb_position,
                    'condition': 'neutral'
                },
                timestamp=datetime.now()
            )

class HybridStrategy(BaseStrategy):
    """
    Hybrid Strategy combining multiple indicators
    Uses voting system with AI override capability
    """

    def __init__(self):
        super().__init__("Hybrid_Multi_Strategy")
        self.strategies = [
            SMAStrategy(),
            RSIStrategy(),
            MACDStrategy(),
            BollingerBandsStrategy(),
            VolumeStrategy()
        ]
        self.strategy_weights = {
            'SMA_Crossover': 0.2,
            'RSI_Mean_Reversion': 0.25,
            'MACD_Trend': 0.25,
            'Bollinger_Bands': 0.15,
            'Volume_Breakout': 0.15
        }

    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Optional[TradingSignal]:
        signals = []
        current_price = data['close'].iloc[-1]

        # Generate signals from all strategies
        for strategy in self.strategies:
            signal = strategy.generate_signal(symbol, data)
            if signal:
                signals.append(signal)

        if not signals:
            return None

        # Count votes for each action
        buy_votes = sum(self.strategy_weights.get(s.strategy, 0.1) for s in signals if s.action == 'BUY')
        sell_votes = sum(self.strategy_weights.get(s.strategy, 0.1) for s in signals if s.action == 'SELL')
        hold_votes = sum(self.strategy_weights.get(s.strategy, 0.1) for s in signals if s.action == 'HOLD')

        # Determine action based on weighted voting
        max_votes = max(buy_votes, sell_votes, hold_votes)

        if max_votes == hold_votes or (buy_votes == sell_votes):
            action = 'HOLD'
            conviction = 50
            reasoning = "Neutral: Mixed signals from strategies"
        elif max_votes == buy_votes:
            action = 'BUY'
            conviction = min(95, 50 + (buy_votes / max(buy_votes + sell_votes, 0.1)) * 45)
            reasoning = f"Bullish: {buy_votes:.2f} vs {sell_votes:.2f} weighted votes"
        else:
            action = 'SELL'
            conviction = min(95, 50 + (sell_votes / max(buy_votes + sell_votes, 0.1)) * 45)
            reasoning = f"Bearish: {sell_votes:.2f} vs {buy_votes:.2f} weighted votes"

        # Combine indicators from all strategies
        combined_indicators = {}
        for signal in signals:
            combined_indicators.update(signal.indicators)

        return TradingSignal(
            symbol=symbol,
            strategy=self.name,
            action=action,
            price=current_price,
            conviction=conviction,
            reasoning=reasoning,
            indicators=combined_indicators,
            timestamp=datetime.now()
        )

class StrategyManager:
    """
    Strategy Manager for handling multiple trading strategies
    """

    def __init__(self):
        self.strategies = {
            'sma': SMAStrategy(),
            'rsi': RSIStrategy(),
            'macd': MACDStrategy(),
            'bollinger': BollingerBandsStrategy(),
            'volume': VolumeStrategy(),
            'hybrid': HybridStrategy()
        }
        self.active_strategy = 'hybrid'
        self.signal_history = []

    def set_strategy(self, strategy_name: str) -> bool:
        """Set the active trading strategy"""
        if strategy_name in self.strategies:
            self.active_strategy = strategy_name
            print(f"✅ Strategy set to: {strategy_name}")
            return True
        else:
            print(f"❌ Strategy '{strategy_name}' not found. Available: {list(self.strategies.keys())}")
            return False

    def get_signal(self, symbol: str, data: pd.DataFrame) -> Optional[TradingSignal]:
        """Get trading signal from active strategy"""
        if self.active_strategy not in self.strategies:
            return None

        strategy = self.strategies[self.active_strategy]
        signal = strategy.generate_signal(symbol, data)

        if signal:
            self.signal_history.append(signal)
            # Keep only last 1000 signals
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-1000:]

        return signal

    def get_all_signals(self, symbol: str, data: pd.DataFrame) -> Dict[str, TradingSignal]:
        """Get signals from all strategies for comparison"""
        signals = {}

        for name, strategy in self.strategies.items():
            signal = strategy.generate_signal(symbol, data)
            if signal:
                signals[name] = signal

        return signals

    def get_strategy_performance(self) -> Dict:
        """Analyze performance of different strategies"""
        strategy_stats = {}

        for strategy_name in self.strategies.keys():
            strategy_signals = [s for s in self.signal_history if s.strategy == strategy_name]

            if strategy_signals:
                buy_signals = [s for s in strategy_signals if s.action == 'BUY']
                sell_signals = [s for s in strategy_signals if s.action == 'SELL']
                hold_signals = [s for s in strategy_signals if s.action == 'HOLD']

                avg_conviction = np.mean([s.conviction for s in strategy_signals])
                avg_conviction_buy = np.mean([s.conviction for s in buy_signals]) if buy_signals else 0
                avg_conviction_sell = np.mean([s.conviction for s in sell_signals]) if sell_signals else 0

                strategy_stats[strategy_name] = {
                    'total_signals': len(strategy_signals),
                    'buy_signals': len(buy_signals),
                    'sell_signals': len(sell_signals),
                    'hold_signals': len(hold_signals),
                    'avg_conviction': avg_conviction,
                    'avg_conviction_buy': avg_conviction_buy,
                    'avg_conviction_sell': avg_conviction_sell
                }
            else:
                strategy_stats[strategy_name] = {
                    'total_signals': 0,
                    'buy_signals': 0,
                    'sell_signals': 0,
                    'hold_signals': 0,
                    'avg_conviction': 0,
                    'avg_conviction_buy': 0,
                    'avg_conviction_sell': 0
                }

        return strategy_stats

if __name__ == "__main__":
    # Example usage with sample data
    manager = StrategyManager()

    # Create sample price data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.02)
    volumes = 1000000 + np.random.randn(100) * 100000

    sample_data = pd.DataFrame({
        'close': prices,
        'volume': volumes,
        'high': prices * (1 + np.abs(np.random.randn(100) * 0.01)),
        'low': prices * (1 - np.abs(np.random.randn(100) * 0.01)),
        'open': np.roll(prices, 1)
    }, index=dates)

    # Test all strategies
    print("🧪 Testing trading strategies...")

    symbol = 'BTC'
    all_signals = manager.get_all_signals(symbol, sample_data)

    for strategy_name, signal in all_signals.items():
        print(f"\n📊 {strategy_name} Strategy:")
        print(f"   Action: {signal.action}")
        print(f"   Conviction: {signal.conviction:.1f}%")
        print(f"   Reasoning: {signal.reasoning}")

    print(f"\n✅ Active strategy: {manager.active_strategy}")
    print(f"📈 Strategy performance: {manager.get_strategy_performance()}")