import os
import time
import sys
import argparse
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Import our custom modules
from paper_trading import PaperTradingPortfolio
from strategies import StrategyManager, TradingSignal
from ai_analyzer import AIAnalyzer
from data_sources import MarketDataManager
from risk_manager import RiskManager, RiskParameters
from portfolio_optimizer import PortfolioOptimizer
from delta_integration import DeltaIntegration
from backtester import Backtester

# Load environment variables
load_dotenv()

class EnhancedAITradingBot:
    """
    Enhanced AI Trading Bot with paper trading, multiple strategies, and risk management
    Designed for ₹1000 initial investment with 95%+ conviction requirements
    """

    def __init__(self, config):
        self.config = config
        self.trading_mode = config['trading_mode']  # 'paper' or 'live'
        self.strategy_name = config['strategy']
        self.symbols = config['symbols']
        self.initial_capital = config['initial_capital']
        self.conviction_threshold = config['conviction_threshold']
        self.check_interval = config['check_interval']

        # Initialize components
        self.data_manager = MarketDataManager()
        self.strategy_manager = StrategyManager()
        self.risk_manager = RiskManager(self.initial_capital)
        self.portfolio = PaperTradingPortfolio(self.initial_capital)

        # AI Analyzer (only initialize if API key available)
        if os.getenv('GEMINI_API_KEY'):
            self.ai_analyzer = AIAnalyzer(os.getenv('GEMINI_API_KEY'))
        else:
            self.ai_analyzer = None
            print("⚠️ GEMINI_API_KEY not found - AI features disabled")

        # Delta Exchange integration (only for live trading)
        if self.trading_mode == 'live':
            if os.getenv('DELTA_WEBHOOK_URL'):
                self.delta_integration = DeltaIntegration(os.getenv('DELTA_WEBHOOK_URL'))
            else:
                print("❌ DELTA_WEBHOOK_URL required for live trading")
                sys.exit(1)
        else:
            self.delta_integration = None

        # Set strategy
        self.strategy_manager.set_strategy(self.strategy_name)

        # Performance tracking
        self.start_time = datetime.now()
        self.last_analysis_time = {}
        self.performance_log = []

        print(f"🤖 Enhanced AI Trading Bot Started")
        print(f"📊 Mode: {self.trading_mode.upper()}")
        print(f"🎯 Strategy: {self.strategy_name}")
        print(f"💰 Initial Capital: ₹{self.initial_capital}")
        print(f"🎯 Symbols: {', '.join(self.symbols)}")
        print(f"📈 Conviction Threshold: {self.conviction_threshold}%")
        print(f"⏰ Check Interval: {self.check_interval} seconds")
        print("="*60)

    def get_market_data(self, symbol: str):
        """Get current market data for symbol"""
        try:
            price = self.data_manager.get_current_price(symbol)
            if price is None:
                print(f"❌ Failed to fetch price for {symbol}")
                return None

            # Get historical data for technical analysis
            historical_data = self.data_manager.get_historical_data(symbol, period='30d', interval='1h')
            if historical_data is None or historical_data.empty:
                print(f"❌ Failed to fetch historical data for {symbol}")
                return None

            return price, historical_data

        except Exception as e:
            print(f"❌ Error getting market data for {symbol}: {e}")
            return None

    def generate_trading_signal(self, symbol: str, price: float, data):
        """Generate trading signal using configured strategy"""
        try:
            # Get signal from strategy
            signal = self.strategy_manager.get_signal(symbol, data)

            # Enhance with AI analysis if available
            if self.ai_analyzer and len(data) >= 50:
                ai_signal = self.ai_analyzer.analyze_symbol(symbol, data)
                if ai_signal:
                    # Combine strategy and AI signals
                    signal = self._combine_signals(signal, ai_signal)

            return signal

        except Exception as e:
            print(f"❌ Error generating signal for {symbol}: {e}")
            return None

    def _combine_signals(self, strategy_signal: TradingSignal, ai_signal):
        """Combine strategy and AI signals with conviction weighting"""
        # Weighted average of convictions
        combined_conviction = (strategy_signal.conviction * 0.4) + (ai_signal.conviction * 0.6)

        # Determine action based on higher conviction signal
        if ai_signal.conviction > strategy_signal.conviction:
            action = ai_signal.action
            reasoning = f"AI: {ai_signal.reasoning[:100]}..."
        else:
            action = strategy_signal.action
            reasoning = f"Strategy: {strategy_signal.reasoning[:100]}..."

        # Create enhanced signal
        combined_signal = TradingSignal(
            symbol=strategy_signal.symbol,
            strategy=f"Hybrid_{strategy_signal.strategy}_AI",
            action=action,
            price=strategy_signal.price,
            conviction=combined_conviction,
            reasoning=reasoning,
            indicators={**strategy_signal.indicators, **ai_signal.technical_analysis},
            timestamp=datetime.now()
        )

        return combined_signal

    def execute_trade(self, signal: TradingSignal):
        """Execute trade based on signal and risk management"""
        try:
            # Check conviction threshold for live trading
            if self.trading_mode == 'live' and signal.conviction < self.conviction_threshold:
                print(f"⏸️ {signal.symbol}: Conviction {signal.conviction:.1f}% below threshold ({self.conviction_threshold}%)")
                return False

            # Risk management check
            can_trade, reason = self.risk_manager.can_open_position(
                signal.symbol,
                signal.action,
                signal.conviction,
                self.portfolio.current_capital
            )

            if not can_trade:
                print(f"🛑 {signal.symbol}: {reason}")
                return False

            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                signal.symbol,
                signal.price,
                signal.price * 0.98,  # Assuming 2% stop loss
                signal.conviction,
                self.portfolio.current_capital
            )

            if position_size <= 0:
                print(f"⚠️ {signal.symbol}: Position size too small")
                return False

            # Execute trade
            if self.trading_mode == 'paper':
                success = self.portfolio.open_position(
                    signal.symbol,
                    signal.action,
                    signal.price,
                    signal.conviction,
                    signal.strategy
                )
            else:  # live trading
                success = self.delta_integration.send_order(
                    symbol=signal.symbol,
                    action=signal.action,
                    quantity=position_size,
                    price=signal.price
                )
                if success:
                    # Track in portfolio for live trading too
                    self.portfolio.open_position(
                        signal.symbol,
                        signal.action,
                        signal.price,
                        signal.conviction,
                        signal.strategy
                    )

            if success:
                print(f"✅ {signal.symbol}: {signal.action} {position_size:.6f} at ${signal.price:.6f}")
                print(f"   Conviction: {signal.conviction:.1f}% | Reasoning: {signal.reasoning[:50]}...")
                return True
            else:
                print(f"❌ {signal.symbol}: Failed to execute {signal.action}")
                return False

        except Exception as e:
            print(f"❌ Error executing trade for {signal.symbol}: {e}")
            return False

    def check_exit_conditions(self):
        """Check for stop loss, take profit, and other exit conditions"""
        try:
            # Get current prices for all positions
            current_prices = {}
            for symbol in self.portfolio.positions.keys():
                price_data = self.get_market_data(symbol)
                if price_data:
                    current_prices[symbol] = price_data[0]

            # Check stop loss and take profit
            symbols_to_close = self.portfolio.check_stop_loss_take_profit(current_prices)

            for symbol in symbols_to_close:
                current_price = current_prices.get(symbol, 0)
                if current_price > 0:
                    if self.trading_mode == 'paper':
                        self.portfolio.close_position(symbol, current_price, "Stop Loss/Take Profit")
                    else:
                        # Send exit order to Delta Exchange
                        success = self.delta_integration.close_position(symbol, current_price)
                        if success:
                            self.portfolio.close_position(symbol, current_price, "Stop Loss/Take Profit")

        except Exception as e:
            print(f"❌ Error checking exit conditions: {e}")

    def log_performance(self):
        """Log performance metrics"""
        try:
            summary = self.portfolio.get_portfolio_summary()

            performance_entry = {
                'timestamp': datetime.now().isoformat(),
                'portfolio_value': summary['total_value'],
                'available_capital': summary['available_capital'],
                'active_positions': summary['active_positions'],
                'total_return_pct': summary['total_return_pct'],
                'daily_pnl': summary['daily_pnl'],
                'max_drawdown_pct': summary['max_drawdown_pct'],
                'win_rate_pct': summary['win_rate_pct'],
                'total_trades': summary['total_trades']
            }

            self.performance_log.append(performance_entry)

            # Keep only last 1000 entries
            if len(self.performance_log) > 1000:
                self.performance_log = self.performance_log[-1000:]

        except Exception as e:
            print(f"❌ Error logging performance: {e}")

    def display_status(self):
        """Display current bot status"""
        try:
            summary = self.portfolio.get_portfolio_summary()
            runtime = datetime.now() - self.start_time

            print(f"\n📊 Bot Status - Runtime: {runtime}")
            print(f"💰 Portfolio Value: ₹{summary['total_value']:.2f}")
            print(f"💵 Available Capital: ₹{summary['available_capital']:.2f}")
            print(f"📈 Total Return: {summary['total_return_pct']:+.2f}%")
            print(f"📉 Max Drawdown: {summary['max_drawdown_pct']:.2f}%")
            print(f"🎯 Win Rate: {summary['win_rate_pct']:.1f}%")
            print(f"📊 Active Positions: {summary['active_positions']}")
            print(f"📈 Total Trades: {summary['total_trades']}")

            if summary['active_positions'] > 0:
                print("\n📍 Active Positions:")
                for symbol, position in summary['positions'].items():
                    print(f"   {symbol}: {position['side']} {position['quantity']:.6f} @ ${position['entry_price']:.6f}")

        except Exception as e:
            print(f"❌ Error displaying status: {e}")

    def run_analysis_cycle(self):
        """Run one complete analysis cycle"""
        try:
            print(f"\n⏰ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Analysis Cycle Started")

            # Reset daily limits
            self.portfolio.reset_daily_limits()

            # Analyze each symbol
            signals_generated = 0
            trades_executed = 0

            for symbol in self.symbols:
                # Rate limiting per symbol (analyze at most once per cycle)
                last_time = self.last_analysis_time.get(symbol, datetime.min)
                if datetime.now() - last_time < timedelta(seconds=self.check_interval / len(self.symbols)):
                    continue

                print(f"\n🔍 Analyzing {symbol}...")

                # Get market data
                market_data = self.get_market_data(symbol)
                if not market_data:
                    continue

                price, historical_data = market_data
                print(f"💰 {symbol} Price: ${price:.6f}")

                # Generate trading signal
                signal = self.generate_trading_signal(symbol, price, historical_data)
                if signal:
                    signals_generated += 1
                    print(f"🎯 {signal.action} signal (conviction: {signal.conviction:.1f}%)")

                    # Execute trade if meets criteria
                    if signal.action in ['BUY', 'SELL']:
                        if self.execute_trade(signal):
                            trades_executed += 1

                self.last_analysis_time[symbol] = datetime.now()

            # Check exit conditions
            self.check_exit_conditions()

            # Log performance
            self.log_performance()

            # Display status every hour
            if len(self.performance_log) % 12 == 0:  # Assuming 5-minute intervals
                self.display_status()

            print(f"\n✅ Cycle Complete: {signals_generated} signals, {trades_executed} trades")
            print(f"⏳ Next cycle in {self.check_interval} seconds")

        except Exception as e:
            print(f"❌ Error in analysis cycle: {e}")
            time.sleep(60)  # Wait 1 minute on error

    def run_backtest(self):
        """Run historical backtesting before live trading"""
        print("🧪 Running Historical Backtesting...")

        try:
            backtester = Backtester(self.initial_capital)

            # Run backtest for selected strategy
            results = backtester.run_comprehensive_backtest(self.symbols, [self.strategy_name])

            if results:
                # Show best result
                best_key = max(results.keys(), key=lambda k: results[k].total_return_pct)
                best_result = results[best_key]

                print(f"\n📈 Best Backtest Result ({best_key}):")
                print(f"   Total Return: {best_result.total_return_pct:+.2f}%")
                print(f"   Win Rate: {best_result.win_rate_pct:.1f}%")
                print(f"   Sharpe Ratio: {best_result.sharpe_ratio:.2f}")
                print(f"   Max Drawdown: {best_result.max_drawdown_pct:.2f}%")
                print(f"   Total Trades: {best_result.total_trades}")

                # Ask user if they want to proceed
                if best_result.total_return_pct < 0:
                    response = input("\n⚠️ Backtest shows negative returns. Continue? (y/N): ")
                    if response.lower() != 'y':
                        print("🛑 Trading aborted based on backtest results")
                        return False
                elif best_result.win_rate_pct < 50:
                    response = input("\n⚠️ Backtest shows low win rate. Continue? (y/N): ")
                    if response.lower() != 'y':
                        print("🛑 Trading aborted based on backtest results")
                        return False

            else:
                print("⚠️ No backtest results available")

            return True

        except Exception as e:
            print(f"❌ Error running backtest: {e}")
            return False

    def run(self):
        """Main bot execution"""
        try:
            # Run backtest first for paper trading
            if self.trading_mode == 'paper':
                if not self.run_backtest():
                    return

                print("\n🚀 Starting Paper Trading...")
                input("Press Enter to begin paper trading...")

            # Main trading loop
            while True:
                try:
                    self.run_analysis_cycle()
                    time.sleep(self.check_interval)

                except KeyboardInterrupt:
                    print("\n🛑 Bot stopped by user")
                    self.display_final_summary()
                    break
                except Exception as e:
                    print(f"❌ Unexpected error: {e}")
                    time.sleep(60)  # Wait 1 minute before retrying

        except Exception as e:
            print(f"❌ Fatal error: {e}")

    def display_final_summary(self):
        """Display final trading summary"""
        try:
            summary = self.portfolio.get_portfolio_summary()
            runtime = datetime.now() - self.start_time

            print("\n" + "="*60)
            print("📊 FINAL TRADING SUMMARY")
            print("="*60)
            print(f"⏰ Total Runtime: {runtime}")
            print(f"💰 Final Portfolio Value: ₹{summary['total_value']:.2f}")
            print(f"📈 Total Return: {summary['total_return_pct']:+.2f}%")
            print(f"💰 Profit/Loss: ₹{summary['total_value'] - self.initial_capital:+.2f}")
            print(f"📉 Maximum Drawdown: {summary['max_drawdown_pct']:.2f}%")
            print(f"🎯 Win Rate: {summary['win_rate_pct']:.1f}%")
            print(f"📊 Total Trades: {summary['total_trades']}")
            print(f"📈 Profit Factor: {summary['profit_factor']:.2f}")
            print(f"📊 Sharpe Ratio: {summary.get('sharpe_ratio', 0):.2f}")

            # Save final portfolio state
            self.portfolio.save_to_file()

        except Exception as e:
            print(f"❌ Error displaying final summary: {e}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced AI Trading Bot')

    parser.add_argument('--mode', choices=['paper', 'live'], default='paper',
                       help='Trading mode (default: paper)')
    parser.add_argument('--strategy', choices=['sma', 'rsi', 'macd', 'bollinger', 'volume', 'hybrid'],
                       default='hybrid', help='Trading strategy (default: hybrid)')
    parser.add_argument('--symbols', nargs='+', default=['BTC', 'ETH', 'ADA', 'DOGE', 'SHIB'],
                       help='Symbols to trade (default: BTC ETH ADA DOGE SHIB)')
    parser.add_argument('--capital', type=float, default=1000,
                       help='Initial capital in INR (default: 1000)')
    parser.add_argument('--conviction', type=float, default=95,
                       help='Conviction threshold for live trading (default: 95)')
    parser.add_argument('--interval', type=int, default=300,
                       help='Check interval in seconds (default: 300)')

    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()

    # Create configuration
    config = {
        'trading_mode': args.mode,
        'strategy': args.strategy,
        'symbols': args.symbols,
        'initial_capital': args.capital,
        'conviction_threshold': args.conviction,
        'check_interval': args.interval
    }

    # Verify required environment variables
    if args.mode == 'live':
        if not os.getenv('GEMINI_API_KEY'):
            print("❌ ERROR: GEMINI_API_KEY environment variable required for live trading!")
            sys.exit(1)
        if not os.getenv('DELTA_WEBHOOK_URL'):
            print("❌ ERROR: DELTA_WEBHOOK_URL environment variable required for live trading!")
            sys.exit(1)

    # Create and run bot
    bot = EnhancedAITradingBot(config)
    bot.run()
