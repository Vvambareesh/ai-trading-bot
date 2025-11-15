import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
from strategies import StrategyManager, TradingSignal
from paper_trading import PaperTradingPortfolio

@dataclass
class BacktestResult:
    strategy: str
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate_pct: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration_days: float
    monthly_returns: List[float]
    equity_curve: List[Tuple[str, float]]
    trades: List[Dict]

class Backtester:
    """
    Historical backtesting system for strategy optimization
    Tests strategies on 6 months of historical data
    """

    def __init__(self, initial_capital: float = 1000):
        self.initial_capital = initial_capital
        self.strategy_manager = StrategyManager()
        self.results = {}

    def fetch_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch historical data for backtesting
        In production, this would use real API data
        For now, generates realistic sample data
        """
        days = (end_date - start_date).days
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # Generate realistic price data with trend and volatility
        np.random.seed(42)  # For reproducible results

        # Base trend (slightly upward for crypto)
        trend = np.linspace(100, 120, len(dates))

        # Add volatility
        volatility = 0.02  # 2% daily volatility
        random_walk = np.cumsum(np.random.randn(len(dates)) * volatility)

        # Combine trend and volatility
        prices = trend * (1 + random_walk)

        # Generate OHLCV data
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            # Generate intraday variation
            high = close * (1 + abs(np.random.randn() * 0.01))
            low = close * (1 - abs(np.random.randn() * 0.01))
            open_price = data[-1]['close'] if i > 0 else close

            # Generate volume (higher on bigger price moves)
            volume_base = 1000000
            price_change_pct = abs(close - open_price) / open_price if open_price > 0 else 0
            volume = volume_base * (1 + price_change_pct * 10) * (0.5 + np.random.random())

            data.append({
                'date': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })

        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)

        return df

    def run_backtest(self, symbol: str, strategy_name: str,
                     start_date: datetime, end_date: datetime) -> BacktestResult:
        """
        Run backtest for a specific strategy on given symbol and date range
        """
        print(f"🔄 Backtesting {strategy_name} on {symbol} from {start_date.date()} to {end_date.date()}")

        # Fetch historical data
        data = self.fetch_historical_data(symbol, start_date, end_date)

        # Initialize portfolio for backtesting
        portfolio = PaperTradingPortfolio(initial_capital_inr=self.initial_capital)

        # Set strategy
        self.strategy_manager.set_strategy(strategy_name)

        # Track equity curve
        equity_curve = []
        trades_executed = []

        # Process data day by day
        for i in range(50, len(data)):  # Start from 50 to have enough data for indicators
            current_data = data.iloc[:i+1]
            current_price = current_data['close'].iloc[-1]
            current_date = current_data.index[-1]

            # Reset daily limits if new day
            portfolio.reset_daily_limits()

            # Get trading signal
            signal = self.strategy_manager.get_signal(symbol, current_data)

            if signal and signal.action in ['BUY', 'SELL']:
                # Check if we can open position
                can_open, reason = portfolio.can_open_position(symbol, signal.action, signal.conviction)
                if can_open:
                    # Open position
                    success = portfolio.open_position(
                        symbol=symbol,
                        side=signal.action,
                        price=current_price,
                        conviction_score=signal.conviction,
                        strategy=strategy_name
                    )

                    if success:
                        trades_executed.append({
                            'date': current_date.isoformat(),
                            'action': signal.action,
                            'price': current_price,
                            'conviction': signal.conviction,
                            'reasoning': signal.reasoning
                        })

            # Check existing positions for stop loss/take profit
            current_prices = {symbol: current_price}
            symbols_to_close = portfolio.check_stop_loss_take_profit(current_prices)

            for close_symbol in symbols_to_close:
                portfolio.close_position(close_symbol, current_price, "Stop Loss/Take Profit")

            # Update equity curve
            portfolio_value = portfolio.get_portfolio_summary()['total_value']
            equity_curve.append((current_date.isoformat(), portfolio_value))

        # Close any remaining positions at the end
        final_price = data['close'].iloc[-1]
        for symbol_pos in list(portfolio.positions.keys()):
            portfolio.close_position(symbol_pos, final_price, "End of backtest")

        # Calculate final metrics
        final_summary = portfolio.get_portfolio_summary()
        completed_trades = [t for t in portfolio.trades if t.pnl is not None]

        # Calculate additional metrics
        monthly_returns = self._calculate_monthly_returns(equity_curve)
        sharpe_ratio = self._calculate_sharpe_ratio(equity_curve)

        # Calculate trade duration
        durations = []
        for trade in completed_trades:
            if trade.entry_time and trade.exit_time:
                duration = (trade.exit_time - trade.entry_time).days
                durations.append(duration)
        avg_duration = np.mean(durations) if durations else 0

        # Create result object
        result = BacktestResult(
            strategy=strategy_name,
            symbol=symbol,
            start_date=start_date.date().isoformat(),
            end_date=end_date.date().isoformat(),
            initial_capital=self.initial_capital,
            final_capital=final_summary['total_value'],
            total_return_pct=final_summary['total_return_pct'],
            max_drawdown_pct=final_summary['max_drawdown_pct'],
            sharpe_ratio=sharpe_ratio,
            win_rate_pct=final_summary['win_rate_pct'],
            profit_factor=final_summary['profit_factor'],
            total_trades=len(completed_trades),
            winning_trades=len([t for t in completed_trades if t.pnl > 0]),
            losing_trades=len([t for t in completed_trades if t.pnl <= 0]),
            avg_win=final_summary.get('avg_win', 0),
            avg_loss=final_summary.get('avg_loss', 0),
            largest_win=max([t.pnl for t in completed_trades]) if completed_trades else 0,
            largest_loss=min([t.pnl for t in completed_trades]) if completed_trades else 0,
            avg_trade_duration_days=avg_duration,
            monthly_returns=monthly_returns,
            equity_curve=equity_curve,
            trades=trades_executed
        )

        print(f"✅ Backtest completed for {strategy_name}")
        print(f"   Total Return: {result.total_return_pct:+.2f}%")
        print(f"   Win Rate: {result.win_rate_pct:.1f}%")
        print(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")

        return result

    def run_comprehensive_backtest(self, symbols: List[str], strategies: List[str]) -> Dict[str, BacktestResult]:
        """
        Run backtest for all strategy-symbol combinations
        """
        # Set date range (6 months of data)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)

        all_results = {}

        for symbol in symbols:
            for strategy in strategies:
                key = f"{symbol}_{strategy}"
                print(f"\n📊 Testing {key}...")

                try:
                    result = self.run_backtest(symbol, strategy, start_date, end_date)
                    all_results[key] = result
                except Exception as e:
                    print(f"❌ Error backtesting {key}: {e}")

        self.results = all_results
        return all_results

    def compare_strategies(self, results: Dict[str, BacktestResult] = None) -> pd.DataFrame:
        """
        Compare strategies and return performance ranking
        """
        if results is None:
            results = self.results

        if not results:
            print("❌ No backtest results to compare")
            return pd.DataFrame()

        comparison_data = []

        for key, result in results.items():
            comparison_data.append({
                'Strategy_Symbol': key,
                'Strategy': result.strategy,
                'Symbol': result.symbol,
                'Total_Return_%': result.total_return_pct,
                'Sharpe_Ratio': result.sharpe_ratio,
                'Win_Rate_%': result.win_rate_pct,
                'Profit_Factor': result.profit_factor,
                'Max_Drawdown_%': result.max_drawdown_pct,
                'Total_Trades': result.total_trades,
                'Monthly_Volatility': np.std(result.monthly_returns) * 100 if result.monthly_returns else 0
            })

        df = pd.DataFrame(comparison_data)

        # Calculate composite score (weighted)
        weights = {
            'Total_Return_%': 0.3,
            'Sharpe_Ratio': 0.25,
            'Win_Rate_%': 0.2,
            'Profit_Factor': 0.15,
            'Max_Drawdown_%': -0.1  # Negative weight (lower drawdown is better)
        }

        # Normalize metrics to 0-1 scale
        for metric, weight in weights.items():
            if metric in df.columns:
                if metric == 'Max_Drawdown_%':
                    # For drawdown, lower is better
                    df[f'{metric}_norm'] = 1 - (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
                else:
                    # For other metrics, higher is better
                    df[f'{metric}_norm'] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
                df[f'{metric}_score'] = df[f'{metric}_norm'] * abs(weight)

        # Calculate composite score
        score_columns = [col for col in df.columns if col.endswith('_score')]
        df['Composite_Score'] = df[score_columns].sum(axis=1)

        # Sort by composite score
        df = df.sort_values('Composite_Score', ascending=False)

        return df[['Strategy_Symbol', 'Strategy', 'Symbol', 'Composite_Score'] +
                  [col for col in df.columns if col not in ['Strategy_Symbol', 'Strategy', 'Symbol', 'Composite_Score']]]

    def optimize_parameters(self, symbol: str, strategy_name: str) -> Dict:
        """
        Optimize strategy parameters using walk-forward analysis
        """
        print(f"🔧 Optimizing parameters for {strategy_name} on {symbol}")

        # Define parameter grids for each strategy
        param_grids = {
            'sma': {'fast_period': [5, 10, 15], 'slow_period': [20, 25, 30]},
            'rsi': {'rsi_period': [10, 14, 20], 'oversold': [25, 30, 35], 'overbought': [65, 70, 75]},
            'macd': {'fast_period': [8, 12, 16], 'slow_period': [20, 26, 30], 'signal_period': [6, 9, 12]},
            'bollinger': {'period': [15, 20, 25], 'std_dev': [1.5, 2.0, 2.5]},
            'volume': {'volume_period': [15, 20, 25], 'volume_multiplier': [1.2, 1.5, 1.8]}
        }

        if strategy_name not in param_grids:
            print(f"❌ No parameter grid defined for {strategy_name}")
            return {}

        best_params = {}
        best_score = -np.inf
        optimization_results = []

        # Set date ranges for walk-forward analysis
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)

        # Get all parameter combinations
        import itertools
        param_names = list(param_grids[strategy_name].keys())
        param_values = list(param_grids[strategy_name].values())
        param_combinations = list(itertools.product(*param_values))

        for i, param_combo in enumerate(param_combinations):
            params = dict(zip(param_names, param_combo))

            try:
                # Create temporary strategy with these parameters
                temp_strategy_manager = StrategyManager()

                # Update strategy parameters (this would need strategy-specific implementation)
                # For now, we'll use default parameters

                # Run backtest with these parameters
                result = self.run_backtest(symbol, strategy_name, start_date, end_date)

                # Calculate score (same as comparison)
                score = (result.total_return_pct * 0.3 +
                        result.sharpe_ratio * 25 +
                        result.win_rate_pct * 0.2 +
                        result.profit_factor * 15 -
                        result.max_drawdown_pct * 0.1)

                optimization_results.append({
                    'params': params,
                    'score': score,
                    'return_pct': result.total_return_pct,
                    'sharpe': result.sharpe_ratio,
                    'win_rate': result.win_rate_pct,
                    'max_drawdown': result.max_drawdown_pct
                })

                if score > best_score:
                    best_score = score
                    best_params = params

                print(f"   Test {i+1}/{len(param_combinations)}: Score {score:.2f}")

            except Exception as e:
                print(f"   Error with params {params}: {e}")

        # Sort results by score
        optimization_results.sort(key=lambda x: x['score'], reverse=True)

        return {
            'best_params': best_params,
            'best_score': best_score,
            'top_5_results': optimization_results[:5],
            'all_results': optimization_results
        }

    def _calculate_monthly_returns(self, equity_curve: List[Tuple[str, float]]) -> List[float]:
        """Calculate monthly returns from equity curve"""
        if not equity_curve:
            return []

        df = pd.DataFrame(equity_curve, columns=['date', 'value'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # Resample to monthly
        monthly_values = df.resample('M')['value'].last()

        if len(monthly_values) < 2:
            return []

        returns = monthly_values.pct_change().dropna()
        return returns.tolist()

    def _calculate_sharpe_ratio(self, equity_curve: List[Tuple[str, float]], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio from equity curve"""
        if len(equity_curve) < 2:
            return 0

        df = pd.DataFrame(equity_curve, columns=['date', 'value'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # Calculate daily returns
        daily_returns = df['value'].pct_change().dropna()

        if len(daily_returns) == 0 or daily_returns.std() == 0:
            return 0

        # Annualized Sharpe ratio
        excess_returns = daily_returns - risk_free_rate / 252  # Daily risk-free rate
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)

        return sharpe

    def save_results(self, filename: str = None):
        """Save backtest results to file"""
        if filename is None:
            filename = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Convert results to JSON-serializable format
        serializable_results = {}
        for key, result in self.results.items():
            serializable_results[key] = asdict(result)

        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"💾 Backtest results saved to {filename}")

    def load_results(self, filename: str):
        """Load backtest results from file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            self.results = {}
            for key, result_data in data.items():
                self.results[key] = BacktestResult(**result_data)

            print(f"📂 Backtest results loaded from {filename}")

        except FileNotFoundError:
            print(f"❌ Backtest file {filename} not found")
        except Exception as e:
            print(f"❌ Error loading backtest results: {e}")

if __name__ == "__main__":
    # Example usage
    print("🚀 Starting comprehensive backtesting...")

    backtester = Backtester(initial_capital=1000)

    # Test symbols and strategies
    symbols = ['BTC', 'ETH', 'ADA', 'DOGE', 'SHIB']
    strategies = ['sma', 'rsi', 'macd', 'bollinger', 'volume', 'hybrid']

    # Run comprehensive backtest
    results = backtester.run_comprehensive_backtest(symbols, strategies)

    # Compare strategies
    comparison = backtester.compare_strategies(results)
    print("\n📊 Strategy Comparison (Top 10):")
    print(comparison.head(10).to_string())

    # Save results
    backtester.save_results()

    # Optimize parameters for best performing strategy
    if not comparison.empty:
        best_strategy_symbol = comparison.iloc[0]['Strategy_Symbol']
        best_strategy = comparison.iloc[0]['Strategy']
        symbol = comparison.iloc[0]['Symbol']

        print(f"\n🔧 Optimizing parameters for best strategy: {best_strategy}")
        optimization = backtester.optimize_parameters(symbol, best_strategy)

        if optimization:
            print(f"Best parameters: {optimization['best_params']}")
            print(f"Best score: {optimization['best_score']:.2f}")