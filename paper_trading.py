import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from decimal import Decimal, ROUND_DOWN

@dataclass
class Position:
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    def current_pnl(self, current_price: float) -> float:
        """Calculate current P&L for position"""
        if self.side == 'BUY':
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity

    def pnl_percentage(self, current_price: float) -> float:
        """Calculate P&L as percentage"""
        if self.side == 'BUY':
            return ((current_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - current_price) / self.entry_price) * 100

@dataclass
class Trade:
    symbol: str
    side: str
    quantity: float
    entry_price: float
    exit_price: Optional[float]
    entry_time: datetime
    exit_time: Optional[datetime]
    pnl: Optional[float]
    fees: float
    strategy: str
    conviction_score: float

class PaperTradingPortfolio:
    """
    Paper trading portfolio management system
    Designed for ₹1000 initial investment with proper risk management
    """

    def __init__(self, initial_capital_inr: float = 1000):
        self.initial_capital = initial_capital_inr
        self.current_capital = initial_capital_inr
        self.available_capital = initial_capital_inr
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.fee_rate = 0.001  # 0.1% per trade
        self.max_positions = 3
        self.max_risk_per_trade = 0.05  # 5% of portfolio
        self.daily_loss_limit = 0.10  # 10% daily loss limit

        # Performance tracking
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_capital = initial_capital_inr
        self.last_reset_date = datetime.now().date()

        # Supported cryptocurrencies
        self.supported_symbols = ['BTC', 'ETH', 'ADA', 'DOGE', 'SHIB']

    def reset_daily_limits(self):
        """Reset daily P&L and loss limits if new day"""
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.daily_pnl = 0.0
            self.last_reset_date = current_date

    def calculate_position_size(self, symbol: str, price: float, conviction_score: float) -> float:
        """
        Calculate optimal position size based on risk management rules
        Returns quantity in units of the cryptocurrency
        """
        # Base position size on maximum risk per trade
        max_risk_amount = self.current_capital * self.max_risk_per_trade  # ₹50 max risk

        # Adjust position size based on conviction score
        if conviction_score >= 95:
            risk_multiplier = 1.0  # Full position
        elif conviction_score >= 80:
            risk_multiplier = 0.6  # 60% position
        elif conviction_score >= 60:
            risk_multiplier = 0.3  # 30% position (paper trading only)
        else:
            return 0.0  # No position for low conviction

        # Calculate position size with 2% stop loss
        stop_loss_distance = price * 0.02  # 2% stop loss
        position_value = min(max_risk_amount * risk_multiplier / 0.02, self.available_capital * 0.8)
        quantity = position_value / price

        return quantity

    def can_open_position(self, symbol: str, side: str, conviction_score: float) -> Tuple[bool, str]:
        """
        Check if we can open a new position based on risk management rules
        """
        # Check conviction threshold for real trading (paper trading allows 60%+)
        if conviction_score < 60:
            return False, f"Conviction score {conviction_score}% too low"

        # Check maximum concurrent positions
        if len(self.positions) >= self.max_positions:
            return False, f"Maximum {self.max_positions} positions already open"

        # Check if we already have position in this symbol
        if symbol in self.positions:
            return False, f"Already have position in {symbol}"

        # Check daily loss limit
        if self.daily_pnl <= -self.current_capital * self.daily_loss_limit:
            return False, f"Daily loss limit of {self.daily_loss_limit*100}% reached"

        # Check available capital
        if self.available_capital <= self.current_capital * 0.1:  # Keep 10% reserve
            return False, "Insufficient available capital"

        return True, "Position can be opened"

    def open_position(self, symbol: str, side: str, price: float, conviction_score: float,
                     strategy: str = "Unknown") -> bool:
        """
        Open a new position with proper risk management
        """
        # Check if position can be opened
        can_open, reason = self.can_open_position(symbol, side, conviction_score)
        if not can_open:
            print(f"❌ Cannot open {symbol} {side} position: {reason}")
            return False

        # Calculate position size
        quantity = self.calculate_position_size(symbol, price, conviction_score)
        if quantity <= 0:
            print(f"❌ Position size too small for {symbol}")
            return False

        # Calculate fees
        trade_value = quantity * price
        fees = trade_value * self.fee_rate
        total_cost = trade_value + fees

        # Check if we have enough capital
        if total_cost > self.available_capital:
            print(f"❌ Insufficient capital for {symbol} position")
            return False

        # Calculate stop loss and take profit
        if side == 'BUY':
            stop_loss = price * 0.98  # 2% below entry
            take_profit = price * 1.06  # 6% above entry (3:1 risk:reward)
        else:  # SELL
            stop_loss = price * 1.02  # 2% above entry
            take_profit = price * 0.94  # 6% below entry

        # Create position
        position = Position(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=price,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        # Update portfolio
        self.positions[symbol] = position
        self.available_capital -= total_cost

        # Create trade record
        trade = Trade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=price,
            exit_price=None,
            entry_time=position.entry_time,
            exit_time=None,
            pnl=None,
            fees=fees,
            strategy=strategy,
            conviction_score=conviction_score
        )
        self.trades.append(trade)

        print(f"✅ Opened {symbol} {side} position:")
        print(f"   Quantity: {quantity:.6f}")
        print(f"   Entry Price: ${price:.6f}")
        print(f"   Stop Loss: ${stop_loss:.6f}")
        print(f"   Take Profit: ${take_profit:.6f}")
        print(f"   Fees: ${fees:.4f}")
        print(f"   Conviction: {conviction_score}%")

        return True

    def close_position(self, symbol: str, exit_price: float, reason: str = "Manual") -> bool:
        """
        Close an existing position
        """
        if symbol not in self.positions:
            print(f"❌ No position found in {symbol}")
            return False

        position = self.positions[symbol]

        # Calculate P&L
        pnl = position.current_pnl(exit_price)
        exit_fees = position.quantity * exit_price * self.fee_rate
        net_pnl = pnl - exit_fees

        # Update capital
        if position.side == 'BUY':
            self.available_capital += (position.quantity * exit_price) - exit_fees
        else:  # SELL
            self.available_capital += (position.quantity * position.entry_price * 2) - (position.quantity * exit_price) - exit_fees

        # Update daily P&L
        self.daily_pnl += net_pnl

        # Update peak capital and drawdown
        self.current_capital += net_pnl
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital

        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        # Update trade record
        for trade in reversed(self.trades):
            if trade.symbol == symbol and trade.exit_price is None:
                trade.exit_price = exit_price
                trade.exit_time = datetime.now()
                trade.pnl = net_pnl
                break

        # Remove position
        del self.positions[symbol]

        print(f"✅ Closed {symbol} {position.side} position:")
        print(f"   Exit Price: ${exit_price:.6f}")
        print(f"   P&L: ${net_pnl:.4f} ({position.pnl_percentage(exit_price):+.2f}%)")
        print(f"   Reason: {reason}")

        return True

    def check_stop_loss_take_profit(self, current_prices: Dict[str, float]) -> List[str]:
        """
        Check if any positions hit stop loss or take profit levels
        Returns list of symbols to close
        """
        symbols_to_close = []

        for symbol, position in self.positions.items():
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]

            # Check stop loss
            if position.side == 'BUY' and current_price <= position.stop_loss:
                symbols_to_close.append(symbol)
                print(f"⚠️ Stop loss triggered for {symbol} at ${current_price:.6f}")
            elif position.side == 'SELL' and current_price >= position.stop_loss:
                symbols_to_close.append(symbol)
                print(f"⚠️ Stop loss triggered for {symbol} at ${current_price:.6f}")

            # Check take profit
            elif position.side == 'BUY' and current_price >= position.take_profit:
                symbols_to_close.append(symbol)
                print(f"🎯 Take profit triggered for {symbol} at ${current_price:.6f}")
            elif position.side == 'SELL' and current_price <= position.take_profit:
                symbols_to_close.append(symbol)
                print(f"🎯 Take profit triggered for {symbol} at ${current_price:.6f}")

        return symbols_to_close

    def get_portfolio_summary(self) -> Dict:
        """
        Get comprehensive portfolio summary
        """
        total_value = self.current_capital
        active_positions_value = 0

        # Calculate value of active positions (estimated)
        for symbol, position in self.positions.items():
            # This would need current price data for accurate calculation
            active_positions_value += position.quantity * position.entry_price

        total_value += active_positions_value

        # Calculate performance metrics
        total_return = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
        win_trades = [t for t in self.trades if t.pnl is not None and t.pnl > 0]
        lose_trades = [t for t in self.trades if t.pnl is not None and t.pnl <= 0]

        win_rate = len(win_trades) / len([t for t in self.trades if t.pnl is not None]) * 100 if self.trades else 0

        avg_win = np.mean([t.pnl for t in win_trades]) if win_trades else 0
        avg_loss = np.mean([t.pnl for t in lose_trades]) if lose_trades else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        return {
            'total_value': total_value,
            'available_capital': self.available_capital,
            'active_positions': len(self.positions),
            'total_trades': len(self.trades),
            'total_return_pct': total_return,
            'daily_pnl': self.daily_pnl,
            'max_drawdown_pct': self.max_drawdown * 100,
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor,
            'positions': {symbol: asdict(pos) for symbol, pos in self.positions.items()}
        }

    def save_to_file(self, filename: str = None):
        """
        Save portfolio state to file for persistence
        """
        if filename is None:
            filename = f"portfolio_{datetime.now().strftime('%Y%m%d')}.json"

        data = {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'available_capital': self.available_capital,
            'trades': [asdict(trade) for trade in self.trades],
            'daily_pnl': self.daily_pnl,
            'max_drawdown': self.max_drawdown,
            'peak_capital': self.peak_capital,
            'last_reset_date': self.last_reset_date.isoformat()
        }

        # Convert datetime objects for JSON serialization
        for trade in data['trades']:
            if trade['entry_time']:
                trade['entry_time'] = trade['entry_time'].isoformat()
            if trade['exit_time']:
                trade['exit_time'] = trade['exit_time'].isoformat()

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"💾 Portfolio saved to {filename}")

    def load_from_file(self, filename: str):
        """
        Load portfolio state from file
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            self.initial_capital = data['initial_capital']
            self.current_capital = data['current_capital']
            self.available_capital = data['available_capital']
            self.daily_pnl = data['daily_pnl']
            self.max_drawdown = data['max_drawdown']
            self.peak_capital = data['peak_capital']
            self.last_reset_date = datetime.fromisoformat(data['last_reset_date']).date()

            # Recreate trades
            self.trades = []
            for trade_data in data['trades']:
                trade = Trade(**trade_data)
                if trade.entry_time:
                    trade.entry_time = datetime.fromisoformat(trade.entry_time)
                if trade.exit_time:
                    trade.exit_time = datetime.fromisoformat(trade.exit_time)
                self.trades.append(trade)

            print(f"📂 Portfolio loaded from {filename}")

        except FileNotFoundError:
            print(f"❌ Portfolio file {filename} not found")
        except Exception as e:
            print(f"❌ Error loading portfolio: {e}")

if __name__ == "__main__":
    # Example usage
    portfolio = PaperTradingPortfolio(initial_capital_inr=1000)

    # Test opening a position
    print("🧪 Testing paper trading portfolio...")

    # Example: Open BTC position with high conviction
    portfolio.open_position(
        symbol='BTC',
        side='BUY',
        price=45000,
        conviction_score=95,
        strategy='AI_Hybrid'
    )

    # Check portfolio summary
    summary = portfolio.get_portfolio_summary()
    print(f"\n📊 Portfolio Summary:")
    for key, value in summary.items():
        if key != 'positions':
            print(f"{key}: {value}")

    # Test closing position
    portfolio.close_position('BTC', 46000, "Take profit")

    # Final summary
    final_summary = portfolio.get_portfolio_summary()
    print(f"\n📈 Final Portfolio Value: ${final_summary['total_value']:.2f}")
    print(f"📈 Total Return: {final_summary['total_return_pct']:+.2f}%")