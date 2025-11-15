import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

@dataclass
class RiskParameters:
    max_portfolio_risk: float = 0.05  # 5% max portfolio risk
    max_position_size: float = 0.20   # 20% max position size
    max_concurrent_positions: int = 3
    stop_loss_pct: float = 0.02       # 2% stop loss
    take_profit_pct: float = 0.06     # 6% take profit (3:1 ratio)
    daily_loss_limit_pct: float = 0.10 # 10% daily loss limit
    correlation_threshold: float = 0.7  # Max correlation between positions
    volatility_lookback: int = 20      # 20-day volatility window
    max_drawdown_limit: float = 0.25   # 25% max drawdown limit
    position_sizing_method: str = 'kelly'  # 'kelly', 'fixed', 'volatility'

@dataclass
class RiskMetrics:
    current_portfolio_risk: float
    max_drawdown: float
    value_at_risk_95: float  # VaR at 95% confidence
    expected_shortfall: float
    sharpe_ratio: float
    volatility: float
    beta: float
    correlation_matrix: Dict
    concentration_risk: float

class RiskManager:
    """
    Advanced risk management system for cryptocurrency trading
    Implements position sizing, portfolio risk control, and real-time monitoring
    """

    def __init__(self, initial_capital: float = 1000, params: RiskParameters = None):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.params = params or RiskParameters()

        # Risk tracking
        self.daily_pnl = 0.0
        self.peak_capital = initial_capital
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0

        # Position tracking
        self.active_positions = {}
        self.position_history = []
        self.daily_returns = []
        self.portfolio_values = []

        # Risk limits tracking
        self.daily_trades_count = 0
        self.last_reset_date = datetime.now().date()

        # Correlation data cache
        self.correlation_cache = {}
        self.volatility_cache = {}

    def reset_daily_limits(self):
        """Reset daily risk limits if new day"""
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.daily_pnl = 0.0
            self.daily_trades_count = 0
            self.last_reset_date = current_date

    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss_price: float,
                             conviction_score: float, portfolio_value: float,
                             volatility: float = None) -> float:
        """
        Calculate optimal position size using multiple methods
        """
        # Base position size from conviction score
        conviction_multiplier = self._get_conviction_multiplier(conviction_score)

        # Calculate risk amount
        risk_per_share = abs(entry_price - stop_loss_price)
        if risk_per_share <= 0:
            return 0.0

        # Maximum risk per trade
        max_risk_amount = portfolio_value * self.params.max_portfolio_risk

        if self.params.position_sizing_method == 'kelly':
            position_value = self._kelly_criterion_size(
                portfolio_value, risk_per_share, entry_price, conviction_score
            )
        elif self.params.position_sizing_method == 'volatility':
            position_value = self._volatility_adjusted_size(
                portfolio_value, volatility or 0.02, conviction_score
            )
        else:  # fixed
            position_value = portfolio_value * 0.1 * conviction_multiplier

        # Apply position limits
        position_value = min(position_value, portfolio_value * self.params.max_position_size)
        position_value = min(position_value, max_risk_amount / risk_per_share * entry_price)

        # Calculate quantity
        quantity = position_value / entry_price

        return quantity

    def _get_conviction_multiplier(self, conviction_score: float) -> float:
        """Get position size multiplier based on conviction score"""
        if conviction_score >= 95:
            return 1.0  # Full position
        elif conviction_score >= 80:
            return 0.7  # 70% position
        elif conviction_score >= 60:
            return 0.4  # 40% position (paper trading only)
        else:
            return 0.0  # No position

    def _kelly_criterion_size(self, portfolio_value: float, risk_per_share: float,
                            entry_price: float, conviction_score: float) -> float:
        """
        Kelly Criterion position sizing
        f* = (bp - q) / b where b is odds, p is win probability, q is loss probability
        """
        # Convert conviction to win probability
        win_prob = conviction_score / 100
        loss_prob = 1 - win_prob

        # Calculate odds (risk:reward ratio)
        reward_per_share = entry_price * self.params.take_profit_pct / self.params.stop_loss_pct
        odds = reward_per_share / risk_per_share if risk_per_share > 0 else 1

        # Kelly fraction
        kelly_fraction = (odds * win_prob - loss_prob) / odds

        # Apply Kelly fraction with safety (half-Kelly for safety)
        kelly_fraction = max(0, min(kelly_fraction * 0.5, 0.25))  # Max 25% of portfolio

        return portfolio_value * kelly_fraction

    def _volatility_adjusted_size(self, portfolio_value: float, volatility: float,
                                conviction_score: float) -> float:
        """
        Volatility-adjusted position sizing
        """
        # Base position
        base_position = portfolio_value * 0.15

        # Adjust for volatility (inverse relationship)
        if volatility > 0:
            volatility_adjustment = min(0.02 / volatility, 2.0)  # Cap at 2x
        else:
            volatility_adjustment = 1.0

        # Apply conviction multiplier
        conviction_multiplier = self._get_conviction_multiplier(conviction_score)

        position_value = base_position * volatility_adjustment * conviction_multiplier

        return position_value

    def can_open_position(self, symbol: str, action: str, conviction_score: float,
                         portfolio_value: float) -> Tuple[bool, str]:
        """
        Check if position can be opened based on risk rules
        """
        # Check conviction threshold
        if conviction_score < 60:
            return False, f"Conviction score {conviction_score}% too low (minimum 60%)"

        # Check daily loss limit
        if self.daily_pnl <= -portfolio_value * self.params.daily_loss_limit_pct:
            return False, f"Daily loss limit of {self.params.daily_loss_limit_pct*100}% reached"

        # Check maximum concurrent positions
        if len(self.active_positions) >= self.params.max_concurrent_positions:
            return False, f"Maximum {self.params.max_concurrent_positions} positions already open"

        # Check if already have position in this symbol
        if symbol in self.active_positions:
            return False, f"Already have position in {symbol}"

        # Check portfolio drawdown
        if self.current_drawdown >= self.params.max_drawdown_limit:
            return False, f"Maximum drawdown limit of {self.params.max_drawdown_limit*100}% reached"

        # Check available capital
        min_capital_required = portfolio_value * 0.1  # Keep 10% reserve
        if self.current_capital < min_capital_required:
            return False, f"Insufficient capital (need at least {min_capital_required:.2f})"

        return True, "Position meets all risk criteria"

    def validate_correlation(self, new_symbol: str, portfolio_value: float) -> Tuple[bool, str]:
        """
        Check if adding new position would exceed correlation limits
        """
        if not self.active_positions:
            return True, "No existing positions"

        # Get correlation with existing positions
        for existing_symbol in self.active_positions.keys():
            correlation = self._get_correlation(new_symbol, existing_symbol)
            if correlation > self.params.correlation_threshold:
                return False, f"High correlation ({correlation:.2f}) with {existing_symbol}"

        return True, "Correlation within acceptable limits"

    def _get_correlation(self, symbol1: str, symbol2: str) -> float:
        """
        Get correlation between two symbols
        Uses cached values or defaults
        """
        cache_key = f"{symbol1}_{symbol2}"

        if cache_key in self.correlation_cache:
            return self.correlation_cache[cache_key]

        # Default correlations (in production, would calculate from historical data)
        default_correlations = {
            'BTC_ETH': 0.7, 'BTC_ADA': 0.5, 'BTC_DOGE': 0.4, 'BTC_SHIB': 0.3,
            'ETH_ADA': 0.6, 'ETH_DOGE': 0.5, 'ETH_SHIB': 0.4,
            'ADA_DOGE': 0.3, 'ADA_SHIB': 0.2, 'DOGE_SHIB': 0.6
        }

        # Try both combinations
        key1 = f"{symbol1}_{symbol2}"
        key2 = f"{symbol2}_{symbol1}"

        correlation = default_correlations.get(key1, default_correlations.get(key2, 0.3))

        # Cache the result
        self.correlation_cache[cache_key] = correlation

        return correlation

    def calculate_portfolio_risk(self) -> RiskMetrics:
        """
        Calculate comprehensive portfolio risk metrics
        """
        if not self.portfolio_values:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, {}, 0)

        # Convert to pandas series for calculations
        values = pd.Series(self.portfolio_values)
        returns = values.pct_change().dropna()

        if len(returns) == 0:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, {}, 0)

        # Calculate metrics
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        sharpe_ratio = (returns.mean() * 252) / volatility if volatility > 0 else 0

        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5) * np.sqrt(252)

        # Expected Shortfall (Conditional VaR)
        worst_5_percent = returns[returns <= np.percentile(returns, 5)]
        expected_shortfall = worst_5_percent.mean() * np.sqrt(252) if len(worst_5_percent) > 0 else 0

        # Maximum drawdown
        rolling_max = values.expanding().max()
        drawdown = (values - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Current portfolio risk (sum of individual position risks)
        portfolio_risk = sum(pos.get('risk_amount', 0) for pos in self.active_positions.values())

        # Beta (assuming market is BTC)
        beta = self._calculate_portfolio_beta()

        # Correlation matrix
        correlation_matrix = self._calculate_correlation_matrix()

        # Concentration risk
        concentration_risk = self._calculate_concentration_risk()

        return RiskMetrics(
            current_portfolio_risk=portfolio_risk,
            max_drawdown=abs(max_drawdown),
            value_at_risk_95=abs(var_95),
            expected_shortfall=abs(expected_shortfall),
            sharpe_ratio=sharpe_ratio,
            volatility=volatility,
            beta=beta,
            correlation_matrix=correlation_matrix,
            concentration_risk=concentration_risk
        )

    def _calculate_portfolio_beta(self) -> float:
        """Calculate portfolio beta relative to BTC"""
        if not self.active_positions:
            return 1.0

        # Default betas for cryptocurrencies relative to BTC
        symbol_betas = {
            'BTC': 1.0,
            'ETH': 0.9,
            'ADA': 0.8,
            'DOGE': 1.1,
            'SHIB': 1.2
        }

        weighted_beta = 0.0
        total_weight = 0.0

        for symbol, position in self.active_positions.items():
            weight = position.get('weight', 0)
            beta = symbol_betas.get(symbol, 1.0)
            weighted_beta += weight * beta
            total_weight += weight

        return weighted_beta / total_weight if total_weight > 0 else 1.0

    def _calculate_correlation_matrix(self) -> Dict:
        """Calculate correlation matrix for active positions"""
        symbols = list(self.active_positions.keys())
        if len(symbols) < 2:
            return {}

        correlation_matrix = {}
        for symbol1 in symbols:
            correlation_matrix[symbol1] = {}
            for symbol2 in symbols:
                if symbol1 == symbol2:
                    correlation_matrix[symbol1][symbol2] = 1.0
                else:
                    correlation = self._get_correlation(symbol1, symbol2)
                    correlation_matrix[symbol1][symbol2] = correlation

        return correlation_matrix

    def _calculate_concentration_risk(self) -> float:
        """Calculate concentration risk (largest position weight)"""
        if not self.active_positions:
            return 0.0

        weights = [pos.get('weight', 0) for pos in self.active_positions.values()]
        return max(weights) if weights else 0.0

    def update_position(self, symbol: str, action: str, quantity: float, entry_price: float,
                      current_price: float, conviction_score: float):
        """
        Update or add position tracking
        """
        # Calculate position weight
        position_value = quantity * current_price
        portfolio_value = self.current_capital + sum(pos.get('value', 0) for pos in self.active_positions.values())
        weight = position_value / portfolio_value if portfolio_value > 0 else 0

        # Calculate risk amount
        stop_loss_price = entry_price * (1 - self.params.stop_loss_pct) if action == 'BUY' else entry_price * (1 + self.params.stop_loss_pct)
        risk_amount = abs(entry_price - stop_loss_price) * quantity

        position_data = {
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'entry_price': entry_price,
            'current_price': current_price,
            'value': position_value,
            'weight': weight,
            'risk_amount': risk_amount,
            'unrealized_pnl': (current_price - entry_price) * quantity if action == 'BUY' else (entry_price - current_price) * quantity,
            'conviction_score': conviction_score,
            'entry_time': datetime.now(),
            'stop_loss': stop_loss_price,
            'take_profit': entry_price * (1 + self.params.take_profit_pct) if action == 'BUY' else entry_price * (1 - self.params.take_profit_pct)
        }

        self.active_positions[symbol] = position_data

    def close_position(self, symbol: str, exit_price: float) -> float:
        """
        Close position and calculate realized P&L
        """
        if symbol not in self.active_positions:
            return 0.0

        position = self.active_positions[symbol]
        entry_price = position['entry_price']
        quantity = position['quantity']
        action = position['action']

        # Calculate realized P&L
        if action == 'BUY':
            realized_pnl = (exit_price - entry_price) * quantity
        else:  # SELL
            realized_pnl = (entry_price - exit_price) * quantity

        # Update capital and P&L tracking
        self.current_capital += realized_pnl
        self.daily_pnl += realized_pnl
        self.daily_trades_count += 1

        # Add to position history
        self.position_history.append({
            'symbol': symbol,
            'action': action,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'realized_pnl': realized_pnl,
            'conviction_score': position['conviction_score'],
            'entry_time': position['entry_time'],
            'exit_time': datetime.now(),
            'duration_hours': (datetime.now() - position['entry_time']).total_seconds() / 3600
        })

        # Remove from active positions
        del self.active_positions[symbol]

        # Update portfolio values for risk calculations
        self._update_portfolio_tracking()

        return realized_pnl

    def _update_portfolio_tracking(self):
        """Update portfolio value and drawdown tracking"""
        current_portfolio_value = self.current_capital + sum(pos.get('value', 0) for pos in self.active_positions.values())
        self.portfolio_values.append(current_portfolio_value)

        # Update drawdown
        if current_portfolio_value > self.peak_capital:
            self.peak_capital = current_portfolio_value

        self.current_drawdown = (self.peak_capital - current_portfolio_value) / self.peak_capital
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

    def get_risk_report(self) -> Dict:
        """
        Generate comprehensive risk report
        """
        risk_metrics = self.calculate_portfolio_risk()

        report = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_summary': {
                'current_capital': self.current_capital,
                'peak_capital': self.peak_capital,
                'daily_pnl': self.daily_pnl,
                'current_drawdown': self.current_drawdown,
                'max_drawdown': self.max_drawdown,
                'active_positions': len(self.active_positions),
                'daily_trades': self.daily_trades_count
            },
            'risk_metrics': {
                'portfolio_risk': risk_metrics.current_portfolio_risk,
                'volatility': risk_metrics.volatility,
                'sharpe_ratio': risk_metrics.sharpe_ratio,
                'var_95': risk_metrics.value_at_risk_95,
                'expected_shortfall': risk_metrics.expected_shortfall,
                'beta': risk_metrics.beta,
                'concentration_risk': risk_metrics.concentration_risk
            },
            'positions': self.active_positions,
            'risk_limits': {
                'max_position_risk': self.params.max_portfolio_risk,
                'max_concurrent_positions': self.params.max_concurrent_positions,
                'max_daily_loss': self.params.daily_loss_limit_pct,
                'max_drawdown': self.params.max_drawdown_limit,
                'correlation_threshold': self.params.correlation_threshold
            },
            'warnings': self._generate_risk_warnings(risk_metrics)
        }

        return report

    def _generate_risk_warnings(self, risk_metrics: RiskMetrics) -> List[str]:
        """Generate risk warnings based on current metrics"""
        warnings = []

        if risk_metrics.max_drawdown > self.params.max_drawdown_limit * 0.8:
            warnings.append(f"High drawdown: {risk_metrics.max_drawdown*100:.1f}% (limit: {self.params.max_drawdown_limit*100:.1f}%)")

        if self.daily_pnl < -self.current_capital * self.params.daily_loss_limit_pct * 0.8:
            warnings.append(f"Approaching daily loss limit: {self.daily_pnl:.2f}")

        if risk_metrics.volatility > 0.5:  # 50% annual volatility
            warnings.append(f"High portfolio volatility: {risk_metrics.volatility*100:.1f}%")

        if risk_metrics.concentration_risk > 0.4:  # 40% in single position
            warnings.append(f"High concentration risk: {risk_metrics.concentration_risk*100:.1f}%")

        if len(self.active_positions) >= self.params.max_concurrent_positions * 0.8:
            warnings.append(f"Approaching max positions: {len(self.active_positions)}/{self.params.max_concurrent_positions}")

        return warnings

    def save_risk_report(self, filename: str = None):
        """Save risk report to file"""
        if filename is None:
            filename = f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        report = self.get_risk_report()

        # Convert datetime objects to strings for JSON serialization
        def json_serial(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=json_serial)

        print(f"💾 Risk report saved to {filename}")

if __name__ == "__main__":
    # Example usage
    print("🛡️ Testing Risk Manager...")

    risk_manager = RiskManager(initial_capital=1000)

    # Test position sizing
    position_size = risk_manager.calculate_position_size(
        symbol='BTC',
        entry_price=45000,
        stop_loss_price=44100,  # 2% below
        conviction_score=85,
        portfolio_value=1000,
        volatility=0.03
    )

    print(f"\n📏 Position Size Calculation:")
    print(f"   Quantity: {position_size:.6f} BTC")
    print(f"   Position Value: ${position_size * 45000:.2f}")

    # Test position opening validation
    can_open, reason = risk_manager.can_open_position(
        symbol='BTC',
        action='BUY',
        conviction_score=85,
        portfolio_value=1000
    )

    print(f"\n✅ Position Opening Check: {can_open}")
    print(f"   Reason: {reason}")

    # Test correlation validation
    can_open_correlation, correlation_reason = risk_manager.validate_correlation('ETH', 1000)
    print(f"\n🔗 Correlation Check: {can_open_correlation}")
    print(f"   Reason: {correlation_reason}")

    # Add sample position
    risk_manager.update_position(
        symbol='BTC',
        action='BUY',
        quantity=0.01,
        entry_price=45000,
        current_price=45500,
        conviction_score=85
    )

    # Generate risk report
    risk_report = risk_manager.get_risk_report()
    print(f"\n📊 Risk Report:")
    print(f"   Portfolio Risk: {risk_report['risk_metrics']['portfolio_risk']:.2f}")
    print(f"   Current Drawdown: {risk_report['portfolio_summary']['current_drawdown']*100:.1f}%")
    print(f"   Warnings: {len(risk_report['warnings'])}")

    for warning in risk_report['warnings']:
        print(f"   ⚠️ {warning}")

    # Save report
    risk_manager.save_risk_report()