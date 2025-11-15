import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

@dataclass
class AssetData:
    symbol: str
    returns: pd.Series
    expected_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    beta: float
    market_cap_rank: int
    trading_volume: float

@dataclass
class PortfolioAllocation:
    symbol: str
    weight: float
    allocation_amount: float
    expected_return: float
    risk_contribution: float

@dataclass
class OptimizationResult:
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    diversification_ratio: float
    risk_parity_score: float
    allocation_breakdown: List[PortfolioAllocation]

class PortfolioOptimizer:
    """
    Advanced portfolio optimization for cryptocurrency assets
    Implements multiple optimization strategies and risk metrics
    """

    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.supported_symbols = ['BTC', 'ETH', 'ADA', 'DOGE', 'SHIB']
        self.min_weight = 0.05  # Minimum 5% allocation
        self.max_weight = 0.40   # Maximum 40% allocation
        self.rebalance_threshold = 0.10  # 10% deviation triggers rebalancing

        # Asset characteristics for crypto assets
        self.asset_characteristics = {
            'BTC': {'beta': 1.0, 'market_cap_rank': 1, 'volatility_factor': 0.8},
            'ETH': {'beta': 0.9, 'market_cap_rank': 2, 'volatility_factor': 0.9},
            'ADA': {'beta': 0.8, 'market_cap_rank': 8, 'volatility_factor': 1.0},
            'DOGE': {'beta': 1.1, 'market_cap_rank': 10, 'volatility_factor': 1.2},
            'SHIB': {'beta': 1.2, 'market_cap_rank': 16, 'volatility_factor': 1.3}
        }

    def prepare_asset_data(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, AssetData]:
        """
        Prepare asset data for optimization
        """
        asset_data = {}

        for symbol, data in price_data.items():
            if data.empty or 'close' not in data.columns:
                continue

            # Calculate daily returns
            returns = data['close'].pct_change().dropna()

            if len(returns) < 30:  # Need at least 30 days of data
                continue

            # Calculate metrics
            expected_return = returns.mean() * 252  # Annualized return
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            sharpe_ratio = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0

            # Calculate maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()

            # Get asset characteristics
            characteristics = self.asset_characteristics.get(symbol, {'beta': 1.0, 'market_cap_rank': 999, 'volatility_factor': 1.0})

            # Estimate trading volume (use average daily volume)
            trading_volume = data['volume'].mean() if 'volume' in data.columns else 0

            asset_data[symbol] = AssetData(
                symbol=symbol,
                returns=returns,
                expected_return=expected_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=abs(max_drawdown),
                beta=characteristics['beta'],
                market_cap_rank=characteristics['market_cap_rank'],
                trading_volume=trading_volume
            )

        return asset_data

    def calculate_correlation_matrix(self, asset_data: Dict[str, AssetData]) -> pd.DataFrame:
        """
        Calculate correlation matrix using shrinkage estimator for stability
        """
        if len(asset_data) < 2:
            return pd.DataFrame()

        # Create returns matrix
        symbols = list(asset_data.keys())
        returns_matrix = pd.DataFrame({symbol: asset_data[symbol].returns for symbol in symbols})

        # Use Ledoit-Wolf shrinkage estimator for stable covariance estimation
        lw = LedoitWolf()
        lw.fit(returns_matrix.dropna())
        cov_matrix = pd.DataFrame(lw.covariance_, index=symbols, columns=symbols)

        # Convert to correlation matrix
        vol_vector = np.sqrt(np.diag(cov_matrix))
        correlation_matrix = cov_matrix / np.outer(vol_vector, vol_vector)

        return correlation_matrix

    def optimize_mean_variance(self, asset_data: Dict[str, AssetData],
                             target_return: Optional[float] = None) -> OptimizationResult:
        """
        Mean-variance optimization (Markowitz)
        """
        symbols = list(asset_data.keys())
        n_assets = len(symbols)

        if n_assets < 2:
            # Single asset case
            symbol = symbols[0]
            return OptimizationResult(
                weights={symbol: 1.0},
                expected_return=asset_data[symbol].expected_return,
                volatility=asset_data[symbol].volatility,
                sharpe_ratio=asset_data[symbol].sharpe_ratio,
                max_drawdown=asset_data[symbol].max_drawdown,
                diversification_ratio=0.0,
                risk_parity_score=0.0,
                allocation_breakdown=[]
            )

        # Prepare expected returns and covariance matrix
        expected_returns = np.array([asset_data[symbol].expected_return for symbol in symbols])
        returns_matrix = pd.DataFrame({symbol: asset_data[symbol].returns for symbol in symbols})
        cov_matrix = returns_matrix.cov() * 252  # Annualized covariance

        # Optimization constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
        ]

        # Add weight bounds
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]

        # Objective function: minimize volatility for given return (or maximize Sharpe)
        def objective(weights):
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            if target_return is None:
                # Maximize Sharpe ratio (minimize negative Sharpe)
                portfolio_return = np.dot(weights, expected_returns)
                sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility
                return -sharpe
            else:
                # Minimize volatility subject to target return
                portfolio_return = np.dot(weights, expected_returns)
                return portfolio_volatility + 1000 * abs(portfolio_return - target_return)

        # Add target return constraint if specified
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.dot(x, expected_returns) - target_return
            })

        # Initial guess (equal weight)
        x0 = np.array([1.0 / n_assets] * n_assets)

        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        if not result.success:
            # Fallback to equal weights
            weights = np.array([1.0 / n_assets] * n_assets)
        else:
            weights = result.x

        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        portfolio_sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility

        # Calculate additional metrics
        correlation_matrix = self.calculate_correlation_matrix(asset_data)
        diversification_ratio = self._calculate_diversification_ratio(weights, cov_matrix)
        risk_parity_score = self._calculate_risk_parity_score(weights, cov_matrix)

        # Create allocation breakdown
        allocation_breakdown = []
        for i, symbol in enumerate(symbols):
            allocation_breakdown.append(PortfolioAllocation(
                symbol=symbol,
                weight=weights[i],
                allocation_amount=weights[i] * 1000,  # Assuming ₹1000 portfolio
                expected_return=asset_data[symbol].expected_return,
                risk_contribution=self._calculate_risk_contribution(weights, cov_matrix, i)
            ))

        return OptimizationResult(
            weights=dict(zip(symbols, weights)),
            expected_return=portfolio_return,
            volatility=portfolio_volatility,
            sharpe_ratio=portfolio_sharpe,
            max_drawdown=self._estimate_portfolio_max_drawdown(asset_data, weights),
            diversification_ratio=diversification_ratio,
            risk_parity_score=risk_parity_score,
            allocation_breakdown=allocation_breakdown
        )

    def optimize_equal_risk_contribution(self, asset_data: Dict[str, AssetData]) -> OptimizationResult:
        """
        Risk Parity optimization - equal risk contribution from each asset
        """
        symbols = list(asset_data.keys())
        n_assets = len(symbols)

        if n_assets < 2:
            # Single asset case
            symbol = symbols[0]
            return OptimizationResult(
                weights={symbol: 1.0},
                expected_return=asset_data[symbol].expected_return,
                volatility=asset_data[symbol].volatility,
                sharpe_ratio=asset_data[symbol].sharpe_ratio,
                max_drawdown=asset_data[symbol].max_drawdown,
                diversification_ratio=0.0,
                risk_parity_score=1.0,
                allocation_breakdown=[]
            )

        # Prepare covariance matrix
        returns_matrix = pd.DataFrame({symbol: asset_data[symbol].returns for symbol in symbols})
        cov_matrix = returns_matrix.cov() * 252  # Annualized covariance

        # Objective function: minimize risk contribution differences
        def objective(weights):
            risk_contributions = self._calculate_all_risk_contributions(weights, cov_matrix)
            # Minimize squared differences from equal risk contribution
            target_risk = 1.0 / n_assets
            return np.sum((risk_contributions - target_risk) ** 2)

        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]

        # Initial guess (equal weights)
        x0 = np.array([1.0 / n_assets] * n_assets)

        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        if not result.success:
            # Fallback to equal weights
            weights = np.array([1.0 / n_assets] * n_assets)
        else:
            weights = result.x

        # Calculate portfolio metrics
        expected_returns = np.array([asset_data[symbol].expected_return for symbol in symbols])
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        portfolio_sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility

        # Additional metrics
        diversification_ratio = self._calculate_diversification_ratio(weights, cov_matrix)
        risk_parity_score = self._calculate_risk_parity_score(weights, cov_matrix)

        # Create allocation breakdown
        allocation_breakdown = []
        for i, symbol in enumerate(symbols):
            allocation_breakdown.append(PortfolioAllocation(
                symbol=symbol,
                weight=weights[i],
                allocation_amount=weights[i] * 1000,
                expected_return=asset_data[symbol].expected_return,
                risk_contribution=self._calculate_risk_contribution(weights, cov_matrix, i)
            ))

        return OptimizationResult(
            weights=dict(zip(symbols, weights)),
            expected_return=portfolio_return,
            volatility=portfolio_volatility,
            sharpe_ratio=portfolio_sharpe,
            max_drawdown=self._estimate_portfolio_max_drawdown(asset_data, weights),
            diversification_ratio=diversification_ratio,
            risk_parity_score=risk_parity_score,
            allocation_breakdown=allocation_breakdown
        )

    def optimize_minimum_variance(self, asset_data: Dict[str, AssetData]) -> OptimizationResult:
        """
        Minimum variance optimization
        """
        symbols = list(asset_data.keys())
        n_assets = len(symbols)

        if n_assets < 2:
            # Single asset case
            symbol = symbols[0]
            return OptimizationResult(
                weights={symbol: 1.0},
                expected_return=asset_data[symbol].expected_return,
                volatility=asset_data[symbol].volatility,
                sharpe_ratio=asset_data[symbol].sharpe_ratio,
                max_drawdown=asset_data[symbol].max_drawdown,
                diversification_ratio=0.0,
                risk_parity_score=0.0,
                allocation_breakdown=[]
            )

        # Prepare covariance matrix
        returns_matrix = pd.DataFrame({symbol: asset_data[symbol].returns for symbol in symbols})
        cov_matrix = returns_matrix.cov() * 252  # Annualized covariance

        # Objective function: minimize portfolio variance
        def objective(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))

        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]

        # Initial guess (equal weights)
        x0 = np.array([1.0 / n_assets] * n_assets)

        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        if not result.success:
            # Fallback to equal weights
            weights = np.array([1.0 / n_assets] * n_assets)
        else:
            weights = result.x

        # Calculate portfolio metrics
        expected_returns = np.array([asset_data[symbol].expected_return for symbol in symbols])
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        portfolio_sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility

        # Additional metrics
        diversification_ratio = self._calculate_diversification_ratio(weights, cov_matrix)
        risk_parity_score = self._calculate_risk_parity_score(weights, cov_matrix)

        # Create allocation breakdown
        allocation_breakdown = []
        for i, symbol in enumerate(symbols):
            allocation_breakdown.append(PortfolioAllocation(
                symbol=symbol,
                weight=weights[i],
                allocation_amount=weights[i] * 1000,
                expected_return=asset_data[symbol].expected_return,
                risk_contribution=self._calculate_risk_contribution(weights, cov_matrix, i)
            ))

        return OptimizationResult(
            weights=dict(zip(symbols, weights)),
            expected_return=portfolio_return,
            volatility=portfolio_volatility,
            sharpe_ratio=portfolio_sharpe,
            max_drawdown=self._estimate_portfolio_max_drawdown(asset_data, weights),
            diversification_ratio=diversification_ratio,
            risk_parity_score=risk_parity_score,
            allocation_breakdown=allocation_breakdown
        )

    def optimize_hybrid(self, asset_data: Dict[str, AssetData]) -> OptimizationResult:
        """
        Hybrid optimization combining multiple objectives
        """
        # Get results from different optimization methods
        mv_result = self.optimize_mean_variance(asset_data)
        rp_result = self.optimize_equal_risk_contribution(asset_data)
        mv_result = self.optimize_minimum_variance(asset_data)

        # Combine weights with different emphasis
        symbols = list(asset_data.keys())
        combined_weights = {}

        for symbol in symbols:
            # Weight average: 50% mean-variance, 30% risk parity, 20% minimum variance
            weight = (
                0.5 * mv_result.weights.get(symbol, 0) +
                0.3 * rp_result.weights.get(symbol, 0) +
                0.2 * mv_result.weights.get(symbol, 0)
            )
            combined_weights[symbol] = weight

        # Normalize weights to sum to 1
        total_weight = sum(combined_weights.values())
        if total_weight > 0:
            combined_weights = {k: v/total_weight for k, v in combined_weights.items()}

        # Calculate portfolio metrics for combined weights
        expected_returns = np.array([asset_data[symbol].expected_return for symbol in symbols])
        returns_matrix = pd.DataFrame({symbol: asset_data[symbol].returns for symbol in symbols})
        cov_matrix = returns_matrix.cov() * 252

        weights_array = np.array([combined_weights[symbol] for symbol in symbols])
        portfolio_return = np.dot(weights_array, expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights_array.T, np.dot(cov_matrix, weights_array)))
        portfolio_sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility

        # Additional metrics
        diversification_ratio = self._calculate_diversification_ratio(weights_array, cov_matrix)
        risk_parity_score = self._calculate_risk_parity_score(weights_array, cov_matrix)

        # Create allocation breakdown
        allocation_breakdown = []
        for i, symbol in enumerate(symbols):
            allocation_breakdown.append(PortfolioAllocation(
                symbol=symbol,
                weight=combined_weights[symbol],
                allocation_amount=combined_weights[symbol] * 1000,
                expected_return=asset_data[symbol].expected_return,
                risk_contribution=self._calculate_risk_contribution(weights_array, cov_matrix, i)
            ))

        return OptimizationResult(
            weights=combined_weights,
            expected_return=portfolio_return,
            volatility=portfolio_volatility,
            sharpe_ratio=portfolio_sharpe,
            max_drawdown=self._estimate_portfolio_max_drawdown(asset_data, weights_array),
            diversification_ratio=diversification_ratio,
            risk_parity_score=risk_parity_score,
            allocation_breakdown=allocation_breakdown
        )

    def _calculate_diversification_ratio(self, weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
        """
        Calculate diversification ratio
        """
        weighted_vol = np.sqrt(np.sum((weights ** 2) * np.diag(cov_matrix)))
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        return weighted_vol / portfolio_vol if portfolio_vol > 0 else 0

    def _calculate_risk_parity_score(self, weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
        """
        Calculate risk parity score (how equal risk contributions are)
        """
        risk_contributions = self._calculate_all_risk_contributions(weights, cov_matrix)
        target_risk = 1.0 / len(weights)

        # Calculate coefficient of variation of risk contributions
        mean_risk = np.mean(risk_contributions)
        std_risk = np.std(risk_contributions)

        return 1 - (std_risk / mean_risk) if mean_risk > 0 else 0

    def _calculate_all_risk_contributions(self, weights: np.ndarray, cov_matrix: pd.DataFrame) -> np.ndarray:
        """
        Calculate risk contributions for all assets
        """
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
        risk_contributions = weights * marginal_contrib

        return risk_contributions

    def _calculate_risk_contribution(self, weights: np.ndarray, cov_matrix: pd.DataFrame, asset_index: int) -> float:
        """
        Calculate risk contribution for a specific asset
        """
        risk_contributions = self._calculate_all_risk_contributions(weights, cov_matrix)
        return risk_contributions[asset_index]

    def _estimate_portfolio_max_drawdown(self, asset_data: Dict[str, AssetData], weights: np.ndarray) -> float:
        """
        Estimate portfolio maximum drawdown based on individual asset drawdowns
        """
        symbols = list(asset_data.keys())
        weighted_drawdown = 0

        for i, symbol in enumerate(symbols):
            weighted_drawdown += weights[i] * asset_data[symbol].max_drawdown

        return weighted_drawdown

    def generate_efficient_frontier(self, asset_data: Dict[str, AssetData],
                                  num_portfolios: int = 50) -> pd.DataFrame:
        """
        Generate efficient frontier for visualization
        """
        symbols = list(asset_data.keys())
        expected_returns = np.array([asset_data[symbol].expected_return for symbol in symbols])
        returns_matrix = pd.DataFrame({symbol: asset_data[symbol].returns for symbol in symbols})
        cov_matrix = returns_matrix.cov() * 252

        # Calculate min and max returns
        min_ret = min(expected_returns)
        max_ret = max(expected_returns)
        target_returns = np.linspace(min_ret, max_ret, num_portfolios)

        frontier_data = []

        for target_return in target_returns:
            try:
                result = self.optimize_mean_variance(asset_data, target_return)
                frontier_data.append({
                    'target_return': target_return,
                    'portfolio_return': result.expected_return,
                    'portfolio_volatility': result.volatility,
                    'sharpe_ratio': result.sharpe_ratio
                })
            except:
                continue

        return pd.DataFrame(frontier_data)

    def should_rebalance(self, current_weights: Dict[str, float],
                        target_weights: Dict[str, float]) -> bool:
        """
        Check if portfolio needs rebalancing
        """
        for symbol in target_weights:
            current_weight = current_weights.get(symbol, 0)
            target_weight = target_weights[symbol]

            if abs(current_weight - target_weight) > self.rebalance_threshold:
                return True

        return False

    def get_rebalancing_orders(self, current_weights: Dict[str, float],
                              target_weights: Dict[str, float],
                              portfolio_value: float) -> List[Dict]:
        """
        Calculate rebalancing orders
        """
        orders = []

        for symbol, target_weight in target_weights.items():
            current_weight = current_weights.get(symbol, 0)
            weight_diff = target_weight - current_weight

            if abs(weight_diff) > 0.01:  # Only if difference > 1%
                order_value = weight_diff * portfolio_value

                orders.append({
                    'symbol': symbol,
                    'action': 'BUY' if weight_diff > 0 else 'SELL',
                    'weight_diff': weight_diff,
                    'order_value': order_value,
                    'current_weight': current_weight,
                    'target_weight': target_weight
                })

        return orders

    def generate_portfolio_report(self, optimization_result: OptimizationResult,
                                 method: str = "Hybrid") -> Dict:
        """
        Generate comprehensive portfolio optimization report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'optimization_method': method,
            'summary': {
                'expected_return': optimization_result.expected_return,
                'volatility': optimization_result.volatility,
                'sharpe_ratio': optimization_result.sharpe_ratio,
                'max_drawdown': optimization_result.max_drawdown,
                'diversification_ratio': optimization_result.diversification_ratio,
                'risk_parity_score': optimization_result.risk_parity_score
            },
            'allocations': [],
            'risk_analysis': {
                'largest_allocation': max(optimization_result.weights.values()) if optimization_result.weights else 0,
                'smallest_allocation': min(optimization_result.weights.values()) if optimization_result.weights else 0,
                'number_of_assets': len(optimization_result.weights),
                'allocation_dispersion': np.std(list(optimization_result.weights.values())) if optimization_result.weights else 0
            },
            'recommendations': []
        }

        # Add allocation details
        for allocation in optimization_result.allocation_breakdown:
            report['allocations'].append({
                'symbol': allocation.symbol,
                'weight': allocation.weight,
                'amount': allocation.allocation_amount,
                'expected_return': allocation.expected_return,
                'risk_contribution': allocation.risk_contribution
            })

        # Generate recommendations
        if optimization_result.sharpe_ratio < 0.5:
            report['recommendations'].append("Low Sharpe ratio - consider different asset mix or risk management")

        if optimization_result.diversification_ratio < 1.2:
            report['recommendations'].append("Low diversification - consider adding uncorrelated assets")

        if optimization_result.risk_parity_score < 0.7:
            report['recommendations'].append("Unequal risk contributions - consider risk parity approach")

        if report['risk_analysis']['largest_allocation'] > 0.4:
            report['recommendations'].append("High concentration in single asset - consider diversification")

        return report

if __name__ == "__main__":
    # Example usage
    print("📊 Testing Portfolio Optimizer...")

    # Create sample price data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=90, freq='D')

    symbols = ['BTC', 'ETH', 'ADA', 'DOGE']
    price_data = {}

    # Generate correlated price movements
    base_returns = np.random.randn(90) * 0.03

    for i, symbol in enumerate(symbols):
        # Add symbol-specific characteristics
        volatility_multiplier = {'BTC': 0.8, 'ETH': 0.9, 'ADA': 1.0, 'DOGE': 1.2}[symbol]
        symbol_returns = base_returns * volatility_multiplier + np.random.randn(90) * 0.01

        # Generate price series
        prices = [100]
        for ret in symbol_returns:
            prices.append(prices[-1] * (1 + ret))

        df = pd.DataFrame({
            'close': prices[1:],
            'volume': np.random.lognormal(10, 1, 90)
        }, index=dates)

        price_data[symbol] = df

    # Test optimization
    optimizer = PortfolioOptimizer()

    # Prepare asset data
    asset_data = optimizer.prepare_asset_data(price_data)
    print(f"\n📈 Prepared data for {len(asset_data)} assets")

    # Run different optimizations
    methods = ['Mean-Variance', 'Risk Parity', 'Minimum Variance', 'Hybrid']
    results = {}

    for method in methods:
        if method == 'Mean-Variance':
            result = optimizer.optimize_mean_variance(asset_data)
        elif method == 'Risk Parity':
            result = optimizer.optimize_equal_risk_contribution(asset_data)
        elif method == 'Minimum Variance':
            result = optimizer.optimize_minimum_variance(asset_data)
        else:  # Hybrid
            result = optimizer.optimize_hybrid(asset_data)

        results[method] = result

        print(f"\n🎯 {method} Optimization:")
        print(f"   Expected Return: {result.expected_return:.2%}")
        print(f"   Volatility: {result.volatility:.2%}")
        print(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"   Diversification Ratio: {result.diversification_ratio:.2f}")

    # Generate efficient frontier
    frontier = optimizer.generate_efficient_frontier(asset_data, num_portfolios=20)
    print(f"\n📈 Efficient Frontier: {len(frontier)} points generated")

    # Test rebalancing
    current_weights = {'BTC': 0.4, 'ETH': 0.3, 'ADA': 0.2, 'DOGE': 0.1}
    hybrid_result = results['Hybrid']

    needs_rebalance = optimizer.should_rebalance(current_weights, hybrid_result.weights)
    print(f"\n🔄 Rebalancing needed: {needs_rebalance}")

    if needs_rebalance:
        orders = optimizer.get_rebalancing_orders(current_weights, hybrid_result.weights, 1000)
        print(f"   Rebalancing orders: {len(orders)}")
        for order in orders:
            print(f"   {order['symbol']}: {order['action']} {order['weight_diff']:+.1%}")

    # Generate comprehensive report
    report = optimizer.generate_portfolio_report(hybrid_result, "Hybrid")
    print(f"\n📋 Portfolio Report:")
    for key, value in report['summary'].items():
        print(f"   {key}: {value:.3f}")

    print(f"\n💡 Recommendations: {len(report['recommendations'])}")
    for rec in report['recommendations']:
        print(f"   • {rec}")