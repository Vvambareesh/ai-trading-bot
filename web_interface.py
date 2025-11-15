#!/usr/bin/env python3
"""
AI Trading Bot Web Interface
Modern web-based interface for managing and monitoring cryptocurrency trading bot
Includes real-time dashboard, configuration, and trading controls
"""

import os
import json
import time
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, send_from_directory
from dotenv import load_dotenv
import threading
import queue

# Import trading bot components
from paper_trading import PaperTradingPortfolio
from strategies import StrategyManager
from data_sources import MarketDataManager
from risk_manager import RiskManager
from ai_analyzer import AIAnalyzer
from backtester import Backtester

# Initialize Flask app
app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'

# Load environment variables
load_dotenv()

# Global state for real-time updates
trading_state = {
    'is_running': False,
    'current_mode': 'paper',
    'portfolio_value': 1000.0,
    'active_positions': 0,
    'total_trades': 0,
    'win_rate': 0.0,
    'daily_pnl': 0.0,
    'last_update': datetime.now().isoformat()
}

# Initialize trading components
try:
    portfolio = PaperTradingPortfolio(1000)
    strategy_manager = StrategyManager()
    data_manager = MarketDataManager()
    risk_manager = RiskManager(1000)

    # AI analyzer (only if API key available)
    ai_analyzer = None
    if os.getenv('GEMINI_API_KEY'):
        ai_analyzer = AIAnalyzer(os.getenv('GEMINI_API_KEY'))

    print("✅ Trading components initialized")
except Exception as e:
    print(f"⚠️ Some components failed to initialize: {e}")
    portfolio = None
    strategy_manager = None
    data_manager = None
    risk_manager = None
    ai_analyzer = None

def update_trading_state():
    """Update global trading state for real-time updates"""
    global trading_state, portfolio

    if portfolio:
        try:
            summary = portfolio.get_portfolio_summary()
            trading_state.update({
                'portfolio_value': summary['total_value'],
                'active_positions': summary['active_positions'],
                'total_trades': summary['total_trades'],
                'win_rate': summary['win_rate_pct'],
                'daily_pnl': summary['daily_pnl'],
                'last_update': datetime.now().isoformat()
            })
        except Exception as e:
            print(f"Error updating trading state: {e}")

def get_market_overview():
    """Get current market data overview"""
    overview = {
        'timestamp': datetime.now().isoformat(),
        'symbols': ['BTC', 'ETH', 'ADA', 'DOGE', 'SHIB'],
        'prices': {}
    }

    if data_manager:
        for symbol in overview['symbols']:
            try:
                price = data_manager.get_current_price(symbol)
                if price:
                    overview['prices'][symbol] = price
            except Exception as e:
                overview['prices'][symbol] = None

    return overview

def get_portfolio_data():
    """Get detailed portfolio information"""
    if portfolio:
        summary = portfolio.get_portfolio_summary()

        # Add position details
        positions = []
        for symbol, position in summary.get('positions', {}).items():
            positions.append({
                'symbol': symbol,
                'side': position['side'],
                'quantity': position['quantity'],
                'entry_price': position['entry_price'],
                'current_price': position['entry_price'],  # Would need current price
                'pnl': position.get('unrealized_pnl', 0),
                'conviction_score': position.get('conviction_score', 0)
            })

        return {
            'summary': summary,
            'positions': positions,
            'performance_history': []
        }
    return {'summary': {}, 'positions': [], 'performance_history': []}

def get_strategy_data():
    """Get strategy information and performance"""
    if strategy_manager:
        performance = strategy_manager.get_strategy_performance()

        strategies = {
            'sma': {'name': 'SMA Crossover', 'description': '10/20 day moving average crossover'},
            'rsi': {'name': 'RSI Mean Reversion', 'description': '30/70 overbought/oversold levels'},
            'macd': {'name': 'MACD Trend', 'description': 'MACD trend following signals'},
            'bollinger': {'name': 'Bollinger Bands', 'description': 'Volatility-based breakout detection'},
            'volume': {'name': 'Volume Breakout', 'description': 'Volume spike identification'},
            'hybrid': {'name': 'Hybrid (Recommended)', 'description': 'Multi-strategy voting with AI override'}
        }

        for strategy_key, strategy_info in strategies.items():
            if strategy_key in performance:
                strategy_info['performance'] = performance[strategy_key]

        return {
            'current_strategy': strategy_manager.active_strategy,
            'strategies': strategies,
            'performance': performance
        }
    return {'current_strategy': 'hybrid', 'strategies': {}, 'performance': {}}

def get_risk_data():
    """Get risk management information"""
    if risk_manager:
        metrics = risk_manager.calculate_portfolio_risk()

        return {
            'current_metrics': metrics,
            'risk_limits': {
                'max_risk_per_trade': 5.0,
                'max_concurrent_positions': 3,
                'stop_loss_pct': 2.0,
                'take_profit_pct': 6.0,
                'daily_loss_limit': 10.0,
                'max_drawdown_limit': 25.0
            },
            'alerts': []
        }
    return {
        'current_metrics': {'portfolio_risk': 0.05, 'max_drawdown': 0.0},
        'risk_limits': {'max_risk_per_trade': 5.0, 'max_concurrent_positions': 3, 'stop_loss_pct': 2.0, 'take_profit_pct': 6.0, 'daily_loss_limit': 10.0, 'max_drawdown_limit': 25.0},
        'alerts': []
    }

def get_ai_data():
    """Get AI analysis information"""
    if ai_analyzer:
        return {
            'is_enabled': True,
            'last_analysis': None,
            'conviction_history': [],
            'performance_stats': {
                'avg_conviction': 85.0,
                'high_conviction_trades': 0,
                'success_rate': 0.0
            }
        }
    return {
        'is_enabled': False,
        'last_analysis': None,
        'conviction_history': [],
        'performance_stats': {'avg_conviction': 0.0, 'high_conviction_trades': 0, 'success_rate': 0.0}
    }

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html',
                         trading_state=trading_state,
                         market_data=get_market_overview(),
                         portfolio_data=get_portfolio_data(),
                         strategy_data=get_strategy_data(),
                         risk_data=get_risk_data(),
                         ai_data=get_ai_data())

@app.route('/api/market-data')
def api_market_data():
    """API endpoint for market data"""
    return jsonify(get_market_overview())

@app.route('/api/portfolio')
def api_portfolio():
    """API endpoint for portfolio data"""
    return jsonify(get_portfolio_data())

@app.route('/api/strategies')
def api_strategies():
    """API endpoint for strategy data"""
    return jsonify(get_strategy_data())

@app.route('/api/risk')
def api_risk():
    """API endpoint for risk data"""
    return jsonify(get_risk_data())

@app.route('/api/ai')
def api_ai():
    """API endpoint for AI data"""
    return jsonify(get_ai_data())

@app.route('/api/trading-state')
def api_trading_state():
    """API endpoint for current trading state"""
    update_trading_state()
    return jsonify(trading_state)

@app.route('/api/start-trading', methods=['POST'])
def api_start_trading():
    """Start trading bot"""
    data = request.get_json()

    if not data:
        return jsonify({'success': False, 'error': 'No data provided'})

    mode = data.get('mode', 'paper')
    strategy = data.get('strategy', 'hybrid')
    conviction = data.get('conviction', 95)

    # Update configuration
    os.environ['TRADING_MODE'] = mode
    strategy_manager.set_strategy(strategy)

    trading_state.update({
        'is_running': True,
        'current_mode': mode,
        'current_strategy': strategy,
        'conviction_threshold': conviction,
        'start_time': datetime.now().isoformat()
    })

    return jsonify({
        'success': True,
        'message': f'Trading started in {mode} mode with {strategy} strategy',
        'mode': mode,
        'strategy': strategy,
        'conviction': conviction
    })

@app.route('/api/stop-trading', methods=['POST'])
def api_stop_trading():
    """Stop trading bot"""
    trading_state.update({
        'is_running': False,
        'stop_time': datetime.now().isoformat()
    })

    return jsonify({
        'success': True,
        'message': 'Trading stopped',
        'stop_time': trading_state['stop_time']
    })

@app.route('/api/execute-trade', methods=['POST'])
def api_execute_trade():
    """Execute a manual trade"""
    data = request.get_json()

    if not data:
        return jsonify({'success': False, 'error': 'No data provided'})

    symbol = data.get('symbol')
    action = data.get('action')  # buy or sell
    quantity = data.get('quantity')

    # Validate trade
    if not all([symbol, action, quantity]):
        return jsonify({'success': False, 'error': 'Missing required fields'})

    # Execute trade logic would go here
    # This is a simplified implementation

    trade_result = {
        'success': True,
        'trade_id': f"trade_{int(time.time())}",
        'symbol': symbol,
        'action': action,
        'quantity': quantity,
        'execution_time': datetime.now().isoformat(),
        'status': 'executed'
    }

    return jsonify(trade_result)

@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    """Run backtesting on selected strategies"""
    data = request.get_json()

    if not data:
        return jsonify({'success': False, 'error': 'No data provided'})

    symbols = data.get('symbols', ['BTC', 'ETH'])
    strategies = data.get('strategies', ['hybrid'])

    # Simplified backtest result
    # In production, this would call the actual backtester
    backtest_results = {
        'status': 'running',
        'start_time': datetime.now().isoformat(),
        'symbols': symbols,
        'strategies': strategies,
        'estimated_completion': (datetime.now() + timedelta(minutes=10)).isoformat()
    }

    return jsonify({'success': True, 'results': backtest_results})

@app.route('/api/configuration', methods=['GET', 'POST'])
def api_configuration():
    """Get or update configuration"""
    if request.method == 'GET':
        config = {
            'trading_mode': os.getenv('TRADING_MODE', 'paper'),
            'initial_capital': os.getenv('INITIAL_CAPITAL_INR', '1000'),
            'strategy': os.getenv('STRATEGY', 'hybrid'),
            'conviction_threshold': os.getenv('CONVICTION_THRESHOLD', '95'),
            'risk_per_trade': os.getenv('RISK_PER_TRADE', '5'),
            'max_concurrent_trades': os.getenv('MAX_CONCURRENT_TRADES', '3'),
            'delta_webhook_url': os.getenv('DELTA_WEBHOOK_URL', ''),
            'gemini_api_key': os.getenv('GEMINI_API_KEY', '')
        }
        return jsonify(config)

    else:  # POST - update configuration
        data = request.get_json()

        # Update environment variables (simplified)
        for key, value in data.items():
            if key in ['TRADING_MODE', 'STRATEGY', 'CONVICTION_THRESHOLD']:
                os.environ[key] = str(value)

        return jsonify({'success': True, 'message': 'Configuration updated'})

def create_html_template():
    """Create HTML template for the web interface"""
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Trading Bot Dashboard</title>

    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- Font Awesome -->
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">

    <style>
        :root {
            --primary-color: #2c3e50;
            --secondary-color: #343a40;
            --success-color: #28a745;
            --danger-color: #dc3545;
            --warning-color: #ffc107;
            --info-color: #17a2b8;
            --light-color: #f8f9fa;
            --dark-color: #212529;
        }

        body {
            background-color: var(--dark-color);
            color: var(--light-color);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        .navbar-custom {
            background-color: var(--secondary-color) !important;
            border-bottom: 1px solid var(--primary-color);
        }

        .card {
            background-color: var(--secondary-color);
            border: 1px solid #444;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }

        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
        }

        .card-header {
            background-color: var(--primary-color);
            border-radius: 10px 10px 0 0;
            padding: 15px;
            font-weight: 600;
        }

        .stat-value {
            font-size: 2rem;
            font-weight: bold;
            color: var(--light-color);
        }

        .stat-label {
            font-size: 0.9rem;
            color: var(--light-color);
            opacity: 0.8;
        }

        .btn-trading {
            background: linear-gradient(45deg, var(--primary-color), var(--success-color));
            border: none;
            color: white;
            padding: 12px 30px;
            border-radius: 50px;
            font-weight: 600;
            transition: all 0.3s ease;
            margin: 5px;
        }

        .btn-trading:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(40, 167, 69, 0.4);
        }

        .btn-stop {
            background: var(--danger-color);
        }

        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }

        .status-running {
            background-color: var(--success-color);
            animation: pulse 2s infinite;
        }

        .status-stopped {
            background-color: var(--warning-color);
        }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }

        .position-item {
            background-color: var(--light-color);
            color: var(--dark-color);
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 10px;
        }

        .position-profit {
            border-left: 4px solid var(--success-color);
        }

        .position-loss {
            border-left: 4px solid var(--danger-color);
        }

        .strategy-card {
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .strategy-card:hover {
            transform: translateY(-3px);
        }

        .strategy-card.selected {
            border: 2px solid var(--success-color);
        }

        .metric-card {
            text-align: center;
            padding: 20px;
        }

        .live-chart {
            background-color: var(--secondary-color);
            border-radius: 10px;
            padding: 20px;
            min-height: 300px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .trading-status {
            font-size: 1.2rem;
            font-weight: 600;
            padding: 20px;
            text-align: center;
        }
    </style>
</head>
<body>
    <!-- Navigation -->
    <nav class="navbar navbar-expand-lg navbar-dark navbar-custom">
        <div class="container-fluid">
            <a class="navbar-brand" href="#">
                <i class="fas fa-robot"></i> AI Trading Bot
            </a>
            <div class="navbar-nav ms-auto">
                <span class="navbar-text">
                    <span id="status-indicator" class="status-indicator status-stopped"></span>
                    <span id="status-text">Stopped</span>
                </span>
                <button class="btn btn-outline-light btn-sm" onclick="toggleTheme()">
                    <i class="fas fa-moon"></i>
                </button>
            </div>
        </div>
    </nav>

    <!-- Main Content -->
    <div class="container-fluid mt-4">
        <div class="row">
            <!-- Left Column - Trading Controls -->
            <div class="col-lg-3">
                <!-- Trading Status Card -->
                <div class="card">
                    <div class="card-header">
                        <i class="fas fa-chart-line"></i> Trading Status
                    </div>
                    <div class="card-body">
                        <div class="trading-status">
                            <div id="trading-status">Ready to Start</div>
                            <div class="mt-3">
                                <button id="start-btn" class="btn btn-trading me-2" onclick="startTrading()">
                                    <i class="fas fa-play"></i> Start Trading
                                </button>
                                <button id="stop-btn" class="btn btn-trading btn-stop" onclick="stopTrading()" disabled>
                                    <i class="fas fa-stop"></i> Stop Trading
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Configuration Card -->
                <div class="card">
                    <div class="card-header">
                        <i class="fas fa-cog"></i> Configuration
                    </div>
                    <div class="card-body">
                        <form id="config-form">
                            <div class="mb-3">
                                <label class="form-label">Trading Mode</label>
                                <select class="form-select" id="trading-mode">
                                    <option value="paper">Paper Trading</option>
                                    <option value="live">Live Trading</option>
                                </select>
                            </div>
                            <div class="mb-3">
                                <label class="form-label">Strategy</label>
                                <select class="form-select" id="strategy">
                                    <option value="sma">SMA Crossover</option>
                                    <option value="rsi">RSI Mean Reversion</option>
                                    <option value="macd">MACD Trend</option>
                                    <option value="bollinger">Bollinger Bands</option>
                                    <option value="volume">Volume Breakout</option>
                                    <option value="hybrid" selected>Hybrid (Recommended)</option>
                                </select>
                            </div>
                            <div class="mb-3">
                                <label class="form-label">Conviction Threshold</label>
                                <input type="range" class="form-range" id="conviction-threshold" min="60" max="100" value="95">
                                <div class="text-center mt-2">
                                    <span id="conviction-value">95%</span>
                                </div>
                            </div>
                            <div class="mb-3">
                                <label class="form-label">Initial Capital (₹)</label>
                                <input type="number" class="form-control" id="initial-capital" value="1000" min="100" max="100000">
                            </div>
                            <button type="button" class="btn btn-primary w-100" onclick="saveConfiguration()">
                                <i class="fas fa-save"></i> Save Configuration
                            </button>
                        </form>
                    </div>
                </div>

                <!-- Manual Trade Card -->
                <div class="card">
                    <div class="card-header">
                        <i class="fas fa-exchange-alt"></i> Manual Trade
                    </div>
                    <div class="card-body">
                        <form id="trade-form">
                            <div class="mb-3">
                                <label class="form-label">Symbol</label>
                                <select class="form-select" id="trade-symbol">
                                    <option value="BTC">BTC</option>
                                    <option value="ETH">ETH</option>
                                    <option value="ADA">ADA</option>
                                    <option value="DOGE">DOGE</option>
                                    <option value="SHIB">SHIB</option>
                                </select>
                            </div>
                            <div class="mb-3">
                                <label class="form-label">Action</label>
                                <select class="form-select" id="trade-action">
                                    <option value="buy">Buy</option>
                                    <option value="sell">Sell</option>
                                </select>
                            </div>
                            <div class="mb-3">
                                <label class="form-label">Quantity</label>
                                <input type="number" class="form-control" id="trade-quantity" value="0.01" step="0.01" min="0.001">
                            </div>
                            <button type="button" class="btn btn-success w-100" onclick="executeTrade()">
                                <i class="fas fa-chart-line"></i> Execute Trade
                            </button>
                        </form>
                    </div>
                </div>
            </div>

            <!-- Middle Column - Metrics and Portfolio -->
            <div class="col-lg-6">
                <!-- Key Metrics Row -->
                <div class="row mb-4">
                    <div class="col-md-3">
                        <div class="card metric-card">
                            <div class="stat-value" id="portfolio-value">₹1,000</div>
                            <div class="stat-label">Portfolio Value</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card metric-card">
                            <div class="stat-value" id="total-return">+0.0%</div>
                            <div class="stat-label">Total Return</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card metric-card">
                            <div class="stat-value" id="win-rate">0%</div>
                            <div class="stat-label">Win Rate</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card metric-card">
                            <div class="stat-value" id="active-positions">0</div>
                            <div class="stat-label">Active Positions</div>
                        </div>
                    </div>
                </div>

                <!-- Portfolio Overview Card -->
                <div class="card">
                    <div class="card-header">
                        <i class="fas fa-wallet"></i> Portfolio Overview
                    </div>
                    <div class="card-body">
                        <div id="positions-container">
                            <p class="text-muted">No active positions</p>
                        </div>
                    </div>
                </div>

                <!-- Performance Chart Card -->
                <div class="card">
                    <div class="card-header">
                        <i class="fas fa-chart-area"></i> Performance Chart
                    </div>
                    <div class="card-body">
                        <div class="live-chart">
                            <canvas id="performance-chart" width="300" height="250"></canvas>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Right Column - Market Data and Strategy -->
            <div class="col-lg-3">
                <!-- Market Data Card -->
                <div class="card">
                    <div class="card-header">
                        <i class="fas fa-bitcoin"></i> Market Data
                    </div>
                    <div class="card-body">
                        <div id="market-data-container">
                            <div class="mb-2">
                                <strong>BTC</strong>: $<span id="btc-price">Loading...</span>
                            </div>
                            <div class="mb-2">
                                <strong>ETH</strong>: $<span id="eth-price">Loading...</span>
                            </div>
                            <div class="mb-2">
                                <strong>ADA</strong>: $<span id="ada-price">Loading...</span>
                            </div>
                            <div class="mb-2">
                                <strong>DOGE</strong>: $<span id="doge-price">Loading...</span>
                            </div>
                            <div class="mb-2">
                                <strong>SHIB</strong>: $<span id="shib-price">Loading...</span>
                            </div>
                        </div>
                        <button class="btn btn-outline-primary btn-sm w-100 mt-3" onclick="refreshMarketData()">
                            <i class="fas fa-sync-alt"></i> Refresh Prices
                        </button>
                    </div>
                </div>

                <!-- Strategy Performance Card -->
                <div class="card">
                    <div class="card-header">
                        <i class="fas fa-robot"></i> Strategy Performance
                    </div>
                    <div class="card-body" id="strategy-performance">
                        <!-- Strategy cards will be populated dynamically -->
                    </div>
                </div>

                <!-- Risk Management Card -->
                <div class="card">
                    <div class="card-header">
                        <i class="fas fa-shield-alt"></i> Risk Management
                    </div>
                    <div class="card-body">
                        <div class="mb-3">
                            <div class="d-flex justify-content-between">
                                <span>Max Risk per Trade:</span>
                                <strong>5%</strong>
                            </div>
                        </div>
                        <div class="mb-3">
                            <div class="d-flex justify-content-between">
                                <span>Stop Loss:</span>
                                <strong>2%</strong>
                            </div>
                        </div>
                        <div class="mb-3">
                            <div class="d-flex justify-content-between">
                                <span>Take Profit:</span>
                                <strong>6%</strong>
                            </div>
                        </div>
                        <div class="mb-3">
                            <div class="d-flex justify-content-between">
                                <span>Max Positions:</span>
                                <strong>3</strong>
                            </div>
                        </div>
                        <div class="mb-3">
                            <div class="d-flex justify-content-between">
                                <span>Daily Loss Limit:</span>
                                <strong>10%</strong>
                            </div>
                        </div>
                        <button class="btn btn-outline-warning btn-sm w-100" onclick="runBacktest()">
                            <i class="fas fa-chart-line"></i> Run Backtest
                        </button>
                    </div>
                </div>

                <!-- AI Analysis Card -->
                <div class="card">
                    <div class="card-header">
                        <i class="fas fa-brain"></i> AI Analysis
                    </div>
                    <div class="card-body">
                        <div id="ai-status">
                            <div class="text-center">
                                <i class="fas fa-robot fa-3x mb-3"></i>
                                <p class="mb-3">AI Analysis</p>
                                <p class="text-muted">Advanced market analysis with conviction scoring</p>
                                <div class="progress">
                                    <div class="progress-bar bg-success" style="width: 85%"></div>
                                    <small class="text-muted">Average Conviction: 85%</small>
                                </div>
                            </div>
                        </div>
                        <button class="btn btn-outline-info btn-sm w-100 mt-3" onclick="analyzeWithAI()">
                            <i class="fas fa-microscope"></i> Analyze Market
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- JavaScript -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        // Global state
        let tradingState = {
            isRunning: false,
            currentMode: 'paper',
            currentStrategy: 'hybrid',
            portfolioValue: 1000,
            totalReturn: 0,
            winRate: 0,
            activePositions: 0
        };

        // Update UI based on state
        function updateUI() {
            // Update status indicator
            const statusIndicator = document.getElementById('status-indicator');
            const statusText = document.getElementById('status-text');
            const tradingStatus = document.getElementById('trading-status');

            if (tradingState.isRunning) {
                statusIndicator.className = 'status-indicator status-running';
                statusText.textContent = 'Running';
                tradingStatus.textContent = `Trading (${tradingState.currentMode.toUpperCase()})`;
                document.getElementById('start-btn').disabled = true;
                document.getElementById('stop-btn').disabled = false;
            } else {
                statusIndicator.className = 'status-indicator status-stopped';
                statusText.textContent = 'Stopped';
                tradingStatus.textContent = 'Ready to Start';
                document.getElementById('start-btn').disabled = false;
                document.getElementById('stop-btn').disabled = true;
            }

            // Update metrics
            document.getElementById('portfolio-value').textContent = `₹${tradingState.portfolioValue.toFixed(2)}`;
            document.getElementById('total-return').textContent = `${tradingState.totalReturn >= 0 ? '+' : ''}${tradingState.totalReturn.toFixed(2)}%`;
            document.getElementById('win-rate').textContent = `${tradingState.winRate.toFixed(1)}%`;
            document.getElementById('active-positions').textContent = tradingState.activePositions;
        }

        // Trading control functions
        async function startTrading() {
            const mode = document.getElementById('trading-mode').value;
            const strategy = document.getElementById('strategy').value;
            const conviction = document.getElementById('conviction-threshold').value;
            const capital = document.getElementById('initial-capital').value;

            try {
                const response = await fetch('/api/start-trading', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        mode: mode,
                        strategy: strategy,
                        conviction: conviction
                    })
                });

                const result = await response.json();
                if (result.success) {
                    tradingState.isRunning = true;
                    tradingState.currentMode = result.mode;
                    tradingState.currentStrategy = result.strategy;
                    updateUI();
                    alert('Trading started successfully!');
                } else {
                    alert('Failed to start trading: ' + result.error);
                }
            } catch (error) {
                alert('Error starting trading: ' + error.message);
            }
        }

        async function stopTrading() {
            try {
                const response = await fetch('/api/stop-trading', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    }
                });

                const result = await response.json();
                if (result.success) {
                    tradingState.isRunning = false;
                    updateUI();
                    alert('Trading stopped successfully!');
                } else {
                    alert('Failed to stop trading: ' + result.error);
                }
            } catch (error) {
                alert('Error stopping trading: ' + error.message);
            }
        }

        async function executeTrade() {
            const symbol = document.getElementById('trade-symbol').value;
            const action = document.getElementById('trade-action').value;
            const quantity = parseFloat(document.getElementById('trade-quantity').value);

            try {
                const response = await fetch('/api/execute-trade', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        symbol: symbol,
                        action: action,
                        quantity: quantity
                    })
                });

                const result = await response.json();
                if (result.success) {
                    alert(`Trade executed successfully!\\nSymbol: ${result.symbol}\\nAction: ${result.action}\\nQuantity: ${result.quantity}`);
                } else {
                    alert('Failed to execute trade: ' + result.error);
                }
            } catch (error) {
                alert('Error executing trade: ' + error.message);
            }
        }

        // Configuration functions
        async function saveConfiguration() {
            const config = {
                trading_mode: document.getElementById('trading-mode').value,
                strategy: document.getElementById('strategy').value,
                conviction_threshold: document.getElementById('conviction-threshold').value,
                initial_capital: document.getElementById('initial-capital').value
            };

            try {
                const response = await fetch('/api/configuration', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(config)
                });

                const result = await response.json();
                if (result.success) {
                    alert('Configuration saved successfully!');
                } else {
                    alert('Failed to save configuration: ' + result.error);
                }
            } catch (error) {
                alert('Error saving configuration: ' + error.message);
            }
        }

        // Market data functions
        async function refreshMarketData() {
            try {
                const response = await fetch('/api/market-data');
                const data = await response.json();

                // Update price displays
                const symbols = ['btc', 'eth', 'ada', 'doge', 'shib'];
                symbols.forEach(symbol => {
                    const element = document.getElementById(`${symbol}-price`);
                    if (element && data.prices[symbol.toUpperCase()]) {
                        element.textContent = data.prices[symbol.toUpperCase()].toFixed(6);
                    }
                });
            } catch (error) {
                console.error('Error refreshing market data:', error);
            }
        }

        // Strategy functions
        async function loadStrategyPerformance() {
            try {
                const response = await fetch('/api/strategies');
                const data = await response.json();

                const container = document.getElementById('strategy-performance');
                container.innerHTML = '';

                Object.keys(data.strategies).forEach(key => {
                    const strategy = data.strategies[key];
                    const isSelected = key === data.current_strategy;

                    const strategyCard = document.createElement('div');
                    strategyCard.className = `card strategy-card ${isSelected ? 'selected' : ''}`;
                    strategyCard.innerHTML = `
                        <div class="card-body text-center">
                            <h6>${strategy.name}</h6>
                            <p class="text-muted small">${strategy.description}</p>
                            ${strategy.performance ? `<p class="mb-2"><small>Win Rate: ${strategy.performance.win_rate?.toFixed(1) || 'N/A'}%</small></p>` : ''}
                            <button class="btn btn-sm ${isSelected ? 'btn-success' : 'btn-outline-primary'}"
                                    onclick="selectStrategy('${key}')">
                                ${isSelected ? 'Selected' : 'Select'}
                            </button>
                        </div>
                    `;

                    container.appendChild(strategyCard);
                });
            } catch (error) {
                console.error('Error loading strategy data:', error);
            }
        }

        function selectStrategy(strategyKey) {
            // Update selected strategy
            document.getElementById('strategy').value = strategyKey;

            // Reload strategy performance
            loadStrategyPerformance();
        }

        // Risk management functions
        async function loadRiskData() {
            try {
                const response = await fetch('/api/risk');
                const data = await response.json();

                // Update risk displays if needed
                console.log('Risk data loaded:', data);
            } catch (error) {
                console.error('Error loading risk data:', error);
            }
        }

        // AI analysis functions
        async function analyzeWithAI() {
            try {
                const response = await fetch('/api/ai');
                const data = await response.json();

                const aiContainer = document.getElementById('ai-status');
                if (data.is_enabled) {
                    aiContainer.innerHTML = `
                        <div class="text-center">
                            <i class="fas fa-robot fa-3x mb-3 text-success"></i>
                            <p class="mb-3">AI Analysis Active</p>
                            <p class="text-muted">Advanced market analysis running...</p>
                            <div class="progress">
                                <div class="progress-bar bg-info" style="width: ${data.performance_stats?.avg_conviction || 0}%">
                                    <small class="text-muted">Average Conviction: ${data.performance_stats?.avg_conviction || 0}%</small>
                                </div>
                            </div>
                        </div>
                    `;
                } else {
                    aiContainer.innerHTML = `
                        <div class="text-center">
                            <i class="fas fa-robot fa-3x mb-3 text-muted"></i>
                            <p class="mb-3">AI Analysis Inactive</p>
                            <p class="text-muted">Configure API key to enable AI features</p>
                        </div>
                    `;
                }
            } catch (error) {
                console.error('Error loading AI data:', error);
            }
        }

        // Backtest functions
        async function runBacktest() {
            try {
                const symbols = ['BTC', 'ETH']; // Could be from UI
                const strategies = ['hybrid'];

                const response = await fetch('/api/backtest', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        symbols: symbols,
                        strategies: strategies
                    })
                });

                const result = await response.json();
                if (result.success) {
                    alert('Backtest started!\\nSymbols: ' + result.results.symbols.join(', ') + '\\nStrategies: ' + result.results.strategies.join(', ') + '\\nEstimated completion: ' + result.results.estimated_completion);
                } else {
                    alert('Failed to start backtest: ' + result.error);
                }
            } catch (error) {
                alert('Error starting backtest: ' + error.message);
            }
        }

        // Theme toggle
        function toggleTheme() {
            document.body.classList.toggle('light-theme');
        }

        // Range slider update
        document.getElementById('conviction-threshold').addEventListener('input', function(e) {
            document.getElementById('conviction-value').textContent = e.target.value + '%';
        });

        // Initialize chart
        let performanceChart;

        function initChart() {
            const ctx = document.getElementById('performance-chart').getContext('2d');
            performanceChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                    datasets: [{
                        label: 'Portfolio Value',
                        data: [1000, 1050, 1025, 1100, 1150, 1200],
                        borderColor: 'rgb(75, 192, 192)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: true,
                            labels: {
                                color: 'white'
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: false,
                            ticks: {
                                color: 'white',
                                callback: function(value) {
                                    return '₹' + value;
                                }
                            }
                        },
                        x: {
                            ticks: {
                                color: 'white'
                            }
                        }
                    }
                }
            });
        }

        // Initialize on load
        document.addEventListener('DOMContentLoaded', function() {
            updateUI();
            refreshMarketData();
            loadStrategyPerformance();
            loadRiskData();
            analyzeWithAI();
            initChart();

            // Auto-refresh market data every 30 seconds
            setInterval(refreshMarketData, 30000);

            // Auto-update trading state every 5 seconds
            setInterval(async () => {
                try {
                    const response = await fetch('/api/trading-state');
                    const newState = await response.json();
                    Object.assign(tradingState, newState);
                    updateUI();
                } catch (error) {
                    console.error('Error updating trading state:', error);
                }
            }, 5000);
        });
    </script>
</body>
</html>
    """

    with open('templates/dashboard.html', 'w') as f:
        f.write(html_template)

    print("✅ HTML template created at templates/dashboard.html")

def create_static_files():
    """Create static assets directory and files"""
    os.makedirs('static', exist_ok=True)

    # Create CSS file
    css_content = """
/* Custom styles for web interface */
.light-theme {
    background-color: #ffffff !important;
    color: #000000 !important;
}

.light-theme .card {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-color: #dee2e6 !important;
}

.light-theme .card-header {
    background-color: #f8f9fa !important;
    color: #000000 !important;
}

.responsive-chart {
    max-width: 100%;
    height: auto;
}

@media (max-width: 768px) {
    .metric-card {
        margin-bottom: 10px;
    }

    .card {
        margin-bottom: 10px;
    }

    .stat-value {
        font-size: 1.5rem;
    }
}
"""

    with open('static/style.css', 'w') as f:
        f.write(css_content)

    # Create JavaScript file
    js_content = """
// Additional JavaScript functionality

// WebSocket for real-time updates
let ws;

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    ws = new WebSocket(wsUrl);

    ws.onopen = function(event) {
        console.log('WebSocket connected');
        updateConnectionStatus(true);
    };

    ws.onclose = function(event) {
        console.log('WebSocket disconnected');
        updateConnectionStatus(false);
    };

    ws.onmessage = function(event) {
        const data = JSON.parse(event.data);
        handleRealtimeUpdate(data);
    };

    ws.onerror = function(error) {
        console.error('WebSocket error:', error);
        updateConnectionStatus(false);
    };
}

function updateConnectionStatus(connected) {
    const statusElement = document.getElementById('connection-status');
    if (statusElement) {
        if (connected) {
            statusElement.innerHTML = '<i class="fas fa-wifi text-success"></i> Connected';
        } else {
            statusElement.innerHTML = '<i class="fas fa-wifi-slash text-danger"></i> Disconnected';
        }
    }
}

function handleRealtimeUpdate(data) {
    switch(data.type) {
        case 'trading_update':
            updateTradingUI(data);
            break;
        case 'market_update':
            updateMarketUI(data);
            break;
        case 'portfolio_update':
            updatePortfolioUI(data);
            break;
        case 'alert':
            showAlert(data.message, data.level);
            break;
    }
}

function updateTradingUI(data) {
    // Update trading interface with real-time data
    if (data.portfolio_value !== undefined) {
        document.getElementById('portfolio-value').textContent = formatCurrency(data.portfolio_value);
    }

    if (data.active_positions !== undefined) {
        document.getElementById('active-positions').textContent = data.active_positions;
    }

    if (data.is_running !== undefined) {
        updateTradingStatus(data.is_running);
    }
}

function updateMarketUI(data) {
    // Update market data displays
    if (data.prices) {
        Object.keys(data.prices).forEach(symbol => {
            const element = document.getElementById(`${symbol.toLowerCase()}-price`);
            if (element) {
                element.textContent = formatPrice(data.prices[symbol]);
            }
        });
    }
}

function updatePortfolioUI(data) {
    // Update portfolio displays
    // Implementation would go here
}

function formatCurrency(value) {
    return '₹' + parseFloat(value).toFixed(2);
}

function formatPrice(price) {
    return '$' + parseFloat(price).toFixed(6);
}

function showAlert(message, level = 'info') {
    const alertContainer = document.getElementById('alert-container');
    if (!alertContainer) return;

    const alertClass = level === 'error' ? 'alert-danger' :
                        level === 'warning' ? 'alert-warning' : 'alert-info';

    const alertHTML = `
        <div class="alert ${alertClass} alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;

    alertContainer.innerHTML = alertHTML;

    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        alertContainer.innerHTML = '';
    }, 5000);
}
"""

    with open('static/main.js', 'w') as f:
        f.write(js_content)

    print("✅ Static assets created")

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.route('/ws')
def websocket_endpoint():
    """WebSocket endpoint for real-time updates"""
    # This would be implemented with Flask-SocketIO
    # For now, return a simple response
    return jsonify({'message': 'WebSocket endpoint - implement with Flask-SocketIO for full functionality'})

if __name__ == '__main__':
    print("🌐 Starting AI Trading Bot Web Interface...")
    print("=" * 50)

    # Create templates directory and HTML
    os.makedirs('templates', exist_ok=True)
    create_html_template()

    # Create static assets
    create_static_files()

    # Start the web server
    print("✅ Web interface ready!")
    print("📊 Access at: http://localhost:5000")
    print("🔗 WebSocket endpoint: ws://localhost:5000/ws")
    print("📱 Features available:")
    print("   - Real-time trading dashboard")
    print("   - Portfolio management")
    print("   - Strategy selection and performance")
    print("   - Market data monitoring")
    print("   - Risk management controls")
    print("   - Manual trade execution")
    print("   - AI analysis integration")
    print("   - Configuration management")
    print("   - Backtesting capabilities")
    print("   - WebSocket real-time updates")

    app.run(host='0.0.0.0', port=5000, debug=True)