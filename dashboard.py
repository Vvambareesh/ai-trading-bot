import os
import time
import json
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from threading import Thread
import queue

# Import our custom modules
from paper_trading import PaperTradingPortfolio
from strategies import StrategyManager
from data_sources import MarketDataManager
from risk_manager import RiskManager
from ai_analyzer import AIAnalyzer
from backtester import Backtester

class TradingDashboard:
    """
    Real-time trading dashboard for monitoring AI trading bot performance
    Provides comprehensive visualization of portfolio, strategies, and risk metrics
    """

    def __init__(self, portfolio: PaperTradingPortfolio = None):
        self.portfolio = portfolio or PaperTradingPortfolio(1000)
        self.data_manager = MarketDataManager()
        self.strategy_manager = StrategyManager()
        self.risk_manager = RiskManager(self.portfolio.initial_capital)

        # Data queue for real-time updates
        self.update_queue = queue.Queue()
        self.performance_history = []
        self.max_history_points = 500

        # Initialize Dash app
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
        self.app.title = "AI Trading Bot Dashboard"

        # Setup layout and callbacks
        self.setup_layout()
        self.setup_callbacks()

        # Start background update thread
        self.running = True
        self.update_thread = Thread(target=self.background_updater, daemon=True)
        self.update_thread.start()

    def setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("🤖 AI Trading Bot Dashboard", className="text-center mb-4"),
                    html.Hr()
                ])
            ]),

            # Key Metrics Row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("₹0", id="portfolio-value", className="text-success"),
                            html.P("Portfolio Value", className="card-text")
                        ])
                    ], color="dark", outline=True)
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("0%", id="total-return", className="text-info"),
                            html.P("Total Return", className="card-text")
                        ])
                    ], color="dark", outline=True)
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("0%", id="win-rate", className="text-warning"),
                            html.P("Win Rate", className="card-text")
                        ])
                    ], color="dark", outline=True)
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("0", id="active-positions", className="text-primary"),
                            html.P("Active Positions", className="card-text")
                        ])
                    ], color="dark", outline=True)
                ], width=3)
            ], className="mb-4"),

            # Charts Row
            dbc.Row([
                # Portfolio Performance Chart
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Portfolio Performance"),
                        dbc.CardBody([
                            dcc.Graph(id="portfolio-chart", config={'displayModeBar': False})
                        ])
                    ])
                ], width=8),

                # Risk Metrics
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Risk Metrics"),
                        dbc.CardBody([
                            html.Div([
                                html.P("Max Drawdown:", className="mb-2"),
                                html.H5("0%", id="max-drawdown", className="text-danger mb-3"),

                                html.P("Sharpe Ratio:", className="mb-2"),
                                html.H5("0.0", id="sharpe-ratio", className="text-info mb-3"),

                                html.P("Daily P&L:", className="mb-2"),
                                html.H5("₹0", id="daily-pnl", className="text-success")
                            ])
                        ])
                    ])
                ], width=4)
            ], className="mb-4"),

            # Second Row Charts
            dbc.Row([
                # Asset Allocation
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Asset Allocation"),
                        dbc.CardBody([
                            dcc.Graph(id="allocation-chart", config={'displayModeBar': False})
                        ])
                    ])
                ], width=6),

                # Recent Trades
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Recent Trades"),
                        dbc.CardBody([
                            html.Div(id="recent-trades")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),

            # Strategy Performance
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Strategy Performance"),
                        dbc.CardBody([
                            dcc.Graph(id="strategy-chart", config={'displayModeBar': False})
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),

            # Control Panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Control Panel"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Select Strategy:"),
                                    dcc.Dropdown(
                                        id='strategy-dropdown',
                                        options=[
                                            {'label': 'SMA Crossover', 'value': 'sma'},
                                            {'label': 'RSI Mean Reversion', 'value': 'rsi'},
                                            {'label': 'MACD Trend', 'value': 'macd'},
                                            {'label': 'Bollinger Bands', 'value': 'bollinger'},
                                            {'label': 'Volume Breakout', 'value': 'volume'},
                                            {'label': 'Hybrid (Recommended)', 'value': 'hybrid'}
                                        ],
                                        value='hybrid',
                                        className="mb-3"
                                    )
                                ], width=4),
                                dbc.Col([
                                    html.Label("Auto-refresh (seconds):"),
                                    dcc.Dropdown(
                                        id='refresh-dropdown',
                                        options=[
                                            {'label': '5 seconds', 'value': 5},
                                            {'label': '10 seconds', 'value': 10},
                                            {'label': '30 seconds', 'value': 30},
                                            {'label': '1 minute', 'value': 60},
                                            {'label': 'Off', 'value': 0}
                                        ],
                                        value=10,
                                        className="mb-3"
                                    )
                                ], width=4),
                                dbc.Col([
                                    html.Label("Actions:"),
                                    html.Div([
                                        dbc.Button("🔄 Refresh", id="refresh-btn", color="primary", size="sm", className="me-2"),
                                        dbc.Button("📊 Backtest", id="backtest-btn", color="success", size="sm", className="me-2"),
                                        dbc.Button("💾 Save Report", id="save-btn", color="info", size="sm")
                                    ])
                                ], width=4)
                            ])
                        ])
                    ])
                ], width=12)
            ]),

            # Auto-refresh component
            dcc.Interval(
                id='interval-component',
                interval=10*1000,  # 10 seconds default
                n_intervals=0
            ),

            # Store for data
            dcc.Store(id='dashboard-data')

        ], fluid=True, className="p-4")

    def setup_callbacks(self):
        """Setup dashboard callbacks"""

        @self.app.callback(
            [Output('portfolio-value', 'children'),
             Output('total-return', 'children'),
             Output('win-rate', 'children'),
             Output('active-positions', 'children'),
             Output('max-drawdown', 'children'),
             Output('sharpe-ratio', 'children'),
             Output('daily-pnl', 'children'),
             Output('portfolio-chart', 'figure'),
             Output('allocation-chart', 'figure'),
             Output('strategy-chart', 'figure'),
             Output('recent-trades', 'children')],
            [Input('interval-component', 'n_intervals'),
             Input('refresh-btn', 'n_clicks'),
             Input('strategy-dropdown', 'value')]
        )
        def update_dashboard(n_intervals, n_clicks, selected_strategy):
            """Update dashboard with current data"""
            try:
                # Get portfolio summary
                summary = self.portfolio.get_portfolio_summary()

                # Update key metrics
                portfolio_value = f"₹{summary['total_value']:.2f}"
                total_return = f"{summary['total_return_pct']:+.2f}%"
                win_rate = f"{summary['win_rate_pct']:.1f}%"
                active_positions = str(summary['active_positions'])
                max_drawdown = f"{summary['max_drawdown_pct']:.2f}%"
                sharpe_ratio = f"{summary.get('sharpe_ratio', 0):.2f}"
                daily_pnl = f"₹{summary['daily_pnl']:+.2f}"

                # Update performance history
                current_time = datetime.now()
                self.performance_history.append({
                    'timestamp': current_time,
                    'portfolio_value': summary['total_value'],
                    'return_pct': summary['total_return_pct']
                })

                # Keep only recent history
                if len(self.performance_history) > self.max_history_points:
                    self.performance_history = self.performance_history[-self.max_history_points:]

                # Portfolio performance chart
                portfolio_fig = self.create_portfolio_chart()

                # Asset allocation chart
                allocation_fig = self.create_allocation_chart()

                # Strategy performance chart
                strategy_fig = self.create_strategy_chart(selected_strategy)

                # Recent trades
                recent_trades_html = self.create_recent_trades_html()

                return (portfolio_value, total_return, win_rate, active_positions,
                       max_drawdown, sharpe_ratio, daily_pnl,
                       portfolio_fig, allocation_fig, strategy_fig,
                       recent_trades_html)

            except Exception as e:
                print(f"Error updating dashboard: {e}")
                # Return default values on error
                empty_fig = go.Figure()
                return ("₹0", "0%", "0%", "0", "0%", "0.0", "₹0",
                       empty_fig, empty_fig, empty_fig, "<p>No data available</p>")

        @self.app.callback(
            Output('interval-component', 'interval'),
            [Input('refresh-dropdown', 'value')]
        )
        def update_refresh_interval(refresh_rate):
            """Update auto-refresh interval"""
            if refresh_rate == 0:
                return 10**9  # Very large number to effectively disable
            return refresh_rate * 1000  # Convert to milliseconds

        @self.app.callback(
            Output("backtest-output", "children"),
            [Input("backtest-btn", "n_clicks")],
            prevent_initial_call=True
        )
        def run_backtest(n_clicks):
            """Run backtest and show results"""
            if n_clicks:
                try:
                    # This is a simplified backtest - in production would be more comprehensive
                    backtester = Backtester(self.portfolio.initial_capital)

                    # Run quick backtest
                    results = backtester.run_comprehensive_backtest(
                        ['BTC', 'ETH', 'ADA'],
                        ['hybrid']
                    )

                    if results:
                        best_result = max(results.values(), key=lambda x: x.total_return_pct)

                        return dbc.Alert([
                            html.H5("🧪 Backtest Results"),
                            html.P(f"Best Return: {best_result.total_return_pct:+.2f}%"),
                            html.P(f"Win Rate: {best_result.win_rate_pct:.1f}%"),
                            html.P(f"Sharpe Ratio: {best_result.sharpe_ratio:.2f}")
                        ], color="success")
                    else:
                        return dbc.Alert("No backtest results available", color="warning")

                except Exception as e:
                    return dbc.Alert(f"Backtest error: {str(e)}", color="danger")

            return ""

    def create_portfolio_chart(self):
        """Create portfolio performance chart"""
        if not self.performance_history:
            return go.Figure()

        df = pd.DataFrame(self.performance_history)

        fig = go.Figure()

        # Portfolio value line
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='green', width=2)
        ))

        # Add zero line for reference
        fig.add_hline(
            y=self.portfolio.initial_capital,
            line_dash="dash",
            line_color="white",
            annotation_text="Initial Capital",
            annotation_position="bottom right"
        )

        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Time",
            yaxis_title="Portfolio Value (₹)",
            template="plotly_dark",
            height=300,
            margin=dict(l=0, r=0, t=30, b=0)
        )

        return fig

    def create_allocation_chart(self):
        """Create asset allocation pie chart"""
        summary = self.portfolio.get_portfolio_summary()

        if not summary['positions']:
            # No positions, show cash only
            fig = go.Figure(data=[go.Pie(
                labels=['Cash'],
                values=[summary['available_capital']],
                hole=0.3
            )])
        else:
            # Calculate allocation for each position
            labels = []
            values = []

            # Add cash
            if summary['available_capital'] > 0:
                labels.append('Cash')
                values.append(summary['available_capital'])

            # Add positions
            for symbol, position in summary['positions'].items():
                value = position['quantity'] * position['entry_price']
                labels.append(symbol)
                values.append(value)

            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.3
            )])

        fig.update_layout(
            title="Asset Allocation",
            template="plotly_dark",
            height=250,
            margin=dict(l=0, r=0, t=30, b=0)
        )

        return fig

    def create_strategy_chart(self, strategy_name):
        """Create strategy performance comparison chart"""
        try:
            # Get strategy performance from strategy manager
            performance = self.strategy_manager.get_strategy_performance()

            if not performance:
                return go.Figure()

            # Prepare data for visualization
            strategies = list(performance.keys())
            win_rates = [performance[s]['win_rate_pct'] for s in strategies]
            avg_convictions = [performance[s]['avg_conviction'] for s in strategies]
            total_signals = [performance[s]['total_signals'] for s in strategies]

            fig = go.Figure()

            # Win rate bars
            fig.add_trace(go.Bar(
                x=strategies,
                y=win_rates,
                name='Win Rate (%)',
                marker_color='lightblue'
            ))

            # Highlight selected strategy
            if strategy_name in strategies:
                idx = strategies.index(strategy_name)
                fig.update_traces(
                    selector={'x': [strategy_name]},
                    marker_color='orange'
                )

            fig.update_layout(
                title=f"Strategy Performance (Current: {strategy_name.upper()})",
                xaxis_title="Strategy",
                yaxis_title="Win Rate (%)",
                template="plotly_dark",
                height=250,
                margin=dict(l=0, r=0, t=30, b=0)
            )

            return fig

        except Exception as e:
            print(f"Error creating strategy chart: {e}")
            return go.Figure()

    def create_recent_trades_html(self):
        """Create HTML for recent trades"""
        recent_trades = self.portfolio.trades[-10:]  # Last 10 trades

        if not recent_trades:
            return html.P("No trades yet", className="text-muted")

        trades_html = []
        for trade in reversed(recent_trades):
            if trade.pnl is not None:
                pnl_class = "text-success" if trade.pnl > 0 else "text-danger"
                pnl_text = f"₹{trade.pnl:+.2f}"
            else:
                pnl_class = "text-warning"
                pnl_text = "Open"

            trades_html.append(
                html.Div([
                    html.Div([
                        html.Span(f"{trade.symbol} {trade.side}", className="fw-bold"),
                        html.Span(f" | {trade.entry_time.strftime('%H:%M')}", className="text-muted ms-2")
                    ], className="d-flex justify-content-between"),
                    html.Div([
                        html.Span(pnl_text, className=pnl_class),
                        html.Span(f"{trade.conviction:.0f}%", className="text-muted ms-2")
                    ], className="d-flex justify-content-between")
                ], className="mb-2 p-2 border rounded")
            )

        return html.Div(trades_html)

    def background_updater(self):
        """Background thread for updating data"""
        while self.running:
            try:
                # Simulate some trading activity (in production, this would be real data)
                time.sleep(5)

                # Here you would fetch real-time data and update portfolio
                # For now, we'll just update the queue with current portfolio state
                summary = self.portfolio.get_portfolio_summary()

                # This would be replaced with actual real-time updates
                # self.update_queue.put(summary)

            except Exception as e:
                print(f"Background updater error: {e}")
                time.sleep(10)

    def run(self, host='127.0.0.1', port=8050, debug=False):
        """Run the dashboard"""
        print(f"🌐 Starting Trading Dashboard at http://{host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)

    def stop(self):
        """Stop the dashboard"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)

if __name__ == "__main__":
    # Create sample portfolio for demonstration
    portfolio = PaperTradingPortfolio(1000)

    # Add some sample trades for visualization
    portfolio.open_position('BTC', 'BUY', 45000, 85, 'hybrid')
    portfolio.open_position('ETH', 'BUY', 3000, 75, 'hybrid')

    # Create and run dashboard
    dashboard = TradingDashboard(portfolio)

    try:
        dashboard.run(debug=True)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
        dashboard.stop()