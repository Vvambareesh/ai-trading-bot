import os
import time
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import ccxt
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
from abc import ABC, abstractmethod

@dataclass
class MarketData:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str
    quality_score: float  # 0-1 data quality rating

class DataSource(ABC):
    """Abstract base class for market data sources"""

    @abstractmethod
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        pass

    @abstractmethod
    def get_historical_data(self, symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Get historical OHLCV data"""
        pass

    @abstractmethod
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols"""
        pass

class CoinGeckoDataSource(DataSource):
    """
    CoinGecko API data source (free tier)
    Primary data source for cryptocurrency market data
    """

    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.rate_limit = 50  # calls per minute (free tier)
        self.last_call_time = 0
        self.call_count = 0

        # Symbol mapping (CoinGecko IDs)
        self.symbol_map = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'ADA': 'cardano',
            'DOGE': 'dogecoin',
            'SHIB': 'shiba-inu'
        }

    def _rate_limit_check(self):
        """Check and enforce rate limiting"""
        current_time = time.time()
        if current_time - self.last_call_time < 60:  # Within same minute
            if self.call_count >= self.rate_limit:
                sleep_time = 60 - (current_time - self.last_call_time)
                print(f"⏳ CoinGecko rate limit reached. Waiting {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
                self.call_count = 0
                self.last_call_time = time.time()
        else:
            # New minute
            self.call_count = 0
            self.last_call_time = current_time

        self.call_count += 1

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from CoinGecko"""
        try:
            self._rate_limit_check()

            if symbol not in self.symbol_map:
                print(f"❌ Symbol {symbol} not supported by CoinGecko")
                return None

            coin_id = self.symbol_map[symbol]
            url = f"{self.base_url}/simple/price"
            params = {
                'ids': coin_id,
                'vs_currencies': 'usd',
                'include_24hr_change': 'true',
                'include_24hr_vol': 'true'
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            price = data[coin_id]['usd']

            return float(price)

        except requests.exceptions.RequestException as e:
            print(f"❌ CoinGecko API error for {symbol}: {e}")
            return None
        except KeyError as e:
            print(f"❌ Unexpected CoinGecko response format for {symbol}: {e}")
            return None

    def get_historical_data(self, symbol: str, period: str = '30d', interval: str = 'daily') -> Optional[pd.DataFrame]:
        """Get historical data from CoinGecko"""
        try:
            self._rate_limit_check()

            if symbol not in self.symbol_map:
                print(f"❌ Symbol {symbol} not supported by CoinGecko")
                return None

            coin_id = self.symbol_map[symbol]

            # Convert period to days
            period_map = {
                '1d': 1, '7d': 7, '30d': 30, '90d': 90, '180d': 180, '1y': 365
            }
            days = period_map.get(period, 30)

            url = f"{self.base_url}/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': interval
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Convert to DataFrame
            prices = data['prices']
            volumes = data['total_volumes']

            df = pd.DataFrame(prices, columns=['timestamp', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Add volume data
            vol_df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
            vol_df['timestamp'] = pd.to_datetime(vol_df['timestamp'], unit='ms')
            vol_df.set_index('timestamp', inplace=True)

            df = df.join(vol_df)

            # Calculate OHLC from close prices (approximation)
            df['open'] = df['close'].shift(1)
            df['high'] = df['close'] * 1.02  # Approximate 2% intraday high
            df['low'] = df['close'] * 0.98   # Approximate 2% intraday low

            # Remove first row (no open price)
            df = df.iloc[1:]

            # Fill missing values
            df.fillna(method='ffill', inplace=True)

            return df[['open', 'high', 'low', 'close', 'volume']]

        except requests.exceptions.RequestException as e:
            print(f"❌ CoinGecko historical data error for {symbol}: {e}")
            return None
        except Exception as e:
            print(f"❌ Error processing CoinGecko data for {symbol}: {e}")
            return None

    def get_supported_symbols(self) -> List[str]:
        """Get supported symbols"""
        return list(self.symbol_map.keys())

class YFinanceDataSource(DataSource):
    """
    Yahoo Finance data source
    Backup data source for cryptocurrency market data
    """

    def __init__(self):
        # Yahoo Finance symbol mapping
        self.symbol_map = {
            'BTC': 'BTC-USD',
            'ETH': 'ETH-USD',
            'ADA': 'ADA-USD',
            'DOGE': 'DOGE-USD',
            'SHIB': 'SHIB-USD'
        }

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from Yahoo Finance"""
        try:
            if symbol not in self.symbol_map:
                print(f"❌ Symbol {symbol} not supported by Yahoo Finance")
                return None

            yf_symbol = self.symbol_map[symbol]
            ticker = yf.Ticker(yf_symbol)
            data = ticker.history(period='1d', interval='1m')

            if data.empty:
                return None

            return float(data['Close'].iloc[-1])

        except Exception as e:
            print(f"❌ Yahoo Finance error for {symbol}: {e}")
            return None

    def get_historical_data(self, symbol: str, period: str = '30d', interval: str = '1d') -> Optional[pd.DataFrame]:
        """Get historical data from Yahoo Finance"""
        try:
            if symbol not in self.symbol_map:
                print(f"❌ Symbol {symbol} not supported by Yahoo Finance")
                return None

            yf_symbol = self.symbol_map[symbol]
            ticker = yf.Ticker(yf_symbol)

            # Map interval for yfinance
            interval_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '1h': '1h',
                '1d': '1d', '5d': '5d', '1wk': '1wk', '1mo': '1mo'
            }
            yf_interval = interval_map.get(interval, '1d')

            # Map period for yfinance
            period_map = {
                '1d': '1d', '5d': '5d', '1mo': '1mo', '3mo': '3mo',
                '6mo': '6mo', '1y': '1y', '2y': '2y', '5y': '5y'
            }
            yf_period = period_map.get(period, '1mo')

            data = ticker.history(period=yf_period, interval=yf_interval)

            if data.empty:
                return None

            # Rename columns to match standard format
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]

            return data[['open', 'high', 'low', 'close', 'volume']]

        except Exception as e:
            print(f"❌ Yahoo Finance historical data error for {symbol}: {e}")
            return None

    def get_supported_symbols(self) -> List[str]:
        """Get supported symbols"""
        return list(self.symbol_map.keys())

class CCXTDataSource(DataSource):
    """
    CCXT data source for multiple exchanges
    Provides real-time data from various cryptocurrency exchanges
    """

    def __init__(self, exchange_name: str = 'binance'):
        self.exchange_name = exchange_name
        try:
            self.exchange = getattr(ccxt, exchange_name)()
            print(f"✅ Connected to {exchange_name} exchange")
        except AttributeError:
            print(f"❌ Exchange {exchange_name} not available in CCXT")
            self.exchange = None

        # Standardize symbol format for CCXT
        self.symbol_map = {
            'BTC': 'BTC/USDT',
            'ETH': 'ETH/USDT',
            'ADA': 'ADA/USDT',
            'DOGE': 'DOGE/USDT',
            'SHIB': 'SHIB/USDT'
        }

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from CCXT exchange"""
        try:
            if not self.exchange or symbol not in self.symbol_map:
                print(f"❌ Symbol {symbol} not supported by {self.exchange_name}")
                return None

            ccxt_symbol = self.symbol_map[symbol]
            ticker = self.exchange.fetch_ticker(ccxt_symbol)

            return float(ticker['last'])

        except Exception as e:
            print(f"❌ {self.exchange_name} error for {symbol}: {e}")
            return None

    def get_historical_data(self, symbol: str, period: str = '30d', interval: str = '1h') -> Optional[pd.DataFrame]:
        """Get historical data from CCXT exchange"""
        try:
            if not self.exchange or symbol not in self.symbol_map:
                print(f"❌ Symbol {symbol} not supported by {self.exchange_name}")
                return None

            ccxt_symbol = self.symbol_map[symbol]

            # Map period and interval to CCXT format
            since = None
            limit = None

            # Convert period to limit
            period_map = {
                '1d': (1440, '1h'),    # 1440 hours = 60 days of hourly data
                '7d': (168, '1h'),     # 168 hours = 7 days of hourly data
                '30d': (720, '1h'),    # 720 hours = 30 days of hourly data
                '90d': (2160, '1h'),   # 2160 hours = 90 days of hourly data
            }

            # Map interval to timeframe
            interval_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '1h': '1h',
                '4h': '4h', '1d': '1d'
            }

            timeframe = interval_map.get(interval, '1h')

            if period in period_map:
                limit, _ = period_map[period]

            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(ccxt_symbol, timeframe, since, limit)

            if not ohlcv:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            return df

        except Exception as e:
            print(f"❌ {self.exchange_name} historical data error for {symbol}: {e}")
            return None

    def get_supported_symbols(self) -> List[str]:
        """Get supported symbols"""
        return list(self.symbol_map.keys()) if self.exchange else []

class MarketDataManager:
    """
    Manages multiple data sources with failover and data quality assessment
    """

    def __init__(self):
        self.data_sources = []

        # Initialize data sources (in order of preference)
        try:
            self.data_sources.append(CoinGeckoDataSource())
            print("✅ CoinGecko data source initialized")
        except Exception as e:
            print(f"⚠️ Failed to initialize CoinGecko: {e}")

        try:
            self.data_sources.append(YFinanceDataSource())
            print("✅ Yahoo Finance data source initialized")
        except Exception as e:
            print(f"⚠️ Failed to initialize Yahoo Finance: {e}")

        try:
            self.data_sources.append(CCXTDataSource('binance'))
            print("✅ Binance data source initialized")
        except Exception as e:
            print(f"⚠️ Failed to initialize Binance: {e}")

        # Cache for data quality and performance
        self.data_quality_cache = {}
        self.source_performance = {source.__class__.__name__: {'success': 0, 'failure': 0}
                                 for source in self.data_sources}

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price with automatic failover"""
        for source in self.data_sources:
            try:
                price = source.get_current_price(symbol)
                if price is not None and price > 0:
                    self.source_performance[source.__class__.__name__]['success'] += 1
                    return price
                else:
                    self.source_performance[source.__class__.__name__]['failure'] += 1
            except Exception as e:
                print(f"⚠️ {source.__class__.__name__} failed for {symbol}: {e}")
                self.source_performance[source.__class__.__name__]['failure'] += 1

        print(f"❌ All data sources failed for {symbol}")
        return None

    def get_historical_data(self, symbol: str, period: str = '30d', interval: str = '1d') -> Optional[pd.DataFrame]:
        """Get historical data with automatic failover"""
        for source in self.data_sources:
            try:
                data = source.get_historical_data(symbol, period, interval)
                if data is not None and not data.empty:
                    # Validate data quality
                    quality_score = self._assess_data_quality(data)
                    if quality_score > 0.7:  # Acceptable quality threshold
                        self.source_performance[source.__class__.__name__]['success'] += 1
                        return data
                    else:
                        print(f"⚠️ {source.__class__.__name__} data quality too low for {symbol}: {quality_score:.2f}")
                        self.source_performance[source.__class__.__name__]['failure'] += 1
                else:
                    self.source_performance[source.__class__.__name__]['failure'] += 1
            except Exception as e:
                print(f"⚠️ {source.__class__.__name__} failed for {symbol}: {e}")
                self.source_performance[source.__class__.__name__]['failure'] += 1

        print(f"❌ All data sources failed for {symbol} historical data")
        return None

    def _assess_data_quality(self, data: pd.DataFrame) -> float:
        """Assess the quality of market data"""
        if data.empty:
            return 0.0

        quality_score = 1.0

        # Check for missing values
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        quality_score -= missing_ratio * 0.3

        # Check for outliers (price changes > 20% in single period)
        if 'close' in data.columns:
            price_changes = data['close'].pct_change().dropna()
            outlier_ratio = (abs(price_changes) > 0.2).sum() / len(price_changes)
            quality_score -= outlier_ratio * 0.2

        # Check data recency (should be recent)
        last_timestamp = data.index[-1] if len(data) > 0 else None
        if last_timestamp:
            age_hours = (datetime.now() - last_timestamp).total_seconds() / 3600
            if age_hours > 24:  # Data older than 24 hours
                quality_score -= min(0.5, age_hours / 48)

        # Check minimum data length
        if len(data) < 10:
            quality_score -= 0.3

        return max(0.0, min(1.0, quality_score))

    def get_multi_symbol_data(self, symbols: List[str], period: str = '30d', interval: str = '1d') -> Dict[str, pd.DataFrame]:
        """Get historical data for multiple symbols"""
        data_dict = {}

        for symbol in symbols:
            data = self.get_historical_data(symbol, period, interval)
            if data is not None:
                data_dict[symbol] = data

        return data_dict

    def get_source_performance_report(self) -> Dict:
        """Get performance report for data sources"""
        report = {}

        for source_name, stats in self.source_performance.items():
            total = stats['success'] + stats['failure']
            success_rate = (stats['success'] / total * 100) if total > 0 else 0
            report[source_name] = {
                'success_count': stats['success'],
                'failure_count': stats['failure'],
                'success_rate_pct': success_rate,
                'total_requests': total
            }

        return report

    def get_real_time_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for multiple symbols"""
        prices = {}

        for symbol in symbols:
            price = self.get_current_price(symbol)
            if price is not None:
                prices[symbol] = price

        return prices

    def validate_data_completeness(self, data: pd.DataFrame, expected_periods: int) -> bool:
        """Validate if data has expected number of periods"""
        if data.empty:
            return False

        # Allow for some missing periods (5% tolerance)
        min_periods = expected_periods * 0.95
        return len(data) >= min_periods

    def resample_data(self, data: pd.DataFrame, target_interval: str) -> pd.DataFrame:
        """Resample data to different interval"""
        if data.empty:
            return data

        # Define resampling rules
        resample_rules = {
            '1h': '1H',
            '4h': '4H',
            '1d': '1D',
            '1w': '1W'
        }

        rule = resample_rules.get(target_interval, '1D')

        ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }

        # Filter columns that exist in data
        available_ohlc = {col: rule for col, rule in ohlc_dict.items() if col in data.columns}

        if not available_ohlc:
            return data

        resampled = data.resample(rule).agg(available_ohlc).dropna()

        return resampled

if __name__ == "__main__":
    # Example usage
    print("📊 Testing Market Data Manager...")

    manager = MarketDataManager()

    # Test getting current price
    symbols = ['BTC', 'ETH', 'ADA']
    print(f"\n💰 Current Prices:")
    for symbol in symbols:
        price = manager.get_current_price(symbol)
        if price:
            print(f"   {symbol}: ${price:.2f}")

    # Test getting historical data
    print(f"\n📈 Historical Data Test:")
    for symbol in ['BTC', 'ETH']:
        data = manager.get_historical_data(symbol, period='7d', interval='1h')
        if data is not None:
            print(f"   {symbol}: {len(data)} records")
            print(f"   Latest price: ${data['close'].iloc[-1]:.2f}")
            print(f"   Data quality: {manager._assess_data_quality(data):.2f}")

    # Get performance report
    print(f"\n📊 Data Source Performance:")
    performance = manager.get_source_performance_report()
    for source, stats in performance.items():
        print(f"   {source}: {stats['success_rate_pct']:.1f}% success rate ({stats['total_requests']} requests)")

    # Test multi-symbol data
    print(f"\n🔄 Multi-Symbol Data Test:")
    multi_data = manager.get_multi_symbol_data(symbols, period='3d', interval='4h')
    for symbol, data in multi_data.items():
        print(f"   {symbol}: {len(data)} records")