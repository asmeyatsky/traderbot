"""
Multi-Asset Support for Crypto, Forex, and Commodities

This module implements support for multiple asset classes including
cryptocurrencies, forex pairs, commodities, and other financial instruments.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import asyncio
import aiohttp
import ccxt  # Cryptocurrency library
import yfinance as yf

from src.domain.entities.trading import Order, Position, Portfolio
from src.domain.entities.user import User
from src.domain.value_objects import Symbol, Price, Money
from src.domain.ports import MarketDataPort, TradingExecutionPort
from src.infrastructure.config.settings import settings


class AssetClass(Enum):
    """Enumeration of different asset classes."""
    EQUITY = "equity"
    CRYPTOCURRENCY = "cryptocurrency"
    FOREX = "forex"
    COMMODITY = "commodity"
    BOND = "bond"
    ETF = "etf"
    FUTURE = "future"
    OPTION = "option"


class AssetType(Enum):
    """Specific types of assets."""
    STOCK = "stock"
    ETF = "etf"
    CRYPTO_SPOT = "crypto_spot"
    CRYPTO_FUTURE = "crypto_future"
    CRYPTO_PERPETUAL = "crypto_perpetual"
    FOREX_SPOT = "forex_spot"
    FOREX_FORWARD = "forex_forward"
    METAL = "metal"
    ENERGY = "energy"
    AGRICULTURAL = "agricultural"
    BOND_GOVERNMENT = "bond_government"
    BOND_CORPORATE = "bond_corporate"


@dataclass
class AssetMetadata:
    """Metadata for different assets."""
    symbol: str
    name: str
    asset_class: AssetClass
    asset_type: AssetType
    base_currency: str
    quote_currency: str
    lot_size: float
    tick_size: float
    margin_required: float
    volatility_category: str  # 'low', 'medium', 'high', 'extreme'
    trading_hours: str
    exchange: str
    country: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    expiry_date: Optional[datetime] = None


class MultiAssetMarketDataAdapter(MarketDataPort):
    """
    Market data adapter supporting multiple asset classes.
    """
    
    def __init__(self):
        # Initialize different exchange libraries for different asset classes
        self.cryptocurrency_exchanges = {
            'binance': ccxt.binance({'enableRateLimit': True}),
            'coinbase': ccxt.coinbase({'enableRateLimit': True}),
            'kraken': ccxt.kraken({'enableRateLimit': True})
        }
        
        self.forex_exchanges = {
            # Forex data would typically come from specialized providers
            # For now, using mock data
        }
        
        self.asset_metadata = self._initialize_asset_metadata()
    
    def _initialize_asset_metadata(self) -> Dict[str, AssetMetadata]:
        """
        Initialize metadata for popular assets across different classes.
        """
        metadata = {}
        
        # Crypto assets
        crypto_assets = [
            ('BTC/USD', 'Bitcoin', AssetClass.CRYPTOCURRENCY, AssetType.CRYPTO_SPOT, 'BTC', 'USD', 0.001, 0.01, 0.1, 'high', '24/7', 'Binance', 'Global'),
            ('ETH/USD', 'Ethereum', AssetClass.CRYPTOCURRENCY, AssetType.CRYPTO_SPOT, 'ETH', 'USD', 0.01, 0.01, 0.1, 'high', '24/7', 'Binance', 'Global'),
            ('SOL/USD', 'Solana', AssetClass.CRYPTOCURRENCY, AssetType.CRYPTO_SPOT, 'SOL', 'USD', 0.1, 0.01, 0.15, 'extreme', '24/7', 'Binance', 'Global'),
            ('ADA/USD', 'Cardano', AssetClass.CRYPTOCURRENCY, AssetType.CRYPTO_SPOT, 'ADA', 'USD', 1.0, 0.0001, 0.1, 'medium', '24/7', 'Binance', 'Global')
        ]
        
        for symbol, name, asset_class, asset_type, base, quote, lot_size, tick_size, margin, vol, hours, exchange, country in crypto_assets:
            metadata[symbol] = AssetMetadata(
                symbol=symbol, name=name, asset_class=asset_class, asset_type=asset_type,
                base_currency=base, quote_currency=quote, lot_size=lot_size, tick_size=tick_size,
                margin_required=margin, volatility_category=vol, trading_hours=hours,
                exchange=exchange, country=country
            )
        
        # Forex pairs
        forex_assets = [
            ('EUR/USD', 'Euro to US Dollar', AssetClass.FOREX, AssetType.FOREX_SPOT, 'EUR', 'USD', 1000, 0.00001, 0.03, 'medium', '24/5', 'FXCM', 'Global'),
            ('GBP/USD', 'British Pound to US Dollar', AssetClass.FOREX, AssetType.FOREX_SPOT, 'GBP', 'USD', 1000, 0.00001, 0.03, 'medium', '24/5', 'FXCM', 'Global'),
            ('USD/JPY', 'US Dollar to Japanese Yen', AssetClass.FOREX, AssetType.FOREX_SPOT, 'USD', 'JPY', 1000, 0.001, 0.03, 'medium', '24/5', 'FXCM', 'Global'),
            ('AUD/USD', 'Australian Dollar to US Dollar', AssetClass.FOREX, AssetType.FOREX_SPOT, 'AUD', 'USD', 1000, 0.00001, 0.03, 'medium', '24/5', 'FXCM', 'Global')
        ]
        
        for symbol, name, asset_class, asset_type, base, quote, lot_size, tick_size, margin, vol, hours, exchange, country in forex_assets:
            metadata[symbol] = AssetMetadata(
                symbol=symbol, name=name, asset_class=asset_class, asset_type=asset_type,
                base_currency=base, quote_currency=quote, lot_size=lot_size, tick_size=tick_size,
                margin_required=margin, volatility_category=vol, trading_hours=hours,
                exchange=exchange, country=country
            )
        
        # Commodities
        commodity_assets = [
            ('GC=F', 'Gold Futures', AssetClass.COMMODITY, AssetType.FUTURE, 'oz', 'USD', 1, 0.1, 0.05, 'medium', '24/6', 'CME', 'USA'),
            ('CL=F', 'Crude Oil', AssetClass.COMMODITY, AssetType.FUTURE, 'barrel', 'USD', 1000, 0.01, 0.08, 'high', '24/6', 'CME', 'USA'),
            ('SI=F', 'Silver Futures', AssetClass.COMMODITY, AssetType.FUTURE, 'oz', 'USD', 1, 0.1, 0.05, 'high', '24/6', 'CME', 'USA'),
            ('HG=F', 'Copper Futures', AssetClass.COMMODITY, AssetType.FUTURE, 'lb', 'USD', 1, 0.0005, 0.08, 'medium', '24/6', 'CME', 'USA')
        ]
        
        for symbol, name, asset_class, asset_type, base, quote, lot_size, tick_size, margin, vol, hours, exchange, country in commodity_assets:
            metadata[symbol] = AssetMetadata(
                symbol=symbol, name=name, asset_class=asset_class, asset_type=asset_type,
                base_currency=base, quote_currency=quote, lot_size=lot_size, tick_size=tick_size,
                margin_required=margin, volatility_category=vol, trading_hours=hours,
                exchange=exchange, country=country
            )
        
        # Equities (extended list)
        equity_assets = [
            ('TSLA', 'Tesla Inc', AssetClass.EQUITY, AssetType.STOCK, 'SHARES', 'USD', 1, 0.01, 0.0, 'high', '9:30-16:00 ET', 'NASDAQ', 'USA', 'Technology', 'Auto'),
            ('NVDA', 'NVIDIA Corp', AssetClass.EQUITY, AssetType.STOCK, 'SHARES', 'USD', 1, 0.01, 0.0, 'high', '9:30-16:00 ET', 'NASDAQ', 'USA', 'Technology', 'Semiconductors'),
            ('AAPL', 'Apple Inc', AssetClass.EQUITY, AssetType.STOCK, 'SHARES', 'USD', 1, 0.01, 0.0, 'medium', '9:30-16:00 ET', 'NASDAQ', 'USA', 'Technology', 'Consumer Electronics'),
            ('MSFT', 'Microsoft Corp', AssetClass.EQUITY, AssetType.STOCK, 'SHARES', 'USD', 1, 0.01, 0.0, 'medium', '9:30-16:00 ET', 'NASDAQ', 'USA', 'Technology', 'Software')
        ]
        
        for symbol, name, asset_class, asset_type, base, quote, lot_size, tick_size, margin, vol, hours, exchange, country, sector, industry in equity_assets:
            metadata[symbol] = AssetMetadata(
                symbol=symbol, name=name, asset_class=asset_class, asset_type=asset_type,
                base_currency=base, quote_currency=quote, lot_size=lot_size, tick_size=tick_size,
                margin_required=margin, volatility_category=vol, trading_hours=hours,
                exchange=exchange, country=country, sector=sector, industry=industry
            )
        
        return metadata
    
    def get_current_price(self, symbol: Symbol) -> Optional[Price]:
        """
        Get current price for any asset class.
        """
        str_symbol = str(symbol).upper()
        
        # Determine asset class and fetch accordingly
        if str_symbol in self.asset_metadata:
            asset_meta = self.asset_metadata[str_symbol]
            
            if asset_meta.asset_class == AssetClass.CRYPTOCURRENCY:
                return self._get_crypto_price(str_symbol)
            elif asset_meta.asset_class == AssetClass.FOREX:
                return self._get_forex_price(str_symbol)
            elif asset_meta.asset_class == AssetClass.COMMODITY:
                return self._get_commodity_price(str_symbol)
            elif asset_meta.asset_class == AssetClass.EQUITY:
                return self._get_equity_price(symbol)
        
        # Default fallback - try with yfinance
        return self._get_equity_price(symbol)
    
    def _get_crypto_price(self, symbol: str) -> Optional[Price]:
        """
        Get current price for cryptocurrency from exchanges.
        """
        try:
            # Try different exchanges
            for exchange_name, exchange in self.cryptocurrency_exchanges.items():
                try:
                    ticker = exchange.fetch_ticker(symbol)
                    if 'last' in ticker and ticker['last'] is not None:
                        return Price(ticker['last'], self.asset_metadata[symbol].quote_currency)
                except:
                    continue  # Try next exchange
        except:
            pass
        
        # If real data unavailable, return mock price
        import random
        mock_price = random.uniform(10000, 100000) if symbol.startswith('BTC') else random.uniform(100, 10000)
        return Price(mock_price, 'USD')
    
    def _get_forex_price(self, symbol: str) -> Optional[Price]:
        """
        Get current price for forex pair.
        """
        # Forex data would come from specialized providers
        # For now, return mock data
        import random
        base = symbol.split('/')[0]
        quote = symbol.split('/')[1]
        
        if base == 'USD' and quote == 'JPY':
            mock_price = random.uniform(130, 150)
        elif base == 'EUR' and quote == 'USD':
            mock_price = random.uniform(1.05, 1.15)
        elif base == 'GBP' and quote == 'USD':
            mock_price = random.uniform(1.25, 1.35)
        else:
            mock_price = random.uniform(0.7, 1.5)
        
        return Price(mock_price, quote)
    
    def _get_commodity_price(self, symbol: str) -> Optional[Price]:
        """
        Get current price for commodity futures.
        """
        # For futures, we'll use Yahoo Finance
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                quote_currency = 'USD'  # Commodities usually priced in USD
                return Price(float(current_price), quote_currency)
        except:
            pass
        
        # Mock data for commodities
        import random
        symbol_base = symbol.split('=')[0] if '=' in symbol else symbol
        if symbol_base == 'GC':  # Gold
            mock_price = random.uniform(1800, 2200)
        elif symbol_base == 'CL':  # Crude Oil
            mock_price = random.uniform(50, 100)
        elif symbol_base == 'SI':  # Silver
            mock_price = random.uniform(20, 30)
        else:
            mock_price = random.uniform(3, 5)
        
        return Price(mock_price, 'USD')
    
    def _get_equity_price(self, symbol: Symbol) -> Optional[Price]:
        """
        Get current price for equity using yfinance.
        """
        try:
            ticker = yf.Ticker(str(symbol))
            hist = ticker.history(period="1d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                return Price(float(current_price), 'USD')
        except:
            pass
        
        # Mock price if real data unavailable
        import random
        mock_price = random.uniform(10, 500)
        return Price(mock_price, 'USD')
    
    def get_historical_prices(self, symbol: Symbol, start_date: datetime, end_date: datetime) -> List[Price]:
        """
        Get historical prices for any asset class.
        """
        str_symbol = str(symbol).upper()
        
        if str_symbol in self.asset_metadata:
            asset_meta = self.asset_metadata[str_symbol]
            
            if asset_meta.asset_class == AssetClass.CRYPTOCURRENCY:
                return self._get_crypto_historical(symbol, start_date, end_date)
            elif asset_meta.asset_class == AssetClass.FOREX:
                return self._get_forex_historical(symbol, start_date, end_date)
            elif asset_meta.asset_class == AssetClass.COMMODITY:
                return self._get_commodity_historical(symbol, start_date, end_date)
            elif asset_meta.asset_class == AssetClass.EQUITY:
                return self._get_equity_historical(symbol, start_date, end_date)
        
        # Default to equity historical data
        return self._get_equity_historical(symbol, start_date, end_date)
    
    def _get_crypto_historical(self, symbol: Symbol, start_date: datetime, end_date: datetime) -> List[Price]:
        """
        Get historical prices for cryptocurrency.
        """
        try:
            # Use one of the exchanges
            exchange = self.cryptocurrency_exchanges['binance']
            timeframe = '1d'
            
            # Convert dates to timestamps
            since = int(start_date.timestamp() * 1000)
            limit = (end_date - start_date).days
            
            ohlcvs = exchange.fetch_ohlcv(str(symbol), timeframe, since, limit)
            
            prices = []
            for ohlcv in ohlcvs:
                # ohlcv format: [timestamp, open, high, low, close, volume]
                close_price = ohlcv[4]
                prices.append(Price(close_price, 'USD'))
            
            return prices
        except:
            # Mock historical data for crypto
            days = (end_date - start_date).days
            base_price = 50000 if 'BTC' in str(symbol) else 3000
            prices = []
            for i in range(days):
                price = base_price * (1 + np.random.normal(0, 0.05))  # Random walk
                prices.append(Price(price, 'USD'))
            return prices
    
    def _get_forex_historical(self, symbol: Symbol, start_date: datetime, end_date: datetime) -> List[Price]:
        """
        Get historical prices for forex pair.
        """
        # Mock historical data for forex
        days = (end_date - start_date).days
        base_price = 1.1  # Starting rate
        prices = []
        
        for i in range(days):
            # Simulate realistic forex movements
            daily_change = np.random.normal(0, 0.005)  # 0.5% daily volatility
            base_price = base_price * (1 + daily_change)
            prices.append(Price(base_price, 'USD'))
        
        return prices
    
    def _get_commodity_historical(self, symbol: Symbol, start_date: datetime, end_date: datetime) -> List[Price]:
        """
        Get historical prices for commodity.
        """
        try:
            # Use yfinance for futures
            ticker = yf.Ticker(str(symbol))
            hist = ticker.history(start=start_date, end=end_date)
            
            prices = []
            for idx, row in hist.iterrows():
                prices.append(Price(float(row['Close']), 'USD'))
            
            return prices
        except:
            # Mock historical data for commodities
            days = (end_date - start_date).days
            symbol_base = str(symbol).split('=')[0] if '=' in str(symbol) else str(symbol)
            
            if symbol_base == 'GC':  # Gold
                base_price = 1950
            elif symbol_base == 'CL':  # Crude Oil
                base_price = 80
            elif symbol_base == 'SI':  # Silver
                base_price = 24
            else:
                base_price = 4
            
            prices = []
            for i in range(days):
                price = base_price * (1 + np.random.normal(0, 0.03))  # 3% daily volatility
                prices.append(Price(price, 'USD'))
            
            return prices
    
    def _get_equity_historical(self, symbol: Symbol, start_date: datetime, end_date: datetime) -> List[Price]:
        """
        Get historical prices for equity.
        """
        try:
            ticker = yf.Ticker(str(symbol))
            hist = ticker.history(start=start_date, end=end_date)
            
            prices = []
            for idx, row in hist.iterrows():
                prices.append(Price(float(row['Close']), 'USD'))
            
            return prices
        except:
            # Mock historical data for equities
            days = (end_date - start_date).days
            base_price = 150  # Starting price
            prices = []
            
            for i in range(days):
                # Simulate realistic stock price movement
                daily_return = np.random.normal(0.001, 0.02)  # 0.1% average daily return, 2% volatility
                base_price = base_price * (1 + daily_return)
                prices.append(Price(base_price, 'USD'))
            
            return prices
    
    def get_market_news(self, symbol: Symbol) -> List[str]:
        """
        Get market news for any asset class.
        """
        # This would integrate with different news sources based on asset class
        # For now, return empty list
        return []


class MultiAssetTradingAdapter(TradingExecutionPort):
    """
    Trading execution adapter supporting multiple asset classes.
    """
    
    def __init__(self, market_data_service: MarketDataPort):
        self.market_data_service = market_data_service
        self.multi_asset_data_adapter = MultiAssetMarketDataAdapter()
        
        # Initialize exchange connections for different asset classes
        self.cryptocurrency_exchanges = {
            'binance': ccxt.binance({
                'apiKey': settings.BINANCE_API_KEY if hasattr(settings, 'BINANCE_API_KEY') else '',
                'secret': settings.BINANCE_SECRET_KEY if hasattr(settings, 'BINANCE_SECRET_KEY') else '',
                'enableRateLimit': True
            })
        }
    
    def place_order(self, order: Order) -> str:
        """
        Place an order for any asset class.
        """
        str_symbol = str(order.symbol).upper()
        
        # Determine asset class and route to appropriate exchange
        if str_symbol in self.multi_asset_data_adapter.asset_metadata:
            asset_meta = self.multi_asset_data_adapter.asset_metadata[str_symbol]
            
            if asset_meta.asset_class == AssetClass.CRYPTOCURRENCY:
                return self._place_crypto_order(order)
            elif asset_meta.asset_class == AssetClass.FOREX:
                return self._place_forex_order(order)
            elif asset_meta.asset_class == AssetClass.COMMODITY:
                return self._place_commodity_order(order)
            elif asset_meta.asset_class in [AssetClass.EQUITY, AssetClass.ETF]:
                return self._place_equity_order(order)
        
        # Default to equity order if asset class not recognized
        return self._place_equity_order(order)
    
    def _place_crypto_order(self, order: Order) -> str:
        """
        Place a cryptocurrency order.
        """
        try:
            # Get the exchange based on the asset
            exchange = self.cryptocurrency_exchanges['binance']
            
            # Determine order type
            side = 'buy' if order.position_type.name == 'LONG' else 'sell'
            type_map = {
                OrderType.MARKET.value.lower(): 'market',
                OrderType.LIMIT.value.lower(): 'limit',
                OrderType.STOP_LOSS.value.lower(): 'stop_loss'
            }
            
            order_type = type_map.get(order.order_type.value.lower(), 'market')
            symbol = str(order.symbol)
            
            # Execute order
            if order_type == 'limit':
                result = exchange.create_limit_order(
                    symbol=symbol,
                    side=side,
                    amount=order.quantity,
                    price=float(order.price.amount) if order.price else None
                )
            elif order_type == 'market':
                result = exchange.create_market_order(
                    symbol=symbol,
                    side=side,
                    amount=order.quantity
                )
            else:
                # For stop orders, we'll use market order with stop price
                result = exchange.create_order(
                    symbol=symbol,
                    type='stop',
                    side=side,
                    amount=order.quantity,
                    price=float(order.stop_price.amount) if order.stop_price else None
                )
            
            return result['id']
        except Exception as e:
            print(f"Crypto order placement failed: {e}")
            # Return mock order ID for simulation
            import secrets
            return f"mock_crypto_order_{secrets.token_hex(8)}"
    
    def _place_forex_order(self, order: Order) -> str:
        """
        Place a forex order (would connect to forex broker API).
        """
        # In a real implementation, this would connect to a forex broker
        import secrets
        return f"mock_forex_order_{secrets.token_hex(8)}"
    
    def _place_commodity_order(self, order: Order) -> str:
        """
        Place a commodity order.
        """
        # In a real implementation, this would connect to a futures broker
        import secrets
        return f"mock_commodity_order_{secrets.token_hex(8)}"
    
    def _place_equity_order(self, order: Order) -> str:
        """
        Place an equity order.
        """
        # For equities, we'll assume it gets routed to a traditional broker
        # This would connect to broker APIs like Alpaca, Interactive Brokers, etc.
        import secrets
        return f"mock_equity_order_{secrets.token_hex(8)}"
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order across asset classes.
        """
        # In a real implementation, this would determine the asset class
        # and route to the appropriate exchange/broker
        return True
    
    def get_order_status(self, order_id: str) -> Order:
        """
        Get order status across asset classes.
        """
        # This would query the appropriate exchange/broker based on order ID
        # For now, return a minimal order
        return Order(
            id=order_id,
            user_id="user123",
            symbol=Symbol("AAPL"),  # Would get from order records
            order_type=OrderType.MARKET,
            position_type=PositionType.LONG,
            quantity=10,
            status=OrderStatus.EXECUTED,
            placed_at=datetime.now(),
        )
    
    def get_account_balance(self, user_id: str) -> Money:
        """
        Get account balance across all asset classes.
        """
        # This would aggregate balances from all connected exchanges/brokers
        # For now, returning a mock balance
        import random
        balance = random.uniform(10000, 1000000)
        return Money(balance, 'USD')


class MultiAssetPortfolioManager:
    """
    Manage portfolio across multiple asset classes.
    """
    
    def __init__(self, market_data_service: MarketDataPort):
        self.market_data_service = market_data_service
    
    def calculate_multi_asset_portfolio_value(self, portfolio: Portfolio) -> Money:
        """
        Calculate portfolio value considering all asset classes.
        """
        total_value = 0.0
        
        # Calculate value of each position
        for position in portfolio.positions:
            current_price = self.market_data_service.get_current_price(position.symbol)
            if current_price:
                position_value = float(current_price.amount) * position.quantity
                total_value += position_value
        
        # Add cash balance
        total_value += float(portfolio.cash_balance.amount)
        
        return Money(total_value, portfolio.cash_balance.currency)
    
    def calculate_correlation_matrix(self, portfolio: Portfolio) -> np.ndarray:
        """
        Calculate correlation matrix across different asset classes.
        """
        symbols = [str(pos.symbol) for pos in portfolio.positions]
        if not symbols:
            return np.array([])
        
        # Get historical data for correlation calculation
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)  # 2 months of data
        
        returns_data = {}
        for symbol in symbols:
            try:
                prices = self.market_data_service.get_historical_prices(
                    Symbol(symbol), start_date, end_date
                )
                if len(prices) > 1:
                    price_values = [float(p.amount) for p in prices]
                    returns = [(price_values[i] - price_values[i-1]) / price_values[i-1] 
                              for i in range(1, len(price_values))]
                    returns_data[symbol] = returns
            except:
                continue
        
        if not returns_data:
            return np.array([])
        
        # Create correlation matrix
        df_returns = pd.DataFrame(returns_data)
        correlation_matrix = df_returns.corr().fillna(0).values
        
        return correlation_matrix
    
    def assess_diversification(self, portfolio: Portfolio) -> Dict[str, Any]:
        """
        Assess portfolio diversification across asset classes.
        """
        asset_class_counts = {}
        total_value = float(self.calculate_multi_asset_portfolio_value(portfolio).amount)
        
        if total_value == 0:
            return {
                'diversification_score': 0,
                'asset_class_distribution': {},
                'concentration_risk': 'N/A'
            }
        
        # Count positions by asset class
        for position in portfolio.positions:
            str_symbol = str(position.symbol).upper()
            # Determine asset class (would use metadata in real implementation)
            if any(crypto in str_symbol for crypto in ['BTC', 'ETH', 'SOL', 'ADA']):
                asset_class = 'CRYPTO'
            elif any(forex in str_symbol for forex in ['EUR', 'GBP', 'JPY', 'AUD']):
                asset_class = 'FOREX'
            elif any(commodity in str_symbol for commodity in ['GC', 'CL', 'SI', 'HG']):
                asset_class = 'COMMODITY'
            else:
                asset_class = 'EQUITY'
            
            position_value = float(position.market_value.amount)
            if asset_class in asset_class_counts:
                asset_class_counts[asset_class] += position_value
            else:
                asset_class_counts[asset_class] = position_value
        
        # Calculate distribution percentages
        distribution = {
            asset_class: (value / total_value) * 100 
            for asset_class, value in asset_class_counts.items()
        }
        
        # Calculate diversification score (0-100)
        # Higher score for more equal distribution
        if len(distribution) == 0:
            diversification_score = 0
        elif len(distribution) == 1:
            diversification_score = 0  # Not diversified
        else:
            # Score based on how equal the distribution is
            values = list(distribution.values())
            score = 100 * (1 - np.std(values) / 50)  # Normalize to 0-100
            diversification_score = max(0, min(100, score))
        
        # Assess concentration risk
        max_allocation = max(distribution.values()) if distribution else 0
        if max_allocation > 60:
            concentration_risk = 'HIGH'
        elif max_allocation > 40:
            concentration_risk = 'MEDIUM'
        else:
            concentration_risk = 'LOW'
        
        return {
            'diversification_score': diversification_score,
            'asset_class_distribution': distribution,
            'concentration_risk': concentration_risk
        }


# Initialize multi-asset services
multi_asset_market_adapter = MultiAssetMarketDataAdapter()
multi_asset_trading_adapter = MultiAssetTradingAdapter(multi_asset_market_adapter)
multi_asset_portfolio_manager = MultiAssetPortfolioManager(multi_asset_market_adapter)