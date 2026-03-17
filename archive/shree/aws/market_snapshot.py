"""
Market Snapshot Builder for AWS Agent Integration

This module builds standardized market snapshots from various data sources
for consumption by Bedrock Agents.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger


class MarketSnapshotBuilder:
    """Build standardized market snapshots for agent consumption.
    
    This class aggregates data from:
    - Real-time price data
    - Technical indicators
    - Higher timeframe levels (PDH, PDL, etc.)
    - Market regime classification
    """
    
    def __init__(self, symbol: str = "ES"):
        """Initialize Market Snapshot Builder.
        
        Args:
            symbol: Trading symbol (default: ES for E-mini S&P 500)
        """
        self.symbol = symbol
        self._last_snapshot = None
        self._snapshot_count = 0
        
        # Historical levels (set at startup)
        self._historical_levels = {
            'pdh': 0,
            'pdl': 0,
            'prev_close': 0,
            'weekly_high': 0,
            'weekly_low': 0,
        }
    
    def set_historical_levels(
        self,
        pdh: float,
        pdl: float,
        prev_close: float,
        weekly_high: float = 0,
        weekly_low: float = 0,
    ):
        """Set historical levels loaded at startup.
        
        Args:
            pdh: Previous day high
            pdl: Previous day low  
            prev_close: Previous day close
            weekly_high: Weekly high
            weekly_low: Weekly low
        """
        self._historical_levels = {
            'pdh': pdh,
            'pdl': pdl,
            'prev_close': prev_close,
            'weekly_high': weekly_high,
            'weekly_low': weekly_low,
        }
        logger.info(f"Historical levels set: PDH={pdh:.2f}, PDL={pdl:.2f}, PrevClose={prev_close:.2f}")
    
    def build_snapshot(
        self,
        price_data: Dict[str, Any],
        indicators: Dict[str, Any] = None,
        levels: Dict[str, Any] = None,
        sentiment: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Build a complete market snapshot.
        
        Args:
            price_data: Current price information
                {'price': float, 'bid': float, 'ask': float, 'volume': int}
            indicators: Technical indicators
                {'rsi': float, 'macd': float, 'ema_9': float, 'ema_20': float, 'atr': float}
            levels: Key price levels
                {'pdh': float, 'pdl': float, 'week_high': float, 'week_low': float}
            sentiment: Sentiment data (optional)
                {'score': float, 'source': str}
                
        Returns:
            Standardized market snapshot dictionary
        """
        indicators = indicators or {}
        levels = levels or {}
        sentiment = sentiment or {}
        
        # Extract price data
        current_price = price_data.get('price', 0)
        
        # Use stored historical levels as defaults, override with provided levels
        pdh = levels.get('pdh') or self._historical_levels.get('pdh', current_price)
        pdl = levels.get('pdl') or self._historical_levels.get('pdl', current_price)
        prev_close = levels.get('prev_close') or self._historical_levels.get('prev_close', current_price)
        weekly_high = levels.get('week_high') or self._historical_levels.get('weekly_high', pdh)
        weekly_low = levels.get('week_low') or self._historical_levels.get('weekly_low', pdl)
        
        # Calculate deltas from key levels
        pdh_delta = ((current_price - pdh) / pdh * 100) if pdh > 0 else 0
        pdl_delta = ((current_price - pdl) / pdl * 100) if pdl > 0 else 0
        prev_close_delta = ((current_price - prev_close) / prev_close * 100) if prev_close > 0 else 0
        
        # Classify regime
        regime = self._classify_regime(indicators, price_data)
        
        # Classify volatility
        volatility = self._classify_volatility(indicators.get('atr', 0))
        
        # Classify time of day
        time_of_day = self._classify_time_of_day()
        
        # Determine daily bias based on price vs PDH/PDL
        daily_bias = "NEUTRAL"
        if pdl_delta < -0.3:  # Price significantly below PDL
            daily_bias = "BEARISH"
        elif pdh_delta > 0.3:  # Price significantly above PDH
            daily_bias = "BULLISH"
        
        snapshot = {
            # Identity
            'symbol': self.symbol,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            
            # Price data
            'price': current_price,
            'bid': price_data.get('bid', current_price),
            'ask': price_data.get('ask', current_price),
            'spread': price_data.get('ask', current_price) - price_data.get('bid', current_price),
            'volume': price_data.get('volume', 0),
            
            # Technical indicators
            'rsi': indicators.get('rsi', 50),
            'macd': indicators.get('macd', 0),
            'macd_histogram': indicators.get('macd_histogram', 0),
            'macd_signal': indicators.get('macd_signal', 0),
            'ema_9': indicators.get('ema_9', current_price),
            'ema_20': indicators.get('ema_20', current_price),
            'atr': indicators.get('atr', 0),
            'adx': indicators.get('adx', 0),
            
            # Key levels (historical data loaded at startup)
            'pdh': pdh,
            'pdl': pdl,
            'prev_close': prev_close,
            'PDH_delta': pdh_delta,
            'PDL_delta': pdl_delta,
            'prev_close_delta': prev_close_delta,
            'week_high': weekly_high,
            'week_low': weekly_low,
            'prev_week_high': levels.get('prev_week_high', 0),
            'prev_week_low': levels.get('prev_week_low', 0),
            
            # Classifications
            'trend': regime,
            'regime': regime,
            'volatility': volatility,
            'time_of_day': time_of_day,
            'daily_bias': daily_bias,  # BULLISH/BEARISH/NEUTRAL based on PDH/PDL
            
            # Sentiment (if available)
            'sentiment_score': sentiment.get('score', 0),
            'sentiment_source': sentiment.get('source', 'none'),
            
            # Computed fields
            'adr': self._calculate_adr(levels),
            'range_position': self._calculate_range_position(current_price, pdh, pdl),
        }
        
        self._last_snapshot = snapshot
        self._snapshot_count += 1
        
        return snapshot
    
    def build_from_dataframe(
        self,
        df,
        index: int = -1,
        levels: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Build snapshot from a pandas DataFrame row.
        
        Args:
            df: DataFrame with OHLCV and indicator columns
            index: Row index to use (default: -1 for latest)
            levels: Key price levels
            
        Returns:
            Standardized market snapshot
        """
        row = df.iloc[index]
        
        price_data = {
            'price': float(row.get('close', row.get('Close', 0))),
            'bid': float(row.get('bid', row.get('close', 0))),
            'ask': float(row.get('ask', row.get('close', 0))),
            'volume': int(row.get('volume', row.get('Volume', 0))),
        }
        
        indicators = {
            'rsi': float(row.get('rsi', row.get('RSI', 50))),
            'macd': float(row.get('macd', row.get('MACD', 0))),
            'macd_histogram': float(row.get('macd_histogram', row.get('MACD_Histogram', 0))),
            'macd_signal': float(row.get('macd_signal', row.get('MACD_Signal', 0))),
            'ema_9': float(row.get('ema_9', row.get('EMA_9', price_data['price']))),
            'ema_20': float(row.get('ema_20', row.get('EMA_20', price_data['price']))),
            'atr': float(row.get('atr', row.get('ATR', 0))),
            'adx': float(row.get('adx', row.get('ADX', 0))),
        }
        
        return self.build_snapshot(price_data, indicators, levels)
    
    def _classify_regime(
        self,
        indicators: Dict[str, Any],
        price_data: Dict[str, Any],
    ) -> str:
        """Classify market regime based on indicators."""
        ema_9 = indicators.get('ema_9', 0)
        ema_20 = indicators.get('ema_20', 0)
        adx = indicators.get('adx', 0)
        price = price_data.get('price', 0)
        
        # Strong trend detection
        if adx > 25:
            if ema_9 > ema_20 and price > ema_9:
                return 'UPTREND'
            elif ema_9 < ema_20 and price < ema_9:
                return 'DOWNTREND'
        
        # Weak trend or ranging
        if ema_9 > ema_20:
            return 'UPTREND' if price > ema_20 else 'RANGE'
        elif ema_9 < ema_20:
            return 'DOWNTREND' if price < ema_20 else 'RANGE'
        
        return 'RANGE'
    
    def _classify_volatility(self, atr: float) -> str:
        """Classify volatility based on ATR."""
        if atr > 3.0:
            return 'HIGH'
        elif atr > 1.5:
            return 'MED'
        else:
            return 'LOW'
    
    def _classify_time_of_day(self) -> str:
        """Classify current time of day for trading context."""
        # Get current hour in CST (UTC-6)
        now = datetime.now(timezone.utc)
        hour = (now.hour - 6) % 24
        
        if 8 <= hour < 10:
            return 'MORNING'  # Market open volatility
        elif 10 <= hour < 12:
            return 'MIDDAY'
        elif 12 <= hour < 14:
            return 'AFTERNOON'
        elif 14 <= hour < 16:
            return 'CLOSE'  # Approaching close
        else:
            return 'EXTENDED'  # Extended hours
    
    def _calculate_adr(self, levels: Dict[str, Any]) -> float:
        """Calculate Average Daily Range."""
        pdh = levels.get('pdh', 0)
        pdl = levels.get('pdl', 0)
        
        if pdh > 0 and pdl > 0:
            return pdh - pdl
        return 0
    
    def _calculate_range_position(
        self,
        price: float,
        high: float,
        low: float,
    ) -> float:
        """Calculate position within range (0-1)."""
        if high > low:
            return (price - low) / (high - low)
        return 0.5
    
    def build(
        self,
        symbol: str = None,
        price: float = 6000.0,
        trend: str = "RANGE",
        volatility: str = "MED",
        rsi: float = 50.0,
        atr: float = 10.0,
        ema_9: float = None,
        ema_20: float = None,
        volume: int = 50000,
        pdh: float = None,
        pdl: float = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Build market snapshot from individual parameters.
        
        This is a convenience method for quick snapshot creation.
        
        Args:
            symbol: Trading symbol
            price: Current price
            trend: Market trend (UPTREND, DOWNTREND, RANGE)
            volatility: Volatility level (HIGH, MED, LOW)
            rsi: RSI value (0-100)
            atr: ATR value
            ema_9: EMA 9 value
            ema_20: EMA 20 value
            volume: Current volume
            pdh: Previous day high
            pdl: Previous day low
            **kwargs: Additional fields to include
            
        Returns:
            Standardized market snapshot
        """
        symbol = symbol or self.symbol
        ema_9 = ema_9 or price
        ema_20 = ema_20 or price
        pdh = pdh or price * 1.005  # Default 0.5% above
        pdl = pdl or price * 0.995  # Default 0.5% below
        
        price_data = {
            'price': price,
            'bid': price - 0.25,
            'ask': price + 0.25,
            'volume': volume,
        }
        
        indicators = {
            'rsi': rsi,
            'macd': 0,
            'ema_9': ema_9,
            'ema_20': ema_20,
            'atr': atr,
            'adx': 25,
        }
        
        levels = {
            'pdh': pdh,
            'pdl': pdl,
            'week_high': pdh * 1.01,
            'week_low': pdl * 0.99,
        }
        
        snapshot = self.build_snapshot(price_data, indicators, levels)
        
        # Override computed fields if explicitly set
        snapshot['trend'] = trend
        snapshot['regime'] = trend
        snapshot['volatility'] = volatility
        snapshot['symbol'] = symbol
        
        # Add any extra kwargs
        snapshot.update(kwargs)
        
        return snapshot
    
    def build_mock(self) -> Dict[str, Any]:
        """Build a mock market snapshot for testing.
        
        Returns:
            Mock market snapshot with realistic ES futures values
        """
        import random
        
        # Generate realistic mock data for ES futures
        base_price = 6000.0 + random.uniform(-50, 50)
        
        return self.build(
            symbol="ES",
            price=base_price,
            trend=random.choice(["UPTREND", "DOWNTREND", "RANGE"]),
            volatility=random.choice(["HIGH", "MED", "LOW"]),
            rsi=50 + random.uniform(-20, 20),
            atr=10 + random.uniform(-3, 3),
            ema_9=base_price + random.uniform(-5, 5),
            ema_20=base_price + random.uniform(-10, 10),
            volume=random.randint(30000, 80000),
            pdh=base_price + random.uniform(10, 30),
            pdl=base_price - random.uniform(10, 30),
            vix=15 + random.uniform(-5, 10),
        )
    
    def get_last_snapshot(self) -> Optional[Dict[str, Any]]:
        """Get the last generated snapshot."""
        return self._last_snapshot
    
    def get_stats(self) -> Dict[str, Any]:
        """Get builder statistics."""
        return {
            'symbol': self.symbol,
            'snapshot_count': self._snapshot_count,
            'last_snapshot_time': self._last_snapshot.get('timestamp') if self._last_snapshot else None,
        }


def build_snapshot_from_ib_data(
    ticker,
    indicators: Dict[str, Any] = None,
    levels: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Build snapshot from IB ticker data.
    
    Args:
        ticker: IB-insync Ticker object
        indicators: Pre-computed technical indicators
        levels: Key price levels
        
    Returns:
        Standardized market snapshot
    """
    builder = MarketSnapshotBuilder(symbol=ticker.contract.symbol if hasattr(ticker, 'contract') else 'ES')
    
    price_data = {
        'price': ticker.marketPrice() if hasattr(ticker, 'marketPrice') else 0,
        'bid': ticker.bid if hasattr(ticker, 'bid') else 0,
        'ask': ticker.ask if hasattr(ticker, 'ask') else 0,
        'volume': ticker.volume if hasattr(ticker, 'volume') else 0,
    }
    
    return builder.build_snapshot(price_data, indicators, levels)
