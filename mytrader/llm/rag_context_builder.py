"""RAG Context Builder for Bedrock analysis.

This module prepares structured context for Bedrock LLM prompts:
- Summarizes market state (price, indicators, volatility)
- Includes position state
- Optionally includes news headlines
- Produces consistent, JSON-parseable output
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..utils.logger import logger
from .event_detector import EventPayload


@dataclass
class MarketContext:
    """Structured market context for Bedrock analysis."""
    
    # Time and instrument
    timestamp: datetime
    instrument: str
    
    # Price data
    current_price: float
    price_change_pct: float
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    
    # Technical indicators
    momentum: float = 0.0
    atr: float = 0.0
    volatility: float = 0.0
    rsi: float = 50.0
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    bb_percent: Optional[float] = None
    
    # Market conditions
    vix: Optional[float] = None
    market_regime: str = "unknown"
    volatility_regime: str = "normal"
    
    # Position
    position: int = 0
    unrealized_pnl: float = 0.0
    entry_price: Optional[float] = None
    
    # Recent price action
    recent_prices: List[float] = None
    
    # News
    news_headlines: List[str] = None
    
    def __post_init__(self):
        if self.recent_prices is None:
            self.recent_prices = []
        if self.news_headlines is None:
            self.news_headlines = []


class RAGContextBuilder:
    """Builds structured context for Bedrock analysis.
    
    The context is formatted as a structured text block that can be
    easily parsed by the LLM to generate trading insights.
    """
    
    # Default timezone for display
    DISPLAY_TIMEZONE = "US/Central"
    
    def __init__(
        self,
        instrument: str = "MES",
        include_news: bool = True,
        max_news_items: int = 3,
        max_recent_prices: int = 10,
    ):
        """Initialize context builder.
        
        Args:
            instrument: Trading instrument name
            include_news: Whether to include news headlines
            max_news_items: Maximum news items to include
            max_recent_prices: Maximum recent prices to include
        """
        self.instrument = instrument
        self.include_news = include_news
        self.max_news_items = max_news_items
        self.max_recent_prices = max_recent_prices
        
        logger.info(f"RAGContextBuilder initialized for {instrument}")
    
    def build_context(self, payload: EventPayload) -> str:
        """Build structured context string from event payload.
        
        Args:
            payload: Event payload from event detector
            
        Returns:
            Formatted context string for Bedrock prompt
        """
        # Format timestamp for display
        timestamp_str = payload.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
        
        # Build context sections
        sections = []
        
        # Header
        sections.append("=" * 50)
        sections.append("MARKET CONTEXT")
        sections.append("=" * 50)
        
        # Basic info
        sections.append(f"\nTime: {timestamp_str}")
        sections.append(f"Instrument: {payload.symbol} (Micro E-mini S&P 500 Futures)")
        sections.append(f"Trigger: {payload.trigger_type.upper()} - {payload.reason}")
        
        # Price section
        sections.append("\n--- PRICE DATA ---")
        sections.append(f"Current Price: {payload.current_price:.2f}")
        sections.append(f"Price Change: {payload.price_change_pct:+.2%}")
        
        # Include recent price trend
        if payload.recent_prices:
            recent = payload.recent_prices[-self.max_recent_prices:]
            if len(recent) >= 2:
                trend = "↑ RISING" if recent[-1] > recent[0] else "↓ FALLING" if recent[-1] < recent[0] else "→ FLAT"
                price_range = max(recent) - min(recent)
                sections.append(f"Recent Trend: {trend}")
                sections.append(f"Recent Range: {price_range:.2f} points")
                sections.append(f"Recent Prices: {', '.join(f'{p:.2f}' for p in recent[-5:])}")
        
        # Technical indicators section
        sections.append("\n--- TECHNICAL INDICATORS ---")
        sections.append(f"Momentum: {payload.momentum:+.3f}")
        sections.append(f"ATR (5m): {payload.atr:.2f}")
        sections.append(f"Volatility: {payload.volatility:.3f}")
        sections.append(f"RSI (14): {payload.rsi:.1f}")
        
        # RSI interpretation
        if payload.rsi < 30:
            rsi_note = "(OVERSOLD - potential bounce)"
        elif payload.rsi > 70:
            rsi_note = "(OVERBOUGHT - potential pullback)"
        elif payload.rsi < 40:
            rsi_note = "(weak/bearish)"
        elif payload.rsi > 60:
            rsi_note = "(strong/bullish)"
        else:
            rsi_note = "(neutral)"
        sections.append(f"RSI Note: {rsi_note}")
        
        # VIX if available
        if payload.vix is not None:
            if payload.vix < 15:
                vix_note = "(LOW - complacent)"
            elif payload.vix > 25:
                vix_note = "(HIGH - fearful)"
            elif payload.vix > 20:
                vix_note = "(elevated)"
            else:
                vix_note = "(normal)"
            sections.append(f"VIX: {payload.vix:.1f} {vix_note}")
        
        # Position section
        sections.append("\n--- POSITION STATE ---")
        if payload.position == 0:
            sections.append("Position: FLAT (no open position)")
        elif payload.position > 0:
            sections.append(f"Position: LONG {payload.position} contracts")
            sections.append(f"Unrealized P&L: ${payload.unrealized_pnl:+.2f}")
        else:
            sections.append(f"Position: SHORT {abs(payload.position)} contracts")
            sections.append(f"Unrealized P&L: ${payload.unrealized_pnl:+.2f}")
        
        # News section (if enabled and available)
        if self.include_news and payload.news_headlines:
            sections.append("\n--- NEWS / EVENTS ---")
            news_items = payload.news_headlines[:self.max_news_items]
            for i, headline in enumerate(news_items, 1):
                sections.append(f"{i}. {headline}")
        else:
            sections.append("\n--- NEWS / EVENTS ---")
            sections.append("No significant news detected")
        
        # Task instruction
        sections.append("\n" + "=" * 50)
        sections.append("TASK")
        sections.append("=" * 50)
        sections.append("""
Analyze the above market context and provide a directional bias assessment.

Your response will be used as a BIAS MODIFIER only:
- It will NOT override stop-loss, max-loss, or risk rules
- It will NOT directly execute trades
- It provides guidance to the local trading engine

Consider:
1. Price action and momentum direction
2. Technical indicator alignment
3. Current position and risk exposure
4. Recent news/events impact
5. Volatility environment

Return a JSON with: bias, confidence, action, rationale.
""")
        
        return "\n".join(sections)
    
    def build_context_from_snapshot(
        self,
        snapshot: Dict[str, Any],
        trigger_type: str = "manual",
        reason: str = "Manual analysis request",
    ) -> str:
        """Build context from a raw market snapshot dict.
        
        Args:
            snapshot: Market snapshot dictionary
            trigger_type: Type of trigger
            reason: Trigger reason
            
        Returns:
            Formatted context string
        """
        # Convert snapshot to EventPayload
        payload = EventPayload(
            trigger_type=trigger_type,
            reason=reason,
            timestamp=datetime.now(timezone.utc),
            symbol=snapshot.get("symbol", self.instrument),
            current_price=snapshot.get("current_price", 0.0),
            price_change_pct=snapshot.get("price_change_pct", 0.0),
            momentum=snapshot.get("momentum", 0.0),
            atr=snapshot.get("atr", 0.0),
            volatility=snapshot.get("volatility", 0.0),
            rsi=snapshot.get("rsi", 50.0),
            vix=snapshot.get("vix"),
            position=snapshot.get("position", 0),
            unrealized_pnl=snapshot.get("unrealized_pnl", 0.0),
            news_headlines=snapshot.get("news_headlines", []),
            recent_prices=snapshot.get("recent_prices", []),
        )
        
        return self.build_context(payload)
    
    def build_context_from_features(
        self,
        features_df,
        current_price: float,
        position: int = 0,
        unrealized_pnl: float = 0.0,
        trigger_type: str = "manual",
        reason: str = "Manual analysis request",
        news_headlines: Optional[List[str]] = None,
        vix: Optional[float] = None,
    ) -> str:
        """Build context from features DataFrame.
        
        Args:
            features_df: DataFrame with engineered features
            current_price: Current market price
            position: Current position
            unrealized_pnl: Unrealized P&L
            trigger_type: Type of trigger
            reason: Trigger reason
            news_headlines: Optional news headlines
            vix: Optional VIX value
            
        Returns:
            Formatted context string
        """
        # Extract indicators from last row
        last_row = features_df.iloc[-1]
        
        # Get recent prices
        recent_prices = features_df["close"].tail(20).tolist()
        
        # Calculate price change
        if len(recent_prices) >= 2:
            price_change_pct = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        else:
            price_change_pct = 0.0
        
        # Build snapshot dict
        snapshot = {
            "symbol": self.instrument,
            "current_price": current_price,
            "price_change_pct": price_change_pct,
            "momentum": float(last_row.get("momentum", last_row.get("MOM_10", 0.0))),
            "atr": float(last_row.get("ATR_14", last_row.get("atr", 0.0))),
            "volatility": float(last_row.get("volatility", last_row.get("VOLATILITY", 0.0))),
            "rsi": float(last_row.get("RSI_14", last_row.get("rsi", 50.0))),
            "vix": vix,
            "position": position,
            "unrealized_pnl": unrealized_pnl,
            "news_headlines": news_headlines or [],
            "recent_prices": recent_prices,
        }
        
        return self.build_context_from_snapshot(
            snapshot=snapshot,
            trigger_type=trigger_type,
            reason=reason,
        )
    
    def build_summary_context(
        self,
        context_type: str,
        daily_pnl: float,
        total_trades: int,
        winning_trades: int,
        current_position: int,
        key_events: List[str] = None,
    ) -> str:
        """Build summary context for market open/close summaries.
        
        Args:
            context_type: "market_open" or "market_close"
            daily_pnl: Daily P&L
            total_trades: Total trades today
            winning_trades: Winning trades today
            current_position: Current position
            key_events: Key events/notes
            
        Returns:
            Formatted summary context string
        """
        timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        
        sections = []
        sections.append("=" * 50)
        
        if context_type == "market_open":
            sections.append("MARKET OPEN SUMMARY")
            sections.append("=" * 50)
            sections.append(f"\nTime: {timestamp_str}")
            sections.append(f"Instrument: {self.instrument}")
            sections.append("\nThis is the start of the trading session.")
            sections.append("Please provide an outlook for today's session.")
        else:
            sections.append("MARKET CLOSE SUMMARY")
            sections.append("=" * 50)
            sections.append(f"\nTime: {timestamp_str}")
            sections.append(f"Instrument: {self.instrument}")
            sections.append("\n--- SESSION PERFORMANCE ---")
            sections.append(f"Daily P&L: ${daily_pnl:+.2f}")
            sections.append(f"Total Trades: {total_trades}")
            
            if total_trades > 0:
                win_rate = winning_trades / total_trades * 100
                sections.append(f"Win Rate: {win_rate:.1f}%")
            
            sections.append(f"Ending Position: {'FLAT' if current_position == 0 else f'{current_position} contracts'}")
        
        # Key events
        if key_events:
            sections.append("\n--- KEY EVENTS ---")
            for event in key_events[:5]:
                sections.append(f"• {event}")
        
        # Task
        sections.append("\n" + "=" * 50)
        sections.append("TASK")
        sections.append("=" * 50)
        
        if context_type == "market_open":
            sections.append("""
Provide a brief outlook for today's trading session:
1. Expected market direction
2. Key levels to watch
3. Risk considerations
4. Recommended bias for the day

Return JSON with: bias, confidence, action, rationale.
""")
        else:
            sections.append("""
Provide a brief review of today's session:
1. Overall performance assessment
2. What worked well
3. Areas for improvement
4. Outlook for tomorrow

Return JSON with: bias, confidence, action, rationale.
""")
        
        return "\n".join(sections)


def build_context(payload: EventPayload) -> str:
    """Convenience function to build context from payload.
    
    Args:
        payload: Event payload
        
    Returns:
        Formatted context string
    """
    builder = RAGContextBuilder(instrument=payload.symbol)
    return builder.build_context(payload)
