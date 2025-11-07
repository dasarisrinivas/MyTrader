"""Data schemas for LLM input/output structures."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional


@dataclass
class TradingContext:
    """Input context for LLM trade decision."""
    
    # Market data
    symbol: str
    current_price: float
    timestamp: datetime
    
    # Technical indicators
    rsi: float
    macd: float
    macd_signal: float
    macd_hist: float
    atr: float
    adx: Optional[float] = None
    bb_percent: Optional[float] = None
    
    # Sentiment data
    sentiment_score: float = 0.0
    sentiment_sources: Optional[Dict[str, float]] = None
    
    # Position information
    current_position: int = 0
    unrealized_pnl: float = 0.0
    
    # Risk metrics
    portfolio_heat: float = 0.0
    daily_pnl: float = 0.0
    win_rate: float = 0.0
    
    # Market regime
    market_regime: Optional[str] = None
    volatility_regime: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "symbol": self.symbol,
            "current_price": self.current_price,
            "timestamp": self.timestamp.isoformat(),
            "technical_indicators": {
                "rsi": self.rsi,
                "macd": self.macd,
                "macd_signal": self.macd_signal,
                "macd_hist": self.macd_hist,
                "atr": self.atr,
                "adx": self.adx,
                "bb_percent": self.bb_percent,
            },
            "sentiment": {
                "score": self.sentiment_score,
                "sources": self.sentiment_sources or {},
            },
            "position": {
                "current": self.current_position,
                "unrealized_pnl": self.unrealized_pnl,
            },
            "risk_metrics": {
                "portfolio_heat": self.portfolio_heat,
                "daily_pnl": self.daily_pnl,
                "win_rate": self.win_rate,
            },
            "market_conditions": {
                "regime": self.market_regime,
                "volatility": self.volatility_regime,
            }
        }


@dataclass
class TradeRecommendation:
    """LLM-generated trade recommendation."""
    
    # Core decision
    trade_decision: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0.0 to 1.0
    
    # Position sizing
    suggested_position_size: int = 1
    
    # Risk management
    suggested_stop_loss: Optional[float] = None
    suggested_take_profit: Optional[float] = None
    
    # LLM reasoning
    reasoning: str = ""
    key_factors: list[str] = field(default_factory=list)
    risk_assessment: str = ""
    
    # Sentiment analysis
    sentiment_score: float = 0.0  # -1.0 (very bearish) to +1.0 (very bullish)
    
    # Metadata
    model_name: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    processing_time_ms: float = 0.0
    
    # Raw response for audit
    raw_response: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "trade_decision": self.trade_decision,
            "confidence": self.confidence,
            "suggested_position_size": self.suggested_position_size,
            "suggested_stop_loss": self.suggested_stop_loss,
            "suggested_take_profit": self.suggested_take_profit,
            "reasoning": self.reasoning,
            "key_factors": self.key_factors,
            "risk_assessment": self.risk_assessment,
            "sentiment_score": self.sentiment_score,
            "model_name": self.model_name,
            "timestamp": self.timestamp.isoformat(),
            "processing_time_ms": self.processing_time_ms,
            "raw_response": self.raw_response,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> TradeRecommendation:
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class TradeOutcome:
    """Record of trade execution and result."""
    
    # Trade identification
    order_id: int
    symbol: str
    timestamp: datetime
    
    # Trade details
    action: str  # "BUY", "SELL"
    quantity: int
    entry_price: float
    exit_price: Optional[float] = None
    
    # Trade result
    realized_pnl: float = 0.0
    trade_duration_minutes: float = 0.0
    outcome: str = "OPEN"  # "OPEN", "WIN", "LOSS", "BREAKEVEN"
    
    # LLM prediction at entry
    llm_recommendation: Optional[TradeRecommendation] = None
    
    # Market context at entry
    entry_context: Optional[TradingContext] = None
    
    # Exit context (if closed)
    exit_context: Optional[TradingContext] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "realized_pnl": self.realized_pnl,
            "trade_duration_minutes": self.trade_duration_minutes,
            "outcome": self.outcome,
            "llm_recommendation": self.llm_recommendation.to_dict() if self.llm_recommendation else None,
            "entry_context": self.entry_context.to_dict() if self.entry_context else None,
            "exit_context": self.exit_context.to_dict() if self.exit_context else None,
        }
