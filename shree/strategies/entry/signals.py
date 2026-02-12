"""Signal dataclasses used across all entry modules.

Contains the lightweight value objects that carry entry signal data
between entry modules and the strategy engine.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PullbackAnalysis:
    """Detailed pullback analysis result."""

    is_valid: bool = False
    score: float = 0.0
    depth_pct: float = 0.0
    touch_level: str = ""  # "EMA9", "EMA21", "VWAP", "NONE"
    confirmation: str = ""  # "BULLISH_ENGULF", "STRONG_CLOSE", "HIGHER_LOW", "NONE"
    reasons: List[str] = field(default_factory=list)


@dataclass
class EntrySignal:
    """Signal emitted by an entry module.

    Attributes:
        action: ``"BUY"``, ``"SELL"``, or ``"HOLD"``.
        confidence: Probability estimate in [0.0, 1.0].
        reason: Human-readable explanation of why the signal was generated.
        stop_loss: Suggested stop-loss price (optional).
        take_profit: Suggested take-profit price (optional).
        entry_type: Label such as ``"CONTINUATION"``, ``"EXHAUSTION"``, etc.
        session_window: Which session window was active when the signal was generated.
        metadata: Arbitrary key/value bag for downstream consumers.
    """

    action: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0.0 to 1.0
    reason: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_type: str = ""
    session_window: str = ""
    metadata: Optional[Dict[str, Any]] = None

    @property
    def is_actionable(self) -> bool:
        """Return ``True`` if the signal is a BUY or SELL with confidence > 0.5."""
        return self.action in ("BUY", "SELL") and self.confidence > 0.5
