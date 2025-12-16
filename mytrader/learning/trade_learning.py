"""Local trade outcome + history snapshot recorder."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..utils.logger import logger


@dataclass
class ExecutionMetrics:
    slippage: float = 0.0
    latency_ms: Optional[float] = None
    immediate_fill: bool = False


@dataclass
class TradeLearningPayload:
    trade_cycle_id: str
    symbol: str
    contract: str
    quantity: int
    side: str
    entry_time: str
    entry_price: float
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    signal_type: Optional[str] = None
    signal_confidence: Optional[float] = None
    regime: Optional[str] = None
    volatility: Optional[str] = None
    advisory: Optional[Dict[str, Any]] = None
    risk_parameters: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, Any] = field(default_factory=dict)
    execution_metrics: ExecutionMetrics = field(default_factory=ExecutionMetrics)
    reason_codes: List[str] = field(default_factory=list)
    outcome: str = "OPEN"
    pnl: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["execution_metrics"] = asdict(self.execution_metrics)
        return payload


class TradeLearningRecorder:
    """Persist structured trade outcomes + supporting context."""

    def __init__(self, outcomes_dir: str, history_dir: str) -> None:
        self.outcomes_root = Path(outcomes_dir)
        self.history_root = Path(history_dir)
        self.outcomes_root.mkdir(parents=True, exist_ok=True)
        self.history_root.mkdir(parents=True, exist_ok=True)

    def record_outcome(self, payload: TradeLearningPayload) -> Path:
        """Persist trade outcome JSON."""
        trade_date = payload.entry_time[:10]
        target_dir = self.outcomes_root / trade_date
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{payload.trade_cycle_id}.json"
        data = payload.to_dict()
        with target_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        logger.info(f"📘 Learning record written: {target_path}")
        return target_path

    def record_history_snapshot(
        self,
        trade_cycle_id: str,
        candles: Iterable[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Path:
        """Persist a lightweight history snapshot for ingestion."""
        target_path = self.history_root / f"{trade_cycle_id}.json"
        if target_path.exists():
            return target_path
        snapshot = {
            "trade_cycle_id": trade_cycle_id,
            "generated_at": datetime.utcnow().isoformat(),
            "context": context,
            "candles": list(candles),
        }
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("w", encoding="utf-8") as fh:
            json.dump(snapshot, fh, indent=2)
        logger.info(f"📚 History snapshot stored: {target_path}")
        return target_path
