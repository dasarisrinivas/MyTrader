"""Reusable scoring helpers for RAG retrieval."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass
class RetrievalWeightConfig:
    similarity: float = 0.55
    recency: float = 0.30
    pnl_bias: float = 0.15

    def normalized(self) -> "RetrievalWeightConfig":
        total = self.similarity + self.recency + self.pnl_bias
        if total <= 0:
            return self
        return RetrievalWeightConfig(
            similarity=self.similarity / total,
            recency=self.recency / total,
            pnl_bias=self.pnl_bias / total,
        )


def recency_weight_from_timestamp(
    timestamp: str,
    half_life_days: float = 3.0,
    now: Optional[datetime] = None,
) -> float:
    """Return recency weight between 0-1 with exponential decay."""
    if not timestamp:
        return 0.5
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    try:
        ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)
    except ValueError:
        return 0.5
    days = max(0.0, (now - ts).total_seconds() / 86400)
    if days == 0:
        return 1.0
    # exponential decay with configurable half-life
    decay_constant = 0.693 / max(0.1, half_life_days)
    weight = pow(2.71828, -decay_constant * days)
    return max(0.0, min(1.0, weight))


def hybrid_trade_score(
    similarity_rank: float,
    recency_weight: float,
    pnl: Optional[float],
    weights: RetrievalWeightConfig | None = None,
) -> float:
    """Blend similarity, recency, and pnl bias into one priority score."""
    cfg = (weights or RetrievalWeightConfig()).normalized()
    similarity_component = max(0.0, min(1.0, similarity_rank))
    pnl_component = 0.5
    if pnl is not None:
        # Positive pnl -> >0.5, negative pnl -> <0.5
        pnl_component = max(0.0, min(1.0, 0.5 + (pnl / 1000.0)))
    score = (
        similarity_component * cfg.similarity
        + recency_weight * cfg.recency
        + pnl_component * cfg.pnl_bias
    )
    return max(0.0, min(1.0, score))


__all__ = [
    "RetrievalWeightConfig",
    "recency_weight_from_timestamp",
    "hybrid_trade_score",
]
