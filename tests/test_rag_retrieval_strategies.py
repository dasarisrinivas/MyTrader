"""Tests for retrieval scoring helpers."""
from datetime import datetime, timedelta, timezone

from mytrader.rag.retrieval_strategies import (
    recency_weight_from_timestamp,
    hybrid_trade_score,
)


def test_recency_weight_respects_decay():
    now = datetime.now(timezone.utc)
    recent = recency_weight_from_timestamp(now.isoformat())
    ten_days = recency_weight_from_timestamp((now - timedelta(days=10)).isoformat())
    assert recent > 0.95
    assert ten_days < recent


def test_hybrid_trade_score_combines_inputs():
    strong = hybrid_trade_score(1.0, 1.0, 200)
    weak = hybrid_trade_score(0.2, 0.3, -200)
    assert strong > weak
