"""Knowledge-base and context helpers."""

import time
from typing import Any, Dict, Optional

from ...utils.logger import logger
from ...utils.timezone_utils import now_cst


class ContextManager:
    """Manages cached KB lookups and local KB queries."""

    def __init__(self, manager: "LiveTradingManager"):  # noqa: F821 (forward reference)
        self.manager = manager

    def build_kb_cache_key(self, trend: str, volatility: str, action: str) -> str:
        return f"{action}:{trend}:{volatility}"

    def get_cached_kb_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        cached = self.manager._kb_cache.get(cache_key)
        if not cached:
            return None
        expires_at, payload = cached
        if expires_at < time.time():
            self.manager._kb_cache.pop(cache_key, None)
            return None
        return payload

    def set_cached_kb_result(self, cache_key: str, payload: Dict[str, Any]) -> None:
        expires = time.time() + max(5.0, float(self.manager._kb_cache_ttl or 0))
        self.manager._kb_cache[cache_key] = (expires, payload)
        if len(self.manager._kb_cache) > self.manager._kb_cache_limit:
            oldest_key = next(iter(self.manager._kb_cache))
            if oldest_key != cache_key:
                self.manager._kb_cache.pop(oldest_key, None)

    def query_local_knowledge_base(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if not self.manager._local_kb:
            return {}
        try:
            return self.manager._local_kb.query(context)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Local KB query failed: {exc}")
            return {}

    def refresh_hybrid_context(self, pipeline_result: Any) -> None:
        """Update cached hybrid status from pipeline output."""
        if not pipeline_result or not getattr(pipeline_result, "rule_engine", None):
            return
        try:
            rule_engine = pipeline_result.rule_engine
            self.manager.status.hybrid_market_trend = getattr(rule_engine, "market_trend", "")
            self.manager.status.hybrid_volatility_regime = getattr(rule_engine, "volatility_regime", "")
            self.manager.status.filters_applied = getattr(rule_engine, "filters_passed", [])
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Hybrid context refresh skipped: {exc}")

    def fetch_rag_context(
        self,
        features,
        signal_action: str,
        structural_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fetch RAG stats, buckets, and similar trades."""
        m = self.manager
        rag_context: Dict[str, Any] = {}

        if not m.rag_storage:
            return rag_context

        volatility = features.iloc[-1].get("volatility_5m", 0.0)
        if volatility > 0.002:
            vol_bucket = "HIGH"
        elif volatility < 0.0005:
            vol_bucket = "LOW"
        else:
            vol_bucket = "MEDIUM"

        hour = now_cst().hour
        if 8 <= hour < 11:
            time_bucket = "MORNING"
        elif 11 <= hour < 14:
            time_bucket = "MIDDAY"
        else:
            time_bucket = "CLOSE"

        buckets = {
            "volatility": vol_bucket,
            "time_of_day": time_bucket,
            "signal_type": signal_action,
        }

        stats = m.rag_storage.get_bucket_stats(buckets)
        similar_trades = m.rag_storage.retrieve_similar_trades(buckets, limit=5)

        rag_context = {
            "buckets": buckets,
            "stats": stats,
            "similar_trades_count": len(similar_trades),
            "structure": structural_metrics,
        }
        return rag_context
