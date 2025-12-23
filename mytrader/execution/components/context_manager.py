"""Knowledge-base and context helpers."""

import time
from typing import Any, Dict, Optional

from ..utils.logger import logger


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
