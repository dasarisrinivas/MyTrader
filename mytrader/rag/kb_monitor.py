"""Telemetry helper for knowledge-base (RAG) usage."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger


def _now() -> float:
    return time.monotonic()


@dataclass
class KnowledgeBaseUsageTracker:
    """Tracks KB/RAG usage and periodically logs aggregated metrics."""

    backend: str = "off"
    remote_active: bool = False
    interval_seconds: int = 60
    _window_start: float = field(default_factory=_now)
    _queries: int = 0
    _cache_hits: int = 0
    _remote_calls: int = 0
    _avoided_remote: int = 0

    def configure(self, backend: Optional[str], remote_active: bool) -> None:
        """Update backend label + remote status shown in logs."""
        if backend:
            self.backend = backend
        self.remote_active = remote_active

    def record_query(self, cache_hit: bool = False, remote_call: bool = False) -> None:
        """Record a KB query."""
        self._queries += 1
        if cache_hit:
            self._cache_hits += 1
        if remote_call:
            self._remote_calls += 1
        self._maybe_log()

    def record_avoidance(self) -> None:
        """Record that we intentionally skipped a remote OpenSearch call."""
        self._avoided_remote += 1
        self._maybe_log()

    def _maybe_log(self) -> None:
        """Log aggregate metrics once the reporting window elapses."""
        now = _now()
        elapsed = now - self._window_start
        if elapsed < self.interval_seconds:
            return

        qpm = (self._queries / elapsed * 60.0) if elapsed > 0 else 0.0
        hit_rate = (self._cache_hits / self._queries * 100.0) if self._queries else 0.0
        logger.info(
            "📈 KB telemetry: backend=%s remote_active=%s qpm=%.2f cache_hit_rate=%.1f%% "
            "remote_calls=%d avoided_remote=%d",
            self.backend,
            "yes" if self.remote_active else "no",
            qpm,
            hit_rate,
            self._remote_calls,
            self._avoided_remote,
        )
        self._reset_window(now)

    def _reset_window(self, now: float) -> None:
        self._window_start = now
        self._queries = 0
        self._cache_hits = 0
        self._remote_calls = 0
        self._avoided_remote = 0


kb_usage_tracker = KnowledgeBaseUsageTracker()

