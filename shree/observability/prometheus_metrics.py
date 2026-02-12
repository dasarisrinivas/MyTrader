"""Prometheus metrics helpers for Shree.

This module provides an opt-in, minimal instrumentation API. Call
`init_metrics(settings, registry=None)` during startup when Prometheus is
enabled. All helper functions are no-ops if metrics weren't initialized.

Design notes:
- Keep label cardinality low: symbol, timeframe (e.g. "1m"), env
- Use an injected CollectorRegistry in tests to avoid global state
"""
from __future__ import annotations

from typing import Optional

try:
    from prometheus_client import CollectorRegistry, Gauge, Counter, start_http_server
except Exception:  # pragma: no cover - prometheus_client optional
    CollectorRegistry = None
    Gauge = None
    Counter = None
    start_http_server = None


_METRICS = None


class _Metrics:
    def __init__(self, registry: CollectorRegistry, env_label: str = "local"):
        self.registry = registry
        self.env = env_label
        # Gauges
        self.live_bar_age = Gauge(
            "shree_live_bar_age_seconds",
            "Age in seconds of the latest live bar",
            ["symbol", "timeframe", "env"],
            registry=self.registry,
        )
        self.stale_episode_active = Gauge(
            "shree_stale_episode_active",
            "Indicator (0/1) whether a stale-live-bar episode is active",
            ["symbol", "env"],
            registry=self.registry,
        )

        # Counters
        self.stale_blocks_total = Counter(
            "shree_stale_live_bars_blocks_total",
            "Total number of stale-live-bars blocks observed",
            ["symbol", "env"],
            registry=self.registry,
        )
        self.decisions_total = Counter(
            "shree_decisions_total",
            "Decision cycles by action",
            ["symbol", "env", "action"],
            registry=self.registry,
        )
        self.pending_entry_orders_canceled_total = Counter(
            "shree_pending_entry_orders_canceled_total",
            "Number of pending entry parent orders canceled",
            ["symbol", "env", "reason"],
            registry=self.registry,
        )
        self.cancel_entries_calls_total = Counter(
            "shree_cancel_entries_calls_total",
            "Number of calls to cancel pending entry orders (outcome label)",
            ["symbol", "env", "reason", "outcome"],
            registry=self.registry,
        )

    # Helper methods
    def set_bar_age(self, symbol: str, timeframe: str, age_seconds: float) -> None:
        try:
            self.live_bar_age.labels(symbol=symbol, timeframe=timeframe, env=self.env).set(float(age_seconds) if age_seconds is not None else 0.0)
        except Exception:
            pass

    def set_stale_episode_active(self, symbol: str, active: bool) -> None:
        try:
            self.stale_episode_active.labels(symbol=symbol, env=self.env).set(1 if active else 0)
        except Exception:
            pass

    def inc_stale_block(self, symbol: str) -> None:
        try:
            self.stale_blocks_total.labels(symbol=symbol, env=self.env).inc()
        except Exception:
            pass

    def inc_decision(self, symbol: str, action: str) -> None:
        try:
            self.decisions_total.labels(symbol=symbol, env=self.env, action=action).inc()
        except Exception:
            pass

    def inc_canceled_entries(self, symbol: str, reason: str, count: int = 1) -> None:
        try:
            self.pending_entry_orders_canceled_total.labels(symbol=symbol, env=self.env, reason=reason).inc(count)
        except Exception:
            pass

    def inc_cancel_call(self, symbol: str, reason: str, outcome: str) -> None:
        try:
            self.cancel_entries_calls_total.labels(symbol=symbol, env=self.env, reason=reason, outcome=outcome).inc()
        except Exception:
            pass


def init_metrics(settings, registry: Optional["CollectorRegistry"] = None) -> Optional[_Metrics]:
    """Initialize metrics if prometheus_client is available and enabled.

    Returns a metrics handle or None.
    """
    global _METRICS
    if CollectorRegistry is None or Gauge is None or Counter is None:
        # prometheus_client not installed
        import sys
        print(f"[Prometheus Init] prometheus_client not available", file=sys.stderr)
        return None

    obs = getattr(settings, "observability", None)
    import sys
    print(f"[Prometheus Init] observability object: {obs}", file=sys.stderr)
    
    enabled = False
    env_label = "local"
    addr = "0.0.0.0"
    port = 8000
    if obs is not None:
        enabled = bool(getattr(obs, "prometheus_enabled", False))
        env_label = getattr(obs, "env_label", env_label)
        addr = getattr(obs, "prometheus_addr", addr)
        port = int(getattr(obs, "prometheus_port", port))
    
    # Debug: print what we're seeing
    print(f"[Prometheus Init] enabled={enabled}, port={port}, addr={addr}, env={env_label}", file=sys.stderr)

    if not enabled:
        print(f"[Prometheus Init] Prometheus disabled, returning None", file=sys.stderr)
        return None

    reg = registry or CollectorRegistry()
    try:
        _METRICS = _Metrics(reg, env_label=env_label)
    except Exception as e:
        print(f"[Prometheus Init] Failed to create _Metrics: {e}", file=sys.stderr)
        _METRICS = None
        return None

    # Start HTTP server (best-effort). This uses prometheus_client.start_http_server which
    # exposes the default registry. Because we use a custom registry, we can't use the
    # basic start_http_server without exposing the registry via a WSGI app. To keep this
    # simple and safe for tests, only start the default server if no custom registry passed.
    if registry is None and start_http_server is not None:
        try:
            print(f"[Prometheus Init] Starting HTTP server on {addr}:{port}", file=sys.stderr)
            start_http_server(port, addr)
            print(f"[Prometheus Init] HTTP server started successfully", file=sys.stderr)
        except Exception as e:
            print(f"[Prometheus Init] Failed to start HTTP server: {e}", file=sys.stderr)
            pass

    return _METRICS


def get_metrics() -> Optional[_Metrics]:
    return _METRICS


# Convenience wrappers
def set_bar_age(symbol: str, timeframe: str, age_seconds: float) -> None:
    m = get_metrics()
    if m:
        m.set_bar_age(symbol, timeframe, age_seconds)


def set_stale_episode_active(symbol: str, active: bool) -> None:
    m = get_metrics()
    if m:
        m.set_stale_episode_active(symbol, active)


def inc_stale_block(symbol: str) -> None:
    m = get_metrics()
    if m:
        m.inc_stale_block(symbol)


def inc_decision(symbol: str, action: str) -> None:
    m = get_metrics()
    if m:
        m.inc_decision(symbol, action)


def inc_canceled_entries(symbol: str, reason: str, count: int = 1) -> None:
    m = get_metrics()
    if m:
        m.inc_canceled_entries(symbol, reason, count)


def inc_cancel_call(symbol: str, reason: str, outcome: str) -> None:
    m = get_metrics()
    if m:
        m.inc_cancel_call(symbol, reason, outcome)
