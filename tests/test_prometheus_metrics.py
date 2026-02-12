import time
from prometheus_client import CollectorRegistry

from shree.observability import prometheus_metrics as pm


def test_metrics_init_and_increment():
    settings = type("S", (), {})()
    settings.observability = type("O", (), {})()
    settings.observability.prometheus_enabled = True
    settings.observability.prometheus_addr = "127.0.0.1"
    settings.observability.prometheus_port = 9000
    settings.observability.env_label = "test"

    registry = CollectorRegistry()
    m = pm.init_metrics(settings, registry=registry)
    assert m is not None

    symbol = "MES"
    # set bar age
    pm.set_bar_age(symbol, "1m", 12.3)
    # increment stale block
    pm.inc_stale_block(symbol)
    pm.inc_decision(symbol, "HOLD")
    pm.inc_cancel_call(symbol, "STALE_LIVE_BARS", "none")
    pm.inc_canceled_entries(symbol, "STALE_LIVE_BARS", 2)

    # Verify some metrics present in registry
    # Look up by metric name (Counter base names without _total suffix)
    names = {m.name for m in registry.collect()}
    assert "shree_live_bar_age_seconds" in names
    assert "shree_stale_live_bars_blocks" in names
    assert "shree_pending_entry_orders_canceled" in names
