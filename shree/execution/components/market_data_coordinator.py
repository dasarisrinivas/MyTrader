"""Handles market data context fan-out to other agents."""

import json
from typing import Dict, Any, Optional

from ...utils.logger import logger
from ...utils.timezone_utils import now_cst


class MarketDataCoordinator:
    """Publishes feature snapshots and keeps cached context fresh."""

    def __init__(self, manager: "LiveTradingManager"):  # noqa: F821 (forward reference)
        self.manager = manager

    def publish_feature_snapshot(self, latest: Dict[str, Any], current_price: float, timestamp: Optional[str] = None) -> None:
        if not self.manager.agent_bus:
            return
        payload = {
            "timestamp": timestamp or now_cst().isoformat(),
            "price": float(latest.get("close", current_price)),
            "rsi": float(latest.get("RSI_14", 50)),
            "macd_hist": float(latest.get("MACDhist_12_26_9", 0)),
            "atr": float(latest.get("ATR_14", 0)),
            "volume_ratio": float(latest.get("volume_ratio", 1)),
        }
        self.manager.agent_bus.publish("features", payload, producer="live_manager", ttl_seconds=120)

    def refresh_external_context(self) -> None:
        if not self.manager.agent_bus:
            return
        directory = self.manager._external_context_dir
        if not directory.exists():
            return
        context_files = [
            ("news", "news_sentiment.json", "news"),
            ("macro", "macro_state.json", "macro"),
        ]
        for label, filename, channel in context_files:
            path = directory / filename
            if not path.exists():
                continue
            try:
                mtime = path.stat().st_mtime
                if self.manager._context_refresh_mtimes.get(channel) == mtime:
                    continue
                data = json.loads(path.read_text())
                data.setdefault("source", label)
                data.setdefault("timestamp", now_cst().isoformat())
                self.manager.agent_bus.publish(channel, data, producer="context_cache", ttl_seconds=900)
                self.manager._context_refresh_mtimes[channel] = mtime
            except Exception as context_error:  # noqa: BLE001
                logger.debug(f"Context refresh failed for {path}: {context_error}")

    def publish_account_context(self) -> None:
        if not self.manager.agent_bus or not self.manager.tracker:
            return
        self.manager._update_status_from_tracker()
        current_equity = self.manager.tracker.get_current_equity()
        peak_equity = getattr(self.manager.tracker, "peak_equity", current_equity)
        drawdown_pct = 0.0
        if peak_equity:
            drawdown_pct = (current_equity - peak_equity) / peak_equity
        payload = {
            "equity": current_equity,
            "daily_pnl": self.manager.status.daily_pnl,
            "active_drawdown_pct": drawdown_pct,
            "portfolio_heat": getattr(self.manager.risk, "portfolio_heat", 0.0) if self.manager.risk else 0.0,
            "timestamp": now_cst().isoformat(),
        }
        self.manager.agent_bus.publish("account", payload, producer="risk_monitor", ttl_seconds=300)
