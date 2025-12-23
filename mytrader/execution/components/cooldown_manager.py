"""Cooldown tracking and persistence."""

from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from ...utils.logger import logger
from ...utils.timezone_utils import format_cst, utc_to_cst


class CooldownManager:
    """Handles cooldown calculations and persistence across restarts."""

    DEFAULT_COOLDOWN_SECONDS = 300
    MIN_COOLDOWN_MINUTES = 1
    MAX_COOLDOWN_MINUTES = 60
    COOLDOWN_WARNING_MINUTES = 30
    PERSISTED_COOLDOWN_MAX_AGE = timedelta(days=7)
    FUTURE_COOLDOWN_TOLERANCE_SECONDS = 120

    def __init__(self, manager: "LiveTradingManager"):  # noqa: F821 (forward reference)
        self.manager = manager

    def sanitize_cooldown_minutes(self, raw_value: Any) -> int:
        default_minutes = self.DEFAULT_COOLDOWN_SECONDS // 60
        try:
            minutes = int(raw_value)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid trade_cooldown_minutes=%r; defaulting to %d minutes",
                raw_value,
                default_minutes,
            )
            return default_minutes
        clamped = max(self.MIN_COOLDOWN_MINUTES, min(self.MAX_COOLDOWN_MINUTES, minutes))
        if clamped != minutes:
            logger.warning(
                "trade_cooldown_minutes=%d outside [%d, %d]; clamped to %d",
                minutes,
                self.MIN_COOLDOWN_MINUTES,
                self.MAX_COOLDOWN_MINUTES,
                clamped,
            )
        elif clamped >= self.COOLDOWN_WARNING_MINUTES:
            logger.warning(
                "trade_cooldown_minutes=%d is unusually high; verify this is intentional",
                clamped,
            )
        return clamped

    def record_last_trade_timestamp(self, timestamp: Optional[datetime] = None) -> None:
        manager = self.manager
        ts = (timestamp or datetime.now(timezone.utc)).astimezone(timezone.utc)
        manager._last_trade_time = ts
        tracker = getattr(getattr(manager.executor, "order_tracker", None), "record_last_trade_time", None)
        if tracker:
            try:
                manager.executor.order_tracker.record_last_trade_time(manager.settings.data.ibkr_symbol, ts)
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"Cooldown persistence skipped: {exc}")

    def load_persistent_cooldown_state(self) -> None:
        manager = self.manager
        tracker = getattr(getattr(manager.executor, "order_tracker", None), "get_last_trade_time", None)
        if not tracker:
            return
        try:
            last_trade = manager.executor.order_tracker.get_last_trade_time(manager.settings.data.ibkr_symbol)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Cooldown state restore skipped: {exc}")
            return
        validated = self.validate_persisted_trade_time(last_trade)
        if validated:
            manager._last_trade_time = validated
            logger.info(
                "⏱️ Cooldown resume: last trade at %s",
                format_cst(utc_to_cst(validated)),
            )
        else:
            manager._last_trade_time = None

    def validate_persisted_trade_time(self, timestamp: Optional[datetime]) -> Optional[datetime]:
        if not timestamp:
            return None
        now_utc = datetime.now(timezone.utc)
        if timestamp > now_utc + timedelta(seconds=self.FUTURE_COOLDOWN_TOLERANCE_SECONDS):
            logger.warning(
                "Ignoring future last trade timestamp: %s",
                format_cst(utc_to_cst(timestamp)),
            )
            return None
        if now_utc - timestamp > self.PERSISTED_COOLDOWN_MAX_AGE:
            logger.warning(
                "Ignoring stale last trade timestamp from %s (> %s old)",
                format_cst(utc_to_cst(timestamp)),
                self.PERSISTED_COOLDOWN_MAX_AGE,
            )
            return None
        return timestamp

    def apply_manual_state_reset(self) -> None:
        manager = self.manager
        if not manager.executor:
            return
        symbol = getattr(manager.settings.data, "ibkr_symbol", "UNKNOWN")
        tracker = getattr(manager.executor, "order_tracker", None)
        if tracker and hasattr(tracker, "reset_symbol_state"):
            try:
                tracker.reset_symbol_state(symbol)
            except Exception as exc:  # noqa: BLE001
                logger.warning("⚠️  Failed to clear persisted cooldown for %s: %s", symbol, exc)
        manager._last_trade_time = None
        manager.status.cooldown_remaining_seconds = 0
        force_release = getattr(manager.executor, "force_release_order_lock", None)
        if callable(force_release):
            force_release("manual reset")
        else:
            release = getattr(manager.executor, "_release_order_lock", None)
            if callable(release):
                release("manual reset")
        logger.warning("♻️  Manual override applied: cooldown cleared and order lock released")
