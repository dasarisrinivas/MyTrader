"""Telegram alert notifier for SPY options bot.

Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the environment to enable.
If not configured, all alerts fall back to log-only — the bot runs normally.

No new external dependencies: uses stdlib urllib.request for HTTP.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

from logger import logger

ET = ZoneInfo("America/New_York")

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class Notifier:
    """Send trade and risk alerts via Telegram; falls back to log if unconfigured."""

    def __init__(self) -> None:
        self.token: str | None = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id: str | None = os.getenv("TELEGRAM_CHAT_ID")
        self.enabled: bool = bool(self.token and self.chat_id)

        if self.enabled:
            logger.info("Notifier: Telegram alerts enabled")
        else:
            logger.info(
                "Notifier: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — "
                "alerts will be logged only"
            )

    # ------------------------------------------------------------------
    # Core send
    # ------------------------------------------------------------------

    def send(self, message: str) -> None:
        """Send a message. Falls back to logger.info if Telegram not configured."""
        logger.info(f"[ALERT] {message}")
        if not self.enabled:
            return
        try:
            url = _TELEGRAM_API.format(token=self.token)
            payload = json.dumps({
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status != 200:
                    logger.warning(f"Telegram returned HTTP {resp.status}")
        except urllib.error.URLError as exc:
            logger.warning(f"Telegram send failed (network): {exc}")
        except Exception as exc:
            logger.warning(f"Telegram send failed: {exc}")

    # ------------------------------------------------------------------
    # Convenience formatters
    # ------------------------------------------------------------------

    def on_trade_open(self, position: dict) -> None:
        """Alert when a new position is opened."""
        right_label = "PUT" if position.get("right") == "P" else "CALL"
        expiry = position.get("expiry", "")
        expiry_fmt = f"{expiry[4:6]}-{expiry[6:8]}-{expiry[:4]}" if len(expiry) == 8 else expiry
        pdt_used = position.get("_pdt_count", "?")

        msg = (
            f"🟢 <b>TRADE OPENED</b>\n"
            f"{position.get('symbol')} {int(position.get('strike', 0))}{right_label} exp {expiry_fmt}\n"
            f"Premium: ${position.get('entry_premium', 0):.2f} | "
            f"Delta: {position.get('entry_delta', 0):.2f} | "
            f"Theta: {position.get('entry_theta', 0):.4f}\n"
            f"Target: ${position.get('profit_target_price', 0):.2f} | "
            f"Stop: ${position.get('stop_loss_price', 0):.2f}\n"
            f"Strategy: {position.get('strategy_type', '?')}\n"
            f"Mode: {_trading_mode()}"
        )
        self.send(msg)

    def on_trade_close(self, position: dict) -> None:
        """Alert when a position is closed (any reason)."""
        right_label = "PUT" if position.get("right") == "P" else "CALL"
        expiry = position.get("expiry", "")
        expiry_fmt = f"{expiry[4:6]}-{expiry[6:8]}-{expiry[:4]}" if len(expiry) == 8 else expiry
        pnl = position.get("pnl", 0) or 0
        pnl_sign = "+" if pnl >= 0 else ""
        reason = position.get("close_reason", "unknown")

        msg = (
            f"{'🟢' if pnl >= 0 else '🔴'} <b>POSITION CLOSED</b> — {reason}\n"
            f"{position.get('symbol')} {int(position.get('strike', 0))}{right_label} exp {expiry_fmt}\n"
            f"Entry: ${position.get('entry_premium', 0):.2f} → "
            f"Close: ${position.get('close_premium', 0):.2f}\n"
            f"P&amp;L: {pnl_sign}${pnl:.2f}\n"
            f"Mode: {_trading_mode()}"
        )
        self.send(msg)

    def on_pdt_warning(self, slots_remaining: int) -> None:
        """Alert when PDT slots are running low."""
        msg = (
            f"⚠️ <b>PDT WARNING</b>\n"
            f"Only {slots_remaining}/3 PDT slot(s) remaining this week.\n"
            f"Strangle entries require 2 slots — may degrade to single-leg."
        )
        self.send(msg)

    def on_risk_stop(self, position: dict, reason: str) -> None:
        """Alert when a risk stop (loss or delta) fires."""
        label = position.get("label", str(position.get("position_id", "?")))
        msg = (
            f"🔴 <b>RISK STOP TRIGGERED</b> — {reason}\n"
            f"Position: {label}\n"
            f"Closing order placed."
        )
        self.send(msg)

    def on_connection_lost(self) -> None:
        msg = "📡 <b>IBKR CONNECTION LOST</b>\nBot attempting to reconnect..."
        self.send(msg)

    def on_connection_restored(self) -> None:
        msg = "✅ <b>IBKR CONNECTION RESTORED</b>\nResuming position monitoring."
        self.send(msg)

    def on_emergency_close(self, position: dict, reason: str) -> None:
        """Alert for emergency gamma close (Thursday rapid move)."""
        msg = (
            f"🚨 <b>EMERGENCY CLOSE</b>\n"
            f"Reason: {reason}\n"
            f"All positions being closed immediately."
        )
        self.send(msg)


def _trading_mode() -> str:
    """Return 'LIVE' or 'PAPER' based on config."""
    try:
        import config as cfg
        return "LIVE" if cfg.LIVE_TRADING else "PAPER"
    except Exception:
        return "?"
