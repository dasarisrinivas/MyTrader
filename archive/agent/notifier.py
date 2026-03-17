"""Notifier — sends Telegram alerts for anomaly reports.

Uses the Telegram Bot API directly (aiohttp not required — uses urllib).
Gracefully degrades if Telegram is disabled or credentials are missing.
Reads credentials from env vars, falling back to config.yaml.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
import urllib.error
import urllib.parse
import yaml
from typing import Any, Dict, Optional
from pathlib import Path

log = logging.getLogger("agent.notifier")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_telegram_from_yaml() -> Dict[str, str]:
    """Attempt to read bot_token and chat_id from config.yaml."""
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        tg = cfg.get("telegram", {})
        return {
            "bot_token": str(tg.get("bot_token", "")),
            "chat_id": str(tg.get("chat_id", "")),
        }
    except Exception:
        return {}


class TelegramNotifier:
    """Sends alert messages via Telegram Bot API."""

    def __init__(self, config: Dict[str, Any]):
        tg_cfg = config.get("telegram", {})
        self.enabled = tg_cfg.get("enabled", False)
        self.max_message_length = tg_cfg.get("max_message_length", 4096)

        # Read credentials from environment variables first
        token_env = tg_cfg.get("bot_token_env", "TELEGRAM_BOT_TOKEN")
        chat_env = tg_cfg.get("chat_id_env", "TELEGRAM_CHAT_ID")
        self.bot_token = os.environ.get(token_env, "")
        self.chat_id = os.environ.get(chat_env, "")

        # Fallback: read from config.yaml if env vars are empty
        if not self.bot_token or not self.chat_id:
            yaml_creds = _load_telegram_from_yaml()
            if not self.bot_token:
                self.bot_token = yaml_creds.get("bot_token", "")
            if not self.chat_id:
                self.chat_id = yaml_creds.get("chat_id", "")

        if self.enabled:
            if not self.bot_token or self.bot_token.startswith("your-"):
                log.warning("Telegram disabled: invalid bot token (set %s env var)", token_env)
                self.enabled = False
            elif not self.chat_id or self.chat_id.startswith("your-"):
                log.warning("Telegram disabled: invalid chat ID (set %s env var)", chat_env)
                self.enabled = False
            else:
                log.info("Telegram notifications enabled (chat: %s)", self.chat_id)

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message via Telegram Bot API.
        
        Args:
            text: Message text (supports HTML formatting)
            parse_mode: "HTML" or "Markdown"
            
        Returns:
            True if sent successfully, False otherwise.
        """
        if not self.enabled:
            log.debug("Telegram disabled — message not sent")
            return False

        # Truncate if needed
        if len(text) > self.max_message_length:
            text = text[: self.max_message_length - 20] + "\n... [truncated]"

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = json.dumps({
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    log.info("Telegram message sent successfully")
                    return True
                else:
                    log.warning("Telegram API returned status %d", resp.status)
                    return False
        except urllib.error.HTTPError as exc:
            log.warning("Telegram API HTTP error: %s %s", exc.code, exc.reason)
            return False
        except urllib.error.URLError as exc:
            log.warning("Telegram API URL error: %s", exc.reason)
            return False
        except Exception as exc:
            log.error("Telegram send failed: %s", exc)
            return False

    def send_anomaly_alert(self, alert_text: str) -> bool:
        """Convenience wrapper: send an anomaly alert."""
        return self.send_message(alert_text, parse_mode="HTML")

    def send_heartbeat(self) -> bool:
        """Send a periodic heartbeat message to confirm agent is running."""
        if not self.enabled:
            return False
        from datetime import datetime
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("America/Chicago"))
        text = f"💓 <b>Trading Analyst Agent Heartbeat</b>\n<code>{now.strftime('%Y-%m-%d %H:%M CST')}</code>\nAgent is running and monitoring."
        return self.send_message(text)
