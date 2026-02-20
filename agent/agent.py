#!/usr/bin/env python3
"""Autonomous Trading Analyst Agent — main event loop.

Monitors a live MES futures trading system and autonomously initiates
diagnostic sessions with GitHub Copilot when anomalies are detected.

ADVISORY ONLY — never modifies trading logic or parameters.

Usage:
    python3 agent/agent.py                    # Run continuously
    python3 agent/agent.py --once             # Single check, then exit
    python3 agent/agent.py --dry-run          # Detect anomalies but skip Copilot

Trading Window:
    Sunday 6PM ET – Friday 5PM ET
    Daily break: 5PM–6PM ET (CME maintenance)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

# ── Ensure agent/ is on sys.path so sibling imports work ──
AGENT_DIR = Path(__file__).resolve().parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from context_collector import ContextCollector
from anomaly_detector import AnomalyDetector, Anomaly
from copilot_interface import CopilotInterface
from report_builder import ReportBuilder
from notifier import TelegramNotifier
from prompt_templates import build_prompt

# ── Timezones ──
ET = ZoneInfo("America/New_York")
CST = ZoneInfo("America/Chicago")

# ── Logging Setup ──
LOG_DIR = AGENT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s CST | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(LOG_DIR / "agent.log", mode="a"),
    ],
)
log = logging.getLogger("agent")


# ═══════════════════════════════════════════════════════════════════════
#  TRADING WINDOW
# ═══════════════════════════════════════════════════════════════════════

class TradingWindowChecker:
    """Determines if we are inside the CME ES/MES trading window."""

    def __init__(self, config: Dict[str, Any]):
        tw = config.get("trading_window", {})
        self.week_start_day = tw.get("week_start_day", "Sunday")
        self.week_start_hour = tw.get("week_start_hour_et", 18)
        self.week_end_day = tw.get("week_end_day", "Friday")
        self.week_end_hour = tw.get("week_end_hour_et", 17)
        self.break_start_hour = tw.get("daily_break_start_hour_et", 17)
        self.break_end_hour = tw.get("daily_break_end_hour_et", 18)

    def is_inside_trading_window(self) -> bool:
        """Check if current time is inside the CME ES/MES trading window.
        
        Trading window: Sunday 6PM ET – Friday 5PM ET
        Daily break: 5PM–6PM ET (maintenance)
        """
        now_et = datetime.now(ET)
        weekday = now_et.weekday()  # Mon=0 ... Sun=6
        hour = now_et.hour

        # Saturday: always closed
        if weekday == 5:
            return False

        # Sunday: only open after 6PM ET
        if weekday == 6:
            return hour >= self.week_start_hour

        # Friday: closed after 5PM ET
        if weekday == 4 and hour >= self.week_end_hour:
            return False

        # Mon-Fri: closed during 5PM-6PM ET maintenance
        if self.break_start_hour <= hour < self.break_end_hour:
            return False

        return True

    def next_window_open(self) -> str:
        """Return human-readable time until next trading window opens."""
        now_et = datetime.now(ET)
        weekday = now_et.weekday()

        if weekday == 5:
            # Saturday → Sunday 6PM
            hours_until = (24 - now_et.hour) + 18
            return f"~{hours_until}h (Sunday 6PM ET)"
        if weekday == 6 and now_et.hour < self.week_start_hour:
            return f"~{self.week_start_hour - now_et.hour}h (Sunday 6PM ET)"
        if self.break_start_hour <= now_et.hour < self.break_end_hour:
            return f"~{self.break_end_hour - now_et.hour}h (daily break ends {self.break_end_hour}:00 ET)"
        if weekday == 4 and now_et.hour >= self.week_end_hour:
            hours_until = (24 - now_et.hour) + 24 + 18  # Sat + to Sun 6PM
            return f"~{hours_until}h (Sunday 6PM ET)"
        return "Now (window is open)"


# ═══════════════════════════════════════════════════════════════════════
#  AGENT CORE
# ═══════════════════════════════════════════════════════════════════════

class TradingAnalystAgent:
    """Main agent: polls for anomalies and orchestrates diagnostic sessions."""

    def __init__(self, config: Dict[str, Any], dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self._shutdown = False

        # Components
        self.window_checker = TradingWindowChecker(config)
        self.collector = ContextCollector(config)
        self.detector = AnomalyDetector(config, self.collector)
        self.copilot = CopilotInterface(config)
        self.report_builder = ReportBuilder(config)
        self.notifier = TelegramNotifier(config)

        # Polling config
        polling = config.get("polling", {})
        self.check_interval = polling.get("check_interval_seconds", 60)
        self.heartbeat_interval = polling.get("heartbeat_interval_seconds", 300)
        self._last_heartbeat = time.time()

        # Session history (in-memory)
        self._session_count = 0

    # ─── Signal Handling ──────────────────────────────────────────────

    def _setup_signal_handlers(self) -> None:
        """Setup graceful shutdown on SIGINT / SIGTERM."""
        def _handle_shutdown(signum, frame):
            signame = signal.Signals(signum).name
            log.info("Received %s — initiating graceful shutdown...", signame)
            self._shutdown = True

        signal.signal(signal.SIGINT, _handle_shutdown)
        signal.signal(signal.SIGTERM, _handle_shutdown)

    # ─── Single Anomaly Processing ────────────────────────────────────

    def _process_anomaly(self, anomaly: Anomaly) -> None:
        """Process a single detected anomaly through the full pipeline.
        
        Copilot is only invoked when anomaly.needs_copilot is True (losses,
        first NO_TRADE of the day).  Lightweight anomalies (wins, repeat
        NO_TRADE re-alerts) skip Copilot and just send a Telegram notification.
        """
        log.info(
            "Processing anomaly: %s [%s] severity=%s copilot=%s",
            anomaly.anomaly_type, anomaly.anomaly_id, anomaly.severity,
            anomaly.needs_copilot,
        )

        # STEP 1: Collect context
        try:
            context = self.collector.collect_full_context()
        except Exception as exc:
            log.error("Context collection failed: %s", exc)
            context = {"error": str(exc), "timestamp_cst": datetime.now(CST).isoformat()}

        # STEP 2 & 3: Build prompt + interact with Copilot (only if warranted)
        session = None
        if anomaly.needs_copilot and not self.dry_run:
            try:
                prompt = build_prompt(anomaly, context)
            except Exception as exc:
                log.error("Prompt building failed: %s", exc)
                prompt = None

            if prompt:
                try:
                    session = self.copilot.run_diagnostic_session(
                        initial_prompt=prompt,
                        anomaly_id=anomaly.anomaly_id,
                        anomaly_type=anomaly.anomaly_type,
                    )
                    self._session_count += 1
                except Exception as exc:
                    log.error("Copilot session failed: %s", exc)
        elif anomaly.needs_copilot and self.dry_run:
            try:
                prompt = build_prompt(anomaly, context)
                log.info("[DRY RUN] Skipping Copilot session — prompt generated (%d chars)", len(prompt))
                prompt_path = LOG_DIR / f"prompt_{anomaly.anomaly_id}_{datetime.now(CST).strftime('%Y%m%d_%H%M%S')}.txt"
                with open(prompt_path, "w") as f:
                    f.write(prompt)
                log.info("Prompt saved: %s", prompt_path)
            except Exception:
                pass
        else:
            log.info("Skipping Copilot — lightweight alert for %s", anomaly.anomaly_type)

        # STEP 4: Produce reports
        try:
            # JSON report
            report = self.report_builder.build_json_report(anomaly, context, session)
            self.report_builder.save_json_report(report)

            # Markdown summary
            self.report_builder.save_markdown_report(anomaly, context, session)

            # Telegram alert
            alert_text = self.report_builder.build_telegram_alert(anomaly, context, session)
            self.notifier.send_anomaly_alert(alert_text)

            log.info("Report pipeline complete for anomaly %s", anomaly.anomaly_id)

        except Exception as exc:
            log.error("Report generation failed: %s", exc)

    # ─── Main Loop ────────────────────────────────────────────────────

    def run_once(self) -> int:
        """Run a single detection cycle. Returns count of anomalies found."""
        # Check trading window
        if not self.window_checker.is_inside_trading_window():
            next_open = self.window_checker.next_window_open()
            log.debug("Outside trading window — next open: %s", next_open)
            return 0

        # Run anomaly detection
        try:
            anomalies = self.detector.detect_all()
        except Exception as exc:
            log.error("Anomaly detection failed: %s", exc)
            return 0

        if not anomalies:
            log.debug("No anomalies detected")
            return 0

        log.info("Detected %d anomalie(s)", len(anomalies))

        # Process each anomaly (highest severity first — already sorted by detector)
        for anomaly in anomalies:
            try:
                self._process_anomaly(anomaly)
            except Exception as exc:
                log.error(
                    "Failed to process anomaly %s: %s",
                    anomaly.anomaly_id, exc,
                )

        return len(anomalies)

    def run_forever(self) -> None:
        """Run the agent continuously until shutdown signal."""
        self._setup_signal_handlers()

        log.info("=" * 60)
        log.info("  Trading Analyst Agent starting")
        log.info("  Mode: %s", "DRY RUN" if self.dry_run else "LIVE")
        log.info("  Check interval: %ds", self.check_interval)
        log.info("  Copilot available: %s", self.copilot.is_available())
        log.info("  Telegram enabled: %s", self.notifier.enabled)
        log.info("=" * 60)

        while not self._shutdown:
            try:
                count = self.run_once()

                # Periodic heartbeat
                if time.time() - self._last_heartbeat > self.heartbeat_interval:
                    self._heartbeat()
                    self._last_heartbeat = time.time()

                # Periodic cleanup
                if self._session_count > 0 and self._session_count % 10 == 0:
                    self.report_builder.cleanup_old_reports()

            except Exception as exc:
                log.error("Unexpected error in main loop: %s", exc, exc_info=True)

            # Sleep between checks (interruptible)
            for _ in range(self.check_interval):
                if self._shutdown:
                    break
                time.sleep(1)

        log.info("Agent shutdown complete. Sessions run: %d", self._session_count)

    def _heartbeat(self) -> None:
        """Log a heartbeat and optionally send to Telegram."""
        now = datetime.now(CST)
        in_window = self.window_checker.is_inside_trading_window()
        log.info(
            "💓 Heartbeat | %s | Window: %s | Sessions: %d | Anomaly history: %d",
            now.strftime("%H:%M CST"),
            "OPEN" if in_window else "CLOSED",
            self._session_count,
            len(self.detector.get_history()),
        )


# ═══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def load_config() -> Dict[str, Any]:
    """Load agent config from config.json."""
    config_path = AGENT_DIR / "config.json"
    if not config_path.exists():
        log.warning("No config.json found at %s — using defaults", config_path)
        return {}
    with open(config_path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous Trading Analyst Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 agent/agent.py                # Run continuously
  python3 agent/agent.py --once         # Single check
  python3 agent/agent.py --dry-run      # No Copilot calls
  python3 agent/agent.py --once --dry-run  # Debug mode
        """,
    )
    parser.add_argument("--once", action="store_true", help="Run one check cycle and exit")
    parser.add_argument("--dry-run", action="store_true", help="Detect anomalies but skip Copilot")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = load_config()
    agent = TradingAnalystAgent(config, dry_run=args.dry_run)

    if args.once:
        count = agent.run_once()
        log.info("Single check complete: %d anomalies found", count)
        sys.exit(0 if count == 0 else 1)
    else:
        agent.run_forever()


if __name__ == "__main__":
    main()
