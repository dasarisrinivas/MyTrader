"""Anomaly Detector — identifies no-trade and loss conditions.

Anomaly types:
  NO_TRADE   — No executions within inactivity_minutes_threshold while inside trading window.
  TRADE_LOSS — A single trade loss exceeds max_loss_per_trade_usd.
  DAILY_LOSS — Cumulative daily PnL drops below daily_loss_threshold_usd.
  CONSEC_LOSS — Consecutive losing trades exceed threshold.

Each anomaly is tagged with a unique ID (hash of type + key attributes) to prevent
duplicate triggering within the dedup window.  Cooldowns are enforced per-type.

State is persisted to disk so that cooldowns survive process restarts.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from zoneinfo import ZoneInfo

from context_collector import ContextCollector

CST = ZoneInfo("America/Chicago")
AGENT_DIR = Path(__file__).resolve().parent

log = logging.getLogger("agent.anomaly_detector")


def _now_cst() -> datetime:
    return datetime.now(CST)


def _format_idle_duration(minutes: float) -> str:
    """Format idle duration in a human-friendly way.

    Examples:
        45 min  → "45 min"
        90 min  → "1h 30m"
        1500 min → "1d 1h"
    """
    if minutes < 120:
        return f"{int(minutes)} min"
    hours = minutes / 60
    if hours < 48:
        h = int(hours)
        m = int(minutes - h * 60)
        return f"{h}h {m}m" if m else f"{h}h"
    days = int(hours / 24)
    remaining_h = int(hours - days * 24)
    return f"{days}d {remaining_h}h" if remaining_h else f"{days}d"


@dataclass
class Anomaly:
    """Represents a detected anomaly event."""
    anomaly_id: str
    anomaly_type: str  # NO_TRADE | TRADE_LOSS | DAILY_LOSS | CONSEC_LOSS | TRADE_COMPLETE
    severity: str      # LOW | MEDIUM | HIGH | CRITICAL
    description: str
    detected_at: str   # ISO timestamp CST
    payload: Dict[str, Any] = field(default_factory=dict)
    needs_copilot: bool = True  # Whether this anomaly warrants a Copilot session

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anomaly_id": self.anomaly_id,
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
            "description": self.description,
            "detected_at": self.detected_at,
            "payload": self.payload,
            "needs_copilot": self.needs_copilot,
        }


class AnomalyDetector:
    """Detects trading anomalies and enforces dedup / cooldown logic.
    
    State (triggered timestamps, anomaly history) is persisted to disk
    so cooldowns survive process restarts and --once invocations.
    """

    STATE_FILE = AGENT_DIR / "logs" / ".anomaly_state.json"

    def __init__(self, config: Dict[str, Any], collector: ContextCollector):
        self.config = config
        self.collector = collector

        ad = config.get("anomaly_detection", {})
        self.inactivity_minutes = ad.get("inactivity_minutes_threshold", 60)
        self.max_loss_per_trade = ad.get("max_loss_per_trade_usd", 100.0)
        self.daily_loss_threshold = ad.get("daily_loss_threshold_usd", 200.0)
        self.consecutive_losses_alert = ad.get("consecutive_losses_alert", 3)

        cd = config.get("cooldowns", {})
        self.analysis_cooldown_min = cd.get("analysis_cooldown_minutes", 30)
        self.no_trade_cooldown_min = cd.get("no_trade_cooldown_minutes", 45)
        self.loss_cooldown_min = cd.get("loss_cooldown_minutes", 20)
        self.dedup_window_min = cd.get("dedup_window_minutes", 60)
        self.max_analyses_per_hour = cd.get("max_analyses_per_hour", 3)

        # State: tracks anomaly IDs and their trigger times
        self._triggered: Dict[str, datetime] = {}
        self._analysis_timestamps: List[datetime] = []
        self._anomaly_history: List[Anomaly] = []
        self._last_seen_trade_id: Optional[str] = None  # Tracks last processed trade
        self._no_trade_fired_today: bool = False  # First NO_TRADE of the day gets Copilot

        # Restore persisted state (survives restarts)
        self._load_state()

    # ─── Dedup & Cooldown ─────────────────────────────────────────────

    def _make_anomaly_id(self, anomaly_type: str, key_data: str) -> str:
        """Create a deterministic anomaly ID from type + key data."""
        raw = f"{anomaly_type}:{key_data}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _is_duplicate(self, anomaly_id: str) -> bool:
        """Check if this anomaly was already triggered within dedup window."""
        if anomaly_id not in self._triggered:
            return False
        last_trigger = self._triggered[anomaly_id]
        return (_now_cst() - last_trigger) < timedelta(minutes=self.dedup_window_min)

    def _is_cooldown_active(self, anomaly_type: str) -> bool:
        """Check if cooldown is active for the given anomaly type."""
        cooldown_map = {
            "NO_TRADE": self.no_trade_cooldown_min,
            "TRADE_LOSS": self.loss_cooldown_min,
            "DAILY_LOSS": self.loss_cooldown_min,
            "CONSEC_LOSS": self.loss_cooldown_min,
        }
        cooldown_min = cooldown_map.get(anomaly_type, self.analysis_cooldown_min)
        # Check most recent trigger of this type
        for anomaly in reversed(self._anomaly_history):
            if anomaly.anomaly_type == anomaly_type:
                trigger_time = datetime.fromisoformat(anomaly.detected_at)
                if (_now_cst() - trigger_time) < timedelta(minutes=cooldown_min):
                    return True
                break
        return False

    def _check_rate_limit(self) -> bool:
        """Ensure we don't exceed max_analyses_per_hour."""
        now = _now_cst()
        one_hour_ago = now - timedelta(hours=1)
        self._analysis_timestamps = [t for t in self._analysis_timestamps if t > one_hour_ago]
        return len(self._analysis_timestamps) < self.max_analyses_per_hour

    def _record_trigger(self, anomaly: Anomaly) -> None:
        """Record that an anomaly was triggered."""
        self._triggered[anomaly.anomaly_id] = _now_cst()
        self._anomaly_history.append(anomaly)
        self._analysis_timestamps.append(_now_cst())
        # Keep history bounded
        if len(self._anomaly_history) > 500:
            self._anomaly_history = self._anomaly_history[-250:]
        # Persist to disk so cooldowns survive restarts
        self._save_state()

    # ─── State Persistence ────────────────────────────────────────────

    def _save_state(self) -> None:
        """Persist triggered timestamps and anomaly history to disk."""
        try:
            self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "triggered": {
                    aid: ts.isoformat() for aid, ts in self._triggered.items()
                },
                "anomaly_history": [a.to_dict() for a in self._anomaly_history[-50:]],
                "analysis_timestamps": [t.isoformat() for t in self._analysis_timestamps],
                "last_seen_trade_id": self._last_seen_trade_id,
                "no_trade_fired_date": _now_cst().strftime("%Y-%m-%d") if self._no_trade_fired_today else None,
                "saved_at": _now_cst().isoformat(),
            }
            with open(self.STATE_FILE, "w") as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as exc:
            log.warning("Failed to save anomaly state: %s", exc)

    def _load_state(self) -> None:
        """Restore persisted state from disk. Prunes entries older than dedup window."""
        if not self.STATE_FILE.exists():
            return
        try:
            with open(self.STATE_FILE) as f:
                state = json.load(f)

            now = _now_cst()
            max_age = timedelta(minutes=max(self.dedup_window_min, self.no_trade_cooldown_min, 120))

            # Restore triggered timestamps (only those still within relevance window)
            for aid, ts_str in state.get("triggered", {}).items():
                ts = datetime.fromisoformat(ts_str)
                if (now - ts) < max_age:
                    self._triggered[aid] = ts

            # Restore anomaly history (only recent entries)
            for entry in state.get("anomaly_history", []):
                detected = datetime.fromisoformat(entry["detected_at"])
                if (now - detected) < max_age:
                    self._anomaly_history.append(Anomaly(
                        anomaly_id=entry["anomaly_id"],
                        anomaly_type=entry["anomaly_type"],
                        severity=entry["severity"],
                        description=entry["description"],
                        detected_at=entry["detected_at"],
                        payload=entry.get("payload", {}),
                        needs_copilot=entry.get("needs_copilot", True),
                    ))

            # Restore analysis timestamps
            for ts_str in state.get("analysis_timestamps", []):
                ts = datetime.fromisoformat(ts_str)
                if (now - ts) < timedelta(hours=1):
                    self._analysis_timestamps.append(ts)

            # Restore last-seen trade ID
            self._last_seen_trade_id = state.get("last_seen_trade_id")

            # Restore no-trade-fired-today flag (reset on new day)
            fired_date = state.get("no_trade_fired_date")
            if fired_date == now.strftime("%Y-%m-%d"):
                self._no_trade_fired_today = True

            if self._triggered or self._anomaly_history:
                log.info(
                    "Restored anomaly state: %d triggered, %d history, %d rate-limit entries",
                    len(self._triggered), len(self._anomaly_history), len(self._analysis_timestamps),
                )
        except Exception as exc:
            log.warning("Failed to load anomaly state (starting fresh): %s", exc)

    # ─── Detection Methods ────────────────────────────────────────────

    def check_no_trade(self) -> Optional[Anomaly]:
        """Check if no trades have been executed within the inactivity window."""
        last_ts = self.collector.get_last_trade_timestamp()
        now = _now_cst()

        if last_ts is None:
            # No trades ever — treat as inactivity anomaly
            minutes_idle = self.inactivity_minutes + 1
        else:
            minutes_idle = (now - last_ts).total_seconds() / 60

        if minutes_idle < self.inactivity_minutes:
            return None

        # Build anomaly — key is anchored to date + last trade time so the same
        # inactivity gap produces the same ID.  The cooldown (45 min) governs
        # how often you get re-alerted for the *same* ongoing gap.
        last_ts_key = last_ts.strftime('%Y%m%d_%H%M') if last_ts else "never"
        key = f"no_trade_{now.strftime('%Y%m%d')}_{last_ts_key}"
        anomaly_id = self._make_anomaly_id("NO_TRADE", key)

        if self._is_duplicate(anomaly_id):
            return None
        if self._is_cooldown_active("NO_TRADE"):
            return None
        if not self._check_rate_limit():
            return None

        severity = "MEDIUM" if minutes_idle < 120 else "HIGH"

        # Human-friendly idle description
        idle_desc = _format_idle_duration(minutes_idle)
        if last_ts:
            # Show relative: "since yesterday 3:15 PM" or "since 9:30 AM today"
            if last_ts.date() == now.date():
                last_trade_str = f"today at {last_ts.strftime('%-I:%M %p CST')}"
            elif last_ts.date() == (now - timedelta(days=1)).date():
                last_trade_str = f"yesterday at {last_ts.strftime('%-I:%M %p CST')}"
            else:
                last_trade_str = last_ts.strftime('%b %d at %-I:%M %p CST')
        else:
            last_trade_str = "never"

        # Only the first NO_TRADE of the day gets a Copilot session;
        # subsequent re-alerts are Telegram-only to avoid spamming Copilot.
        first_today = not self._no_trade_fired_today
        self._no_trade_fired_today = True

        anomaly = Anomaly(
            anomaly_id=anomaly_id,
            anomaly_type="NO_TRADE",
            severity=severity,
            description=(
                f"No trades for {idle_desc}. "
                f"Last trade: {last_trade_str}. "
                f"Threshold: {self.inactivity_minutes} min."
            ),
            detected_at=now.isoformat(),
            payload={
                "minutes_idle": round(minutes_idle, 1),
                "idle_display": idle_desc,
                "threshold": self.inactivity_minutes,
                "last_trade": last_ts.isoformat() if last_ts else None,
                "last_trade_display": last_trade_str,
            },
            needs_copilot=first_today,
        )
        self._record_trigger(anomaly)
        return anomaly

    def check_trade_complete(self) -> Optional[Anomaly]:
        """Check if a new trade has completed since last check.
        
        Fires once per new trade. Wins get a lightweight Telegram-only alert.
        Losses below the TRADE_LOSS threshold also get a Telegram-only alert.
        (Losses above threshold are caught separately by check_trade_loss.)
        """
        trades = self.collector.get_recent_trades(3)
        if not trades:
            return None

        # Find the most recent trade with a fill
        latest = None
        for t in trades:
            if t.get("fill_price") is not None and t.get("order_id"):
                latest = t
                break
        if latest is None:
            return None

        order_id = str(latest.get("order_id", ""))
        if not order_id or order_id == self._last_seen_trade_id:
            return None  # Already processed this trade

        # New trade found!
        self._last_seen_trade_id = order_id

        pnl = 0.0
        try:
            pnl = float(latest.get("realized_pnl") or 0)
        except (TypeError, ValueError):
            pass

        # Skip if this is a big loss — check_trade_loss will handle it with Copilot
        if pnl < 0 and abs(pnl) >= self.max_loss_per_trade:
            self._save_state()  # Persist the updated last_seen_trade_id
            return None

        key = f"trade_complete_{order_id}"
        anomaly_id = self._make_anomaly_id("TRADE_COMPLETE", key)

        if self._is_duplicate(anomaly_id):
            return None

        action = latest.get("action", "?")
        fill = latest.get("fill_price", "?")
        pnl_str = f"${pnl:+.2f}" if pnl != 0 else "flat"
        is_loss = pnl < 0

        severity = "MEDIUM" if is_loss else "LOW"
        # Losses get Copilot analysis; wins are Telegram-only
        needs_copilot = is_loss

        anomaly = Anomaly(
            anomaly_id=anomaly_id,
            anomaly_type="TRADE_COMPLETE",
            severity=severity,
            description=(
                f"Trade completed: {action} filled at {fill}, P&L: {pnl_str}. "
                f"Order: {order_id}."
            ),
            detected_at=_now_cst().isoformat(),
            payload={
                "order_id": order_id,
                "action": action,
                "fill_price": fill,
                "realized_pnl": pnl,
                "pnl_display": pnl_str,
                "symbol": latest.get("symbol", "MES"),
            },
            needs_copilot=needs_copilot,
        )
        self._record_trigger(anomaly)
        return anomaly

    def check_trade_loss(self) -> Optional[Anomaly]:
        """Check if the most recent trade had a loss exceeding threshold."""
        trades = self.collector.get_recent_trades(5)
        if not trades:
            return None

        for trade in trades:
            pnl = trade.get("realized_pnl")
            if pnl is None:
                continue
            try:
                pnl = float(pnl)
            except (TypeError, ValueError):
                continue
            if pnl >= 0:
                continue
            if abs(pnl) < self.max_loss_per_trade:
                continue

            # Found a losing trade above threshold
            order_id = trade.get("order_id", "unknown")
            key = f"trade_loss_{order_id}"
            anomaly_id = self._make_anomaly_id("TRADE_LOSS", key)

            if self._is_duplicate(anomaly_id):
                continue
            if self._is_cooldown_active("TRADE_LOSS"):
                return None
            if not self._check_rate_limit():
                return None

            severity = "HIGH" if abs(pnl) > self.max_loss_per_trade * 1.5 else "MEDIUM"
            anomaly = Anomaly(
                anomaly_id=anomaly_id,
                anomaly_type="TRADE_LOSS",
                severity=severity,
                description=(
                    f"Trade loss of ${abs(pnl):.2f} exceeds threshold of ${self.max_loss_per_trade:.2f}. "
                    f"Order: {order_id}, Action: {trade.get('action')}, "
                    f"Fill: {trade.get('fill_price')}."
                ),
                detected_at=_now_cst().isoformat(),
                payload={
                    "trade": trade,
                    "loss_usd": round(abs(pnl), 2),
                    "threshold_usd": self.max_loss_per_trade,
                },
            )
            self._record_trigger(anomaly)
            return anomaly

        return None

    def check_daily_loss(self) -> Optional[Anomaly]:
        """Check if cumulative daily PnL dropped below threshold."""
        daily = self.collector.get_daily_pnl()
        realized = daily.get("realized_pnl", 0.0)

        if realized >= 0 or abs(realized) < self.daily_loss_threshold:
            return None

        key = f"daily_loss_{daily['date']}"
        anomaly_id = self._make_anomaly_id("DAILY_LOSS", key)

        if self._is_duplicate(anomaly_id):
            return None
        if self._is_cooldown_active("DAILY_LOSS"):
            return None
        if not self._check_rate_limit():
            return None

        severity = "CRITICAL" if abs(realized) > self.daily_loss_threshold * 1.5 else "HIGH"
        anomaly = Anomaly(
            anomaly_id=anomaly_id,
            anomaly_type="DAILY_LOSS",
            severity=severity,
            description=(
                f"Daily cumulative loss of ${abs(realized):.2f} exceeds threshold of "
                f"${self.daily_loss_threshold:.2f}. "
                f"Date: {daily['date']}, Trades: {daily['trade_count']}, "
                f"Wins: {daily['wins']}, Losses: {daily['losses']}."
            ),
            detected_at=_now_cst().isoformat(),
            payload={
                "daily_pnl": daily,
                "loss_usd": round(abs(realized), 2),
                "threshold_usd": self.daily_loss_threshold,
            },
        )
        self._record_trigger(anomaly)
        return anomaly

    def check_consecutive_losses(self) -> Optional[Anomaly]:
        """Check for a streak of consecutive losing trades."""
        trades = self.collector.get_recent_trades(20)
        if not trades:
            return None

        streak = 0
        for trade in trades:
            pnl = trade.get("realized_pnl")
            if pnl is None:
                continue
            try:
                pnl = float(pnl)
            except (TypeError, ValueError):
                continue
            if pnl < 0:
                streak += 1
            else:
                break  # Streak broken

        if streak < self.consecutive_losses_alert:
            return None

        key = f"consec_loss_{_now_cst().strftime('%Y%m%d')}_{streak}"
        anomaly_id = self._make_anomaly_id("CONSEC_LOSS", key)

        if self._is_duplicate(anomaly_id):
            return None
        if self._is_cooldown_active("CONSEC_LOSS"):
            return None
        if not self._check_rate_limit():
            return None

        severity = "HIGH" if streak >= 5 else "MEDIUM"
        anomaly = Anomaly(
            anomaly_id=anomaly_id,
            anomaly_type="CONSEC_LOSS",
            severity=severity,
            description=(
                f"{streak} consecutive losing trades detected (threshold: {self.consecutive_losses_alert}). "
                f"Possible strategy degradation or regime mismatch."
            ),
            detected_at=_now_cst().isoformat(),
            payload={
                "streak": streak,
                "threshold": self.consecutive_losses_alert,
                "recent_trades": trades[:streak],
            },
        )
        self._record_trigger(anomaly)
        return anomaly

    # ─── Main Detection Loop ─────────────────────────────────────────

    def detect_all(self) -> List[Anomaly]:
        """Run all anomaly checks and return any new anomalies found.
        
        Returns anomalies in priority order (CRITICAL first).
        """
        anomalies: List[Anomaly] = []

        checks = [
            self.check_daily_loss,
            self.check_trade_loss,
            self.check_trade_complete,
            self.check_consecutive_losses,
            self.check_no_trade,
        ]

        for check_fn in checks:
            try:
                anomaly = check_fn()
                if anomaly is not None:
                    anomalies.append(anomaly)
            except Exception as exc:
                # Never let a detection crash the loop
                anomalies.append(
                    Anomaly(
                        anomaly_id=self._make_anomaly_id("ERROR", str(exc)[:50]),
                        anomaly_type="DETECTION_ERROR",
                        severity="LOW",
                        description=f"Error in {check_fn.__name__}: {exc}",
                        detected_at=_now_cst().isoformat(),
                        payload={"error": str(exc), "check": check_fn.__name__},
                    )
                )

        # Sort: CRITICAL > HIGH > MEDIUM > LOW
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        anomalies.sort(key=lambda a: severity_order.get(a.severity, 99))
        return anomalies

    def get_history(self) -> List[Dict[str, Any]]:
        """Return the anomaly history for reporting."""
        return [a.to_dict() for a in self._anomaly_history]
