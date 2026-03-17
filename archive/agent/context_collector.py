"""Context Collector — gathers live trading system state for anomaly analysis.

Reads from:
  - SQLite orders database (data/orders.db)
  - Bot log files (logs/bot.log, logs/live_trading.log)
  - config.yaml (strategy & risk parameters)
  - Status files (if any)

Deep log parsing extracts:
  - Strategy signal diagnostics (NO_SIGNAL diag lines)
  - Entry block/skip reasons
  - Trade executions and PnL
  - Cooldown/kill-switch events
  - OR (Opening Range) computations
  - Heartbeat status

All timestamps returned in CST (America/Chicago).
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import yaml
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

CST = ZoneInfo("America/Chicago")
ET = ZoneInfo("America/New_York")

# Project root — agent/ lives at project_root/agent/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _now_cst() -> datetime:
    return datetime.now(CST)


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


class ContextCollector:
    """Gathers full trading system context for diagnostic prompts."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        cc = config.get("context_collection", {})
        self.orders_db_path = PROJECT_ROOT / cc.get("orders_db_path", "data/orders.db")
        self.log_file_path = PROJECT_ROOT / cc.get("log_file_path", "logs/live_trading.log")
        self.config_yaml_path = PROJECT_ROOT / cc.get("config_yaml_path", "config.yaml")
        self.status_dir = PROJECT_ROOT / cc.get("status_dir", "status")
        self.recent_trades_count = cc.get("recent_trades_count", 20)
        self.log_tail_lines = cc.get("log_tail_lines", 200)

    # ─── Trade History ────────────────────────────────────────────────

    def get_recent_trades(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch recent trades from the orders SQLite database."""
        limit = limit or self.recent_trades_count
        if not self.orders_db_path.exists():
            return []
        try:
            with sqlite3.connect(str(self.orders_db_path)) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT
                        o.order_id,
                        o.symbol,
                        o.action,
                        o.quantity,
                        o.order_type,
                        o.limit_price,
                        o.stop_price,
                        o.status,
                        o.trade_cycle_id,
                        o.created_at,
                        e.price AS fill_price,
                        e.realized_pnl,
                        e.timestamp AS fill_timestamp
                    FROM orders o
                    LEFT JOIN executions e ON o.order_id = e.order_id
                    ORDER BY COALESCE(e.timestamp, o.created_at) DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as exc:
            return [{"error": str(exc)}]

    def get_todays_trades(self) -> List[Dict[str, Any]]:
        """Fetch all trades from today (CST date boundary)."""
        today_str = _now_cst().strftime("%Y-%m-%d")
        if not self.orders_db_path.exists():
            return []
        try:
            with sqlite3.connect(str(self.orders_db_path)) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT
                        o.order_id, o.symbol, o.action, o.quantity,
                        o.order_type, o.status, o.trade_cycle_id,
                        o.created_at,
                        e.price AS fill_price,
                        e.realized_pnl,
                        e.timestamp AS fill_timestamp
                    FROM orders o
                    LEFT JOIN executions e ON o.order_id = e.order_id
                    WHERE DATE(COALESCE(e.timestamp, o.created_at)) = ?
                    ORDER BY COALESCE(e.timestamp, o.created_at) DESC
                    """,
                    (today_str,),
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as exc:
            return [{"error": str(exc)}]

    def get_last_trade_timestamp(self) -> Optional[datetime]:
        """Return the timestamp of the most recent trade execution."""
        if not self.orders_db_path.exists():
            return None
        try:
            with sqlite3.connect(str(self.orders_db_path)) as conn:
                row = conn.execute(
                    "SELECT MAX(timestamp) as ts FROM executions"
                ).fetchone()
                if row and row[0]:
                    ts = datetime.fromisoformat(str(row[0]))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    return ts.astimezone(CST)
        except Exception:
            pass
        return None

    def get_daily_pnl(self) -> Dict[str, Any]:
        """Compute today's realized PnL from executions table."""
        today_str = _now_cst().strftime("%Y-%m-%d")
        result = {"date": today_str, "realized_pnl": 0.0, "trade_count": 0, "wins": 0, "losses": 0}
        if not self.orders_db_path.exists():
            return result
        try:
            with sqlite3.connect(str(self.orders_db_path)) as conn:
                rows = conn.execute(
                    """
                    SELECT realized_pnl FROM executions
                    WHERE DATE(timestamp) = ? AND realized_pnl IS NOT NULL AND realized_pnl != 0
                    """,
                    (today_str,),
                ).fetchall()
                for row in rows:
                    pnl = _safe_float(row[0])
                    result["realized_pnl"] += pnl
                    result["trade_count"] += 1
                    if pnl > 0:
                        result["wins"] += 1
                    elif pnl < 0:
                        result["losses"] += 1
        except Exception:
            pass
        return result

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Detect open positions by checking for unmatched entries.
        
        Uses trade_cycle_id grouping: if a cycle has an entry but no 
        exit execution, it's considered open.
        """
        if not self.orders_db_path.exists():
            return []
        try:
            with sqlite3.connect(str(self.orders_db_path)) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT o.trade_cycle_id, o.action, o.quantity, e.price, e.timestamp
                    FROM orders o
                    JOIN executions e ON o.order_id = e.order_id
                    WHERE o.status = 'Filled'
                    AND o.trade_cycle_id IS NOT NULL
                    GROUP BY o.trade_cycle_id
                    HAVING COUNT(*) = 1
                    ORDER BY e.timestamp DESC
                    LIMIT 5
                    """,
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as exc:
            return [{"error": str(exc)}]

    # ─── Configuration & Strategy Parameters ──────────────────────────

    def get_strategy_config(self) -> Dict[str, Any]:
        """Read strategy-relevant config from config.yaml."""
        if not self.config_yaml_path.exists():
            return {"error": "config.yaml not found"}
        try:
            with open(self.config_yaml_path, "r") as f:
                cfg = yaml.safe_load(f)
            # Extract the strategy-relevant sections
            return {
                "trading": cfg.get("trading", {}),
                "risk_gate": cfg.get("risk_gate", {}),
                "vix_feed": cfg.get("vix_feed", {}),
                "hybrid": cfg.get("hybrid", {}),
                "structural_support_floor": cfg.get("structural_support_floor"),
                "high_impact_event_dates": cfg.get("high_impact_event_dates", []),
            }
        except Exception as exc:
            return {"error": str(exc)}

    def get_confidence_thresholds(self) -> Dict[str, Any]:
        """Extract confidence-related thresholds from config."""
        cfg = self.get_strategy_config()
        trading = cfg.get("trading", {})
        return {
            "min_weighted_confidence": trading.get("min_weighted_confidence"),
            "confidence_threshold": trading.get("confidence_threshold"),
            "min_confidence_for_trade": trading.get("min_confidence_for_trade"),
            "min_risk_reward_ratio": trading.get("min_risk_reward_ratio"),
            "entry_filters": trading.get("entry_filters", {}),
        }

    def get_risk_parameters(self) -> Dict[str, Any]:
        """Extract risk management parameters."""
        cfg = self.get_strategy_config()
        return {
            "risk_gate": cfg.get("risk_gate", {}),
            "position_limits": {
                "max_position_size": cfg.get("trading", {}).get("max_position_size"),
                "max_contracts_limit": cfg.get("trading", {}).get("max_contracts_limit"),
                "max_daily_loss": cfg.get("trading", {}).get("max_daily_loss"),
                "max_daily_trades": cfg.get("trading", {}).get("max_daily_trades"),
            },
            "stop_target": {
                "stop_loss_ticks": cfg.get("trading", {}).get("stop_loss_ticks"),
                "take_profit_ticks": cfg.get("trading", {}).get("take_profit_ticks"),
                "min_stop_distance_ticks": cfg.get("trading", {}).get("min_stop_distance_ticks"),
                "max_stop_distance_ticks": cfg.get("trading", {}).get("max_stop_distance_ticks"),
            },
            "cooldowns": {
                "trade_cooldown_minutes": cfg.get("trading", {}).get("trade_cooldown_minutes"),
                "cooldown_on_loss_minutes": cfg.get("trading", {}).get("cooldown_on_loss_minutes"),
                "cooldown_on_consecutive_losses_minutes": cfg.get("trading", {}).get(
                    "cooldown_on_consecutive_losses_minutes"
                ),
            },
        }

    # ─── VIX / Volatility ────────────────────────────────────────────

    def get_vix_status(self) -> Dict[str, Any]:
        """Check VIX feed configuration and any cached VIX value."""
        cfg = self.get_strategy_config()
        vix_cfg = cfg.get("vix_feed", {})
        # Try to read cached VIX from status file
        vix_file = self.status_dir / "vix_latest.json"
        cached_vix = None
        if vix_file.exists():
            try:
                with open(vix_file) as f:
                    cached_vix = json.load(f)
            except Exception:
                pass
        return {
            "config": vix_cfg,
            "cached_value": cached_vix,
        }

    # ═══════════════════════════════════════════════════════════════════
    #  RAW LOG READERS
    # ═══════════════════════════════════════════════════════════════════

    def _tail_file(self, path: Path, lines: int) -> str:
        """Tail a log file, return raw text."""
        if not path.exists():
            return ""
        try:
            result = subprocess.run(
                ["tail", "-n", str(lines), str(path)],
                capture_output=True, text=True, timeout=10,
            )
            return result.stdout if result.returncode == 0 else ""
        except Exception:
            return ""

    def get_recent_logs(self, lines: Optional[int] = None) -> str:
        """Tail the primary bot log file."""
        lines = lines or self.log_tail_lines
        text = self._tail_file(self.log_file_path, lines)
        if not text:
            # Try alternative log paths
            for alt in [PROJECT_ROOT / "logs" / "bot.log",
                        PROJECT_ROOT / "logs" / "live_trading.log",
                        PROJECT_ROOT / "logs" / "shreebot.log"]:
                if alt.exists():
                    self.log_file_path = alt
                    text = self._tail_file(alt, lines)
                    if text:
                        break
        return text or "[No log file found]"

    def _get_detail_logs(self, lines: int = 200) -> str:
        """Tail the detailed trading log (live_trading.log)."""
        detail_path = PROJECT_ROOT / self.config.get(
            "context_collection", {}).get("detail_log_file_path", "logs/live_trading.log")
        return self._tail_file(detail_path, lines)

    # ═══════════════════════════════════════════════════════════════════
    #  DEEP LOG PARSING — Strategy Signal Diagnostics
    # ═══════════════════════════════════════════════════════════════════

    # Regex patterns (compiled once)
    _RE_NO_SIGNAL = re.compile(
        r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+CST.*NO_SIGNAL diag:\s*(.*)"
    )
    _RE_STRATEGY_SIGNAL = re.compile(
        r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+CST.*Strategy Signal:\s*(\w+)\s*\(conf=([0-9.]+),\s*meta=(\w+)\)"
    )
    _RE_OR_COMPUTED = re.compile(
        r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+CST.*OR computed:\s*HIGH=([0-9.]+)\s*LOW=([0-9.]+)(?:\s*\((\d+) bars\))?"
    )
    _RE_ENTRY_BLOCKED = re.compile(
        r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+CST.*Entry blocked:\s*(.*)"
    )
    _RE_HEARTBEAT = re.compile(
        r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+CST.*Heartbeat:.*bars=(\d+)\s*\|\s*price=(\S+)"
    )
    _RE_VX_FEED = re.compile(
        r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+CST.*VX Feed: Price=([0-9.]+)\s*\(multiplier=([0-9.]+)x\)"
    )
    _RE_COOLDOWN = re.compile(
        r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+CST.*[Cc]ooldown.*?:\s*(.*)"
    )
    _RE_SKIPPING = re.compile(
        r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+CST.*Skipping trade\s*\((\w+)\)"
    )
    _RE_CYCLE_START = re.compile(
        r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+CST.*Cycle start"
    )
    _RE_RISK_CONSOLIDATION = re.compile(
        r"RISK CONSOLIDATION:\s*(.*)"
    )
    _RE_KILL_SWITCH = re.compile(
        r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+CST.*[Kk]ill.?[Ss]witch\s*(.*)"
    )
    _RE_DRAWDOWN = re.compile(
        r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+CST.*[Dd]rawdown.*(halt|triggered|limit)(.*)"
    )

    # Signal label explanations for Copilot context
    SIGNAL_LABEL_MAP = {
        "A": "EMA21 Pullback Long (price dips to EMA21, bounces in uptrend)",
        "B": "Opening Range Breakout Long (first close above OR high)",
        "D": "EMA21 Pullback Short (price rallies to EMA21, rejects in downtrend)",
        "E": "Opening Range Breakdown Short (first close below OR low)",
        "F": "Trend Continuation (strong momentum, aligned EMA stack)",
    }

    def parse_strategy_diagnostics(self, logs: str) -> Dict[str, Any]:
        """Parse NO_SIGNAL diag lines to understand why each signal type failed.

        Extracts per-signal failure reasons across multiple candle cycles.
        Returns a structured dict with failure counts, latest reasons, and
        a human-readable summary.
        """
        diag_entries = []  # list of (timestamp, raw_diag_string)
        for line in logs.split("\n"):
            m = self._RE_NO_SIGNAL.search(line)
            if m:
                diag_entries.append((m.group(1), m.group(2).strip()))

        if not diag_entries:
            return {"has_diagnostics": False, "summary": "No NO_SIGNAL diag lines found in logs"}

        # Parse individual signal parts (A:reason | B:reason | ...)
        signal_failures: Dict[str, List[str]] = {}  # label -> [reason, reason, ...]
        for ts, raw in diag_entries:
            parts = [p.strip() for p in raw.split("|")]
            for part in parts:
                colon_idx = part.find(":")
                if colon_idx > 0:
                    label = part[:colon_idx].strip()
                    reason = part[colon_idx + 1:].strip()
                    signal_failures.setdefault(label, []).append(reason)

        # Compute per-signal summary
        signal_summary = {}
        for label in sorted(signal_failures.keys()):
            reasons = signal_failures[label]
            # Deduplicate: count how many times each unique reason appeared
            reason_counts: Dict[str, int] = {}
            for r in reasons:
                reason_counts[r] = reason_counts.get(r, 0) + 1
            # Sort by frequency
            top_reasons = sorted(reason_counts.items(), key=lambda x: -x[1])
            signal_summary[label] = {
                "description": self.SIGNAL_LABEL_MAP.get(label, f"Signal {label}"),
                "failure_count": len(reasons),
                "unique_reasons": len(reason_counts),
                "top_reasons": [
                    {"reason": r, "count": c} for r, c in top_reasons[:5]
                ],
            }

        # Human-readable summary
        lines = []
        for label, info in signal_summary.items():
            top = info["top_reasons"][0] if info["top_reasons"] else None
            reason_str = top["reason"] if top else "unknown"
            lines.append(
                f"  Signal {label} ({info['description']}): "
                f"failed {info['failure_count']}x — primary reason: {reason_str}"
            )

        return {
            "has_diagnostics": True,
            "candle_cycles_analyzed": len(diag_entries),
            "latest_diag_timestamp": diag_entries[-1][0] if diag_entries else None,
            "latest_diag_raw": diag_entries[-1][1] if diag_entries else None,
            "signal_summary": signal_summary,
            "human_readable": "\n".join(lines),
        }

    def parse_strategy_signals(self, logs: str) -> Dict[str, Any]:
        """Parse Strategy Signal lines (BUY, SELL, HOLD) and their confidence."""
        signals = []
        for line in logs.split("\n"):
            m = self._RE_STRATEGY_SIGNAL.search(line)
            if m:
                signals.append({
                    "timestamp": m.group(1),
                    "action": m.group(2),
                    "confidence": float(m.group(3)),
                    "meta": m.group(4),
                })
        if not signals:
            return {"total": 0, "summary": "No strategy signals found"}

        actions = [s["action"] for s in signals]
        return {
            "total": len(signals),
            "buys": actions.count("BUY"),
            "sells": actions.count("SELL"),
            "holds": actions.count("HOLD"),
            "avg_confidence": sum(s["confidence"] for s in signals) / len(signals),
            "max_confidence": max(s["confidence"] for s in signals),
            "latest": signals[-1],
            "all_signals": signals[-20:],  # last 20
        }

    def parse_opening_range(self, logs: str) -> Dict[str, Any]:
        """Parse OR computed lines to get today's opening range."""
        entries = []
        for line in logs.split("\n"):
            m = self._RE_OR_COMPUTED.search(line)
            if m:
                entries.append({
                    "timestamp": m.group(1),
                    "or_high": float(m.group(2)),
                    "or_low": float(m.group(3)),
                    "bars": int(m.group(4)) if m.group(4) else None,
                })
        if not entries:
            return {"computed": False}
        latest = entries[-1]
        return {
            "computed": True,
            "or_high": latest["or_high"],
            "or_low": latest["or_low"],
            "or_range_points": round(latest["or_high"] - latest["or_low"], 2),
            "or_range_dollars": round((latest["or_high"] - latest["or_low"]) * 5, 2),
            "bars_used": latest["bars"],
            "timestamp": latest["timestamp"],
        }

    def parse_entry_blocks(self, logs: str) -> List[Dict[str, Any]]:
        """Parse Entry blocked lines — these prevent trades even if signals fire."""
        blocks = []
        for line in logs.split("\n"):
            m = self._RE_ENTRY_BLOCKED.search(line)
            if m:
                blocks.append({
                    "timestamp": m.group(1),
                    "reason": m.group(2).strip(),
                })
        return blocks

    def parse_heartbeats(self, logs: str) -> Dict[str, Any]:
        """Parse Heartbeat lines to get bot liveness and latest price."""
        entries = []
        for line in logs.split("\n"):
            m = self._RE_HEARTBEAT.search(line)
            if m:
                price_str = m.group(3)
                price = float(price_str) if price_str != "None" else None
                entries.append({
                    "timestamp": m.group(1),
                    "bars": int(m.group(2)),
                    "price": price,
                })
        if not entries:
            return {"alive": False, "summary": "No heartbeat lines found — bot may not be running"}
        latest = entries[-1]
        return {
            "alive": True,
            "latest_heartbeat": latest["timestamp"],
            "total_heartbeats": len(entries),
            "latest_price": latest["price"],
            "latest_bars": latest["bars"],
        }

    def parse_vx_status(self, logs: str) -> Dict[str, Any]:
        """Parse VX Feed lines for live VIX futures data."""
        entries = []
        for line in logs.split("\n"):
            m = self._RE_VX_FEED.search(line)
            if m:
                entries.append({
                    "timestamp": m.group(1),
                    "vx_price": float(m.group(2)),
                    "multiplier": float(m.group(3)),
                })
        if not entries:
            return {"available": False}
        latest = entries[-1]
        return {
            "available": True,
            "vx_price": latest["vx_price"],
            "multiplier": latest["multiplier"],
            "timestamp": latest["timestamp"],
        }

    def parse_skips(self, logs: str) -> Dict[str, Any]:
        """Parse Skipping trade lines."""
        skips = []
        for line in logs.split("\n"):
            m = self._RE_SKIPPING.search(line)
            if m:
                skips.append({"timestamp": m.group(1), "reason": m.group(2)})
        return {
            "total_skips": len(skips),
            "skip_reasons": {r: sum(1 for s in skips if s["reason"] == r) for r in set(s["reason"] for s in skips)},
            "latest": skips[-1] if skips else None,
        }

    def parse_cycle_count(self, logs: str) -> int:
        """Count how many trading cycles (15m candle evaluations) occurred."""
        return len(self._RE_CYCLE_START.findall(logs))

    def parse_dampening_status(self, logs: str) -> Dict[str, Any]:
        """Parse cooldown, kill-switch, and drawdown events from logs."""
        dampening = {
            "cooldown_active": False,
            "kill_switch_active": False,
            "drawdown_halt": False,
            "maintenance_block": False,
            "cooldown_details": [],
            "risk_consolidation": [],
            "details": [],
        }

        for line in logs.split("\n"):
            ll = line.lower()

            m = self._RE_COOLDOWN.search(line)
            if m:
                dampening["cooldown_active"] = True
                dampening["cooldown_details"].append({
                    "timestamp": m.group(1), "detail": m.group(2).strip()
                })

            m = self._RE_KILL_SWITCH.search(line)
            if m:
                dampening["kill_switch_active"] = True
                dampening["details"].append(f"Kill switch at {m.group(1)}: {m.group(2).strip()}")

            m = self._RE_DRAWDOWN.search(line)
            if m:
                dampening["drawdown_halt"] = True
                dampening["details"].append(f"Drawdown {m.group(2)} at {m.group(1)}")

            if "maintenance" in ll and ("block" in ll or "window" in ll):
                dampening["maintenance_block"] = True
                dampening["details"].append(line.strip()[-150:])

            m = self._RE_RISK_CONSOLIDATION.search(line)
            if m:
                dampening["risk_consolidation"].append(m.group(1).strip())

        # Trim
        dampening["cooldown_details"] = dampening["cooldown_details"][-5:]
        dampening["details"] = dampening["details"][-10:]
        dampening["risk_consolidation"] = dampening["risk_consolidation"][-3:]
        return dampening

    # ═══════════════════════════════════════════════════════════════════
    #  HIGH-LEVEL DEEP LOG INTELLIGENCE
    # ═══════════════════════════════════════════════════════════════════

    def get_deep_log_intelligence(self) -> Dict[str, Any]:
        """Master method: parse both log files and return structured intelligence.

        Reads bot.log (primary, compact) and live_trading.log (detail) and
        merges all parsed data into one dict.
        """
        primary_logs = self.get_recent_logs()
        detail_logs = self._get_detail_logs(200)
        # Merge: use primary for most parsing; detail for extra context
        combined = primary_logs + "\n" + detail_logs

        diagnostics = self.parse_strategy_diagnostics(combined)
        signals = self.parse_strategy_signals(combined)
        opening_range = self.parse_opening_range(combined)
        entry_blocks = self.parse_entry_blocks(combined)
        heartbeats = self.parse_heartbeats(combined)
        vx = self.parse_vx_status(combined)
        skips = self.parse_skips(combined)
        cycles = self.parse_cycle_count(combined)
        dampening = self.parse_dampening_status(combined)

        # Build overall bot status
        bot_running = heartbeats.get("alive", False)
        latest_price = heartbeats.get("latest_price")

        return {
            "bot_status": {
                "running": bot_running,
                "latest_heartbeat": heartbeats.get("latest_heartbeat"),
                "latest_price": latest_price,
                "bars_accumulated": heartbeats.get("latest_bars"),
                "total_heartbeats": heartbeats.get("total_heartbeats", 0),
            },
            "strategy_diagnostics": diagnostics,
            "strategy_signals": signals,
            "opening_range": opening_range,
            "entry_blocks": entry_blocks,
            "skips": skips,
            "trading_cycles": cycles,
            "vx_live": vx,
            "dampening": dampening,
        }

    # ─── Market Regime (best-effort) ──────────────────────────────────

    def get_market_regime_snapshot(self) -> Dict[str, Any]:
        """Attempt to read any cached market regime/volatility snapshot."""
        regime_file = self.status_dir / "market_regime.json"
        if regime_file.exists():
            try:
                with open(regime_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"status": "no_regime_data_available"}

    # ─── Legacy compatibility wrappers ────────────────────────────────

    def get_signal_metrics(self) -> Dict[str, Any]:
        """Extract signal generation stats — delegates to deep parsing."""
        logs = self.get_recent_logs()
        signals = self.parse_strategy_signals(logs)
        return {
            "signals_generated": signals.get("buys", 0) + signals.get("sells", 0),
            "signals_blocked": len(self.parse_entry_blocks(logs)),
            "confidence_values": [s["confidence"] for s in signals.get("all_signals", [])],
            "block_reasons": [b["reason"] for b in self.parse_entry_blocks(logs)[:10]],
        }

    def get_dampening_status(self) -> Dict[str, Any]:
        """Check dampening/throttling — delegates to deep parsing."""
        logs = self.get_recent_logs()
        return self.parse_dampening_status(logs)

    # ═══════════════════════════════════════════════════════════════════
    #  FULL CONTEXT BUNDLE
    # ═══════════════════════════════════════════════════════════════════

    def collect_full_context(self) -> Dict[str, Any]:
        """Gather all available context into a single structured dict.

        This is the main method called by the anomaly detector before
        generating a diagnostic prompt.  Includes deep log intelligence.
        """
        now = _now_cst()
        last_ts = self.get_last_trade_timestamp()

        # Deep log intelligence (all parsed log data)
        deep = self.get_deep_log_intelligence()

        return {
            "timestamp_cst": now.isoformat(),
            # ── Trade data ──
            "recent_trades": self.get_recent_trades(),
            "todays_trades": self.get_todays_trades(),
            "last_trade_timestamp": last_ts.isoformat() if last_ts else None,
            "daily_pnl": self.get_daily_pnl(),
            "open_positions": self.get_open_positions(),
            # ── Config ──
            "strategy_config": self.get_strategy_config(),
            "confidence_thresholds": self.get_confidence_thresholds(),
            "risk_parameters": self.get_risk_parameters(),
            "vix_status": self.get_vix_status(),
            "market_regime": self.get_market_regime_snapshot(),
            # ── Deep log intelligence ──
            "deep_log": deep,
            "signal_metrics": self.get_signal_metrics(),
            "dampening_status": deep.get("dampening", {}),
            # ── Raw logs (truncated for prompt) ──
            "recent_logs": self.get_recent_logs(80),
        }
