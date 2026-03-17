"""Report Builder — generates structured JSON reports and Markdown summaries.

Produces:
  1. JSON report with all diagnostic data
  2. Markdown summary for human review
  3. Severity classification
  4. Telegram-ready alert text
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from anomaly_detector import Anomaly
from copilot_interface import CopilotSession

CST = ZoneInfo("America/Chicago")
AGENT_DIR = Path(__file__).resolve().parent
REPORTS_DIR = AGENT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("agent.report_builder")


class ReportBuilder:
    """Builds structured diagnostic reports from anomaly + Copilot session data."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        rpt_cfg = config.get("reporting", {})
        self.reports_dir = Path(rpt_cfg.get("reports_dir", str(REPORTS_DIR)))
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.max_report_age_days = rpt_cfg.get("max_report_age_days", 30)

    # ─── Severity Assessment ──────────────────────────────────────────

    def assess_severity(
        self,
        anomaly: Anomaly,
        session: Optional[CopilotSession] = None,
    ) -> str:
        """Determine final severity based on anomaly + Copilot analysis.
        
        Copilot may upgrade/downgrade severity in its final round.
        """
        base_severity = anomaly.severity

        if session and session.final_summary:
            summary_lower = session.final_summary.lower()
            if "critical" in summary_lower:
                return "CRITICAL"
            if "high" in summary_lower and base_severity in ("MEDIUM", "LOW"):
                return "HIGH"
            if "low" in summary_lower and "false alarm" in summary_lower:
                return "LOW"

        return base_severity

    # ─── JSON Report ──────────────────────────────────────────────────

    def build_json_report(
        self,
        anomaly: Anomaly,
        context: Dict[str, Any],
        session: Optional[CopilotSession] = None,
    ) -> Dict[str, Any]:
        """Build a comprehensive JSON report."""
        severity = self.assess_severity(anomaly, session)
        now = datetime.now(CST)

        report = {
            "report_id": f"RPT_{anomaly.anomaly_id}_{now.strftime('%Y%m%d_%H%M%S')}",
            "generated_at": now.isoformat(),
            "severity": severity,
            "anomaly": anomaly.to_dict(),
            "context_snapshot": {
                "timestamp": context.get("timestamp_cst"),
                "daily_pnl": context.get("daily_pnl"),
                "open_positions": context.get("open_positions"),
                "last_trade_timestamp": context.get("last_trade_timestamp"),
                "signal_metrics": context.get("signal_metrics"),
                "dampening_status": context.get("dampening_status"),
                "market_regime": context.get("market_regime"),
                "vix_status": context.get("vix_status"),
            },
            "strategy_config": context.get("strategy_config"),
            "risk_parameters": context.get("risk_parameters"),
            "confidence_thresholds": context.get("confidence_thresholds"),
            "copilot_analysis": session.to_dict() if session else None,
            "recommendations": session.suggestions if session else [],
            "advisory_only": True,
            "disclaimer": (
                "This report is advisory only. No automated parameter changes "
                "or strategy modifications should be made without human review."
            ),
        }

        return report

    def save_json_report(self, report: Dict[str, Any]) -> Path:
        """Save JSON report to disk."""
        report_id = report.get("report_id", f"report_{datetime.now(CST).strftime('%Y%m%d_%H%M%S')}")
        filepath = self.reports_dir / f"{report_id}.json"
        try:
            with open(filepath, "w") as f:
                json.dump(report, f, indent=2, default=str)
            log.info("JSON report saved: %s", filepath)
        except Exception as exc:
            log.error("Failed to save JSON report: %s", exc)
        return filepath

    # ─── Markdown Summary ─────────────────────────────────────────────

    def build_markdown_summary(
        self,
        anomaly: Anomaly,
        context: Dict[str, Any],
        session: Optional[CopilotSession] = None,
    ) -> str:
        """Build a human-readable Markdown summary."""
        severity = self.assess_severity(anomaly, session)
        now = datetime.now(CST)
        daily = context.get("daily_pnl", {})
        signal_metrics = context.get("signal_metrics", {})
        dampening = context.get("dampening_status", {})

        severity_emoji = {
            "CRITICAL": "🔴",
            "HIGH": "🟠",
            "MEDIUM": "🟡",
            "LOW": "🟢",
        }

        lines = [
            f"# {severity_emoji.get(severity, '⚪')} Trading Analyst Report — {severity}",
            f"**Generated:** {now.strftime('%Y-%m-%d %H:%M:%S CST')}",
            f"**Anomaly ID:** `{anomaly.anomaly_id}`",
            f"**Type:** {anomaly.anomaly_type}",
            "",
            "---",
            "",
            "## 📋 Observed Behavior",
            anomaly.description,
            "",
            "## 📊 Recent Metrics",
            f"- **Daily P&L:** ${daily.get('realized_pnl', 0):.2f}",
            f"- **Trades Today:** {daily.get('trade_count', 0)} "
            f"(W: {daily.get('wins', 0)} / L: {daily.get('losses', 0)})",
            f"- **Last Trade:** {context.get('last_trade_timestamp', 'N/A')}",
            f"- **Signals Generated:** {signal_metrics.get('signals_generated', 'N/A')}",
            f"- **Signals Blocked:** {signal_metrics.get('signals_blocked', 'N/A')}",
            "",
            "## 🛡️ System State",
            f"- **Cooldown Active:** {dampening.get('cooldown_active', False)}",
            f"- **Kill Switch:** {dampening.get('kill_switch_active', False)}",
            f"- **Drawdown Halt:** {dampening.get('drawdown_halt', False)}",
            f"- **Maintenance Block:** {dampening.get('maintenance_block', False)}",
            "",
            "## ⚙️ Expected Behavior",
            "- The ES 15-minute strategy should generate ~3-4 trades per day during RTH.",
            "- EMA21 pullback and OR breakout signals should fire when ADX > 20 and trend is aligned.",
            "- Daily loss should not exceed the configured threshold.",
            "",
        ]

        if session and session.suggestions:
            lines.extend([
                "## 💡 Copilot Suggestions",
            ])
            for i, suggestion in enumerate(session.suggestions, 1):
                lines.append(f"{i}. {suggestion}")
            lines.append("")

        if session and session.final_summary:
            lines.extend([
                "## 🤖 Copilot Analysis Summary",
                session.final_summary,
                "",
            ])

        if session:
            lines.extend([
                f"## 📝 Copilot Session",
                f"- Duration: {session.duration_seconds}s",
                f"- Success: {'✅' if session.success else '❌'}",
            ])
            if session.error:
                lines.append(f"- Error: {session.error}")
            lines.append("")

        # Potential constraints
        block_reasons = signal_metrics.get("block_reasons", [])
        if block_reasons:
            lines.extend([
                "## 🚧 Potential Constraints",
                "Recent signal blocks detected in logs:",
            ])
            for reason in block_reasons[:5]:
                lines.append(f"- `{reason[:120]}`")
            lines.append("")

        lines.extend([
            "---",
            "⚠️ **This report is advisory only.** Do not auto-deploy changes.",
        ])

        return "\n".join(lines)

    def save_markdown_report(
        self,
        anomaly: Anomaly,
        context: Dict[str, Any],
        session: Optional[CopilotSession] = None,
    ) -> Path:
        """Save markdown report to disk."""
        md = self.build_markdown_summary(anomaly, context, session)
        now = datetime.now(CST)
        filepath = self.reports_dir / f"report_{anomaly.anomaly_id}_{now.strftime('%Y%m%d_%H%M%S')}.md"
        try:
            with open(filepath, "w") as f:
                f.write(md)
            log.info("Markdown report saved: %s", filepath)
        except Exception as exc:
            log.error("Failed to save Markdown report: %s", exc)
        return filepath

    # ─── Telegram Alert Text ──────────────────────────────────────────

    def build_telegram_alert(
        self,
        anomaly: Anomaly,
        context: Dict[str, Any],
        session: Optional[CopilotSession] = None,
    ) -> str:
        """Build a concise Telegram-ready alert message (HTML format)."""
        severity = self.assess_severity(anomaly, session)
        daily = context.get("daily_pnl", {})
        now = datetime.now(CST)

        severity_icon = {
            "CRITICAL": "🔴",
            "HIGH": "🟠",
            "MEDIUM": "🟡",
            "LOW": "🟢",
        }

        lines = [
            f"{severity_icon.get(severity, '⚪')} <b>Trading Analyst Alert — {severity}</b>",
            f"<code>{now.strftime('%H:%M CST')}</code>",
            "",
            f"<b>Type:</b> {anomaly.anomaly_type}",
            anomaly.description[:400],
            "",
            f"<b>Daily P&amp;L:</b> ${daily.get('realized_pnl', 0):.2f}",
            f"<b>Trades today:</b> {daily.get('trade_count', 0)}",
        ]

        if session and session.suggestions:
            lines.append("")
            lines.append("<b>Top Suggestions:</b>")
            for suggestion in session.suggestions[:3]:
                lines.append(f"• {suggestion[:100]}")

        lines.extend([
            "",
            "<i>Advisory only — no auto-changes made.</i>",
        ])

        text = "\n".join(lines)
        # Telegram message limit
        max_len = self.config.get("telegram", {}).get("max_message_length", 4096)
        if len(text) > max_len:
            text = text[: max_len - 20] + "\n... [truncated]"
        return text

    # ─── Cleanup ──────────────────────────────────────────────────────

    def cleanup_old_reports(self) -> int:
        """Remove reports older than max_report_age_days."""
        import time as _time
        cutoff = _time.time() - (self.max_report_age_days * 86400)
        removed = 0
        for f in self.reports_dir.iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        if removed:
            log.info("Cleaned up %d old reports", removed)
        return removed
