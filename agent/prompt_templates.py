"""Prompt Templates — compact diagnostic prompts for Copilot CLI.

Prompts are kept SHORT (~2K chars). Our code already parsed the logs —
we send only the distilled findings + file paths so Copilot can look
at the actual source files on disk if it needs more detail.
"""
from __future__ import annotations

from typing import Any, Dict

from anomaly_detector import Anomaly

PROJECT_ROOT = "/Users/svss/Documents/code/ShreeBot"


def _signal_summary(deep_log: dict) -> str:
    """One-liner per signal from parsed diagnostics."""
    diag = deep_log.get("strategy_diagnostics", {})
    if not diag.get("has_diagnostics"):
        return "  No signal diagnostics in logs."
    lines = []
    for label, info in diag.get("signal_summary", {}).items():
        top = info["top_reasons"][0]["reason"] if info.get("top_reasons") else "?"
        lines.append(f"  {label} ({info['description']}): {info['failure_count']}x fail — {top}")
    return "\n".join(lines)


def _bot_oneliner(deep_log: dict) -> str:
    bs = deep_log.get("bot_status", {})
    if not bs.get("running"):
        return "BOT NOT RUNNING (no heartbeat)"
    return f"RUNNING | price={bs.get('latest_price')} | bars={bs.get('bars_accumulated')} | hb={bs.get('latest_heartbeat')}"


def _or_oneliner(deep_log: dict) -> str:
    od = deep_log.get("opening_range", {})
    if not od.get("computed"):
        return "OR: not computed"
    return f"OR: H={od['or_high']} L={od['or_low']} range={od['or_range_points']}pts (${od['or_range_dollars']})"


# ═══════════════════════════════════════════════════════════════════════
#  NO-TRADE DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════

def build_no_trade_prompt(anomaly: Anomaly, context: Dict[str, Any]) -> str:
    daily = context.get("daily_pnl", {})
    deep = context.get("deep_log", {})
    dampening = context.get("dampening_status", {})
    diag = deep.get("strategy_diagnostics", {})
    blocks = deep.get("entry_blocks", [])
    skips = deep.get("skips", {})
    vx = deep.get("vx_live", {})

    block_lines = "\n".join(f"  {b['timestamp']}: {b['reason']}" for b in blocks[-3:]) or "  none"

    return f"""You are diagnosing a live MES futures trading bot that has NOT traded.

PROBLEM: {anomaly.description}

BOT: {_bot_oneliner(deep)}
{_or_oneliner(deep)}
VX: {vx.get('vx_price','N/A')} (mult={vx.get('multiplier','N/A')}x)
P&L today: ${daily.get('realized_pnl',0):.2f} | trades: {daily.get('trade_count',0)}
Last trade: {context.get('last_trade_timestamp','N/A')}
Cycles evaluated: {deep.get('trading_cycles','?')} | Skips: {skips.get('total_skips',0)} (all HOLD)
Cooldown: {dampening.get('cooldown_active',False)} | Kill-switch: {dampening.get('kill_switch_active',False)}

SIGNAL FAILURES (parsed from NO_SIGNAL diag — {diag.get('candle_cycles_analyzed',0)} cycles):
{_signal_summary(deep)}

Latest raw diag: {diag.get('latest_diag_raw','N/A')}

Entry blocks:
{block_lines}

STRATEGY REFERENCE:
  A=EMA21 Pullback Long (touch EMA21 + bounce + bullish bar + MACD>0)
  B=OR Breakout Long (prev close < OR_H, current close > OR_H)
  D=EMA21 Pullback Short (mirror of A in downtrend)
  E=OR Breakdown Short (mirror of B below OR_L)
  F=Trend Continuation (aligned EMA stack 9>21>50 or 9<21<50)

KEY FILES you can inspect on this machine for deeper analysis:
  Strategy code: {PROJECT_ROOT}/shree/strategies/es_fifteen_min.py
  Live bot log:  {PROJECT_ROOT}/logs/bot.log
  Config:        {PROJECT_ROOT}/config.yaml
  Orders DB:     {PROJECT_ROOT}/data/orders.db

DIAGNOSE:
1. Root cause — cite the specific signal failure values above.
2. Market vs System issue?
3. Is price just too far from EMA21 / stuck inside OR / no trend?
4. What needs to change for trades to fire?
5. Severity: CRITICAL / HIGH / MEDIUM / LOW?
"""


# ═══════════════════════════════════════════════════════════════════════
#  LOSS ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def build_loss_prompt(anomaly: Anomaly, context: Dict[str, Any]) -> str:
    daily = context.get("daily_pnl", {})
    deep = context.get("deep_log", {})
    dampening = context.get("dampening_status", {})
    trade = anomaly.payload.get("trade", {})
    loss = anomaly.payload.get("loss_usd", 0)
    vx = deep.get("vx_live", {})

    return f"""You are doing root-cause analysis on a losing MES futures trade.

PROBLEM: {anomaly.description}
Trade: {trade.get('action','?')} @ {trade.get('fill_price','?')} | PnL: ${loss:.2f} | cycle: {trade.get('trade_cycle_id','?')}

BOT: {_bot_oneliner(deep)}
{_or_oneliner(deep)}
VX: {vx.get('vx_price','N/A')} (mult={vx.get('multiplier','N/A')}x)
P&L today: ${daily.get('realized_pnl',0):.2f} | trades: {daily.get('trade_count',0)}
Cooldown: {dampening.get('cooldown_active',False)}

SIGNAL CONTEXT (what signals looked like around this trade):
{_signal_summary(deep)}

KEY FILES you can inspect on this machine:
  Strategy code: {PROJECT_ROOT}/shree/strategies/es_fifteen_min.py
  Live bot log:  {PROJECT_ROOT}/logs/bot.log
  Config:        {PROJECT_ROOT}/config.yaml

ANALYZE:
1. Was stop-loss appropriate for current VX/volatility?
2. Was entry signal marginal or solid?
3. Normal loss or system issue?
4. What adjustments would help?
5. Severity: CRITICAL / HIGH / MEDIUM / LOW?
"""


# ═══════════════════════════════════════════════════════════════════════
#  CONSECUTIVE LOSS
# ═══════════════════════════════════════════════════════════════════════

def build_consecutive_loss_prompt(anomaly: Anomaly, context: Dict[str, Any]) -> str:
    daily = context.get("daily_pnl", {})
    deep = context.get("deep_log", {})
    streak = anomaly.payload.get("streak", 0)
    vx = deep.get("vx_live", {})

    return f"""You are investigating {streak} consecutive losing trades in a live MES futures bot.

PROBLEM: {anomaly.description}

BOT: {_bot_oneliner(deep)}
{_or_oneliner(deep)}
VX: {vx.get('vx_price','N/A')}
P&L today: ${daily.get('realized_pnl',0):.2f} | trades: {daily.get('trade_count',0)}

SIGNAL DIAGNOSTICS:
{_signal_summary(deep)}

KEY FILES you can inspect on this machine:
  Strategy code: {PROJECT_ROOT}/shree/strategies/es_fifteen_min.py
  Live bot log:  {PROJECT_ROOT}/logs/bot.log
  Orders DB:     {PROJECT_ROOT}/data/orders.db

ANALYZE:
1. Is {streak}-loss streak expected at ~56% win rate?
2. Regime mismatch? Concentrated in one signal type?
3. Are cooldown escalations working?
4. Should bot self-pause or within normal variance?
5. Severity: CRITICAL / HIGH / MEDIUM / LOW?
"""


# ═══════════════════════════════════════════════════════════════════════
#  TRADE COMPLETE (loss below big threshold — lightweight Copilot check)
# ═══════════════════════════════════════════════════════════════════════

def build_trade_complete_prompt(anomaly: Anomaly, context: Dict[str, Any]) -> str:
    daily = context.get("daily_pnl", {})
    deep = context.get("deep_log", {})
    payload = anomaly.payload or {}
    vx = deep.get("vx_live", {})

    return f"""Quick review of a completed MES futures trade.

TRADE: {payload.get('action','?')} filled at {payload.get('fill_price','?')} | PnL: {payload.get('pnl_display','?')} | order: {payload.get('order_id','?')}

BOT: {_bot_oneliner(deep)}
{_or_oneliner(deep)}
VX: {vx.get('vx_price','N/A')}
P&L today: ${daily.get('realized_pnl',0):.2f} | trades: {daily.get('trade_count',0)}

SIGNAL CONTEXT:
{_signal_summary(deep)}

KEY FILES on this machine:
  Strategy: {PROJECT_ROOT}/shree/strategies/es_fifteen_min.py
  Log:      {PROJECT_ROOT}/logs/bot.log

REVIEW:
1. Was entry quality good?
2. Was stop distance appropriate?
3. Any quick improvement?
4. Severity: LOW / MEDIUM?
"""


# ═══════════════════════════════════════════════════════════════════════
#  DISPATCHER
# ═══════════════════════════════════════════════════════════════════════

def build_prompt(anomaly: Anomaly, context: Dict[str, Any]) -> str:
    dispatch = {
        "NO_TRADE": build_no_trade_prompt,
        "TRADE_LOSS": build_loss_prompt,
        "DAILY_LOSS": build_loss_prompt,
        "CONSEC_LOSS": build_consecutive_loss_prompt,
        "TRADE_COMPLETE": build_trade_complete_prompt,
    }
    builder = dispatch.get(anomaly.anomaly_type, build_no_trade_prompt)
    return builder(anomaly, context)
