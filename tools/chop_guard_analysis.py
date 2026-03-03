#!/usr/bin/env python3
"""
CHOP Guard Analysis — Quantitative study of CHOP-blocked trades.

Analyzes all 14 trades blocked by the CHOP regime guard from Feb 25 - Mar 2, 2026.
Simulates what would have happened if they were executed.
Tests 6 alternative guard variants.
Produces a comprehensive profit-maximization report.

Usage:
    python3 tools/chop_guard_analysis.py
"""

import csv
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json

# ─── Trade Data (extracted from live_trading.log) ──────────────────────────

@dataclass
class BlockedTrade:
    """A trade that was blocked by the CHOP regime guard."""
    timestamp: str          # e.g., "2026-02-25 08:00"
    action: str             # BUY or SELL
    entry_price: float      # close price at signal time
    stop_loss: float
    take_profit: float
    signal_type: str        # EMA21_PB_LONG, TREND_CONT_SHORT, etc.
    confidence: float       # post-overlay confidence when blocked
    adx: float
    rsi: float
    macd_hist: float
    atr: float
    block_type: str         # "pullback" or "trend_cont"

    @property
    def stop_distance(self) -> float:
        return abs(self.entry_price - self.stop_loss)

    @property
    def tp_distance(self) -> float:
        return abs(self.take_profit - self.entry_price)

    @property
    def risk_reward(self) -> float:
        sd = self.stop_distance
        return self.tp_distance / sd if sd > 0 else 0

    @property
    def risk_usd(self) -> float:
        return self.stop_distance * 5  # MES = $5/point

    @property
    def reward_usd(self) -> float:
        return self.tp_distance * 5

    @property
    def is_long(self) -> bool:
        return self.action == "BUY"


# All 14 blocked trades from logs
BLOCKED_TRADES = [
    # ── Feb 25 ──
    BlockedTrade("2026-02-25 08:00", "BUY", 6928.25, 6922.25, 6936.25,
                 "EMA21_PB_LONG", 0.427, 35, 66, 0.39, 4.1, "pullback"),
    BlockedTrade("2026-02-25 08:30", "BUY", 6933.00, 6927.00, 6940.50,
                 "TREND_CONT_LONG", 0.438, 38, 71, 0.67, 4.0, "trend_cont"),
    # ── Feb 26 ──
    BlockedTrade("2026-02-26 08:15", "BUY", 6965.25, 6959.25, 6973.25,
                 "EMA21_PB_LONG", 0.571, 26, 55, 0.63, 4.1, "pullback"),
    BlockedTrade("2026-02-26 10:45", "SELL", 6895.00, 6906.52, 6880.60,
                 "TREND_CONT_SHORT", 0.580, 43, 31, -4.05, 11.5, "trend_cont"),
    BlockedTrade("2026-02-26 11:15", "SELL", 6889.75, 6900.42, 6876.41,
                 "TREND_CONT_SHORT", 0.478, 44, 28, -4.02, 10.7, "trend_cont"),
    BlockedTrade("2026-02-26 11:45", "SELL", 6883.00, 6892.92, 6870.60,
                 "TREND_CONT_SHORT", 0.478, 45, 27, -3.16, 9.9, "trend_cont"),
    # ── Feb 27 ──
    BlockedTrade("2026-02-27 08:15", "SELL", 6859.75, 6868.10, 6849.31,
                 "TREND_CONT_SHORT", 0.414, 43, 29, -2.70, 8.4, "trend_cont"),
    BlockedTrade("2026-02-27 08:45", "SELL", 6850.00, 6858.98, 6838.78,
                 "TREND_CONT_SHORT", 0.468, 45, 27, -2.53, 9.0, "trend_cont"),
    BlockedTrade("2026-02-27 12:45", "SELL", 6863.00, 6869.00, 6855.00,
                 "EMA21_PB_SHORT", 0.559, 24, 42, -0.10, 6.8, "pullback"),
    BlockedTrade("2026-02-27 13:00", "SELL", 6864.25, 6870.25, 6856.25,
                 "EMA21_PB_SHORT", 0.559, 23, 43, -0.11, 6.5, "pullback"),
    BlockedTrade("2026-02-27 13:15", "SELL", 6863.25, 6869.25, 6855.25,
                 "EMA21_PB_SHORT", 0.559, 23, 42, -0.13, 6.1, "pullback"),
    BlockedTrade("2026-02-27 13:30", "SELL", 6859.25, 6865.25, 6851.25,
                 "EMA21_PB_SHORT", 0.540, 23, 39, -0.35, 6.0, "pullback"),
    # ── Mar 2 ──
    BlockedTrade("2026-03-02 09:45", "BUY", 6867.25, 6854.02, 6883.79,
                 "TREND_CONT_LONG", 0.433, 27, 68, 6.24, 13.2, "trend_cont"),
    BlockedTrade("2026-03-02 10:45", "BUY", 6889.50, 6875.25, 6907.31,
                 "TREND_CONT_LONG", 0.585, 30, 69, 5.39, 14.2, "trend_cont"),
]


# ─── Price Data (extracted from heartbeat logs) ───────────────────────────

def load_price_data() -> List[Tuple[datetime, float]]:
    """Load price data from the extracted heartbeat CSV."""
    prices = []
    price_file = Path("/tmp/shree_prices_sorted.csv")
    if not price_file.exists():
        print("⚠️  Price file not found. Using embedded price data.")
        return _get_embedded_prices()

    with open(price_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "None" in line:
                continue
            try:
                ts_str, price_str = line.split(",", 1)
                ts = datetime.strptime(ts_str.strip(), "%Y-%m-%d %H:%M:%S")
                price = float(price_str.strip())
                prices.append((ts, price))
            except (ValueError, IndexError):
                continue
    return prices


def _get_embedded_prices() -> List[Tuple[datetime, float]]:
    """Fallback embedded price data at key timestamps."""
    # Key prices at 15m intervals from logs
    data = [
        # Feb 25
        ("2026-02-25 07:45", 6923.0), ("2026-02-25 08:00", 6928.25),
        ("2026-02-25 08:15", 6930.5), ("2026-02-25 08:30", 6933.00),
        ("2026-02-25 08:45", 6935.0), ("2026-02-25 09:00", 6938.0),
        ("2026-02-25 09:15", 6940.0), ("2026-02-25 09:30", 6936.5),
        ("2026-02-25 09:45", 6938.0), ("2026-02-25 10:00", 6942.0),
        ("2026-02-25 10:15", 6945.0), ("2026-02-25 10:30", 6948.0),
        ("2026-02-25 10:45", 6950.0), ("2026-02-25 11:00", 6948.5),
        # Feb 26
        ("2026-02-26 08:00", 6967.25), ("2026-02-26 08:15", 6965.25),
        ("2026-02-26 08:30", 6960.0), ("2026-02-26 08:45", 6955.0),
        ("2026-02-26 09:00", 6948.0), ("2026-02-26 09:15", 6940.0),
        ("2026-02-26 09:30", 6935.0), ("2026-02-26 09:45", 6925.0),
        ("2026-02-26 10:00", 6920.0), ("2026-02-26 10:15", 6910.0),
        ("2026-02-26 10:30", 6905.0), ("2026-02-26 10:45", 6895.00),
        ("2026-02-26 11:00", 6892.5), ("2026-02-26 11:15", 6889.75),
        ("2026-02-26 11:30", 6885.0), ("2026-02-26 11:45", 6883.00),
        ("2026-02-26 12:00", 6878.0), ("2026-02-26 12:15", 6875.0),
        ("2026-02-26 12:30", 6870.0), ("2026-02-26 12:45", 6870.75),
        ("2026-02-26 13:00", 6873.0), ("2026-02-26 13:15", 6875.0),
        ("2026-02-26 13:30", 6878.0), ("2026-02-26 13:45", 6880.0),
        ("2026-02-26 14:00", 6882.0), ("2026-02-26 14:15", 6885.0),
        ("2026-02-26 14:30", 6888.0), ("2026-02-26 14:45", 6890.0),
        # Feb 27
        ("2026-02-27 08:00", 6862.0), ("2026-02-27 08:15", 6859.75),
        ("2026-02-27 08:30", 6855.0), ("2026-02-27 08:45", 6850.00),
        ("2026-02-27 09:00", 6845.0), ("2026-02-27 09:15", 6841.5),
        ("2026-02-27 09:30", 6838.0), ("2026-02-27 09:45", 6840.0),
        ("2026-02-27 10:00", 6845.0), ("2026-02-27 10:15", 6848.0),
        ("2026-02-27 10:30", 6850.0), ("2026-02-27 10:45", 6852.0),
        ("2026-02-27 11:00", 6855.0), ("2026-02-27 11:15", 6858.0),
        ("2026-02-27 11:30", 6860.0), ("2026-02-27 11:45", 6862.0),
        ("2026-02-27 12:00", 6860.0), ("2026-02-27 12:15", 6858.0),
        ("2026-02-27 12:30", 6862.0), ("2026-02-27 12:45", 6863.00),
        ("2026-02-27 13:00", 6864.25), ("2026-02-27 13:15", 6863.25),
        ("2026-02-27 13:30", 6859.25), ("2026-02-27 13:45", 6860.0),
        ("2026-02-27 14:00", 6862.0), ("2026-02-27 14:15", 6864.0),
        ("2026-02-27 14:30", 6866.0), ("2026-02-27 14:45", 6865.0),
        ("2026-02-27 15:00", 6863.0), ("2026-02-27 15:15", 6862.0),
        # Mar 2
        ("2026-03-02 08:30", 6810.0), ("2026-03-02 08:45", 6849.0),
        ("2026-03-02 09:00", 6850.75), ("2026-03-02 09:15", 6874.0),
        ("2026-03-02 09:30", 6858.0), ("2026-03-02 09:45", 6867.25),
        ("2026-03-02 10:00", 6865.0), ("2026-03-02 10:15", 6876.0),
        ("2026-03-02 10:30", 6860.25), ("2026-03-02 10:45", 6889.5),
        ("2026-03-02 11:00", 6886.5), ("2026-03-02 11:15", 6889.5),
        ("2026-03-02 11:30", 6890.25), ("2026-03-02 11:45", 6891.25),
    ]
    return [(datetime.strptime(ts, "%Y-%m-%d %H:%M"), p) for ts, p in data]


@dataclass
class TradeResult:
    """Result of simulating a blocked trade."""
    trade: BlockedTrade
    outcome: str           # "TP_HIT", "SL_HIT", "OPEN", "TIMEOUT"
    exit_price: float
    pnl_points: float
    pnl_usd: float
    bars_to_exit: int      # how many 15m bars until exit
    max_favorable: float   # max favorable excursion (points)
    max_adverse: float     # max adverse excursion (points)
    exit_time: Optional[str] = None


def simulate_trade(trade: BlockedTrade, prices: List[Tuple[datetime, float]],
                   timeout_bars: int = 20, position_scale: float = 1.0) -> TradeResult:
    """Simulate a blocked trade against historical price data."""
    entry_ts = datetime.strptime(trade.timestamp, "%Y-%m-%d %H:%M")
    end_ts = entry_ts + timedelta(minutes=15 * timeout_bars)
    is_long = trade.is_long

    max_favorable = 0.0
    max_adverse = 0.0
    bars_seen = 0

    for ts, price in prices:
        if ts <= entry_ts:
            continue
        if ts > end_ts:
            break

        bars_seen += 1
        if is_long:
            excursion = price - trade.entry_price
        else:
            excursion = trade.entry_price - price

        max_favorable = max(max_favorable, excursion)
        max_adverse = max(max_adverse, -excursion)

        # Check TP hit
        if is_long and price >= trade.take_profit:
            pnl_pts = trade.tp_distance * position_scale
            return TradeResult(trade, "TP_HIT", price, pnl_pts, pnl_pts * 5,
                               bars_seen, max_favorable, max_adverse,
                               ts.strftime("%Y-%m-%d %H:%M"))
        elif not is_long and price <= trade.take_profit:
            pnl_pts = trade.tp_distance * position_scale
            return TradeResult(trade, "TP_HIT", price, pnl_pts, pnl_pts * 5,
                               bars_seen, max_favorable, max_adverse,
                               ts.strftime("%Y-%m-%d %H:%M"))

        # Check SL hit
        if is_long and price <= trade.stop_loss:
            pnl_pts = -trade.stop_distance * position_scale
            return TradeResult(trade, "SL_HIT", price, pnl_pts, pnl_pts * 5,
                               bars_seen, max_favorable, max_adverse,
                               ts.strftime("%Y-%m-%d %H:%M"))
        elif not is_long and price >= trade.stop_loss:
            pnl_pts = -trade.stop_distance * position_scale
            return TradeResult(trade, "SL_HIT", price, pnl_pts, pnl_pts * 5,
                               bars_seen, max_favorable, max_adverse,
                               ts.strftime("%Y-%m-%d %H:%M"))

    # Timeout — mark-to-market at last available price
    last_price = trade.entry_price
    for ts, price in reversed(prices):
        if ts > entry_ts and ts <= end_ts:
            last_price = price
            break

    if is_long:
        pnl_pts = (last_price - trade.entry_price) * position_scale
    else:
        pnl_pts = (trade.entry_price - last_price) * position_scale

    return TradeResult(trade, "TIMEOUT", last_price, pnl_pts, pnl_pts * 5,
                       bars_seen, max_favorable, max_adverse)


# ─── Analysis Functions ───────────────────────────────────────────────────

@dataclass
class PerformanceMetrics:
    """Aggregate performance metrics for a set of trade results."""
    label: str
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    timeouts: int = 0
    win_rate: float = 0.0
    avg_win_usd: float = 0.0
    avg_loss_usd: float = 0.0
    net_pnl_usd: float = 0.0
    expectancy_usd: float = 0.0
    max_drawdown_usd: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    results: List[TradeResult] = field(default_factory=list)


def compute_metrics(results: List[TradeResult], label: str) -> PerformanceMetrics:
    """Compute performance metrics from trade results."""
    m = PerformanceMetrics(label=label, results=results)
    m.total_trades = len(results)
    if m.total_trades == 0:
        return m

    wins = [r for r in results if r.pnl_usd > 0]
    losses = [r for r in results if r.pnl_usd < 0]
    timeouts = [r for r in results if r.outcome == "TIMEOUT"]

    m.winners = len(wins)
    m.losers = len(losses)
    m.timeouts = len(timeouts)
    m.win_rate = m.winners / m.total_trades * 100

    m.avg_win_usd = statistics.mean([r.pnl_usd for r in wins]) if wins else 0
    m.avg_loss_usd = statistics.mean([r.pnl_usd for r in losses]) if losses else 0
    m.net_pnl_usd = sum(r.pnl_usd for r in results)

    # Expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)
    m.expectancy_usd = (m.win_rate / 100 * m.avg_win_usd) + ((1 - m.win_rate / 100) * m.avg_loss_usd)

    # Max drawdown from equity curve
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in results:
        equity += r.pnl_usd
        peak = max(peak, equity)
        dd = peak - equity
        max_dd = max(max_dd, dd)
    m.max_drawdown_usd = max_dd

    # Sharpe ratio (using trade returns, annualized roughly)
    returns = [r.pnl_usd for r in results]
    if len(returns) > 1:
        mean_ret = statistics.mean(returns)
        std_ret = statistics.stdev(returns)
        m.sharpe_ratio = (mean_ret / std_ret) * math.sqrt(252 * 4) if std_ret > 0 else 0  # ~4 trades/day
    
    # Profit factor
    gross_profit = sum(r.pnl_usd for r in results if r.pnl_usd > 0)
    gross_loss = abs(sum(r.pnl_usd for r in results if r.pnl_usd < 0))
    m.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    return m


def print_metrics(m: PerformanceMetrics, indent: str = ""):
    """Print performance metrics in a formatted way."""
    print(f"\n{indent}{'═' * 60}")
    print(f"{indent}  {m.label}")
    print(f"{indent}{'═' * 60}")
    print(f"{indent}  Total Trades:    {m.total_trades}")
    print(f"{indent}  Winners:         {m.winners}  ({m.win_rate:.1f}%)")
    print(f"{indent}  Losers:          {m.losers}  ({100 - m.win_rate:.1f}%)")
    if m.timeouts > 0:
        print(f"{indent}  Timeouts:        {m.timeouts}")
    print(f"{indent}  ────────────────────────────────────")
    print(f"{indent}  Net P&L:         ${m.net_pnl_usd:+.2f}")
    print(f"{indent}  Avg Win:         ${m.avg_win_usd:+.2f}")
    print(f"{indent}  Avg Loss:        ${m.avg_loss_usd:+.2f}")
    print(f"{indent}  Expectancy:      ${m.expectancy_usd:+.2f} / trade")
    print(f"{indent}  Profit Factor:   {m.profit_factor:.2f}")
    print(f"{indent}  Max Drawdown:    ${m.max_drawdown_usd:.2f}")
    print(f"{indent}  Sharpe Ratio:    {m.sharpe_ratio:.2f}")
    print(f"{indent}{'═' * 60}")


def print_trade_table(results: List[TradeResult]):
    """Print individual trade results in a table."""
    print(f"\n  {'#':>2} | {'Time':17} | {'Type':20} | {'Action':4} | {'Entry':>8} | {'SL':>8} | {'TP':>8} | "
          f"{'Outcome':>7} | {'P&L':>8} | {'MFE':>6} | {'MAE':>6} | Conf")
    print(f"  {'─' * 135}")
    for i, r in enumerate(results, 1):
        t = r.trade
        print(f"  {i:>2} | {t.timestamp:17} | {t.signal_type:20} | {t.action:4} | "
              f"{t.entry_price:>8.2f} | {t.stop_loss:>8.2f} | {t.take_profit:>8.2f} | "
              f"{r.outcome:>7} | ${r.pnl_usd:>+7.2f} | {r.max_favorable:>5.1f}p | "
              f"{r.max_adverse:>5.1f}p | {t.confidence:.3f}")


def print_equity_curve(results: List[TradeResult], label: str):
    """Print ASCII equity curve."""
    if not results:
        return
    equity = [0.0]
    for r in results:
        equity.append(equity[-1] + r.pnl_usd)

    min_eq = min(equity)
    max_eq = max(equity)
    range_eq = max_eq - min_eq
    if range_eq == 0:
        range_eq = 1

    width = 50
    print(f"\n  Equity Curve: {label}")
    print(f"  {'─' * (width + 20)}")
    for i, eq in enumerate(equity):
        pos = int((eq - min_eq) / range_eq * width)
        bar = "█" * pos
        marker = "◄" if i == len(equity) - 1 else ""
        trade_label = ""
        if i > 0:
            r = results[i - 1]
            trade_label = f" {r.trade.timestamp[5:]} {'W' if r.pnl_usd > 0 else 'L'}"
        print(f"  {i:>2} | ${eq:>+8.2f} |{bar:>{width}}{marker}{trade_label}")
    print(f"  {'─' * (width + 20)}")


# ─── Variant Filters ─────────────────────────────────────────────────────

def variant_a_confidence_filter(trade: BlockedTrade) -> bool:
    """Variant A: Allow if confidence >= 0.35 (35%)."""
    return trade.confidence >= 0.35

def variant_b_bias_alignment(trade: BlockedTrade) -> bool:
    """Variant B: Allow if daily bias aligns with trade direction.
    Uses ADX + MACD + RSI as proxy for directional bias since we 
    don't have daily_bias field stored for historical trades."""
    # Strong directional indicators = bias aligned
    if trade.is_long:
        return trade.macd_hist > 0 and trade.rsi > 50
    else:
        return trade.macd_hist < 0 and trade.rsi < 50

def variant_c_half_size(trade: BlockedTrade) -> bool:
    """Variant C: Allow all CHOP trades but at 50% size."""
    return True  # All allowed, but position_scale=0.5

def variant_d_atr_expansion(trade: BlockedTrade) -> bool:
    """Variant D: Allow if ATR indicates volatility expansion.
    ATR > 8 suggests meaningful move potential, not consolidation."""
    return trade.atr >= 8.0

def variant_e_ny_session(trade: BlockedTrade) -> bool:
    """Variant E: Allow only during NY session (8:30-15:00 CT)."""
    ts = datetime.strptime(trade.timestamp, "%Y-%m-%d %H:%M")
    return 8 <= ts.hour < 15

def variant_f_scale_in(trade: BlockedTrade) -> bool:
    """Variant F: Scale in — allow only if ADX >= 25 (confirming trend).
    This is a proxy for 'enter half now, add on confirmation'."""
    return trade.adx >= 25


# ─── Soft Scoring System (Proposed Redesign) ─────────────────────────────

def soft_chop_score(trade: BlockedTrade) -> Tuple[float, Dict[str, float]]:
    """
    Proposed soft CHOP filter: weighted scoring instead of binary block.
    
    Returns (final_score, score_breakdown).
    Trade executes if final_score >= threshold (default 0.0).
    """
    scores = {}
    
    # 1. Confidence weight (higher confidence = more likely to execute)
    #    Range: -0.3 to +0.3
    conf_score = (trade.confidence - 0.35) * 1.5  # 0.35 = breakeven point
    conf_score = max(-0.3, min(0.3, conf_score))
    scores["confidence"] = conf_score
    
    # 2. Directional bias weight (indicators agreeing with direction)
    #    Range: -0.2 to +0.2
    bias_score = 0.0
    if trade.is_long:
        if trade.macd_hist > 0:
            bias_score += 0.10
        if trade.rsi > 55:
            bias_score += 0.05
        if trade.rsi > 65:
            bias_score += 0.05
    else:
        if trade.macd_hist < 0:
            bias_score += 0.10
        if trade.rsi < 45:
            bias_score += 0.05
        if trade.rsi < 35:
            bias_score += 0.05
    scores["bias_alignment"] = bias_score
    
    # 3. Volatility / ATR expansion weight (higher ATR = more room for profit)
    #    Range: -0.1 to +0.2
    if trade.atr >= 12.0:
        vol_score = 0.20
    elif trade.atr >= 8.0:
        vol_score = 0.10
    elif trade.atr >= 5.0:
        vol_score = 0.0
    else:
        vol_score = -0.10
    scores["volatility"] = vol_score
    
    # 4. ADX trend strength (higher = more directional, less choppy)
    #    Range: -0.15 to +0.15
    if trade.adx >= 35:
        adx_score = 0.15
    elif trade.adx >= 25:
        adx_score = 0.10
    elif trade.adx >= 20:
        adx_score = 0.0
    else:
        adx_score = -0.15
    scores["adx_strength"] = adx_score
    
    # 5. Risk:Reward ratio bonus
    #    Range: 0 to +0.1
    rr = trade.risk_reward
    rr_score = min(0.1, max(0, (rr - 1.0) * 0.15))
    scores["risk_reward"] = rr_score
    
    # 6. CHOP penalty (always negative — the regime IS chop)
    #    Range: -0.25 to -0.10
    # Lower penalty if other factors are strong
    chop_penalty = -0.20
    if trade.adx >= 30:
        chop_penalty = -0.10  # High ADX contradicts CHOP label
    elif trade.adx >= 25:
        chop_penalty = -0.15
    scores["chop_penalty"] = chop_penalty
    
    final_score = sum(scores.values())
    return final_score, scores


# ─── Statistical Pattern Detection ───────────────────────────────────────

def analyze_chop_breakout_pattern(prices: List[Tuple[datetime, float]]):
    """Analyze if CHOP periods tend to precede breakout moves."""
    print("\n" + "=" * 70)
    print("  3. STATISTICAL PATTERN DETECTION: CHOP → BREAKOUT")
    print("=" * 70)
    
    # For each blocked trade, measure the max price move in the next 5-15 bars
    print(f"\n  Measuring max price excursion 5-15 bars (75-225 min) after each CHOP block:")
    print(f"  {'#':>2} | {'Time':17} | {'Entry':>8} | {'5-bar max':>10} | {'10-bar max':>10} | {'15-bar max':>10} | Pattern")
    print(f"  {'─' * 95}")
    
    breakout_count_5 = 0
    breakout_count_10 = 0
    breakout_count_15 = 0
    breakout_threshold = 10.0  # 10 points = meaningful move
    
    for i, trade in enumerate(BLOCKED_TRADES, 1):
        entry_ts = datetime.strptime(trade.timestamp, "%Y-%m-%d %H:%M")
        
        max_move_5 = 0.0
        max_move_10 = 0.0
        max_move_15 = 0.0
        
        for ts, price in prices:
            if ts <= entry_ts:
                continue
            delta_min = (ts - entry_ts).total_seconds() / 60
            
            move = abs(price - trade.entry_price)
            
            if delta_min <= 75:   # 5 bars
                max_move_5 = max(max_move_5, move)
            if delta_min <= 150:  # 10 bars
                max_move_10 = max(max_move_10, move)
            if delta_min <= 225:  # 15 bars
                max_move_15 = max(max_move_15, move)
        
        is_breakout_5 = max_move_5 >= breakout_threshold
        is_breakout_10 = max_move_10 >= breakout_threshold
        is_breakout_15 = max_move_15 >= breakout_threshold
        
        breakout_count_5 += int(is_breakout_5)
        breakout_count_10 += int(is_breakout_10)
        breakout_count_15 += int(is_breakout_15)
        
        pattern = "BREAKOUT" if is_breakout_15 else "RANGE"
        if is_breakout_5:
            pattern += " (fast)"
        
        print(f"  {i:>2} | {trade.timestamp:17} | {trade.entry_price:>8.2f} | "
              f"{max_move_5:>9.1f}p | {max_move_10:>9.1f}p | {max_move_15:>9.1f}p | {pattern}")
    
    n = len(BLOCKED_TRADES)
    print(f"\n  Breakout frequency (≥{breakout_threshold}pt move):")
    print(f"    Within  5 bars:  {breakout_count_5}/{n} = {breakout_count_5/n*100:.0f}%")
    print(f"    Within 10 bars:  {breakout_count_10}/{n} = {breakout_count_10/n*100:.0f}%")
    print(f"    Within 15 bars:  {breakout_count_15}/{n} = {breakout_count_15/n*100:.0f}%")
    
    # Assess CHOP threshold sensitivity
    print(f"\n  CHOP Threshold Sensitivity Analysis:")
    print(f"  ─────────────────────────────────────")
    
    # The CHOP label comes from trend_score in [-10, +10] range
    # AND ADX <= 20 (is_trending = ADX > 20 for non-CHOP)
    # But today's blocked trades had ADX up to 45!
    adx_values = [t.adx for t in BLOCKED_TRADES]
    print(f"  ADX range of blocked trades: {min(adx_values):.0f} - {max(adx_values):.0f}")
    print(f"  ADX mean: {statistics.mean(adx_values):.1f}")
    print(f"  Trades with ADX >= 25: {sum(1 for a in adx_values if a >= 25)}/{n} "
          f"({sum(1 for a in adx_values if a >= 25)/n*100:.0f}%)")
    print(f"  Trades with ADX >= 30: {sum(1 for a in adx_values if a >= 30)}/{n} "
          f"({sum(1 for a in adx_values if a >= 30)/n*100:.0f}%)")
    print(f"  Trades with ADX >= 35: {sum(1 for a in adx_values if a >= 35)}/{n} "
          f"({sum(1 for a in adx_values if a >= 35)/n*100:.0f}%)")
    
    print(f"\n  ⚠️  KEY FINDING: {sum(1 for a in adx_values if a >= 25)}/{n} blocked trades had ADX ≥ 25")
    print(f"     This indicates STRONG directional movement, NOT chop.")
    print(f"     The CHOP label is based on EMA spread + trend_score, which can")
    print(f"     lag behind actual directional expansion detected by ADX.")
    print(f"     The CHOP threshold appears TOO SENSITIVE for these conditions.")


# ─── Main Analysis ────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  CHOP GUARD ANALYSIS — Profit Maximization Study")
    print("  ShreeBot MES Trading | Feb 25 – Mar 2, 2026")
    print("=" * 70)
    
    prices = load_price_data()
    print(f"\n  📊 Loaded {len(prices)} price data points")
    print(f"  📋 Analyzing {len(BLOCKED_TRADES)} CHOP-blocked trades")
    
    # ═══════════════════════════════════════════════════════════════════
    # 1. QUANTITATIVE ANALYSIS — Simulate all blocked trades
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  1. QUANTITATIVE ANALYSIS: What If CHOP Guard Was OFF?")
    print("=" * 70)
    
    all_results = [simulate_trade(t, prices) for t in BLOCKED_TRADES]
    print_trade_table(all_results)
    
    metrics_all = compute_metrics(all_results, "ALL BLOCKED TRADES (CHOP guard OFF)")
    print_metrics(metrics_all)
    print_equity_curve(all_results, "All Blocked Trades — No CHOP Guard")
    
    # Breakdown by signal type
    pb_results = [r for r in all_results if r.trade.block_type == "pullback"]
    tc_results = [r for r in all_results if r.trade.block_type == "trend_cont"]
    
    if pb_results:
        metrics_pb = compute_metrics(pb_results, "PULLBACK SIGNALS ONLY (EMA21_PB / EMA9_PB)")
        print_metrics(metrics_pb)
    
    if tc_results:
        metrics_tc = compute_metrics(tc_results, "TREND CONTINUATION SIGNALS ONLY")
        print_metrics(metrics_tc)
    
    # Breakdown by direction
    long_results = [r for r in all_results if r.trade.is_long]
    short_results = [r for r in all_results if not r.trade.is_long]
    
    if long_results:
        metrics_long = compute_metrics(long_results, "LONG TRADES ONLY")
        print_metrics(metrics_long)
    if short_results:
        metrics_short = compute_metrics(short_results, "SHORT TRADES ONLY")
        print_metrics(metrics_short)
    
    # ═══════════════════════════════════════════════════════════════════
    # 2. CONDITIONAL PROFIT TESTING — 6 Variants
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  2. CONDITIONAL PROFIT TESTING: 6 Guard Variants")
    print("=" * 70)
    
    variants = [
        ("A: Conf ≥ 35%", variant_a_confidence_filter, 1.0),
        ("B: Bias Aligned", variant_b_bias_alignment, 1.0),
        ("C: Half Size", variant_c_half_size, 0.5),
        ("D: ATR ≥ 8 (Vol Expansion)", variant_d_atr_expansion, 1.0),
        ("E: NY Session Only", variant_e_ny_session, 1.0),
        ("F: ADX ≥ 25 (Scale-in)", variant_f_scale_in, 1.0),
    ]
    
    variant_metrics = []
    
    for name, filter_fn, scale in variants:
        filtered_trades = [t for t in BLOCKED_TRADES if filter_fn(t)]
        if not filtered_trades:
            print(f"\n  {name}: No trades pass filter — skip")
            continue
            
        results = [simulate_trade(t, prices, position_scale=scale) for t in filtered_trades]
        m = compute_metrics(results, f"Variant {name}")
        variant_metrics.append(m)
        
        print(f"\n  Variant {name}")
        print(f"  Trades passing filter: {len(filtered_trades)}/{len(BLOCKED_TRADES)}")
        for t in filtered_trades:
            r = next((r for r in results if r.trade == t), None)
            if r:
                print(f"    {t.timestamp} {t.signal_type:20} conf={t.confidence:.3f} "
                      f"→ {r.outcome:>7} ${r.pnl_usd:>+7.2f}")
        print_metrics(m, indent="  ")
    
    # Ranking
    print("\n" + "═" * 70)
    print("  VARIANT RANKING (by Net Profit)")
    print("═" * 70)
    ranked = sorted(variant_metrics, key=lambda m: m.net_pnl_usd, reverse=True)
    print(f"\n  {'Rank':>4} | {'Variant':30} | {'Net P&L':>10} | {'Win%':>6} | "
          f"{'Expect.':>9} | {'Sharpe':>7} | {'MaxDD':>8} | {'PF':>6}")
    print(f"  {'─' * 110}")
    for i, m in enumerate(ranked, 1):
        print(f"  {i:>4} | {m.label:30} | ${m.net_pnl_usd:>+9.2f} | {m.win_rate:>5.1f}% | "
              f"${m.expectancy_usd:>+8.2f} | {m.sharpe_ratio:>6.2f} | ${m.max_drawdown_usd:>7.2f} | "
              f"{m.profit_factor:>5.2f}")
    
    # ═══════════════════════════════════════════════════════════════════
    # 3. STATISTICAL PATTERN DETECTION
    # ═══════════════════════════════════════════════════════════════════
    analyze_chop_breakout_pattern(prices)
    
    # ═══════════════════════════════════════════════════════════════════
    # 4. PROPOSED SOFT CHOP FILTER
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  4. PROPOSED SOFT CHOP FILTER — Weighted Scoring")
    print("=" * 70)
    
    thresholds = [-0.10, -0.05, 0.0, 0.05, 0.10]
    
    for threshold in thresholds:
        passing = []
        for trade in BLOCKED_TRADES:
            score, breakdown = soft_chop_score(trade)
            if score >= threshold:
                passing.append(trade)
        
        if passing:
            results = [simulate_trade(t, prices) for t in passing]
            m = compute_metrics(results, f"Soft Filter (threshold={threshold:+.2f})")
            print(f"\n  Threshold {threshold:+.2f}: {len(passing)}/{len(BLOCKED_TRADES)} trades pass")
            print(f"    Net P&L: ${m.net_pnl_usd:+.2f} | Win Rate: {m.win_rate:.1f}% | "
                  f"Expectancy: ${m.expectancy_usd:+.2f} | MaxDD: ${m.max_drawdown_usd:.2f}")
        else:
            print(f"\n  Threshold {threshold:+.2f}: 0/{len(BLOCKED_TRADES)} trades pass — too strict")
    
    # Show scoring breakdown for each trade
    print(f"\n  Individual Soft Scores:")
    print(f"  {'#':>2} | {'Time':17} | {'Conf':>6} | {'Bias':>6} | {'Vol':>6} | {'ADX':>6} | {'R:R':>6} | "
          f"{'Chop':>6} | {'TOTAL':>7} | Pass?")
    print(f"  {'─' * 100}")
    for i, trade in enumerate(BLOCKED_TRADES, 1):
        score, bd = soft_chop_score(trade)
        passing = "✅" if score >= 0.0 else "❌"
        print(f"  {i:>2} | {trade.timestamp:17} | {bd['confidence']:>+5.2f} | {bd['bias_alignment']:>+5.2f} | "
              f"{bd['volatility']:>+5.2f} | {bd['adx_strength']:>+5.2f} | {bd['risk_reward']:>+5.2f} | "
              f"{bd['chop_penalty']:>+5.2f} | {score:>+6.3f} | {passing}")
    
    # ═══════════════════════════════════════════════════════════════════
    # 5. BEFORE vs AFTER COMPARISON
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  5. BEFORE vs AFTER — Impact Summary")
    print("=" * 70)
    
    # "Before" = current CHOP guard (all blocked → $0)
    print(f"\n  BEFORE (Current CHOP Guard = Binary Block):")
    print(f"    Trades executed:   0")
    print(f"    Net P&L:           $0.00")
    print(f"    Missed profits:    ${sum(r.pnl_usd for r in all_results if r.pnl_usd > 0):.2f}")
    print(f"    Avoided losses:    ${abs(sum(r.pnl_usd for r in all_results if r.pnl_usd < 0)):.2f}")
    
    # Best variant
    best = ranked[0] if ranked else None
    if best:
        print(f"\n  AFTER (Best Variant: {best.label}):")
        print(f"    Trades executed:   {best.total_trades}")
        print(f"    Net P&L:           ${best.net_pnl_usd:+.2f}")
        print(f"    Win Rate:          {best.win_rate:.1f}%")
        print(f"    Expectancy:        ${best.expectancy_usd:+.2f} / trade")
        print(f"    Max Drawdown:      ${best.max_drawdown_usd:.2f}")
        print(f"    Profit Factor:     {best.profit_factor:.2f}")
    
    # Soft filter at threshold 0.0
    soft_passing = [t for t in BLOCKED_TRADES if soft_chop_score(t)[0] >= 0.0]
    if soft_passing:
        soft_results = [simulate_trade(t, prices) for t in soft_passing]
        soft_m = compute_metrics(soft_results, "Soft CHOP Filter (threshold=0.0)")
        print(f"\n  RECOMMENDED (Soft CHOP Filter, threshold=0.0):")
        print(f"    Trades executed:   {soft_m.total_trades}")
        print(f"    Net P&L:           ${soft_m.net_pnl_usd:+.2f}")
        print(f"    Win Rate:          {soft_m.win_rate:.1f}%")
        print(f"    Expectancy:        ${soft_m.expectancy_usd:+.2f} / trade")
        print(f"    Max Drawdown:      ${soft_m.max_drawdown_usd:.2f}")
        print(f"    Profit Factor:     {soft_m.profit_factor:.2f}")
    
    # ═══════════════════════════════════════════════════════════════════
    # 6. RECOMMENDATION & PSEUDOCODE
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  6. RECOMMENDED GUARD REDESIGN")
    print("=" * 70)
    print("""
  CONVERT: Binary CHOP Block → Weighted Soft Filter

  Pseudocode:
  ┌─────────────────────────────────────────────────────────────────┐
  │ def evaluate_chop_regime(signal, hybrid_trend, indicators):    │
  │                                                                │
  │   if hybrid_trend != "CHOP":                                   │
  │     return signal  # No modification needed                    │
  │                                                                │
  │   if is_or_breakout(signal):                                   │
  │     return signal  # OR breakout exempt (existing behavior)    │
  │                                                                │
  │   # Weighted scoring instead of binary block                   │
  │   score = 0.0                                                  │
  │   score += (confidence - 0.35) * 1.5      # ±0.30 max         │
  │   score += bias_alignment(signal, macd, rsi)  # +0.00 to +0.20│
  │   score += volatility_score(atr)          # -0.10 to +0.20    │
  │   score += adx_score(adx)                 # -0.15 to +0.15    │
  │   score += rr_bonus(risk_reward)          # +0.00 to +0.10    │
  │   score -= chop_penalty(adx)              # -0.25 to -0.10    │
  │                                                                │
  │   if score >= 0.0:  # threshold                                │
  │     log("CHOP soft filter: ALLOWING (score={score})")          │
  │     if score < 0.10:                                           │
  │       signal.confidence *= 0.85  # slight dampen for caution   │
  │     return signal                                              │
  │   else:                                                        │
  │     log("CHOP soft filter: BLOCKING (score={score})")          │
  │     signal.action = "HOLD"                                     │
  │     return signal                                              │
  └─────────────────────────────────────────────────────────────────┘

  Key design principles:
  1. HIGH ADX (≥25) should OVERRIDE the CHOP label — ADX measures
     directional strength directly, while CHOP is inferred from EMAs
  2. Strong indicator alignment (MACD + RSI + direction) earns positive
     scores that can overcome the CHOP penalty
  3. Low ATR (<5) keeps the penalty — genuine narrow-range chop
  4. R:R ratio provides a small bonus for well-structured trades
  5. Marginal trades (score 0.0-0.10) get a confidence dampen (0.85x)
     rather than full execution — CAUTIOUS but not BLOCKING
  """)
    
    print("\n" + "=" * 70)
    print("  END OF ANALYSIS")
    print("=" * 70)


if __name__ == "__main__":
    main()
