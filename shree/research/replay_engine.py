"""Real historical option replay engine — the canonical strategy evaluator.

    Signal -> choose contract -> historical NBBO -> execution model
           -> production exit logic -> option P&L -> standardized scorecard

Every future strategy is graded here, not on the SPY barrier. Observation only;
imports no production trading code. ThetaData supplies NBBO; fills are realistic
(buy@ask, sell@bid, commission) — this is a better simulation, not a perfect one.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

RESEARCH_ENGINE_VERSION = "v2.0-real-option-replay"
ET = ZoneInfo("America/New_York")


# ── execution model (Priority-5 hooks; v2.0 defaults) ────────────────────────

@dataclass
class ExecutionModel:
    commission_per_side: float = 0.65   # IBKR options, $/contract
    entry_lag_ms: int = 0               # v3: latency before entry fill
    fill: str = "marketable"            # buy ask / sell bid (spread = cost)
    # v3 hooks (not yet applied): partial_fill, queue_position, limit_fill_prob

    @property
    def commission_rt(self) -> float:
        return 2.0 * self.commission_per_side


class ContractChoice(Enum):
    PRODUCTION = "production"    # exactly what the bot selected
    ATM_0DTE = "atm_0dte"
    ATM_1DTE = "atm_1dte"
    ATM_2DTE = "atm_2dte"
    DELTA_40 = "delta_40"       # ~0.3% OTM proxy (short-dated)
    DELTA_25 = "delta_25"       # ~0.7% OTM proxy
    NEAREST_LIQUID = "nearest_liquid"


@dataclass
class ReplayResult:
    signal_id: int
    choice: str
    ok: bool
    reason: str = ""
    entry_prem: Optional[float] = None
    exit_prem: Optional[float] = None
    net_dollar: Optional[float] = None      # per 1 contract, after commission
    ret_pct: Optional[float] = None
    hold_min: Optional[float] = None
    spread_cost_entry: Optional[float] = None   # (ask-bid) at entry, $/contract
    direction_correct: Optional[bool] = None
    barrier_pnl: Optional[float] = None     # legacy Outcome A for side-by-side
    classification: Optional[str] = None    # 4-way


# ── timestamp helpers ────────────────────────────────────────────────────────

def _utc_to_et(s: str) -> datetime:
    s = s.split(".")[0].rstrip("Z")
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).astimezone(ET)


def _et_key(dt: datetime) -> str:
    return dt.replace(tzinfo=None).isoformat(timespec="milliseconds")


# ── NBBO access (cached per (expiry,strike,right,session)) ───────────────────

class QuoteCache:
    def __init__(self, client):
        self.client = client
        self._cache: Dict[tuple, tuple] = {}

    def get(self, expiry_ymd: str, strike: float, right: str, session_ymd: str):
        key = (expiry_ymd, strike, right, session_ymd)
        if key in self._cache:
            return self._cache[key]
        k = str(int(strike)) if float(strike).is_integer() else str(strike)
        rows = self.client.get_csv("/v3/option/history/quote", {
            "symbol": "SPY", "expiration": expiry_ymd, "strike": k,
            "right": right, "start_date": session_ymd, "end_date": session_ymd})
        quotes = []
        for r in rows:
            ts = (r.get("timestamp") or "").strip().strip('"').replace("Z", "")
            if not ts:
                continue
            quotes.append((ts, r.get("bid"), r.get("ask")))
        quotes.sort(key=lambda x: x[0])
        keys = [q[0] for q in quotes]
        self._cache[key] = (quotes, keys)
        return quotes, keys


def _prevailing(quotes, keys, ts_iso) -> Tuple[Optional[float], Optional[float]]:
    i = bisect.bisect_right(keys, ts_iso) - 1
    while i >= 0:
        try:
            b, a = float(quotes[i][1]), float(quotes[i][2])
        except (TypeError, ValueError):
            i -= 1
            continue
        if b > 0 and a > 0:
            return b, a
        i -= 1
    return None, None


# ── contract resolution ──────────────────────────────────────────────────────

def resolve_contract(client, signal, choice: ContractChoice, entry_et: datetime
                     ) -> Optional[Tuple[str, float, str]]:
    """Return (expiry_ymd, strike, right) for the chosen contract, or None."""
    right = signal["right"]
    spot = signal["spy_price"]
    if choice == ContractChoice.PRODUCTION:
        return signal["expiry_date"], float(signal["strike"]), right

    # ATM / delta choices need the chain for a target expiry
    exps = client.list_expirations("SPY")
    sess = entry_et.date()
    future = sorted(e for e in exps if e >= sess)
    dte_map = {ContractChoice.ATM_0DTE: 0, ContractChoice.ATM_1DTE: 1,
               ContractChoice.ATM_2DTE: 2}
    if choice in dte_map:
        idx = dte_map[choice]
        if idx >= len(future):
            return None
        expiry = future[idx]
    else:
        expiry = future[0] if future else None  # delta/liquid on 0DTE
    if expiry is None:
        return None
    strikes = client.list_strikes("SPY", expiry)
    if not strikes:
        return None
    # moneyness proxy for delta targets (calls OTM above spot, puts below)
    off = {ContractChoice.DELTA_40: 0.003, ContractChoice.DELTA_25: 0.007}.get(choice, 0.0)
    sign = 1 if right == "C" else -1
    target = spot * (1 + sign * off)
    strike = min(strikes, key=lambda k: abs(k - target))
    return expiry.strftime("%Y%m%d"), float(strike), right


# ── the replay ───────────────────────────────────────────────────────────────

def replay_signal(cache: QuoteCache, client, signal, choice: ContractChoice,
                  ex: ExecutionModel) -> ReplayResult:
    sid = signal["id"]
    entry = _utc_to_et(signal["sent_at"])
    exit_ = _utc_to_et(signal["exit_at"])
    if ex.entry_lag_ms:
        entry = entry + timedelta(milliseconds=ex.entry_lag_ms)
    if entry.date() != exit_.date():
        return ReplayResult(sid, choice.value, False, "cross-session (v2 intraday only)")
    con = resolve_contract(client, signal, choice, entry)
    if con is None:
        return ReplayResult(sid, choice.value, False, "no contract")
    expiry_ymd, strike, right = con
    try:
        quotes, keys = cache.get(expiry_ymd, strike, right, entry.strftime("%Y%m%d"))
    except Exception as e:
        return ReplayResult(sid, choice.value, False, f"pull err {e}")
    if not quotes:
        return ReplayResult(sid, choice.value, False, "no quotes")
    eb, ea = _prevailing(quotes, keys, _et_key(entry))
    xb, xa = _prevailing(quotes, keys, _et_key(exit_))
    if not ea or not xb:
        return ReplayResult(sid, choice.value, False, "no NBBO at entry/exit")
    net = (xb - ea) * 100.0 - ex.commission_rt          # buy ask, sell bid
    ret = net / (ea * 100.0)
    dir_ok = signal["spy_price_exit"] > signal["spy_price"] if right == "C" \
        else signal["spy_price_exit"] < signal["spy_price"]
    win = net > 0
    cls = ("TRUE_EDGE" if dir_ok and win else
           "EXEC_CONTRACT" if dir_ok and not win else
           "MECH_DOMINATED" if not dir_ok and win else "SIGNAL_FAIL")
    return ReplayResult(
        sid, choice.value, True, entry_prem=ea, exit_prem=xb, net_dollar=net,
        ret_pct=ret, hold_min=(exit_ - entry).total_seconds() / 60.0,
        spread_cost_entry=(ea - eb), direction_correct=dir_ok,
        barrier_pnl=signal["pnl_pct"], classification=cls)


# ── standardized scorecard (Priority-4) ──────────────────────────────────────

def scorecard(results: List[ReplayResult]) -> Dict:
    ok = [r for r in results if r.ok]
    n = len(ok)
    out = {"engine_version": RESEARCH_ENGINE_VERSION, "n_replayed": n,
           "n_failed": len(results) - n}
    if not n:
        return out
    rets = [r.ret_pct for r in ok]
    nets = [r.net_dollar for r in ok]
    wins = [r for r in ok if r.net_dollar > 0]
    losses = [r for r in ok if r.net_dollar <= 0]
    wr = len(wins) / n
    sum_w = sum(r.net_dollar for r in wins)
    sum_l = -sum(r.net_dollar for r in losses)
    avg_w = (sum_w / len(wins)) if wins else 0.0
    avg_l = (sum_l / len(losses)) if losses else 0.0
    # Kelly with real $ win/loss ratio
    b = (avg_w / avg_l) if avg_l > 0 else 0.0
    kelly = (wr - (1 - wr) / b) if b > 0 else 0.0
    srt = sorted(rets)
    p5 = srt[max(0, int(0.05 * n))]        # 95th-percentile tail loss
    # max drawdown on the cumulative $ curve (trade order)
    cum = 0.0
    peak = 0.0
    mdd = 0.0
    for r in ok:
        cum += r.net_dollar
        peak = max(peak, cum)
        mdd = min(mdd, cum - peak)
    out.update({
        "direction_hit_rate": round(sum(1 for r in ok if r.direction_correct) / n, 3),
        "option_win_rate": round(wr, 3),
        "expectancy_pct": round(sum(rets) / n, 4),
        "expectancy_dollar": round(sum(nets) / n, 2),
        "profit_factor": round(sum_w / sum_l, 3) if sum_l > 0 else None,
        "kelly": round(kelly, 3),
        "tail_loss_p95_pct": round(p5, 3),
        "max_drawdown_dollar": round(mdd, 2),
        "avg_hold_min": round(sum(r.hold_min for r in ok) / n, 1),
        "avg_spread_cost_dollar": round(
            sum((r.spread_cost_entry or 0) * 100 for r in ok) / n, 2),
        "classification": {
            c: sum(1 for r in ok if r.classification == c)
            for c in ("TRUE_EDGE", "EXEC_CONTRACT", "MECH_DOMINATED", "SIGNAL_FAIL")},
        # side-by-side vs legacy barrier
        "legacy_barrier_win_rate": round(
            sum(1 for r in ok if (r.barrier_pnl or 0) > 0) / n, 3),
        "legacy_barrier_expectancy_pct": round(
            sum(r.barrier_pnl or 0 for r in ok) / n, 4),
    })
    return out
