"""Cold-start expectancy priors + shadow computation for the MES R:R gate.

SHADOW-ONLY. Nothing in this module gates a live decision. It is consumed
exclusively by decision_log.append_decision to attach a `shadow_expectancy`
block to each logged MES decision (Phase 0 of docs/expectancy_gate_proposal.md).
The live decision continues to come from rules.py unchanged.

------------------------------------------------------------------------------
WR priors — provenance (versioned, frozen 2026-05-22)
------------------------------------------------------------------------------
Source: reports/bt_baseline/backtest_MES_2025-02-01_2026-01-31_20260309_111038_trades.csv
  (full-year MES 15m backtest at the strategy's real exit logic;
   exit mix = 20 take_profit / 19 stop_loss / 14 flatten).
WR = realized_pnl > 0, pooled across regime, split by signal_type (== reason's
first token, matching signal_watcher.Signal.signal_type).

  signal_type          n   wins    WR
  EMA21_PB_LONG        22    9    40.91%
  TREND_CONT_LONG      10    7    70.00%
  OR_BREAK_LONG         3    1    33.33%   <- thin, below MIN_PRIOR_N
  TREND_CONT_SHORT     11    4    36.36%
  EMA21_PB_SHORT        7    3    42.86%   <- thin, below MIN_PRIOR_N

These priors are SMALL-SAMPLE and pooled across regime. They are adequate for a
shadow-mode prior whose only job is to be logged and audited; they are NOT yet
trustworthy enough to gate live trades (proposal §6.3 / §7). The shadow record
logs n_prior and prior_source so trustworthiness can be judged from the data.
------------------------------------------------------------------------------
"""
from __future__ import annotations

from typing import Optional

# Frozen for the Phase-0 trial. Any change after shadow data is observed
# requires bumping PRIORS_VERSION and a new pre-registration (proposal §6).
PRIORS_VERSION = "2026-05-22.bt_baseline"

# Pre-registered parameters (docs/expectancy_gate_proposal.md §3b).
MIN_E = 0.10            # require expectancy >= +0.10R (risk leg fixed at 1xATR)
DEFAULT_WR = 0.45       # conservative fallback when no/insufficient prior
MIN_PRIOR_N = 10        # below this, treat the backtest WR as not seedable -> DEFAULT_WR

# (wr, n) per signal_type, frozen from the full-year MES baseline backtest.
_WR_PRIORS = {
    "EMA21_PB_LONG":    (0.4091, 22),
    "TREND_CONT_LONG":  (0.7000, 10),
    "OR_BREAK_LONG":    (0.3333, 3),
    "TREND_CONT_SHORT": (0.3636, 11),
    "EMA21_PB_SHORT":   (0.4286, 7),
}


def required_rr_for(wr: float, min_e: float = MIN_E) -> float:
    """Breakeven-plus-margin R:R for a given prior win rate.

        E_R(rr) = wr*rr - (1-wr)*1   (risk leg fixed at 1xATR)
        require E_R >= min_e
        => required_rr = (1 - wr + min_e) / wr
    """
    if wr <= 0:
        return float("inf")
    return (1.0 - wr + min_e) / wr


def expected_r(wr: float, rr: float) -> float:
    """Expectancy in R for a setup with win rate `wr` and reward:risk `rr`."""
    return wr * rr - (1.0 - wr)


def shadow_expectancy(signal_type: str, rr: float) -> Optional[dict]:
    """Compute the would-be cold-start expectancy decision for an MES signal.

    Pure/side-effect-free. Returns a dict suitable for logging alongside the
    live decision, or None if rr is unusable. Does NOT decide anything live.

    prior_source:
      "backtest"      - seedable backtest WR (n >= MIN_PRIOR_N) drove the floor
      "default_thin"  - backtest WR exists but n < MIN_PRIOR_N -> DEFAULT_WR used
      "default_absent"- no backtest WR for this signal_type -> DEFAULT_WR used
    """
    try:
        rr = float(rr)
    except (TypeError, ValueError):
        return None
    if rr <= 0:
        return None

    bt = _WR_PRIORS.get(signal_type)
    if bt is None:
        wr_prior, n_prior, source, bt_wr = DEFAULT_WR, 0, "default_absent", None
    else:
        bt_wr, n_prior = bt
        if n_prior >= MIN_PRIOR_N:
            wr_prior, source = bt_wr, "backtest"
        else:
            wr_prior, source = DEFAULT_WR, "default_thin"

    req = required_rr_for(wr_prior)
    return {
        "version": PRIORS_VERSION,
        "wr_prior": round(wr_prior, 4),
        "n_prior": n_prior,
        "prior_source": source,
        "backtest_wr": (round(bt_wr, 4) if bt_wr is not None else None),
        "min_e": MIN_E,
        "required_rr": round(req, 4),
        "e_r": round(expected_r(wr_prior, rr), 4),
        "would_pass": bool(rr >= req),
        # The live static floor for side-by-side comparison in the shadow log.
        "static_floor": 2.0,
    }
