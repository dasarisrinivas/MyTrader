"""PROMOTION FRAMEWORK CONSTITUTION — frozen, versioned, single source of truth.

Changing ANY value here changes history: prior promotion decisions were made
under a different constitution and are no longer comparable. Amendments require
(1) a version bump, (2) migration notes, (3) re-running historical comparison.

Effective 2026-07-27.
"""
from __future__ import annotations

PROMOTION_FRAMEWORK_VERSION = "2.1"
EFFECTIVE_DATE = "2026-07-27"

# ── AMENDMENT LOG ────────────────────────────────────────────────────────────
# v2.1 (2026-07-27) — G8 completeness 1.00 -> 0.99, denominator redefined,
#   itemization made mandatory.
#   Reason: v2.0's G8 was UNSATISFIABLE. Contracts with no market produce no
#   NBBO, so 100% can never be reached and G8 blocked every promotion on a
#   technicality. Evidence: 4/1048 signals unreplayable, ALL from 2026-04-03
#   (market closed — Good Friday; vol=0, oi=0; vendor returned HTTP 472).
#   0 vendor gaps, 0 replay-engine defects.
#   Also fixed: the completeness DENOMINATOR silently excluded 13 signals whose
#   session had <2 price points, so v2.0 reported 99.6% (1031/1035) when true
#   completeness vs all in-window signals was 98.4% (1031/1048).
#   HISTORICAL IMPACT: NONE. Every v2.0 verdict was NO for 4-10 gates besides
#   G8, so no promotion decision changes.
# v2.0 (2026-07-27) — initial frozen constitution.

# ── The gates (G1..G10). Deterministic thresholds; no per-run tuning. ─────────
G1_MIN_SIGNALS = 50            # replayed signals
G2_MIN_EV = 0.0                # EV must exceed this ($/trade)
G3_CI_LOWER_ABOVE = 0.0        # bootstrap 95% CI lower bound must exceed this
G4_MIN_WILSON_WR = 0.40        # Wilson lower bound on win rate
G5_WINDOWS = (10, 30, 90)      # all must be +EV; lifetime non-negative
G6_MAX_BETA_CORR = 0.50        # |corr(P&L, SPY session return)|
G6_MAX_BETA_SLOPE = 5000.0     # |regression slope| ($ per 1.0 SPY return unit)
G7_MIN_RESIDUAL_ALPHA = 0.0    # direction-residualized EV must exceed this
G8_MIN_REPLAY_COMPLETENESS = 0.99  # v2.1. Denominator = ALL signals in the window
# with a resolvable contract; NO signal may be excluded before counting. Every
# unreplayable signal MUST be itemized (id, contract, reason) in the report.
# BIAS NOTE: unreplayable contracts skew illiquid/never-traded, so dropping them
# biases EV UPWARD — itemization exists to keep that skew visible.
G9_DETERMINISTIC_ONLY = True   # capped/sampled runs are EXPLORATORY, never binding
G10_MULTI_LEG = {"BULL_CALL_SPREAD", "BEAR_PUT_SPREAD", "LONG_STRADDLE"}

# ── Evidence coverage: data-driven, not time-driven ──────────────────────────
MIN_COMPLETE_SESSIONS = 30
REQUIRED_REGIMES = ("bull", "bear", "range", "high_vix", "low_vix")
BULL_BEAR_THRESHOLD = 0.003    # |SPY session return| > 0.3% => bull/bear, else range
HIGH_VIX_THRESHOLD = 20.0      # session mean VIX >= 20 => high_vix

# ── Lifecycle stages ─────────────────────────────────────────────────────────
STAGES = ("research", "shadow", "candidate", "pilot", "active",
          "probation", "archived", "ineligible")
# probation: a LIVE strategy whose evidence has deteriorated but which does not
# yet meet archival criteria — sized down / watched, not yet retired.

# ── Metric classification (prevents overweighting exploratory findings) ──────
PROMOTION_METRICS = ("n", "sessions", "ev", "ev_ci95", "wilson_lo",
                     "window_ev", "up_ev", "dn_ev", "beta_corr", "beta_slope",
                     "alpha_residual_ev", "replay_completeness",
                     "regime_coverage")
EXPLORATORY_METRICS = ("alpha_residual_sharpe", "flat_ev", "consec_pos_sessions",
                       "win_rate", "avg_hold_min")


def gate_summary() -> str:
    return (
        f"Promotion Framework v{PROMOTION_FRAMEWORK_VERSION} (eff. {EFFECTIVE_DATE})\n"
        f"  G1 n>={G1_MIN_SIGNALS}\n"
        f"  G2 EV>{G2_MIN_EV}\n"
        f"  G3 95% CI lower>{G3_CI_LOWER_ABOVE}\n"
        f"  G4 Wilson lower>{G4_MIN_WILSON_WR}\n"
        f"  G5 +EV at {G5_WINDOWS} + lifetime non-negative\n"
        f"  G6 |beta corr|<{G6_MAX_BETA_CORR} and |slope|<{G6_MAX_BETA_SLOPE}\n"
        f"  G7 residual alpha>{G7_MIN_RESIDUAL_ALPHA}\n"
        f"  G8 replay completeness=={G8_MIN_REPLAY_COMPLETENESS:.0%}\n"
        f"  G9 deterministic replay only (no sampling)\n"
        f"  G10 multi-leg ineligible until engine v3\n"
        f"  COVERAGE >={MIN_COMPLETE_SESSIONS} sessions spanning {REQUIRED_REGIMES}"
    )
