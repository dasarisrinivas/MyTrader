"""Trading Manager runtime configuration.

All thresholds live here so the bot's config.yaml stays the source of
truth for strategy parameters and the manager owns the *risk* parameters.
Override with env vars prefixed TM_ if needed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _envf(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v is not None else default


def _envi(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v is not None else default


def _envs(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class ManagerConfig:
    # === Capital ===
    # Account equity — the manager treats this as authoritative for sizing.
    # Update via env TM_ACCOUNT_EQUITY when balance changes materially.
    account_equity: float = _envf("TM_ACCOUNT_EQUITY", 4499.77)

    # === Risk caps (% of equity) ===
    risk_per_trade_pref_pct: float = _envf("TM_RISK_PER_TRADE_PCT", 0.01)   # 1%
    risk_per_trade_max_pct: float = _envf("TM_RISK_PER_TRADE_MAX_PCT", 0.02)  # 2%
    daily_loss_hard_pct: float = _envf("TM_DAILY_LOSS_PCT", 0.03)           # 3%
    daily_loss_warn_pct: float = _envf("TM_DAILY_LOSS_WARN_PCT", 0.015)     # 1.5% → cut size

    # === Defined-risk spread cap (MAY 20 2026) ===========================
    # Opt-in, structure-aware override for SPY *debit spreads* only
    # (BULL_CALL_SPREAD / BEAR_PUT_SPREAD). These are defined-risk: their
    # true max-loss is the net debit, which is far below the naked-premium
    # number the flat 2% cap was designed for. When this knob is > 0, a
    # tagged debit spread may risk up to this many dollars instead of the
    # flat risk_per_trade_max_dollars. When 0 (the default) the feature is
    # DISABLED and every structure uses the standard 2% cap — i.e. behaviour
    # is byte-for-byte unchanged until an operator explicitly sets it.
    #
    # Naked longs and straddles (undefined / larger risk) are NEVER affected
    # by this knob — they always use the flat 2% cap.
    #
    # Suggested starting value: ~1 contract of a 5-wide SPY debit spread,
    # e.g. TM_DEFINED_RISK_MAX_DOLLARS=300.
    defined_risk_max_dollars: float = _envf("TM_DEFINED_RISK_MAX_DOLLARS", 0.0)

    # === Trade frequency ===
    max_concurrent_positions: int = _envi("TM_MAX_CONCURRENT", 2)
    max_trades_per_day: int = _envi("TM_MAX_TRADES_DAY", 4)

    # === Streak rules ===
    # 2-loss streak: SOFT pause — keep trading at reduced size with tighter gates
    consec_losses_pause: int = _envi("TM_CONSEC_LOSSES_PAUSE", 2)
    # 5-loss streak: HARD pause — reject all entries until next session rolls
    consec_losses_hard_pause: int = _envi("TM_CONSEC_LOSSES_HARD", 5)
    consec_wins_size_up: int = _envi("TM_CONSEC_WINS_SIZE_UP", 3)
    # When in soft pause, raise these gates above their static defaults
    soft_pause_min_confidence: float = _envf("TM_SOFT_PAUSE_MIN_CONF", 0.70)
    soft_pause_min_rr: float = _envf("TM_SOFT_PAUSE_MIN_RR", 1.5)

    # === Quality gates ===
    min_rr_ratio: float = _envf("TM_MIN_RR", 1.2)
    min_adx_for_trend_strategy: float = _envf("TM_MIN_ADX_TREND", 18.0)
    min_confidence: float = _envf("TM_MIN_CONFIDENCE", 0.50)

    # === Lookback windows ===
    last_n_for_pattern_check: int = _envi("TM_LAST_N_PATTERN", 5)
    pattern_loss_threshold: int = _envi("TM_PATTERN_LOSS_THRESHOLD", 3)

    # === Files (relative to repo root) ===
    decisions_jsonl: str = _envs("TM_DECISIONS_FILE", "logs/decisions.jsonl")
    spy_signals_jsonl: str = _envs("TM_SPY_SIGNALS_FILE", "logs/spy_signals.jsonl")
    manager_jsonl: str = _envs("TM_MANAGER_FILE", "logs/manager_decisions.jsonl")
    state_file: str = _envs("TM_STATE_FILE", "logs/trading_manager_state.json")
    log_file: str = _envs("TM_LOG_FILE", "logs/trading_manager.log")
    pid_file: str = _envs("TM_PID_FILE", "logs/trading_manager.pid")
    bot_pid_file: str = _envs("TM_BOT_PID", "logs/bot.pid")
    spy_pid_file: str = _envs("TM_SPY_PID", "logs/spy_options.pid")
    orders_db: str = _envs("TM_ORDERS_DB", "data/orders.db")
    trade_journal_db: str = _envs("TM_JOURNAL_DB", "data/trade_journal.db")
    spy_signals_db: str = _envs("TM_SPY_DB", "data/spy_options_signals.db")
    learning_db: str = _envs("TM_LEARNING_DB", "data/learning.db")
    # Refresh empirical bucket stats every N seconds (also rebuilds from any
    # newly-closed trades). 300 = every 5 minutes — cheap.
    learning_refresh_seconds: int = _envi("TM_LEARNING_REFRESH_S", 300)

    # === Loop ===
    poll_interval_seconds: float = _envf("TM_POLL_SECONDS", 1.0)
    # Health check cadence (multi-day metrics — cheap, not time-critical)
    health_check_interval_seconds: int = _envi("TM_HEALTH_CHECK_S", 600)  # 10 min

    # Health history cutoff — ISO date string (YYYY-MM-DD or full ISO timestamp).
    # When set, compute_metrics only reads executions *on or after* this date,
    # preventing old-book / pre-config-change / paper trades from poisoning the
    # rolling health window (cold-start deadlock #2). Default empty = no cutoff,
    # original behavior unchanged.
    # Example: TM_HEALTH_SINCE=2026-05-27
    health_since: str = _envs("TM_HEALTH_SINCE", "")

    # Unlock CLI drops a marker here; manager picks it up
    unlock_marker_file: str = _envs("TM_UNLOCK_MARKER", "logs/trading_manager_unlock.marker")

    # === MES contract specs (used for $ risk math from points) ===
    mes_multiplier: float = _envf("TM_MES_MULT", 5.0)

    # === Operational ===
    dry_run: bool = bool(int(_envs("TM_DRY_RUN", "0")))  # if 1, never kills bots

    # Derived helpers
    @property
    def risk_per_trade_pref_dollars(self) -> float:
        return round(self.account_equity * self.risk_per_trade_pref_pct, 2)

    @property
    def risk_per_trade_max_dollars(self) -> float:
        return round(self.account_equity * self.risk_per_trade_max_pct, 2)

    @property
    def daily_loss_hard_dollars(self) -> float:
        return round(self.account_equity * self.daily_loss_hard_pct, 2)

    @property
    def daily_loss_warn_dollars(self) -> float:
        return round(self.account_equity * self.daily_loss_warn_pct, 2)

    # Defined-risk debit spreads (DESCRIBED ABOVE). Naked longs and straddles
    # always get the flat 2% cap. A tagged debit spread gets the larger of the
    # flat cap and the opt-in spread cap — so enabling the knob can only ever
    # *loosen* the limit for spreads, never tighten it below the 2% baseline,
    # and a value of 0.0 leaves the flat cap fully in force.
    _DEFINED_RISK_STRUCTURES = ("BULL_CALL_SPREAD", "BEAR_PUT_SPREAD")

    def cap_for_structure(self, structure: str) -> float:
        """Return the per-trade $ risk cap that applies to ``structure``.

        Defaults to the flat 2% cap. Only tagged debit spreads, and only when
        ``defined_risk_max_dollars`` is set > 0, are allowed a higher ceiling.
        """
        flat = self.risk_per_trade_max_dollars
        s = (structure or "").upper()
        if s in self._DEFINED_RISK_STRUCTURES and self.defined_risk_max_dollars > 0:
            return max(flat, self.defined_risk_max_dollars)
        return flat


CONFIG = ManagerConfig()
