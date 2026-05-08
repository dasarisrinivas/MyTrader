"""The decision framework — the 5 questions, plus risk/sizing math.

Pure functions, no I/O. Takes a Signal + ManagerState + ManagerConfig +
recent trade outcomes, returns a Decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from .config import ManagerConfig
from .executions_reader import TradeOutcome
from .learning import adaptive_lookup, AdaptiveThresholds
from .signal_watcher import Signal
from .spy_signal_watcher import SpySignal
from .state import (
    ManagerState,
    POSTURE_DEFENSIVE,
    POSTURE_KILLED,
    POSTURE_LOCKED,
    POSTURE_NORMAL,
    POSTURE_PROBATION,
    POSTURE_SIT_OUT,
)


@dataclass
class Decision:
    decision: str         # APPROVE | REJECT | MODIFY
    confidence: int       # 0-100
    position_size: str    # small | normal | aggressive
    reasoning: str
    risk_notes: str
    override: bool = False
    checks: List[Tuple[str, bool, str]] = field(default_factory=list)  # (name, passed, note)
    posture_after: str = POSTURE_NORMAL


def _market_regime(sig: Signal) -> str:
    """Trending / Choppy / Mixed based on ADX."""
    if sig.adx >= 22:
        return "TRENDING"
    if sig.adx <= 15:
        return "CHOPPY"
    return "MIXED"


def _strategy_fits_regime(sig: Signal, regime: str) -> bool:
    """Trend strategies need trend; OR breakouts can work in mixed/choppy starts."""
    st = sig.signal_type.upper()
    if "TREND_CONT" in st or "EMA21_PB" in st:
        return regime in ("TRENDING", "MIXED")
    if "OR_BREAK" in st:
        return regime in ("TRENDING", "MIXED", "CHOPPY")  # ORB is regime-tolerant
    if "PB_FADE" in st or "REVERSAL" in st:
        return regime in ("CHOPPY", "MIXED")
    # Unknown — be conservative, only allow in clearly trending
    return regime == "TRENDING"


def _similar_recent_lost(
    sig: Signal,
    recent: List[TradeOutcome],
    cfg: ManagerConfig,
) -> Tuple[bool, str]:
    """We don't have signal_type on each TradeOutcome (executions table is
    direction-agnostic). Use last-N as a proxy: if the last N closed trades
    have >= threshold losers, the regime isn't paying.
    """
    n = cfg.last_n_for_pattern_check
    sample = recent[:n]
    if len(sample) < n:
        return False, f"insufficient history ({len(sample)}/{n})"
    losers = sum(1 for t in sample if t.is_loss)
    if losers >= cfg.pattern_loss_threshold:
        return True, f"{losers}/{n} of last trades were losses — pattern unfavorable"
    return False, f"{losers}/{n} losers in last {n} — within tolerance"


def _position_size_label(sig: Signal, state: ManagerState, cfg: ManagerConfig) -> str:
    """Map W/L streak + posture to size bucket."""
    if state.posture == POSTURE_DEFENSIVE:
        return "small"
    if state.consec_wins >= cfg.consec_wins_size_up:
        return "aggressive"
    if state.consec_losses >= 1:
        return "small"
    return "normal"


def evaluate(
    sig: Signal,
    state: ManagerState,
    recent: List[TradeOutcome],
    cfg: ManagerConfig,
) -> Decision:
    """Run the 5-question framework against a fresh signal."""
    checks: List[Tuple[str, bool, str]] = []

    # Hard kill switches first — these dominate everything else.
    if state.posture == POSTURE_KILLED:
        return Decision(
            decision="REJECT",
            confidence=100,
            position_size="small",
            reasoning="Bots are in KILLED posture (daily-loss hard stop hit). No new entries until next session.",
            risk_notes="Manual reset required, or wait for daily session roll.",
            override=True,
            checks=[("kill_switch_active", False, "POSTURE=KILLED")],
            posture_after=POSTURE_KILLED,
        )

    # Layer 1.5: LOCKED — multi-day strategy failure detected by health module.
    # Manual unlock required (writes logs/trading_manager_unlock.marker).
    if state.posture == POSTURE_LOCKED:
        return Decision(
            decision="REJECT",
            confidence=100,
            position_size="small",
            reasoning=(
                "🔒 LOCKED — multi-day strategy/regime failure detected. "
                f"Reason: {state.lock_reason or '(see health log)'}. "
                "Run `python -m shree.trading_manager.unlock` to enter probation."
            ),
            risk_notes=f"Locked since {state.lock_since}. Triggers: {state.health_triggers}",
            override=True,
            checks=[("posture_locked", False, str(state.health_triggers))],
            posture_after=POSTURE_LOCKED,
        )

    # Layer 1.5: PROBATION — exactly 1 small probe trade allowed. Must clear
    # all the normal gates AND meet an extreme-confidence bar. The probation
    # outcome handler in manager.py decides next state on the closed trade.
    if state.posture == POSTURE_PROBATION:
        if state.probation_trade_count >= 1:
            return Decision(
                decision="REJECT",
                confidence=95,
                position_size="small",
                reasoning=(
                    "Probation already used (1 probe trade pending evaluation). "
                    "Wait for outcome before next entry."
                ),
                risk_notes="Probation: only 1 probe allowed at a time.",
                override=True,
                checks=[("probation_already_used", False,
                         f"count={state.probation_trade_count}")],
                posture_after=POSTURE_PROBATION,
            )
        # Allow only signals with >= 0.85 confidence to be the probe
        sig_conf = float(sig.raw.get("confidence") or 0.0) if hasattr(sig, "raw") else 0.0
        if sig_conf > 0 and sig_conf < 0.85:
            return Decision(
                decision="REJECT",
                confidence=85,
                position_size="small",
                reasoning=(
                    f"PROBATION: probe trade requires >= 0.85 confidence "
                    f"(have {sig_conf:.2f}). Wait for an A++ setup."
                ),
                risk_notes="Probation: a single losing probe sends us back to LOCKED.",
                override=True,
                checks=[("probation_high_conf_only", False, f"conf={sig_conf:.2f}")],
                posture_after=POSTURE_PROBATION,
            )
        # Otherwise fall through — the signal will be evaluated by the rest of
        # the framework, but with size forced to "small" and posture remaining
        # PROBATION until the trade closes.

    if state.realized_pnl_today <= -cfg.daily_loss_hard_dollars:
        return Decision(
            decision="REJECT",
            confidence=100,
            position_size="small",
            reasoning=(
                f"Daily loss limit BREACHED: realized {state.realized_pnl_today:+.2f} "
                f"<= -{cfg.daily_loss_hard_dollars:.2f} (3% of equity). HARD STOP."
            ),
            risk_notes="Forcing posture=KILLED. Bots will be paused.",
            override=True,
            checks=[("daily_loss_limit", False, f"{state.realized_pnl_today:+.2f}")],
            posture_after=POSTURE_KILLED,
        )

    if state.trades_today >= cfg.max_trades_per_day:
        return Decision(
            decision="REJECT",
            confidence=95,
            position_size="small",
            reasoning=(
                f"Trade count {state.trades_today} >= cap {cfg.max_trades_per_day}. "
                "Mandate: no overtrading."
            ),
            risk_notes="Sit out the rest of the session.",
            override=True,
            checks=[("max_trades_per_day", False, f"{state.trades_today}/{cfg.max_trades_per_day}")],
            posture_after=POSTURE_DEFENSIVE,
        )

    # ── STREAK LADDER (MAY 8 2026 update) ─────────────────────────────────
    # Three-tier response based on consecutive loss count:
    #   < 2 losses:                 normal operation
    #   2..(hard_pause-1) losses:   SOFT pause — keep trading but at reduced
    #                               size with tighter confidence + R:R gates
    #   >= hard_pause losses:       HARD halt — reject all entries
    # Streak resets at session roll OR on a single winning trade.
    if state.consec_losses >= cfg.consec_losses_hard_pause:
        return Decision(
            decision="REJECT",
            confidence=95,
            position_size="small",
            reasoning=(
                f"{state.consec_losses} consecutive losses — HARD pause "
                f"(>= {cfg.consec_losses_hard_pause}). Halting until session "
                "rolls or a winning trade breaks the streak."
            ),
            risk_notes="Capital preservation: real losing run detected.",
            override=True,
            checks=[("consec_losses_hard", False,
                     f"{state.consec_losses}/{cfg.consec_losses_hard_pause}")],
            posture_after=POSTURE_DEFENSIVE,
        )

    # SOFT pause: don't return REJECT here — instead, raise the bar so only
    # the cleanest setups make it through, and force size to "small". The
    # downstream gates handle the actual rejection if the signal isn't good
    # enough. This gives us a path out of the streak via a winning trade.
    in_soft_pause = state.consec_losses >= cfg.consec_losses_pause
    if in_soft_pause:
        checks.append((
            "streak_soft_pause",
            True,
            f"{state.consec_losses} losses → tighter gates "
            f"(min_conf {cfg.soft_pause_min_confidence:.2f}, min_rr {cfg.soft_pause_min_rr:.1f})",
        ))

    # === The 5 questions ===

    # Q1: trending or choppy?
    regime = _market_regime(sig)
    q1_pass = regime != "CHOPPY"
    checks.append((
        "Q1_regime",
        q1_pass,
        f"ADX={sig.adx:.1f} → {regime}",
    ))

    # Q2: does this strategy fit?
    q2_pass = _strategy_fits_regime(sig, regime)
    checks.append((
        "Q2_strategy_fit",
        q2_pass,
        f"{sig.signal_type} in {regime}",
    ))

    # Q3: are we overtrading?
    # Soft trigger at 75% of cap, hard at cap (handled above).
    soft_cap = max(1, int(cfg.max_trades_per_day * 0.75))
    q3_pass = state.trades_today < soft_cap
    checks.append((
        "Q3_overtrading",
        q3_pass,
        f"{state.trades_today}/{cfg.max_trades_per_day} (soft cap {soft_cap})",
    ))

    # Q4: did similar trades recently lose?
    similar_lost, q4_note = _similar_recent_lost(sig, recent, cfg)
    q4_pass = not similar_lost
    checks.append(("Q4_recent_pattern", q4_pass, q4_note))

    # ── ADAPTIVE LAYER (Layer 1 intelligence) ────────────────────────────
    # Consult the empirical learning DB for this signal's bucket. If the
    # bucket has been bleeding historically, AUTO-SUPPRESS. If it pays
    # asymmetrically, relax the R:R requirement. Falls through silently
    # when no usable data exists.
    adaptive = adaptive_lookup(
        cfg.learning_db,
        signal_type=sig.signal_type,
        bot="mes",
        regime=_market_regime(sig),
        ts_iso=sig.ts,
        vix=None,  # MES decisions don't carry VIX
        base_min_confidence=cfg.min_confidence,
        base_min_rr=cfg.min_rr_ratio,
    )
    if adaptive is not None:
        checks.append((
            "adaptive_bucket",
            not adaptive.auto_suppress,
            adaptive.rationale,
        ))
        if adaptive.auto_suppress:
            return Decision(
                decision="REJECT",
                confidence=90,
                position_size="small",
                reasoning=(
                    f"LEARNED REJECTION: this bucket "
                    f"({sig.signal_type}/{_market_regime(sig)}) has "
                    f"{adaptive.win_rate*100:.0f}% WR over {adaptive.n_trades} "
                    f"historical trades with negative expectancy. The math doesn't pay."
                ),
                risk_notes=adaptive.rationale,
                override=True,
                checks=checks,
                posture_after=state.posture,
            )
        effective_min_rr = adaptive.min_rr_required
    else:
        checks.append(("adaptive_bucket", True, "no empirical data — using static rules"))
        effective_min_rr = cfg.min_rr_ratio

    # SOFT-PAUSE override: raise R:R floor when in a 2-4 loss streak.
    # Take the max of any active source (adaptive, soft pause, static).
    if in_soft_pause:
        effective_min_rr = max(effective_min_rr, cfg.soft_pause_min_rr)

    # Q5: R:R >= effective minimum (adaptive, soft-pause, or static)?
    rr = sig.rr
    q5_pass = rr >= effective_min_rr
    rr_tag = ""
    if in_soft_pause and effective_min_rr == cfg.soft_pause_min_rr:
        rr_tag = " [SOFT_PAUSE]"
    elif adaptive and effective_min_rr != cfg.min_rr_ratio:
        rr_tag = " [ADAPTIVE]"
    checks.append((
        "Q5_rr_ratio",
        q5_pass,
        f"R:R={rr:.2f} (need >= {effective_min_rr:.1f}{rr_tag})",
    ))

    # Risk in $ — does this trade fit our per-trade cap?
    risk_dollars = sig.risk_points * cfg.mes_multiplier
    risk_pct = (risk_dollars / cfg.account_equity) * 100.0 if cfg.account_equity else 0.0
    risk_within_cap = risk_dollars <= cfg.risk_per_trade_max_dollars
    checks.append((
        "risk_within_cap",
        risk_within_cap,
        f"${risk_dollars:.2f} ({risk_pct:.2f}% of equity), cap ${cfg.risk_per_trade_max_dollars:.2f}",
    ))

    # Daily-loss warning band — cut size if we're inside it
    in_warn_band = state.realized_pnl_today <= -cfg.daily_loss_warn_dollars
    if in_warn_band:
        checks.append((
            "daily_loss_warning_band",
            False,
            f"PnL {state.realized_pnl_today:+.2f} <= -{cfg.daily_loss_warn_dollars:.2f} (warn) — cut size",
        ))

    # Aggregate
    fails = [c for c in checks if not c[1]]
    n_q_fails = sum(
        1 for c in checks
        if c[0].startswith("Q") and not c[1]
    )

    # Decision logic
    posture_after = state.posture

    if not risk_within_cap:
        # Hard reject — risk per trade is non-negotiable
        return Decision(
            decision="REJECT",
            confidence=95,
            position_size="small",
            reasoning=(
                f"Per-trade risk ${risk_dollars:.2f} exceeds cap "
                f"${cfg.risk_per_trade_max_dollars:.2f} (2% of equity). Stop is too wide."
            ),
            risk_notes=f"Stop distance {sig.risk_points:.2f} pts. Need <= "
                       f"{cfg.risk_per_trade_max_dollars/cfg.mes_multiplier:.2f} pts.",
            override=True,
            checks=checks,
            posture_after=posture_after,
        )

    if not q5_pass:
        # Hard reject — R:R is non-negotiable for new entries
        rr_source = (
            "soft-pause" if (in_soft_pause and effective_min_rr == cfg.soft_pause_min_rr)
            else "adaptive" if (adaptive and effective_min_rr != cfg.min_rr_ratio)
            else "static"
        )
        return Decision(
            decision="REJECT",
            confidence=85,
            position_size="small",
            reasoning=(
                f"R:R {rr:.2f} < required {effective_min_rr:.1f} ({rr_source}). "
                f"Risk {sig.risk_points:.2f} pts vs reward {sig.reward_points:.2f} pts. "
                "Mandate requires asymmetric payoff."
            ),
            risk_notes=(
                "Streak in soft pause — only A+ setups (R:R >= 2.5) clear here."
                if in_soft_pause
                else "If strategy keeps producing sub-2:1 setups, the ATR-adaptive "
                     "sizing logic needs review (R:R compression bug)."
            ),
            override=True,
            checks=checks,
            posture_after=POSTURE_DEFENSIVE if in_soft_pause else posture_after,
        )

    if n_q_fails >= 2:
        # Multi-failure: REJECT
        fail_names = ", ".join(c[0] for c in checks if c[0].startswith("Q") and not c[1])
        return Decision(
            decision="REJECT",
            confidence=80,
            position_size="small",
            reasoning=(
                f"{n_q_fails} framework checks failed: {fail_names}. "
                "Mandate: ANY negative answer → REJECT or MODIFY. Multiple = REJECT."
            ),
            risk_notes="Wait for regime/setup to align.",
            override=False,
            checks=checks,
            posture_after=posture_after,
        )

    if n_q_fails == 1 or in_warn_band or in_soft_pause:
        # Single issue, warning band, OR soft-pause streak: MODIFY (reduce size).
        # Soft-pause adds an explicit confidence-floor gate at 0.70 against the
        # bot's own signal confidence (when present in metadata).
        sig_conf = float(sig.raw.get("confidence") or 0.0) if hasattr(sig, "raw") else 0.0
        if in_soft_pause and sig_conf > 0 and sig_conf < cfg.soft_pause_min_confidence:
            return Decision(
                decision="REJECT",
                confidence=80,
                position_size="small",
                reasoning=(
                    f"Soft pause active ({state.consec_losses} losses) AND "
                    f"signal confidence {sig_conf:.2f} < soft-pause floor "
                    f"{cfg.soft_pause_min_confidence:.2f}. Need cleaner setup."
                ),
                risk_notes="Streak resets on a winner OR session roll.",
                override=False,
                checks=checks,
                posture_after=POSTURE_DEFENSIVE,
            )
        reasons = []
        if n_q_fails == 1:
            reasons.append("one framework check failed")
        if in_warn_band:
            reasons.append("daily-loss warning band entered")
        if in_soft_pause:
            reasons.append(f"{state.consec_losses}-loss streak (soft pause)")
        return Decision(
            decision="MODIFY",
            confidence=60 if in_soft_pause else 65,
            position_size="small",
            reasoning=(
                f"Approving at reduced size — {', '.join(reasons)}. "
                "Mandate: MODIFY rather than REJECT when concerns are bounded."
            ),
            risk_notes=(
                "Cut to half of normal size. Tight management on this one."
                + (" Soft pause: streak breaks on a winning trade." if in_soft_pause else "")
            ),
            override=False,
            checks=checks,
            posture_after=POSTURE_DEFENSIVE if in_soft_pause else posture_after,
        )

    # Clean signal — but if we're in soft pause, force small size + DEFENSIVE
    size = _position_size_label(sig, state, cfg)
    if in_soft_pause:
        size = "small"
        posture_after = POSTURE_DEFENSIVE
    return Decision(
        decision="APPROVE",
        confidence=85,
        position_size=size,
        reasoning=(
            f"All 5 framework checks pass. Regime={regime}, ADX={sig.adx:.1f}, "
            f"R:R={rr:.2f}, risk ${risk_dollars:.2f} ({risk_pct:.2f}% equity). "
            f"Streak: {state.consec_wins}W / {state.consec_losses}L → size '{size}'."
            + (" [SOFT_PAUSE: small size enforced]" if in_soft_pause else "")
        ),
        risk_notes=(
            "Standard tail risk: stop can slip on news/illiquidity. "
            f"Daily PnL buffer remaining: ${cfg.daily_loss_hard_dollars + state.realized_pnl_today:.2f}."
        ),
        override=False,
        checks=checks,
        posture_after=posture_after,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SPY OPTIONS — separate evaluator
#
# Options-specific gates that the futures rules don't cover:
#   • Premium-based $ risk (50% premium loss heuristic) instead of point risk
#   • Spread % filter (illiquid contracts kill PnL on entry+exit)
#   • DTE bounds (no 0DTE gamma-bombs, no >45DTE theta drag)
#   • Regime-fit per signal type (PC_RATIO needs trending tape, ORB needs early-AM)
#   • Confidence floor (cfg.min_confidence)
#
# Hard kill-switches (KILLED posture, daily loss breach, max trades) are shared
# with the MES path — same state, same caps, same pool.
# ─────────────────────────────────────────────────────────────────────────────


def _spy_signal_fits_regime(sig: SpySignal) -> Tuple[bool, str]:
    """Return (passes, note). Each SPY signal type has a regime in which it
    works and one in which it bleeds.
    """
    st = (sig.signal_type or "").upper()
    regime = (sig.regime or "").upper()

    # PC_RATIO_EXTREME is a contrarian fade — needs CHOP/RANGE_BOUND tape
    # where the extreme gets reverted, not TREND tape where it just keeps going.
    if "PC_RATIO" in st:
        ok = regime in ("RANGE_BOUND", "CHOP", "MIXED", "")
        return ok, f"{st} in {regime or 'unknown'}"

    # ORB_BREAKOUT needs a trending or breakout regime to follow through.
    if "ORB" in st or "ORB_BREAKOUT" in st:
        ok = regime in ("TREND_UP", "TREND_DOWN", "BREAKOUT", "MIXED", "")
        return ok, f"{st} in {regime or 'unknown'}"

    # TREND_CONTINUATION is rules_v2's pullback-rejection — by definition it
    # only fires in TREND_UP / TREND_DOWN. Trust the regime classification.
    if "TREND_CONT" in st:
        ok = regime in ("TREND_UP", "TREND_DOWN")
        return ok, f"{st} in {regime or 'unknown'}"

    # Sweep / spread / straddle / IV signals — accept any regime, the engine
    # has already filtered for confidence.
    return True, f"{st} regime-agnostic"


def _spy_position_size_label(state: ManagerState, cfg: ManagerConfig) -> str:
    if state.posture == POSTURE_DEFENSIVE:
        return "small"
    if state.consec_wins >= cfg.consec_wins_size_up:
        return "aggressive"
    if state.consec_losses >= 1:
        return "small"
    return "normal"


def evaluate_spy(
    sig: SpySignal,
    state: ManagerState,
    recent: List[TradeOutcome],
    cfg: ManagerConfig,
) -> Decision:
    """Run the 5-question framework against a SPY options signal.

    Hard kill-switches are checked first (shared with MES path), then options-
    specific quality gates.
    """
    checks: List[Tuple[str, bool, str]] = []

    # ── Shared hard kill-switches ─────────────────────────────────────────
    if state.posture == POSTURE_KILLED:
        return Decision(
            decision="REJECT",
            confidence=100,
            position_size="small",
            reasoning="Bots are in KILLED posture (daily-loss hard stop hit). No new SPY entries.",
            risk_notes="Manual reset required, or wait for daily session roll.",
            override=True,
            checks=[("kill_switch_active", False, "POSTURE=KILLED")],
            posture_after=POSTURE_KILLED,
        )

    if state.realized_pnl_today <= -cfg.daily_loss_hard_dollars:
        return Decision(
            decision="REJECT",
            confidence=100,
            position_size="small",
            reasoning=(
                f"Daily loss limit BREACHED: realized {state.realized_pnl_today:+.2f} "
                f"<= -{cfg.daily_loss_hard_dollars:.2f} (3% of equity). HARD STOP."
            ),
            risk_notes="Forcing posture=KILLED. Bots will be paused.",
            override=True,
            checks=[("daily_loss_limit", False, f"{state.realized_pnl_today:+.2f}")],
            posture_after=POSTURE_KILLED,
        )

    if state.trades_today >= cfg.max_trades_per_day:
        return Decision(
            decision="REJECT",
            confidence=95,
            position_size="small",
            reasoning=(
                f"Trade count {state.trades_today} >= cap {cfg.max_trades_per_day}. "
                "Mandate: no overtrading. SPY options counts toward the same pool."
            ),
            risk_notes="Sit out the rest of the session.",
            override=True,
            checks=[("max_trades_per_day", False, f"{state.trades_today}/{cfg.max_trades_per_day}")],
            posture_after=POSTURE_DEFENSIVE,
        )

    # ── STREAK LADDER (MAY 8 2026 — same tiering as MES path) ────────────
    if state.consec_losses >= cfg.consec_losses_hard_pause:
        return Decision(
            decision="REJECT",
            confidence=95,
            position_size="small",
            reasoning=(
                f"{state.consec_losses} consecutive losses — HARD pause "
                f"(>= {cfg.consec_losses_hard_pause}). Halting all SPY entries until "
                "session rolls or a winning trade breaks the streak."
            ),
            risk_notes="Capital preservation: real losing run detected.",
            override=True,
            checks=[("consec_losses_hard", False,
                     f"{state.consec_losses}/{cfg.consec_losses_hard_pause}")],
            posture_after=POSTURE_DEFENSIVE,
        )

    in_soft_pause = state.consec_losses >= cfg.consec_losses_pause
    if in_soft_pause:
        checks.append((
            "streak_soft_pause",
            True,
            f"{state.consec_losses} losses → SPY soft pause "
            f"(min_conf {cfg.soft_pause_min_confidence:.2f})",
        ))

    # ── ADAPTIVE LAYER (Layer 1 intelligence) ────────────────────────────
    # Same idea as the MES path. The SPY bot writes VIX to each signal so
    # we get a richer bucket key.
    adaptive = adaptive_lookup(
        cfg.learning_db,
        signal_type=sig.signal_type,
        bot="spy_options",
        regime=sig.regime,
        ts_iso=sig.ts,
        vix=sig.vix if sig.vix > 0 else None,
        base_min_confidence=cfg.min_confidence,
        base_min_rr=cfg.min_rr_ratio,
    )
    if adaptive is not None:
        checks.append((
            "adaptive_bucket",
            not adaptive.auto_suppress,
            adaptive.rationale,
        ))
        if adaptive.auto_suppress:
            return Decision(
                decision="REJECT",
                confidence=90,
                position_size="small",
                reasoning=(
                    f"LEARNED REJECTION: SPY bucket "
                    f"({sig.signal_type}/{sig.regime}) has "
                    f"{adaptive.win_rate*100:.0f}% WR over {adaptive.n_trades} "
                    f"historical trades with negative expectancy."
                ),
                risk_notes=adaptive.rationale,
                override=True,
                checks=checks,
                posture_after=state.posture,
            )
        effective_min_confidence = adaptive.confidence_floor
    else:
        checks.append(("adaptive_bucket", True, "no empirical data — using static rules"))
        effective_min_confidence = cfg.min_confidence

    # SOFT-PAUSE override: raise confidence floor when in a 2-4 loss streak.
    if in_soft_pause:
        effective_min_confidence = max(effective_min_confidence, cfg.soft_pause_min_confidence)

    # ── Options-specific quality gates ────────────────────────────────────

    # G1: Confidence floor (adaptive, soft-pause, or static — whichever is highest)
    g1_pass = sig.confidence >= effective_min_confidence
    g1_tag = ""
    if in_soft_pause and effective_min_confidence == cfg.soft_pause_min_confidence:
        g1_tag = " [SOFT_PAUSE]"
    elif adaptive and effective_min_confidence != cfg.min_confidence:
        g1_tag = " [ADAPTIVE]"
    checks.append((
        "G1_min_confidence",
        g1_pass,
        f"conf={sig.confidence:.2f} (need >= {effective_min_confidence:.2f}{g1_tag})",
    ))

    # G2: Spread filter — anything wider than 5% spread bleeds on entry+exit
    g2_pass = sig.spread_pct <= 5.0 if sig.spread_pct > 0 else True  # 0 = unknown, accept
    checks.append((
        "G2_spread",
        g2_pass,
        f"spread={sig.spread_pct:.2f}% (need <= 5.0%)",
    ))

    # G3: DTE bounds — avoid 0DTE gamma bombs and >45DTE theta drag
    g3_pass = 1 <= sig.dte <= 45
    checks.append((
        "G3_dte",
        g3_pass,
        f"DTE={sig.dte} (need 1..45)",
    ))

    # G4: Regime-fit
    fit_ok, fit_note = _spy_signal_fits_regime(sig)
    checks.append(("G4_regime_fit", fit_ok, fit_note))

    # G5: Premium-based per-trade $ risk vs. cap
    # SPY contracts are 100 shares; assume 1-contract size.
    # estimated_risk = 50% premium loss × 100 × legs
    risk_dollars = sig.estimated_risk_per_contract_usd
    # If we have no premium info (mid=0), treat as "size unknown" — flag but don't hard-reject.
    if risk_dollars <= 0:
        risk_within_cap = True
        risk_note = "premium not available — assuming OK"
    else:
        risk_within_cap = risk_dollars <= cfg.risk_per_trade_max_dollars
        risk_note = (
            f"~${risk_dollars:.2f} (50% premium × 100 × legs), "
            f"cap ${cfg.risk_per_trade_max_dollars:.2f}"
        )
    checks.append(("risk_within_cap", risk_within_cap, risk_note))

    # G6: Daily-loss warning band — same as MES
    in_warn_band = state.realized_pnl_today <= -cfg.daily_loss_warn_dollars
    if in_warn_band:
        checks.append((
            "daily_loss_warning_band",
            False,
            f"PnL {state.realized_pnl_today:+.2f} <= -{cfg.daily_loss_warn_dollars:.2f} (warn) — cut size",
        ))

    # ── Decision logic ────────────────────────────────────────────────────

    if not risk_within_cap:
        return Decision(
            decision="REJECT",
            confidence=95,
            position_size="small",
            reasoning=(
                f"Per-contract risk ~${risk_dollars:.2f} exceeds cap "
                f"${cfg.risk_per_trade_max_dollars:.2f} (2% of equity). "
                f"Premium too rich for this account size."
            ),
            risk_notes=(
                f"Mid {sig.mid:.2f}. Need premium <= "
                f"${cfg.risk_per_trade_max_dollars / 50.0:.2f} for the 50%-loss heuristic to fit."
            ),
            override=True,
            checks=checks,
            posture_after=state.posture,
        )

    if not g1_pass:
        return Decision(
            decision="REJECT",
            confidence=80,
            position_size="small",
            reasoning=(
                f"Confidence {sig.confidence:.2f} below floor {cfg.min_confidence:.2f}. "
                "Mandate: be selective."
            ),
            risk_notes="Wait for higher-conviction setup.",
            override=False,
            checks=checks,
            posture_after=state.posture,
        )

    if not g2_pass:
        return Decision(
            decision="REJECT",
            confidence=80,
            position_size="small",
            reasoning=(
                f"Spread {sig.spread_pct:.2f}% > 5% — round-trip cost will eat the edge. "
                "Liquidity gate."
            ),
            risk_notes="Skip illiquid contracts; the math doesn't work.",
            override=False,
            checks=checks,
            posture_after=state.posture,
        )

    if not g3_pass:
        return Decision(
            decision="REJECT",
            confidence=80,
            position_size="small",
            reasoning=(
                f"DTE={sig.dte} outside acceptable band [1, 45]. "
                "0DTE = gamma bomb; >45DTE = theta-heavy and slow."
            ),
            risk_notes="If you want 0DTE, that's a different mandate.",
            override=False,
            checks=checks,
            posture_after=state.posture,
        )

    if not fit_ok:
        return Decision(
            decision="REJECT",
            confidence=75,
            position_size="small",
            reasoning=(
                f"Regime fit failed: {fit_note}. "
                "rules_v2 should already block this — escalating."
            ),
            risk_notes="If this fires often, signal-engine and rules_v2 disagree on regime.",
            override=False,
            checks=checks,
            posture_after=state.posture,
        )

    # Passes hard gates. warning band OR soft-pause → MODIFY. Otherwise APPROVE.
    if in_warn_band or in_soft_pause:
        reasons = []
        if in_warn_band:
            reasons.append("daily-loss warning band entered")
        if in_soft_pause:
            reasons.append(f"{state.consec_losses}-loss streak (soft pause)")
        return Decision(
            decision="MODIFY",
            confidence=60 if in_soft_pause else 65,
            position_size="small",
            reasoning=(
                f"All SPY quality gates pass — {', '.join(reasons)} — "
                "approving at reduced size only."
            ),
            risk_notes=(
                "Cut to half of normal size. Tight management on this one."
                + (" Soft pause: streak breaks on a winning trade." if in_soft_pause else "")
            ),
            override=False,
            checks=checks,
            posture_after=POSTURE_DEFENSIVE if in_soft_pause else state.posture,
        )

    size = _spy_position_size_label(state, cfg)
    posture_after = state.posture
    if in_soft_pause:
        size = "small"
        posture_after = POSTURE_DEFENSIVE
    return Decision(
        decision="APPROVE",
        confidence=int(round(min(95.0, 60.0 + 35.0 * sig.confidence))),
        position_size=size,
        reasoning=(
            f"All SPY gates pass. {sig.signal_type} {sig.right}{sig.strike:.0f} "
            f"({sig.expiry}, {sig.dte}dte) | conf={sig.confidence:.2f} "
            f"| regime={sig.regime} | risk ~${risk_dollars:.2f} "
            f"({(risk_dollars/cfg.account_equity)*100.0:.2f}% equity). "
            f"Streak: {state.consec_wins}W/{state.consec_losses}L → size '{size}'."
            + (" [SOFT_PAUSE: small size enforced]" if in_soft_pause else "")
        ),
        risk_notes=(
            f"Greeks: Δ={sig.delta:.2f} Γ={sig.gamma:.3f} Θ={sig.theta:.2f} V={sig.vega:.2f}. "
            f"IV={sig.impl_vol:.2f}, IVR={sig.iv_rank:.0f}. "
            f"Daily PnL buffer remaining: ${cfg.daily_loss_hard_dollars + state.realized_pnl_today:.2f}."
        ),
        override=False,
        checks=checks,
        posture_after=posture_after,
    )
