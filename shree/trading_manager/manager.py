"""Trading Manager main loop.

Run forever:
  - tail decisions.jsonl for new SIGNAL rows
  - on each new signal, refresh state from orders.db, evaluate, log decision
  - poll daily PnL every N seconds; if it crosses the hard-stop, kill bots
  - persist state on every change
"""
from __future__ import annotations

import logging
import os
import signal as signal_mod
import sys
import time
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from .backfill_learning import run_backfill
from .bot_control import bots_alive, kill_bots
from .config import CONFIG
from .decision_log import append_decision, append_spy_decision
from .executions_reader import (
    last_n_closed_trades,
    open_position_count,
    realized_pnl_for_session,
    streaks_from_recent,
)
from .health import (
    HEALTH_HEALTHY,
    HEALTH_DEGRADED,
    HEALTH_SUSPECT,
    HEALTH_LOCKED,
    HEALTH_PROBATION,
    compute_metrics,
    render_summary,
    status_from_triggers,
)
from .learning import summary as learning_summary
from .rules import evaluate, evaluate_spy
from .signal_watcher import SignalTailer
from .spy_signal_watcher import SpySignalTailer
from .state import (
    POSTURE_DEFENSIVE,
    POSTURE_KILLED,
    POSTURE_LOCKED,
    POSTURE_NORMAL,
    POSTURE_PROBATION,
    POSTURE_SIT_OUT,
    ManagerState,
    load_state,
    roll_session_if_needed,
    save_state,
)


CT = ZoneInfo("America/Chicago")


def _setup_logging(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    logger = logging.getLogger("trading_manager")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    )
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.handlers = [fh, sh]
    logger.propagate = False
    # silence sub-loggers
    for name in ("trading_manager.bot_control",):
        logging.getLogger(name).handlers = [fh, sh]
        logging.getLogger(name).setLevel(logging.INFO)
        logging.getLogger(name).propagate = False
    return logger


def _today_ct() -> str:
    return datetime.now(CT).strftime("%Y-%m-%d")


def _write_pid(pid_path: str) -> None:
    os.makedirs(os.path.dirname(pid_path) or ".", exist_ok=True)
    with open(pid_path, "w") as f:
        f.write(str(os.getpid()))


def _refresh_state_from_disk(state: ManagerState, cfg) -> None:
    """Recompute today's PnL/trade count and streaks from the truth source.

    MAY 8 2026: streaks are now bounded to the current CT trading session.
    Without this bound, a multi-day losing run permanently locks the system
    (no entries → no outcomes → streak never breaks). `last_n_outcomes`
    keeps the broader rolling history for journals/reports — only the
    *streak counter* is session-scoped.
    """
    today = _today_ct()
    rolled = roll_session_if_needed(state, today)

    pnl, n = realized_pnl_for_session(cfg.orders_db, today)
    state.realized_pnl_today = pnl
    state.trades_today = n

    recent = last_n_closed_trades(cfg.orders_db, n=20)

    # Session start in UTC ISO ~ 22:00 prior CT day or 17:00 CT same day.
    # The CME session boundary is 17:00 CT (5pm settlement). For streak
    # scoping purposes, "today's session" means trades with timestamp >=
    # today's CT date 00:00. Tighter than the actual CME session but
    # cleaner — and the executions table writes UTC ISO so we filter
    # on the date prefix of the local CT date converted back to UTC.
    # Simpler approach: filter on the calendar day prefix of the trade's
    # timestamp. The executions table stores UTC; today's CT trades will
    # have UTC timestamps either today or yesterday's UTC depending on time.
    # We pass the earliest UTC timestamp that could correspond to today's
    # CT session start (today CT 00:00 = today UTC 05:00 / 06:00 depending on DST).
    from datetime import datetime as _dt, timedelta
    ct_midnight = _dt.now(CT).replace(hour=0, minute=0, second=0, microsecond=0)
    session_start_utc_iso = ct_midnight.astimezone(ZoneInfo("UTC")).isoformat(timespec="seconds")
    # Strip the timezone suffix because executions.timestamp is naive UTC ISO
    # (e.g. "2026-05-07T15:57:35.509979" — no offset).
    session_start_naive_utc = session_start_utc_iso.replace("+00:00", "").rstrip("Z")

    wins, losses = streaks_from_recent(recent, since_iso=session_start_naive_utc)
    state.consec_wins = wins
    state.consec_losses = losses
    state.last_n_outcomes = [
        "WIN" if t.is_win else "LOSS" for t in recent[: cfg.last_n_for_pattern_check]
    ]
    return rolled


def _enforce_kill_switch(state: ManagerState, cfg, log) -> bool:
    """If realized PnL has breached hard stop, set posture=KILLED and SIGTERM bots.
    Returns True if the kill switch fired this cycle.
    """
    if state.realized_pnl_today <= -cfg.daily_loss_hard_dollars:
        if state.posture != POSTURE_KILLED:
            log.error(
                "🚨 DAILY LOSS HARD STOP: %.2f <= -%.2f. KILLING BOTS.",
                state.realized_pnl_today,
                cfg.daily_loss_hard_dollars,
            )
            res = kill_bots([cfg.bot_pid_file, cfg.spy_pid_file], dry_run=cfg.dry_run)
            for pf, pid, status in res:
                log.warning("  kill: %s pid=%s status=%s", pf, pid, status)
            state.posture = POSTURE_KILLED
            save_state(state, cfg.state_file)
            return True
    return False


def _maybe_update_posture(state: ManagerState, cfg, log) -> None:
    """Update posture based on PnL warn band, streak, etc. (non-killing)."""
    if state.posture == POSTURE_KILLED:
        return  # only daily roll can leave KILLED
    new_posture = POSTURE_NORMAL
    if state.realized_pnl_today <= -cfg.daily_loss_warn_dollars:
        new_posture = POSTURE_DEFENSIVE
    if state.consec_losses >= cfg.consec_losses_pause:
        # 2-loss streak → DEFENSIVE (soft pause: small size, tighter gates)
        new_posture = POSTURE_DEFENSIVE
    if state.consec_losses >= cfg.consec_losses_hard_pause:
        # 5-loss streak → SIT_OUT (hard pause: reject all entries)
        new_posture = POSTURE_SIT_OUT
    if state.trades_today >= cfg.max_trades_per_day:
        new_posture = POSTURE_SIT_OUT
    if new_posture != state.posture:
        log.warning(
            "Posture %s → %s (pnl=%+.2f, trades=%d, consec_L=%d)",
            state.posture, new_posture,
            state.realized_pnl_today, state.trades_today, state.consec_losses,
        )
        state.posture = new_posture


def _run_health_check(state: ManagerState, cfg, log) -> None:
    """Compute the four-metric health snapshot and update state.health_*.

    Posture transition rules (driven purely by trigger count):
      • >=3 triggers           → LOCKED (manual unlock required)
      • 2 triggers             → SIT_OUT (hard pause for the day)
      • 1 trigger              → DEFENSIVE (escalate one tier)
      • 0 triggers             → no change (don't auto-relax existing
                                 KILLED/streak-driven postures)

    LOCKED and PROBATION are multi-day flags that survive session rolls;
    only the unlock CLI (or the probation winner) clears them.
    """
    try:
        m = compute_metrics(cfg.orders_db, account_equity=cfg.account_equity)
    except Exception as exc:
        log.warning("Health check failed (non-fatal): %s", exc)
        return

    state.health_summary = render_summary(m)
    state.health_triggers = m.triggers
    state.health_last_check = datetime.now().astimezone().isoformat(timespec="seconds")
    new_status = status_from_triggers(m.trigger_count)

    # Don't auto-clear PROBATION via health check — only a winning probation
    # trade clears it. PROBATION is set by the unlock CLI handler.
    if state.posture == POSTURE_PROBATION:
        state.health_status = HEALTH_PROBATION
        return

    state.health_status = new_status

    if new_status == HEALTH_LOCKED:
        if state.posture != POSTURE_LOCKED:
            log.error(
                "🔒 HEALTH CHECK: %d triggers (%s) — entering LOCKED posture. "
                "Manual unlock required: python -m shree.trading_manager.unlock",
                m.trigger_count, ", ".join(m.triggers),
            )
            state.posture = POSTURE_LOCKED
            state.lock_reason = (
                f"{m.trigger_count} health triggers fired: {', '.join(m.triggers)}. "
                f"Snapshot: {state.health_summary}"
            )
            state.lock_since = state.health_last_check
        return

    if new_status == HEALTH_SUSPECT:
        # 2 triggers — hard pause for the day, not multi-day lock
        if state.posture not in (POSTURE_LOCKED, POSTURE_KILLED):
            if state.posture != POSTURE_SIT_OUT:
                log.warning(
                    "⚠️  HEALTH CHECK: 2 triggers (%s) — posture → SIT_OUT for the day",
                    ", ".join(m.triggers),
                )
            state.posture = POSTURE_SIT_OUT
        return

    if new_status == HEALTH_DEGRADED:
        if state.posture not in (POSTURE_LOCKED, POSTURE_KILLED, POSTURE_SIT_OUT):
            if state.posture != POSTURE_DEFENSIVE:
                log.info(
                    "🟡 HEALTH CHECK: 1 trigger (%s) — posture → DEFENSIVE",
                    ", ".join(m.triggers),
                )
            state.posture = POSTURE_DEFENSIVE
        return

    # HEALTHY — clear LOCKED only if the unlock-CLI path puts us there.
    # Don't auto-relax DEFENSIVE/SIT_OUT here; the streak/PnL rules manage them.


def _check_unlock_marker(state: ManagerState, cfg, log) -> None:
    """If the unlock CLI has dropped a marker file, transition LOCKED → PROBATION.

    The CLI writes `logs/trading_manager_unlock.marker` containing a reason
    line. We consume it (delete it) and move to PROBATION.
    """
    marker_path = cfg.unlock_marker_file
    if not os.path.exists(marker_path):
        return
    if state.posture != POSTURE_LOCKED:
        # User dropped a marker but we're not LOCKED — clean it up and move on
        try:
            os.remove(marker_path)
        except OSError:
            pass
        log.info("Unlock marker present but posture=%s — ignored.", state.posture)
        return
    try:
        with open(marker_path, "r") as f:
            reason = f.read().strip()
    except OSError:
        reason = "(unreadable)"
    try:
        os.remove(marker_path)
    except OSError:
        pass
    log.warning(
        "🔓 UNLOCK marker received: '%s' — transitioning LOCKED → PROBATION. "
        "Next trade is the probe; must win to clear.",
        reason or "(no reason)",
    )
    state.posture = POSTURE_PROBATION
    state.health_status = HEALTH_PROBATION
    state.probation_started = datetime.now().astimezone().isoformat(timespec="seconds")
    state.probation_trade_count = 0
    state.lock_reason = ""


def _check_probation_outcome(state: ManagerState, cfg, log) -> None:
    """If we're in PROBATION and a new closed trade has appeared, evaluate it.

    A winning probe → clear PROBATION (posture → NORMAL).
    A losing probe  → back to LOCKED, increment probation_trade_count, log loud.
    """
    if state.posture != POSTURE_PROBATION:
        return
    # Look at the single most-recent closed trade
    recent = last_n_closed_trades(cfg.orders_db, n=1)
    if not recent:
        return
    t = recent[0]
    # Only evaluate trades that happened AFTER probation started
    if state.probation_started and t.timestamp:
        try:
            prob_dt = datetime.fromisoformat(state.probation_started.replace("Z", "+00:00"))
            # t.timestamp is naive UTC ISO
            trade_dt = datetime.fromisoformat(t.timestamp).replace(tzinfo=ZoneInfo("UTC"))
            if trade_dt < prob_dt:
                return  # not a probation trade
        except (ValueError, AttributeError):
            return

    state.probation_trade_count += 1
    if t.is_win:
        log.warning(
            "✅ PROBATION CLEARED: probe trade WON (+$%.2f). Posture → NORMAL.",
            t.net_pnl,
        )
        state.posture = POSTURE_NORMAL
        state.health_status = HEALTH_HEALTHY
        state.probation_started = ""
        state.probation_trade_count = 0
    else:
        log.error(
            "❌ PROBATION FAILED: probe trade LOST (-$%.2f). Back to LOCKED. "
            "Manual unlock required.",
            abs(t.net_pnl),
        )
        state.posture = POSTURE_LOCKED
        state.health_status = HEALTH_LOCKED
        state.lock_reason = (
            f"Probation probe failed (lost ${abs(t.net_pnl):.2f}). "
            f"Original lock reason: {state.lock_reason or '(unknown)'}"
        )
        state.lock_since = datetime.now().astimezone().isoformat(timespec="seconds")
        state.probation_started = ""


def _heartbeat(state: ManagerState, cfg, log) -> None:
    alive = bots_alive([cfg.bot_pid_file, cfg.spy_pid_file])
    alive_str = ", ".join(f"{os.path.basename(pf)}=pid{pid}{'✓' if a else '✗'}"
                          for pf, pid, a in alive)
    # Add learning summary
    try:
        s = learning_summary(cfg.learning_db)
        learn_str = f"{s.get('buckets', 0)}b/{s.get('trades', 0)}t"
    except Exception:
        learn_str = "n/a"
    log.info(
        "💓 TM heartbeat | %s | pnl=%+.2f / -$%.2f cap | trades=%d/%d | "
        "streak=%dW/%dL | posture=%s | health=%s | learn=%s | bots: %s",
        state.session_date,
        state.realized_pnl_today,
        cfg.daily_loss_hard_dollars,
        state.trades_today,
        cfg.max_trades_per_day,
        state.consec_wins,
        state.consec_losses,
        state.posture,
        state.health_status or "n/a",
        learn_str,
        alive_str,
    )


def run() -> int:
    cfg = CONFIG
    log = _setup_logging(cfg.log_file)
    log.info("=" * 70)
    log.info("Trading Manager starting")
    log.info("Equity=$%.2f | per-trade cap $%.2f-%.2f | daily stop -$%.2f",
             cfg.account_equity,
             cfg.risk_per_trade_pref_dollars,
             cfg.risk_per_trade_max_dollars,
             cfg.daily_loss_hard_dollars)
    log.info("Decisions in: %s | Decisions out: %s | State: %s",
             cfg.decisions_jsonl, cfg.manager_jsonl, cfg.state_file)
    log.info("dry_run=%s", cfg.dry_run)
    log.info("=" * 70)

    _write_pid(cfg.pid_file)

    state = load_state(cfg.state_file)
    rolled = _refresh_state_from_disk(state, cfg)
    if rolled:
        log.info("Session rolled to %s — daily counters reset", state.session_date)
    save_state(state, cfg.state_file)

    tailer = SignalTailer(cfg.decisions_jsonl, start_at_end=True)
    spy_tailer = SpySignalTailer(cfg.spy_signals_jsonl, start_at_end=True)
    log.info("MES tailer: %s | SPY tailer: %s",
             cfg.decisions_jsonl, cfg.spy_signals_jsonl)

    # Heartbeat cadence
    last_hb = 0.0
    HB_INTERVAL = 60.0
    last_pnl_refresh = 0.0
    PNL_REFRESH_INTERVAL = 5.0  # seconds — cheap query
    last_learning_refresh = 0.0
    LEARNING_REFRESH_INTERVAL = float(cfg.learning_refresh_seconds)
    last_health_check = 0.0
    HEALTH_CHECK_INTERVAL = float(cfg.health_check_interval_seconds)

    # Initial health check at startup — populates state.health_* immediately
    # so the heartbeat shows real values from cycle 0.
    try:
        _run_health_check(state, cfg, log)
        log.info("Initial health: %s | %s", state.health_status, state.health_summary)
    except Exception as exc:
        log.warning("Initial health check failed (non-fatal): %s", exc)

    # Initial learning backfill: run once at startup so the rules engine has
    # historical context immediately. Safe + idempotent.
    try:
        res = run_backfill(
            learning_db=cfg.learning_db,
            orders_db=cfg.orders_db,
            decisions_jsonl=cfg.decisions_jsonl,
            spy_db=cfg.spy_signals_db,
        )
        log.info(
            "Learning backfill: MES new=%d skip=%d | SPY new=%d skip=%d",
            res["mes"]["new"], res["mes"]["skipped"],
            res["spy"]["new"], res["spy"]["skipped"],
        )
        s = learning_summary(cfg.learning_db)
        log.info("Learning DB: %d buckets, %d trades", s.get("buckets", 0), s.get("trades", 0))
    except Exception as exc:
        log.warning("Learning backfill failed (non-fatal): %s", exc)

    stop_flag = {"stop": False}
    def _onsig(signum, frame):
        log.warning("Signal %s received — shutting down", signum)
        stop_flag["stop"] = True
    signal_mod.signal(signal_mod.SIGTERM, _onsig)
    signal_mod.signal(signal_mod.SIGINT, _onsig)

    log.info("Entering main loop (poll every %.1fs)", cfg.poll_interval_seconds)

    try:
        while not stop_flag["stop"]:
            now = time.time()

            # Refresh PnL/trade count from db cheaply
            if now - last_pnl_refresh >= PNL_REFRESH_INTERVAL:
                _refresh_state_from_disk(state, cfg)
                _enforce_kill_switch(state, cfg, log)
                _maybe_update_posture(state, cfg, log)
                # Layer 1.5: probation outcome check (only fires when state
                # is PROBATION and a new closed trade has appeared)
                _check_probation_outcome(state, cfg, log)
                # Unlock marker watch — instant transition LOCKED → PROBATION
                _check_unlock_marker(state, cfg, log)
                save_state(state, cfg.state_file)
                last_pnl_refresh = now

            # Layer 1.5 health check (multi-day forest view)
            if now - last_health_check >= HEALTH_CHECK_INTERVAL:
                _run_health_check(state, cfg, log)
                save_state(state, cfg.state_file)
                last_health_check = now

            # Refresh learning DB from any newly-closed trades.
            # This keeps Layer-1 intelligence up-to-date as trades close.
            if now - last_learning_refresh >= LEARNING_REFRESH_INTERVAL:
                try:
                    res = run_backfill(
                        learning_db=cfg.learning_db,
                        orders_db=cfg.orders_db,
                        decisions_jsonl=cfg.decisions_jsonl,
                        spy_db=cfg.spy_signals_db,
                    )
                    new_total = res["mes"]["new"] + res["spy"]["new"]
                    if new_total > 0:
                        log.info(
                            "Learning refresh: %d new trade events ingested "
                            "(MES=%d, SPY=%d)",
                            new_total, res["mes"]["new"], res["spy"]["new"],
                        )
                except Exception as exc:
                    log.warning("Learning refresh failed (non-fatal): %s", exc)
                last_learning_refresh = now

            # Process new signals
            for sig in tailer.poll():
                if not sig.action:
                    continue
                state.last_signal_ts = sig.ts
                state.last_decision_id += 1
                recent = last_n_closed_trades(cfg.orders_db, n=20)
                decision = evaluate(sig, state, recent, cfg)
                append_decision(
                    cfg.manager_jsonl,
                    state.last_decision_id,
                    sig,
                    decision,
                    state,
                )
                log.info(
                    "Signal #%d %s %s @ %.2f | %s | conf=%d | size=%s | %s",
                    state.last_decision_id,
                    sig.action,
                    sig.signal_type,
                    sig.close,
                    decision.decision,
                    decision.confidence,
                    decision.position_size,
                    decision.reasoning[:120],
                )
                # Apply posture changes the rules requested
                if decision.posture_after != state.posture:
                    log.warning("Posture %s → %s (rule-driven)",
                                state.posture, decision.posture_after)
                    state.posture = decision.posture_after
                # If a rule moved us to KILLED, fire the kill switch immediately
                if state.posture == POSTURE_KILLED:
                    res = kill_bots(
                        [cfg.bot_pid_file, cfg.spy_pid_file],
                        dry_run=cfg.dry_run,
                    )
                    for pf, pid, status in res:
                        log.warning("  kill: %s pid=%s status=%s", pf, pid, status)
                save_state(state, cfg.state_file)

            # Process new SPY options signals
            for spy_sig in spy_tailer.poll():
                state.last_signal_ts = spy_sig.ts
                state.last_decision_id += 1
                recent = last_n_closed_trades(cfg.orders_db, n=20)
                spy_decision = evaluate_spy(spy_sig, state, recent, cfg)
                append_spy_decision(
                    cfg.manager_jsonl,
                    state.last_decision_id,
                    spy_sig,
                    spy_decision,
                    state,
                )
                log.info(
                    "SPY #%d %s %s%g %s/%dDTE | %s | conf=%d | size=%s | %s",
                    state.last_decision_id,
                    spy_sig.signal_type,
                    spy_sig.right,
                    spy_sig.strike,
                    spy_sig.expiry,
                    spy_sig.dte,
                    spy_decision.decision,
                    spy_decision.confidence,
                    spy_decision.position_size,
                    spy_decision.reasoning[:120],
                )
                if spy_decision.posture_after != state.posture:
                    log.warning("Posture %s → %s (SPY rule-driven)",
                                state.posture, spy_decision.posture_after)
                    state.posture = spy_decision.posture_after
                if state.posture == POSTURE_KILLED:
                    res = kill_bots(
                        [cfg.bot_pid_file, cfg.spy_pid_file],
                        dry_run=cfg.dry_run,
                    )
                    for pf, pid, status in res:
                        log.warning("  kill: %s pid=%s status=%s", pf, pid, status)
                save_state(state, cfg.state_file)

            # Heartbeat
            if now - last_hb >= HB_INTERVAL:
                _heartbeat(state, cfg, log)
                last_hb = now

            time.sleep(cfg.poll_interval_seconds)
    finally:
        tailer.close()
        spy_tailer.close()
        try:
            os.remove(cfg.pid_file)
        except OSError:
            pass
        log.info("Trading Manager stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
