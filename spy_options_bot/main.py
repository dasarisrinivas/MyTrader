"""SPY Weekly Options Selling Bot — entry point.

Usage:
    python spy_options_bot/main.py [--dry-run] [--reset-pdt] [--once]

Flags:
    --dry-run     Simulate everything but never call placeOrder
    --reset-pdt   Prompt to clear the PDT log, then exit
    --once        Run one evaluation cycle then exit (useful for testing)
"""
from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# Ensure spy_options_bot/ is on the path when run from repo root
_here = Path(__file__).parent
sys.path.insert(0, str(_here))

import config
from ibkr_connection import IBKRConnection
from logger import configure_logging, logger
from notifier import Notifier
from option_chain import fetch_option_chain, select_best_call, select_best_put
from order_manager import OrderManager, load_open_positions
from pdt_tracker import PDTTracker
from risk_manager import RiskManager
from signal_engine import Strategy, evaluate_signals

ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def _now_et() -> datetime:
    return datetime.now(ET)


def _is_entry_window() -> bool:
    """True if current time is within entry window (Mon–Wed, 9:35–15:30 ET)."""
    now = _now_et()
    if now.weekday() not in config.ENTRY_DAYS:
        return False
    after_open = (now.hour > config.MARKET_OPEN_HOUR) or (
        now.hour == config.MARKET_OPEN_HOUR and now.minute >= config.MARKET_OPEN_MINUTE
    )
    before_close = (now.hour < config.MARKET_CLOSE_HOUR) or (
        now.hour == config.MARKET_CLOSE_HOUR and now.minute <= config.MARKET_CLOSE_MINUTE
    )
    return after_open and before_close


def _is_market_hours() -> bool:
    """True between 9:30–16:00 ET Mon–Fri (for position monitoring)."""
    now = _now_et()
    if now.weekday() >= 5:
        return False
    after_open = (now.hour > 9) or (now.hour == 9 and now.minute >= 30)
    before_close = now.hour < 16
    return after_open and before_close


# ---------------------------------------------------------------------------
# Core trading cycle
# ---------------------------------------------------------------------------

async def run_cycle(
    ib_conn: IBKRConnection,
    order_mgr: OrderManager,
    risk_mgr: RiskManager,
    pdt: PDTTracker,
    notifier: Notifier,
    dry_run: bool,
) -> None:
    """One evaluation cycle: monitor positions, then consider new entry."""
    now = _now_et()
    logger.info(f"--- Cycle start: {now.strftime('%Y-%m-%d %H:%M:%S')} ET ---")

    # Always monitor open positions during market hours
    if _is_market_hours():
        await risk_mgr.monitor_positions()

    # Skip new entry logic outside entry window
    if not _is_entry_window():
        logger.info(
            f"Outside entry window (weekday={now.weekday()}, "
            f"{now.strftime('%H:%M')} ET) — skipping entry evaluation"
        )
        return

    # PDT check
    slots = pdt.slots_remaining()
    logger.info(
        f"PDT status: {pdt.get_weekly_count()}/{pdt.max_trades} trades used | "
        f"{slots} slot(s) remaining"
    )
    if not pdt.can_trade():
        logger.warning("PDT limit reached — no new entries this week")
        return

    if slots == 1:
        notifier.on_pdt_warning(slots)

    # Portfolio-level risk check
    can_enter, reason = await risk_mgr.can_enter()
    if not can_enter:
        logger.info(f"Entry blocked by risk check: {reason}")
        return

    # Signal evaluation (PDT-aware: strangle downgrades if slots < 2)
    signal = await evaluate_signals(ib_conn.ib, pdt=pdt)
    if signal.strategy == Strategy.NO_TRADE:
        logger.info(f"No trade signal: {signal.reason}")
        return

    # Fetch option chain
    logger.info(f"Fetching option chain for strategy: {signal.strategy.value}")
    candidates = await fetch_option_chain(ib_conn.ib)

    if not candidates:
        logger.warning("No valid option candidates found after filtering")
        return

    # Determine strategy_type for PDT slot tracking
    is_strangle = signal.strategy == Strategy.SELL_STRANGLE
    strategy_type = "strangle" if is_strangle else "single"

    opened_positions = []

    if signal.strategy in (Strategy.SELL_PUT, Strategy.SELL_STRANGLE):
        put = select_best_put(candidates)
        if put:
            logger.info(
                f"Selected PUT: strike={put.strike} delta={put.delta:.3f} "
                f"theta={put.theta:.4f} mid=${put.mid:.2f}"
            )
            pos = await order_mgr.sell_option(put, strategy_type=strategy_type)
            if pos:
                opened_positions.append(pos)
        else:
            logger.warning("No suitable PUT found")

    if signal.strategy in (Strategy.SELL_CALL, Strategy.SELL_STRANGLE):
        # For strangle: only add call leg if put was successfully placed
        if signal.strategy == Strategy.SELL_CALL or opened_positions:
            call = select_best_call(candidates)
            if call:
                logger.info(
                    f"Selected CALL: strike={call.strike} delta={call.delta:.3f} "
                    f"theta={call.theta:.4f} mid=${call.mid:.2f}"
                )
                pos = await order_mgr.sell_option(call, strategy_type=strategy_type)
                if pos:
                    opened_positions.append(pos)
            else:
                logger.warning("No suitable CALL found")

    if opened_positions:
        logger.info(
            f"Opened {len(opened_positions)} position(s) | strategy: {signal.strategy.value}"
        )
    else:
        logger.info("No positions opened this cycle")


# ---------------------------------------------------------------------------
# Shutdown handler
# ---------------------------------------------------------------------------

def _install_signal_handlers(
    loop: asyncio.AbstractEventLoop,
    ib_conn: IBKRConnection,
    order_mgr: OrderManager,
    shutdown_event: asyncio.Event,
) -> None:
    """Register SIGINT / SIGTERM for graceful shutdown."""

    def _handle_signal(sig_name: str) -> None:
        logger.warning(f"Received {sig_name} — initiating graceful shutdown")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal, sig.name)


async def _graceful_shutdown(ib_conn: IBKRConnection, order_mgr: OrderManager) -> None:
    """Cancel open orders, log positions, disconnect."""
    logger.info("Graceful shutdown: cancelling open orders...")
    try:
        open_trades = ib_conn.ib.openTrades()
        for t in open_trades:
            ib_conn.ib.cancelOrder(t.order)
            logger.info(f"Cancelled order: {t.order.orderId}")
    except Exception as exc:
        logger.error(f"Error cancelling orders during shutdown: {exc}")

    positions = load_open_positions()
    if positions:
        logger.info(f"Shutdown with {len(positions)} open position(s):")
        for p in positions:
            logger.info(
                f"  {p.position_id} | entry=${p.entry_premium:.2f} | "
                f"target=${p.profit_target_price:.2f} | stop=${p.stop_loss_price:.2f}"
            )
    else:
        logger.info("No open positions at shutdown.")

    await ib_conn.disconnect()
    logger.info("Disconnected from IBKR. Bot stopped.")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def main(dry_run: bool = False, run_once: bool = False) -> None:
    configure_logging(log_file=config.LOG_FILE, level=config.LOG_LEVEL)

    logger.info("=" * 60)
    logger.info("SPY Options Bot starting")
    logger.info(f"Mode: {'DRY RUN' if dry_run else config.TRADING_MODE.upper()}")
    logger.info(f"IBKR: {config.IBKR_HOST}:{config.IBKR_PORT} clientId={config.CLIENT_ID}")
    logger.info("=" * 60)

    notifier = Notifier()

    pdt = PDTTracker(filepath=config.PDT_LOG_FILE, max_trades=config.MAX_WEEKLY_TRADES)
    slots = pdt.slots_remaining()
    logger.info(
        f"PDT status: {pdt.get_weekly_count()}/{pdt.max_trades} trades used. "
        f"{slots} slot(s) remaining."
    )
    if slots == 1:
        notifier.on_pdt_warning(slots)

    ib_conn = IBKRConnection(
        host=config.IBKR_HOST,
        port=config.IBKR_PORT,
        client_id=config.CLIENT_ID,
    )

    async def _on_reconnect() -> None:
        logger.info("Reconnected — resuming monitoring")
        notifier.on_connection_restored()

    ib_conn.on_reconnect(_on_reconnect)

    # Wrap disconnect event to alert
    _orig_on_disconnected = ib_conn._on_disconnected

    def _on_disconnected_with_alert() -> None:
        notifier.on_connection_lost()
        _orig_on_disconnected()

    ib_conn._on_disconnected = _on_disconnected_with_alert  # type: ignore[method-assign]

    await ib_conn.connect()

    order_mgr = OrderManager(ib_conn.ib, dry_run=dry_run, notifier=notifier)
    risk_mgr = RiskManager(ib_conn.ib, order_mgr, notifier=notifier)

    # Hook PDT recording onto every close (fires after close_position succeeds)
    _original_close = order_mgr.close_position

    async def _close_with_pdt(pos, reason: str) -> bool:
        result = await _original_close(pos, reason)
        if result:
            # Each leg close = 1 FINRA round-trip, regardless of strategy_type
            pdt.record_round_trip(
                strategy_type="single",
                symbol=pos.symbol,
                description=f"{pos.right}{pos.strike} {pos.expiry} | reason={reason}",
            )
        return result

    order_mgr.close_position = _close_with_pdt  # type: ignore[method-assign]

    if run_once:
        await run_cycle(ib_conn, order_mgr, risk_mgr, pdt, notifier, dry_run)
        await _graceful_shutdown(ib_conn, order_mgr)
        return

    # Main loop with dynamic poll interval and graceful shutdown
    loop = asyncio.get_event_loop()
    shutdown_event = asyncio.Event()
    _install_signal_handlers(loop, ib_conn, order_mgr, shutdown_event)

    logger.info("Starting main loop...")
    while not shutdown_event.is_set():
        try:
            await run_cycle(ib_conn, order_mgr, risk_mgr, pdt, notifier, dry_run)
        except Exception as exc:
            logger.exception(f"Unhandled error in cycle: {exc}")

        # Dynamic interval: tighter on Thursdays / low DTE
        interval = risk_mgr.get_poll_interval()
        logger.debug(f"Sleeping {interval}s until next cycle...")

        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass  # Normal — timeout means no shutdown signal, continue loop

    await _graceful_shutdown(ib_conn, order_mgr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPY Weekly Options Selling Bot")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate everything without placing real orders",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one evaluation cycle and exit",
    )
    parser.add_argument(
        "--reset-pdt",
        action="store_true",
        help="Prompt to clear the PDT log file and exit",
    )
    args = parser.parse_args()

    if args.reset_pdt:
        path = Path(config.PDT_LOG_FILE)
        if path.exists():
            confirm = input(
                f"Are you sure? This will clear the PDT log at '{path}'. "
                "Type YES to confirm: "
            ).strip()
            if confirm == "YES":
                path.unlink()
                print(f"PDT log cleared: {path}")
            else:
                print("Aborted — PDT log unchanged.")
        else:
            print(f"PDT log not found at '{path}' — nothing to clear.")
        sys.exit(0)

    asyncio.run(main(dry_run=args.dry_run, run_once=args.once))
