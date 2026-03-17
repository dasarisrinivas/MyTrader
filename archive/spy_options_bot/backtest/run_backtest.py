"""CLI entry point for SPY options backtest.

Usage:
    # Run with defaults (1 year, $25k starting capital)
    python spy_options_bot/backtest/run_backtest.py

    # Custom date range
    python spy_options_bot/backtest/run_backtest.py --start 2025-01-01 --end 2025-12-31

    # Custom capital
    python spy_options_bot/backtest/run_backtest.py --capital 50000

    # Skip IBKR download (use cached data)
    python spy_options_bot/backtest/run_backtest.py --use-cache

    # Download only — do not run simulation
    python spy_options_bot/backtest/run_backtest.py --download-only

    # Strategy variants
    python spy_options_bot/backtest/run_backtest.py --strategy puts_only
    python spy_options_bot/backtest/run_backtest.py --strategy strangles_only
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path

# ── Path setup: allow running from repo root or spy_options_bot/ ──────────────
_file = Path(__file__).resolve()
_backtest_dir = _file.parent          # spy_options_bot/backtest/
_bot_dir = _backtest_dir.parent       # spy_options_bot/
_repo_root = _bot_dir.parent          # repo root

sys.path.insert(0, str(_bot_dir))     # lets us import config, logger, etc.
sys.path.insert(0, str(_backtest_dir.parent))  # lets backtest/ find its siblings

from logger import configure_logging, logger

CACHE_DIR = _repo_root / "backtest_results" / "cache"
OUTPUT_DIR = _repo_root / "backtest_results"


# ---------------------------------------------------------------------------
# Async data download (requires IBKR connection)
# ---------------------------------------------------------------------------

async def download_data(end_date_str: str = "") -> None:
    """Connect to IBKR on clientId=3 and download required historical data."""
    import config
    from ibkr_connection import IBKRConnection
    from backtest.data_downloader import DataDownloader

    logger.info(
        f"Connecting to IBKR {config.IBKR_HOST}:{config.IBKR_PAPER_PORT} "
        f"clientId=3 for data download..."
    )
    conn = IBKRConnection(
        host=config.IBKR_HOST,
        port=config.IBKR_PAPER_PORT,
        client_id=3,
    )
    await conn.connect()

    try:
        downloader = DataDownloader(ib=conn.ib, cache_dir=CACHE_DIR)
        paths = await downloader.download_all(end_date=end_date_str)
        for name, path in paths.items():
            logger.info(f"  {name}: {path}")
    finally:
        await conn.disconnect()


# ---------------------------------------------------------------------------
# Synchronous backtest + reporting
# ---------------------------------------------------------------------------

def run_simulation(
    start: date,
    end: date,
    capital: float,
    strategy: str,
    max_vix: float = float("inf"),
) -> None:
    from backtest.backtest_engine import BacktestEngine
    from backtest.metrics import calculate_metrics
    from backtest.reporter import Reporter

    vix_note = f" | VIX ≤ {max_vix}" if max_vix < float("inf") else ""
    logger.info(f"Running backtest: {start} → {end} | capital=${capital:,.0f} | strategy={strategy}{vix_note}")

    engine = BacktestEngine(cache_dir=CACHE_DIR)
    results = engine.run(start_date=start, end_date=end,
                         initial_capital=capital, strategy=strategy,
                         max_vix=max_vix)

    logger.info(f"Simulation complete: {len(results.trades)} trades")

    metrics = calculate_metrics(results)

    reporter = Reporter(results, metrics, output_dir=OUTPUT_DIR)
    paths = reporter.generate()

    _print_summary(metrics, results.initial_capital, paths)


def _print_summary(m: dict, initial: float, paths: dict[str, Path]) -> None:
    dd_peak = m.get("drawdown_peak_date", "N/A")
    dd_recover = m.get("drawdown_recovery_date", "N/A")

    win_str = f"{m.get('win_rate_pct',0):.1f}%  ({m.get('winning_trades',0)}/{m.get('total_trades',0)} trades)"
    dd_str = f"{m.get('max_drawdown_pct',0):.1f}%  (peaked {dd_peak}, recovered {dd_recover})"

    print()
    print("=" * 60)
    print("  SPY OPTIONS BOT — BACKTEST RESULTS")
    print(f"  {m.get('start_date','')} → {m.get('end_date','')}")
    print("=" * 60)
    print(f"  Starting Capital   :  ${initial:>12,.2f}")
    print(f"  Ending Capital     :  ${m.get('final_equity', initial):>12,.2f}")
    print(f"  Total Return       :  {m.get('total_return_pct', 0):>+10.2f}%")
    print(f"  Annualized Return  :  {m.get('annualized_return_pct', 0):>+10.2f}%")
    print(f"  Win Rate           :  {win_str}")
    print(f"  Sharpe Ratio       :  {m.get('sharpe_ratio', 0):>12.2f}")
    print(f"  Max Drawdown       :  {dd_str}")
    print(f"  Profit Factor      :  {m.get('profit_factor', 0):>12.2f}")
    print(f"  Avg P&L / Trade    :  ${m.get('avg_pnl_per_trade', 0):>+11.2f}")
    print(f"  PDT-Blocked Weeks  :  {m.get('pdt_blocked_weeks', 0)}")
    print("=" * 60)
    print(f"  Full report saved  → {paths.get('report_html', 'N/A')}")
    print(f"  Trade log saved    → {paths.get('trades_csv', 'N/A')}")
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    one_year_ago = date.today() - timedelta(days=365)
    today = date.today()

    parser = argparse.ArgumentParser(
        description="SPY Options Bot — Backtest Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start", default=one_year_ago.isoformat(),
                        help="Start date YYYY-MM-DD (default: 1 year ago)")
    parser.add_argument("--end", default=today.isoformat(),
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--capital", type=float, default=25_000.0,
                        help="Starting capital in USD (default: 25000)")
    parser.add_argument("--strategy",
                        choices=["auto", "puts_only", "calls_only", "strangles_only"],
                        default="auto",
                        help="Strategy variant (default: auto = signal_engine decides)")
    parser.add_argument("--use-cache", action="store_true",
                        help="Skip IBKR download; use cached data only")
    parser.add_argument("--download-only", action="store_true",
                        help="Download data from IBKR and exit (no simulation)")
    parser.add_argument("--vix-max", type=float, default=float("inf"),
                        help="Skip entry if VIX opens above this level (panic-spike filter)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING"],
                        help="Logging verbosity (default: INFO)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    configure_logging(
        log_file=str(OUTPUT_DIR / "backtest.log"),
        level=args.log_level,
    )

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    if start >= end:
        print(f"Error: --start ({start}) must be before --end ({end})")
        sys.exit(1)

    # Download phase
    if not args.use_cache:
        logger.info("Downloading historical data from IBKR...")
        asyncio.run(download_data(end_date_str=args.end.replace("-", "") + " 16:00:00"))
    else:
        logger.info("--use-cache: skipping IBKR download")

    if args.download_only:
        logger.info("--download-only: data download complete, exiting")
        return

    # Simulation + reporting phase
    run_simulation(start=start, end=end, capital=args.capital, strategy=args.strategy,
                   max_vix=args.vix_max)


if __name__ == "__main__":
    main()
