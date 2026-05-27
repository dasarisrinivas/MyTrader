#!/usr/bin/env python3
"""
Backtest CLI Entrypoint
=======================

Usage:
    python -m backtest.run --symbol MES --start 2024-01-09 --end 2026-01-09 --bar 1m --bar2 5m

This script runs a comprehensive 2-year historical backtest using the exact same
strategy/risk/execution logic as the live trading bot.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger

# Force unbuffered output so progress is visible in real-time
import os
os.environ["PYTHONUNBUFFERED"] = "1"

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {message}")

# Add file logging
log_path = Path("logs/backtest.log")
log_path.parent.mkdir(exist_ok=True)
logger.add(log_path, level="DEBUG", rotation="10 MB")

print("=" * 60, flush=True)
print("  BACKTEST RUNNER STARTING", flush=True)
print("=" * 60, flush=True)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run historical backtest using live trading logic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic 2-year backtest
  python -m backtest.run --symbol MES --start 2024-01-09 --end 2026-01-09
  
  # With custom config file
  python -m backtest.run --config configs/backtest.yaml
  
  # Using SPY proxy data
  python -m backtest.run --symbol MES --start 2024-01-01 --end 2026-01-01 --proxy-mode
  
  # Generate HTML report
  python -m backtest.run --symbol MES --start 2024-01-01 --end 2026-01-01 --report html
"""
    )
    
    # Required arguments
    parser.add_argument(
        "--symbol", "-s",
        default="MES",
        help="Symbol to backtest (default: MES)"
    )
    
    parser.add_argument(
        "--start",
        type=str,
        required=False,
        default="2024-01-09",
        help="Start date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end",
        type=str,
        required=False,
        default="2026-01-09",
        help="End date (YYYY-MM-DD)"
    )
    
    # Data options
    parser.add_argument(
        "--bar", "-b",
        default="1m",
        choices=["1m", "5m", "15m"],
        help="Primary bar size (default: 1m)"
    )
    
    parser.add_argument(
        "--bar2",
        default="5m",
        help="Secondary bar size for MTF analysis (default: 5m)"
    )
    
    parser.add_argument(
        "--session",
        default="full",
        choices=["full", "rth"],
        help="Trading session: full (24h) or rth (regular hours only)"
    )
    
    # Data source options
    parser.add_argument(
        "--data-source",
        default="auto",
        choices=["auto", "databento", "polygon", "ib", "file", "proxy"],
        help="Data source: auto (best available), databento, polygon, ib (Interactive Brokers), file (local), proxy (SPY/yfinance)"
    )
    
    parser.add_argument(
        "--data-file",
        type=str,
        help="Path to local data file (parquet/csv)"
    )
    
    parser.add_argument(
        "--proxy-mode",
        action="store_true",
        help="Use SPY as proxy for ES/MES (labeled in results)"
    )
    
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force re-download data (don't use cache)"
    )
    
    # Configuration
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to backtest config YAML file"
    )
    
    parser.add_argument(
        "--capital",
        type=float,
        default=50000.0,
        help="Initial capital (default: 50000)"
    )
    
    parser.add_argument(
        "--slippage",
        type=float,
        default=1.0,
        help="Slippage in ticks (default: 1.0)"
    )
    
    parser.add_argument(
        "--commission",
        type=float,
        default=2.40,
        help="Commission per contract round-trip (default: 2.40)"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="reports",
        help="Output directory for results (default: reports)"
    )
    
    parser.add_argument(
        "--report",
        default="html",
        choices=["html", "md", "both", "none"],
        help="Report format (default: html)"
    )
    
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Enable decision trace mode for live comparison"
    )
    
    # Feature flags
    parser.add_argument(
        "--no-optimizer",
        action="store_true",
        help="Disable TrendContinuationOptimizer"
    )
    
    parser.add_argument(
        "--no-mtf",
        action="store_true",
        help="Disable multi-timeframe analysis"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    # MAY 27 2026: run the REAL live Trade Manager gate inside the backtest so
    # research == live (closes the parity gap that hid the R:R deadlock for months).
    parser.add_argument(
        "--with-manager",
        action="store_true",
        help="Enable the live trading_manager approval gate (rules.evaluate) in the backtest"
    )
    parser.add_argument(
        "--tm-min-rr",
        type=float,
        default=None,
        help="Override the Trade Manager R:R floor for --with-manager (live default 2.0; strategy needs ~1.2)"
    )

    args = parser.parse_args()
    # Plumb manager flags to the engine via env (engine reads BT_WITH_MANAGER/BT_MIN_RR).
    if getattr(args, "with_manager", False):
        os.environ["BT_WITH_MANAGER"] = "1"
        if args.tm_min_rr is not None:
            os.environ["BT_MIN_RR"] = str(args.tm_min_rr)
    return args


def load_config(config_path: Optional[str]) -> dict:
    """Load configuration from YAML file."""
    if not config_path:
        return {}
    
    config_file = Path(config_path)
    if not config_file.exists():
        logger.warning(f"Config file not found: {config_path}")
        return {}
    
    with open(config_file) as f:
        return yaml.safe_load(f) or {}


async def download_data(args: argparse.Namespace) -> tuple:
    """Download or load data based on arguments."""
    import pandas as pd
    import os
    
    start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    
    df_1m = None
    df_5m = None
    df_15m = None
    df_30m = None
    data_mode = "live"
    
    # Auto-detect local files for 'auto' or 'ib' if not explicitly provided
    if args.data_source in ("auto", "ib") and not args.data_file and not args.no_cache:
        # Check standard locations
        potential_files = [
            Path(f"data/ib/{args.symbol}_1m_1y.parquet"),
            Path(f"data/ib/{args.symbol}_1m.parquet"),
        ]
        
        for p in potential_files:
            if p.exists():
                logger.info(f"📂 Auto-detected local data file: {p}")
                # Switch mode to file to reuse logic below
                args.data_source = "file"
                args.data_file = str(p)
                break
        
        # FEB 2026: If no 1m file found, try 15m file directly
        if not args.data_file:
            potential_15m = [
                Path(f"data/ib/{args.symbol}_15m_1y.parquet"),
                Path(f"data/ib/{args.symbol}_15m.parquet"),
            ]
            for p in potential_15m:
                if p.exists():
                    logger.info(f"📂 Auto-detected 15m data file: {p}")
                    args.data_source = "file"
                    args.data_file = str(p)
                    break

        # FEB 2026: If no 1m/15m file found, try 30m file directly
        if not args.data_file:
            potential_30m = [
                Path(f"data/ib/{args.symbol}_30m_1y.parquet"),
                Path(f"data/ib/{args.symbol}_30m.parquet"),
            ]
            for p in potential_30m:
                if p.exists():
                    logger.info(f"📂 Auto-detected 30m data file: {p}")
                    args.data_source = "file"
                    args.data_file = str(p)
                    break

    # Load from file
    if args.data_source == "file" and args.data_file:
        data_path = Path(args.data_file)
        if data_path.suffix == ".parquet":
            raw = pd.read_parquet(data_path)
            
            # FEB 2026: Detect if the loaded file is 15m data (not 1m)
            if "15m" in data_path.name:
                # This IS 15m data — load directly, don't treat as 1m
                df_15m = raw
                if "timestamp" in df_15m.columns:
                    df_15m["timestamp"] = pd.to_datetime(df_15m["timestamp"])
                    df_15m.set_index("timestamp", inplace=True)
                if df_15m.index.tzinfo is None:
                    df_15m.index = df_15m.index.tz_localize("UTC")
                # df_1m stays None — engine.load_15m_only will handle it
                logger.info(f"Loaded {len(df_15m)} 15m bars from {data_path}")
                return df_1m, df_5m, df_15m, df_30m, data_mode

            # FEB 2026: Detect if the loaded file is 30m data
            if "30m" in data_path.name:
                df_30m = raw
                if "timestamp" in df_30m.columns:
                    df_30m["timestamp"] = pd.to_datetime(df_30m["timestamp"])
                    df_30m.set_index("timestamp", inplace=True)
                if df_30m.index.tzinfo is None:
                    df_30m.index = df_30m.index.tz_localize("UTC")
                logger.info(f"Loaded {len(df_30m)} 30m bars from {data_path}")
                return df_1m, df_5m, df_15m, df_30m, data_mode
            
            # Otherwise treat as 1m data (legacy path)
            df_1m = raw
            
            # JAN 17 2026: Try to load matching 15m/30m files if available
            # Pattern: ES_1m_1y.parquet -> ES_15m_1y.parquet
            if "1m" in data_path.name:
                path_15m = data_path.with_name(data_path.name.replace("1m", "15m"))
                if path_15m.exists():
                    logger.info(f"Loading paired 15m data: {path_15m}")
                    df_15m = pd.read_parquet(path_15m)
                    if "timestamp" in df_15m.columns:
                        df_15m["timestamp"] = pd.to_datetime(df_15m["timestamp"])
                        df_15m.set_index("timestamp", inplace=True)
                    if df_15m.index.tzinfo is None:
                        df_15m.index = df_15m.index.tz_localize("UTC")
                
                path_30m = data_path.with_name(data_path.name.replace("1m", "30m"))
                if path_30m.exists():
                    logger.info(f"Loading paired 30m data: {path_30m}")
                    df_30m = pd.read_parquet(path_30m)
                    if "timestamp" in df_30m.columns:
                        df_30m["timestamp"] = pd.to_datetime(df_30m["timestamp"])
                        df_30m.set_index("timestamp", inplace=True)
                    if df_30m.index.tzinfo is None:
                        df_30m.index = df_30m.index.tz_localize("UTC")
        else:
            df_1m = pd.read_csv(data_path)
        
        # Ensure datetime index
        if "timestamp" in df_1m.columns:
            df_1m["timestamp"] = pd.to_datetime(df_1m["timestamp"])
            df_1m.set_index("timestamp", inplace=True)
        
        if df_1m.index.tzinfo is None:
            df_1m.index = df_1m.index.tz_localize("UTC")
        
        logger.info(f"Loaded {len(df_1m)} bars from {data_path}")
        return df_1m, df_5m, df_15m, df_30m, data_mode
    
    # Use SPY proxy
    if args.data_source == "proxy" or args.proxy_mode:
        from .data.ib_downloader import FallbackDataDownloader
        
        logger.warning("⚠️  PROXY MODE: Using SPY as ES/MES proxy")
        downloader = FallbackDataDownloader()
        
        df_1m, data_mode = downloader.download_spy_proxy(
            start=start_date,
            end=end_date,
            bar_size=args.bar.replace("m", " min")
        )
        
        return df_1m, df_5m, df_15m, df_30m, data_mode
    
    # Use Databento
    if args.data_source in ("databento", "auto") and os.environ.get("DATABENTO_API_KEY"):
        try:
            from .data.ib_downloader import DatabentoDownloader
            
            logger.info("📊 Using Databento for historical futures data")
            downloader = DatabentoDownloader()
            
            # Convert bar size to Databento format
            bar_size_db = args.bar.replace("m", " min")
            if bar_size_db == "5 min":
                bar_size_db = "5 mins"
            
            df_1m = downloader.download(
                symbol=args.symbol,
                start=start_date,
                end=end_date,
                bar_size=bar_size_db
            )
            
            # Download secondary timeframe if MTF enabled
            if not args.no_mtf and args.bar2:
                bar2_db = args.bar2.replace("m", " min")
                if bar2_db != "1 min":
                    bar2_db = bar2_db + "s"  # e.g., "5 mins"
                
                df_5m = downloader.download(
                    symbol=args.symbol,
                    start=start_date,
                    end=end_date,
                    bar_size=bar2_db
                )
            
            data_mode = "databento"
            return df_1m, df_5m, df_15m, df_30m, data_mode
            
        except Exception as e:
            logger.warning(f"Databento failed: {e}")
            if args.data_source == "databento":
                raise  # User explicitly requested Databento
            logger.info("Falling back to next available source...")
    
    # Use Polygon
    if args.data_source in ("polygon", "auto") and os.environ.get("POLYGON_API_KEY"):
        try:
            from .data.ib_downloader import PolygonDownloader
            
            logger.info("📊 Using Polygon for historical futures data")
            downloader = PolygonDownloader()
            
            # Convert bar size
            bar_size_poly = args.bar.replace("m", " min")
            if bar_size_poly == "5 min":
                bar_size_poly = "5 mins"
            
            df_1m = downloader.download(
                symbol=args.symbol,
                start=start_date,
                end=end_date,
                bar_size=bar_size_poly
            )
            
            # Download secondary timeframe if MTF enabled
            if not args.no_mtf and args.bar2:
                bar2_poly = args.bar2.replace("m", " min")
                if bar2_poly != "1 min":
                    bar2_poly = bar2_poly + "s"
                
                df_5m = downloader.download(
                    symbol=args.symbol,
                    start=start_date,
                    end=end_date,
                    bar_size=bar2_poly
                )
            
            data_mode = "polygon"
            return df_1m, df_5m, df_15m, df_30m, data_mode
            
        except Exception as e:
            logger.warning(f"Polygon failed: {e}")
            if args.data_source == "polygon":
                raise  # User explicitly requested Polygon
            logger.info("Falling back to next available source...")
    
    # Download from IB
    if args.data_source in ("ib", "auto"):
        # ... logic omitted for brevity in thought but needs to be in replace ...
        # I'll rely on old string matching
        pass

    
    # Download from IB
    if args.data_source in ("ib", "auto"):
        from .data.ib_downloader import IBHistoricalDownloader, DownloadConfig
        
        config = DownloadConfig()
        downloader = IBHistoricalDownloader(config)
        
        try:
            # Download 1m data using continuous futures for long date ranges
            bar_size_ib = args.bar.replace("m", " min").replace("1 min", "1 min")
            
            logger.info(f"Downloading {args.symbol} {bar_size_ib} continuous futures data from IB...")
            
            # Use continuous download for futures to handle historical contracts
            if args.symbol in ("MES", "ES", "NQ", "MNQ", "RTY", "M2K"):
                df_1m = await downloader.download_continuous(
                    symbol=args.symbol,
                    start=start_date,
                    end=end_date,
                    bar_size=bar_size_ib,
                    use_cache=not args.no_cache
                )
            else:
                df_1m = await downloader.download(
                    symbol=args.symbol,
                    start=start_date,
                    end=end_date,
                    bar_size=bar_size_ib,
                    use_cache=not args.no_cache
                )
            
            # Download 5m data if MTF enabled
            if not args.no_mtf and args.bar2:
                # IB requires plural "mins" for bar sizes > 1 min
                bar2_ib = args.bar2.replace("m", " mins") if args.bar2 != "1m" else "1 min"
                logger.info(f"Downloading {args.symbol} {bar2_ib} data from IB...")
                
                if args.symbol in ("MES", "ES", "NQ", "MNQ", "RTY", "M2K"):
                    df_5m = await downloader.download_continuous(
                        symbol=args.symbol,
                        start=start_date,
                        end=end_date,
                        bar_size=bar2_ib,
                        use_cache=not args.no_cache
                    )
                else:
                    df_5m = await downloader.download(
                    symbol=args.symbol,
                    start=start_date,
                    end=end_date,
                    bar_size=bar2_ib,
                    use_cache=not args.no_cache
                )
            
            # Validate data reception to trigger fallback if empty
            if df_1m is None or df_1m.empty:
                raise ValueError(f"IB returned no data for {args.symbol} (likely due to contract expiration/stitching issues)")
            
            data_mode = "ib"
            return df_1m, df_5m, df_15m, df_30m, data_mode
            
        except Exception as e:
            logger.error(f"IB download failed: {e}")
            if args.data_source == "ib":
                raise  # User explicitly requested IB
            logger.info("Falling back to SPY proxy...")
            
        finally:
            downloader.disconnect()
    
    # Final fallback: SPY proxy
    from .data.ib_downloader import FallbackDataDownloader
    logger.warning("⚠️  PROXY MODE: Using SPY as ES/MES proxy (fallback)")
    downloader = FallbackDataDownloader()
    df_1m, data_mode = downloader.download_spy_proxy(
        start=start_date,
        end=end_date
    )
    
    return df_1m, df_5m, df_15m, df_30m, data_mode



def run_backtest(df_1m, df_5m, df_15m, df_30m, args: argparse.Namespace, config: dict):
    """Run the backtest engine."""
    from .engine import BacktestEngine, BacktestConfig
    from shree.config import OneMinuteStrategyConfig, TradingConfig
    from shree.risk.risk_gate import RiskGateConfig  # Use the one from risk_gate module
    
    start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    
    # Build strategy config
    strategy_config = OneMinuteStrategyConfig()
    if "strategy" in config:
        for key, value in config["strategy"].items():
            if hasattr(strategy_config, key):
                setattr(strategy_config, key, value)
    
    # FEB 2026: Only apply 1m-specific tuning when NOT using 15m strategy
    if not getattr(strategy_config, 'use_15m_strategy', False):
        # TUNING (Jan 18 2026): Defaults for RTH Trading per user request (More Trades)
        # FEB 6 2026 FIX: Only set these if NOT already configured via YAML.
        if "strategy" not in config or "stop_atr_multiplier" not in config.get("strategy", {}):
            strategy_config.stop_atr_multiplier = 3.0
        if "strategy" not in config or "take_profit_multiple" not in config.get("strategy", {}):
            strategy_config.take_profit_multiple = 1.0
        if "strategy" not in config or "trend_adx_threshold" not in config.get("strategy", {}):
            strategy_config.trend_adx_threshold = 18.0
        # Ensure MTF is enabled to use wider 15m ATR for stops (avoid RiskGate minimums)
        strategy_config.use_mtf_regime = True
        
        # JAN 17 2026 FIX: WARMUP LOCK
        # window_bars must be >= warmup_bars (800) otherwise generation exits with "WARMUP"
        strategy_config.window_bars = 1000
    else:
        # FEB 7 2026: Wire ft_max_hold_bars → max_hold_minutes for 15m strategy
        # ft_max_hold_bars is in 15m bars, so multiply by 15 to get minutes
        ft_bars = getattr(strategy_config, 'ft_max_hold_bars', 6)
        strategy_config.max_hold_minutes = ft_bars * 15
        logger.info(f"15m max_hold: {ft_bars} bars × 15 min = {strategy_config.max_hold_minutes} min")
    
    # Ensure Risk Gate allows this - configured below in RiskGateConfig section
    
    # Build trading config
    trading_config = TradingConfig()
    if "trading" in config:
        for key, value in config["trading"].items():
            if hasattr(trading_config, key):
                setattr(trading_config, key, value)
    
    # Build risk gate config
    risk_gate_config = RiskGateConfig()
    
    # FEB 2026: Only apply 1m-specific risk tuning when NOT using 15m strategy
    if not getattr(strategy_config, 'use_15m_strategy', False):
        # TUNING (Jan 18 2026): Increase hard caps for 1m High ATR strategy
        risk_gate_config.max_stop_points = 50.0
        risk_gate_config.risk_per_trade_max = 250.0
        risk_gate_config.risk_per_trade_usd = 250.0
        risk_gate_config.min_stop_points = 3.0
    
    if "risk_gate" in config:
        for key, value in config["risk_gate"].items():
            if hasattr(risk_gate_config, key):
                setattr(risk_gate_config, key, value)
    
    # Create backtest config
    bt_config = BacktestConfig(
        symbol=args.symbol,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.capital,
        slippage_ticks=args.slippage,
        commission_per_contract=args.commission,
        session_type=args.session,
        strategy_config=strategy_config,
        trading_config=trading_config,
        risk_gate_config=risk_gate_config,
        enable_trend_optimizer=not args.no_optimizer,
        trace_mode=args.trace,
    )
    
    # Run backtest
    engine = BacktestEngine(bt_config)
    
    # FEB 2026: Strategy dispatch — 30m, 15m, or legacy 1m
    if getattr(strategy_config, 'use_30m_strategy', False):
        if df_30m is None or df_30m.empty:
            # Try to resample from 1m data if available
            if df_1m is not None and not df_1m.empty:
                logger.info("No 30m data found, resampling from 1m...")
                engine.load_data(df_1m, df_5m, df_15m, df_30m)
            else:
                raise ValueError("30m data required for use_30m_strategy but none loaded")
        else:
            engine.load_30m_only(df_30m)
        logger.info("Running 30m overnight backtest...")
        results = engine.run_30m_only()
    elif strategy_config.use_15m_strategy:
        if df_15m is None or df_15m.empty:
            raise ValueError("15m data required for use_15m_strategy but none loaded")
        engine.load_15m_only(df_15m)
        logger.info("Running 15m-only backtest...")
        results = engine.run_15m_only()
    else:
        engine.load_data(df_1m, df_5m, df_15m, df_30m)
        logger.info("Running backtest...")
        results = engine.run()
    
    return results


def analyze_and_report(results: dict, args: argparse.Namespace, data_mode: str):
    """Analyze results and generate reports."""
    from .analysis import BacktestAnalyzer
    from .report import ReportGenerator, ReportConfig
    import json
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze
    analyzer = BacktestAnalyzer(
        trades=results.get("trades", []),
        equity_curve=results.get("equity_curve"),
        block_reasons=results.get("block_reasons", {}),
        optimizer_stats=results.get("optimizer_stats", {}),
        initial_capital=args.capital,
    )
    
    analysis = analyzer.analyze()
    
    # Add data mode warning to learnings
    if data_mode == "proxy_spy":
        analysis["learnings"]["regime_notes"].insert(
            0, "⚠️ PROXY MODE: Results based on SPY, not actual ES/MES futures. "
               "Actual futures may have different behavior during overnight sessions."
        )
    
    # Save analysis JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"backtest_{args.symbol}_{args.start}_{args.end}_{timestamp}"
    
    with open(output_dir / f"{base_name}_metrics.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    
    # Save trades CSV
    if results.get("trades"):
        import pandas as pd
        trades_df = pd.DataFrame(results["trades"])
        trades_df.to_csv(output_dir / f"{base_name}_trades.csv", index=False)
    
    # Save equity curve
    if results.get("equity_curve") is not None and not results["equity_curve"].empty:
        results["equity_curve"].to_csv(output_dir / f"{base_name}_equity.csv")
    
    # Generate reports
    if args.report != "none":
        report_config = ReportConfig(
            title=f"Backtest Report: {args.symbol}",
            symbol=args.symbol,
            strategy_name="MES 1-Minute Trend Strategy",
        )
        
        generator = ReportGenerator(
            analysis=analysis,
            equity_curve=results.get("equity_curve"),
            trades=results.get("trades", []),
            config=report_config
        )
        
        if args.report in ("html", "both"):
            html_path = output_dir / f"{base_name}.html"
            report_config.format = "html"
            generator.config = report_config
            generator.generate(html_path)
            logger.info(f"📊 HTML report saved: {html_path}")
        
        if args.report in ("md", "both"):
            md_path = output_dir / f"{base_name}.md"
            report_config.format = "md"
            generator.config = report_config
            generator.generate(md_path)
            logger.info(f"📝 Markdown report saved: {md_path}")
    
    # Print summary to console
    metrics = analysis.get("metrics", {})
    print("\n" + "=" * 60)
    print(f"BACKTEST SUMMARY: {args.symbol} ({args.start} to {args.end})")
    print("=" * 60)
    print(f"  Total Return:    {metrics.get('total_return', 0)*100:+.2f}%")
    print(f"  Total P&L:       ${metrics.get('total_pnl', 0):,.2f}")
    print(f"  Sharpe Ratio:    {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  Max Drawdown:    {metrics.get('max_drawdown', 0)*100:.2f}%")
    print(f"  Win Rate:        {metrics.get('win_rate', 0)*100:.1f}%")
    print(f"  Profit Factor:   {metrics.get('profit_factor', 0):.2f}")
    print(f"  Total Trades:    {metrics.get('total_trades', 0)}")
    print("=" * 60)
    
    if data_mode == "proxy_spy":
        print("⚠️  WARNING: Results based on SPY proxy data")
    
    print(f"\nResults saved to: {output_dir}")
    
    return analysis


async def main():
    """Main entry point."""
    args = parse_args()
    
    if args.verbose:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG", format="<level>{time:HH:mm:ss}</level> | <level>{message}</level>")
    
    logger.info(f"Starting backtest: {args.symbol} from {args.start} to {args.end}")
    
    # Load config
    config = load_config(args.config)
    
    # Download/load data
    logger.info("Loading data...")
    df_1m, df_5m, df_15m, df_30m, data_mode = await download_data(args)
    
    # FEB 2026: Check available data — 1m, 15m, or 30m
    has_1m = df_1m is not None and not df_1m.empty
    has_15m = df_15m is not None and not df_15m.empty
    has_30m = df_30m is not None and not df_30m.empty
    
    if not has_1m and not has_15m and not has_30m:
        logger.error("No data available (neither 1m, 15m, nor 30m). Exiting.")
        sys.exit(1)
    
    if has_1m:
        logger.info(f"Data loaded: {len(df_1m)} 1m bars ({data_mode} mode)")
    if has_15m:
        logger.info(f"Data loaded: {len(df_15m)} 15m bars ({data_mode} mode)")
    if has_30m:
        logger.info(f"Data loaded: {len(df_30m)} 30m bars ({data_mode} mode)")
    
    # Run backtest
    results = run_backtest(df_1m, df_5m, df_15m, df_30m, args, config)
    
    # Analyze and report
    analyze_and_report(results, args, data_mode)
    
    logger.info("Backtest complete!")


if __name__ == "__main__":
    asyncio.run(main())
