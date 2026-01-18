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

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO", format="<level>{time:HH:mm:ss}</level> | <level>{message}</level>")

# Add file logging
log_path = Path("logs/backtest.log")
log_path.parent.mkdir(exist_ok=True)
logger.add(log_path, level="DEBUG", rotation="10 MB")


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
    
    return parser.parse_args()


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
    data_mode = "live"
    
    # Load from file
    if args.data_source == "file" and args.data_file:
        data_path = Path(args.data_file)
        if data_path.suffix == ".parquet":
            df_1m = pd.read_parquet(data_path)
        else:
            df_1m = pd.read_csv(data_path)
        
        # Ensure datetime index
        if "timestamp" in df_1m.columns:
            df_1m["timestamp"] = pd.to_datetime(df_1m["timestamp"])
            df_1m.set_index("timestamp", inplace=True)
        
        if df_1m.index.tzinfo is None:
            df_1m.index = df_1m.index.tz_localize("UTC")
        
        logger.info(f"Loaded {len(df_1m)} bars from {data_path}")
        return df_1m, df_5m, data_mode
    
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
        
        return df_1m, df_5m, data_mode
    
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
            return df_1m, df_5m, data_mode
            
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
            return df_1m, df_5m, data_mode
            
        except Exception as e:
            logger.warning(f"Polygon failed: {e}")
            if args.data_source == "polygon":
                raise  # User explicitly requested Polygon
            logger.info("Falling back to next available source...")
    
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
            
            data_mode = "ib"
            return df_1m, df_5m, data_mode
            
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
    
    return df_1m, df_5m, data_mode


def run_backtest(df_1m, df_5m, args: argparse.Namespace, config: dict):
    """Run the backtest engine."""
    from .engine import BacktestEngine, BacktestConfig
    from mytrader.config import OneMinuteStrategyConfig, TradingConfig
    from mytrader.risk.risk_gate import RiskGateConfig  # Use the one from risk_gate module
    
    start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    
    # Build strategy config
    strategy_config = OneMinuteStrategyConfig()
    if "strategy" in config:
        for key, value in config["strategy"].items():
            if hasattr(strategy_config, key):
                setattr(strategy_config, key, value)
    
    # Build trading config
    trading_config = TradingConfig()
    if "trading" in config:
        for key, value in config["trading"].items():
            if hasattr(trading_config, key):
                setattr(trading_config, key, value)
    
    # Build risk gate config
    risk_gate_config = RiskGateConfig()
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
    engine.load_data(df_1m, df_5m)
    
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
    df_1m, df_5m, data_mode = await download_data(args)
    
    if df_1m is None or df_1m.empty:
        logger.error("No data available. Exiting.")
        sys.exit(1)
    
    logger.info(f"Data loaded: {len(df_1m)} bars ({data_mode} mode)")
    
    # Run backtest
    results = run_backtest(df_1m, df_5m, args, config)
    
    # Analyze and report
    analyze_and_report(results, args, data_mode)
    
    logger.info("Backtest complete!")


if __name__ == "__main__":
    asyncio.run(main())
