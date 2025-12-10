"""Simulate Replay - Replay historical data through hybrid decision engine.

This script replays yesterday's (or specified date's) market data through
the hybrid decision engine to validate behavior without risking capital.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import yaml

from mytrader.hybrid import HybridDecisionEngine, DeterministicEngine, HeuristicEngine
from mytrader.hybrid.confidence import ConfidenceScorer
from mytrader.hybrid.safety import SafetyManager
from mytrader.hybrid.decision_logger import DecisionLogger
from mytrader.features.feature_engineer import engineer_features
from mytrader.utils.logger import configure_logging, logger


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_historical_data(data_path: str, date: str) -> pd.DataFrame:
    """Load historical data for a specific date.
    
    Args:
        data_path: Path to historical data file
        date: Date string (YYYY-MM-DD)
        
    Returns:
        DataFrame with OHLCV data
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Parse timestamp column
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
    
    # Filter to specified date
    if date:
        target_date = datetime.strptime(date, "%Y-%m-%d").date()
        df = df[df.index.date == target_date]
    
    return df


def run_simulation(
    data: pd.DataFrame,
    config: dict,
    dry_run: bool = True,
    report_dir: str = "reports",
) -> dict:
    """Run simulation through hybrid decision engine.
    
    Args:
        data: Historical OHLCV data
        config: Configuration dictionary
        dry_run: Run in dry-run mode (always True for simulation)
        report_dir: Directory for reports
        
    Returns:
        Simulation results dictionary
    """
    logger.info(f"Starting simulation with {len(data)} bars")
    
    # Initialize engine components
    hybrid_config = config.get("hybrid", {})
    
    d_engine = DeterministicEngine(config=hybrid_config.get("d_engine"))
    
    h_engine = HeuristicEngine(
        llm_client=None,  # No LLM in simulation (or use mock)
        rag_storage=None,
        config=hybrid_config.get("h_engine"),
    )
    
    confidence_scorer = ConfidenceScorer(
        weights=hybrid_config.get("confidence", {}).get("weights"),
        confidence_threshold=hybrid_config.get("confidence", {}).get("threshold", 0.60),
    )
    
    safety_manager = SafetyManager(
        cooldown_minutes=hybrid_config.get("safety", {}).get("order_cooldown_minutes", 5),
        max_orders_per_window=hybrid_config.get("safety", {}).get("max_orders_per_15min", 3),
        dry_run=True,  # Always dry-run in simulation
    )
    
    decision_logger = DecisionLogger(
        log_dir=report_dir,
        json_file="simulation_decisions.json",
        csv_file="simulation_decisions.csv",
    )
    
    engine = HybridDecisionEngine(
        d_engine=d_engine,
        h_engine=h_engine,
        confidence_scorer=confidence_scorer,
        safety_manager=safety_manager,
        decision_logger=decision_logger,
        config=hybrid_config,
    )
    
    # Engineer features
    features = engineer_features(data[["open", "high", "low", "close", "volume"]], None)
    
    if features.empty:
        logger.error("Feature engineering returned empty DataFrame")
        return {"error": "Feature engineering failed"}
    
    logger.info(f"Features engineered: {len(features)} rows, {len(features.columns)} columns")
    
    # Calculate price levels from data
    if len(features) > 0:
        pdh = float(features["high"].max())
        pdl = float(features["low"].min())
        engine.set_price_levels(pdh=pdh, pdl=pdl)
        logger.info(f"Price levels: PDH={pdh:.2f}, PDL={pdl:.2f}")
    
    # Simulation results tracking
    results = {
        "date": data.index[0].strftime("%Y-%m-%d") if len(data) > 0 else "unknown",
        "bars_processed": 0,
        "candidates": 0,
        "decisions": [],
        "trades": [],
        "h_engine_calls": 0,
        "errors": [],
    }
    
    # Process each bar
    min_bars = 50  # Minimum bars for feature calculation
    
    for i in range(min_bars, len(features)):
        try:
            # Get feature window
            feature_window = features.iloc[:i+1]
            current_price = float(feature_window.iloc[-1]["close"])
            candle_time = feature_window.index[-1]
            
            # Evaluate
            decision = engine.evaluate(
                features=feature_window,
                current_price=current_price,
                candle_time=candle_time,
            )
            
            results["bars_processed"] += 1
            
            if decision.d_signal and decision.d_signal.is_candidate:
                results["candidates"] += 1
            
            if decision.should_execute:
                results["trades"].append({
                    "timestamp": candle_time.isoformat(),
                    "action": decision.action,
                    "price": current_price,
                    "confidence": decision.final_confidence,
                    "stop_loss": decision.stop_loss,
                    "take_profit": decision.take_profit,
                })
            
            # Progress logging every 100 bars
            if results["bars_processed"] % 100 == 0:
                logger.info(f"Processed {results['bars_processed']}/{len(features)-min_bars} bars")
                
        except Exception as e:
            logger.error(f"Error at bar {i}: {e}")
            results["errors"].append(str(e))
    
    # Get final stats
    stats = engine.get_stats()
    results["h_engine_calls"] = stats.get("h_engine_calls", 0)
    results["final_stats"] = stats
    
    # Save report
    report_path = Path(report_dir) / f"simulation_report_{results['date']}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Simulation complete. Report saved to {report_path}")
    
    return results


def print_summary(results: dict):
    """Print simulation summary."""
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    print(f"Date: {results.get('date', 'unknown')}")
    print(f"Bars processed: {results.get('bars_processed', 0)}")
    print(f"Candidates generated: {results.get('candidates', 0)}")
    print(f"Trades (would execute): {len(results.get('trades', []))}")
    print(f"H-engine calls: {results.get('h_engine_calls', 0)}")
    print(f"Errors: {len(results.get('errors', []))}")
    
    if results.get("trades"):
        print("\nTrades:")
        for trade in results["trades"][:10]:  # Show first 10
            print(f"  {trade['timestamp']}: {trade['action']} @ {trade['price']:.2f} (conf={trade['confidence']:.2f})")
    
    if results.get("errors"):
        print(f"\nFirst 5 errors:")
        for err in results["errors"][:5]:
            print(f"  - {err}")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Simulate replay of historical data")
    parser.add_argument("--date", type=str, help="Date to replay (YYYY-MM-DD), default: yesterday")
    parser.add_argument("--data", type=str, default="data/es_historical.csv", help="Path to historical data")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--report-dir", type=str, default="reports", help="Directory for reports")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Run in dry-run mode (always true)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = "DEBUG" if args.verbose else "INFO"
    configure_logging(log_file="logs/simulation.log", level=log_level, serialize=False)
    
    # Default to yesterday
    if not args.date:
        args.date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    logger.info(f"Simulation replay for {args.date}")
    
    try:
        # Load config
        config = load_config(args.config)
        
        # Load data
        data = load_historical_data(args.data, args.date)
        
        if data.empty:
            logger.error(f"No data found for {args.date}")
            print(f"ERROR: No data found for {args.date}")
            sys.exit(1)
        
        # Run simulation
        results = run_simulation(
            data=data,
            config=config,
            dry_run=args.dry_run,
            report_dir=args.report_dir,
        )
        
        # Print summary
        print_summary(results)
        
        # Exit code based on errors
        if results.get("errors"):
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        print(f"ERROR: Simulation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
