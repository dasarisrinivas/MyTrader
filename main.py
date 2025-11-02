"""Entry point for MyTrader."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import pandas as pd

from mytrader.backtesting.engine import BacktestingEngine
from mytrader.config import Settings
from mytrader.data.ibkr import IBKRCollector
from mytrader.data.pipeline import MarketDataPipeline
from mytrader.data.sentiment import TwitterSentimentCollector
from mytrader.data.tradingview import TradingViewCollector
from mytrader.execution.ib_executor import TradeExecutor
from mytrader.features.feature_engineer import engineer_features
from mytrader.monitoring.live_tracker import LivePerformanceTracker
from mytrader.optimization.optimizer import ParameterOptimizer
from mytrader.risk.manager import RiskManager
from mytrader.strategies.engine import StrategyEngine
from mytrader.strategies.momentum_reversal import MomentumReversalStrategy
from mytrader.strategies.rsi_macd_sentiment import RsiMacdSentimentStrategy
from mytrader.utils.logger import configure_logging, logger
from mytrader.utils.settings_loader import load_settings

try:
    from ib_insync import IB
except ImportError as exc:  # noqa: BLE001
    raise SystemExit("ib_insync is required. Install with pip install ib_insync") from exc


async def run_live(settings: Settings) -> None:
    """Run live trading with enhanced monitoring and risk management."""
    configure_logging(level="INFO")

    # Initialize data collectors with reconnection support
    ib_collector = IBKRCollector(
        host=settings.data.ibkr_host,
        port=settings.data.ibkr_port,
        client_id=settings.data.ibkr_client_id,
        symbol=settings.data.ibkr_symbol,
        exchange=settings.data.ibkr_exchange,
        currency=settings.data.ibkr_currency,
        max_retries=5,
        base_delay=1.0
    )

    collectors = [
        ib_collector,
        TradingViewCollector(
            settings.data.tradingview_webhook_url or "http://localhost:8000", 
            settings.data.tradingview_symbol, 
            settings.data.tradingview_interval,
            max_retries=3,
            rate_limit_delay=1.0
        ),
    ]

    if settings.data.twitter_bearer_token:
        collectors.append(TwitterSentimentCollector(
            settings.data.twitter_bearer_token, 
            ["SPY", "ES", "S&P 500"],
            max_retries=3
        ))

    pipeline = MarketDataPipeline(collectors)
    strategies = [RsiMacdSentimentStrategy(), MomentumReversalStrategy()]
    optimizer = ParameterOptimizer(strategies)
    engine = StrategyEngine(strategies)
    
    # Initialize risk manager with Kelly Criterion support
    risk = RiskManager(settings.trading, position_sizing_method="kelly")
    
    # Initialize live performance tracker
    tracker = LivePerformanceTracker(
        initial_capital=settings.trading.initial_capital,
        risk_free_rate=settings.backtest.risk_free_rate
    )

    ib = IB()
    executor = TradeExecutor(ib, settings.trading, settings.data.ibkr_symbol)
    await executor.connect(settings.data.ibkr_host, settings.data.ibkr_port, settings.data.ibkr_client_id)

    account_value = settings.trading.initial_capital
    step = 0
    opt_interval = max(1, settings.optimization.retrain_interval // 60)
    status_log_interval = 60  # Log performance every 60 iterations

    try:
        async for snapshot in pipeline.stream():
            try:
                # Validate data
                if not {"open", "high", "low", "close", "volume"}.issubset(snapshot.columns):
                    continue
                price_df = snapshot[["open", "high", "low", "close", "volume"]].dropna(subset=["open", "high", "low", "close", "volume"], how="any")
                if price_df.empty:
                    continue
                    
                sentiment_cols = [c for c in snapshot.columns if "sentiment" in c]
                sentiment_df = snapshot[sentiment_cols].dropna(how="all") if sentiment_cols else None
                
                # Engineer features with enhanced indicators
                features = engineer_features(price_df, sentiment_df)
                if features.empty or len(features) < 50:
                    continue

                returns = features["close"].pct_change().dropna()

                # Periodic optimization
                if (
                    settings.optimization.parameter_grid
                    and step % opt_interval == 0
                    and len(features) >= settings.optimization.window_length
                ):
                    window = features.tail(settings.optimization.window_length)
                    result = optimizer.optimize(window, settings.optimization.parameter_grid)
                    logger.info("Optimizer applied params %s (score %.4f)", result.best_params, result.best_score)

                step += 1

                # Generate trading signal
                signal = engine.evaluate(features, returns)
                logger.info("Signal %s confidence %.2f", signal.action, signal.confidence)
                
                if signal.action == "HOLD":
                    continue

                # Get current position and update tracker
                current_position = await executor.get_current_position()
                last_price = float(features.iloc[-1]["close"])
                
                if current_position:
                    unrealized_pnl = await executor.get_unrealized_pnl()
                    tracker.update_equity(last_price, realized_pnl=0.0)

                # Position sizing with Kelly Criterion
                risk_stats = risk.get_statistics()
                qty = risk.position_size(
                    account_value, 
                    signal.confidence,
                    win_rate=risk_stats.get("win_rate"),
                    avg_win=risk_stats.get("avg_win"),
                    avg_loss=risk_stats.get("avg_loss")
                )
                
                if not risk.can_trade(qty):
                    logger.warning("Risk limits exceeded. Skipping trade.")
                    continue

                # Calculate ATR-based dynamic stops if ATR available
                atr = features.iloc[-1].get("ATR_14")
                if atr and atr > 0:
                    stop_loss, take_profit = risk.calculate_dynamic_stops(
                        entry_price=last_price,
                        current_atr=atr,
                        direction="long" if signal.action == "BUY" else "short",
                        atr_multiplier=2.0
                    )
                else:
                    # Fallback to fixed tick stops
                    tick = settings.trading.tick_size
                    stop_distance = settings.trading.stop_loss_ticks * tick
                    target_distance = settings.trading.take_profit_ticks * tick

                    if signal.action == "BUY":
                        stop_loss = last_price - stop_distance
                        take_profit = last_price + target_distance
                    else:
                        stop_loss = last_price + stop_distance
                        take_profit = last_price - target_distance

                # Place order with bracket orders
                result = await executor.place_order(
                    action=signal.action,
                    quantity=qty,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                )
                
                logger.info("Order result status %s, filled=%d @ %.2f", 
                           result.status, result.filled_quantity, 
                           result.fill_price if result.fill_price else 0.0)
                
                if result.status not in {"Cancelled", "Inactive"}:
                    risk.register_trade()
                    # Record trade in tracker
                    if result.fill_price:
                        tracker.record_trade(
                            action=signal.action,
                            price=result.fill_price,
                            quantity=qty
                        )
                
                # Periodic performance logging
                if step % status_log_interval == 0:
                    tracker.log_status()
                    
            except Exception as exc:  # noqa: BLE001
                logger.exception("Live loop error: %s", exc)
                await asyncio.sleep(1)
                
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
        # Export final performance snapshot
        tracker.export_snapshot("reports/final_performance.json")
        logger.info("Final performance snapshot saved")
    finally:
        if ib.isConnected():
            ib.disconnect()


def run_backtest(settings: Settings, data_path: Path | None) -> None:
    """Run backtest with enhanced performance metrics and reporting."""
    configure_logging(level="INFO")
    
    # Load data
    path = data_path or settings.backtest.data_path
    if isinstance(path, str):
        path = Path(path)
    
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    
    # Initialize strategies and engine
    strategies = [RsiMacdSentimentStrategy(), MomentumReversalStrategy()]
    engine = BacktestingEngine(strategies, settings.trading, settings.backtest)
    
    # Run backtest
    logger.info("Starting backtest on %d bars...", len(df))
    result = engine.run(df)
    
    # Log comprehensive metrics
    logger.info("=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)
    for key, value in result.metrics.items():
        if isinstance(value, float):
            logger.info("  %s: %.4f", key.replace("_", " ").title(), value)
        else:
            logger.info("  %s: %s", key.replace("_", " ").title(), value)
    logger.info("=" * 60)
    
    # Export detailed report
    from mytrader.backtesting.performance import export_report
    
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    
    # Export JSON report
    json_path = output_dir / "backtest_report.json"
    export_report(result.metrics, result.trades, json_path, format="json")
    logger.info("Detailed report saved to %s", json_path)
    
    # Export CSV report
    csv_path = output_dir / "backtest_report.csv"
    export_report(result.metrics, result.trades, csv_path, format="csv")
    logger.info("CSV reports saved to %s", output_dir)
    
    # Export equity curve
    if not result.equity_curve.empty:
        equity_path = output_dir / "equity_curve.csv"
        result.equity_curve.to_csv(equity_path)
        logger.info("Equity curve saved to %s", equity_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MyTrader autonomous trading bot")
    parser.add_argument("mode", choices=["live", "backtest", "optimize"], help="Execution mode")
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML config")
    parser.add_argument("--data", type=Path, default=None, help="Historical data path for backtest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    if args.mode == "live":
        asyncio.run(run_live(settings))
    elif args.mode == "backtest":
        run_backtest(settings, args.data)
    elif args.mode == "optimize":
        raise NotImplementedError("Optimization CLI not yet implemented")


if __name__ == "__main__":
    main()
