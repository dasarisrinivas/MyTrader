"""Entry point for MyTrader."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import nest_asyncio
import pandas as pd

# Apply nest_asyncio to allow nested event loops (required for ib_insync)
nest_asyncio.apply()

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
from mytrader.strategies.multi_strategy import MultiStrategy
from mytrader.utils.logger import configure_logging, logger
from mytrader.utils.settings_loader import load_settings

try:
    from ib_insync import IB
except ImportError as exc:  # noqa: BLE001
    raise SystemExit("ib_insync is required. Install with pip install ib_insync") from exc


async def run_live(settings: Settings) -> None:
    """Run live trading with enhanced monitoring and risk management."""
    configure_logging(level="INFO")

    # For actual live trading, we use a simpler approach:
    # 1. Connect to IBKR via executor
    # 2. Get real-time quotes from IBKR
    # 3. Build feature history from live bars
    # 4. Generate signals and execute trades
    # 
    # This avoids Error 162 from historical data requests in the data pipeline
    
    # Initialize multi-strategy system
    # You can change strategy_mode: "trend_following", "breakout", "mean_reversion", or "auto"
    multi_strategy = MultiStrategy(
        strategy_mode="auto",  # Auto-selects best strategy based on market conditions
        reward_risk_ratio=2.0,  # 2:1 reward:risk
        use_trailing_stop=True,
        trailing_stop_pct=0.5,  # 0.5% trailing stop
        min_confidence=0.65
    )
    
    # Wrap multi-strategy with LLM enhancement if enabled in config
    from mytrader.strategies.llm_enhanced_strategy import LLMEnhancedStrategy
    
    # Force enable LLM for trading decisions
    llm_enabled = True  # Hardcoded to use AWS Bedrock Claude
    
    if llm_enabled:
        logger.info("🤖 LLM Enhancement ENABLED - using AWS Bedrock Claude for signal analysis")
        multi_strategy = LLMEnhancedStrategy(
            base_strategy=multi_strategy,
            enable_llm=True,
            min_llm_confidence=0.55,  # Only execute if AI confidence >= 55% (lowered for testing)
            llm_override_mode=True  # Override mode: Let LLM override weak traditional signals
        )
        logger.info("   📋 LLM Config: model=claude-3-sonnet, min_confidence=0.55, mode=override")
    else:
        logger.info("⚠️  LLM Enhancement DISABLED - using traditional signals only")
    
    # Keep original strategies as fallback
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

    # Create IB instance and connect
    import random
    from ib_insync import util
    util.logToConsole('ERROR')  # Reduce log noise
    
    client_id = random.randint(10, 999)  # Use random client ID to avoid conflicts
    logger.info("Initializing IB connection to %s:%s (client_id=%d)", settings.data.ibkr_host, settings.data.ibkr_port, client_id)
    ib = IB()
    
    # Connect directly first to test
    try:
        logger.info("Attempting direct IB connection...")
        await asyncio.wait_for(
            ib.connectAsync(settings.data.ibkr_host, settings.data.ibkr_port, clientId=client_id, timeout=10),
            timeout=15
        )
        logger.info("✅ Connected to IBKR successfully")
    except asyncio.TimeoutError:
        logger.error("❌ Connection timeout after 15 seconds")
        logger.error("Check: 1) IB Gateway is running 2) API is enabled 3) Port 4002 is open")
        raise
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        raise
    
    # Now wrap with executor
    executor = TradeExecutor(ib, settings.trading, settings.data.ibkr_symbol, settings.data.ibkr_exchange)
    executor._connection_host = settings.data.ibkr_host
    executor._connection_port = settings.data.ibkr_port
    executor._connection_client_id = client_id
    
    # Set up executor event handlers
    ib.reqMarketDataType(3)  # Delayed market data
    logger.info("Using delayed market data (15-min delay, free)")
    ib.orderStatusEvent += executor._on_order_status
    ib.execDetailsEvent += executor._on_execution
    await executor._cancel_all_existing_orders()
    await executor._reconcile_positions()
    await executor._start_keepalive()

    account_value = settings.trading.initial_capital
    step = 0
    opt_interval = max(1, settings.optimization.retrain_interval // 60)
    status_log_interval = 10  # Log performance every 10 iterations
    
    # Initialize price history buffer (we'll build this from live data)
    price_history = []
    min_bars_needed = 15  # Minimum bars for feature engineering (reduced for faster start)
    poll_interval = 5  # seconds between price checks

    logger.info("Starting live trading loop (polling every %ds)...", poll_interval)
    logger.info("Will start generating signals after collecting %d bars", min_bars_needed)

    try:
        while True:
            try:
                # Get current market price from IBKR
                current_price = await executor.get_current_price()
                if not current_price:
                    logger.warning("No price data available, retrying...")
                    await asyncio.sleep(poll_interval)
                    continue
                
                # Add to price history (simplified: using last price for OHLC)
                # In production, you'd want proper bar aggregation
                from datetime import datetime, timezone
                price_bar = {
                    'timestamp': datetime.now(timezone.utc),
                    'open': current_price,
                    'high': current_price,
                    'low': current_price,
                    'close': current_price,
                    'volume': 0  # Not critical for these strategies
                }
                price_history.append(price_bar)
                
                logger.debug(f"Price bar added: {current_price:.2f} (total history: {len(price_history)} bars)")
                
                # Keep only recent history (last 500 bars for efficiency)
                if len(price_history) > 500:
                    price_history = price_history[-500:]
                
                # Need minimum bars before we can trade
                if len(price_history) < min_bars_needed:
                    logger.info(f"Building history: {len(price_history)}/{min_bars_needed} bars")
                    await asyncio.sleep(poll_interval)
                    continue
                
                logger.info(f"✅ History complete with {len(price_history)} bars - starting signal generation")
                
                # Convert to DataFrame and engineer features
                df = pd.DataFrame(price_history)
                df.set_index('timestamp', inplace=True)
                
                logger.info(f"🔧 Engineering features from {len(df)} bars...")
                features = engineer_features(df[['open', 'high', 'low', 'close', 'volume']], None)
                
                if features.empty:
                    logger.warning("⚠️  Feature engineering returned empty DataFrame - skipping iteration")
                    await asyncio.sleep(poll_interval)
                    continue
                
                logger.info(f"✅ Features engineered: {len(features)} rows, {len(features.columns)} columns")

                returns = features["close"].pct_change().dropna()
                logger.info(f"📈 Returns calculated: {len(returns)} values")

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

                # Get current position before generating signal
                current_position = await executor.get_current_position()
                current_qty = current_position.quantity if current_position else 0
                logger.info(f"📦 Current position: {current_qty} contracts")

                # Generate trading signal using multi-strategy (with optional LLM enhancement)
                logger.info(f"🤔 Evaluating strategy with {len(features)} feature rows...")
                try:
                    # Check if we're using LLM-enhanced strategy
                    if isinstance(multi_strategy, LLMEnhancedStrategy):
                        # Use the enhanced generate method which calls LLM
                        signal = multi_strategy.generate(features)
                        action = signal.action
                        confidence = signal.confidence
                        
                        # Extract risk params from metadata if base strategy provided them
                        risk_params = signal.metadata.get("risk_params", {})
                        
                        # Log LLM details if available
                        if "llm_recommendation" in signal.metadata:
                            llm_rec = signal.metadata["llm_recommendation"]
                            logger.info(f"🤖 LLM Recommendation: {llm_rec.get('action')} (confidence: {llm_rec.get('confidence', 0):.2f})")
                            logger.info(f"   💬 Reasoning: {llm_rec.get('reasoning', 'N/A')[:100]}...")
                        
                        # Get market context from base strategy if it's a MultiStrategy
                        if hasattr(multi_strategy.base_strategy, 'market_bias'):
                            market_bias = multi_strategy.base_strategy.market_bias
                            volatility_level = multi_strategy.base_strategy.volatility_level
                        else:
                            market_bias = "unknown"
                            volatility_level = "unknown"
                    else:
                        # Traditional multi-strategy without LLM
                        action, confidence, risk_params = multi_strategy.generate_signal(
                            df=features,
                            current_position=current_qty
                        )
                        market_bias = multi_strategy.market_bias
                        volatility_level = multi_strategy.volatility_level
                        
                        # Create signal object for compatibility
                        from dataclasses import dataclass
                        @dataclass
                        class Signal:
                            action: str
                            confidence: float
                            metadata: dict = None
                        
                        signal = Signal(
                            action=action,
                            confidence=confidence,
                            metadata={
                                "strategy": "multi_strategy",
                                "risk_params": risk_params,
                                "market_bias": market_bias,
                                "volatility": volatility_level
                            }
                        )
                    
                    logger.info(f"📊 SIGNAL GENERATED: {signal.action}, confidence={signal.confidence:.2f}")
                    if risk_params:
                        logger.info(f"   📍 Stop Loss: {risk_params.get('stop_loss_long', 'N/A'):.2f}")
                        logger.info(f"   🎯 Take Profit: {risk_params.get('take_profit_long', 'N/A'):.2f}")
                        logger.info(f"   📏 ATR: {risk_params.get('atr', 'N/A'):.2f}")
                    logger.info(f"   📊 Market: {market_bias}, Volatility: {volatility_level}")
                    
                except Exception as signal_error:
                    logger.error(f"❌ Error generating signal: {signal_error}")
                    logger.exception("Signal generation traceback:")
                    await asyncio.sleep(poll_interval)
                    continue
                
                # Update tracker
                if current_position:
                    unrealized_pnl = await executor.get_unrealized_pnl()
                    tracker.update_equity(current_price, realized_pnl=0.0)
                    
                    # Check exit conditions using multi-strategy risk management
                    logger.info(f"🔍 Checking exit: position={current_position.quantity}, avg_cost={current_position.avg_cost:.2f}, current={current_price:.2f}, risk_params={risk_params}")
                    if risk_params and hasattr(current_position, 'avg_cost'):
                        should_exit, exit_reason = multi_strategy.should_exit_position(
                            df=features,
                            entry_price=current_position.avg_cost,
                            position=current_position.quantity,
                            risk_params=risk_params
                        )
                        logger.info(f"   → Exit check result: should_exit={should_exit}, reason={exit_reason}")
                        
                        if should_exit:
                            logger.info(f"🛑 EXIT SIGNAL: {exit_reason}")
                            close_result = await executor.close_position()
                            if close_result and close_result.fill_price:
                                tracker.record_trade(
                                    action="SELL" if current_position.quantity > 0 else "BUY",
                                    price=close_result.fill_price,
                                    quantity=abs(current_position.quantity)
                                )
                                realized = (close_result.fill_price - current_position.avg_cost) * current_position.quantity
                                risk.update_pnl(realized)
                                tracker.update_equity(current_price, realized)
                                logger.info(f"✅ Position closed: {exit_reason}, realized PnL: {realized:.2f}")
                            await asyncio.sleep(poll_interval)
                            continue
                    
                    # Update trailing stops if configured (matches backtest logic)
                    atr_val = float(features.iloc[-1].get("ATR_14", 0.0))
                    await executor.update_trailing_stops(current_price, atr_val)
                    
                    # Check for opposite signal exit (matches backtest logic)
                    direction = "BUY" if current_position.quantity > 0 else "SELL"
                    opposite_action = "SELL" if direction == "BUY" else "BUY"
                    if signal.action == opposite_action and signal.action != "HOLD":
                        logger.info("🔄 Opposite signal detected - closing position")
                        close_result = await executor.close_position()
                        if close_result and close_result.fill_price:
                            # Record the exit
                            tracker.record_trade(
                                action=opposite_action,
                                price=close_result.fill_price,
                                quantity=abs(current_position.quantity)
                            )
                            realized = (close_result.fill_price - current_position.avg_cost) * current_position.quantity
                            risk.update_pnl(realized)
                            tracker.update_equity(current_price, realized)
                            logger.info("Position closed on opposite signal, realized PnL: %.2f", realized)
                        await asyncio.sleep(poll_interval)
                        continue
                
                if signal.action == "HOLD":
                    logger.info("⏸️  HOLD signal - no action taken")
                    await asyncio.sleep(poll_interval)
                    continue
                
                # Don't open new position if we already have one
                if current_position and current_position.quantity != 0:
                    logger.info("⚠️  Already have open position, skipping new signal")
                    await asyncio.sleep(poll_interval)
                    continue

                logger.info(f"🎯 Preparing to execute {signal.action} signal...")

                # Position sizing with Kelly Criterion
                risk_stats = risk.get_statistics()
                qty = risk.position_size(
                    account_value, 
                    signal.confidence,
                    win_rate=risk_stats.get("win_rate"),
                    avg_win=risk_stats.get("avg_win"),
                    avg_loss=risk_stats.get("avg_loss")
                )
                
                # Apply position scaler from metadata (like backtest)
                metadata = signal.metadata if isinstance(signal.metadata, dict) else {}
                scaler = float(metadata.get("position_scaler", 1.0))
                if scaler > 0:
                    qty = max(1, int(round(qty * scaler)))
                
                qty = min(qty, settings.trading.max_position_size)
                
                if not risk.can_trade(qty):
                    logger.warning("Risk limits exceeded. Skipping trade.")
                    await asyncio.sleep(poll_interval)
                    continue

                # Calculate stop-loss and take-profit using SAME logic as backtest
                direction = 1 if signal.action == "BUY" else -1
                row = features.iloc[-1]
                
                # Check for explicit prices in metadata first
                raw_stop = metadata.get("stop_loss_price")
                stop_loss = float(raw_stop) if isinstance(raw_stop, (int, float)) else None
                
                raw_target = metadata.get("take_profit_price")
                take_profit = float(raw_target) if isinstance(raw_target, (int, float)) else None
                
                # Get ATR value (from metadata or features)
                atr = float(metadata.get("atr_value", 0.0))
                if atr <= 0:
                    atr = float(row.get("ATR_14", 0.0))
                
                tick_size = settings.trading.tick_size
                default_stop_offset = settings.trading.stop_loss_ticks * tick_size
                default_target_offset = settings.trading.take_profit_ticks * tick_size
                
                # Calculate stop-loss if not provided
                if stop_loss is None:
                    atr_multiplier = float(metadata.get("atr_stop_multiplier", 0.0))
                    if atr_multiplier > 0 and atr > 0:
                        stop_offset = atr * atr_multiplier
                    else:
                        stop_offset = default_stop_offset
                    
                    if stop_offset <= 0:
                        stop_offset = default_stop_offset
                    
                    stop_loss = current_price - stop_offset if direction > 0 else current_price + stop_offset
                
                # Calculate take-profit if not provided
                if take_profit is None:
                    risk_reward = float(metadata.get("risk_reward", 0.0))
                    if risk_reward <= 0:
                        if default_stop_offset > 0:
                            risk_reward = settings.trading.take_profit_ticks / max(1e-6, settings.trading.stop_loss_ticks)
                        else:
                            risk_reward = 2.0
                    
                    stop_distance = abs(current_price - stop_loss)
                    if stop_distance <= 0:
                        stop_distance = default_stop_offset
                    
                    target_offset = stop_distance * risk_reward if risk_reward > 0 else default_target_offset
                    take_profit = current_price + target_offset if direction > 0 else current_price - target_offset
                
                logger.info(f"📈 Trade levels: entry={current_price:.2f} stop={stop_loss:.2f} target={take_profit:.2f} (ATR={atr:.2f})")
                logger.info(f"💰 Position size: {qty} contracts (account value: ${account_value:.2f})")

                # Store trailing stop parameters from metadata
                trailing_atr_multiplier = float(metadata.get("trailing_atr_multiplier", 0.0)) or None
                trailing_percent = float(metadata.get("trailing_percent", 0.0)) or None
                
                logger.info(f"🚀 Placing {signal.action} order for {qty} contracts...")
                
                # Place order with bracket orders
                result = await executor.place_order(
                    action=signal.action,
                    quantity=qty,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        "trailing_atr_multiplier": trailing_atr_multiplier,
                        "trailing_percent": trailing_percent,
                        "atr_value": atr,
                        "entry_metadata": metadata
                    }
                )
                
                logger.info(f"✅ Order result: status={result.status}, filled={result.filled_quantity} @ {result.fill_price if result.fill_price else 0.0:.2f}")
                
                if result.status not in {"Cancelled", "Inactive"}:
                    risk.register_trade()
                    # Record trade in tracker with metadata
                    if result.fill_price:
                        tracker.record_trade(
                            action=signal.action,
                            price=result.fill_price,
                            quantity=qty
                        )
                
                # Periodic performance logging
                if step % status_log_interval == 0:
                    tracker.log_status()
                
                # Wait before next poll
                await asyncio.sleep(poll_interval)
                    
            except Exception as exc:  # noqa: BLE001
                logger.exception("Live loop error: %s", exc)
                await asyncio.sleep(poll_interval)
                
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
