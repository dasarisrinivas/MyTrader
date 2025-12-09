"""
MyTrader Dashboard API
FastAPI backend providing REST API and WebSocket for real-time trading dashboard
"""

import sys
from pathlib import Path

# Add project root to Python path (both parent directories to ensure imports work)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
from datetime import datetime, timedelta
import json
import subprocess
import random

from mytrader.utils.settings_loader import load_settings
from mytrader.monitoring.live_tracker import LivePerformanceTracker
from mytrader.monitoring.order_tracker import OrderTracker
from mytrader.monitoring.pnl_calculator import calculate_pnl_for_orders, calculate_unrealized_pnl
from mytrader.utils.logger import configure_logging, logger
from mytrader.execution.live_trading_manager import LiveTradingManager
from mytrader.execution.ib_executor import TradeExecutor

# Import RAG API router
try:
    from dashboard.backend.rag_api import include_rag_router
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logger.warning("RAG API not available - install dependencies: pip install faiss-cpu")

# Initialize FastAPI app
app = FastAPI(
    title="MyTrader Dashboard API",
    description="Real-time trading dashboard backend",
    version="1.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
live_trading_manager: Optional[LiveTradingManager] = None
trading_session = None
performance_tracker = None
websocket_clients: List[WebSocket] = []


# Pydantic models
class TradingConfig(BaseModel):
    max_position_size: int = 2
    max_daily_loss: float = 1500.0
    stop_loss_ticks: float = 20.0
    take_profit_ticks: float = 40.0


class StartTradingRequest(BaseModel):
    config_path: str = "config.yaml"
    strategy: str = "rsi_macd_sentiment"


class BacktestRequest(BaseModel):
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD
    download_data: bool = True  # Whether to download fresh data
    config_path: str = "config.yaml"


class DataDownloadRequest(BaseModel):
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD


class TradeResponse(BaseModel):
    timestamp: str
    action: str
    quantity: int
    price: float
    pnl: float
    status: str


class PerformanceSnapshot(BaseModel):
    total_pnl: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int


# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            logger.debug("No active WebSocket connections to broadcast to")
            return
            
        logger.info(f"Broadcasting {message['type']} to {len(self.active_connections)} clients")
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")


manager = ConnectionManager()


async def run_integrated_live_trading(settings):
    """
    Integrated live trading loop that runs inside the dashboard process.
    This is the same logic as main.py but integrated for real-time UI updates.
    """
    from mytrader.strategies.multi_strategy import MultiStrategy
    from mytrader.strategies.momentum_reversal import MomentumReversalStrategy
    from mytrader.strategies.rsi_macd_sentiment import RsiMacdSentimentStrategy
    from mytrader.features.feature_engineer import engineer_features
    from mytrader.risk.manager import RiskManager
    from mytrader.monitoring.live_tracker import LivePerformanceTracker
    from ib_insync import IB
    import pandas as pd
    from datetime import datetime, timezone
    
    try:
        # Initialize multi-strategy system
        multi_strategy = MultiStrategy(
            strategy_mode="auto",
            reward_risk_ratio=2.0,
            use_trailing_stop=True,
            trailing_stop_pct=0.5,
            min_confidence=0.65
        )
        
        # Wrap with LLM enhancement if enabled in config
        from mytrader.strategies.llm_enhanced_strategy import LLMEnhancedStrategy
        llm_enabled = settings.llm.enabled if hasattr(settings, 'llm') else False
        
        if llm_enabled:
            logger.info("🤖 LLM Enhancement ENABLED - using AWS Bedrock Claude")
            multi_strategy = LLMEnhancedStrategy(
                base_strategy=multi_strategy,
                enable_llm=True,
                min_llm_confidence=settings.llm.min_confidence_threshold,
                llm_override_mode=settings.llm.override_mode
            )
            
            # Broadcast LLM status
            await manager.broadcast({
                "type": "system",
                "message": "🤖 LLM Enhancement Active - AWS Bedrock Claude analyzing trades",
                "timestamp": datetime.now().isoformat()
            })
        else:
            logger.info("⚠️  LLM Enhancement DISABLED")
        
        # Initialize risk manager with Kelly Criterion support
        risk = RiskManager(settings.trading, position_sizing_method="kelly")
        
        # Initialize live performance tracker
        tracker = LivePerformanceTracker(
            initial_capital=settings.trading.initial_capital,
            risk_free_rate=settings.backtest.risk_free_rate
        )
        
        # Initialize Telegram notifier if configured
        from mytrader.utils.telegram_notifier import TelegramNotifier
        telegram_notifier = None
        if hasattr(settings, 'telegram') and settings.telegram.enabled:
            telegram_notifier = TelegramNotifier(
                bot_token=settings.telegram.bot_token,
                chat_id=settings.telegram.chat_id,
                enabled=True
            )
            logger.info("✅ Telegram notifications initialized")
        
        # Create IB instance and connect
        client_id = random.randint(100, 999)  # Use random client ID like standalone bot
        logger.info(f"🔌 Connecting to IBKR from dashboard (client_id={client_id})...")
        ib = IB()
        executor = TradeExecutor(
            ib, 
            settings.trading, 
            settings.data.ibkr_symbol, 
            settings.data.ibkr_exchange,
            telegram_notifier=telegram_notifier
        )
        
        try:
            await executor.connect(
                settings.data.ibkr_host, 
                settings.data.ibkr_port, 
                client_id=client_id,
                timeout=30  # Shorter timeout like standalone bot
            )
            logger.info(f"✅ Connected to IBKR successfully (client_id={client_id})")
        except TimeoutError as e:
            error_msg = f"❌ Failed to connect to IB Gateway (timeout): {e}"
            logger.error(error_msg)
            await manager.broadcast({
                "type": "error",
                "message": error_msg,
                "timestamp": datetime.now().isoformat()
            })
            live_trading_manager = None
            raise HTTPException(status_code=503, detail=error_msg)
        except Exception as e:
            error_msg = f"❌ Failed to connect to IB Gateway: {e}"
            logger.error(error_msg)
            await manager.broadcast({
                "type": "error",
                "message": error_msg,
                "timestamp": datetime.now().isoformat()
            })
            live_trading_manager = None
            raise HTTPException(status_code=503, detail=error_msg)
        
        # Broadcast connection status
        await manager.broadcast({
            "type": "system",
            "message": "Connected to IBKR and starting trading",
            "timestamp": datetime.now().isoformat()
        })
        
        account_value = settings.trading.initial_capital
        price_history = []
        min_bars_needed = 50
        poll_interval = 5
        
        logger.info("🚀 Starting integrated live trading loop...")
        
        while live_trading_manager and live_trading_manager.running:
            try:
                # Get current market price
                current_price = await executor.get_current_price()
                if not current_price:
                    logger.warning("No price data available, retrying...")
                    await asyncio.sleep(poll_interval)
                    continue
                
                # Broadcast price update
                await manager.broadcast({
                    "type": "price_update",
                    "price": current_price,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Add to price history
                price_bar = {
                    'timestamp': datetime.now(timezone.utc),
                    'open': current_price,
                    'high': current_price,
                    'low': current_price,
                    'close': current_price,
                    'volume': 0
                }
                price_history.append(price_bar)
                
                # Keep only recent history
                if len(price_history) > 500:
                    price_history = price_history[-500:]
                
                # Need minimum bars before we can trade
                if len(price_history) < min_bars_needed:
                    logger.info(f"Building history: {len(price_history)}/{min_bars_needed} bars")
                    await manager.broadcast({
                        "type": "progress",
                        "bars_collected": len(price_history),
                        "min_bars_needed": min_bars_needed,
                        "timestamp": datetime.now().isoformat()
                    })
                    await asyncio.sleep(poll_interval)
                    continue
                
                # Convert to DataFrame and engineer features
                df = pd.DataFrame(price_history)
                df.set_index('timestamp', inplace=True)
                
                features = engineer_features(df[['open', 'high', 'low', 'close', 'volume']], None)
                
                if features.empty:
                    logger.warning("Feature engineering returned empty DataFrame")
                    await asyncio.sleep(poll_interval)
                    continue
                
                # Get current position
                current_position = await executor.get_current_position()
                current_qty = current_position.quantity if current_position else 0
                
                # Generate trading signal using multi-strategy (with optional LLM)
                try:
                    # Check if using LLM-enhanced strategy
                    if isinstance(multi_strategy, LLMEnhancedStrategy):
                        # Use enhanced generate method which calls LLM
                        signal = multi_strategy.generate(features)
                        action = signal.action
                        confidence = signal.confidence
                        risk_params = signal.metadata.get("risk_params", {})
                        
                        # Broadcast LLM recommendation if available
                        if "llm_recommendation" in signal.metadata:
                            llm_rec = signal.metadata["llm_recommendation"]
                            await manager.broadcast({
                                "type": "llm_analysis",
                                "action": llm_rec.get('trade_decision'),
                                "confidence": llm_rec.get('confidence'),
                                "sentiment_score": llm_rec.get('sentiment_score', 0.0),
                                "reasoning": llm_rec.get('reasoning', '')[:200],
                                "timestamp": datetime.now().isoformat()
                            })
                            logger.info(
                                f"🤖 LLM: {llm_rec.get('trade_decision')} "
                                f"(conf: {llm_rec.get('confidence', 0):.2f}, "
                                f"sentiment: {llm_rec.get('sentiment_score', 0):+.2f})"
                            )
                        
                        # Get market context from base strategy
                        if hasattr(multi_strategy.base_strategy, 'market_bias'):
                            market_bias = multi_strategy.base_strategy.market_bias
                            volatility_level = multi_strategy.base_strategy.volatility_level
                        else:
                            market_bias = "unknown"
                            volatility_level = "unknown"
                    else:
                        # Traditional multi-strategy
                        action, confidence, risk_params = multi_strategy.generate_signal(
                            df=features,
                            current_position=current_qty
                        )
                        market_bias = multi_strategy.market_bias
                        volatility_level = multi_strategy.volatility_level
                    
                    # Broadcast signal to UI
                    await manager.broadcast({
                        "type": "signal",
                        "signal": action,
                        "confidence": confidence,
                        "market_bias": market_bias,
                        "volatility": volatility_level,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    logger.info(f"📊 SIGNAL: {action}, confidence={confidence:.2f}")
                    
                except Exception as signal_error:
                    logger.error(f"Error generating signal: {signal_error}")
                    await asyncio.sleep(poll_interval)
                    continue
                
                # Update tracker
                if current_position:
                    unrealized_pnl = await executor.get_unrealized_pnl()
                    tracker.update_equity(current_price, realized_pnl=0.0)
                    
                    # Check exit conditions
                    if risk_params and hasattr(current_position, 'avg_cost'):
                        should_exit, exit_reason = multi_strategy.should_exit_position(
                            df=features,
                            entry_price=current_position.avg_cost,
                            position=current_position.quantity,
                            risk_params=risk_params
                        )
                        
                        if should_exit:
                            logger.info(f"🛑 EXIT SIGNAL: {exit_reason}")
                            await manager.broadcast({
                                "type": "exit_signal",
                                "reason": exit_reason,
                                "timestamp": datetime.now().isoformat()
                            })
                            
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
                                
                                # Broadcast trade execution
                                await manager.broadcast({
                                    "type": "trade_executed",
                                    "action": "CLOSE",
                                    "price": close_result.fill_price,
                                    "quantity": abs(current_position.quantity),
                                    "pnl": realized,
                                    "timestamp": datetime.now().isoformat()
                                })
                            
                            await asyncio.sleep(poll_interval)
                            continue
                    
                    # Update trailing stops
                    atr_val = float(features.iloc[-1].get("ATR_14", 0.0))
                    await executor.update_trailing_stops(current_price, atr_val)
                    
                    # Check for opposite signal
                    direction = "BUY" if current_position.quantity > 0 else "SELL"
                    opposite_action = "SELL" if direction == "BUY" else "BUY"
                    if action == opposite_action and action != "HOLD":
                        logger.info("🔄 Opposite signal - closing position")
                        close_result = await executor.close_position()
                        if close_result and close_result.fill_price:
                            tracker.record_trade(
                                action=opposite_action,
                                price=close_result.fill_price,
                                quantity=abs(current_position.quantity)
                            )
                            realized = (close_result.fill_price - current_position.avg_cost) * current_position.quantity
                            risk.update_pnl(realized)
                            tracker.update_equity(current_price, realized)
                            
                            await manager.broadcast({
                                "type": "trade_executed",
                                "action": "CLOSE_OPPOSITE",
                                "price": close_result.fill_price,
                                "quantity": abs(current_position.quantity),
                                "pnl": realized,
                                "timestamp": datetime.now().isoformat()
                            })
                        await asyncio.sleep(poll_interval)
                        continue
                
                if action == "HOLD":
                    logger.info("⏸️ HOLD signal")
                    await asyncio.sleep(poll_interval)
                    continue
                
                # Don't open new position if we already have one
                if current_position and current_position.quantity != 0:
                    logger.info("Already have position, skipping")
                    await asyncio.sleep(poll_interval)
                    continue
                
                # Position sizing
                risk_stats = risk.get_statistics()
                qty = risk.position_size(
                    account_value, 
                    confidence,
                    win_rate=risk_stats.get("win_rate"),
                    avg_win=risk_stats.get("avg_win"),
                    avg_loss=risk_stats.get("avg_loss")
                )
                
                metadata = {}
                scaler = float(metadata.get("position_scaler", 1.0))
                if scaler > 0:
                    qty = max(1, int(round(qty * scaler)))
                
                qty = min(qty, settings.trading.max_position_size)
                
                if not risk.can_trade(qty):
                    logger.warning("Risk limits exceeded")
                    await manager.broadcast({
                        "type": "risk_limit",
                        "message": "Risk limits exceeded, skipping trade",
                        "timestamp": datetime.now().isoformat()
                    })
                    await asyncio.sleep(poll_interval)
                    continue
                
                # Calculate stop-loss and take-profit
                direction = 1 if action == "BUY" else -1
                row = features.iloc[-1]
                atr = float(row.get("ATR_14", 0.0))
                
                tick_size = settings.trading.tick_size
                default_stop_offset = settings.trading.stop_loss_ticks * tick_size
                default_target_offset = settings.trading.take_profit_ticks * tick_size
                
                stop_offset = default_stop_offset
                stop_loss = current_price - stop_offset if direction > 0 else current_price + stop_offset
                
                risk_reward = 2.0
                target_offset = default_target_offset
                take_profit = current_price + target_offset if direction > 0 else current_price - target_offset
                
                logger.info(f"📈 Trade levels: entry={current_price:.2f} stop={stop_loss:.2f} target={take_profit:.2f}")
                
                # Broadcast order placement
                await manager.broadcast({
                    "type": "order_placing",
                    "action": action,
                    "quantity": qty,
                    "entry_price": current_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Place order
                result = await executor.place_order(
                    action=action,
                    quantity=qty,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={"atr_value": atr}
                )
                
                logger.info(f"✅ Order result: {result.status}")
                
                # Broadcast order result
                await manager.broadcast({
                    "type": "order_placed",
                    "action": action,
                    "status": result.status,
                    "filled_quantity": result.filled_quantity,
                    "fill_price": result.fill_price,
                    "timestamp": datetime.now().isoformat()
                })
                
                if result.status not in {"Cancelled", "Inactive"}:
                    risk.register_trade()
                    if result.fill_price:
                        tracker.record_trade(
                            action=action,
                            price=result.fill_price,
                            quantity=qty
                        )
                        
                        await manager.broadcast({
                            "type": "trade_executed",
                            "action": action,
                            "price": result.fill_price,
                            "quantity": qty,
                            "timestamp": datetime.now().isoformat()
                        })
                
                await asyncio.sleep(poll_interval)
                
            except Exception as loop_error:
                logger.exception(f"Live loop error: {loop_error}")
                await manager.broadcast({
                    "type": "error",
                    "message": str(loop_error),
                    "timestamp": datetime.now().isoformat()
                })
                await asyncio.sleep(poll_interval)
    
    except asyncio.CancelledError:
        logger.info("Trading loop cancelled")
        if ib.isConnected():
            ib.disconnect()
        raise
    except Exception as e:
        logger.exception(f"Fatal error in trading loop: {e}")
        await manager.broadcast({
            "type": "fatal_error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        })
        if ib.isConnected():
            ib.disconnect()
    finally:
        logger.info("Trading loop stopped")
        if ib.isConnected():
            ib.disconnect()


# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "MyTrader Dashboard API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/status")
async def get_status():
    """Get current trading session status."""
    global trading_session, performance_tracker
    
    is_running = trading_session is not None and getattr(trading_session, 'running', False)
    
    status = {
        "is_running": is_running,
        "session_start": getattr(trading_session, 'session_start', None),
        "total_signals": getattr(trading_session, 'total_signals', 0),
        "total_trades": getattr(trading_session, 'total_trades', 0),
    }
    
    if performance_tracker:
        snapshot = performance_tracker.get_snapshot()
        # Calculate return percentage
        total_return = (snapshot.equity / performance_tracker.initial_capital - 1) * 100
        status.update({
            "total_pnl": snapshot.total_pnl,
            "total_return": total_return,
            "sharpe_ratio": snapshot.sharpe_ratio,
            "max_drawdown": snapshot.max_drawdown,
        })
    
    return status


@app.post("/api/trading/start")
async def start_trading(request: StartTradingRequest):
    """Start a new live trading session - runs the trading bot integrated into dashboard."""
    global live_trading_manager
    
    if live_trading_manager and live_trading_manager.running:
        raise HTTPException(status_code=400, detail="Trading session already running")
    
    try:
        import os
        
        # Load settings - use absolute path from project root
        config_path = project_root / request.config_path
        
        logger.info(f"Attempting to load config from: {config_path}")
        logger.info(f"Config exists: {config_path.exists()}")
        logger.info(f"Project root: {project_root}")
        
        if not config_path.exists():
            raise HTTPException(status_code=404, detail=f"Config file not found: {config_path}")
        
        # Import the live trading function
        from mytrader.config import Settings
        from mytrader.utils.settings_loader import load_settings as load_settings_func
        
        # Load settings
        settings = load_settings_func(config_path)
        
        # Create a manager to track the trading session
        from dataclasses import dataclass
        import asyncio
        
        @dataclass
        class IntegratedTradingManager:
            task: asyncio.Task
            running: bool = True
            settings: Settings = None
            
            async def stop(self):
                """Stop the trading session gracefully."""
                self.running = False
                if self.task and not self.task.done():
                    self.task.cancel()
                    try:
                        await self.task
                    except asyncio.CancelledError:
                        pass
        
        # Start the live trading loop as a background task
        trading_task = asyncio.create_task(run_integrated_live_trading(settings))
        live_trading_manager = IntegratedTradingManager(task=trading_task, settings=settings)
        
        logger.info(f"✅ Started integrated live trading session")
        logger.info(f"� Strategy: {request.strategy}")
        logger.info(f"⚙️  Config: {config_path}")
        
        return {
            "status": "started",
            "message": "Live trading session started successfully (integrated mode)",
            "strategy": request.strategy,
            "config": str(config_path),
            "timestamp": datetime.now().isoformat(),
            "mode": "integrated"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start live trading: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trading/stop")
async def stop_trading():
    """Stop the current live trading session."""
    global live_trading_manager
    
    if not live_trading_manager:
        raise HTTPException(status_code=400, detail="No trading session running")
    
    try:
        # Stop the integrated trading session
        await live_trading_manager.stop()
        live_trading_manager = None
        
        logger.info("✅ Trading session stopped successfully")
        
        return {
            "status": "stopped",
            "message": "Trading session stopped successfully",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to stop trading: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trading/emergency-exit")
async def emergency_exit():
    """Emergency exit: Close all positions and stop the bot."""
    global live_trading_manager
    
    try:
        positions_closed = 0
        errors = []
        
        # Step 1: Close all open positions
        if live_trading_manager and live_trading_manager.executor:
            try:
                executor = live_trading_manager.executor
                current_position = await executor.get_current_position()
                
                if current_position != 0:
                    logger.info(f"🚨 EMERGENCY EXIT: Closing position of {current_position} contracts")
                    
                    # Close position with market order for immediate execution
                    action = "SELL" if current_position > 0 else "BUY"
                    quantity = abs(current_position)
                    
                    order_id = await executor.place_market_order(
                        action=action,
                        quantity=quantity,
                        reason="EMERGENCY_EXIT"
                    )
                    
                    if order_id:
                        positions_closed = quantity
                        logger.info(f"✅ Emergency exit order placed: {order_id}")
                    else:
                        errors.append("Failed to place exit order")
                else:
                    logger.info("No open positions to close")
                    
            except Exception as e:
                error_msg = f"Failed to close positions: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        # Step 2: Cancel all pending orders
        if live_trading_manager and live_trading_manager.executor:
            try:
                await live_trading_manager.executor.cancel_all_orders()
                logger.info("✅ All pending orders cancelled")
            except Exception as e:
                error_msg = f"Failed to cancel orders: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        # Step 3: Stop the trading bot
        if live_trading_manager:
            try:
                await live_trading_manager.stop()
                live_trading_manager = None
                logger.info("✅ Trading bot stopped")
            except Exception as e:
                error_msg = f"Failed to stop bot: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        return {
            "status": "completed",
            "positions_closed": positions_closed,
            "errors": errors,
            "timestamp": datetime.now().isoformat(),
            "message": "Emergency exit completed" if not errors else "Emergency exit completed with errors"
        }
        
    except Exception as e:
        logger.error(f"Emergency exit failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trading/status")
async def get_trading_status():
    """Get detailed live trading status."""
    global live_trading_manager
    
    if not live_trading_manager or not live_trading_manager.running:
        return {
            "is_running": False,
            "message": "No trading session"
        }
    
    # Check if task is still running
    if live_trading_manager.task.done():
        live_trading_manager.running = False
        return {
            "is_running": False,
            "message": "Trading task completed or crashed"
        }
    
    return {
        "is_running": True,
        "mode": "integrated",
        "message": "Trading session running (integrated mode)"
    }


@app.get("/api/market/status")
async def get_market_status():
    """
    Get detailed market status including conditions for trading resume.
    This endpoint provides information about when HOLD signals will transition to active trading.
    """
    try:
        # Read the latest data from the live trading log
        log_file = project_root / "logs" / "live_trading.log"
        
        market_status = {
            "symbol": "ES",
            "exchange": "CME",
            "contract": "ESZ5",
            "current_price": None,
            "last_signal": "HOLD",
            "signal_confidence": 0.0,
            "market_bias": "neutral",
            "volatility_level": "medium",
            "active_strategy": "breakout",
            "atr": None,
            "stop_loss": None,
            "take_profit": None,
            "resume_conditions": {
                "breakout_detected": False,
                "confidence_threshold_met": False,
                "market_context_changed": False,
                "strategy_switched": False
            },
            "resume_triggers": {
                "breakout": "Price breaks above resistance or below support levels",
                "confidence": "Signal confidence exceeds 0.65 threshold",
                "market_bias": "Market switches from neutral to bullish/bearish",
                "volatility": "Volatility increases to create trading opportunities",
                "strategy": "System switches from breakout to mean-reversion or trend-following"
            },
            "last_update": datetime.now().isoformat()
        }
        
        # Parse the log file for latest information
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    
                    # Read last 100 lines to get recent data
                    recent_lines = lines[-100:] if len(lines) > 100 else lines
                    
                    for line in reversed(recent_lines):
                        # Extract current price
                        if "Got last price:" in line:
                            try:
                                price = float(line.split("Got last price:")[1].strip())
                                if market_status["current_price"] is None:
                                    market_status["current_price"] = price
                            except:
                                pass
                        
                        # Extract signal information
                        if "SIGNAL GENERATED:" in line:
                            try:
                                parts = line.split("SIGNAL GENERATED:")[1].strip()
                                signal_parts = parts.split(",")
                                if len(signal_parts) >= 2:
                                    market_status["last_signal"] = signal_parts[0].strip()
                                    conf_str = signal_parts[1].split("=")[1].strip()
                                    market_status["signal_confidence"] = float(conf_str)
                            except:
                                pass
                        
                        # Extract market context
                        if "Market Context:" in line:
                            try:
                                context = line.split("Market Context:")[1].strip()
                                if "bullish" in context:
                                    market_status["market_bias"] = "bullish"
                                elif "bearish" in context:
                                    market_status["market_bias"] = "bearish"
                                else:
                                    market_status["market_bias"] = "neutral"
                                    
                                if "high volatility" in context:
                                    market_status["volatility_level"] = "high"
                                elif "low volatility" in context:
                                    market_status["volatility_level"] = "low"
                                else:
                                    market_status["volatility_level"] = "medium"
                            except:
                                pass
                        
                        # Extract strategy
                        if "Using strategy:" in line:
                            try:
                                strategy = line.split("Using strategy:")[1].strip()
                                market_status["active_strategy"] = strategy
                            except:
                                pass
                        
                        # Extract ATR
                        if "📏 ATR:" in line:
                            try:
                                atr_str = line.split("📏 ATR:")[1].strip()
                                market_status["atr"] = float(atr_str)
                            except:
                                pass
                        
                        # Extract Stop Loss
                        if "📍 Stop Loss:" in line:
                            try:
                                sl_str = line.split("📍 Stop Loss:")[1].strip()
                                market_status["stop_loss"] = float(sl_str)
                            except:
                                pass
                        
                        # Extract Take Profit
                        if "🎯 Take Profit:" in line:
                            try:
                                tp_str = line.split("🎯 Take Profit:")[1].strip()
                                market_status["take_profit"] = float(tp_str)
                            except:
                                pass
                        
                        # Extract contract
                        if "Qualified contract:" in line:
                            try:
                                contract = line.split("Qualified contract:")[1].split("(")[0].strip()
                                market_status["contract"] = contract
                            except:
                                pass
                
                # Determine resume conditions
                market_status["resume_conditions"]["confidence_threshold_met"] = \
                    market_status["signal_confidence"] >= 0.65
                
                market_status["resume_conditions"]["market_context_changed"] = \
                    market_status["market_bias"] != "neutral" or market_status["volatility_level"] == "high"
                
                market_status["resume_conditions"]["breakout_detected"] = \
                    market_status["last_signal"] in ["BUY", "SELL"]
                
                market_status["resume_conditions"]["strategy_switched"] = \
                    market_status["active_strategy"] not in ["breakout", ""]
                
            except Exception as e:
                logger.error(f"Error parsing log file: {e}")
        
        return market_status
        
    except Exception as e:
        logger.error(f"Error getting market status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trades")
async def get_trades(limit: int = 50):
    """Get recent trades from live trading log."""
    log_file = project_root / "logs" / "live_trading.log"
    
    if not log_file.exists():
        return {"trades": [], "count": 0, "error": "Log file not found"}
    
    try:
        trades = []
        orders = []
        executions = []
        
        # Parse log file for order and execution information
        with open(log_file, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            # Look for SIGNAL GENERATED lines
            if "SIGNAL GENERATED:" in line and i > 0:
                try:
                    timestamp_str = line.split("|")[0].strip()
                    timestamp = datetime.fromisoformat(timestamp_str.replace(" ", "T"))
                    
                    # Extract signal details
                    if "BUY" in line:
                        action = "BUY"
                    elif "SELL" in line:
                        action = "SELL"
                    elif "HOLD" in line:
                        action = "HOLD"
                    else:
                        continue
                    
                    # Get confidence from the signal line
                    confidence = 0.0
                    if "confidence=" in line:
                        conf_part = line.split("confidence=")[1].split()[0].replace(",", "")
                        confidence = float(conf_part)
                    
                    # Look ahead for price, stop loss, take profit
                    price = None
                    stop_loss = None
                    take_profit = None
                    atr = None
                    
                    for j in range(i+1, min(i+10, len(lines))):
                        next_line = lines[j]
                        if "Got last price:" in next_line:
                            try:
                                price = float(next_line.split("Got last price:")[1].strip())
                            except:
                                pass
                        if "Stop Loss:" in next_line:
                            try:
                                stop_loss = float(next_line.split("Stop Loss:")[1].strip().split()[0])
                            except:
                                pass
                        if "Take Profit:" in next_line:
                            try:
                                take_profit = float(next_line.split("Take Profit:")[1].strip().split()[0])
                            except:
                                pass
                        if "ATR:" in next_line:
                            try:
                                atr = float(next_line.split("ATR:")[1].strip().split()[0])
                            except:
                                pass
                    
                    if action != "HOLD":  # Only add actual trade signals
                        trades.append({
                            "timestamp": timestamp.isoformat(),
                            "action": action,
                            "price": price,
                            "confidence": confidence,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "atr": atr,
                            "status": "signal"
                        })
                        
                except Exception as e:
                    logger.debug(f"Error parsing signal line: {e}")
                    continue
            
            # Look for order execution information
            if "Execution: order_id=" in line or "Order" in line and "status update:" in line:
                try:
                    timestamp_str = line.split("|")[0].strip()
                    timestamp = datetime.fromisoformat(timestamp_str.replace(" ", "T"))
                    
                    executions.append({
                        "timestamp": timestamp.isoformat(),
                        "message": line.split("|")[-1].strip(),
                        "type": "execution"
                    })
                except:
                    pass
            
            # Look for closed positions
            if "Closing position:" in line:
                try:
                    timestamp_str = line.split("|")[0].strip()
                    timestamp = datetime.fromisoformat(timestamp_str.replace(" ", "T"))
                    
                    orders.append({
                        "timestamp": timestamp.isoformat(),
                        "action": "CLOSE",
                        "message": line.split("|")[-1].strip(),
                        "type": "close"
                    })
                except:
                    pass
        
        # Return most recent trades
        all_activity = trades + orders + executions
        all_activity.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return {
            "trades": all_activity[:limit],
            "count": len(all_activity),
            "signals": len(trades),
            "executions": len(executions),
            "orders": len(orders)
        }
        
    except Exception as e:
        logger.error(f"Error parsing trades from log: {e}")
        return {"trades": [], "count": 0, "error": str(e)}


@app.get("/api/orders/detailed")
async def get_detailed_orders():
    """Get detailed order information from SQLite database."""
    try:
        tracker = OrderTracker()
        orders = tracker.get_all_orders(limit=100)
        
        logger.info(f"📥 Got {len(orders)} orders from database")
        
        # Transform for frontend
        formatted_orders = []
        for order in orders:
            logger.info(f"Processing order {order['order_id']}")
            # Get detailed info including events
            details = tracker.get_order_details(order['order_id'])
            if not details:
                logger.warning(f"No details for order {order['order_id']}, skipping")
                continue
            
            formatted_order = {
                "order_id": order['order_id'],
                "parent_order_id": order.get('parent_order_id'),
                "timestamp": order['timestamp'],
                "symbol": order['symbol'],
                "action": order['action'],
                "quantity": order['quantity'],
                "order_type": order['order_type'],
                "entry_price": order.get('entry_price'),
                "stop_loss": order.get('stop_loss'),
                "take_profit": order.get('take_profit'),
                "status": order['status'],
                "filled_quantity": order.get('filled_quantity', 0),
                "avg_fill_price": order.get('avg_fill_price'),
                "confidence": order.get('confidence'),
                "atr": order.get('atr'),
                "rationale": order.get('rationale'),
                "features": order.get('features'),
                "market_regime": order.get('market_regime'),
                "commission": order.get('commission'),
                "realized_pnl": order.get('realized_pnl'),
                "execution_time": None,
                "updates": []
            }
            
            # Add events as updates
            for event in details.get('events', []):
                update = {
                    "timestamp": event['timestamp'],
                    "status": event.get('status', event['event_type']),
                    "message": event.get('message', '')
                }
                formatted_order['updates'].append(update)
                
                # Set execution time on first fill
                if event.get('status') == 'Filled' and not formatted_order['execution_time']:
                    formatted_order['execution_time'] = event['timestamp']
            
            # Add execution details
            if details.get('executions'):
                exec_info = []
                for exec_detail in details['executions']:
                    exec_info.append({
                        "timestamp": exec_detail['timestamp'],
                        "quantity": exec_detail['quantity'],
                        "price": exec_detail['price'],
                        "commission": exec_detail.get('commission'),
                        "realized_pnl": exec_detail.get('realized_pnl')
                    })
                formatted_order['executions'] = exec_info
            
            formatted_orders.append(formatted_order)
        
        # Calculate P&L for all orders
        formatted_orders = calculate_pnl_for_orders(formatted_orders)
        
        # Ensure orders are sorted by order_id DESC (latest/highest ID first)
        # Using order_id ensures consistent ordering as IB assigns sequential IDs
        formatted_orders.sort(key=lambda x: x['order_id'], reverse=True)
        
        logger.info(f"📊 Returning {len(formatted_orders)} orders from SQLite database")
        return {
            "orders": formatted_orders,
            "count": len(formatted_orders)
        }
        
    except Exception as e:
        logger.error(f"Error getting orders from database: {e}")
        return {"orders": [], "count": 0, "error": str(e)}
    
    try:
        orders_map = {}  # order_id -> order_info
        
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        current_order = None
        
        for i, line in enumerate(lines):
            try:
                # Extract timestamp
                timestamp_str = line.split("|")[0].strip()
                timestamp = datetime.fromisoformat(timestamp_str.replace(" ", "T"))
                
                # Look for "Placing ORDER" lines - match actual log format
                if "Placing" in line and "order for" in line.lower() and ("BUY" in line or "SELL" in line) and "contracts" in line:
                    action = "BUY" if "BUY" in line else "SELL"
                    
                    # Extract quantity
                    quantity = 0
                    if "for" in line and "contracts" in line:
                        try:
                            qty_str = line.split("for")[1].split("contracts")[0].strip()
                            quantity = int(qty_str)
                        except:
                            pass
                    
                    # Look back and ahead for context
                    price = None
                    stop_loss = None
                    take_profit = None
                    confidence = 0.0
                    atr = None
                    status = "Placed"  # Default status
                    
                    # Look back for signal, price, and trade levels
                    for j in range(max(0, i-15), i):
                        prev_line = lines[j]
                        if "SIGNAL GENERATED:" in prev_line and action in prev_line:
                            if "confidence=" in prev_line:
                                conf_str = prev_line.split("confidence=")[1].split()[0].replace(",", "")
                                confidence = float(conf_str)
                        if "Got last price:" in prev_line or "Current price:" in prev_line:
                            try:
                                price_str = prev_line.split(":")[-1].strip()
                                price = float(price_str)
                            except:
                                pass
                        if "📏 ATR:" in prev_line or "ATR=" in prev_line:
                            try:
                                atr_str = prev_line.split("ATR")[-1].replace(":", "").replace("=", "").strip().split()[0]
                                atr = float(atr_str)
                            except:
                                pass
                        # Look for trade levels line
                        if "Trade levels:" in prev_line:
                            try:
                                if "stop=" in prev_line:
                                    stop_str = prev_line.split("stop=")[1].split()[0]
                                    stop_loss = float(stop_str)
                                if "target=" in prev_line:
                                    target_str = prev_line.split("target=")[1].split()[0]
                                    take_profit = float(target_str)
                                if "entry=" in prev_line:
                                    entry_str = prev_line.split("entry=")[1].split()[0]
                                    price = float(entry_str)
                            except:
                                pass
                    
                    # Look ahead for order details (order ID and status)
                    found_order = False
                    for j in range(i+1, min(i+20, len(lines))):
                        next_line = lines[j]
                        
                        # Look for "Order result:" line with status
                        if "Order result:" in next_line or "✅ Order result:" in next_line:
                            if "status=" in next_line:
                                status_part = next_line.split("status=")[1].split(",")[0].strip()
                                status = status_part
                            if "filled=" in next_line:
                                filled_str = next_line.split("filled=")[1].split()[0].strip()
                                try:
                                    quantity = int(filled_str)
                                except:
                                    pass
                        
                        # Look for "Order ... placed: orderId=" pattern
                        if "placed: orderId=" in next_line or ("Order" in next_line and "orderId=" in next_line):
                            # Extract order ID
                            order_id_str = next_line.split("orderId=")[1].split()[0]
                            order_id = int(order_id_str)
                            
                            # Extract status if present
                            if "status=" in next_line:
                                status = next_line.split("status=")[1].strip()
                            
                            found_order = True
                            break
                    
                    # Create order entry even if we don't have orderId (use timestamp as ID)
                    if not found_order:
                        order_id = int(timestamp.timestamp() * 1000)  # Use timestamp as pseudo ID
                    
                    # Create order entry
                    current_order = {
                        "order_id": order_id,
                        "timestamp": timestamp.isoformat(),
                        "action": action,
                        "quantity": quantity,
                        "entry_price": price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "status": status,
                        "confidence": confidence,
                        "atr": atr,
                        "filled_quantity": 0,
                        "avg_fill_price": None,
                        "execution_time": None,
                        "updates": [{"timestamp": timestamp.isoformat(), "status": "Placed", "message": f"{action} {quantity} contracts"}]
                    }
                    
                    orders_map[order_id] = current_order
                    logger.info(f"Detected order from log: id={order_id}, action={action}, qty={quantity}, entry={price}, SL={stop_loss}, TP={take_profit}")
                
                # Look for order status updates
                if "Order" in line and "status update:" in line:
                    try:
                        order_id = int(line.split("Order ")[1].split()[0])
                        status = line.split("status update:")[1].strip()
                        
                        if order_id in orders_map:
                            orders_map[order_id]["status"] = status
                            orders_map[order_id]["updates"].append({
                                "timestamp": timestamp.isoformat(),
                                "status": status,
                                "message": status
                            })
                            
                            if status == "Filled" and not orders_map[order_id]["execution_time"]:
                                orders_map[order_id]["execution_time"] = timestamp.isoformat()
                    except:
                        pass
                
                # Look for execution details
                if "Execution: order_id=" in line:
                    try:
                        order_id = int(line.split("order_id=")[1].split()[0])
                        qty = int(line.split("qty=")[1].split()[0])
                        fill_price = float(line.split("price=")[1].strip().split()[0])
                        
                        if order_id in orders_map:
                            orders_map[order_id]["filled_quantity"] = qty
                            orders_map[order_id]["avg_fill_price"] = fill_price
                            if not orders_map[order_id]["entry_price"]:
                                orders_map[order_id]["entry_price"] = fill_price
                            orders_map[order_id]["updates"].append({
                                "timestamp": timestamp.isoformat(),
                                "status": "Executed",
                                "message": f"Filled {qty} @ {fill_price:.2f}"
                            })
                    except:
                        pass
                
                # Look for stop loss/take profit triggers
                if ("Stop loss triggered" in line or "Take profit triggered" in line or 
                    "Trailing stop updated" in line):
                    msg = line.split("|")[-1].strip()
                    # Try to associate with most recent order
                    if orders_map:
                        latest_order_id = max(orders_map.keys())
                        orders_map[latest_order_id]["updates"].append({
                            "timestamp": timestamp.isoformat(),
                            "status": "Update",
                            "message": msg
                        })
                
            except Exception as e:
                logger.debug(f"Error parsing order line: {e}")
                continue
        
        # Convert to list and sort by timestamp
        orders_list = list(orders_map.values())
        orders_list.sort(key=lambda x: x["timestamp"], reverse=True)
        
        logger.info(f"Returning {len(orders_list)} orders from API")
        return {
            "orders": orders_list[:50],  # Last 50 orders
            "count": len(orders_list)
        }
        
    except Exception as e:
        logger.error(f"Error getting orders from database: {e}")
        return {"orders": [], "count": 0, "error": str(e)}


@app.get("/api/pnl/summary")
async def get_pnl_summary():
    """Get P&L summary from order database."""
    try:
        tracker = OrderTracker()
        orders = tracker.get_all_orders(limit=1000)
        
        # Calculate P&L
        enhanced_orders = calculate_pnl_for_orders(orders)
        
        # Calculate summary
        total_realized_pnl = 0.0
        total_commission = 0.0
        total_trades = 0
        winning_trades = 0
        losing_trades = 0
        
        # Track current position for unrealized P&L
        current_position = 0
        avg_entry_price = 0.0
        
        for order in enhanced_orders:
            if order['status'] == 'Filled':
                total_trades += 1
                
                # Add realized P&L from round trips
                if 'calculated_pnl' in order and order['calculated_pnl']:
                    pnl = order['calculated_pnl']
                    total_realized_pnl += pnl
                    if pnl > 0:
                        winning_trades += 1
                    elif pnl < 0:
                        losing_trades += 1
                
                # Track commission
                if order.get('commission'):
                    total_commission += order['commission']
                
                # Update current position
                if 'position_after' in order:
                    current_position = order['position_after']
                if 'avg_entry_price' in order and order['avg_entry_price']:
                    avg_entry_price = order['avg_entry_price']
        
        # Get current price for unrealized P&L
        unrealized_pnl = 0.0
        current_price = None
        try:
            # Try to get current price from trading status
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:8000/api/trading/status') as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        current_price = data.get('current_price')
        except:
            pass
        
        if current_price and current_position != 0 and avg_entry_price > 0:
            unrealized_pnl = calculate_unrealized_pnl(current_position, avg_entry_price, current_price)
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        return {
            "total_realized_pnl": round(total_realized_pnl, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "total_pnl": round(total_realized_pnl + unrealized_pnl, 2),
            "total_commission": round(total_commission, 2),
            "net_pnl": round(total_realized_pnl + unrealized_pnl - total_commission, 2),
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": round(win_rate, 2),
            "current_position": current_position,
            "avg_entry_price": round(avg_entry_price, 2) if avg_entry_price > 0 else None,
            "current_price": round(current_price, 2) if current_price else None
        }
        
    except Exception as e:
        logger.error(f"Error calculating P&L summary: {e}")
        return {"error": str(e)}


@app.get("/api/account/balance")
async def get_account_balance():
    """Get account balance from Interactive Brokers."""
    try:
        from ib_insync import IB
        import nest_asyncio
        nest_asyncio.apply()
        
        # Load settings
        settings = load_settings()
        
        # Try to connect
        client_ids_to_try = [998, 997, 996, 995, 994]
        ib = IB()
        connected = False
        
        for client_id in client_ids_to_try:
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(
                        ib.connect,
                        settings.data.ibkr_host,
                        settings.data.ibkr_port,
                        clientId=client_id,
                        timeout=10,
                        readonly=True
                    ),
                    timeout=15.0
                )
                
                if ib.isConnected():
                    connected = True
                    logger.info(f"✅ Connected to IB for account balance (client {client_id})")
                    break
            except:
                if ib.isConnected():
                    try:
                        await asyncio.to_thread(ib.disconnect)
                    except:
                        pass
                continue
        
        if not connected:
            return {
                "success": False,
                "message": "Could not connect to IB"
            }
        
        try:
            # Get account summary
            account_values = await asyncio.to_thread(lambda: ib.accountSummary())
            
            # Extract relevant values
            balance_info = {
                "net_liquidation": 0.0,
                "total_cash_value": 0.0,
                "buying_power": 0.0,
                "excess_liquidity": 0.0,
                "available_funds": 0.0,
                "gross_position_value": 0.0,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0,
                "currency": "USD"
            }
            
            for av in account_values:
                if av.tag == "NetLiquidation" and av.currency == "USD":
                    balance_info["net_liquidation"] = float(av.value)
                elif av.tag == "TotalCashValue" and av.currency == "USD":
                    balance_info["total_cash_value"] = float(av.value)
                elif av.tag == "BuyingPower" and av.currency == "USD":
                    balance_info["buying_power"] = float(av.value)
                elif av.tag == "ExcessLiquidity" and av.currency == "USD":
                    balance_info["excess_liquidity"] = float(av.value)
                elif av.tag == "AvailableFunds" and av.currency == "USD":
                    balance_info["available_funds"] = float(av.value)
                elif av.tag == "GrossPositionValue" and av.currency == "USD":
                    balance_info["gross_position_value"] = float(av.value)
                elif av.tag == "UnrealizedPnL" and av.currency == "USD":
                    balance_info["unrealized_pnl"] = float(av.value)
                elif av.tag == "RealizedPnL" and av.currency == "USD":
                    balance_info["realized_pnl"] = float(av.value)
            
            # Disconnect
            await asyncio.to_thread(ib.disconnect)
            
            return {
                "success": True,
                **balance_info
            }
            
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            if ib.isConnected():
                try:
                    await asyncio.to_thread(ib.disconnect)
                except:
                    pass
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to retrieve account balance"
            }
            
    except Exception as e:
        logger.error(f"Error setting up account balance request: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to initialize connection for account balance"
        }


@app.post("/api/orders/sync")
async def sync_orders_from_ib():
    """Sync orders from Interactive Brokers API to database."""
    try:
        from ib_insync import IB
        import nest_asyncio
        nest_asyncio.apply()
        
        # Load settings to get IB connection info
        settings = load_settings()
        
        # Try multiple client IDs in case some are in use
        client_ids_to_try = [999, 998, 997, 996, 995]
        ib = IB()
        connected = False
        used_client_id = None
        
        logger.info("🔌 Attempting to connect to IB for order sync...")
        
        for client_id in client_ids_to_try:
            try:
                logger.info(f"Trying client ID {client_id}...")
                await asyncio.wait_for(
                    asyncio.to_thread(
                        ib.connect,
                        settings.data.ibkr_host,
                        settings.data.ibkr_port,
                        clientId=client_id,
                        timeout=15,
                        readonly=True  # Read-only connection
                    ),
                    timeout=20.0
                )
                
                if ib.isConnected():
                    connected = True
                    used_client_id = client_id
                    logger.info(f"✅ Connected to IB with client ID {client_id}")
                    break
            except Exception as e:
                logger.debug(f"Client ID {client_id} failed: {e}")
                if ib.isConnected():
                    try:
                        await asyncio.to_thread(ib.disconnect)
                    except:
                        pass
                continue
        
        if not connected:
            return {
                "success": False,
                "error": "Connection failed",
                "message": "Could not connect to IB Gateway/TWS. All client IDs in use or IB not running."
            }
        
        try:
            # Get all trades (open and completed) - run in thread pool
            trades = await asyncio.to_thread(lambda: ib.trades())
            logger.info(f"📥 Found {len(trades)} trades from IB")
            
            # Get order IDs from IB
            ib_order_ids = set(trade.order.orderId for trade in trades)
            
            tracker = OrderTracker()
            synced_count = 0
            updated_count = 0
            deleted_count = 0
            
            # First, delete orders from DB that are not in IB anymore
            db_orders = tracker.get_all_orders(limit=10000)
            logger.info(f"📊 Checking {len(db_orders)} DB orders vs {len(ib_order_ids)} IB orders")
            
            for db_order in db_orders:
                order_id = db_order['order_id']
                order_status = db_order['status']
                
                # Check if order is in IB
                if order_id not in ib_order_ids:
                    # Delete non-filled, non-cancelled orders
                    if order_status not in ['Filled', 'Cancelled']:
                        try:
                            tracker.conn.execute("DELETE FROM orders WHERE order_id = ?", (order_id,))
                            tracker.conn.execute("DELETE FROM order_events WHERE order_id = ?", (order_id,))
                            tracker.conn.execute("DELETE FROM executions WHERE order_id = ?", (order_id,))
                            tracker.conn.commit()
                            deleted_count += 1
                            logger.info(f"🗑️ Deleted {order_id} (status={order_status})")
                        except Exception as e:
                            logger.error(f"Error deleting {order_id}: {e}")
                    else:
                        logger.info(f"📋 Keeping {order_id} (status={order_status}, historical)")
            
            logger.info(f"🧹 Deleted {deleted_count} stale orders")
            
            # Now sync orders from IB
            for trade in trades:
                try:
                    order = trade.order
                    contract = trade.contract
                    order_status = trade.orderStatus
                    
                    # Check if order already exists in database
                    existing = tracker.get_order_details(order.orderId)
                    
                    if existing:
                        # Update existing order
                        tracker.update_order_status(
                            order_id=order.orderId,
                            status=order_status.status,
                            filled=int(order_status.filled),
                            remaining=int(order_status.remaining),
                            avg_fill_price=float(order_status.avgFillPrice) if order_status.avgFillPrice > 0 else None
                        )
                        updated_count += 1
                    else:
                        # Add new order to database
                        tracker.record_order_placement(
                            order_id=order.orderId,
                            parent_order_id=order.parentId if order.parentId > 0 else None,
                            symbol=contract.symbol,
                            action=order.action,
                            quantity=int(order.totalQuantity),
                            order_type=order.orderType,
                            entry_price=float(order.lmtPrice) if order.lmtPrice else None,
                            stop_loss=float(order.auxPrice) if order.auxPrice else None,
                            take_profit=None,  # Will be set from bracket orders
                            confidence=None,
                            atr=None
                        )
                        synced_count += 1
                    
                    # Record fills if available
                    if order_status.filled > 0 and order_status.avgFillPrice > 0:
                        # Check if execution already recorded
                        details = tracker.get_order_details(order.orderId)
                        if details and not details.get('executions'):
                            tracker.record_execution(
                                order_id=order.orderId,
                                quantity=int(order_status.filled),
                                price=float(order_status.avgFillPrice),
                                commission=float(order_status.commission) if order_status.commission else None,
                                realized_pnl=float(order_status.realizedPNL) if order_status.realizedPNL else None
                            )
                except Exception as e:
                    logger.error(f"Error processing trade {trade.order.orderId}: {e}")
                    continue
            
            # Disconnect
            await asyncio.to_thread(ib.disconnect)
            logger.info(f"✅ Sync complete: {synced_count} new, {updated_count} updated, {deleted_count} deleted")
            
            return {
                "success": True,
                "synced": synced_count,
                "updated": updated_count,
                "deleted": deleted_count,
                "total": len(trades),
                "message": f"Synced {synced_count} new, updated {updated_count}, deleted {deleted_count} stale orders from IB"
            }
            
        except Exception as e:
            logger.error(f"Error during IB sync: {e}")
            if ib.isConnected():
                try:
                    await asyncio.to_thread(ib.disconnect)
                except:
                    pass
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to sync orders from IB: {str(e)}"
            }
            
    except Exception as e:
        logger.error(f"Error setting up IB sync: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to initialize IB connection for sync"
        }


@app.get("/api/trading/today-summary")
async def get_today_trading_summary():
    """
    Get today's trading summary including orders, fills, P&L, and statistics.
    
    This endpoint provides a comprehensive view of today's trading activity
    from the orders database and optionally parses the live trading log for
    additional details like realized P&L from IB executions.
    """
    try:
        from pathlib import Path
        import re
        import pytz
        
        # CST timezone
        cst = pytz.timezone('America/Chicago')
        now_cst = datetime.now(cst)
        today_cst = now_cst.date().isoformat()
        
        tracker = OrderTracker()
        today = datetime.now().date().isoformat()
        tomorrow = (datetime.now().date() + timedelta(days=1)).isoformat()
        
        # Get today's orders from database
        orders = tracker.get_orders_by_date(today, tomorrow)
        
        # Calculate statistics
        total_orders = len(orders)
        filled_orders = [o for o in orders if o['status'] == 'Filled']
        cancelled_orders = [o for o in orders if o['status'] == 'Cancelled']
        pending_orders = [o for o in orders if o['status'] not in ['Filled', 'Cancelled', 'Inactive']]
        
        # Calculate P&L from database
        db_total_pnl = sum(o.get('realized_pnl') or 0 for o in orders)
        db_total_commission = sum(o.get('commission') or 0 for o in orders)
        
        # Parse live trading log for additional P&L data
        log_pnl_entries = []
        log_total_pnl = 0.0
        log_total_commission = 0.0
        winning_trades = 0
        losing_trades = 0
        
        # Try multiple possible log paths
        log_paths = [
            Path("logs/live_trading.log"),
            Path("/Users/svss/Documents/code/MyTrader/logs/live_trading.log"),
            project_root / "logs" / "live_trading.log"
        ]
        
        log_file = None
        for lp in log_paths:
            if lp.exists():
                log_file = lp
                break
        
        if log_file:
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        # Only parse today's entries (check date in line)
                        if today not in line:
                            continue
                            
                        # Parse realizedPNL entries (non-zero only for win/loss counting)
                        if "realizedPNL=" in line:
                            match = re.search(r'realizedPNL=(-?[\d.]+)', line)
                            if match:
                                pnl = float(match.group(1))
                                if pnl != 0:  # Only count non-zero P&L for statistics
                                    log_pnl_entries.append(pnl)
                                    log_total_pnl += pnl
                                    if pnl > 0:
                                        winning_trades += 1
                                    elif pnl < 0:
                                        losing_trades += 1
                        
                        # Parse commission entries
                        if "commission=" in line:
                            match = re.search(r'commission=(-?[\d.]+)', line)
                            if match:
                                log_total_commission += float(match.group(1))
            except Exception as e:
                logger.warning(f"Error parsing log file: {e}")
        
        # Use log data if available, otherwise use database data
        total_pnl = log_total_pnl if log_total_pnl != 0 else db_total_pnl
        total_commission = log_total_commission if log_total_commission != 0 else db_total_commission
        net_pnl = total_pnl - total_commission
        
        # Calculate win rate
        total_closed_trades = winning_trades + losing_trades
        win_rate = (winning_trades / total_closed_trades * 100) if total_closed_trades > 0 else 0
        
        # Helper to convert timestamp to CST
        def to_cst(timestamp_str):
            try:
                # Parse the timestamp
                if 'T' in timestamp_str:
                    dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    dt = datetime.fromisoformat(timestamp_str)
                
                # If naive, assume UTC
                if dt.tzinfo is None:
                    dt = pytz.utc.localize(dt)
                
                # Convert to CST
                dt_cst = dt.astimezone(cst)
                return dt_cst.strftime('%Y-%m-%d %I:%M:%S %p CST')
            except:
                return timestamp_str
        
        # Build summary with CST times
        summary = {
            "date": today_cst,
            "date_cst": now_cst.strftime('%Y-%m-%d'),
            "timestamp": now_cst.strftime('%Y-%m-%d %I:%M:%S %p CST'),
            "orders": {
                "total": total_orders,
                "filled": len(filled_orders),
                "cancelled": len(cancelled_orders),
                "pending": len(pending_orders)
            },
            "performance": {
                "realized_pnl": round(total_pnl, 2),
                "commission": round(total_commission, 2),
                "net_pnl": round(net_pnl, 2),
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": round(win_rate, 1),
                "pnl_per_trade": round(total_pnl / total_closed_trades, 2) if total_closed_trades > 0 else 0
            },
            "trades": [
                {
                    "order_id": o['order_id'],
                    "symbol": o['symbol'],
                    "action": o['action'],
                    "quantity": o['quantity'],
                    "status": o['status'],
                    "avg_fill_price": o.get('avg_fill_price'),
                    "realized_pnl": o.get('realized_pnl'),
                    "timestamp": o['timestamp'],
                    "timestamp_cst": to_cst(o['timestamp'])
                }
                for o in filled_orders[:20]  # Last 20 filled trades
            ],
            "status": "success"
        }
        
        logger.info(f"📊 Today's summary: {len(filled_orders)} fills, ${net_pnl:.2f} net P&L, {winning_trades}W/{losing_trades}L")
        return summary
        
    except Exception as e:
        logger.error(f"Error getting today's trading summary: {e}")
        return {
            "status": "error",
            "error": str(e),
            "date": datetime.now().date().isoformat()
        }


class PerformanceAnalysisRequest(BaseModel):
    period: str = "today"  # today, yesterday, week, all


@app.post("/api/analysis/performance")
async def analyze_performance(request: PerformanceAnalysisRequest):
    """Generate AI analysis of trading performance."""
    try:
        from mytrader.llm.bedrock_client import BedrockClient
        from mytrader.utils.settings_loader import load_settings
        
        # Get trades
        tracker = OrderTracker()
        orders = tracker.get_all_orders(limit=100)
        enhanced_orders = calculate_pnl_for_orders(orders)
        
        # Filter by period
        now = datetime.now()
        filtered_orders = []
        
        for order in enhanced_orders:
            order_time = datetime.fromisoformat(order['timestamp'])
            
            if request.period == "today":
                if order_time.date() == now.date():
                    filtered_orders.append(order)
            elif request.period == "yesterday":
                if order_time.date() == (now - timedelta(days=1)).date():
                    filtered_orders.append(order)
            elif request.period == "week":
                if order_time > (now - timedelta(days=7)):
                    filtered_orders.append(order)
            else:
                filtered_orders.append(order)
        
        if not filtered_orders:
            return {
                "analysis": "No trades found for the specified period.",
                "period": request.period
            }
        
        # Prepare data for LLM
        trades_summary = []
        total_pnl = 0
        wins = 0
        losses = 0
        
        for order in filtered_orders:
            if order['status'] == 'Filled':
                pnl = order.get('calculated_pnl', 0)
                total_pnl += pnl
                if pnl > 0: wins += 1
                elif pnl < 0: losses += 1
                
                trades_summary.append(
                    f"Action: {order['action']}, Symbol: {order['symbol']}, "
                    f"PnL: ${pnl:.2f}, Rationale: {order.get('rationale', 'N/A')}, "
                    f"Market Regime: {order.get('market_regime', 'N/A')}"
                )
        
        summary_text = "\n".join(trades_summary)
        
        # Generate analysis
        settings = load_settings()
        client = BedrockClient(
            model_id=settings.llm.model_id,
            region_name=settings.llm.region_name
        )
        
        prompt = f"""
        Analyze the following trading performance for {request.period}:
        
        Total PnL: ${total_pnl:.2f}
        Win Rate: {wins}/{wins+losses} ({wins/(wins+losses)*100:.1f}%)
        
        Trades:
        {summary_text}
        
        Please provide:
        1. A summary of what happened.
        2. Analysis of why losses occurred (look at rationale and regime).
        3. Suggestions for improvement.
        4. Assessment of whether the strategy respected the market regime.
        
        Keep it concise and actionable.
        """
        
        analysis = client.generate_text(prompt)
        
        return {
            "analysis": analysis,
            "period": request.period,
            "stats": {
                "total_pnl": total_pnl,
                "trades": len(filtered_orders),
                "win_rate": wins/(wins+losses) if (wins+losses) > 0 else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/performance")
async def get_performance():
    """Get current performance metrics from live trading logs."""
    log_file = project_root / "logs" / "live_trading.log"
    
    # Default metrics
    metrics = {
        "total_pnl": 0.0,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "total_return": 0.0,
        "current_position": 0,
        "total_trades": 0,
        "total_signals": 0,
        "hold_signals": 0,
        "buy_signals": 0,
        "sell_signals": 0
    }
    
    if not log_file.exists():
        return metrics
    
    try:
        # Parse log for position and signal info
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        buy_count = 0
        sell_count = 0
        hold_count = 0
        last_position = 0
        
        for line in lines:
            # Count signals
            if "SIGNAL GENERATED:" in line:
                if "BUY" in line:
                    buy_count += 1
                elif "SELL" in line:
                    sell_count += 1
                elif "HOLD" in line:
                    hold_count += 1
            
            # Get latest position
            if "Current position:" in line and "contracts" in line:
                try:
                    # Extract position number
                    parts = line.split("Current position:")[1].split("contracts")[0]
                    last_position = int(parts.strip())
                except:
                    pass
        
        metrics.update({
            "total_signals": buy_count + sell_count + hold_count,
            "buy_signals": buy_count,
            "sell_signals": sell_count,
            "hold_signals": hold_count,
            "current_position": last_position,
            "total_trades": buy_count + sell_count  # Trades excluding HOLD
        })
        
        # Try to get P&L from IBKR if the trading system is connected
        # Note: P&L values are logged but with format strings, so we can't parse them reliably
        # The actual P&L is tracked in the executor but not accessible from dashboard
        metrics["note"] = "P&L values require direct IBKR connection. Check IBKR account for realized P&L."
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        return metrics


@app.get("/api/equity-curve")
async def get_equity_curve(limit: int = 100):
    """Get equity curve data."""
    global performance_tracker
    
    if not performance_tracker:
        return {"data": []}
    
    equity_data = []
    if hasattr(performance_tracker, 'equity_curve') and performance_tracker.equity_curve:
        # equity_curve is a deque of (timestamp, equity) tuples
        recent_data = list(performance_tracker.equity_curve)[-limit:]
        equity_data = [
            {
                "timestamp": timestamp.isoformat(),
                "equity": float(equity)
            }
            for timestamp, equity in recent_data
        ]
    
    return {"data": equity_data}


@app.get("/api/config")
async def get_config():
    """Get current trading configuration."""
    try:
        config = load_settings("config.yaml")
        
        return {
            "ibkr_port": config.data.ibkr_port,
            "max_position_size": config.trading.max_position_size,
            "max_daily_loss": config.trading.max_daily_loss,
            "stop_loss_ticks": config.trading.stop_loss_ticks,
            "take_profit_ticks": config.trading.take_profit_ticks,
            "strategies": [s.name for s in config.strategies] if hasattr(config, 'strategies') else []
        }
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {
            "error": str(e),
            "ibkr_port": 4002,
            "max_position_size": 2,
            "max_daily_loss": 1500.0,
        }


@app.get("/api/reports")
async def get_reports():
    """Get list of available performance reports."""
    reports_dir = Path("reports")
    
    if not reports_dir.exists():
        return {"reports": []}
    
    reports = []
    for report_file in reports_dir.glob("paper_trade_*.json"):
        try:
            with open(report_file, 'r') as f:
                data = json.load(f)
            
            reports.append({
                "filename": report_file.name,
                "timestamp": data.get('generated_at', 'unknown'),
                "total_pnl": data.get('snapshot', {}).get('total_pnl', 0),
                "total_trades": data.get('snapshot', {}).get('trade_count', 0),
            })
        except Exception as e:
            logger.error(f"Error reading report {report_file}: {e}")
    
    # Sort by timestamp descending
    reports.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return {"reports": reports[:20]}  # Last 20 reports


# Backtesting endpoints
@app.post("/api/backtest/download-data")
async def download_backtest_data(request: DataDownloadRequest):
    """Download historical data for backtesting."""
    try:
        import subprocess
        from pathlib import Path
        
        # Run the download script
        script_path = Path("scripts/download_data.py")
        output_file = f"data/es_{request.start_date}_to_{request.end_date}.csv"
        
        cmd = [
            "python", str(script_path),
            "--start", request.start_date,
            "--end", request.end_date,
            "--output", output_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Data download failed: {result.stderr}"
            )
        
        return {
            "status": "success",
            "message": "Data downloaded successfully",
            "file": output_file,
            "start_date": request.start_date,
            "end_date": request.end_date
        }
    
    except Exception as e:
        logger.error(f"Failed to download data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/backtest/run")
async def run_backtest(request: BacktestRequest):
    """Run backtest on historical data."""
    try:
        from mytrader.backtesting.engine import BacktestingEngine
        from mytrader.utils.settings_loader import load_settings
        from mytrader.features.feature_engineer import add_technical_indicators
        from mytrader.strategies.rsi_macd_sentiment import RsiMacdSentimentStrategy
        from mytrader.strategies.momentum_reversal import MomentumReversalStrategy
        import pandas as pd
        
        # Download data if requested
        data_file = f"data/es_{request.start_date}_to_{request.end_date}.csv"
        
        if request.download_data:
            download_req = DataDownloadRequest(
                start_date=request.start_date,
                end_date=request.end_date
            )
            await download_backtest_data(download_req)
        
        # Check if data file exists
        if not Path(data_file).exists():
            raise HTTPException(
                status_code=404,
                detail=f"Data file not found: {data_file}. Enable download_data=true to fetch it."
            )
        
        # Load configuration
        config = load_settings(request.config_path)
        
        # Load data
        df = pd.read_csv(data_file, parse_dates=['timestamp'])
        df = df.set_index('timestamp')
        
        # Add technical indicators
        df = add_technical_indicators(df)
        
        # Initialize strategies
        strategies = [
            RsiMacdSentimentStrategy(),
            MomentumReversalStrategy()
        ]
        
        # Initialize backtesting engine with proper config objects
        engine = BacktestingEngine(
            strategies=strategies,
            trading_config=config.trading,
            backtest_config=config.backtest
        )
        
        # Run backtest
        result = engine.run(df)
        
        # Save results
        results_file = f"reports/backtest_{request.start_date}_to_{request.end_date}.json"
        Path("reports").mkdir(exist_ok=True)
        
        backtest_report = {
            "start_date": request.start_date,
            "end_date": request.end_date,
            "config": request.config_path,
            "generated_at": datetime.now().isoformat(),
            "metrics": result.metrics,
            "total_trades": len(result.trades),
            "equity_curve_points": len(result.equity_curve)
        }
        
        with open(results_file, 'w') as f:
            json.dump(backtest_report, f, indent=2)
        
        # Convert equity curve to list for JSON response
        equity_curve_data = [
            {"timestamp": ts.isoformat(), "equity": float(equity)}
            for ts, equity in result.equity_curve.items()
        ]
        
        # Convert trades to list
        trades_data = [
            {
                "timestamp": t["timestamp"].isoformat() if isinstance(t["timestamp"], pd.Timestamp) else t["timestamp"],
                "action": t["action"],
                "price": float(t["price"]),
                "quantity": int(t["qty"]),
                "pnl": float(t.get("realized", 0))
            }
            for t in result.trades
        ]
        
        return {
            "status": "success",
            "message": "Backtest completed successfully",
            "results_file": results_file,
            "metrics": result.metrics,
            "equity_curve": equity_curve_data[-200:],  # Last 200 points
            "trades": trades_data[-50:],  # Last 50 trades
            "total_trades": len(result.trades)
        }
    
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/backtest/results")
async def get_backtest_results():
    """Get list of available backtest results."""
    reports_dir = Path("reports")
    
    if not reports_dir.exists():
        return {"results": []}
    
    results = []
    for report_file in reports_dir.glob("backtest_*.json"):
        try:
            with open(report_file, 'r') as f:
                data = json.load(f)
            
            results.append({
                "filename": report_file.name,
                "start_date": data.get('start_date', 'unknown'),
                "end_date": data.get('end_date', 'unknown'),
                "timestamp": data.get('generated_at', 'unknown'),
                "total_return": data.get('metrics', {}).get('total_return', 0),
                "sharpe_ratio": data.get('metrics', {}).get('sharpe_ratio', 0),
                "max_drawdown": data.get('metrics', {}).get('max_drawdown', 0),
                "total_trades": data.get('total_trades', 0),
            })
        except Exception as e:
            logger.error(f"Error reading backtest result {report_file}: {e}")
    
    # Sort by timestamp descending
    results.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return {"results": results[:20]}  # Last 20 results


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    
    try:
        # Send initial status
        await websocket.send_json({
            "type": "connected",
            "message": "WebSocket connection established",
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep connection alive and handle incoming messages
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)


def parse_log_line(line: str) -> dict:
    """Parse a log line into structured data for the UI."""
    line = line.strip()
    
    # Parse signal generation
    if "📊 SIGNAL GENERATED:" in line:
        parts = line.split("SIGNAL GENERATED:")[1].strip().split(",")
        signal = parts[0].strip()
        confidence = None
        if len(parts) > 1 and "confidence" in parts[1]:
            confidence = float(parts[1].split("=")[1].strip())
        
        return {
            "type": "signal",
            "signal": signal,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
    
    # Parse order placement with details - updated to match actual log format
    elif "� Placing" in line and ("BUY" in line or "SELL" in line):
        action = "BUY" if "BUY" in line else "SELL"
        # Extract quantity if present
        quantity = None
        if "for" in line and "contracts" in line:
            try:
                qty_part = line.split("for")[1].split("contracts")[0].strip()
                quantity = int(qty_part)
            except:
                pass
        
        return {
            "type": "order_placing",
            "action": action,
            "quantity": quantity,
            "status": "placing",
            "message": line.split(":")[-1].strip() if ":" in line else line,
            "timestamp": datetime.now().isoformat()
        }
    
    # Parse order placed confirmation with ID
    elif "Order" in line and "placed: orderId=" in line:
        try:
            order_id = int(line.split("orderId=")[1].split()[0])
            action = "BUY" if "BUY" in line else "SELL"
            status = "Submitted"
            if "status=" in line:
                status = line.split("status=")[1].strip()
            
            return {
                "type": "order_placed",
                "order_id": order_id,
                "action": action,
                "status": status,
                "timestamp": datetime.now().isoformat()
            }
        except:
            pass
    
    # Parse order status updates
    elif "✅ Order filled" in line or "❌ Order" in line:
        status = "filled" if "✅" in line else "rejected"
        order_info = {}
        
        # Try to extract order ID
        if "order_id=" in line or "Order" in line:
            try:
                if "order_id=" in line:
                    order_id = int(line.split("order_id=")[1].split()[0])
                    order_info["order_id"] = order_id
            except:
                pass
        
        return {
            "type": "order_update",
            "status": status,
            "message": line.split(":", 1)[1].strip() if ":" in line else line,
            "timestamp": datetime.now().isoformat(),
            **order_info
        }
    
    # Parse execution details
    elif "Execution: order_id=" in line:
        try:
            order_id = int(line.split("order_id=")[1].split()[0])
            qty = int(line.split("qty=")[1].split()[0])
            price = float(line.split("price=")[1].strip().split()[0])
            
            return {
                "type": "execution",
                "order_id": order_id,
                "quantity": qty,
                "price": price,
                "status": "Filled",
                "timestamp": datetime.now().isoformat()
            }
        except:
            pass
    
    # Parse stop loss and take profit
    elif "📍 Stop Loss:" in line:
        try:
            stop_loss = float(line.split("📍 Stop Loss:")[1].strip().split()[0])
            return {
                "type": "stop_loss",
                "stop_loss": stop_loss,
                "timestamp": datetime.now().isoformat()
            }
        except:
            pass
    
    elif "🎯 Take Profit:" in line:
        try:
            take_profit = float(line.split("🎯 Take Profit:")[1].strip().split()[0])
            return {
                "type": "take_profit",
                "take_profit": take_profit,
                "timestamp": datetime.now().isoformat()
            }
        except:
            pass
    
    # Parse data collection progress - handle both formats
    elif "bars collected" in line.lower() or "Building history:" in line:
        import re
        # Try "Building history: X/50 bars" format first
        match = re.search(r'Building history:\s*(\d+)/(\d+)', line)
        if match:
            bars = int(match.group(1))
            total = int(match.group(2))
            return {
                "type": "progress",
                "bars_collected": bars,
                "min_bars_needed": total,
                "timestamp": datetime.now().isoformat()
            }
        # Try "X bars collected" format
        match = re.search(r'(\d+)\s*bars', line)
        if match:
            bars = int(match.group(1))
            return {
                "type": "progress",
                "bars_collected": bars,
                "timestamp": datetime.now().isoformat()
            }
    
    # Parse current price
    elif "Current price:" in line or "Price:" in line:
        import re
        match = re.search(r'(\d+\.\d+)', line)
        if match:
            price = float(match.group(1))
            return {
                "type": "price_update",
                "price": price,
                "timestamp": datetime.now().isoformat()
            }
    
    return None


# Background task to run trading session
async def run_trading_session():
    """Run trading session in background and broadcast updates."""
    global trading_session, performance_tracker
    
    try:
        # This would be integrated with your actual trading loop
        # For now, we'll simulate it
        
        while trading_session and not trading_session.stop_requested:
            # Update performance tracker
            if hasattr(trading_session, 'tracker'):
                performance_tracker = trading_session.tracker
                
                # Get current snapshot
                snapshot = performance_tracker.get_snapshot()
                
                # Calculate return percentage
                total_return = (snapshot.equity / performance_tracker.initial_capital - 1) * 100
                
                # Broadcast update to all clients
                await manager.broadcast({
                    "type": "performance_update",
                    "data": {
                        "total_pnl": snapshot.total_pnl,
                        "total_return": total_return,
                        "sharpe_ratio": snapshot.sharpe_ratio,
                        "max_drawdown": snapshot.max_drawdown,
                        "total_trades": snapshot.trade_count,
                    },
                    "timestamp": datetime.now().isoformat()
                })
            
            # Sleep for 5 seconds before next update
            await asyncio.sleep(5)
    
    except Exception as e:
        logger.error(f"Error in trading session: {e}")
        await manager.broadcast({
            "type": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        })


# =============================================================================
# SPY FUTURES DAILY INSIGHTS API
# =============================================================================

class SPYTradingSummaryRequest(BaseModel):
    """Request model for SPY Futures trading summary."""
    date: str  # YYYY-MM-DD
    performance: Dict[str, Any]
    observations: List[str]
    suggestions: Dict[str, Any]
    warnings: Optional[List[str]] = []
    insights: Optional[List[Dict]] = []
    profitable_patterns: Optional[List[str]] = []
    losing_patterns: Optional[List[str]] = []


@app.post("/api/trading-summary")
async def post_trading_summary(summary: SPYTradingSummaryRequest):
    """
    Receive SPY Futures daily trading summary and insights.
    
    This endpoint receives structured daily performance data and LLM insights
    for SPY Futures trading, stores it, and broadcasts to connected clients.
    """
    try:
        logger.info(f"Received SPY trading summary for {summary.date}")
        
        # Save to file for persistence
        reports_dir = Path("dashboard/backend/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        summary_file = reports_dir / f"spy_summary_{summary.date}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary.dict(), f, indent=2, default=str)
        
        logger.info(f"Saved summary to {summary_file}")
        
        # Broadcast to all connected WebSocket clients
        await manager.broadcast({
            "type": "spy_trading_summary",
            "data": summary.dict(),
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "status": "success",
            "message": f"Trading summary received for {summary.date}",
            "saved_to": str(summary_file)
        }
    
    except Exception as e:
        logger.error(f"Error processing trading summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/spy-futures/latest-summary")
async def get_latest_spy_summary():
    """
    Get the most recent SPY Futures trading summary.
    
    Returns the latest saved summary from the reports directory.
    """
    try:
        reports_dir = Path("dashboard/backend/reports")
        
        if not reports_dir.exists():
            return {
                "status": "no_data",
                "message": "No reports directory found"
            }
        
        # Find most recent summary file
        summary_files = list(reports_dir.glob("spy_summary_*.json"))
        
        if not summary_files:
            return {
                "status": "no_data",
                "message": "No SPY Futures summaries found"
            }
        
        # Get most recent file
        latest_file = max(summary_files, key=lambda p: p.stat().st_mtime)
        
        # Load and return
        with open(latest_file, 'r') as f:
            data = json.load(f)
        
        return {
            "status": "success",
            "data": data,
            "file": str(latest_file),
            "timestamp": datetime.fromtimestamp(latest_file.stat().st_mtime).isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error loading latest summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/spy-futures/summary-history")
async def get_spy_summary_history(days: int = 7):
    """
    Get SPY Futures summary history for the last N days.
    
    Args:
        days: Number of days of history to retrieve (default: 7)
    """
    try:
        reports_dir = Path("dashboard/backend/reports")
        
        if not reports_dir.exists():
            return {
                "status": "no_data",
                "message": "No reports directory found",
                "summaries": []
            }
        
        # Find all summary files
        summary_files = list(reports_dir.glob("spy_summary_*.json"))
        
        if not summary_files:
            return {
                "status": "no_data",
                "message": "No SPY Futures summaries found",
                "summaries": []
            }
        
        # Sort by modification time (most recent first)
        summary_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        # Load recent summaries
        summaries = []
        for filepath in summary_files[:days]:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    summaries.append(data)
            except Exception as e:
                logger.warning(f"Could not load {filepath}: {e}")
                continue
        
        return {
            "status": "success",
            "count": len(summaries),
            "summaries": summaries
        }
    
    except Exception as e:
        logger.error(f"Error loading summary history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/spy-futures/run-analysis")
async def run_spy_analysis(days: int = 1):
    """
    Trigger SPY Futures daily analysis manually.
    
    This endpoint runs the complete analysis pipeline:
    1. Load recent SPY Futures trades
    2. Compute performance metrics
    3. Generate LLM insights
    4. Save and broadcast results
    
    Args:
        days: Number of days to analyze (default: 1)
    """
    try:
        logger.info(f"Running SPY Futures analysis for last {days} day(s)")
        
        # Import here to avoid circular dependencies
        from mytrader.llm.spy_futures_orchestrator import run_spy_futures_daily_review
        
        # Run analysis
        result = await asyncio.to_thread(run_spy_futures_daily_review, days=days)
        
        if not result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Analysis failed")
            )
        
        # Broadcast results via WebSocket
        await manager.broadcast({
            "type": "spy_analysis_complete",
            "data": result.get("report", {}),
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "status": "success",
            "message": "SPY Futures analysis completed",
            "result": result
        }
    
    except Exception as e:
        logger.error(f"Error running SPY analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# HYBRID BEDROCK ARCHITECTURE ENDPOINTS
# =============================================================================

# Global Bedrock components (initialized on demand)
_hybrid_bedrock_client = None
_event_detector = None


def _get_bedrock_client():
    """Get or initialize the hybrid Bedrock client."""
    global _hybrid_bedrock_client
    
    if _hybrid_bedrock_client is None:
        try:
            from mytrader.llm.bedrock_hybrid_client import init_bedrock_client
            settings = load_settings()
            
            _hybrid_bedrock_client = init_bedrock_client(
                model_id=settings.llm.model_id if hasattr(settings, 'llm') else None,
                region_name=settings.llm.region_name if hasattr(settings, 'llm') else None,
                max_tokens=300,
                db_path="data/bedrock_hybrid.db",
            )
            logger.info("Initialized hybrid Bedrock client for API")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            raise HTTPException(status_code=500, detail=f"Bedrock initialization failed: {e}")
    
    return _hybrid_bedrock_client


def _get_event_detector():
    """Get or initialize the event detector."""
    global _event_detector
    
    if _event_detector is None:
        try:
            from mytrader.llm.event_detector import create_event_detector
            settings = load_settings()
            
            symbol = settings.data.ibkr_symbol if settings.data.ibkr_symbol in ["MES", "ES"] else "MES"
            _event_detector = create_event_detector(symbol=symbol)
            logger.info(f"Initialized event detector for {symbol}")
        except Exception as e:
            logger.error(f"Failed to initialize event detector: {e}")
            raise HTTPException(status_code=500, detail=f"Event detector initialization failed: {e}")
    
    return _event_detector


# === Force Cancel Orders Endpoint ===
class ForceCancelResponse(BaseModel):
    """Response model for force cancel orders."""
    status: str
    canceled_count: int
    remaining_orders: int
    message: str


@app.post("/api/orders/force-cancel", response_model=ForceCancelResponse)
async def force_cancel_orders():
    """
    Force cancel all orders for the trading symbol.
    
    This is a nuclear option for when orders are stuck and blocking new trades.
    Use with caution - it will attempt to cancel all orders including those
    from other client IDs.
    
    Returns:
        - Number of orders canceled
        - Number of orders still remaining
        - Status message
    """
    global live_trading_manager
    
    if not live_trading_manager or not live_trading_manager.executor:
        raise HTTPException(
            status_code=400,
            detail="Trading manager not initialized. Start trading first."
        )
    
    try:
        executor = live_trading_manager.executor
        
        # Get count before
        before_count = executor.get_active_order_count(sync=True)
        logger.info(f"🚨 Force cancel requested. {before_count} orders before cancel.")
        
        # Force cancel
        canceled_count = await executor.force_cancel_all_orders()
        
        # Get count after
        after_count = executor.get_active_order_count(sync=True)
        
        return ForceCancelResponse(
            status="success" if after_count == 0 else "partial",
            canceled_count=canceled_count,
            remaining_orders=after_count,
            message=f"Canceled {canceled_count} orders. {after_count} remaining."
        )
        
    except Exception as e:
        logger.error(f"Force cancel failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class TriggerBedrockRequest(BaseModel):
    """Request model for manual Bedrock trigger."""
    notes: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class TriggerBedrockResponse(BaseModel):
    """Response model for Bedrock trigger."""
    request_id: str
    status: str
    message: str
    result: Optional[Dict[str, Any]] = None


@app.post("/api/trigger-bedrock", response_model=TriggerBedrockResponse)
async def trigger_bedrock(request: TriggerBedrockRequest):
    """
    Manually trigger a Bedrock analysis.
    
    This endpoint allows manual triggering of Bedrock LLM analysis
    for market assessment. The result is a bias modifier, not a trade signal.
    
    Security: Localhost only or requires auth.
    """
    try:
        from mytrader.llm.rag_context_builder import RAGContextBuilder
        from mytrader.llm.event_detector import EventPayload
        import uuid
        
        # Generate request ID
        request_id = str(uuid.uuid4())[:8]
        
        # Get Bedrock client
        bedrock_client = _get_bedrock_client()
        
        # Build context from request or use defaults
        settings = load_settings()
        symbol = settings.data.ibkr_symbol if settings.data.ibkr_symbol in ["MES", "ES"] else "MES"
        
        context_builder = RAGContextBuilder(instrument=symbol)
        
        # Create payload from request context or use defaults
        if request.context:
            context = context_builder.build_context_from_snapshot(
                snapshot=request.context,
                trigger_type="manual",
                reason=request.notes or "Manual trigger via API"
            )
        else:
            # Create minimal context
            payload = EventPayload(
                trigger_type="manual",
                reason=request.notes or "Manual trigger via API",
                timestamp=datetime.now(),
                symbol=symbol,
                current_price=request.context.get("current_price", 0.0) if request.context else 0.0,
            )
            context = context_builder.build_context(payload)
        
        # Call Bedrock
        result = bedrock_client.bedrock_analyze(
            context=context,
            trigger="manual",
        )
        
        # Also mark manual trigger in event detector for main loop
        event_detector = _get_event_detector()
        event_detector.set_manual_trigger(notes=request.notes)
        
        return TriggerBedrockResponse(
            request_id=request_id,
            status="success",
            message=f"Bedrock analysis complete: {result.get('bias', 'NEUTRAL')}",
            result=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in trigger-bedrock: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class BedrockStatusResponse(BaseModel):
    """Response model for Bedrock status."""
    model_id: str
    region: str
    daily_calls: int
    daily_cost: float
    quota_status: str
    within_quota: bool
    recent_calls: List[Dict[str, Any]]
    total_estimated_cost: float


@app.get("/api/bedrock-status", response_model=BedrockStatusResponse)
async def get_bedrock_status():
    """
    Get Bedrock usage status and recent calls.
    
    Returns:
        - Model configuration
        - Daily call count and cost
        - Quota status
        - Last 10 Bedrock calls with details
    """
    try:
        bedrock_client = _get_bedrock_client()
        
        # Get status
        status = bedrock_client.get_status()
        
        # Get recent calls
        recent_calls = bedrock_client.get_recent_calls(limit=10)
        
        # Calculate total cost from recent calls
        total_cost = sum(call.get("cost_estimate", 0.0) for call in recent_calls)
        
        return BedrockStatusResponse(
            model_id=status.get("model_id", "unknown"),
            region=status.get("region", "unknown"),
            daily_calls=status.get("daily_calls", 0),
            daily_cost=status.get("daily_cost", 0.0),
            quota_status=status.get("quota_status", "unknown"),
            within_quota=status.get("within_quota", True),
            recent_calls=recent_calls,
            total_estimated_cost=total_cost
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in bedrock-status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/bedrock-detector-status")
async def get_bedrock_detector_status():
    """
    Get event detector status.
    
    Returns the current state of the event detector including:
    - Market open/close trigger status
    - Volatility baseline
    - Cooldown status
    """
    try:
        event_detector = _get_event_detector()
        return event_detector.get_status()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in bedrock-detector-status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Startup event
@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    configure_logging(level="INFO")
    logger.info("MyTrader Dashboard API started")
    
    # Check if reports directory exists
    Path("reports").mkdir(exist_ok=True)
    Path("dashboard/backend/reports").mkdir(parents=True, exist_ok=True)
    
    # Include RAG API endpoints if available
    if RAG_AVAILABLE:
        try:
            include_rag_router(app)
            logger.info("RAG API endpoints enabled")
        except Exception as e:
            logger.warning(f"Failed to enable RAG API: {e}")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    global trading_session
    
    if trading_session:
        trading_session.stop_requested = True
        trading_session.running = False
    
    logger.info("MyTrader Dashboard API shutdown")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "dashboard_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
