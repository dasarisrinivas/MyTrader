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
from datetime import datetime
import json
import subprocess

from mytrader.utils.settings_loader import load_settings
from mytrader.monitoring.live_tracker import LivePerformanceTracker
from mytrader.utils.logger import configure_logging, logger
from mytrader.execution.live_trading_manager import LiveTradingManager

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
    """Start a new live trading session by launching main.py as subprocess."""
    global live_trading_manager
    
    if live_trading_manager and live_trading_manager.running:
        raise HTTPException(status_code=400, detail="Trading session already running")
    
    try:
        import subprocess
        import os
        
        # Load settings - use absolute path from project root
        config_path = project_root / request.config_path
        
        logger.info(f"Attempting to load config from: {config_path}")
        logger.info(f"Config exists: {config_path.exists()}")
        logger.info(f"Project root: {project_root}")
        
        if not config_path.exists():
            raise HTTPException(status_code=404, detail=f"Config file not found: {config_path}")
        
        # Start main.py live trading as a subprocess
        venv_python = project_root / ".venv" / "bin" / "python3"
        main_py = project_root / "main.py"
        
        log_file = project_root / "logs" / "live_trading.log"
        log_file.parent.mkdir(exist_ok=True)
        
        # Start the live trading process
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                [str(venv_python), str(main_py), "live", "--config", str(config_path)],
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=str(project_root),
                env={**os.environ, "PYTHONUNBUFFERED": "1"}
            )
        
        # Create a simple manager to track the process
        from dataclasses import dataclass
        
        @dataclass
        class ProcessManager:
            process: subprocess.Popen
            running: bool = True
            
            def stop(self):
                if self.process:
                    self.process.terminate()
                    self.process.wait(timeout=10)
                self.running = False
        
        live_trading_manager = ProcessManager(process)
        
        logger.info(f"✅ Started live trading process (PID: {process.pid})")
        logger.info(f"📝 Logs: {log_file}")
        
        # Start log tailer in background
        asyncio.create_task(tail_logs())
        
        return {
            "status": "started",
            "message": f"Live trading session started successfully (PID: {process.pid})",
            "log_file": str(log_file),
            "pid": process.pid,
            "timestamp": datetime.now().isoformat()
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
        # Terminate the subprocess gracefully
        if live_trading_manager.process.poll() is None:
            live_trading_manager.process.terminate()
            # Give it 5 seconds to terminate gracefully
            try:
                live_trading_manager.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate
                live_trading_manager.process.kill()
                live_trading_manager.process.wait()
        
        live_trading_manager.running = False
        live_trading_manager = None
        
        return {
            "status": "stopped",
            "message": "Trading session stopped successfully",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to stop trading: {e}")
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
    
    # Check if process is still running
    if live_trading_manager.process.poll() is not None:
        live_trading_manager.running = False
        return {
            "is_running": False,
            "message": "Trading process terminated"
        }
    
    return {
        "is_running": True,
        "pid": live_trading_manager.process.pid,
        "message": "Trading session running"
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


async def tail_logs():
    """Tail the live trading log file and broadcast updates via WebSocket."""
    log_file = project_root / "logs" / "live_trading.log"
    
    if not log_file.exists():
        logger.warning(f"Log file not found: {log_file}")
        return
    
    logger.info(f"Starting log tailer for: {log_file}")
    
    try:
        with open(log_file, 'r') as f:
            # Start from the beginning to catch all bars
            f.seek(0, 0)
            lines_read = 0
            
            while live_trading_manager and live_trading_manager.running:
                line = f.readline()
                if line:
                    lines_read += 1
                    # Parse the log line and send as structured data
                    parsed = parse_log_line(line)
                    if parsed:
                        logger.info(f"Parsed line {lines_read}: {parsed['type']}")
                        # Broadcast to all connected WebSocket clients
                        await manager.broadcast(parsed)
                else:
                    # No new data, wait a bit
                    await asyncio.sleep(0.5)
    except Exception as e:
        logger.error(f"Error tailing logs: {e}", exc_info=True)


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
    
    # Parse order placement
    elif "📤 PLACING ORDER:" in line:
        parts = line.split("PLACING ORDER:")[1].strip()
        return {
            "type": "order",
            "action": parts,
            "status": "pending",
            "timestamp": datetime.now().isoformat()
        }
    
    # Parse order status
    elif "✅ Order filled" in line or "❌ Order" in line:
        status = "filled" if "✅" in line else "rejected"
        return {
            "type": "order_update",
            "status": status,
            "message": line.split(":", 1)[1].strip() if ":" in line else line,
            "timestamp": datetime.now().isoformat()
        }
    
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


# Startup event
@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    configure_logging(level="INFO")
    logger.info("MyTrader Dashboard API started")
    
    # Check if reports directory exists
    Path("reports").mkdir(exist_ok=True)


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
