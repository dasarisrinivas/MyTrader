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

from mytrader.utils.settings_loader import load_settings
from mytrader.monitoring.live_tracker import LivePerformanceTracker
from mytrader.utils.logger import configure_logging, logger

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
    """Start a new trading session."""
    global trading_session
    
    if trading_session and getattr(trading_session, 'running', False):
        raise HTTPException(status_code=400, detail="Trading session already running")
    
    try:
        # Import here to avoid circular imports
        from scripts.paper_trade import PaperTradingSession
        
        # Create new session
        trading_session = PaperTradingSession(request.config_path)
        
        # Run pre-flight checks (async mode to avoid event loop conflicts)
        checks_passed = trading_session.pre_flight_checks(use_async=True)
        
        if not checks_passed:
            raise HTTPException(status_code=400, detail="Pre-flight checks failed")
        
        # Setup components
        trading_session.setup_components()
        
        # Start trading in background
        asyncio.create_task(run_trading_session())
        
        return {
            "status": "started",
            "message": "Trading session started successfully",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to start trading: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trading/stop")
async def stop_trading():
    """Stop the current trading session."""
    global trading_session
    
    if not trading_session:
        raise HTTPException(status_code=400, detail="No trading session running")
    
    try:
        trading_session.stop_requested = True
        trading_session.running = False
        
        # Broadcast stop event
        await manager.broadcast({
            "type": "trading_stopped",
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "status": "stopped",
            "message": "Trading session stopped successfully",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to stop trading: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trades")
async def get_trades(limit: int = 50):
    """Get recent trades."""
    global performance_tracker
    
    if not performance_tracker:
        return {"trades": []}
    
    # Get trades from tracker
    trades = performance_tracker.trades[-limit:] if hasattr(performance_tracker, 'trades') else []
    
    return {"trades": trades, "count": len(trades)}


@app.get("/api/performance")
async def get_performance():
    """Get current performance metrics."""
    global performance_tracker
    
    if not performance_tracker:
        return {
            "total_pnl": 0.0,
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0
        }
    
    snapshot = performance_tracker.get_snapshot()
    
    # Calculate return percentage
    total_return = (snapshot.equity / performance_tracker.initial_capital - 1) * 100
    
    return {
        "total_pnl": snapshot.total_pnl,
        "total_return": total_return,
        "sharpe_ratio": snapshot.sharpe_ratio,
        "max_drawdown": snapshot.max_drawdown,
        "win_rate": snapshot.win_rate,
        "total_trades": snapshot.trade_count,
        "winning_trades": snapshot.winning_trades,
        "losing_trades": snapshot.losing_trades,
    }


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
            # Echo back or handle commands if needed
            await websocket.send_json({
                "type": "echo",
                "data": data,
                "timestamp": datetime.now().isoformat()
            })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


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
