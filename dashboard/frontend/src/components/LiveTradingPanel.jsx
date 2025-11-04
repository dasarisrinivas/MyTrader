import { useState, useEffect } from 'react';
import { Play, Square, Activity, TrendingUp, AlertCircle, CheckCircle, Clock } from 'lucide-react';
import MarketStatus from './MarketStatus';

const API_URL = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/ws';

export default function LiveTradingPanel() {
  const [isRunning, setIsRunning] = useState(false);
  const [status, setStatus] = useState(null);
  const [signals, setSignals] = useState([]);
  const [orders, setOrders] = useState([]);
  const [ws, setWs] = useState(null);
  const [connecting, setConnecting] = useState(false);
  // Separate state for data collection progress
  const [dataCollectionProgress, setDataCollectionProgress] = useState({
    isCollecting: false,
    barsCollected: 0,
    minBarsNeeded: 50,
    percentage: 0
  });

  // Connect to WebSocket
  useEffect(() => {
    const websocket = new WebSocket(WS_URL);
    
    websocket.onopen = () => {
      console.log('WebSocket connected');
      setWs(websocket);
    };
    
    websocket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      handleWebSocketMessage(message);
    };
    
    websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
    
    websocket.onclose = () => {
      console.log('WebSocket disconnected');
      setWs(null);
      // Attempt to reconnect after 5 seconds
      setTimeout(() => {
        if (!ws || ws.readyState === WebSocket.CLOSED) {
          window.location.reload();
        }
      }, 5000);
    };
    
    return () => {
      if (websocket.readyState === WebSocket.OPEN) {
        websocket.close();
      }
    };
  }, []);

  // Fetch trading status periodically
  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 2000); // Every 2 seconds
    return () => clearInterval(interval);
  }, []);

  const handleWebSocketMessage = (message) => {
    console.log('WebSocket message:', message);
    
    switch (message.type) {
      case 'connected':
        console.log('WebSocket connected:', message.message);
        break;
      
      case 'status_update':
        setStatus(message.data);
        setIsRunning(message.data.is_running);
        break;
      
      case 'signal':
        // Add new signal to the list
        const newSignal = {
          action: message.signal,
          confidence: message.confidence,
          timestamp: message.timestamp
        };
        setSignals(prev => [newSignal, ...prev].slice(0, 20));
        
        // Update status with latest signal
        setStatus(prev => ({
          ...prev,
          last_signal: message.signal,
          signal_confidence: message.confidence
        }));
        break;
      
      case 'order':
        // Add new order
        const newOrder = {
          action: message.action,
          status: message.status,
          timestamp: message.timestamp
        };
        setOrders(prev => [newOrder, ...prev].slice(0, 50));
        break;
      
      case 'order_update':
        // Update order status
        setOrders(prev => {
          const updated = [...prev];
          if (updated.length > 0) {
            updated[0] = {
              ...updated[0],
              status: message.status,
              message: message.message
            };
          }
          return updated;
        });
        break;
      
      case 'progress':
        // Update bars collected - persist in separate state
        const barsCollected = message.bars_collected || 0;
        const minBarsNeeded = message.min_bars_needed || 50;
        const percentage = Math.min(100, (barsCollected / minBarsNeeded) * 100);
        
        console.log(`Progress update: ${barsCollected}/${minBarsNeeded} (${percentage.toFixed(1)}%)`);
        
        setDataCollectionProgress({
          isCollecting: barsCollected < minBarsNeeded,
          barsCollected,
          minBarsNeeded,
          percentage
        });
        
        setStatus(prev => ({
          ...prev,
          bars_collected: barsCollected,
          min_bars_needed: minBarsNeeded
        }));
        break;
      
      case 'price_update':
        // Update current price
        setStatus(prev => ({
          ...prev,
          current_price: message.price
        }));
        break;
      
      case 'error':
        console.error('Trading error:', message.data);
        break;
      
      default:
        console.log('Unknown message type:', message.type);
    }
  };

  const fetchStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/api/trading/status`);
      const data = await response.json();
      setStatus(data);
      setIsRunning(data.is_running);
    } catch (error) {
      console.error('Failed to fetch status:', error);
    }
  };

  const startTrading = async () => {
    try {
      setConnecting(true);
      const response = await fetch(`${API_URL}/api/trading/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          config_path: 'config.yaml',
          strategy: 'rsi_macd_sentiment'
        })
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to start trading');
      }
      
      const data = await response.json();
      console.log('Trading started:', data);
      setIsRunning(true);
      
      // Initialize status with defaults
      setStatus({
        is_running: true,
        bars_collected: 0,
        min_bars_needed: 50,
        current_price: null,
        last_signal: null,
        signal_confidence: null,
        current_position: 0,
        unrealized_pnl: 0,
        message: 'Initializing trading session...'
      });
      
      // Initialize data collection progress
      setDataCollectionProgress({
        isCollecting: true,
        barsCollected: 0,
        minBarsNeeded: 50,
        percentage: 0
      });
      
      // Clear previous data
      setSignals([]);
      setOrders([]);
    } catch (error) {
      console.error('Failed to start trading:', error);
      alert(`Failed to start trading: ${error.message}`);
    } finally {
      setConnecting(false);
    }
  };

  const stopTrading = async () => {
    try {
      const response = await fetch(`${API_URL}/api/trading/stop`, {
        method: 'POST'
      });
      
      if (!response.ok) {
        throw new Error('Failed to stop trading');
      }
      
      setIsRunning(false);
    } catch (error) {
      console.error('Failed to stop trading:', error);
      alert(`Failed to stop trading: ${error.message}`);
    }
  };

  const getSignalColor = (action) => {
    switch (action) {
      case 'BUY': return 'text-green-600';
      case 'SELL': return 'text-red-600';
      case 'HOLD': return 'text-gray-600';
      default: return 'text-gray-600';
    }
  };

  const getSignalBg = (action) => {
    switch (action) {
      case 'BUY': return 'bg-green-100 border-green-200';
      case 'SELL': return 'bg-red-100 border-red-200';
      case 'HOLD': return 'bg-gray-100 border-gray-200';
      default: return 'bg-gray-100 border-gray-200';
    }
  };

  const getOrderStatusColor = (status) => {
    switch (status) {
      case 'Filled': return 'text-green-600';
      case 'placing': return 'text-blue-600';
      case 'Submitted':
      case 'PreSubmitted': return 'text-yellow-600';
      case 'Cancelled':
      case 'Inactive': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const getOrderStatusIcon = (status) => {
    switch (status) {
      case 'Filled': return <CheckCircle className="w-4 h-4" />;
      case 'placing': return <Clock className="w-4 h-4 animate-spin" />;
      case 'Cancelled':
      case 'Inactive': return <AlertCircle className="w-4 h-4" />;
      default: return <Activity className="w-4 h-4" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Market Status - New Component */}
      {isRunning && (
        <MarketStatus />
      )}

      {/* Control Panel */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
            <Activity className="w-6 h-6" />
            Live Trading
          </h2>
          
          <div className="flex gap-3">
            {!isRunning ? (
              <button
                onClick={startTrading}
                disabled={connecting}
                className="flex items-center gap-2 px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
              >
                <Play className="w-5 h-5" />
                {connecting ? 'Starting...' : 'Start Trading'}
              </button>
            ) : (
              <button
                onClick={stopTrading}
                className="flex items-center gap-2 px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
              >
                <Square className="w-5 h-5" />
                Stop Trading
              </button>
            )}
          </div>
        </div>

        {/* Status Display */}
        {status && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600 mb-1">Status</div>
              <div className={`text-lg font-semibold ${isRunning ? 'text-green-600' : 'text-gray-600'}`}>
                {isRunning ? '● Running' : '○ Stopped'}
              </div>
            </div>

            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600 mb-1">Current Price</div>
              <div className="text-lg font-semibold text-gray-800">
                ${status.current_price?.toFixed(2) || '--'}
              </div>
            </div>

            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600 mb-1">Position</div>
              <div className="text-lg font-semibold text-gray-800">
                {status.current_position || 0} contracts
              </div>
            </div>

            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600 mb-1">Unrealized P&L</div>
              <div className={`text-lg font-semibold ${status.unrealized_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                ${status.unrealized_pnl?.toFixed(2) || '0.00'}
              </div>
            </div>
          </div>
        )}

        {/* Progress Bar (when collecting data) */}
        {isRunning && dataCollectionProgress.isCollecting && (
          <div className="mt-6 bg-gradient-to-r from-blue-50 to-indigo-50 border-2 border-blue-200 rounded-lg p-5">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-3">
                <div className="animate-pulse">
                  <Activity className="w-5 h-5 text-blue-600" />
                </div>
                <span className="font-semibold text-gray-800 text-lg">
                  Collecting Market Data
                </span>
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-blue-600">
                  {dataCollectionProgress.barsCollected} / {dataCollectionProgress.minBarsNeeded}
                </div>
                <div className="text-sm text-gray-600">
                  bars collected
                </div>
              </div>
            </div>
            
            <div className="relative">
              <div className="w-full bg-gray-200 rounded-full h-4 shadow-inner">
                <div 
                  className="bg-gradient-to-r from-blue-500 to-indigo-600 h-4 rounded-full transition-all duration-500 ease-out flex items-center justify-end pr-2"
                  style={{ width: `${dataCollectionProgress.percentage}%` }}
                >
                  {dataCollectionProgress.percentage > 10 && (
                    <span className="text-xs font-bold text-white">
                      {dataCollectionProgress.percentage.toFixed(0)}%
                    </span>
                  )}
                </div>
              </div>
              {dataCollectionProgress.percentage <= 10 && (
                <div className="absolute -right-12 top-0 text-sm font-semibold text-blue-600">
                  {dataCollectionProgress.percentage.toFixed(0)}%
                </div>
              )}
            </div>
            
            <div className="mt-3 text-sm text-gray-600 flex items-center gap-2">
              <Clock className="w-4 h-4" />
              <span>
                {dataCollectionProgress.minBarsNeeded - dataCollectionProgress.barsCollected} bars remaining
                {' • '}
                ~{Math.ceil((dataCollectionProgress.minBarsNeeded - dataCollectionProgress.barsCollected) * 5 / 60)} minutes
              </span>
            </div>
          </div>
        )}
        
        {/* Data Collection Complete Message */}
        {isRunning && !dataCollectionProgress.isCollecting && dataCollectionProgress.barsCollected > 0 && (
          <div className="mt-4 p-4 bg-green-50 border-2 border-green-200 rounded-lg flex items-center gap-3">
            <CheckCircle className="w-6 h-6 text-green-600" />
            <div>
              <div className="font-semibold text-green-800">Data Collection Complete</div>
              <div className="text-sm text-green-600">
                Ready to generate trading signals
              </div>
            </div>
          </div>
        )}

        {/* Status Message */}
        {status?.message && (
          <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg text-sm text-blue-800">
            {status.message}
          </div>
        )}

        {/* Performance Metrics */}
        {status?.performance && (
          <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-4 rounded-lg border border-blue-200">
              <div className="text-sm text-blue-600 mb-1">Total Return</div>
              <div className={`text-xl font-bold ${status.performance.total_return >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {status.performance.total_return?.toFixed(2)}%
              </div>
            </div>

            <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-4 rounded-lg border border-purple-200">
              <div className="text-sm text-purple-600 mb-1">Total Trades</div>
              <div className="text-xl font-bold text-gray-800">
                {status.performance.total_trades || 0}
              </div>
            </div>

            <div className="bg-gradient-to-br from-green-50 to-green-100 p-4 rounded-lg border border-green-200">
              <div className="text-sm text-green-600 mb-1">Win Rate</div>
              <div className="text-xl font-bold text-gray-800">
                {status.performance.win_rate?.toFixed(1)}%
              </div>
            </div>

            <div className="bg-gradient-to-br from-orange-50 to-orange-100 p-4 rounded-lg border border-orange-200">
              <div className="text-sm text-orange-600 mb-1">Sharpe Ratio</div>
              <div className="text-xl font-bold text-gray-800">
                {status.performance.sharpe_ratio?.toFixed(2) || '0.00'}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Signals and Orders Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Live Signals */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5" />
            Live Signals
          </h3>
          
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {signals.length === 0 ? (
              <div className="text-center text-gray-500 py-8">
                No signals yet
              </div>
            ) : (
              signals.map((signal, index) => (
                <div 
                  key={index}
                  className={`border rounded-lg p-3 ${getSignalBg(signal.action)}`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <span className={`text-lg font-bold ${getSignalColor(signal.action)}`}>
                        {signal.action}
                      </span>
                      <span className="text-sm text-gray-600">
                        @ ${signal.price?.toFixed(2)}
                      </span>
                    </div>
                    <div className="text-right">
                      <div className="text-sm font-semibold text-gray-700">
                        {(signal.confidence * 100).toFixed(0)}% confidence
                      </div>
                      <div className="text-xs text-gray-500">
                        {new Date(signal.timestamp).toLocaleTimeString()}
                      </div>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Order Status */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
            <Activity className="w-5 h-5" />
            Order Status
          </h3>
          
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {orders.length === 0 ? (
              <div className="text-center text-gray-500 py-8">
                No orders yet
              </div>
            ) : (
              orders.map((order, index) => (
                <div 
                  key={index}
                  className="border border-gray-200 rounded-lg p-3 bg-gray-50"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        {getOrderStatusIcon(order.status)}
                        <span className={`font-semibold ${getOrderStatusColor(order.status)}`}>
                          {order.status}
                        </span>
                        {order.order_id && (
                          <span className="text-xs text-gray-500">
                            #{order.order_id}
                          </span>
                        )}
                      </div>
                      <div className="text-sm text-gray-700">
                        {order.action} {order.quantity} @ {order.entry_price?.toFixed(2) || order.fill_price?.toFixed(2) || '--'}
                      </div>
                      {order.filled_quantity > 0 && (
                        <div className="text-xs text-green-600 mt-1">
                          Filled: {order.filled_quantity} contracts
                        </div>
                      )}
                    </div>
                    <div className="text-right text-xs text-gray-500">
                      {new Date(order.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                  
                  {(order.stop_loss || order.take_profit) && (
                    <div className="mt-2 pt-2 border-t border-gray-200 text-xs text-gray-600 flex gap-4">
                      {order.stop_loss && (
                        <span>SL: ${order.stop_loss.toFixed(2)}</span>
                      )}
                      {order.take_profit && (
                        <span>TP: ${order.take_profit.toFixed(2)}</span>
                      )}
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
