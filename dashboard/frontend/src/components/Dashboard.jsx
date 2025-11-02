import { useState, useEffect } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import { TradingControls } from './TradingControls';
import { PerformanceMetrics } from './PerformanceMetrics';
import { EquityChart } from './EquityChart';
import { TradeHistory } from './TradeHistory';
import BacktestControls from './BacktestControls';
import BacktestResults from './BacktestResults';
import { Activity, WifiOff, Wifi } from 'lucide-react';

const API_URL = 'http://localhost:8000';

export const Dashboard = () => {
  const [status, setStatus] = useState(null);
  const [performance, setPerformance] = useState(null);
  const [trades, setTrades] = useState([]);
  const [equityCurve, setEquityCurve] = useState([]);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('live'); // 'live' or 'backtest'
  const [backtestResults, setBacktestResults] = useState(null);

  const { isConnected, lastMessage } = useWebSocket();

  // Fetch initial data
  useEffect(() => {
    fetchStatus();
    fetchPerformance();
    fetchTrades();
    fetchEquityCurve();

    // Poll for updates every 5 seconds
    const interval = setInterval(() => {
      fetchStatus();
      fetchPerformance();
      fetchTrades();
      fetchEquityCurve();
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  // Handle WebSocket updates
  useEffect(() => {
    if (lastMessage) {
      console.log('WebSocket update:', lastMessage);
      
      if (lastMessage.type === 'status_update') {
        setStatus(lastMessage.data);
      } else if (lastMessage.type === 'performance_update') {
        setPerformance(lastMessage.data);
      } else if (lastMessage.type === 'trade') {
        setTrades(prev => [lastMessage.data, ...prev].slice(0, 20));
      }
    }
  }, [lastMessage]);

  const fetchStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/api/status`);
      const data = await response.json();
      setStatus(data);
    } catch (err) {
      console.error('Failed to fetch status:', err);
    }
  };

  const fetchPerformance = async () => {
    try {
      const response = await fetch(`${API_URL}/api/performance`);
      const data = await response.json();
      setPerformance(data);
    } catch (err) {
      console.error('Failed to fetch performance:', err);
    }
  };

  const fetchTrades = async () => {
    try {
      const response = await fetch(`${API_URL}/api/trades?limit=20`);
      const data = await response.json();
      // API returns {trades: [...]}
      setTrades(data.trades || []);
    } catch (err) {
      console.error('Failed to fetch trades:', err);
    }
  };

  const fetchEquityCurve = async () => {
    try {
      const response = await fetch(`${API_URL}/api/equity-curve`);
      const data = await response.json();
      // API returns {data: [...]}
      setEquityCurve(data.data || []);
    } catch (err) {
      console.error('Failed to fetch equity curve:', err);
    }
  };

  const handleStart = async () => {
    try {
      setError(null);
      const response = await fetch(`${API_URL}/api/trading/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Failed to start trading');
      }
      
      await fetchStatus();
    } catch (err) {
      setError(err.message);
      console.error('Failed to start trading:', err);
    }
  };

  const handleStop = async () => {
    try {
      setError(null);
      const response = await fetch(`${API_URL}/api/trading/stop`, {
        method: 'POST',
      });
      
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Failed to stop trading');
      }
      
      await fetchStatus();
    } catch (err) {
      setError(err.message);
      console.error('Failed to stop trading:', err);
    }
  };

  const handleBacktestComplete = (results) => {
    setBacktestResults(results);
  };

  return (
    <div className="min-h-screen bg-apple-gray-50 font-sf">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Activity className="w-8 h-8 text-blue-500" />
              <h1 className="text-2xl font-semibold text-apple-gray-800">MyTrader Dashboard</h1>
            </div>
            <div className="flex items-center space-x-2">
              {isConnected ? (
                <>
                  <Wifi className="w-5 h-5 text-green-500" />
                  <span className="text-sm text-green-600 font-medium">Live</span>
                </>
              ) : (
                <>
                  <WifiOff className="w-5 h-5 text-red-500" />
                  <span className="text-sm text-red-600 font-medium">Disconnected</span>
                </>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8 space-y-6">
        {/* Error Alert */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-xl p-4">
            <p className="text-red-700">{error}</p>
          </div>
        )}

        {/* Tab Navigation */}
        <div className="bg-white rounded-lg shadow">
          <div className="border-b border-gray-200">
            <nav className="flex -mb-px">
              <button
                onClick={() => setActiveTab('live')}
                className={`px-6 py-3 text-sm font-medium border-b-2 transition-colors ${
                  activeTab === 'live'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                Live Trading
              </button>
              <button
                onClick={() => setActiveTab('backtest')}
                className={`px-6 py-3 text-sm font-medium border-b-2 transition-colors ${
                  activeTab === 'backtest'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                Backtesting
              </button>
            </nav>
          </div>
        </div>

        {/* Live Trading Tab */}
        {activeTab === 'live' && (
          <>
            <TradingControls 
              status={status} 
              onStart={handleStart} 
              onStop={handleStop} 
            />
            <PerformanceMetrics performance={performance} />
            <EquityChart data={equityCurve} />
            <TradeHistory trades={trades} />
          </>
        )}

        {/* Backtesting Tab */}
        {activeTab === 'backtest' && (
          <>
            <BacktestControls onBacktestComplete={handleBacktestComplete} />
            <BacktestResults results={backtestResults} />
          </>
        )}
      </main>
    </div>
  );
};
