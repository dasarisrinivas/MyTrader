import { useState, useEffect } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import BotOverview from './BotOverview';
import DecisionIntelligence from './DecisionIntelligence';
import LiveTradeTrail from './LiveTradeTrail';
import RealTimeCharts from './RealTimeCharts';
import BotHealthIndicator from './BotHealthIndicator';
import BacktestControls from './BacktestControls';
import BacktestResults from './BacktestResults';
import { Activity, Play, Square, BarChart2 } from 'lucide-react';

const API_URL = 'http://localhost:8000';

export const Dashboard = () => {
  const [status, setStatus] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview'); // 'overview', 'intelligence', 'trail', 'charts', 'backtest'
  const [backtestResults, setBacktestResults] = useState(null);
  const [starting, setStarting] = useState(false);

  const { isConnected, lastMessage } = useWebSocket();

  // Fetch initial status
  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 3000);
    return () => clearInterval(interval);
  }, []);

  // Handle WebSocket updates
  useEffect(() => {
    if (lastMessage) {
      console.log('WebSocket update:', lastMessage);
      if (lastMessage.type === 'status_update') {
        setStatus(lastMessage.data);
        setIsRunning(lastMessage.data.is_running);
      }
    }
  }, [lastMessage]);

  const fetchStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/api/trading/status`);
      const data = await response.json();
      setStatus(data);
      setIsRunning(data.is_running);
    } catch (err) {
      console.error('Failed to fetch status:', err);
    }
  };

  const handleStart = async () => {
    try {
      setStarting(true);
      setError(null);
      const response = await fetch(`${API_URL}/api/trading/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          config_path: 'config.yaml',
          strategy: 'rsi_macd_sentiment'
        }),
      });
      
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Failed to start trading');
      }
      
      await fetchStatus();
      setIsRunning(true);
    } catch (err) {
      setError(err.message);
      console.error('Failed to start trading:', err);
    } finally {
      setStarting(false);
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
      setIsRunning(false);
    } catch (err) {
      setError(err.message);
      console.error('Failed to stop trading:', err);
    }
  };

  const handleBacktestComplete = (results) => {
    setBacktestResults(results);
  };

  return (
    <div className="min-h-screen bg-gray-950 font-sf">
      {/* Header */}
      <header className="bg-gradient-to-r from-gray-900 to-gray-800 border-b border-gray-700 shadow-2xl">
        <div className="max-w-7xl mx-auto px-6 py-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="p-2 bg-blue-600 rounded-lg">
                <Activity className="w-7 h-7 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-white">MyTrader AI</h1>
                <p className="text-gray-400 text-sm">Autonomous Trading Bot Dashboard</p>
              </div>
            </div>
            
            {/* Bot Control */}
            <div className="flex items-center gap-4">
              {!isRunning ? (
                <button
                  onClick={handleStart}
                  disabled={starting}
                  className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-green-600 to-green-500 text-white rounded-lg hover:from-green-500 hover:to-green-400 disabled:from-gray-600 disabled:to-gray-500 disabled:cursor-not-allowed transition-all shadow-lg"
                >
                  <Play className="w-5 h-5" />
                  {starting ? 'Starting...' : 'Start Bot'}
                </button>
              ) : (
                <button
                  onClick={handleStop}
                  className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-red-600 to-red-500 text-white rounded-lg hover:from-red-500 hover:to-red-400 transition-all shadow-lg"
                >
                  <Square className="w-5 h-5" />
                  Stop Bot
                </button>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8 space-y-6">
        {/* Bot Health Indicator */}
        <BotHealthIndicator />

        {/* Error Alert */}
        {error && (
          <div className="bg-red-900/20 border border-red-700 rounded-xl p-4">
            <p className="text-red-400">{error}</p>
          </div>
        )}

        {/* Tab Navigation */}
        <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl shadow-2xl border border-gray-700">
          <div className="border-b border-gray-700">
            <nav className="flex -mb-px">
              <button
                onClick={() => setActiveTab('overview')}
                className={`px-6 py-4 text-sm font-semibold border-b-2 transition-all ${
                  activeTab === 'overview'
                    ? 'border-blue-500 text-blue-400 bg-blue-900/20'
                    : 'border-transparent text-gray-400 hover:text-gray-300 hover:border-gray-600'
                }`}
              >
                <Activity className="w-4 h-4 inline mr-2" />
                Overview
              </button>
              <button
                onClick={() => setActiveTab('intelligence')}
                className={`px-6 py-4 text-sm font-semibold border-b-2 transition-all ${
                  activeTab === 'intelligence'
                    ? 'border-purple-500 text-purple-400 bg-purple-900/20'
                    : 'border-transparent text-gray-400 hover:text-gray-300 hover:border-gray-600'
                }`}
              >
                🧠 AI Intelligence
              </button>
              <button
                onClick={() => setActiveTab('trail')}
                className={`px-6 py-4 text-sm font-semibold border-b-2 transition-all ${
                  activeTab === 'trail'
                    ? 'border-green-500 text-green-400 bg-green-900/20'
                    : 'border-transparent text-gray-400 hover:text-gray-300 hover:border-gray-600'
                }`}
              >
                📜 Trade Trail
              </button>
              <button
                onClick={() => setActiveTab('charts')}
                className={`px-6 py-4 text-sm font-semibold border-b-2 transition-all ${
                  activeTab === 'charts'
                    ? 'border-yellow-500 text-yellow-400 bg-yellow-900/20'
                    : 'border-transparent text-gray-400 hover:text-gray-300 hover:border-gray-600'
                }`}
              >
                📊 Analytics
              </button>
              <button
                onClick={() => setActiveTab('backtest')}
                className={`px-6 py-4 text-sm font-semibold border-b-2 transition-all ${
                  activeTab === 'backtest'
                    ? 'border-orange-500 text-orange-400 bg-orange-900/20'
                    : 'border-transparent text-gray-400 hover:text-gray-300 hover:border-gray-600'
                }`}
              >
                <BarChart2 className="w-4 h-4 inline mr-2" />
                Backtest
              </button>
            </nav>
          </div>
        </div>

        {/* Tab Content */}
        <div className="animate-fadeIn">
          {activeTab === 'overview' && <BotOverview />}
          
          {activeTab === 'intelligence' && <DecisionIntelligence />}
          
          {activeTab === 'trail' && <LiveTradeTrail />}
          
          {activeTab === 'charts' && <RealTimeCharts />}
          
          {activeTab === 'backtest' && (
            <div className="space-y-6">
              <BacktestControls onBacktestComplete={handleBacktestComplete} />
              <BacktestResults results={backtestResults} />
            </div>
          )}
        </div>
      </main>
    </div>
  );
};
