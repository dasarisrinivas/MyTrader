import { useState, useEffect } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import BotOverview from './BotOverview';
import DecisionIntelligence from './DecisionIntelligence';
import LiveTradeTrail from './LiveTradeTrail';
import RealTimeCharts from './RealTimeCharts';
import BotHealthIndicator from './BotHealthIndicator';
import BacktestControls from './BacktestControls';
import BacktestResults from './BacktestResults';
import RAGStatusIndicator from './RAGStatusIndicator';
import TradingSignalDisplay from './TradingSignalDisplay';
import TradingSummary from './TradingSummary';
import { ErrorNotification, ConnectionStatus, BackendStatusCard } from './ErrorStates';
import { Activity, Play, Square, BarChart2, AlertTriangle, Brain, PieChart } from 'lucide-react';

const API_URL = 'http://localhost:8000';

export const Dashboard = () => {
  const [status, setStatus] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview'); // 'overview', 'intelligence', 'trail', 'charts', 'backtest', 'signals'
  const [backtestResults, setBacktestResults] = useState(null);
  const [starting, setStarting] = useState(false);
  const [showEmergencyConfirm, setShowEmergencyConfirm] = useState(false);
  const [isEmergencyExiting, setIsEmergencyExiting] = useState(false);
  const [isRetrying, setIsRetrying] = useState(false);

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

  // Auto-retry connection
  useEffect(() => {
    if (!isConnected && !isRetrying) {
      const retryTimer = setTimeout(() => {
        handleRetryConnection();
      }, 5000);
      
      return () => clearTimeout(retryTimer);
    }
  }, [isConnected]);

  const fetchStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/api/trading/status`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      setStatus(data);
      setIsRunning(data.is_running);
      setError(null); // Clear error on successful fetch
    } catch (err) {
      console.error('Failed to fetch status:', err);
      if (!error) { // Only set error if not already set
        setError(`Failed to connect to backend: ${err.message}`);
      }
    }
  };

  const handleRetryConnection = () => {
    setIsRetrying(true);
    fetchStatus().finally(() => {
      setIsRetrying(false);
    });
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

  const handleEmergencyExit = async () => {
    try {
      setIsEmergencyExiting(true);
      setError(null);
      const response = await fetch(`${API_URL}/api/trading/emergency-exit`, {
        method: 'POST',
      });
      
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Failed to execute emergency exit');
      }
      
      const result = await response.json();
      console.log('Emergency exit result:', result);
      
      // Show success message with details
      if (result.positions_closed > 0) {
        alert(`Emergency exit successful!\n\nPositions closed: ${result.positions_closed}\n${result.errors.length > 0 ? '\nWarnings: ' + result.errors.join(', ') : ''}`);
      } else {
        alert('Emergency exit completed. No positions were open.');
      }
      
      await fetchStatus();
      setIsRunning(false);
      setShowEmergencyConfirm(false);
    } catch (err) {
      setError(err.message);
      console.error('Failed to execute emergency exit:', err);
      alert(`Emergency exit failed: ${err.message}`);
    } finally {
      setIsEmergencyExiting(false);
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
              <button
                onClick={handleStart}
                disabled={starting || isRunning}
                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-green-600 to-green-500 text-white rounded-lg hover:from-green-500 hover:to-green-400 disabled:from-gray-600 disabled:to-gray-500 disabled:cursor-not-allowed transition-all shadow-lg"
              >
                <Play className="w-5 h-5" />
                {starting ? 'Starting...' : 'Start Bot'}
              </button>
              
              <button
                onClick={handleStop}
                disabled={!isRunning || isEmergencyExiting}
                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-red-600 to-red-500 text-white rounded-lg hover:from-red-500 hover:to-red-400 disabled:from-gray-600 disabled:to-gray-500 disabled:cursor-not-allowed transition-all shadow-lg"
              >
                <Square className="w-5 h-5" />
                Stop Bot
              </button>
              
              <button
                onClick={() => setShowEmergencyConfirm(true)}
                disabled={!isRunning || isEmergencyExiting}
                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-orange-600 to-orange-500 text-white rounded-lg hover:from-orange-500 hover:to-orange-400 disabled:from-gray-600 disabled:to-gray-500 disabled:cursor-not-allowed transition-all shadow-lg"
              >
                <AlertTriangle className="w-5 h-5" />
                Emergency Exit
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Emergency Exit Confirmation Modal */}
      {showEmergencyConfirm && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-gray-800 border border-gray-700 rounded-2xl p-8 max-w-md mx-4 shadow-2xl">
            <div className="flex items-center space-x-3 mb-4">
              <AlertTriangle className="w-8 h-8 text-orange-500" />
              <h3 className="text-2xl font-bold text-white">Emergency Exit</h3>
            </div>
            
            <p className="text-gray-300 mb-6">
              This will immediately:
            </p>
            <ul className="list-disc list-inside text-gray-300 mb-6 space-y-2">
              <li>Close all open positions (market order)</li>
              <li>Cancel all pending orders</li>
              <li>Stop the trading bot</li>
            </ul>
            
            <p className="text-orange-400 font-semibold mb-6 bg-orange-900/20 p-3 rounded-lg border border-orange-700">
              ⚠️ This action cannot be undone!
            </p>
            
            <div className="flex space-x-3">
              <button
                onClick={() => setShowEmergencyConfirm(false)}
                disabled={isEmergencyExiting}
                className="flex-1 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:cursor-not-allowed text-white px-6 py-3 rounded-lg font-medium transition-colors duration-200"
              >
                Cancel
              </button>
              <button
                onClick={handleEmergencyExit}
                disabled={isEmergencyExiting}
                className="flex-1 bg-gradient-to-r from-orange-600 to-orange-500 hover:from-orange-500 hover:to-orange-400 disabled:from-gray-600 disabled:to-gray-500 disabled:cursor-not-allowed text-white px-6 py-3 rounded-lg font-medium transition-all duration-200 flex items-center justify-center space-x-2"
              >
                {isEmergencyExiting ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    <span>Exiting...</span>
                  </>
                ) : (
                  <span>Confirm Emergency Exit</span>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

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
              <button
                onClick={() => setActiveTab('summary')}
                className={`px-6 py-4 text-sm font-semibold border-b-2 transition-all ${
                  activeTab === 'summary'
                    ? 'border-cyan-500 text-cyan-400 bg-cyan-900/20'
                    : 'border-transparent text-gray-400 hover:text-gray-300 hover:border-gray-600'
                }`}
              >
                <PieChart className="w-4 h-4 inline mr-2" />
                Today's Summary
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

          {activeTab === 'summary' && <TradingSummary />}
        </div>
      </main>
    </div>
  );
};
