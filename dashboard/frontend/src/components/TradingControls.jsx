import { Play, Square, RefreshCw, AlertTriangle } from 'lucide-react';
import { useState } from 'react';

export const TradingControls = ({ status, onStart, onStop, onEmergencyExit }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [isEmergencyExiting, setIsEmergencyExiting] = useState(false);
  const [showEmergencyConfirm, setShowEmergencyConfirm] = useState(false);

  const handleStart = async () => {
    setIsLoading(true);
    try {
      await onStart();
    } finally {
      setIsLoading(false);
    }
  };

  const handleStop = async () => {
    setIsLoading(true);
    try {
      await onStop();
    } finally {
      setIsLoading(false);
    }
  };

  const handleEmergencyExit = async () => {
    setIsEmergencyExiting(true);
    try {
      await onEmergencyExit();
      setShowEmergencyConfirm(false);
    } finally {
      setIsEmergencyExiting(false);
    }
  };

  const isRunning = status?.is_running || false;
  const uptime = status?.uptime || 0;

  const formatUptime = (seconds) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hours}h ${minutes}m ${secs}s`;
  };

  return (
    <div className="bg-white rounded-2xl shadow-apple p-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className={`flex items-center space-x-2 px-4 py-2 rounded-full ${
            isRunning ? 'bg-green-100' : 'bg-gray-100'
          }`}>
            <div className={`w-3 h-3 rounded-full ${
              isRunning ? 'bg-green-500 animate-pulse' : 'bg-gray-400'
            }`} />
            <span className={`text-sm font-medium ${
              isRunning ? 'text-green-700' : 'text-gray-600'
            }`}>
              {isRunning ? 'Active' : 'Stopped'}
            </span>
          </div>
          
          {isRunning && uptime > 0 && (
            <div className="text-sm text-apple-gray-600">
              Uptime: <span className="font-medium">{formatUptime(uptime)}</span>
            </div>
          )}
        </div>

        <div className="flex items-center space-x-3">
          {!isRunning ? (
            <button
              onClick={handleStart}
              disabled={isLoading}
              className="flex items-center space-x-2 bg-green-500 hover:bg-green-600 disabled:bg-gray-400 text-white px-6 py-3 rounded-xl font-medium transition-colors duration-200 shadow-apple hover:shadow-apple-lg"
            >
              {isLoading ? (
                <>
                  <RefreshCw className="w-5 h-5 animate-spin" />
                  <span>Starting...</span>
                </>
              ) : (
                <>
                  <Play className="w-5 h-5" />
                  <span>Start Trading</span>
                </>
              )}
            </button>
          ) : (
            <>
              <button
                onClick={handleStop}
                disabled={isLoading || isEmergencyExiting}
                className="flex items-center space-x-2 bg-red-500 hover:bg-red-600 disabled:bg-gray-400 text-white px-6 py-3 rounded-xl font-medium transition-colors duration-200 shadow-apple hover:shadow-apple-lg"
              >
                {isLoading ? (
                  <>
                    <RefreshCw className="w-5 h-5 animate-spin" />
                    <span>Stopping...</span>
                  </>
                ) : (
                  <>
                    <Square className="w-5 h-5" />
                    <span>Stop Trading</span>
                  </>
                )}
              </button>
              
              <button
                onClick={() => setShowEmergencyConfirm(true)}
                disabled={isLoading || isEmergencyExiting}
                className="flex items-center space-x-2 bg-orange-600 hover:bg-orange-700 disabled:bg-gray-400 text-white px-6 py-3 rounded-xl font-medium transition-colors duration-200 shadow-apple hover:shadow-apple-lg"
              >
                <AlertTriangle className="w-5 h-5" />
                <span>Emergency Exit</span>
              </button>
            </>
          )}
        </div>
      </div>

      {/* Emergency Exit Confirmation Modal */}
      {showEmergencyConfirm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-2xl p-8 max-w-md mx-4 shadow-2xl">
            <div className="flex items-center space-x-3 mb-4">
              <AlertTriangle className="w-8 h-8 text-orange-600" />
              <h3 className="text-2xl font-bold text-gray-900">Emergency Exit</h3>
            </div>
            
            <p className="text-gray-700 mb-6">
              This will immediately:
            </p>
            <ul className="list-disc list-inside text-gray-700 mb-6 space-y-2">
              <li>Close all open positions (market order)</li>
              <li>Cancel all pending orders</li>
              <li>Stop the trading bot</li>
            </ul>
            
            <p className="text-red-600 font-semibold mb-6">
              ⚠️ This action cannot be undone!
            </p>
            
            <div className="flex space-x-3">
              <button
                onClick={() => setShowEmergencyConfirm(false)}
                disabled={isEmergencyExiting}
                className="flex-1 bg-gray-200 hover:bg-gray-300 disabled:bg-gray-100 text-gray-800 px-6 py-3 rounded-xl font-medium transition-colors duration-200"
              >
                Cancel
              </button>
              <button
                onClick={handleEmergencyExit}
                disabled={isEmergencyExiting}
                className="flex-1 bg-orange-600 hover:bg-orange-700 disabled:bg-gray-400 text-white px-6 py-3 rounded-xl font-medium transition-colors duration-200 flex items-center justify-center space-x-2"
              >
                {isEmergencyExiting ? (
                  <>
                    <RefreshCw className="w-5 h-5 animate-spin" />
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

      {status?.current_position && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <div className="flex items-center justify-between text-sm">
            <span className="text-apple-gray-600">Current Position:</span>
            <span className={`font-medium ${
              status.current_position > 0 ? 'text-green-600' : 'text-red-600'
            }`}>
              {status.current_position > 0 ? 'Long' : 'Short'} {Math.abs(status.current_position)} contracts
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
