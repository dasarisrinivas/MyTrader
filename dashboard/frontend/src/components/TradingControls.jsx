import { Play, Square, RefreshCw } from 'lucide-react';
import { useState } from 'react';

export const TradingControls = ({ status, onStart, onStop }) => {
  const [isLoading, setIsLoading] = useState(false);

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
            <button
              onClick={handleStop}
              disabled={isLoading}
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
          )}
        </div>
      </div>

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
