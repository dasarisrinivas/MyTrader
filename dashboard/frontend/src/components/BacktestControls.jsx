import { useState } from 'react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

function BacktestControls({ onBacktestComplete }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [downloadData, setDownloadData] = useState(true);
  const [message, setMessage] = useState('');

  // Set default dates (last Friday to today)
  const getLastFriday = () => {
    const today = new Date();
    const dayOfWeek = today.getDay();
    const daysToLastFriday = (dayOfWeek + 2) % 7;
    const lastFriday = new Date(today);
    lastFriday.setDate(today.getDate() - daysToLastFriday);
    return lastFriday.toISOString().split('T')[0];
  };

  const getToday = () => {
    return new Date().toISOString().split('T')[0];
  };

  // Initialize dates on component mount
  useState(() => {
    if (!startDate) setStartDate(getLastFriday());
    if (!endDate) setEndDate(getToday());
  });

  const handleRunBacktest = async () => {
    setLoading(true);
    setError(null);
    setMessage('');

    try {
      setMessage('Running backtest...');
      
      const response = await axios.post(`${API_BASE_URL}/api/backtest/run`, {
        start_date: startDate,
        end_date: endDate,
        download_data: downloadData,
        config_path: 'config.yaml'
      });

      setMessage('Backtest completed successfully!');
      
      // Notify parent component
      if (onBacktestComplete) {
        onBacktestComplete(response.data);
      }
    } catch (err) {
      console.error('Backtest error:', err);
      setError(err.response?.data?.detail || err.message);
      setMessage('');
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadData = async () => {
    setLoading(true);
    setError(null);
    setMessage('');

    try {
      setMessage('Downloading data...');
      
      await axios.post(`${API_BASE_URL}/api/backtest/download-data`, {
        start_date: startDate,
        end_date: endDate
      });

      setMessage('Data downloaded successfully!');
    } catch (err) {
      console.error('Download error:', err);
      setError(err.response?.data?.detail || err.message);
      setMessage('');
    } finally {
      setLoading(false);
    }
  };

  const handleQuickDate = (days) => {
    const end = new Date();
    const start = new Date();
    start.setDate(end.getDate() - days);
    
    setStartDate(start.toISOString().split('T')[0]);
    setEndDate(end.toISOString().split('T')[0]);
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-2xl font-bold mb-4 text-gray-800">Backtesting</h2>
      
      {/* Date Selection */}
      <div className="space-y-4 mb-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Start Date
            </label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={loading}
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              End Date
            </label>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={loading}
            />
          </div>
        </div>

        {/* Quick Date Buttons */}
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => handleQuickDate(1)}
            className="px-3 py-1 bg-gray-200 text-gray-700 rounded hover:bg-gray-300 text-sm"
            disabled={loading}
          >
            Last 1 Day
          </button>
          <button
            onClick={() => handleQuickDate(7)}
            className="px-3 py-1 bg-gray-200 text-gray-700 rounded hover:bg-gray-300 text-sm"
            disabled={loading}
          >
            Last Week
          </button>
          <button
            onClick={() => handleQuickDate(30)}
            className="px-3 py-1 bg-gray-200 text-gray-700 rounded hover:bg-gray-300 text-sm"
            disabled={loading}
          >
            Last Month
          </button>
          <button
            onClick={() => {
              setStartDate(getLastFriday());
              setEndDate(getToday());
            }}
            className="px-3 py-1 bg-gray-200 text-gray-700 rounded hover:bg-gray-300 text-sm"
            disabled={loading}
          >
            This Friday
          </button>
        </div>

        {/* Download Data Checkbox */}
        <div className="flex items-center">
          <input
            type="checkbox"
            id="downloadData"
            checked={downloadData}
            onChange={(e) => setDownloadData(e.target.checked)}
            className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
            disabled={loading}
          />
          <label htmlFor="downloadData" className="ml-2 block text-sm text-gray-700">
            Download fresh data before backtest
          </label>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex gap-3 mb-4">
        <button
          onClick={handleDownloadData}
          disabled={loading || !startDate || !endDate}
          className="flex-1 bg-gray-600 text-white px-4 py-2 rounded-lg hover:bg-gray-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? 'Processing...' : 'Download Data Only'}
        </button>
        
        <button
          onClick={handleRunBacktest}
          disabled={loading || !startDate || !endDate}
          className="flex-1 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? 'Running...' : 'Run Backtest'}
        </button>
      </div>

      {/* Status Messages */}
      {message && (
        <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg text-blue-700">
          {message}
        </div>
      )}
      
      {error && (
        <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-red-700">
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Info Box */}
      <div className="mt-4 p-3 bg-gray-50 rounded-lg text-sm text-gray-600">
        <p className="font-semibold mb-1">ℹ️ How it works:</p>
        <ul className="list-disc list-inside space-y-1">
          <li>Select date range for historical data</li>
          <li>Download real market data from Yahoo Finance (ES futures)</li>
          <li>Run strategy backtest with your configured settings</li>
          <li>View results: equity curve, trades, and performance metrics</li>
        </ul>
      </div>
    </div>
  );
}

export default BacktestControls;
