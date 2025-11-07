import React, { useState, useEffect } from 'react';
import {
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  Clock,
  Activity,
  Target,
  BarChart3,
  CheckCircle,
  XCircle,
  Info
} from 'lucide-react';

const SPYFuturesInsights = ({ apiUrl = 'http://localhost:8000' }) => {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);

  // Fetch latest summary on mount and set up polling
  useEffect(() => {
    fetchLatestSummary();
    
    // Poll every 30 seconds for updates
    const interval = setInterval(fetchLatestSummary, 30000);
    
    return () => clearInterval(interval);
  }, []);

  const fetchLatestSummary = async () => {
    try {
      const response = await fetch(`${apiUrl}/api/spy-futures/latest-summary`);
      const data = await response.json();
      
      if (data.status === 'success') {
        setSummary(data.data);
        setLastUpdate(new Date(data.timestamp));
        setError(null);
      } else if (data.status === 'no_data') {
        setError('No SPY Futures data available yet. Run analysis first.');
      }
      
      setLoading(false);
    } catch (err) {
      setError(`Failed to load data: ${err.message}`);
      setLoading(false);
    }
  };

  const runAnalysis = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${apiUrl}/api/spy-futures/run-analysis`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ days: 1 })
      });
      
      const data = await response.json();
      
      if (data.status === 'success') {
        // Refresh data
        await fetchLatestSummary();
      } else {
        setError('Analysis failed. Check backend logs.');
      }
    } catch (err) {
      setError(`Failed to run analysis: ${err.message}`);
    }
    setLoading(false);
  };

  if (loading && !summary) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <span className="ml-3 text-gray-600">Loading SPY Futures insights...</span>
        </div>
      </div>
    );
  }

  if (error && !summary) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center text-amber-600">
            <AlertTriangle className="w-5 h-5 mr-2" />
            <span>{error}</span>
          </div>
          <button
            onClick={runAnalysis}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Run Analysis
          </button>
        </div>
      </div>
    );
  }

  if (!summary) return null;

  const { performance, observations, insights, suggestions, warnings, date } = summary;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-blue-800 rounded-lg shadow-md p-6 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold mb-2">SPY Futures Daily Insights</h2>
            <p className="text-blue-100">AI-Powered Performance Analysis</p>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold">{summary.symbol || 'ES'}</div>
            <div className="text-sm text-blue-100">{date}</div>
            {lastUpdate && (
              <div className="text-xs text-blue-200 mt-1">
                Updated: {lastUpdate.toLocaleTimeString()}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Total Trades"
          value={performance.total_trades}
          icon={<Activity className="w-5 h-5" />}
          color="blue"
        />
        <MetricCard
          title="Win Rate"
          value={`${performance.win_rate}%`}
          icon={performance.win_rate >= 50 ? <TrendingUp className="w-5 h-5" /> : <TrendingDown className="w-5 h-5" />}
          color={performance.win_rate >= 50 ? 'green' : 'red'}
        />
        <MetricCard
          title="P&L"
          value={`$${performance.profit_loss.toLocaleString()}`}
          icon={performance.profit_loss >= 0 ? <TrendingUp className="w-5 h-5" /> : <TrendingDown className="w-5 h-5" />}
          color={performance.profit_loss >= 0 ? 'green' : 'red'}
        />
        <MetricCard
          title="Max Drawdown"
          value={`$${Math.abs(performance.max_drawdown).toLocaleString()}`}
          icon={<BarChart3 className="w-5 h-5" />}
          color="orange"
        />
      </div>

      {/* Additional Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <MetricCard
          title="Profit Factor"
          value={performance.profit_factor?.toFixed(2) || 'N/A'}
          icon={<Target className="w-5 h-5" />}
          color="purple"
          small
        />
        <MetricCard
          title="Avg Win"
          value={`$${performance.average_win?.toFixed(2) || 0}`}
          icon={<CheckCircle className="w-5 h-5" />}
          color="green"
          small
        />
        <MetricCard
          title="Avg Hold Time"
          value={`${performance.holding_time_avg?.toFixed(1) || 0} min`}
          icon={<Clock className="w-5 h-5" />}
          color="blue"
          small
        />
      </div>

      {/* Warnings */}
      {warnings && warnings.length > 0 && (
        <div className="bg-red-50 border-l-4 border-red-500 rounded-lg p-4">
          <div className="flex items-start">
            <AlertTriangle className="w-5 h-5 text-red-500 mt-0.5 mr-3 flex-shrink-0" />
            <div className="flex-1">
              <h3 className="text-red-800 font-semibold mb-2">⚠️ Critical Warnings</h3>
              <ul className="space-y-1">
                {warnings.map((warning, idx) => (
                  <li key={idx} className="text-red-700 text-sm">{warning}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Key Observations */}
      {observations && observations.length > 0 && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <Info className="w-5 h-5 mr-2 text-blue-600" />
            Key Observations
          </h3>
          <ul className="space-y-2">
            {observations.map((obs, idx) => (
              <li key={idx} className="flex items-start">
                <span className="text-blue-600 mr-2">•</span>
                <span className="text-gray-700">{obs}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* AI Suggestions */}
      {suggestions && Object.keys(suggestions).length > 0 && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <Target className="w-5 h-5 mr-2 text-green-600" />
            AI Recommendations
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(suggestions).map(([key, value]) => (
              <SuggestionCard key={key} parameter={key} value={value} />
            ))}
          </div>
        </div>
      )}

      {/* Detailed Insights */}
      {insights && insights.length > 0 && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Detailed Analysis</h3>
          <div className="space-y-3">
            {insights.map((insight, idx) => (
              <InsightCard key={idx} insight={insight} />
            ))}
          </div>
        </div>
      )}

      {/* Refresh Button */}
      <div className="flex justify-center">
        <button
          onClick={runAnalysis}
          disabled={loading}
          className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center"
        >
          {loading ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
              Running Analysis...
            </>
          ) : (
            <>
              <Activity className="w-4 h-4 mr-2" />
              Run Fresh Analysis
            </>
          )}
        </button>
      </div>
    </div>
  );
};

// Metric Card Component
const MetricCard = ({ title, value, icon, color, small = false }) => {
  const colorClasses = {
    blue: 'bg-blue-50 text-blue-600 border-blue-200',
    green: 'bg-green-50 text-green-600 border-green-200',
    red: 'bg-red-50 text-red-600 border-red-200',
    orange: 'bg-orange-50 text-orange-600 border-orange-200',
    purple: 'bg-purple-50 text-purple-600 border-purple-200'
  };

  return (
    <div className={`bg-white rounded-lg shadow-md border-l-4 ${colorClasses[color]} p-${small ? '3' : '4'}`}>
      <div className="flex items-center justify-between">
        <div>
          <p className={`text-gray-600 ${small ? 'text-xs' : 'text-sm'} mb-1`}>{title}</p>
          <p className={`font-bold ${small ? 'text-lg' : 'text-2xl'}`}>{value}</p>
        </div>
        <div className={colorClasses[color]}>
          {icon}
        </div>
      </div>
    </div>
  );
};

// Suggestion Card Component
const SuggestionCard = ({ parameter, value }) => {
  const formatParameter = (param) => {
    return param
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  return (
    <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-lg p-4 border border-green-200">
      <div className="font-semibold text-gray-800 mb-1">{formatParameter(parameter)}</div>
      <div className="text-gray-700">{JSON.stringify(value)}</div>
    </div>
  );
};

// Insight Card Component
const InsightCard = ({ insight }) => {
  const severityColors = {
    info: 'bg-blue-50 border-blue-300 text-blue-800',
    warning: 'bg-yellow-50 border-yellow-300 text-yellow-800',
    critical: 'bg-red-50 border-red-300 text-red-800'
  };

  const severityClass = severityColors[insight.severity] || severityColors.info;

  return (
    <div className={`${severityClass} border-l-4 rounded-lg p-4`}>
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="font-semibold mb-1">
            {insight.type.toUpperCase()}: {insight.category}
          </div>
          <p className="text-sm mb-2">{insight.description}</p>
          <p className="text-xs italic">{insight.reasoning}</p>
        </div>
        <div className="ml-4">
          <div className="text-xs font-semibold">
            {Math.round(insight.confidence * 100)}% confidence
          </div>
        </div>
      </div>
    </div>
  );
};

export default SPYFuturesInsights;
