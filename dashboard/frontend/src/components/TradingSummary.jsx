import { useState, useEffect } from 'react';
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Activity, 
  Target, 
  Clock,
  RefreshCw,
  CheckCircle,
  XCircle,
  AlertCircle
} from 'lucide-react';

const API_URL = 'http://localhost:8000';

const TradingSummary = () => {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  const fetchSummary = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_URL}/api/trading/today-summary`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      setSummary(data);
      setLastUpdated(new Date());
      setError(null);
    } catch (err) {
      console.error('Failed to fetch trading summary:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSummary();
    // Auto-refresh every 30 seconds
    const interval = setInterval(fetchSummary, 30000);
    return () => clearInterval(interval);
  }, []);

  if (loading && !summary) {
    return (
      <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-8 border border-gray-700">
        <div className="flex items-center justify-center space-x-3">
          <RefreshCw className="w-6 h-6 text-blue-400 animate-spin" />
          <span className="text-gray-400">Loading trading summary...</span>
        </div>
      </div>
    );
  }

  if (error && !summary) {
    return (
      <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-8 border border-red-700">
        <div className="flex items-center space-x-3 text-red-400">
          <AlertCircle className="w-6 h-6" />
          <span>Failed to load trading summary: {error}</span>
        </div>
        <button
          onClick={fetchSummary}
          className="mt-4 px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  const performance = summary?.performance || {};
  const orders = summary?.orders || {};
  const netPnL = performance.net_pnl || 0;
  const isProfit = netPnL >= 0;

  return (
    <div className="space-y-6">
      {/* Header with refresh button */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">Today's Trading Summary</h2>
          <p className="text-gray-400 text-sm">
            {summary?.date_cst || summary?.date} • Last updated: {summary?.timestamp || 'N/A'}
          </p>
        </div>
        <button
          onClick={fetchSummary}
          disabled={loading}
          className="flex items-center gap-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors disabled:opacity-50"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Main P&L Card */}
      <div className={`bg-gradient-to-br ${isProfit ? 'from-green-900/30 to-green-800/20 border-green-700' : 'from-red-900/30 to-red-800/20 border-red-700'} rounded-xl p-6 border`}>
        <div className="flex items-center justify-between">
          <div>
            <p className="text-gray-400 text-sm uppercase tracking-wide">Net P&L</p>
            <div className="flex items-center gap-3 mt-1">
              {isProfit ? (
                <TrendingUp className="w-8 h-8 text-green-400" />
              ) : (
                <TrendingDown className="w-8 h-8 text-red-400" />
              )}
              <span className={`text-4xl font-bold ${isProfit ? 'text-green-400' : 'text-red-400'}`}>
                ${netPnL.toFixed(2)}
              </span>
            </div>
          </div>
          <div className="text-right">
            <p className="text-gray-400 text-sm">Win Rate</p>
            <p className={`text-3xl font-bold ${(performance.win_rate || 0) >= 50 ? 'text-green-400' : 'text-yellow-400'}`}>
              {(performance.win_rate || 0).toFixed(1)}%
            </p>
          </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {/* Realized P&L */}
        <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-4 border border-gray-700">
          <div className="flex items-center gap-2 mb-2">
            <DollarSign className="w-5 h-5 text-blue-400" />
            <span className="text-gray-400 text-sm">Realized P&L</span>
          </div>
          <p className={`text-2xl font-bold ${(performance.realized_pnl || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            ${(performance.realized_pnl || 0).toFixed(2)}
          </p>
        </div>

        {/* Commission */}
        <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-4 border border-gray-700">
          <div className="flex items-center gap-2 mb-2">
            <Activity className="w-5 h-5 text-orange-400" />
            <span className="text-gray-400 text-sm">Commission</span>
          </div>
          <p className="text-2xl font-bold text-orange-400">
            ${(performance.commission || 0).toFixed(2)}
          </p>
        </div>

        {/* Winning Trades */}
        <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-4 border border-gray-700">
          <div className="flex items-center gap-2 mb-2">
            <CheckCircle className="w-5 h-5 text-green-400" />
            <span className="text-gray-400 text-sm">Winners</span>
          </div>
          <p className="text-2xl font-bold text-green-400">
            {performance.winning_trades || 0}
          </p>
        </div>

        {/* Losing Trades */}
        <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-4 border border-gray-700">
          <div className="flex items-center gap-2 mb-2">
            <XCircle className="w-5 h-5 text-red-400" />
            <span className="text-gray-400 text-sm">Losers</span>
          </div>
          <p className="text-2xl font-bold text-red-400">
            {performance.losing_trades || 0}
          </p>
        </div>
      </div>

      {/* Order Stats */}
      <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Target className="w-5 h-5 text-purple-400" />
          Order Statistics
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-3 bg-gray-800/50 rounded-lg">
            <p className="text-3xl font-bold text-white">{orders.total || 0}</p>
            <p className="text-gray-400 text-sm">Total Orders</p>
          </div>
          <div className="text-center p-3 bg-gray-800/50 rounded-lg">
            <p className="text-3xl font-bold text-green-400">{orders.filled || 0}</p>
            <p className="text-gray-400 text-sm">Filled</p>
          </div>
          <div className="text-center p-3 bg-gray-800/50 rounded-lg">
            <p className="text-3xl font-bold text-yellow-400">{orders.pending || 0}</p>
            <p className="text-gray-400 text-sm">Pending</p>
          </div>
          <div className="text-center p-3 bg-gray-800/50 rounded-lg">
            <p className="text-3xl font-bold text-gray-400">{orders.cancelled || 0}</p>
            <p className="text-gray-400 text-sm">Cancelled</p>
          </div>
        </div>
      </div>

      {/* Recent Trades */}
      {summary?.trades && summary.trades.length > 0 && (
        <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Clock className="w-5 h-5 text-blue-400" />
            Recent Filled Trades
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-400 border-b border-gray-700">
                  <th className="text-left py-2 px-3">Order ID</th>
                  <th className="text-left py-2 px-3">Symbol</th>
                  <th className="text-left py-2 px-3">Action</th>
                  <th className="text-right py-2 px-3">Qty</th>
                  <th className="text-right py-2 px-3">Fill Price</th>
                  <th className="text-right py-2 px-3">P&L</th>
                  <th className="text-left py-2 px-3">Time (CST)</th>
                </tr>
              </thead>
              <tbody>
                {summary.trades.map((trade, idx) => (
                  <tr key={trade.order_id || idx} className="border-b border-gray-800 hover:bg-gray-800/50">
                    <td className="py-2 px-3 text-gray-300 font-mono">{trade.order_id}</td>
                    <td className="py-2 px-3 text-white font-semibold">{trade.symbol}</td>
                    <td className="py-2 px-3">
                      <span className={`px-2 py-1 rounded text-xs font-semibold ${
                        trade.action === 'BUY' ? 'bg-green-900/50 text-green-400' : 'bg-red-900/50 text-red-400'
                      }`}>
                        {trade.action}
                      </span>
                    </td>
                    <td className="py-2 px-3 text-right text-gray-300">{trade.quantity}</td>
                    <td className="py-2 px-3 text-right text-gray-300">
                      {trade.avg_fill_price ? `$${trade.avg_fill_price.toFixed(2)}` : '-'}
                    </td>
                    <td className={`py-2 px-3 text-right font-semibold ${
                      (trade.realized_pnl || 0) >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {trade.realized_pnl ? `$${trade.realized_pnl.toFixed(2)}` : '-'}
                    </td>
                    <td className="py-2 px-3 text-gray-400 text-xs">
                      {trade.timestamp_cst || trade.timestamp}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Avg P&L per Trade */}
      <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-4 border border-gray-700">
        <div className="flex items-center justify-between">
          <span className="text-gray-400">Average P&L per Trade</span>
          <span className={`text-xl font-bold ${(performance.pnl_per_trade || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            ${(performance.pnl_per_trade || 0).toFixed(2)}
          </span>
        </div>
      </div>
    </div>
  );
};

export default TradingSummary;
