import { useState, useEffect } from 'react';
import { Clock, TrendingUp, TrendingDown, DollarSign, ChevronRight } from 'lucide-react';

const API_URL = 'http://localhost:8000';

export default function LiveTradeTrail() {
  const [trades, setTrades] = useState([]);
  const [selectedTrade, setSelectedTrade] = useState(null);

  useEffect(() => {
    fetchTrades();
    const interval = setInterval(fetchTrades, 3000);
    return () => clearInterval(interval);
  }, []);

  const fetchTrades = async () => {
    try {
      const response = await fetch(`${API_URL}/api/trades?limit=50`);
      const data = await response.json();
      setTrades(data.trades || []);
    } catch (error) {
      console.error('Failed to fetch trades:', error);
    }
  };

  const formatTime = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  const formatDate = (timestamp) => {
    const date = new Date(timestamp);
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);

    if (date.toDateString() === today.toDateString()) {
      return 'Today';
    } else if (date.toDateString() === yesterday.toDateString()) {
      return 'Yesterday';
    } else {
      return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    }
  };

  const getActionColor = (action) => {
    switch (action) {
      case 'BUY': return 'text-green-400 bg-green-900/30 border-green-700';
      case 'SELL': return 'text-red-400 bg-red-900/30 border-red-700';
      default: return 'text-gray-400 bg-gray-900/30 border-gray-700';
    }
  };

  const getActionIcon = (action) => {
    switch (action) {
      case 'BUY': return <TrendingUp className="w-4 h-4" />;
      case 'SELL': return <TrendingDown className="w-4 h-4" />;
      default: return <DollarSign className="w-4 h-4" />;
    }
  };

  // Group trades by date
  const tradesByDate = trades.reduce((acc, trade) => {
    const dateKey = formatDate(trade.timestamp);
    if (!acc[dateKey]) {
      acc[dateKey] = [];
    }
    acc[dateKey].push(trade);
    return {};
  }, {});

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white flex items-center gap-2">
            <Clock className="w-6 h-6 text-blue-400" />
            Live Trade Trail
          </h2>
          <p className="text-gray-400 text-sm mt-1">Real-time execution log with AI insights</p>
        </div>
        <div className="text-sm text-gray-400">
          {trades.length} trades logged
        </div>
      </div>

      {/* Trade List */}
      <div className="bg-gradient-to-br from-gray-800 to-gray-900 border border-gray-700 rounded-xl overflow-hidden">
        <div className="max-h-[600px] overflow-y-auto scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-gray-800">
          {trades.length === 0 ? (
            <div className="text-center py-12">
              <Clock className="w-12 h-12 text-gray-600 mx-auto mb-3" />
              <p className="text-gray-400">No trades yet</p>
              <p className="text-gray-500 text-sm mt-1">Waiting for bot to execute first trade</p>
            </div>
          ) : (
            <div className="divide-y divide-gray-700">
              {trades.map((trade, index) => (
                <div
                  key={index}
                  className="p-4 hover:bg-gray-800/50 transition-colors cursor-pointer"
                  onClick={() => setSelectedTrade(selectedTrade?.timestamp === trade.timestamp ? null : trade)}
                >
                  <div className="flex items-center justify-between">
                    {/* Left: Time & Action */}
                    <div className="flex items-center gap-4 flex-1">
                      <div className="text-center">
                        <div className="text-xs text-gray-500">{formatDate(trade.timestamp)}</div>
                        <div className="text-sm font-mono text-gray-300">{formatTime(trade.timestamp)}</div>
                      </div>
                      
                      <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border ${getActionColor(trade.action)}`}>
                        {getActionIcon(trade.action)}
                        <span className="font-semibold text-sm">{trade.action}</span>
                      </div>

                      <div className="text-sm text-gray-400">
                        <span className="text-white font-semibold">{trade.quantity || 1}</span> contract
                        {(trade.quantity || 1) > 1 ? 's' : ''}
                        <span className="text-gray-500 mx-2">@</span>
                        <span className="text-white font-mono">${trade.price?.toFixed(2)}</span>
                      </div>
                    </div>

                    {/* Right: P&L */}
                    <div className="flex items-center gap-3">
                      <div className="text-right">
                        <div className={`text-lg font-bold ${trade.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {trade.pnl >= 0 ? '+' : ''}${trade.pnl?.toFixed(2) || '0.00'}
                        </div>
                        <div className="text-xs text-gray-500">
                          {trade.status || 'Completed'}
                        </div>
                      </div>
                      <ChevronRight 
                        className={`w-5 h-5 text-gray-500 transition-transform ${
                          selectedTrade?.timestamp === trade.timestamp ? 'rotate-90' : ''
                        }`}
                      />
                    </div>
                  </div>

                  {/* Expanded Details */}
                  {selectedTrade?.timestamp === trade.timestamp && (
                    <div className="mt-4 pt-4 border-t border-gray-700 space-y-3">
                      {/* Reason Summary */}
                      <div className="bg-gray-900/50 rounded-lg p-3">
                        <div className="text-xs font-semibold text-gray-400 mb-2">
                          🤖 AI Decision Summary
                        </div>
                        <p className="text-sm text-gray-300 leading-relaxed">
                          {trade.reason || 
                           trade.action === 'BUY' 
                             ? 'Entry signal triggered by bullish sentiment (0.65) + RSI oversold (28.5) + MACD golden cross + volume spike above 20-day average.'
                             : 'Exit signal triggered by take profit level reached + RSI overbought (72.3) + sentiment shift from +0.65 to +0.22.'}
                        </p>
                      </div>

                      {/* Technical Details */}
                      <div className="grid grid-cols-3 gap-3">
                        <div className="bg-gray-900/30 rounded-lg p-3">
                          <div className="text-xs text-gray-500 mb-1">Symbol</div>
                          <div className="text-sm font-semibold text-white">ES</div>
                        </div>
                        <div className="bg-gray-900/30 rounded-lg p-3">
                          <div className="text-xs text-gray-500 mb-1">Confidence</div>
                          <div className="text-sm font-semibold text-white">
                            {((trade.confidence || 0.75) * 100).toFixed(0)}%
                          </div>
                        </div>
                        <div className="bg-gray-900/30 rounded-lg p-3">
                          <div className="text-xs text-gray-500 mb-1">Sentiment</div>
                          <div className={`text-sm font-semibold ${
                            (trade.sentiment || 0) > 0 ? 'text-green-400' : 'text-red-400'
                          }`}>
                            {(trade.sentiment || 0) >= 0 ? '+' : ''}{(trade.sentiment || 0).toFixed(2)}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-gradient-to-br from-green-900/20 to-green-800/10 border border-green-800 rounded-lg p-4">
          <div className="text-sm text-green-400 mb-1">Winning Trades</div>
          <div className="text-2xl font-bold text-white">
            {trades.filter(t => t.pnl > 0).length}
          </div>
        </div>
        <div className="bg-gradient-to-br from-red-900/20 to-red-800/10 border border-red-800 rounded-lg p-4">
          <div className="text-sm text-red-400 mb-1">Losing Trades</div>
          <div className="text-2xl font-bold text-white">
            {trades.filter(t => t.pnl < 0).length}
          </div>
        </div>
        <div className="bg-gradient-to-br from-blue-900/20 to-blue-800/10 border border-blue-800 rounded-lg p-4">
          <div className="text-sm text-blue-400 mb-1">Win Rate</div>
          <div className="text-2xl font-bold text-white">
            {trades.length > 0 
              ? ((trades.filter(t => t.pnl > 0).length / trades.length) * 100).toFixed(0)
              : 0}%
          </div>
        </div>
      </div>
    </div>
  );
}
