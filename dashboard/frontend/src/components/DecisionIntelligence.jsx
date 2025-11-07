import { useState, useEffect } from 'react';
import { Brain, TrendingUp, TrendingDown, AlertCircle, CheckCircle, Minus } from 'lucide-react';

const API_URL = 'http://localhost:8000';

export default function DecisionIntelligence() {
  const [latestSignal, setLatestSignal] = useState(null);
  const [sentiment, setSentiment] = useState(0);
  const [lastTrade, setLastTrade] = useState(null);

  useEffect(() => {
    fetchDecisionData();
    const interval = setInterval(fetchDecisionData, 3000);
    return () => clearInterval(interval);
  }, []);

  const fetchDecisionData = async () => {
    try {
      const [statusRes, tradesRes] = await Promise.all([
        fetch(`${API_URL}/api/trading/status`),
        fetch(`${API_URL}/api/trades?limit=1`)
      ]);

      const status = await statusRes.json();
      const trades = await tradesRes.json();

      setLatestSignal({
        action: status.last_signal || 'HOLD',
        confidence: status.signal_confidence || 0,
        reason: status.signal_reason || 'Analyzing market conditions...'
      });

      setSentiment(status.sentiment_score || 0);

      if (trades.trades && trades.trades.length > 0) {
        setLastTrade(trades.trades[0]);
      }
    } catch (error) {
      console.error('Failed to fetch decision data:', error);
    }
  };

  const getSignalColor = (action) => {
    switch (action) {
      case 'BUY': return 'text-green-400';
      case 'SELL': return 'text-red-400';
      case 'HOLD': return 'text-yellow-400';
      default: return 'text-gray-400';
    }
  };

  const getSignalBg = (action) => {
    switch (action) {
      case 'BUY': return 'from-green-900/40 to-green-800/20 border-green-700';
      case 'SELL': return 'from-red-900/40 to-red-800/20 border-red-700';
      case 'HOLD': return 'from-yellow-900/40 to-yellow-800/20 border-yellow-700';
      default: return 'from-gray-900/40 to-gray-800/20 border-gray-700';
    }
  };

  const getSignalIcon = (action) => {
    switch (action) {
      case 'BUY': return <TrendingUp className="w-8 h-8" />;
      case 'SELL': return <TrendingDown className="w-8 h-8" />;
      case 'HOLD': return <Minus className="w-8 h-8" />;
      default: return <AlertCircle className="w-8 h-8" />;
    }
  };

  const getSentimentLabel = (score) => {
    if (score > 0.3) return { label: 'Bullish', color: 'text-green-400' };
    if (score < -0.3) return { label: 'Bearish', color: 'text-red-400' };
    return { label: 'Neutral', color: 'text-gray-400' };
  };

  const sentimentLabel = getSentimentLabel(sentiment);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Brain className="w-8 h-8 text-purple-400" />
        <div>
          <h2 className="text-2xl font-bold text-white">Decision Intelligence</h2>
          <p className="text-gray-400 text-sm">AI-powered trading signals and reasoning</p>
        </div>
      </div>

      {/* Current Signal */}
      {latestSignal && (
        <div className={`bg-gradient-to-r ${getSignalBg(latestSignal.action)} border rounded-xl p-6`}>
          <div className="flex items-start justify-between mb-4">
            <div className="flex items-center gap-4">
              <div className={`p-4 rounded-xl bg-gray-900/50 ${getSignalColor(latestSignal.action)}`}>
                {getSignalIcon(latestSignal.action)}
              </div>
              <div>
                <div className="text-sm text-gray-400 mb-1">Current Signal</div>
                <div className={`text-3xl font-bold ${getSignalColor(latestSignal.action)}`}>
                  {latestSignal.action}
                </div>
              </div>
            </div>
            <div className="text-right">
              <div className="text-sm text-gray-400 mb-1">Confidence</div>
              <div className="text-3xl font-bold text-white">
                {(latestSignal.confidence * 100).toFixed(0)}%
              </div>
            </div>
          </div>

          {/* Confidence Bar */}
          <div className="mb-4">
            <div className="w-full bg-gray-800 rounded-full h-3 overflow-hidden">
              <div
                className={`h-full transition-all duration-500 ${
                  latestSignal.confidence >= 0.7 ? 'bg-green-500' :
                  latestSignal.confidence >= 0.5 ? 'bg-yellow-500' : 'bg-red-500'
                }`}
                style={{ width: `${latestSignal.confidence * 100}%` }}
              />
            </div>
          </div>

          {/* Reasoning */}
          <div className="bg-gray-900/50 rounded-lg p-4">
            <div className="text-sm font-semibold text-gray-300 mb-2 flex items-center gap-2">
              <Brain className="w-4 h-4" />
              AI Reasoning
            </div>
            <p className="text-gray-400 text-sm leading-relaxed">
              {latestSignal.reason}
            </p>
          </div>
        </div>
      )}

      {/* Market Sentiment */}
      <div className="bg-gradient-to-br from-gray-800 to-gray-900 border border-gray-700 rounded-xl p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <div className="text-sm text-gray-400 mb-1">Market Sentiment</div>
            <div className={`text-2xl font-bold ${sentimentLabel.color}`}>
              {sentimentLabel.label}
            </div>
          </div>
          <div className="text-right">
            <div className="text-sm text-gray-400 mb-1">Score</div>
            <div className={`text-2xl font-bold ${sentimentLabel.color}`}>
              {sentiment >= 0 ? '+' : ''}{sentiment.toFixed(2)}
            </div>
          </div>
        </div>

        {/* Sentiment Gauge */}
        <div className="relative">
          <div className="flex justify-between items-center mb-2">
            <span className="text-xs text-red-400">Bearish</span>
            <span className="text-xs text-gray-400">Neutral</span>
            <span className="text-xs text-green-400">Bullish</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-4 relative overflow-hidden">
            <div className="absolute inset-0 flex">
              <div className="w-1/3 bg-gradient-to-r from-red-600 to-red-500"></div>
              <div className="w-1/3 bg-gray-600"></div>
              <div className="w-1/3 bg-gradient-to-r from-green-500 to-green-600"></div>
            </div>
            <div
              className="absolute top-0 h-full w-1 bg-white shadow-lg transition-all duration-500"
              style={{ left: `${((sentiment + 1) / 2) * 100}%` }}
            >
              <div className="absolute -top-6 left-1/2 transform -translate-x-1/2">
                <div className="w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-white"></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Last Trade Decision */}
      {lastTrade && (
        <div className="bg-gradient-to-br from-gray-800 to-gray-900 border border-gray-700 rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <CheckCircle className="w-5 h-5 text-blue-400" />
            Last Trade Decision
          </h3>

          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <div className="text-sm text-gray-400 mb-1">Action</div>
              <div className={`text-xl font-bold ${getSignalColor(lastTrade.action)}`}>
                {lastTrade.action}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-400 mb-1">P&L</div>
              <div className={`text-xl font-bold ${lastTrade.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                ${lastTrade.pnl?.toFixed(2) || '0.00'}
              </div>
            </div>
          </div>

          {/* Entry Reason */}
          {lastTrade.action !== 'SELL' && (
            <div className="bg-green-900/20 border border-green-800 rounded-lg p-3 mb-3">
              <div className="text-xs font-semibold text-green-400 mb-1">📈 Entry Reason</div>
              <p className="text-sm text-gray-300">
                {lastTrade.entry_reason || 'Bullish sentiment + RSI oversold + volume spike + MACD golden cross'}
              </p>
            </div>
          )}

          {/* Exit Reason */}
          {lastTrade.action === 'SELL' && (
            <div className="bg-red-900/20 border border-red-800 rounded-lg p-3">
              <div className="text-xs font-semibold text-red-400 mb-1">📉 Exit Reason</div>
              <p className="text-sm text-gray-300">
                {lastTrade.exit_reason || 'Take profit target reached + sentiment shift + RSI overbought'}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
