import { useState, useEffect } from 'react';
import { 
  TrendingUp, 
  Activity, 
  BarChart3, 
  Target, 
  Shield,
  AlertCircle,
  CheckCircle2,
  Clock,
  DollarSign
} from 'lucide-react';

const API_URL = 'http://localhost:8000';

export const MarketStatus = () => {
  const [marketStatus, setMarketStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchMarketStatus();
    
    // Poll every 3 seconds for real-time updates
    const interval = setInterval(() => {
      fetchMarketStatus();
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const fetchMarketStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/api/market/status`);
      const data = await response.json();
      setMarketStatus(data);
      setLoading(false);
    } catch (err) {
      console.error('Failed to fetch market status:', err);
      setLoading(false);
    }
  };

  if (loading || !marketStatus) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-4 bg-gray-200 rounded w-1/4"></div>
          <div className="h-4 bg-gray-200 rounded w-1/2"></div>
          <div className="h-4 bg-gray-200 rounded w-3/4"></div>
        </div>
      </div>
    );
  }

  const getSignalColor = (signal) => {
    if (signal === 'BUY') return 'text-green-600 bg-green-50';
    if (signal === 'SELL') return 'text-red-600 bg-red-50';
    return 'text-yellow-600 bg-yellow-50';
  };

  const getMarketBiasColor = (bias) => {
    if (bias === 'bullish') return 'text-green-600';
    if (bias === 'bearish') return 'text-red-600';
    return 'text-gray-600';
  };

  const getVolatilityColor = (vol) => {
    if (vol === 'high') return 'text-orange-600';
    if (vol === 'low') return 'text-blue-600';
    return 'text-gray-600';
  };

  const anyConditionMet = Object.values(marketStatus.resume_conditions || {}).some(v => v === true);

  return (
    <div className="space-y-4">
      {/* Ticker Information Card */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
            <Activity className="w-5 h-5 text-blue-500" />
            Market Information
          </h3>
          <div className="text-xs text-gray-500">
            Updated: {new Date(marketStatus.last_update).toLocaleTimeString()}
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="space-y-1">
            <div className="text-xs text-gray-500 uppercase">Symbol</div>
            <div className="text-xl font-bold text-gray-900">{marketStatus.symbol}</div>
          </div>
          
          <div className="space-y-1">
            <div className="text-xs text-gray-500 uppercase">Exchange</div>
            <div className="text-xl font-bold text-gray-900">{marketStatus.exchange}</div>
          </div>
          
          <div className="space-y-1">
            <div className="text-xs text-gray-500 uppercase">Contract</div>
            <div className="text-xl font-bold text-gray-900">{marketStatus.contract}</div>
          </div>
          
          <div className="space-y-1">
            <div className="text-xs text-gray-500 uppercase flex items-center gap-1">
              <DollarSign className="w-3 h-3" />
              Current Price
            </div>
            <div className="text-xl font-bold text-blue-600">
              {marketStatus.current_price ? marketStatus.current_price.toFixed(2) : 'N/A'}
            </div>
          </div>
        </div>
      </div>

      {/* Signal Status Card */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-blue-500" />
          Current Signal Status
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="space-y-2">
            <div className="text-xs text-gray-500 uppercase">Last Signal</div>
            <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-semibold ${getSignalColor(marketStatus.last_signal)}`}>
              {marketStatus.last_signal}
            </div>
          </div>
          
          <div className="space-y-2">
            <div className="text-xs text-gray-500 uppercase">Confidence</div>
            <div className="flex items-center gap-2">
              <div className="text-lg font-bold text-gray-900">
                {(marketStatus.signal_confidence * 100).toFixed(1)}%
              </div>
              <div className="flex-1 bg-gray-200 rounded-full h-2">
                <div 
                  className={`h-2 rounded-full transition-all ${
                    marketStatus.signal_confidence >= 0.65 ? 'bg-green-500' : 'bg-yellow-500'
                  }`}
                  style={{ width: `${marketStatus.signal_confidence * 100}%` }}
                ></div>
              </div>
            </div>
          </div>
          
          <div className="space-y-2">
            <div className="text-xs text-gray-500 uppercase">Active Strategy</div>
            <div className="text-lg font-bold text-gray-900 capitalize">
              {marketStatus.active_strategy}
            </div>
          </div>
        </div>

        {/* Risk Parameters */}
        <div className="mt-4 pt-4 border-t border-gray-200 grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="space-y-1">
            <div className="text-xs text-gray-500 uppercase">Market Bias</div>
            <div className={`text-sm font-semibold capitalize ${getMarketBiasColor(marketStatus.market_bias)}`}>
              {marketStatus.market_bias}
            </div>
          </div>
          
          <div className="space-y-1">
            <div className="text-xs text-gray-500 uppercase">Volatility</div>
            <div className={`text-sm font-semibold capitalize ${getVolatilityColor(marketStatus.volatility_level)}`}>
              {marketStatus.volatility_level}
            </div>
          </div>
          
          <div className="space-y-1">
            <div className="text-xs text-gray-500 uppercase flex items-center gap-1">
              <Shield className="w-3 h-3" />
              Stop Loss
            </div>
            <div className="text-sm font-semibold text-red-600">
              {marketStatus.stop_loss ? marketStatus.stop_loss.toFixed(2) : 'N/A'}
            </div>
          </div>
          
          <div className="space-y-1">
            <div className="text-xs text-gray-500 uppercase flex items-center gap-1">
              <Target className="w-3 h-3" />
              Take Profit
            </div>
            <div className="text-sm font-semibold text-green-600">
              {marketStatus.take_profit ? marketStatus.take_profit.toFixed(2) : 'N/A'}
            </div>
          </div>
        </div>

        {marketStatus.atr && (
          <div className="mt-2 text-xs text-gray-600">
            ATR: <span className="font-semibold">{marketStatus.atr.toFixed(2)}</span>
          </div>
        )}
      </div>

      {/* Resume Conditions Card */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
            <Clock className="w-5 h-5 text-blue-500" />
            Trading Resume Conditions
          </h3>
          {anyConditionMet ? (
            <span className="px-3 py-1 bg-green-100 text-green-700 text-xs font-semibold rounded-full">
              Conditions Met
            </span>
          ) : (
            <span className="px-3 py-1 bg-yellow-100 text-yellow-700 text-xs font-semibold rounded-full">
              Waiting
            </span>
          )}
        </div>

        <div className="space-y-3">
          {/* Breakout Detection */}
          <div className={`flex items-start gap-3 p-3 rounded-lg ${
            marketStatus.resume_conditions.breakout_detected 
              ? 'bg-green-50 border border-green-200' 
              : 'bg-gray-50 border border-gray-200'
          }`}>
            {marketStatus.resume_conditions.breakout_detected ? (
              <CheckCircle2 className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
            ) : (
              <AlertCircle className="w-5 h-5 text-gray-400 flex-shrink-0 mt-0.5" />
            )}
            <div className="flex-1">
              <div className="font-semibold text-sm text-gray-900 mb-1">Market Breakout</div>
              <div className="text-xs text-gray-600">
                {marketStatus.resume_triggers.breakout}
              </div>
            </div>
          </div>

          {/* Confidence Threshold */}
          <div className={`flex items-start gap-3 p-3 rounded-lg ${
            marketStatus.resume_conditions.confidence_threshold_met 
              ? 'bg-green-50 border border-green-200' 
              : 'bg-gray-50 border border-gray-200'
          }`}>
            {marketStatus.resume_conditions.confidence_threshold_met ? (
              <CheckCircle2 className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
            ) : (
              <AlertCircle className="w-5 h-5 text-gray-400 flex-shrink-0 mt-0.5" />
            )}
            <div className="flex-1">
              <div className="font-semibold text-sm text-gray-900 mb-1">Confidence Threshold</div>
              <div className="text-xs text-gray-600">
                {marketStatus.resume_triggers.confidence}
              </div>
            </div>
          </div>

          {/* Market Context Change */}
          <div className={`flex items-start gap-3 p-3 rounded-lg ${
            marketStatus.resume_conditions.market_context_changed 
              ? 'bg-green-50 border border-green-200' 
              : 'bg-gray-50 border border-gray-200'
          }`}>
            {marketStatus.resume_conditions.market_context_changed ? (
              <CheckCircle2 className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
            ) : (
              <AlertCircle className="w-5 h-5 text-gray-400 flex-shrink-0 mt-0.5" />
            )}
            <div className="flex-1">
              <div className="font-semibold text-sm text-gray-900 mb-1">Market Context Change</div>
              <div className="text-xs text-gray-600">
                {marketStatus.resume_triggers.market_bias} or {marketStatus.resume_triggers.volatility}
              </div>
            </div>
          </div>

          {/* Strategy Switch */}
          <div className={`flex items-start gap-3 p-3 rounded-lg ${
            marketStatus.resume_conditions.strategy_switched 
              ? 'bg-green-50 border border-green-200' 
              : 'bg-gray-50 border border-gray-200'
          }`}>
            {marketStatus.resume_conditions.strategy_switched ? (
              <CheckCircle2 className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
            ) : (
              <AlertCircle className="w-5 h-5 text-gray-400 flex-shrink-0 mt-0.5" />
            )}
            <div className="flex-1">
              <div className="font-semibold text-sm text-gray-900 mb-1">Strategy Switch</div>
              <div className="text-xs text-gray-600">
                {marketStatus.resume_triggers.strategy}
              </div>
            </div>
          </div>
        </div>

        {/* Status Message */}
        <div className="mt-4 pt-4 border-t border-gray-200">
          <div className="flex items-start gap-2 text-sm">
            <TrendingUp className="w-4 h-4 text-blue-500 flex-shrink-0 mt-0.5" />
            <p className="text-gray-600">
              {anyConditionMet ? (
                <span className="text-green-600 font-semibold">
                  System is actively looking for entry opportunities!
                </span>
              ) : (
                <span>
                  System is in <span className="font-semibold text-yellow-600">HOLD</span> mode. 
                  Trading will resume automatically when any of the above conditions are met.
                </span>
              )}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MarketStatus;
