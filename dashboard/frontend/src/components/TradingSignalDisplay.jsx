import { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, Minus, AlertTriangle, CheckCircle, Clock } from 'lucide-react';

const API_URL = 'http://localhost:8000';

export const TradingSignalDisplay = ({ lastMessage }) => {
  const [currentSignal, setCurrentSignal] = useState(null);
  const [signalHistory, setSignalHistory] = useState([]);
  const [ragValidation, setRagValidation] = useState(null);

  useEffect(() => {
    // Update signal from WebSocket messages
    if (lastMessage) {
      if (lastMessage.type === 'signal_update') {
        const signal = lastMessage.data;
        setCurrentSignal(signal);
        
        // Add to history (keep last 10)
        setSignalHistory(prev => [
          { ...signal, timestamp: new Date() },
          ...prev.slice(0, 9)
        ]);
        
        // Extract RAG validation if present
        if (signal.metadata?.llm_recommendation) {
          setRagValidation(signal.metadata.llm_recommendation);
        }
      }
    }
  }, [lastMessage]);

  const getSignalColor = (action) => {
    switch (action) {
      case 'BUY':
        return 'bg-green-500 text-white';
      case 'SELL':
        return 'bg-red-500 text-white';
      case 'HOLD':
        return 'bg-gray-400 text-white';
      default:
        return 'bg-gray-300 text-gray-700';
    }
  };

  const getSignalIcon = (action) => {
    switch (action) {
      case 'BUY':
        return <TrendingUp className="w-8 h-8" />;
      case 'SELL':
        return <TrendingDown className="w-8 h-8" />;
      case 'HOLD':
        return <Minus className="w-8 h-8" />;
      default:
        return <AlertTriangle className="w-8 h-8" />;
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return '';
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit',
      second: '2-digit'
    });
  };

  if (!currentSignal) {
    return (
      <div className="bg-white rounded-2xl shadow-apple p-6">
        <div className="flex items-center justify-center py-8">
          <div className="text-center">
            <Clock className="w-12 h-12 text-gray-400 mx-auto mb-3" />
            <p className="text-gray-500">Waiting for trading signals...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Current Signal Card */}
      <div className="bg-white rounded-2xl shadow-apple p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-apple-gray-900">Current Signal</h3>
          <span className="text-xs text-gray-500">{formatTimestamp(currentSignal.timestamp)}</span>
        </div>

        <div className="flex items-center space-x-6">
          {/* Signal Badge */}
          <div className={`${getSignalColor(currentSignal.action)} rounded-2xl p-6 flex items-center justify-center`}>
            {getSignalIcon(currentSignal.action)}
          </div>

          {/* Signal Details */}
          <div className="flex-1">
            <div className="text-3xl font-bold text-apple-gray-900 mb-2">
              {currentSignal.action}
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-600">Confidence:</span>
                <span className={`text-sm font-semibold ${getConfidenceColor(currentSignal.confidence)}`}>
                  {(currentSignal.confidence * 100).toFixed(1)}%
                </span>
              </div>

              {currentSignal.metadata?.market_bias && (
                <div className="flex items-center space-x-2">
                  <span className="text-sm text-gray-600">Market:</span>
                  <span className="text-sm font-medium text-apple-gray-700">
                    {currentSignal.metadata.market_bias} bias
                  </span>
                  <span className="text-xs text-gray-500">•</span>
                  <span className="text-sm font-medium text-apple-gray-700">
                    {currentSignal.metadata.volatility} volatility
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Risk Parameters */}
        {currentSignal.metadata?.risk_params && (
          <div className="mt-4 pt-4 border-t border-gray-100">
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-gray-50 rounded-lg p-3">
                <div className="text-xs text-gray-500 mb-1">Stop Loss</div>
                <div className="text-sm font-semibold text-red-600">
                  ${currentSignal.metadata.risk_params.stop_loss_long?.toFixed(2) || 
                    currentSignal.metadata.risk_params.stop_loss_short?.toFixed(2) || 'N/A'}
                </div>
              </div>
              
              <div className="bg-gray-50 rounded-lg p-3">
                <div className="text-xs text-gray-500 mb-1">Take Profit</div>
                <div className="text-sm font-semibold text-green-600">
                  ${currentSignal.metadata.risk_params.take_profit_long?.toFixed(2) || 
                    currentSignal.metadata.risk_params.take_profit_short?.toFixed(2) || 'N/A'}
                </div>
              </div>
              
              <div className="bg-gray-50 rounded-lg p-3">
                <div className="text-xs text-gray-500 mb-1">ATR</div>
                <div className="text-sm font-semibold text-apple-gray-700">
                  {currentSignal.metadata.risk_params.atr?.toFixed(2) || 'N/A'}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* RAG Validation */}
        {ragValidation && (
          <div className="mt-4 pt-4 border-t border-gray-100">
            <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
              <div className="flex items-start space-x-3">
                <CheckCircle className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" />
                <div className="flex-1">
                  <div className="text-sm font-semibold text-blue-900 mb-1">
                    RAG Validation
                  </div>
                  <div className="text-xs text-blue-700 mb-2">
                    <span className="font-medium">Recommendation:</span> {ragValidation.action} 
                    <span className="ml-2">({(ragValidation.confidence * 100).toFixed(0)}% confidence)</span>
                  </div>
                  {ragValidation.reasoning && (
                    <div className="text-xs text-blue-600">
                      {ragValidation.reasoning.substring(0, 150)}
                      {ragValidation.reasoning.length > 150 ? '...' : ''}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Signal History */}
      {signalHistory.length > 0 && (
        <div className="bg-white rounded-2xl shadow-apple p-6">
          <h3 className="text-lg font-semibold text-apple-gray-900 mb-4">Recent Signals</h3>
          <div className="space-y-2">
            {signalHistory.slice(0, 5).map((signal, index) => (
              <div 
                key={index}
                className="flex items-center justify-between py-2 px-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
              >
                <div className="flex items-center space-x-3">
                  <span className={`px-3 py-1 rounded-full text-xs font-medium ${getSignalColor(signal.action)}`}>
                    {signal.action}
                  </span>
                  <span className={`text-sm font-medium ${getConfidenceColor(signal.confidence)}`}>
                    {(signal.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <span className="text-xs text-gray-500">
                  {formatTimestamp(signal.timestamp)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default TradingSignalDisplay;
