import { Shield, Target, TrendingUp } from 'lucide-react';

export default function OrderPriceChart({ order, currentPrice }) {
  if (!order || !order.entry_price) return null;

  const { entry_price, stop_loss, take_profit, action } = order;
  
  // Calculate price range for display
  const prices = [entry_price, stop_loss, take_profit, currentPrice].filter(p => p != null);
  const minPrice = Math.min(...prices) * 0.999;
  const maxPrice = Math.max(...prices) * 1.001;
  const priceRange = maxPrice - minPrice;

  const getPosition = (price) => {
    if (!price) return 0;
    return ((price - minPrice) / priceRange) * 100;
  };

  const entryPos = getPosition(entry_price);
  const slPos = stop_loss ? getPosition(stop_loss) : null;
  const tpPos = take_profit ? getPosition(take_profit) : null;
  const currentPos = currentPrice ? getPosition(currentPrice) : null;

  // Calculate P&L
  const calculatePnL = () => {
    if (!currentPrice) return null;
    const multiplier = action === 'BUY' ? 1 : -1;
    const pnl = (currentPrice - entry_price) * multiplier;
    return pnl;
  };

  const pnl = calculatePnL();
  const pnlPercent = pnl ? ((pnl / entry_price) * 100) : 0;

  return (
    <div className="relative w-full h-48 bg-gradient-to-b from-gray-50 to-white border border-gray-200 rounded-lg p-4">
      {/* Price scale */}
      <div className="absolute left-2 top-0 bottom-0 flex flex-col justify-between text-xs text-gray-500 font-mono">
        <div>${maxPrice.toFixed(2)}</div>
        <div>${((maxPrice + minPrice) / 2).toFixed(2)}</div>
        <div>${minPrice.toFixed(2)}</div>
      </div>

      {/* Chart area */}
      <div className="ml-20 mr-8 h-full relative">
        {/* Take Profit Line */}
        {tpPos !== null && (
          <div 
            className="absolute w-full border-t-2 border-green-500 border-dashed"
            style={{ bottom: `${tpPos}%` }}
          >
            <div className="absolute -right-6 -top-3 flex items-center gap-1 text-xs font-semibold text-green-600">
              <Target className="w-3 h-3" />
              TP
            </div>
            <div className="absolute left-2 -top-3 text-xs text-green-600 font-mono">
              ${take_profit.toFixed(2)}
            </div>
          </div>
        )}

        {/* Entry Price Line */}
        <div 
          className="absolute w-full border-t-2 border-blue-500"
          style={{ bottom: `${entryPos}%` }}
        >
          <div className="absolute -right-6 -top-3 flex items-center gap-1 text-xs font-semibold text-blue-600">
            <TrendingUp className="w-3 h-3" />
            Entry
          </div>
          <div className="absolute left-2 -top-3 text-xs text-blue-600 font-mono font-bold">
            ${entry_price.toFixed(2)}
          </div>
        </div>

        {/* Stop Loss Line */}
        {slPos !== null && (
          <div 
            className="absolute w-full border-t-2 border-red-500 border-dashed"
            style={{ bottom: `${slPos}%` }}
          >
            <div className="absolute -right-6 -top-3 flex items-center gap-1 text-xs font-semibold text-red-600">
              <Shield className="w-3 h-3" />
              SL
            </div>
            <div className="absolute left-2 -top-3 text-xs text-red-600 font-mono">
              ${stop_loss.toFixed(2)}
            </div>
          </div>
        )}

        {/* Current Price Line */}
        {currentPos !== null && (
          <>
            <div 
              className="absolute w-full border-t-2 border-purple-500"
              style={{ bottom: `${currentPos}%` }}
            >
              <div className="absolute -right-6 -top-3 text-xs font-semibold text-purple-600">
                Now
              </div>
              <div className="absolute left-2 -top-3 text-xs text-purple-600 font-mono font-bold">
                ${currentPrice.toFixed(2)}
              </div>
            </div>
            
            {/* Animated pulse at current price */}
            <div 
              className="absolute left-0 w-2 h-2 bg-purple-500 rounded-full animate-pulse"
              style={{ bottom: `calc(${currentPos}% - 4px)` }}
            />
          </>
        )}

        {/* Profit/Loss zone shading */}
        {currentPos !== null && entryPos !== null && (
          <div
            className={`absolute left-0 right-0 ${
              currentPos > entryPos 
                ? 'bg-green-100 opacity-20' 
                : 'bg-red-100 opacity-20'
            }`}
            style={{
              bottom: `${Math.min(entryPos, currentPos)}%`,
              height: `${Math.abs(currentPos - entryPos)}%`
            }}
          />
        )}
      </div>

      {/* P&L Display */}
      {pnl !== null && (
        <div className="absolute top-2 right-2 px-3 py-1.5 rounded-lg bg-white border shadow-sm">
          <div className="text-xs text-gray-600 mb-0.5">Unrealized P&L</div>
          <div className={`text-lg font-bold ${pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            {pnl >= 0 ? '+' : ''}${(pnl * 50).toFixed(2)}
          </div>
          <div className={`text-xs font-semibold ${pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            {pnl >= 0 ? '+' : ''}{pnlPercent.toFixed(2)}%
          </div>
        </div>
      )}

      {/* Direction Badge */}
      <div className={`absolute bottom-2 right-2 px-3 py-1 rounded-full text-xs font-semibold ${
        action === 'BUY' 
          ? 'bg-green-100 text-green-700' 
          : 'bg-red-100 text-red-700'
      }`}>
        {action} Position
      </div>
    </div>
  );
}
