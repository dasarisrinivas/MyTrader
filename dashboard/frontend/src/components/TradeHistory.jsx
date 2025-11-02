import { ArrowUpCircle, ArrowDownCircle } from 'lucide-react';

export const TradeHistory = ({ trades }) => {
  // Handle both array and object responses
  const tradesArray = Array.isArray(trades) ? trades : (trades?.trades || []);
  
  if (!tradesArray || tradesArray.length === 0) {
    return (
      <div className="bg-white rounded-2xl shadow-apple p-6">
        <h3 className="text-xl font-semibold text-apple-gray-800 mb-4">Recent Trades</h3>
        <div className="flex items-center justify-center h-32 text-apple-gray-400">
          <p>No trades yet</p>
        </div>
      </div>
    );
  }

  const formatDate = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  return (
    <div className="bg-white rounded-2xl shadow-apple p-6">
      <h3 className="text-xl font-semibold text-apple-gray-800 mb-4">Recent Trades</h3>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-gray-200">
              <th className="text-left py-3 px-4 text-sm font-medium text-apple-gray-600">Time</th>
              <th className="text-left py-3 px-4 text-sm font-medium text-apple-gray-600">Symbol</th>
              <th className="text-left py-3 px-4 text-sm font-medium text-apple-gray-600">Side</th>
              <th className="text-right py-3 px-4 text-sm font-medium text-apple-gray-600">Quantity</th>
              <th className="text-right py-3 px-4 text-sm font-medium text-apple-gray-600">Price</th>
              <th className="text-right py-3 px-4 text-sm font-medium text-apple-gray-600">P&L</th>
            </tr>
          </thead>
          <tbody>
            {tradesArray.map((trade, index) => (
              <tr key={index} className="border-b border-gray-100 hover:bg-gray-50 transition-colors">
                <td className="py-3 px-4 text-sm text-apple-gray-700">
                  {formatDate(trade.timestamp)}
                </td>
                <td className="py-3 px-4 text-sm font-medium text-apple-gray-800">
                  {trade.symbol}
                </td>
                <td className="py-3 px-4">
                  <div className="flex items-center space-x-2">
                    {trade.side === 'BUY' ? (
                      <>
                        <ArrowUpCircle className="w-4 h-4 text-green-500" />
                        <span className="text-sm font-medium text-green-600">Buy</span>
                      </>
                    ) : (
                      <>
                        <ArrowDownCircle className="w-4 h-4 text-red-500" />
                        <span className="text-sm font-medium text-red-600">Sell</span>
                      </>
                    )}
                  </div>
                </td>
                <td className="py-3 px-4 text-sm text-right text-apple-gray-700">
                  {trade.quantity}
                </td>
                <td className="py-3 px-4 text-sm text-right text-apple-gray-700">
                  ${trade.price?.toFixed(2) || '0.00'}
                </td>
                <td className="py-3 px-4 text-sm text-right">
                  <span className={`font-medium ${
                    trade.pnl >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    ${trade.pnl?.toFixed(2) || '0.00'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
