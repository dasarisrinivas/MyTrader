import { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';

function BacktestResults({ results }) {
  const [showAllTrades, setShowAllTrades] = useState(false);

  if (!results) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold mb-4 text-gray-800">Backtest Results</h2>
        <p className="text-gray-500">Run a backtest to see results here.</p>
      </div>
    );
  }

  const { metrics, equity_curve, trades, total_trades } = results;

  // Prepare equity curve data for chart
  const equityChartData = {
    labels: equity_curve?.map(point => {
      const date = new Date(point.timestamp);
      return date.toLocaleTimeString();
    }) || [],
    datasets: [
      {
        label: 'Portfolio Value',
        data: equity_curve?.map(point => point.equity) || [],
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        borderWidth: 2,
        fill: true,
        tension: 0.4,
      }
    ]
  };

  const equityChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: 'top',
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        callbacks: {
          label: function(context) {
            return `Value: $${context.parsed.y.toFixed(2)}`;
          }
        }
      }
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Time'
        }
      },
      y: {
        display: true,
        title: {
          display: true,
          text: 'Portfolio Value ($)'
        },
        ticks: {
          callback: function(value) {
            return '$' + value.toFixed(0);
          }
        }
      }
    }
  };

  const displayedTrades = showAllTrades ? trades : trades?.slice(0, 10);

  // Get total P&L from metrics (backend calculates it)
  const totalPnL = metrics?.total_pnl || 0;
  const initialCapital = metrics?.initial_capital || 100000;
  const finalCapital = metrics?.final_capital || 100000;

  return (
    <div className="space-y-6">
      {/* Performance Metrics */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold mb-4 text-gray-800">Performance Metrics</h2>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-4 rounded-lg border-2 border-blue-200">
            <p className="text-sm text-blue-700 font-semibold">Total Profit/Loss</p>
            <p className={`text-3xl font-bold ${
              totalPnL >= 0 ? 'text-green-600' : 'text-red-600'
            }`}>
              {totalPnL >= 0 ? '+' : ''}${totalPnL.toFixed(2)}
            </p>
            <p className="text-xs text-blue-600 mt-1">
              ${initialCapital.toLocaleString()} → ${finalCapital.toLocaleString()}
            </p>
          </div>

          <div className="bg-gray-50 p-4 rounded-lg">
            <p className="text-sm text-gray-600">Total Return</p>
            <p className={`text-2xl font-bold ${
              metrics?.total_return >= 0 ? 'text-green-600' : 'text-red-600'
            }`}>
              {((metrics?.total_return || 0) * 100).toFixed(2)}%
            </p>
          </div>
          
          <div className="bg-gray-50 p-4 rounded-lg">
            <p className="text-sm text-gray-600">Sharpe Ratio</p>
            <p className={`text-2xl font-bold ${
              (metrics?.sharpe_ratio || 0) >= 1 ? 'text-green-600' : 
              (metrics?.sharpe_ratio || 0) >= 0 ? 'text-blue-600' : 'text-red-600'
            }`}>
              {(metrics?.sharpe_ratio || 0).toFixed(2)}
            </p>
          </div>
          
          <div className="bg-gray-50 p-4 rounded-lg">
            <p className="text-sm text-gray-600">Max Drawdown</p>
            <p className="text-2xl font-bold text-red-600">
              {((metrics?.max_drawdown || 0) * 100).toFixed(2)}%
            </p>
          </div>
          
          <div className="bg-gray-50 p-4 rounded-lg">
            <p className="text-sm text-gray-600">Win Rate</p>
            <p className={`text-2xl font-bold ${
              (metrics?.win_rate || 0) >= 0.5 ? 'text-green-600' : 
              (metrics?.win_rate || 0) >= 0.4 ? 'text-yellow-600' : 'text-red-600'
            }`}>
              {((metrics?.win_rate || 0) * 100).toFixed(1)}%
            </p>
          </div>

          <div className="bg-gray-50 p-4 rounded-lg">
            <p className="text-sm text-gray-600">Total Trades</p>
            <p className="text-2xl font-bold text-gray-800">
              {metrics?.total_trades || total_trades || 0}
            </p>
          </div>

          <div className="bg-gray-50 p-4 rounded-lg">
            <p className="text-sm text-gray-600">Avg Trade P&L</p>
            <p className={`text-2xl font-bold ${
              (metrics?.avg_trade || 0) >= 0 ? 'text-green-600' : 'text-red-600'
            }`}>
              ${(metrics?.avg_trade || 0).toFixed(2)}
            </p>
          </div>

          <div className="bg-gray-50 p-4 rounded-lg">
            <p className="text-sm text-gray-600">Profit Factor</p>
            <p className={`text-2xl font-bold ${
              (metrics?.profit_factor || 0) >= 1.5 ? 'text-green-600' :
              (metrics?.profit_factor || 0) >= 1.0 ? 'text-blue-600' : 'text-red-600'
            }`}>
              {(metrics?.profit_factor || 0).toFixed(2)}
            </p>
          </div>

          <div className="bg-gray-50 p-4 rounded-lg">
            <p className="text-sm text-gray-600">Winning Trades</p>
            <p className="text-2xl font-bold text-green-600">
              {metrics?.winning_trades || 0}
            </p>
          </div>
        </div>
      </div>

      {/* Equity Curve Chart */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold mb-4 text-gray-800">Equity Curve</h2>
        <div style={{ height: '400px' }}>
          <Line data={equityChartData} options={equityChartOptions} />
        </div>
        <p className="text-sm text-gray-500 mt-2 text-center">
          Showing last {equity_curve?.length || 0} data points
        </p>
      </div>

      {/* Trade History */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-bold text-gray-800">Trade History</h2>
          <button
            onClick={() => setShowAllTrades(!showAllTrades)}
            className="text-blue-600 hover:text-blue-700 text-sm font-medium"
          >
            {showAllTrades ? 'Show Less' : `Show All (${trades?.length || 0})`}
          </button>
        </div>
        
        {trades && trades.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Time
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Action
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Price
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Quantity
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    P&L
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {displayedTrades.map((trade, idx) => (
                  <tr key={idx} className="hover:bg-gray-50">
                    <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900">
                      {new Date(trade.timestamp).toLocaleString()}
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap">
                      <span className={`px-2 py-1 text-xs font-medium rounded ${
                        trade.action === 'BUY' 
                          ? 'bg-green-100 text-green-800'
                          : 'bg-red-100 text-red-800'
                      }`}>
                        {trade.action}
                      </span>
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900">
                      ${trade.price.toFixed(2)}
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900">
                      {trade.quantity}
                    </td>
                    <td className={`px-4 py-3 whitespace-nowrap text-sm font-medium ${
                      trade.pnl >= 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      ${trade.pnl.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-gray-500 text-center py-4">No trades in this backtest</p>
        )}
        
        {!showAllTrades && trades && trades.length > 10 && (
          <p className="text-sm text-gray-500 mt-2 text-center">
            Showing 10 of {trades.length} trades
          </p>
        )}
      </div>
    </div>
  );
}

export default BacktestResults;
