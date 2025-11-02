import { TrendingUp, TrendingDown, DollarSign, Target, Activity } from 'lucide-react';

export const PerformanceMetrics = ({ performance }) => {
  if (!performance) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
        {[...Array(5)].map((_, i) => (
          <div key={i} className="bg-white rounded-2xl shadow-apple p-6 animate-pulse">
            <div className="h-4 bg-gray-200 rounded w-3/4 mb-4"></div>
            <div className="h-8 bg-gray-200 rounded w-1/2"></div>
          </div>
        ))}
      </div>
    );
  }

  // Use total_pnl for P&L display, or calculate from total_return if available
  const totalPnL = performance.total_pnl !== undefined ? performance.total_pnl : 0;
  const totalReturn = performance.total_return !== undefined ? performance.total_return : 0;
  const winRate = performance.win_rate !== undefined ? performance.win_rate : 0;
  const sharpeRatio = performance.sharpe_ratio !== undefined ? performance.sharpe_ratio : 0;
  const totalTrades = performance.total_trades !== undefined ? performance.total_trades : 0;
  const maxDrawdown = performance.max_drawdown !== undefined ? performance.max_drawdown : 0;

  const metrics = [
    {
      label: 'Total P&L',
      value: `$${totalPnL.toFixed(2)}`,
      icon: DollarSign,
      color: totalPnL >= 0 ? 'text-green-500' : 'text-red-500',
      bgColor: totalPnL >= 0 ? 'bg-green-50' : 'bg-red-50',
    },
    {
      label: 'Total Return',
      value: `${totalReturn.toFixed(2)}%`,
      icon: totalReturn >= 0 ? TrendingUp : TrendingDown,
      color: totalReturn >= 0 ? 'text-green-500' : 'text-red-500',
      bgColor: totalReturn >= 0 ? 'bg-green-50' : 'bg-red-50',
    },
    {
      label: 'Win Rate',
      value: `${(winRate * 100).toFixed(1)}%`,
      icon: Target,
      color: winRate >= 0.5 ? 'text-green-500' : winRate >= 0.4 ? 'text-yellow-500' : 'text-red-500',
      bgColor: winRate >= 0.5 ? 'bg-green-50' : winRate >= 0.4 ? 'bg-yellow-50' : 'bg-red-50',
    },
    {
      label: 'Sharpe Ratio',
      value: sharpeRatio.toFixed(2),
      icon: Activity,
      color: sharpeRatio >= 1 ? 'text-green-500' : sharpeRatio >= 0 ? 'text-blue-500' : 'text-red-500',
      bgColor: sharpeRatio >= 1 ? 'bg-green-50' : sharpeRatio >= 0 ? 'bg-blue-50' : 'bg-red-50',
    },
    {
      label: 'Total Trades',
      value: totalTrades,
      icon: Activity,
      color: 'text-blue-500',
      bgColor: 'bg-blue-50',
    },
    {
      label: 'Max Drawdown',
      value: `${(maxDrawdown * 100).toFixed(2)}%`,
      icon: TrendingDown,
      color: Math.abs(maxDrawdown) < 0.05 ? 'text-yellow-500' : 'text-red-500',
      bgColor: Math.abs(maxDrawdown) < 0.05 ? 'bg-yellow-50' : 'bg-red-50',
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
      {metrics.map((metric, index) => {
        const Icon = metric.icon;
        return (
          <div
            key={index}
            className="bg-white rounded-2xl shadow-apple p-6 hover:shadow-apple-lg transition-shadow duration-200"
          >
            <div className="flex items-center justify-between mb-3">
              <span className="text-apple-gray-600 text-sm font-medium">{metric.label}</span>
              <div className={`${metric.bgColor} p-2 rounded-lg`}>
                <Icon className={`w-5 h-5 ${metric.color}`} />
              </div>
            </div>
            <div className={`text-3xl font-semibold ${metric.color}`}>
              {metric.value}
            </div>
          </div>
        );
      })}
    </div>
  );
};
