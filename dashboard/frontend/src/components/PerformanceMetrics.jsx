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

  const metrics = [
    {
      label: 'Total P&L',
      value: `$${performance.total_return?.toFixed(2) || '0.00'}`,
      icon: DollarSign,
      color: performance.total_return >= 0 ? 'text-green-500' : 'text-red-500',
      bgColor: performance.total_return >= 0 ? 'bg-green-50' : 'bg-red-50',
    },
    {
      label: 'Win Rate',
      value: `${(performance.win_rate * 100 || 0).toFixed(1)}%`,
      icon: Target,
      color: performance.win_rate >= 0.5 ? 'text-green-500' : 'text-yellow-500',
      bgColor: performance.win_rate >= 0.5 ? 'bg-green-50' : 'bg-yellow-50',
    },
    {
      label: 'Sharpe Ratio',
      value: performance.sharpe_ratio?.toFixed(2) || '0.00',
      icon: Activity,
      color: performance.sharpe_ratio >= 1 ? 'text-green-500' : 'text-blue-500',
      bgColor: performance.sharpe_ratio >= 1 ? 'bg-green-50' : 'bg-blue-50',
    },
    {
      label: 'Total Trades',
      value: performance.total_trades || 0,
      icon: performance.total_return >= 0 ? TrendingUp : TrendingDown,
      color: 'text-blue-500',
      bgColor: 'bg-blue-50',
    },
    {
      label: 'Max Drawdown',
      value: `${(performance.max_drawdown * 100 || 0).toFixed(2)}%`,
      icon: TrendingDown,
      color: 'text-red-500',
      bgColor: 'bg-red-50',
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
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
