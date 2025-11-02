import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { TrendingUp } from 'lucide-react';

export const EquityChart = ({ data }) => {
  // Handle both array and object responses
  const dataArray = Array.isArray(data) ? data : (data?.data || []);
  
  if (!dataArray || dataArray.length === 0) {
    return (
      <div className="bg-white rounded-2xl shadow-apple p-6">
        <h3 className="text-xl font-semibold text-apple-gray-800 mb-4">Equity Curve</h3>
        <div className="flex items-center justify-center h-64 text-apple-gray-400">
          <div className="text-center">
            <TrendingUp className="w-12 h-12 mx-auto mb-2 opacity-50" />
            <p>No trading data yet</p>
          </div>
        </div>
      </div>
    );
  }

  const chartData = dataArray.map(point => ({
    time: new Date(point.timestamp).toLocaleTimeString(),
    equity: point.equity,
    pnl: point.total_pnl,
  }));

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white px-4 py-2 rounded-lg shadow-apple border border-gray-200">
          <p className="text-sm text-apple-gray-600">{payload[0].payload.time}</p>
          <p className="text-lg font-semibold text-green-600">
            ${payload[0].value.toFixed(2)}
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-white rounded-2xl shadow-apple p-6">
      <h3 className="text-xl font-semibold text-apple-gray-800 mb-4">Equity Curve</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis 
            dataKey="time" 
            stroke="#9ca3af"
            style={{ fontSize: '12px' }}
          />
          <YAxis 
            stroke="#9ca3af"
            style={{ fontSize: '12px' }}
            tickFormatter={(value) => `$${value.toFixed(0)}`}
          />
          <Tooltip content={<CustomTooltip />} />
          <Line 
            type="monotone" 
            dataKey="equity" 
            stroke="#10b981" 
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 6 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};
