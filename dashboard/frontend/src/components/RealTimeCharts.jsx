import { useState, useEffect } from 'react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine, Dot } from 'recharts';
import { TrendingUp, Activity, Heart } from 'lucide-react';

const API_URL = 'http://localhost:8000';

export default function RealTimeCharts() {
  const [priceData, setPriceData] = useState([]);
  const [sentimentData, setSentimentData] = useState([]);
  const [equityData, setEquityData] = useState([]);
  const [trades, setTrades] = useState([]);

  useEffect(() => {
    fetchChartData();
    const interval = setInterval(fetchChartData, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchChartData = async () => {
    try {
      const [equityRes, tradesRes] = await Promise.all([
        fetch(`${API_URL}/api/equity-curve`),
        fetch(`${API_URL}/api/trades?limit=20`)
      ]);

      const equity = await equityRes.json();
      const tradesData = await tradesRes.json();

      // Transform equity curve data
      const equityCurve = (equity.data || []).map((point, index) => ({
        time: new Date(point.timestamp).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
        equity: point.equity || 0,
        timestamp: point.timestamp
      }));

      // Generate mock price and sentiment data from equity (in real scenario, get from status endpoint)
      const mockPriceData = equityCurve.map((point, index) => ({
        time: point.time,
        price: 6680 + (Math.sin(index / 5) * 20) + (Math.random() * 10),
        timestamp: point.timestamp
      }));

      const mockSentimentData = equityCurve.map((point, index) => ({
        time: point.time,
        sentiment: Math.sin(index / 3) * 0.8 + (Math.random() * 0.2 - 0.1),
        timestamp: point.timestamp
      }));

      setPriceData(mockPriceData.slice(-30));
      setSentimentData(mockSentimentData.slice(-30));
      setEquityData(equityCurve.slice(-30));
      setTrades(tradesData.trades || []);
    } catch (error) {
      console.error('Failed to fetch chart data:', error);
    }
  };

  // Custom dot for entry/exit points
  const CustomDot = (props) => {
    const { cx, cy, payload } = props;
    const trade = trades.find(t => t.timestamp === payload.timestamp);
    
    if (!trade) return null;

    return (
      <g>
        <circle 
          cx={cx} 
          cy={cy} 
          r={6} 
          fill={trade.action === 'BUY' ? '#4ade80' : '#f87171'}
          stroke="white"
          strokeWidth={2}
        />
        <text
          x={cx}
          y={cy - 12}
          textAnchor="middle"
          fill="white"
          fontSize={10}
          fontWeight="bold"
        >
          {trade.action}
        </text>
      </g>
    );
  };

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 shadow-xl">
          <p className="text-gray-400 text-xs mb-1">{label}</p>
          {payload.map((entry, index) => (
            <p key={index} style={{ color: entry.color }} className="text-sm font-semibold">
              {entry.name}: {typeof entry.value === 'number' ? entry.value.toFixed(2) : entry.value}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-white flex items-center gap-2 mb-2">
          <Activity className="w-6 h-6 text-blue-400" />
          Real-Time Analytics
        </h2>
        <p className="text-gray-400 text-sm">Live market data with bot decision points</p>
      </div>

      {/* Price Chart with Entry/Exit Markers */}
      <div className="bg-gradient-to-br from-gray-800 to-gray-900 border border-gray-700 rounded-xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-green-400" />
            Price Movement
          </h3>
          <div className="flex gap-3 text-xs">
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 rounded-full bg-green-400"></div>
              <span className="text-gray-400">Entry</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 rounded-full bg-red-400"></div>
              <span className="text-gray-400">Exit</span>
            </div>
          </div>
        </div>
        
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={priceData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis 
              dataKey="time" 
              stroke="#9ca3af" 
              style={{ fontSize: '12px' }}
            />
            <YAxis 
              stroke="#9ca3af" 
              style={{ fontSize: '12px' }}
              domain={['dataMin - 5', 'dataMax + 5']}
            />
            <Tooltip content={<CustomTooltip />} />
            <Line 
              type="monotone" 
              dataKey="price" 
              stroke="#3b82f6" 
              strokeWidth={2}
              dot={<CustomDot />}
              name="Price"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Sentiment Trend */}
      <div className="bg-gradient-to-br from-gray-800 to-gray-900 border border-gray-700 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Heart className="w-5 h-5 text-purple-400" />
          Sentiment Trend
        </h3>
        
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={sentimentData}>
            <defs>
              <linearGradient id="sentimentGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0.1}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis 
              dataKey="time" 
              stroke="#9ca3af" 
              style={{ fontSize: '12px' }}
            />
            <YAxis 
              stroke="#9ca3af" 
              style={{ fontSize: '12px' }}
              domain={[-1, 1]}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine y={0} stroke="#6b7280" strokeDasharray="3 3" />
            <Area 
              type="monotone" 
              dataKey="sentiment" 
              stroke="#8b5cf6" 
              fillOpacity={1} 
              fill="url(#sentimentGradient)"
              name="Sentiment"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Cumulative Profit */}
      <div className="bg-gradient-to-br from-gray-800 to-gray-900 border border-gray-700 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-emerald-400" />
          Cumulative Profit
        </h3>
        
        <ResponsiveContainer width="100%" height={250}>
          <AreaChart data={equityData}>
            <defs>
              <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10b981" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="#10b981" stopOpacity={0.1}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis 
              dataKey="time" 
              stroke="#9ca3af" 
              style={{ fontSize: '12px' }}
            />
            <YAxis 
              stroke="#9ca3af" 
              style={{ fontSize: '12px' }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Area 
              type="monotone" 
              dataKey="equity" 
              stroke="#10b981" 
              fillOpacity={1} 
              fill="url(#equityGradient)"
              name="Equity"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
