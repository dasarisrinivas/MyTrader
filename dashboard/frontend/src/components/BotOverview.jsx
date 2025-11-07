import { useState, useEffect } from 'react';
import { Activity, TrendingUp, TrendingDown, DollarSign, BarChart3, Zap } from 'lucide-react';

const API_URL = 'http://localhost:8000';

export default function BotOverview() {
  const [stats, setStats] = useState({
    activeOrders: 0,
    totalPnL: 0,
    todayTrades: 0,
    openPosition: 0,
    currentSymbol: 'ES',
    winRate: 0,
    todayReturn: 0,
    accountValue: 0,
    initialCapital: 250000
  });

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 2000);
    return () => clearInterval(interval);
  }, []);

  const fetchStats = async () => {
    try {
      const [statusRes, pnlRes, tradesRes, balanceRes] = await Promise.all([
        fetch(`${API_URL}/api/trading/status`),
        fetch(`${API_URL}/api/pnl/summary`),
        fetch(`${API_URL}/api/trades?limit=100`),
        fetch(`${API_URL}/api/account/balance`).catch(() => ({ json: () => ({ net_liquidation: 0 }) }))
      ]);

      const status = await statusRes.json();
      const pnl = await pnlRes.json();
      const trades = await tradesRes.json();
      const balance = await balanceRes.json();

      // Filter today's trades
      const today = new Date().toDateString();
      const todayTrades = (trades.trades || []).filter(t => 
        new Date(t.timestamp).toDateString() === today
      );

      // Use net liquidation value from IB, or fall back to initial capital + PnL
      const accountValue = balance.net_liquidation > 0 
        ? balance.net_liquidation 
        : 100000 + (pnl.total_pnl || 0);

      setStats({
        activeOrders: status.active_orders || 0,
        totalPnL: pnl.total_pnl || 0,
        todayTrades: todayTrades.length,
        openPosition: status.current_position || 0,
        currentSymbol: 'ES',
        winRate: pnl.win_rate || 0,
        todayReturn: pnl.today_return || 0,
        accountValue: accountValue,
        initialCapital: 100000
      });
    } catch (error) {
      console.error('Failed to fetch stats:', error);
    }
  };

  const StatCard = ({ icon: Icon, label, value, subValue, trend, colorClass = 'text-blue-600' }) => (
    <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-6 border border-gray-700 hover:border-gray-600 transition-all">
      <div className="flex items-start justify-between mb-3">
        <div className={`p-3 rounded-lg bg-opacity-10 ${colorClass.replace('text', 'bg')}`}>
          <Icon className={`w-6 h-6 ${colorClass}`} />
        </div>
        {trend !== undefined && (
          <div className={`flex items-center gap-1 ${trend >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {trend >= 0 ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
            <span className="text-sm font-semibold">{Math.abs(trend).toFixed(1)}%</span>
          </div>
        )}
      </div>
      <div className="text-sm text-gray-400 mb-1">{label}</div>
      <div className="text-2xl font-bold text-white mb-1">{value}</div>
      {subValue && <div className="text-xs text-gray-500">{subValue}</div>}
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-white mb-2">Bot Analytics Dashboard</h2>
          <p className="text-gray-400">Real-time trading intelligence from your AI bot</p>
        </div>
        <div className="flex items-center gap-2 px-4 py-2 bg-green-900/30 border border-green-700 rounded-lg">
          <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
          <span className="text-green-400 font-semibold">Bot Active</span>
        </div>
      </div>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={DollarSign}
          label="Account Value"
          value={`$${stats.accountValue.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
          subValue={`Starting: $${stats.initialCapital.toLocaleString()}`}
          trend={((stats.accountValue - stats.initialCapital) / stats.initialCapital * 100)}
          colorClass="text-cyan-400"
        />
        
        <StatCard
          icon={DollarSign}
          label="Total P&L"
          value={`$${stats.totalPnL.toFixed(2)}`}
          subValue="All time realized"
          trend={stats.todayReturn}
          colorClass={stats.totalPnL >= 0 ? 'text-green-400' : 'text-red-400'}
        />
        
        <StatCard
          icon={BarChart3}
          label="Today's Trades"
          value={stats.todayTrades}
          subValue={`${stats.winRate.toFixed(0)}% win rate`}
          colorClass="text-blue-400"
        />
        
        <StatCard
          icon={Activity}
          label="Open Position"
          value={`${stats.openPosition} contracts`}
          subValue={stats.currentSymbol}
          colorClass="text-purple-400"
        />
        
        <StatCard
          icon={Zap}
          label="Active Orders"
          value={stats.activeOrders}
          subValue="In execution"
          colorClass="text-yellow-400"
        />
      </div>

      {/* Symbol Information */}
      <div className="bg-gradient-to-r from-blue-900/20 to-purple-900/20 border border-blue-800 rounded-xl p-6">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm text-gray-400 mb-1">Currently Trading</div>
            <div className="text-3xl font-bold text-white">{stats.currentSymbol} Futures</div>
            <div className="text-sm text-gray-400 mt-1">E-mini S&P 500 • CME</div>
          </div>
          <div className="text-right">
            <div className="text-sm text-gray-400 mb-1">Contract Multiplier</div>
            <div className="text-2xl font-bold text-white">$50</div>
            <div className="text-xs text-gray-500 mt-1">per point</div>
          </div>
        </div>
      </div>
    </div>
  );
}
