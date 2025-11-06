import { useState, useEffect } from 'react';
import { 
  Clock, 
  CheckCircle2, 
  AlertCircle, 
  TrendingUp, 
  TrendingDown, 
  Target, 
  Shield,
  Activity,
  ChevronDown,
  ChevronRight,
  Zap,
  Filter
} from 'lucide-react';
import OrderPriceChart from './OrderPriceChart';

const API_URL = 'http://localhost:8000';

  // Convert UTC timestamp to CST (timestamps in DB are UTC without 'Z' suffix)
  const formatTimestamp = (timestamp) => {
    // Add 'Z' to indicate UTC if not present
    const utcTimestamp = timestamp.endsWith('Z') ? timestamp : timestamp + 'Z';
    const date = new Date(utcTimestamp);
    return date.toLocaleString('en-US', {
      timeZone: 'America/Chicago',
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false
    });
  };

export default function OrderBook() {
  const [orders, setOrders] = useState([]);
  const [loading, setLoading] = useState(true);
  const [expandedOrders, setExpandedOrders] = useState(new Set());
  const [currentPrice, setCurrentPrice] = useState(null);
  const [pnlSummary, setPnlSummary] = useState(null);
  const [showAllOrders, setShowAllOrders] = useState(false);
  const [accountBalance, setAccountBalance] = useState(null);

  useEffect(() => {
    fetchOrders();
    fetchCurrentPrice();
    fetchPnlSummary();
    fetchAccountBalance();
    const interval = setInterval(() => {
      fetchOrders();
      fetchCurrentPrice();
      fetchPnlSummary();
      fetchAccountBalance();
    }, 5000); // Refresh every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchOrders = async () => {
    try {
      const response = await fetch(`${API_URL}/api/orders/detailed`);
      const data = await response.json();
      setOrders(data.orders || []);
      setLoading(false);
    } catch (error) {
      console.error('Failed to fetch orders:', error);
      setLoading(false);
    }
  };

  const fetchCurrentPrice = async () => {
    try {
      const response = await fetch(`${API_URL}/api/trading/status`);
      const data = await response.json();
      if (data.current_price) {
        setCurrentPrice(data.current_price);
      }
    } catch (error) {
      console.error('Failed to fetch current price:', error);
    }
  };

  const fetchPnlSummary = async () => {
    try {
      const response = await fetch(`${API_URL}/api/pnl/summary`);
      const data = await response.json();
      setPnlSummary(data);
    } catch (error) {
      console.error('Failed to fetch P&L summary:', error);
    }
  };

  const fetchAccountBalance = async () => {
    try {
      const response = await fetch(`${API_URL}/api/account/balance`);
      const data = await response.json();
      if (data.success) {
        setAccountBalance(data);
      }
    } catch (error) {
      console.error('Failed to fetch account balance:', error);
    }
  };

  const syncOrdersFromIB = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_URL}/api/orders/sync`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      const data = await response.json();
      
      if (data.success) {
        console.log(`✅ Synced ${data.synced} new orders, updated ${data.updated} orders from IB`);
        // Refresh orders after sync
        await fetchOrders();
      } else {
        console.error('Sync failed:', data.message);
        alert(`Failed to sync orders: ${data.message}`);
      }
    } catch (error) {
      console.error('Failed to sync orders from IB:', error);
      alert('Failed to sync orders from IB. Make sure IB Gateway/TWS is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    setLoading(true);
    try {
      // Try to sync from IB first
      const syncResponse = await fetch(`${API_URL}/api/orders/sync`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      const syncData = await syncResponse.json();
      
      if (syncData.success) {
        const deletedMsg = syncData.deleted > 0 ? `, deleted ${syncData.deleted} stale orders` : '';
        console.log(`✅ Synced ${syncData.synced} new, updated ${syncData.updated}${deletedMsg} from IB`);
      } else {
        console.warn('IB sync failed, refreshing from database:', syncData.message);
      }
      
      // Always refresh from database after sync attempt
      await fetchOrders();
      await fetchPnlSummary();
      await fetchCurrentPrice();
    } catch (error) {
      console.error('Refresh error:', error);
      // Still try to refresh from database
      await fetchOrders();
      await fetchPnlSummary();
      await fetchCurrentPrice();
    } finally {
      setLoading(false);
    }
  };

  const toggleExpand = (orderId) => {
    setExpandedOrders(prev => {
      const newSet = new Set(prev);
      if (newSet.has(orderId)) {
        newSet.delete(orderId);
      } else {
        newSet.add(orderId);
      }
      return newSet;
    });
  };

  const getStatusColor = (status) => {
    const statusLower = status?.toLowerCase() || '';
    if (statusLower.includes('filled')) return 'text-green-600 bg-green-50 border-green-200';
    if (statusLower.includes('submitted') || statusLower.includes('presubmitted')) 
      return 'text-blue-600 bg-blue-50 border-blue-200';
    if (statusLower.includes('cancelled') || statusLower.includes('inactive')) 
      return 'text-red-600 bg-red-50 border-red-200';
    if (statusLower.includes('placing')) return 'text-yellow-600 bg-yellow-50 border-yellow-200';
    return 'text-gray-600 bg-gray-50 border-gray-200';
  };

  const getStatusIcon = (status) => {
    const statusLower = status?.toLowerCase() || '';
    if (statusLower.includes('filled')) return <CheckCircle2 className="w-4 h-4" />;
    if (statusLower.includes('cancelled') || statusLower.includes('inactive')) 
      return <AlertCircle className="w-4 h-4" />;
    if (statusLower.includes('placing')) return <Clock className="w-4 h-4 animate-spin" />;
    return <Activity className="w-4 h-4" />;
  };

  const calculateRiskReward = (order) => {
    if (!order.entry_price || !order.stop_loss || !order.take_profit) return null;
    
    const risk = Math.abs(order.entry_price - order.stop_loss);
    const reward = Math.abs(order.take_profit - order.entry_price);
    
    if (risk === 0) return null;
    return (reward / risk).toFixed(2);
  };

  const calculatePotentialPnL = (order) => {
    if (!order.entry_price || !order.avg_fill_price) return null;
    
    const multiplier = order.action === 'BUY' ? 1 : -1;
    const pnl = (order.avg_fill_price - order.entry_price) * multiplier * (order.filled_quantity || 0) * 50; // ES multiplier
    return pnl;
  };

  const formatTime = (timestamp) => {
    // Add 'Z' to indicate UTC if not present
    const utcTimestamp = timestamp.endsWith('Z') ? timestamp : timestamp + 'Z';
    const date = new Date(utcTimestamp);
    return date.toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit', 
      second: '2-digit',
      timeZone: 'America/Chicago'
    });
  };

  const getExecutionTime = (order) => {
    if (!order.execution_time || !order.timestamp) return null;
    
    const start = new Date(order.timestamp);
    const end = new Date(order.execution_time);
    const diff = (end - start) / 1000; // seconds
    
    return diff.toFixed(1);
  };

  // Filter orders - show only today (CST) by default
  const filteredOrders = showAllOrders 
    ? orders 
    : orders.filter(order => {
        // Parse UTC timestamp and convert to CST date
        const utcTimestamp = order.timestamp.endsWith('Z') ? order.timestamp : order.timestamp + 'Z';
        const orderDate = new Date(utcTimestamp);
        
        // Get today's date in CST (start of day)
        const todayCST = new Date();
        todayCST.setHours(0, 0, 0, 0);
        
        // Get CST date string for comparison (format: 2025-11-05)
        const orderDateCST = orderDate.toLocaleDateString('en-US', {
          timeZone: 'America/Chicago',
          year: 'numeric',
          month: '2-digit',
          day: '2-digit'
        });
        
        const todayDateCST = new Date().toLocaleDateString('en-US', {
          timeZone: 'America/Chicago',
          year: 'numeric',
          month: '2-digit',
          day: '2-digit'
        });
        
        return orderDateCST === todayDateCST;
      });

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      {/* Header with P&L Summary */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-4">
          <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
            <Activity className="w-6 h-6 text-blue-500" />
            Order Book
            <span className="text-sm font-normal text-gray-500 ml-2">
              ({filteredOrders.length} {showAllOrders ? 'total' : 'today'})
            </span>
          </h2>
          
          {/* Real-time P&L Display */}
          {pnlSummary && (
            <div className="flex items-center gap-4 ml-6">
              {/* Account Balance */}
              {accountBalance && (
                <div className={`px-4 py-2 rounded-lg border ${
                  accountBalance.net_liquidation >= 2000 
                    ? 'bg-gradient-to-r from-emerald-50 to-emerald-100 border-emerald-200' 
                    : 'bg-gradient-to-r from-red-50 to-red-100 border-red-300'
                }`}>
                  <div className={`text-xs font-medium ${
                    accountBalance.net_liquidation >= 2000 
                      ? 'text-emerald-600' 
                      : 'text-red-600'
                  }`}>
                    Account Balance
                    {accountBalance.net_liquidation < 2000 && ' ⚠️'}
                  </div>
                  <div className={`text-xl font-bold ${
                    accountBalance.net_liquidation >= 2000 
                      ? 'text-emerald-700' 
                      : 'text-red-700'
                  }`}>
                    ${accountBalance.net_liquidation?.toFixed(2) || '0.00'}
                  </div>
                  {accountBalance.net_liquidation < 2000 && (
                    <div className="text-xs text-red-600 mt-1">
                      Min $2,000 required for ES futures
                    </div>
                  )}
                </div>
              )}
              
              <div className="px-4 py-2 bg-gradient-to-r from-blue-50 to-blue-100 rounded-lg border border-blue-200">
                <div className="text-xs text-blue-600 font-medium">Realized P&L</div>
                <div className={`text-xl font-bold ${pnlSummary.total_realized_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {pnlSummary.total_realized_pnl >= 0 ? '+' : ''}${pnlSummary.total_realized_pnl?.toFixed(2) || '0.00'}
                </div>
              </div>
              
              {pnlSummary.unrealized_pnl !== 0 && (
                <div className="px-4 py-2 bg-gradient-to-r from-yellow-50 to-yellow-100 rounded-lg border border-yellow-200">
                  <div className="text-xs text-yellow-600 font-medium">Unrealized P&L</div>
                  <div className={`text-xl font-bold ${pnlSummary.unrealized_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {pnlSummary.unrealized_pnl >= 0 ? '+' : ''}${pnlSummary.unrealized_pnl?.toFixed(2) || '0.00'}
                  </div>
                </div>
              )}
              
              <div className="px-4 py-2 bg-gradient-to-r from-indigo-50 to-indigo-100 rounded-lg border border-indigo-200">
                <div className="text-xs text-indigo-600 font-medium">Total P&L</div>
                <div className={`text-xl font-bold ${pnlSummary.total_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {pnlSummary.total_pnl >= 0 ? '+' : ''}${pnlSummary.total_pnl?.toFixed(2) || '0.00'}
                </div>
              </div>
              
              <div className="px-4 py-2 bg-gradient-to-r from-green-50 to-green-100 rounded-lg border border-green-200">
                <div className="text-xs text-green-600 font-medium">Win Rate</div>
                <div className="text-xl font-bold text-green-700">
                  {pnlSummary.win_rate?.toFixed(0) || '0'}%
                </div>
              </div>
              
              {pnlSummary.current_position !== 0 && (
                <div className="px-4 py-2 bg-gradient-to-r from-purple-50 to-purple-100 rounded-lg border border-purple-200">
                  <div className="text-xs text-purple-600 font-medium">Position</div>
                  <div className="text-xl font-bold text-purple-700">
                    {pnlSummary.current_position > 0 ? '+' : ''}{pnlSummary.current_position}
                  </div>
                </div>
              )}
              
              {currentPrice && (
                <div className="px-4 py-2 bg-gradient-to-r from-gray-50 to-gray-100 rounded-lg border border-gray-200">
                  <div className="text-xs text-gray-600 font-medium">Current Price</div>
                  <div className="text-xl font-bold text-gray-800">
                    ${currentPrice.toFixed(2)}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
        
        <div className="flex items-center gap-2">
          <button 
            onClick={() => setShowAllOrders(!showAllOrders)}
            className="px-4 py-2 text-sm bg-gray-50 text-gray-600 rounded-lg hover:bg-gray-100 transition-colors flex items-center gap-2"
          >
            <Filter className="w-4 h-4" />
            {showAllOrders ? 'Show Today Only' : 'Show All History'}
          </button>
          
          <button 
            onClick={handleRefresh}
            disabled={loading}
            className={`px-4 py-2 text-sm rounded-lg transition-colors flex items-center gap-2 ${
              loading 
                ? 'bg-gray-100 text-gray-400 cursor-not-allowed' 
                : 'bg-blue-50 text-blue-600 hover:bg-blue-100'
            }`}
          >
            <Zap className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            {loading ? 'Syncing from IB...' : 'Sync from IB'}
          </button>
        </div>
      </div>

      {filteredOrders.length === 0 ? (
        <div className="flex flex-col items-center justify-center h-64 text-gray-500">
          <Activity className="w-16 h-16 mb-4 opacity-20" />
          <p className="text-lg">No orders {showAllOrders ? '' : 'in last 24 hours'}</p>
          <p className="text-sm mt-2">
            {showAllOrders 
              ? 'Orders will appear here when trading starts' 
              : 'Click "Show All History" to see older orders'}
          </p>
        </div>
      ) : (
        <div className="space-y-3 max-h-[600px] overflow-y-auto">
        {filteredOrders.map((order) => {
          const isExpanded = expandedOrders.has(order.order_id);
          const riskReward = calculateRiskReward(order);
          const pnl = calculatePotentialPnL(order);
          const execTime = getExecutionTime(order);

          return (
            <div 
              key={order.order_id}
              className="border border-gray-200 rounded-lg overflow-hidden hover:shadow-lg transition-shadow"
            >
              {/* Main Order Row */}
              <div 
                className="p-4 cursor-pointer hover:bg-gray-50 transition-colors"
                onClick={() => toggleExpand(order.order_id)}
              >
                <div className="flex items-start justify-between">
                  {/* Left: Order Details */}
                  <div className="flex-1 space-y-2">
                    <div className="flex items-center gap-3">
                      {/* Expand/Collapse Icon */}
                      {isExpanded ? (
                        <ChevronDown className="w-5 h-5 text-gray-400" />
                      ) : (
                        <ChevronRight className="w-5 h-5 text-gray-400" />
                      )}
                      
                      {/* Order ID */}
                      <span className="text-xs font-mono text-gray-500">
                        #{order.order_id}
                      </span>
                      
                      {/* Action Badge */}
                      <div className={`flex items-center gap-1 px-3 py-1 rounded-full ${
                        order.action === 'BUY' 
                          ? 'bg-green-100 text-green-700' 
                          : 'bg-red-100 text-red-700'
                      }`}>
                        {order.action === 'BUY' ? (
                          <TrendingUp className="w-4 h-4" />
                        ) : (
                          <TrendingDown className="w-4 h-4" />
                        )}
                        <span className="font-semibold text-sm">{order.action}</span>
                      </div>

                      {/* Status Badge */}
                      <div className={`flex items-center gap-1 px-3 py-1 rounded-full border ${getStatusColor(order.status)}`}>
                        {getStatusIcon(order.status)}
                        <span className="text-sm font-medium">{order.status}</span>
                      </div>

                      {/* Execution Time Badge */}
                      {execTime && (
                        <div className="flex items-center gap-1 px-2 py-1 bg-purple-50 text-purple-600 rounded text-xs">
                          <Zap className="w-3 h-3" />
                          {execTime}s
                        </div>
                      )}
                    </div>

                    {/* Price Information */}
                    <div className="flex items-center gap-6 ml-8 text-sm">
                      <div>
                        <span className="text-gray-500">Entry:</span>
                        <span className="ml-2 font-semibold text-gray-800">
                          ${order.entry_price?.toFixed(2) || '--'}
                        </span>
                      </div>

                      {order.avg_fill_price && (
                        <div>
                          <span className="text-gray-500">Fill:</span>
                          <span className="ml-2 font-semibold text-gray-800">
                            ${order.avg_fill_price.toFixed(2)}
                          </span>
                        </div>
                      )}

                      <div>
                        <span className="text-gray-500">Qty:</span>
                        <span className="ml-2 font-semibold text-gray-800">
                          {order.filled_quantity || order.quantity || 0}
                        </span>
                      </div>

                      {order.confidence > 0 && (
                        <div>
                          <span className="text-gray-500">Confidence:</span>
                          <span className="ml-2 font-semibold text-blue-600">
                            {(order.confidence * 100).toFixed(0)}%
                          </span>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Right: Timestamp and P&L */}
                  <div className="text-right space-y-1">
                    <div className="text-xs text-gray-500">
                      {formatTimestamp(order.timestamp)}
                    </div>
                    {order.calculated_pnl && (
                      <div className={`text-lg font-bold ${order.calculated_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {order.calculated_pnl >= 0 ? '+' : ''}${order.calculated_pnl.toFixed(2)}
                      </div>
                    )}
                    {!order.calculated_pnl && order.position_after !== undefined && (
                      <div className="text-xs text-gray-400">
                        Pos: {order.position_after > 0 ? '+' : ''}{order.position_after}
                      </div>
                    )}
                  </div>
                </div>

                {/* Stop Loss & Take Profit - Always Visible */}
                {(order.stop_loss || order.take_profit) && (
                  <div className="mt-3 ml-8 flex items-center gap-4">
                    {order.stop_loss && (
                      <div className="flex items-center gap-2 px-3 py-1.5 bg-red-50 border border-red-200 rounded-lg">
                        <Shield className="w-4 h-4 text-red-600" />
                        <div>
                          <div className="text-xs text-red-600 font-medium">Stop Loss</div>
                          <div className="text-sm font-bold text-red-700">
                            ${order.stop_loss.toFixed(2)}
                          </div>
                        </div>
                      </div>
                    )}

                    {order.take_profit && (
                      <div className="flex items-center gap-2 px-3 py-1.5 bg-green-50 border border-green-200 rounded-lg">
                        <Target className="w-4 h-4 text-green-600" />
                        <div>
                          <div className="text-xs text-green-600 font-medium">Take Profit</div>
                          <div className="text-sm font-bold text-green-700">
                            ${order.take_profit.toFixed(2)}
                          </div>
                        </div>
                      </div>
                    )}

                    {riskReward && (
                      <div className="flex items-center gap-2 px-3 py-1.5 bg-blue-50 border border-blue-200 rounded-lg">
                        <div>
                          <div className="text-xs text-blue-600 font-medium">Risk:Reward</div>
                          <div className="text-sm font-bold text-blue-700">
                            1:{riskReward}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Expanded Section: Timeline and Price Chart */}
              {isExpanded && (
                <div className="border-t border-gray-200 bg-gray-50">
                  {/* Price Chart */}
                  {(order.entry_price || order.avg_fill_price) && (
                    <div className="p-4 border-b border-gray-200 bg-white">
                      <h4 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
                        <TrendingUp className="w-4 h-4" />
                        Price Levels
                      </h4>
                      <OrderPriceChart 
                        order={{
                          ...order,
                          entry_price: order.entry_price || order.avg_fill_price
                        }} 
                        currentPrice={currentPrice}
                      />
                    </div>
                  )}
                  
                  {/* Timeline */}
                  {order.updates && order.updates.length > 0 && (
                    <div className="p-4">
                      <h4 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
                        <Clock className="w-4 h-4" />
                        Order Timeline
                      </h4>
                      <div className="space-y-2">
                        {order.updates.map((update, idx) => (
                          <div key={idx} className="flex items-start gap-3 text-sm">
                            <div className="flex-shrink-0 w-16 text-xs text-gray-500 font-mono pt-0.5">
                              {formatTime(update.timestamp)}
                            </div>
                            <div className="flex-shrink-0">
                              <div className={`w-2 h-2 rounded-full mt-1.5 ${
                                update.status === 'Filled' || update.status === 'Executed' 
                                  ? 'bg-green-500' 
                                  : update.status === 'Cancelled' 
                                    ? 'bg-red-500' 
                                    : 'bg-blue-500'
                              }`}></div>
                            </div>
                            <div className="flex-1">
                              <span className={`font-medium ${
                                update.status === 'Filled' || update.status === 'Executed'
                                  ? 'text-green-700'
                                  : update.status === 'Cancelled'
                                    ? 'text-red-700'
                                    : 'text-gray-700'
                              }`}>
                                {update.status}
                              </span>
                              {update.message && update.message !== update.status && (
                                <span className="text-gray-600 ml-2">— {update.message}</span>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>

                      {/* Additional Order Metadata */}
                      {order.atr && (
                        <div className="mt-4 pt-3 border-t border-gray-200">
                          <div className="text-xs text-gray-600">
                            <span className="font-medium">ATR:</span> {order.atr.toFixed(2)}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
        </div>
      )}
    </div>
  );
}
