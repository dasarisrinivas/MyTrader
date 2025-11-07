# Trading System Restart & P&L Tracking

## What Happens When You Click "Start Trading"

### On Startup, the system:

1. **Connects to Interactive Brokers**
   - Establishes connection to IB Gateway/TWS
   - Requests delayed market data (15-min delay, free)

2. **Cancels Pending Orders**
   - Automatically cancels ALL unfilled orders for the symbol
   - This prevents "too many orders" errors
   - Only affects pending orders (not filled positions)

3. **Reconciles Existing Positions**
   - Checks IBKR for any open positions (from filled orders)
   - Loads current position (long/short/flat)
   - Tracks average entry price and unrealized P&L from IBKR

4. **Starts Fresh**
   - Begins generating new signals
   - Places new orders based on strategy
   - All new orders tracked in SQLite database

### What Happens to Existing Data

**✅ PRESERVED:**
- ✅ Filled orders (already in database)
- ✅ Open positions (retrieved from IBKR)
- ✅ Historical P&L (calculated from closed trades)
- ✅ Position history and average entry prices

**❌ CANCELLED:**
- ❌ Pending orders (PreSubmitted, Submitted)
- ❌ Stop-loss orders (unfilled)
- ❌ Take-profit orders (unfilled)

### Why Orders Are Cancelled

Interactive Brokers has limits:
- Max 15 orders per side (buy/sell)
- Prevents accumulation of stale orders
- Ensures clean state on each restart

---

## P&L Tracking System

### How P&L is Calculated

#### **Realized P&L** (Closed Positions)
When you complete a round trip (BUY → SELL or SELL → BUY):
```
Example:
BUY 1 @ 6802.75
SELL 1 @ 6807.25
Realized P&L = (6807.25 - 6802.75) × 1 contract × $50/point = $225
```

The system tracks:
- Entry price (average for multiple contracts)
- Exit price (when closing)
- Contract multiplier: ES futures = $50 per point
- Commission (from IBKR reports)

#### **Unrealized P&L** (Open Positions)
For positions still open:
```
Current Position: LONG 1 @ 6807.50
Current Price: 6810.00
Unrealized P&L = (6810.00 - 6807.50) × 1 × $50 = $125
```

Updates in real-time as market price changes.

### Current P&L Status

Based on your orders:

| Order | Action | Price | Result |
|-------|--------|-------|--------|
| 4105 | BUY | 6802.75 | Opened position +1 |
| 4125 | BUY | 6802.75 | Added to position +2 |
| 4227 | SELL | 6807.25 | **Closed 1: +$225** ✅ |
| 4231 | SELL | 6807.25 | **Closed 1: +$225** ✅ |
| 4235 | BUY | 6807.50 | Opened position +1 |

**Summary:**
- Total Realized P&L: **$450** (2 winning trades)
- Current Position: **LONG 1 contract @ 6807.50**
- Win Rate: **100%** (2 wins, 0 losses)
- Unrealized P&L: Depends on current market price

---

## Dashboard Features

### Order Book Section Shows:
- 📊 **Order Placement**: Exact timestamp
- 💰 **Fill Price**: Actual execution price
- 📈 **P&L**: Calculated for closed trades
- 📍 **Position Tracking**: Running position after each order
- ⚡ **Status**: Placed → PreSubmitted → Filled
- 📋 **Timeline**: Full event history

### P&L Summary Endpoint
`GET /api/pnl/summary`

Returns:
```json
{
  "total_realized_pnl": 450.0,
  "unrealized_pnl": 0.0,
  "total_pnl": 450.0,
  "net_pnl": 450.0,
  "total_trades": 5,
  "winning_trades": 2,
  "losing_trades": 0,
  "win_rate": 40.0,
  "current_position": 1,
  "avg_entry_price": 6807.5
}
```

---

## Why Commission Might Show $0

**Reasons:**
1. **Paper Trading Account**: No real commissions charged
2. **Free Tier**: Some accounts have commission-free contracts
3. **IB Reporting Delay**: Commissions reported after settlement
4. **Data Not Available**: Commission report not received from IB API

**Note**: Even if commission shows $0, the P&L calculation is correct based on entry/exit prices.

---

## Best Practices

### Before Restarting Trading System:

1. **Check Open Positions**: Know what positions you have
2. **Note Pending Orders**: They will be cancelled
3. **Record P&L**: Take snapshot of current profits
4. **Review Database**: Orders are preserved in SQLite

### After Restart:

1. **Verify Connection**: Check IB Gateway is connected
2. **Confirm Position**: System should show correct position from IBKR
3. **Monitor New Orders**: Watch for signals and order placement
4. **Check P&L**: Historical P&L should match pre-restart

---

## Troubleshooting

**Q: My P&L shows $0 for all orders**
- Old orders placed before P&L tracking was added
- System now calculates P&L for all new orders

**Q: Position doesn't match after restart**
- System reconciles with IBKR on startup
- May take a few seconds to sync
- Check IBKR TWS/Gateway for actual position

**Q: Stop loss/Take profit not showing**
- Bracket orders cancelled on restart
- New orders will have SL/TP tracked
- Old orders may not have this data

**Q: Orders missing from dashboard**
- Check database path (should be project_root/data/orders.db)
- Backend must be using same database as trading system
- Restart dashboard to reload data

---

## Technical Details

### Database Tables:
1. **orders**: Main order records
2. **order_events**: Status updates, state changes
3. **executions**: Fill details, commission, P&L

### P&L Calculation:
- ES futures: $50 per point per contract
- Position tracking: Maintains running position and avg entry
- Round trip detection: Matches BUY/SELL pairs automatically

### APIs:
- `GET /api/orders/detailed` - All orders with P&L
- `GET /api/pnl/summary` - Overall performance summary
