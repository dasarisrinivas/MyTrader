# Critical Trading Bot Fixes - November 19, 2025

## Root Cause Analysis

### What Happened:
The bot accumulated **-16 short contracts** when the limit was 5 contracts due to:

1. **Bracket Orders Creating Opposite Positions**
   - Each BUY entry created 2 SELL orders (stop-loss + take-profit)
   - Stop-losses were triggering immediately due to poor entry pricing
   - Bot kept placing new BUY orders while stop-loss SELLs kept filling

2. **Stop-Loss Too Tight**
   - Original: 5 points ($250 per contract) 
   - Got filled at 6654.00 with stop at 6654.25
   - No room for normal market volatility

3. **No Position Direction Check**
   - Bot would add to existing positions in same direction
   - No check to prevent pyramiding without confirmation

### Execution Timeline:
```
13:04:41 - BUY signal, entry=6659.25, stop=6654.25
13:04:42 - Order 411 FILLED at 6654.00 (below stop!) 
13:04:42 - Orders 412 (SELL TP), 413 (SELL SL) placed
13:05:31 - Order 413 FILLED (stop-loss hit)
... Pattern repeated ...
13:45:07 - Order 2123: BUY 16 contracts (trying to close -16 short)
Result: -16 contract short position
```

## Fixes Implemented

### 1. ✅ Position Direction Check (main.py)
```python
# Don't add to existing position in same direction
if current_qty != 0:
    signal_direction = 1 if signal.action == "BUY" else -1
    position_direction = 1 if current_qty > 0 else -1
    if signal_direction == position_direction:
        logger.warning("Already have position, skipping")
        continue
```

### 2. ✅ Professional Stop-Loss Width (main.py)
```python
# Minimum 10 points ($500) for ES futures
min_stop_points = 10.0  # Prevents stop-hunting
if stop_offset < min_stop_points:
    stop_offset = min_stop_points
```

### 3. ✅ Better Entry Pricing (ib_executor.py)
```python
# Use 2-tick buffer (0.50 points) instead of 1-tick
tick_buffer = 0.50  # Better fill probability
limit_price = current_price + tick_buffer if "BUY" else current_price - tick_buffer
```

### 4. ✅ Wider Stop-Limit Buffer (ib_executor.py)
```python
# Allow 1 point (4 ticks = $50) slippage on stop-loss
offset_ticks = 4  # Was 2, now 4
```

### 5. ✅ Config Updated (config.yaml)
```yaml
stop_loss_ticks: 40.0    # 10 points = $500 per contract
take_profit_ticks: 80.0  # 20 points = $1000 per contract (2:1 R/R)
```

## Risk Parameters (Professional Standards)

### ES Futures Professional Minimums:
- **Stop-Loss**: 10-15 points ($500-$750 per contract)
- **Take-Profit**: 20-30 points ($1000-$1500 per contract)
- **Risk/Reward**: Minimum 2:1 ratio
- **Max Position**: 1-3 contracts for $100K account
- **Daily Loss Limit**: $1500 (3 stop-losses)

### Current Configuration:
- ✅ Stop-Loss: 10 points ($500)
- ✅ Take-Profit: 20 points ($1000)
- ✅ R/R Ratio: 2:1
- ✅ Max Position: 5 contracts
- ✅ Contracts/Order: 1
- ✅ Daily Loss: $1500

## Before Restart: Close Current Position

**Current State**: -16 short contracts at ES

**Action Required**:
```bash
# Option 1: Close in IB Gateway manually
# Option 2: Use cancel script
python bin/cancel_all_orders.py

# Option 3: Let bot close on restart (will try to BUY 16)
```

## Expected Bot Behavior After Fix

### ✅ Correct Flow:
1. Generate BUY signal with 0.65 confidence
2. Check: No existing long position ✅
3. Place BUY 1 contract with:
   - Entry: 6650.00
   - Stop: 6640.00 (10 points below)
   - Target: 6670.00 (20 points above)
4. Wait for fill or manage position
5. Skip new BUY signals while holding long
6. Accept SELL signals to close or reverse

### ❌ Will NOT:
- Add to existing long position
- Use stops < 10 points
- Get filled at or below stop-loss price
- Exceed 5 contract limit
- Place orders every 6 seconds

## Monitoring Checklist

After restart, verify:
- [ ] Only 1 contract per entry
- [ ] Stop-loss ≥ 10 points away
- [ ] No position pyramiding
- [ ] No rapid order spam
- [ ] Position closes at stop or target
- [ ] Max 5 contracts total

## Recovery Steps

1. **Close Current -16 Position**
   ```bash
   # In IB Gateway: Right-click ES position → Close Position
   ```

2. **Restart Bot**
   ```bash
   python main.py --mode live --symbol ES
   ```

3. **Monitor First Trade**
   - Watch entry price vs stop-loss
   - Verify 10+ point stop distance
   - Confirm bracket orders placed correctly

4. **Test Stop-Loss**
   - If market moves against position
   - Stop should trigger at calculated price
   - Should not re-enter immediately

## Key Lessons

1. **ES futures need wide stops** - 5-10 points minimum for volatility
2. **Bracket orders create opposite positions** - Be careful with stop-loss fills
3. **Position tracking is critical** - Always check before adding
4. **Limit orders need buffers** - 2-tick buffer prevents missing fills
5. **Professional risk management** - $500 stops, 2:1 R/R, strict limits
