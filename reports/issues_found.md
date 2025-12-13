# Issues Found - MyTrader Trading Bot

## Severity-1 (Must Fix)

### 1. Trade Submission with Invalid/None SL/TP
**Location**: `mytrader/execution/live_trading_manager.py:_place_exit_order()` (line 1581-1619)

**Issue**: Exit orders are placed with `stop_loss=None, take_profit=None` without validation. If an exit order accidentally becomes an entry (e.g., wrong direction), the position has no protection.

**Evidence**: 
- Line 1600-1601: `stop_loss=None, take_profit=None` hardcoded for exit orders
- Postmortem 2025-12-12: Exit order 14812 filled as BUY when bot was short, creating unprotected long position

**Impact**: Unprotected positions can lead to unlimited losses.

**Fix Required**: 
- Validate exit orders don't accidentally open positions (reduce_only enforcement)
- If exit becomes entry, require valid SL/TP or block trade

---

### 2. Bracket Children Can Be Marketable Immediately (Wrong Side)
**Location**: `mytrader/execution/order_builder.py:validate_bracket_prices()` (line 19-88)

**Issue**: Validation checks SL/TP are on correct side relative to entry, but doesn't verify minimum distance in ticks. If SL/TP are too close to entry, they can fill immediately.

**Evidence**:
- Line 69-81: `_clamp()` function adjusts distances but minimum is `max(tick_size, 1e-6)` which may be too small
- No check for minimum distance in points/ticks (e.g., 4 ticks minimum)

**Impact**: Stop loss can trigger immediately on entry, causing instant loss.

**Fix Required**:
- Enforce minimum distance in ticks (configurable, default 4 ticks = 1 point for ES)
- Reject brackets where SL/TP are within minimum distance of entry

---

### 3. Decision Gating Ignored (AWS WAIT/NO_TRADE)
**Location**: `mytrader/execution/live_trading_manager.py:_process_hybrid_signal()` (line 1486-1523)

**Issue**: When AWS agent returns `WAIT`, the code only blocks if `FF_WAIT_BLOCKING=1` AND `block_on_wait=True`. Otherwise, it applies a confidence penalty but still allows trade.

**Evidence**:
- Line 1487-1503: `wait_should_block` only True if feature flag enabled AND block_on_wait=True
- Line 1512-1519: If WAIT but not blocking, applies penalty but continues
- Postmortem: AWS said WAIT but trade proceeded with reduced confidence

**Impact**: Trades proceed against AWS guardrail advice, leading to losses.

**Fix Required**:
- Default `AWS_WAIT_BLOCKS_TRADE=true` (change default in config)
- Block on WAIT unless explicit override with high confidence threshold
- Log structured event when WAIT is overridden

---

### 4. Duplicate Trade Cycle Runs / Race Conditions
**Location**: `mytrader/execution/live_trading_manager.py:_process_trading_cycle()` (line 814)

**Issue**: No idempotency check to prevent same signal from being processed multiple times if cycle runs faster than expected or if multiple signals arrive.

**Evidence**:
- `_current_cycle_id` is generated at start of cycle but not checked for duplicates
- No lock to prevent concurrent cycle processing
- Active order check (line 1180) may not catch rapid-fire signals

**Impact**: Double-submission of orders, duplicate trades.

**Fix Required**:
- Add cycle idempotency check (skip if same signal+price+timestamp already processed)
- Add async lock to prevent concurrent cycle processing
- Track recent cycle IDs to detect duplicates

---

### 5. Exit Fills Not Linked to Entry (trade_id Missing)
**Location**: `mytrader/execution/live_trading_manager.py:_place_exit_order()` (line 1581)

**Issue**: Exit orders don't preserve `trade_cycle_id` from entry, making it impossible to correlate exit with entry for learning.

**Evidence**:
- Line 1603: Exit order passes `metadata={"trade_cycle_id": self._current_cycle_id}` but this is the NEW cycle ID, not the entry cycle ID
- Postmortem: "No current trade ID for exit logging" - exit couldn't find entry trade_id
- Learning hooks can't match exit to entry without correlation

**Impact**: Learning system can't analyze trade outcomes, can't improve from losses.

**Fix Required**:
- Store entry `trade_cycle_id` in position metadata
- When placing exit, use entry's `trade_cycle_id` not current cycle
- Ensure exit fills resolve back to entry via orderId/permId/execId mapping

---

## Severity-2 (Should Fix)

### 6. No Emergency Stop After Fill If Bracket Fails
**Location**: `mytrader/execution/ib_executor.py:_on_execution()` (line 542)

**Issue**: When an entry order fills, there's no check to verify bracket orders (SL/TP) were successfully placed. If bracket placement fails, position is unprotected.

**Evidence**:
- `_on_execution()` only logs fill and sends Telegram
- No verification that bracket children are active
- No emergency stop placement if bracket missing

**Impact**: Unprotected positions can lead to large losses if bracket fails silently.

**Fix Required**:
- After entry fill, verify bracket orders exist and are active
- If bracket missing, immediately place emergency stop (idempotent + retry safe)
- Log structured event for bracket failure

---

### 7. No Max Loss Per Trade/Day Kill-Switch
**Location**: `mytrader/execution/live_trading_manager.py:_validate_entry_guard()` (line 2402)

**Issue**: Validation checks `max_loss_per_trade` but doesn't check daily loss limit. No emergency stop if daily loss exceeds threshold.

**Evidence**:
- Line 2429: Checks `max_loss_per_trade` per trade
- No check for cumulative daily loss
- No automatic position closure on daily loss limit

**Impact**: Can exceed daily risk limits through multiple trades.

**Fix Required**:
- Add daily loss tracking
- Block new trades if daily loss exceeds limit
- Auto-close positions if daily loss limit hit

---

### 8. Startup Reconciliation Missing
**Location**: `mytrader/execution/live_trading_manager.py:initialize()` (line 200+)

**Issue**: No reconciliation between IB positions/orders and local database on startup. Drift can cause incorrect position tracking.

**Evidence**:
- `ReconcileManager` exists but may not be called on startup
- No verification that local DB matches IB state

**Impact**: Position tracking errors, duplicate orders, incorrect PnL.

**Fix Required**:
- Call reconciliation on startup before trading
- Verify positions/orders match between IB and DB
- Block trading until reconciliation complete

---

### 9. OpenSearch/RAG Hit Every Minute/Tick
**Location**: `mytrader/execution/live_trading_manager.py:_query_aws_knowledge_base()` (line 1621)

**Issue**: KB queries may be made on every trading cycle without sufficient caching or rate limiting.

**Evidence**:
- Line 1652-1675: Cache exists but TTL may be too short
- No bulk indexing, queries may be frequent
- No retention/TTL for old data

**Impact**: Cost/perf leaks, API rate limits, slow trading loop.

**Fix Required**:
- Default `OPENSEARCH_ENABLED=false` locally
- Increase cache TTL, batch queries
- Add retention/TTL for old data
- Rate limit queries per minute

---

## Summary

**Severity-1**: 5 critical issues that can cause unprotected trades, ignored guardrails, and learning failures.

**Severity-2**: 4 important issues that can cause cost/perf problems and tracking errors.

**Priority**: Fix all Severity-1 issues before live trading. Severity-2 can be addressed incrementally.

