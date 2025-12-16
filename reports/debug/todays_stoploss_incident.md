# 2025-12-15 Stop-Loss Incident Report

## Order Audit Summary
| CST Time | Action | Qty | Limit | Stop Loss | Take Profit | Log Reference | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 09:53:01 | SELL | 5 | Market | 6829.02 | 6826.06 | logs/reconcile.log:446 | ✅ bracket present |
| 10:51:03 | BUY | 5 | Market | 6821.44 | 6825.08 | logs/reconcile.log:447 | ✅ bracket present |
| 10:55:03 | BUY | 5 | Market | 6822.44 | 6835.41 | logs/reconcile.log:448 | ✅ bracket present |
| 10:59:01 | BUY | 3 | Market | 6821.57 | 6824.06 | logs/reconcile.log:449 | ✅ bracket present |
| 11:09:05 | BUY | 5 | Market | 6818.90 | 6831.38 | logs/reconcile.log:450 | ✅ bracket present |
| 11:27:05 | BUY | 3 | Market | 6821.00 | 6823.00 | logs/reconcile.log:451 | ✅ bracket present |
| 11:45:05 | SELL | 3 | 6826.25 | — | — | logs/reconcile.log:452, logs/live_trading.log:15419-15485 | ❌ reduce-only exit submitted without SL/TP |
| 12:41:04 | BUY | 3 | 6814.00 | — | — | logs/reconcile.log:453, logs/live_trading.log:18498-18540 | ❌ reduce-only exit (flatten) |
| 12:46:02 | BUY | 3 | 6817.25 | — | — | logs/reconcile.log:454, logs/live_trading.log:18788-18832 | ❌ reduce-only exit (flatten) |
| 13:58:04 | SELL | 3 | 6826.50 | — | — | logs/reconcile.log:455, logs/live_trading.log:20286-20330 | ❌ reduce-only exit (flatten) |
| 14:06:05 | BUY | 3 | 6828.00 | — | — | logs/reconcile.log:456, logs/live_trading.log:20450-20494 | ❌ reduce-only exit (flatten) |
| 14:07:01 | SELL | 3 | 6828.50 | — | — | logs/reconcile.log:457, logs/live_trading.log:20520-20564 | ❌ reduce-only exit (flatten) |
| 15:24:24 | SELL | 2 | 6824.25 | — | — | logs/reconcile.log:458, logs/live_trading.log:6612-6658 | ❌ reduce-only exit (flatten) |
| 15:26:05 | BUY | 2 | 6823.00 | — | — | logs/reconcile.log:459, logs/live_trading.log:6837-6881 | ❌ reduce-only exit (flatten) |
| 15:48:04 | BUY | 2 | 6824.50 | — | — | logs/reconcile.log:460, logs/live_trading.log:8653-8697 | ❌ reduce-only exit (flatten) |

## Findings
- The first six orders placed today (09:53–11:27 CST) included valid stop-loss and take-profit brackets; telemetry confirms `_validate_entry_guard` worked as intended.
- Beginning at 11:45 CST (`logs/live_trading.log:15419-15485`), the bot began issuing `HYBRID EXIT` orders via `_place_exit_order`, which intentionally sets `stop_loss=None` and `take_profit=None` (see `mytrader/execution/live_trading_manager.py:1602-1655`).
- `TradeExecutor.place_order` (`mytrader/execution/ib_executor.py:699-1065`) lacks a final guard to reject non-reduce-only orders without valid protective levels. As a result, any caller that forgets to attach an SL/TP—or any exit flow that reuses the same API—will still hit IB.
- The log snapshot at `logs/live_trading.log:15419-15485` shows the first offending order: exit request logged, `stop_loss=None`, Telegram alerts announcing SL=None, and reconciliation events confirming the trade was recorded without protection.

## Likely Root Cause
1. `_place_exit_order` always lands in `reduce_only=True` mode and explicitly passes `stop_loss=None` / `take_profit=None` to `TradeExecutor.place_order` (`mytrader/execution/live_trading_manager.py:1632-1645`).
2. `TradeExecutor.place_order` never differentiated between protective entry orders and reduce-only exits. It happily proceeded through PositionManager checks and bracket construction even when both SL/TP were missing, and since `reduce_only` orders skip the bracket branch entirely (`order.transmit = True`), IB accepted a naked limit order.
3. No structured logging or guardrails flagged this condition, so downstream tooling (reconcile log, Telegram alerts) silently recorded SL=None, TP=None, meeting the definition of "order submitted without a valid stop loss."

## Code Locations Implicated
- Exit path bypassing protection: `mytrader/execution/live_trading_manager.py:1602-1655` (`stop_loss=None` comment in `_place_exit_order`).
- Missing guard in executor: `mytrader/execution/ib_executor.py:699-1065` (no validation before `placeOrder`).
- Evidence in telemetry: `logs/live_trading.log:15419-15485` (first offending order) plus subsequent exit blocks listed above.

These findings inform the fix: enforce a hard protective guard in `TradeExecutor.place_order` for every non-reduce-only submission, emit structured incidents when tripped, and ensure all callers supply a concrete entry price so IB submissions can be blocked before reaching `placeOrder`.
