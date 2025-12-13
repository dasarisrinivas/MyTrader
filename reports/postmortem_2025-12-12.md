# Postmortem – 2025-12-12 ES Loss

## Incident Summary
- **Trade cycle ID:** derived from exit order `15638` at **09:35:10 CST**
- **Symbol/contract:** ESZ5 (E-mini S&P 500 Dec 2025), 1 contract
- **Entry:** 08:53:13 CST BUY @ **6890.50** (order 14812) – created unintentionally while attempting to flatten a short
- **Exit:** 09:35:22 CST SELL @ **6856.50** (order 15638)
- **Stop/Target:** *None* (exit-order path supplied `SL=None`, `TP=None`)
- **Outcome:** Loss **-$1,700** ((6856.50 − 6890.50) × $50), no protective orders, AWS agents advised `WAIT`
- **Telemetry gap:** No `trade_cycle_id` nor trade logger entry; `mytrader.rag.pipeline_integration:log_trade_exit` warned “No current trade ID for exit logging”

## Timeline (America/Chicago)
| Time | Event |
| --- | --- |
| 08:53:01 | Hybrid pipeline emitted `SCALP_BUY` @ 0.36 confidence while bot was **short** 1 contract. |
| 08:53:13 | AWS Decision Agent returned `WAIT (50%)` but `live_trading_manager` only reduced local confidence (“WAIT advisory only - not blocking trade”). |
| 08:53:13 | `_process_hybrid_signal` treated the BUY signal as an **exit**, calling `_place_exit_order` with `stop_loss=None`, `take_profit=None`. |
| 08:53:20 | Exit order 14812 filled **BUY 1 @ 6890.50**. Because the executor lacked a *reduce-only* guard, this reversed the book from short → long and left the new long **without any bracket children** (`Telegram alert: SL=None, TP=None`). |
| 08:53–09:34 | Pipeline kept emitting HOLD/SELL signals with repeated warnings `Hybrid pipeline produced non-positive protective levels (stop=%s, target=%s)`; bot ignored AWS WAIT advisories. |
| 09:35:04 | Signal `SCALP_SELL` triggered HYBRID EXIT for the unintended long. |
| 09:35:22 | Exit order 15638 filled SELL 1 @ 6856.50; IB commission report recorded **PNL -592** because IB’s average cost snapshot was stale, but actual price delta was -34 points (~$1.7k). |
| 09:35:22 | No trade record was written (`No current trade ID for exit logging`) leaving the loss opaque until log review. |

## Root Causes
1. **Exit orders could open fresh positions** – `_place_exit_order` forwarded the signal’s direction to `ib_executor.place_order` without re-checking holdings or forcing reduce-only semantics, so BUY-to-cover orders became outright BUYs (see historical behavior in `mytrader/execution/live_trading_manager.py:1370-1405` prior to this fix).  
2. **AWS WAIT advisories were treated as cosmetic** – `_process_hybrid_signal` merely shaved confidence and continued to trade, even when the Bedrock decision explicitly said `WAIT`, leading to trades against the guardrail (`mytrader/execution/live_trading_manager.py:1258-1350`).  
3. **Stop/Target validation absent on exit path** – exit orders were allowed with `SL=None`, `TP=None`, so when an exit accidentally became an entry there were no protective brackets (logs show `Telegram alert: SL=None, TP=None`).  
4. **No trade-cycle correlation or learning hooks** – Without a `trade_cycle_id`, telemetry, or learning ingestion, the loss was invisible to dashboards and could not feed the Learning/Data agents for future mitigation.

## Fixes Implemented
1. **Reduce-only exits & guardrails** (`mytrader/execution/live_trading_manager.py`, `mytrader/execution/ib_executor.py`, `mytrader/monitoring/order_tracker.py`):  
   - Exit path now re-checks the live position, derives the correct BUY/SELL action, and calls `place_order(..., reduce_only=True)` which enforces direction/size and blocks if no exposure exists.  
   - Order tracker + executor carry `trade_cycle_id` metadata so fills, telemetry, and structured logs stay linked.
2. **Trade-cycle telemetry & structured logs** (`live_trading_manager.py`):  
   - Every candle cycle receives a UUID shared across signal generation, AWS decisions, risk checks, order submissions, and PnL events via `log_structured_event`.  
   - Reason codes (`AWS_WAIT`, `MISSING_PROTECTION`, `MAX_LOSS_CAP`, etc.) accumulate per cycle to explain future incidents.
3. **Safety guards on entries** (`live_trading_manager.py`, `mytrader/execution/guards.py`):  
   - New `_validate_entry_guard` rejects trades lacking valid SL/TP or exceeding configurable `max_loss_per_trade`.  
   - Wait advisories now block trades by default (`settings.aws_agents.block_on_wait`), with optional high-confidence overrides for advisory-only responses.
4. **Learning & ingestion hooks** (`mytrader/learning/trade_learning.py`):  
   - On trade close the bot records a structured payload (signal context, AWS advice, risk params, reason codes) under `rag_data/training/trade_outcomes/<date>/<trade_cycle_id>.json`.  
   - A matching historical snapshot (recent candle stack, regime metadata) is stored under `rag_data/history_snapshots/`. This gives the Learning and Data Ingestion agents deterministic artifacts to ingest.
5. **Unit tests** (`tests/test_execution_guards.py`):  
   - Added coverage for wait gating and bracket validation to prevent regressions.

## Follow-up / Monitoring
- Validate that new telemetry shows a single `trade_cycle_id` from signal → fill → PnL.  
- Use the recorded outcome for 2025-12-12 loss to seed the Learning Agent and adjust filters for similar `AWS_WAIT` + missing protection scenarios.  
- Guardrail replay: `python3 scripts/replay_trade_from_logs.py --log logs/bot.log --order-id 14812` now reports `Guardrails would block trade: ✅`, confirming the WAIT advisory plus missing SL/TP would prevent the loss in dry-run mode.
