# MyTrader Guardrail Runbook

## Test & Verification Checklist

1. **Unit tests** – run the most relevant suites before deploying guardrail changes:
   ```bash
   python3 -m pytest \
     tests/test_execution_guards.py \
     tests/test_order_state_persistence.py \
     tests/test_hybrid_guardrails.py
   ```
   These cover risk gates, idempotency persistence, cooldown restore, and hybrid HOLD-on-error logic.

2. **Paper / mock IB mode** – launch the bot with simulation enabled to exercise order flow without touching a live account:
   ```bash
   python3 run_bot.py --simulation
   ```
   or instantiate `LiveTradingManager(settings, simulation_mode=True)` inside a notebook/test harness. All guardrails (entry validation, PendingSubmit detection, cooldowns) run in simulation.

3. **Confirm “no entry without confirmed SL/TP”** – watch for:
   - `Entry blocked by hard guardrails` + reason codes in `live_trading.log`.
   - `protective_orders_not_confirmed` structured events when IB fails to acknowledge child orders within ~8s (the executor cancels the parent and logs the incident ID).

4. **Confirm PendingSubmit blocks new orders** – during IB hiccups, `ib_executor` logs show `🔍 Sync: N active orders (... PendingSubmit)` and `LiveTradingManager` will skip placement with `↳ {active_orders} active orders pending`. You can provoke this in simulation by monkeypatching `_confirm_protective_orders` (see `tests/test_execution_guards.py::test_pending_submit_counts_as_active_order`).

5. **Hybrid crash ⇒ HOLD** – trigger an exception in `HybridPipelineIntegration` (e.g., by patching `process` to raise) and verify:
   - `hybrid.pipeline_error` structured event in logs.
   - `LiveTradingManager` broadcast signal is forced to HOLD and no legacy path runs until the next candle.

6. **LLM throttling / caching** – confirm the following telemetry in `logs/live_trading.log` or structured logs:
   - `bedrock.call_start/bedrock.call_complete` events with `latency_ms`, `prompt_chars`, `calls_last_minute`.
   - Only one Bedrock call per candle unless a conflict is detected; repeated prompts should hit the cache (`LLM skipped … using cached response`).
   - `data/hybrid_llm_cache.json` contains the persisted response cache (for offline replays).

7. **IB metrics** – monitor warnings if contract qualifications or snapshot requests exceed healthy per-minute limits:
   - `metrics.qualified_contract_calls` and `metrics.snapshot_price_requests` structured events fire with counts when crossing thresholds.
   - Cached contracts (`Using cached contract ES…`) and cached prices (`Using cached price snapshot …`) should appear in logs after the first request.

## Operational Tips

- **Reconcile cooldown on restart** – `OrderTracker` now persists `last_trade_time` per symbol. After a crash or redeploy, the manager logs `⏱️ Cooldown resume: last trade at …` and respects the remaining cooldown before placing new entries.
- **Idempotency persistence** – duplicate entries are blocked even across restarts (`duplicate_submission_blocked` events). Signatures are derived from symbol, candle timestamp, signal ID, and entry price bucket; verify `data/orders.db` table `submission_signatures` if troubleshooting.
- **Structured log grep cheatsheet**
  - `grep -E "hybrid.pipeline_error|RISK_BLOCKED_INVALID_PROTECTION" logs/live_trading.log`
  - `grep protective_orders_not_confirmed logs/reconcile.log`
  - `grep bedrock.call logs/live_trading.log`

Keep this runbook close when validating hotfixes or during on-call so you can quickly prove the guardrails are engaged without digging through the entire codebase.
