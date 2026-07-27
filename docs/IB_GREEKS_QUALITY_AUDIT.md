# IB Greeks Data-Quality Audit + Phase-1 Instrumentation

**Date:** 2026-07-24. Observation/measurement only. No strategy/allocation/
governor/confidence/exit/risk changes.

## Root cause of the 9% missing-greeks (from historical data)
91/977 dispatched signals had missing/zero greeks. Split by mechanism:
- **~60% (55/91) NO quote at all** (bid=0 & ask=0, median OI=0) — dead/illiquid/
  uninitialized contract. Nothing (incl. ThetaData) can trade a contract with no
  market. Fail-closed rejection is correct.
- **~40% (36/91) quote present, greeks absent** — real contract, real NBBO, IB
  returned no modelGreeks. Mechanism confirmed in code: `get_snapshot_with_greeks`
  does a SINGLE fixed `sleep(greeks_wait_s)` then reads modelGreeks ONCE, no
  retry. Greeks arriving at `greeks_wait_s + ε` are recorded missing → a
  timing/race, fixable by a retry/longer wait — NOT a data-vendor problem.

Concentration: 0DTE miss-rate 3%, but 1-2 DTE 33% / 3-7 DTE 22% / 8+ 20%.
The 11 real fills to date are all 1-2 DTE (7) and 3-7 DTE (4) — the HIGH-miss
bands — so practical impact is NOT provably small; the fill sample (n=11) is just
too thin to conclude. `pnl_pct` of the 20 greek-missing outcomes is coin-flip →
"currently no evidence" of benefit from recovery, NOT "impossible".

## What historical data CANNOT answer (needs live instrumentation)
Quote/greek arrival latency, greek refresh rate, contract-qualification→first-
greek time, and trades-rejected-solely-by-missing-data. Hence Phase 1.

## Phase-1 instrumentation (built, log-only, feature-flagged)
`ib_client.get_snapshot_with_greeks`: during the SAME total `greeks_wait_s` wait
(no behavior change, no timing change, no result change), poll to record
per-contract `first_quote_ms` / `first_greek_ms`; on every missing-greek event
append a JSONL record to `logs/greeks_quality.jsonl` with quote/greek timing,
bid/ask, OI, volume, and a `class` (`no_quote` vs `quote_no_greek`). Gated by
`greeks_quality_log` (default on via getattr — backward compatible). Wrapped in
try/except so instrumentation can never affect the data path.
Analyzer: `scripts/analyze_greeks_quality.py`.

Also: `_compute_iv_rank` docstring corrected — it is a normalized VIX regime
score (0-100), NOT option IV rank/percentile. Read the `iv_rank` field/column as
`vix_regime_score`; a true rename is a tracked migration.

## Roadmap (agreed)
1. **Instrumentation** (this) — log-only, deployed to live_trading.
2. **Measure** ~30-50 sessions → run analyzer: what fraction of missing-greek
   events are `quote_no_greek`, and do greeks arrive late/never within the wait?
3. **Prototype retry** ONLY if data shows meaningful recoverable rate — feature-
   flagged short wait/retry after subscription; measure the execution tradeoff
   (entry delay, price drift, fill quality) — a retry is NOT free.
4. **Reassess ThetaData** only if retries still leave an execution-relevant gap.

Decision stays evidence-based: no ThetaData wiring, no retry, until the live data
justifies it.
