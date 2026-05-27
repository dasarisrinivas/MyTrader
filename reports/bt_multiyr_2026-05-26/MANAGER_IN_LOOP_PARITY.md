# Trade-Manager-in-the-Loop Parity (#3) — 2026-05-27

Goal: make the backtest use the **exact same gate** the live engine uses, then compare
apples-to-apples. Built by replaying the **real `rules.evaluate`** (+ the real
`_maybe_update_posture` and `streaks_from_recent` helpers) over the 392 validated
engine trades in time order, evolving `ManagerState` (posture/streaks/daily PnL) from
the backtest's own approved fills. Not a replica — the same functions the live engine
calls.

**Faithfulness note / finding:** the live manager's *state refresh* is hard-wired to
wall-clock (`datetime.now()`, `_today_ct()`) and reads streaks from `orders.db`. That is
**why it was never backtestable** — it isn't time-injectable. The gate itself
(`rules.evaluate`) is pure, so it can be driven faithfully; the only adaptation is
scoping session/streak state by bar-time instead of `now()`. (My first pass had a bug —
streaks not reset per session — which over-rejected; corrected below.)

## The parity table

| Book | Trades | /mo | WR | Net PnL | Exp/trade |
|---|---|---|---|---|---|
| Engine (no manager) — the "validated" book | 392 | 15.0 | 53.1% | +$2,142 | +$5.46 |
| **Manager @ RR floor 2.0 (current LIVE)** | **0** | **0** | — | **$0** | — |
| **Manager @ RR floor 1.2 (the staged fix)** | **373** | **14.3** | **54.4%** | **+$2,560** | **+$6.86** |

Reject reasons @ 2.0: **R:R = 392 (100%)**. @ 1.2: framework/regime-fit 18, other 1.

## Root cause, proven from the data distribution

R:R of all 392 validated trades:

| R:R band | count |
|---|---|
| < 1.2 | 0 |
| **1.2 – 1.5** | **392 (100%)** — median 1.33, mean 1.31 |
| 1.5 – 2.0 | 0 |
| ≥ 2.0 | 0 |

The strategy is, by construction, a ~1.33:1 system (take-profit ≈ 1.33× stop). The live
manager required ≥ 2.0. **The overlap is exactly zero** — at floor 2.0, 0/392 can ever
pass; at floor 1.2, 392/392 clear the R:R gate. That single incompatibility *is* the
7-month live failure.

## Answering the three diagnostic questions

1. *Manager improves expectancy but over-constrains frequency?* — Only mildly, and only
   at the correct floor: at 1.2 it keeps 373/392 (14.3/mo vs 15.0) and nudges expectancy
   up ($6.86 vs $5.46) by trimming 19 marginal trades. Not the problem.
2. *Manager destroys the edge entirely?* — **Only at floor 2.0**, where it rejects 100% →
   0 trades. That is the live deadlock. At 1.2 the edge is preserved/slightly improved.
3. *Edge only existed because the backtest omitted execution constraints?* — **No.** The
   edge survives the real manager at the correct floor. It does **not** survive the
   *mis-set* floor (2.0) — which is what live has been running.

## Slippage (manager @ 1.2 book)
+$0/tr → +$2,560; +$2.5/tr → +$1,627; +$5/tr → +$695. Survives moderate slippage but
the per-trade edge is thin — consistent with the earlier robustness review (t≈2.1).

## Conclusion

The live behavior is **not** a bug in the strategy, the 15m candles, the regime, or the
data. It is one mis-set parameter: **`TM_MIN_RR` = 2.0 against a strategy that produces
1.2–1.5 R:R.** With the manager wired into the backtest at the live floor (2.0), the
backtest *also* produces 0 trades — i.e. **the backtest now reproduces live exactly.**
That is the parity we were missing. At floor 1.2, the unified system reproduces the
validated book (+$2,560 / 54% / 14/mo).

## Now implemented as a first-class backtest mode (2026-05-27)

The harness was a post-hoc CSV replay (couldn't capture slot-freeing — a manager
rejection lets a later signal take the freed slot). It's now wired **into the engine
loop** so the real gate runs at signal time and rejections free slots:

- `backtest/manager_gate.py` — `BacktestManagerGate`: calls the real `rules.evaluate`
  + `_maybe_update_posture` + `streaks_from_recent`, state evolved from backtest fills
  (bar-time scoped). No live code touched.
- `backtest/engine.py` — gate runs before the risk gate (sees the strategy's proposed
  levels, like the live feed); closed-trade outcomes feed `record_outcome`. **Default
  OFF**; enabled only via env `BT_WITH_MANAGER=1`.
- `backtest/run.py` — `--with-manager` and `--tm-min-rr` CLI flags.

Validated (path-faithful, in-loop, 26 months):

| Mode | Trades | PnL |
|---|---|---|
| No manager (old default backtest) | 392 | +$2,142 |
| `--with-manager --tm-min-rr 2.0` (current LIVE) | **0** | $0 — reproduces the live deadlock |
| `--with-manager --tm-min-rr 1.2` (the fix) | **367** (~14/mo) | **+$2,680** |

So at the live floor the backtest now produces **0 trades exactly like live**, and at
1.2 it reproduces (slightly improves) the validated book. Research and live are finally
the same system. **Use `--with-manager` for all future evaluation.**

## Fixes (ranked)
1. **Activate `TM_MIN_RR=1.2`** (already staged; needs the launchd reload to take — MES
   still shows `required 2.0`, so it has NOT applied yet). This alone restores trading.
2. **Adopt this manager-in-loop harness as the standard backtest** so research and live
   are always the same system. Never validate a config without the manager again.
3. **Make the floor expectancy-based** (win-rate aware) rather than a static number, so a
   high-WR/low-RR strategy can't be silently zeroed out again.
4. Re-run on **real MES data** (not ES proxy) and **forward-validate** before sizing —
   the edge is thin and slippage-sensitive even after the fix.
