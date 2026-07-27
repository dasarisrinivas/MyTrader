# Replay Engine — Validation, Calibration & Sensitivity Baseline

**Engine:** `shree/research/replay_engine.py` v2.0-real-option-replay (canonical).
Observation only; no production changes. Legacy SPY-barrier grade is DEPRECATED
for any tradeability claim — grade on real option P&L here.

## Phase 1 — Engine validated (GATE PASSED)
30 trades across dates/families:
- **Internal consistency: 30/30 exact** — engine's entry-ask/exit-bid/net match an
  independent tick-scan (different code path) to the cent, 0 flagged >1%.
- **Cross-vendor: median 0.2%, p90 1.1%** — ThetaData NBBO-mid vs the bot's
  IB-recorded `entry_mid` agree, confirming quote correctness AND ET timestamp
  alignment across two independent vendors. The >1% items are all cross-vendor
  sub-penny/timing differences, never internal.

## Metric note (important)
`expectancy_%` is outlier-dominated for options (a $0.05→$0.20 print = +300%);
**dollar expectancy is the ground truth.** Reports use $.

## Engine limitation (v3 work)
The engine replays the ONE recorded strike/right → correct for single-leg
families, WRONG for multi-leg (BULL_CALL_SPREAD, BEAR_PUT_SPREAD, LONG_STRADDLE).
Their re-grade numbers are invalid single-leg approximations. Multi-leg replay
is required before grading spreads.

## Phase 2 + Calibration — single-leg families (60/family, real $)

| Strategy | Barrier E% | Real E$ | PF | Kelly | Tail p95 | Barrier bias |
|---|---|---|---|---|---|---|
| CALL_SWEEP | +0.009 | **+0.23** | 1.00 | 0.00 | −87% | overstated tail |
| PC_RATIO_EXTREME | −0.026 | **−22.3** | 0.64 | −0.21 | −70% | +0.007 |
| TREND_CONTINUATION | −0.041 | **−24.0** | 0.26 | −0.96 | −46% | +0.059 |
| PUT_SWEEP | −0.051 | **−25.2** | 0.53 | −0.31 | −87% | +0.101 |
| ORB_BREAKOUT | −0.070 | **−36.0** | 0.23 | −0.95 | −70% | +0.093 |

**Barrier bias is systematically optimistic** (it caps losses at ±0.5% SPY while
real 0DTE OTM options decay to near-zero, tail p95 −70..−87%). Magnitude is
contract-dependent (moneyness/DTE), so prior barrier conclusions must be
**re-run through the engine, not scalar-corrected** — but every qualitative "no
edge" verdict HARDENS (real is worse), so nothing is discarded, only re-graded.

## Sensitivity triage — is any family worth optimizing?
Perfect-execution ceiling = remove commission + recover round-trip spread.

| Strategy | Real E$ | Perfect-exec ceiling | Dir hit | Worth optimizing? |
|---|---|---|---|---|
| CALL_SWEEP | +0.23 | +8.83 (spread tight/liquid → real) | 0.50 | **YES** |
| PC_RATIO_EXTREME | −22.3 | +7.04 *(fictional — $14 illiquid spread not recoverable)* | 0.47 | NO |
| PUT_SWEEP | −25.2 | −16.8 | 0.47 | NO |
| TREND_CONTINUATION | −24.0 | −18.5 | 0.41 | NO |
| ORB_BREAKOUT | −36.0 | −28.3 | 0.35 | NO |

## Signal vs execution — decomposition (avg $/trade, ranked)
1. **Signal quality (direction 0.44 < 0.50)** — dominant: residual EV **−$9.55**
   even at PERFECT execution. No contract fixes sub-coin-flip direction.
2. Spread −$10.62 (≈half is unrecoverable illiquidity tax).
3. Theta / OTM decay (tail −72%).
4. Commission −$1.30 (fixed).

**Signal generation is the primary bottleneck — quantified.** Freeze contract/
execution optimization; it cannot create directional edge that isn't there.

## Decisions
- **PERMANENTLY ARCHIVE** (real E$ deeply negative, ceiling still negative, dir<0.50):
  ORB_BREAKOUT, PUT_SWEEP, TREND_CONTINUATION, PC_RATIO_EXTREME. No contract/exec
  choice recovers $17–28/trade with a losing directional signal.
- **Phase-4 multi-contract replay: CALL_SWEEP ONLY** — the sole liquid, breakeven,
  50%-direction candidate where contract selection could plausibly cross positive.
- **No new strategies / features / tuning** until signal-generation edge exists.

## Phase 4 — variance decomposition (CALL_SWEEP, 138 signals × 6 contracts)
Two-way ANOVA on returns (`scripts/phase4_contract_variance.py`):

| Factor | η² |
|---|---|
| **Signal direction** | **77.2%** |
| **Contract choice** | **0.7%** |
| interaction (idiosyncratic) | 22.1% |

Per-contract: expectancy stays ≈0-to-negative for ALL contracts (mean range
driven by a Δ25 penny outlier); volatility swings 4× (ATM-2DTE std 28% → Δ25
118%). **Contract choice is a VOLATILITY lever, not an expectancy lever.**

**Architectural conclusion (generalizes):** for this architecture, signal
direction dominates P&L variance (77%); contract selection is a second-order
effect (<1%) that only reshapes risk. Measured on the best, most-liquid,
50%-direction family — so contract optimization is a second-order avenue for the
currently-tested families. Empirical, not just the theoretical ceiling.
Recommendation: freeze contract/execution optimization as a research avenue; the
first-order lever is signal generation.

## RESEARCH ENGINE v2 — COMPLETE (frozen)
The engine has answered its architectural question. FREEZE:
- API (`shree/research/replay_engine.py`) and the evaluation methodology.
- Every future strategy is graded through this engine UNCHANGED — no weekly
  drift, so results stay comparable across experiments.
- Real option dollar-P&L is the canonical metric; SPY barrier is retired.

Known v3 work (only reason to unfreeze): **multi-leg replay** (spreads/straddles
are currently mis-graded single-leg). Do NOT expand the engine for anything else.

## Research version lineage
v1 barrier (deprecated) · **v2 real option replay — COMPLETE/FROZEN** ·
v3 +multi-leg replay (only planned change) · v4 +execution latency (deferred).
