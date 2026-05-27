# Live-vs-Backtest Forensic: why ~0 trades and mostly losers (2026-05-27)

Root-cause investigation (no optimization). Question: backtest does ~15 trades/month
profitably; live does 3–4/month, mostly losers. Where do the trades disappear, and
why do the survivors lose?

## Data caveats (read first)
- Live logs span only ~2 months: `live_trading.log` from 2026-03-29, `trading_manager.log`
  from 2026-05-05. The full 7-month history is rotated/gone, so per-day counts cover the
  recent window, not all 7 months. The *mechanisms* below are current and confirmed.
- `orders.db` is **contaminated** (mixes backtest/paper/live — Feb-2026 shows
  "+$137,988" on a $4,499 account). Absolute trade counts from it are unreliable; only
  trends/signs are used.

## The funnel — where trades disappear

| Stage | Count (recent window) | Note |
|---|---|---|
| Strategy signals emitted (`📊`) | **78 in ~2 months (~39/mo)** | **NOT starved** — more than the backtest's trade rate |
| Manager decisions on MES (May 5–27) | **10,105 REJECT, 68 MODIFY, 0 APPROVE** | re-scored per cycle; **zero approvals** |
| — rejected for **R:R < floor (2.0)** | **9,561 (95%)** | the deadlock |
| — rejected for risk > $90 cap | 489 | small-account cap |
| — rejected regime-fit / streak | 41 / 14 | minor |
| Actual fills | ~0 recently (was 34→14→2 Mar→Apr→May) | collapse tracks the manager gating |

**The collapse is 100% at the manager layer, not signal generation.** The strategy
fires plenty; the manager vetoes ~everything because it demands R:R ≥ 2.0 while the
strategy is built for ~1.3:1. **The backtest never ran this gate** (it uses
`RiskManager` only) — so the +$2,142 book and the live system are two different systems.

## Why the survivors lose
- Win rate collapsed monotonically as the regime/old-config played out: Dec 55% → Jan
  42% → Mar 29% → Apr 14% → May 0%.
- Pre-deadlock live trades were **old-config** (shorts enabled, overnight entries) — the
  exact things the validated config removes. Earlier analysis: live longs 19% WR driven
  by overnight (9% WR); shorts −$380.
- The edge is **thin and slippage-fragile** (t-stat 2.09; dies at +2–3 ticks slippage —
  see ARCHITECTURE_AUDIT.md). In any non-ideal stretch it nets negative.

## Hypotheses RULED OUT
- **15m candle behavior / finalization / timing / data alignment:** live signal
  timestamps match backtest bars **78/78 exactly**; closes match within a **~4pt median**
  (the ES-as-MES proxy basis, not a candle bug). Signals fire on properly finalized,
  correctly-aligned 15m bars. **Not the problem.**
- **Signal starvation / over-filtering at the strategy layer:** ~39 signals/month
  emitted. **Not the problem.**
- **Market-regime mismatch:** recent ~7mo is *as favorable or more* than the profitable
  2025 window — higher ATR (10.8 vs 8.7), more trending (ADX≥25 55% vs 52%), more
  uptrend (48% vs 45%), less chop (14% vs 17%). **Not the cause.**
- **Data integrity** (missing/dup/stale candles): backtest 1m data is clean (consistent
  monthly counts, only weekend gaps); live bars that fired are correctly timestamped.
  No evidence of corruption at the decision points.

## Top 5 root causes (ranked by probability × impact)
1. **Manager R:R ≥ 2.0 deadlock (PRIMARY, confirmed).** 9,561/10,105 rejects; 0
   approvals. Strategy R:R ~1.3 can never clear a 2.0 floor; adaptive relaxation needs
   winning history it can't accrue (cold-start deadlock). → near-zero trades. *This alone
   explains the frequency collapse.*
2. **Backtest ≠ live gating (structural).** The validated edge was measured WITHOUT the
   manager. Live adds it. Until both use the same gate, live can never resemble backtest.
3. **Pre-deadlock losses = old config + thin/fragile edge.** Shorts + overnight in a
   choppy/high-ATR stretch, on an edge that's marginal (t≈2.1) and slippage-sensitive.
4. **$90/trade risk cap on a $4,499 account.** 489 rejects — any setup with a stop wider
   than ~18 pts (1 MES) is auto-rejected, so live can only ever take the *tightest*-stop
   subset, which is not the validated population.
5. **ES-as-MES proxy basis (~4 pt/trade).** Backtest priced ES; live trades MES. On 6–8pt
   stops, a 4pt median basis is material noise → live outcomes diverge from backtest even
   with identical logic.

## Verdict
Not architecture-broken, not a 15m-candle bug, not regime, not data. The dominant cause
is **execution gating** (the manager R:R deadlock) on top of a **research/live mismatch**
(backtest never gated by the manager), with **a thin, slippage-fragile, proxy-noised
edge** explaining why the few survivors lose. The system isn't "broken" — it's *blocked*,
and what little leaks through is the wrong (old-config, tightest-stop) subset.

## Fixes ranked by expected impact
1. **Unify the gate, then re-validate.** Either (a) remove/replace the manager's static
   R:R≥2.0 with an expectancy-based floor matching this high-WR/~1.3 strategy, AND re-run
   the backtest WITH the manager in the loop — so backtest and live are the *same system*.
   This is the only way live can resemble backtest. (`TM_MIN_RR=1.2` applied; needs
   manager restart — SPY already approving today, MES still rejecting as of May 26.)
2. **Confirm the restart actually took** and watch the manager log flip MES to APPROVE.
3. **Right-size the account/cap or the stops** so the $90 cap stops auto-rejecting the
   wider-stop (often better) setups.
4. **Backtest on real MES 1m**, not ES proxy, to remove the 4pt basis noise before trusting
   any number.
5. **Do not size up** until the unified-gate book is forward-validated; the edge is too
   thin (t≈2.1, slippage-fragile) to run on faith.
