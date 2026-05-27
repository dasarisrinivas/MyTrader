# Finer-timeframe (5m) prototype — lifts frequency, loses the edge (2026-05-26)

## Question

After new orthogonal signals failed to add edge, the remaining frequency lever was
a **finer decision timeframe**. Prototype: run the validated 15-minute strategy on
5-minute bars (3× the decision points) and see whether the added frequency keeps the
edge.

## Setup

- Resampled `ES_1m_multiyr` → 5m (155,053 bars, 26 months, open-labeled).
- Ran the **live-faithful** config (shorts-off, opening/Friday blocks, ADX-12) on 5m
  bars through the same engine path as the 15m book. Indicators (EMA 9/21/50, ADX/ATR/
  RSI 14) are bar-count based, so at 5m they span 1/3 the wall-clock — i.e. a faster-
  trading variant, which is what trading this strategy at 5m actually means.
- Compared 5m vs the validated 15m book on identical windows.

## Result — frequency up, edge gone

| Window | 15m (validated) | 5m (prototype) |
|---|---|---|
| Feb–May 2025 | 29 trades, **+$219**, 48.3% WR | 71 trades, **−$227**, 39.4% WR |
| Jun–Sep 2024 | 55 trades, **+$166**, 49.1% WR | 98 trades, **−$175**, 45.9% WR |

The 5m timeframe **does** lift frequency — ~1.8× to 2.4× the trades. But PnL flips
**negative** in both windows and win rate drops 3–9 points. Consistent across the
favorable 2025 tape and the harder 2024 tape, so it isn't a one-regime fluke.

## Why

The strategy's gates — ADX floors, RSI bands, ATR floors, stop/TP multipliers,
opening-range and exhaustion windows — were tuned for 15-minute bars. On 5m bars the
indicators are 3× faster and far noisier, so the same thresholds admit many more but
much lower-quality entries. More trades, negative expectancy.

## Recommendation — stay on 15m

A *naive* 5m run fails. A *proper* 5m strategy would require re-calibrating every gate
for 5m's noise characteristics — effectively building and validating a new strategy
variant (months of walk-forward work, real overfitting risk), and it should be
shadow-traded before any live capital. Given the 15m book is already validated and
positive on 26 months, that speculative project isn't justified for a small live
account right now.

**Net of this whole frequency investigation:** the one frequency gain that survived
faithful testing is the one already applied — `ft_adx_min` 18 → 12 (+44% trades,
+58% PnL, validated on 26 months). New signals (momentum breakout + five more long
triggers) and the finer 5m timeframe were all tested and did not hold up. The
current 15m configuration is the validated optimum.

Artifacts: `p5m_probe` / `p15m_probe` (Feb–May 2025), `p5m_2024` / `p15m_2024`
(Jun–Sep 2024); 5m data at `data/ib/ES_15m_from5m.parquet` (5m bars, named for the
engine's df_15m routing).
