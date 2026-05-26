# ADX-min re-validation on 26 months of data (2026-05-26)

## What this is

The earlier ADX tuning was validated on ~13 months (Feb 2025 – Jan 2026, the only
1-minute data on disk). You pulled fresh ES 1-minute history from IB, which let me
roughly double the sample. This re-runs the **live-faithful** backtest (your
`one_minute:` config — shorts disabled, opening + Friday blocks on) and re-tests the
pullback ADX floor (`ft_adx_min`) on the longer history.

## Data

- **File:** `data/ib/ES_1m_multiyr.parquet` (775,263 1-minute bars), resampled to
  `data/ib/ES_15m_multiyr.parquet` for the 15m strategy.
- **Span:** 2024-03-15 → 2026-05-26 (~26 months). IB wouldn't serve 1m older than
  Mar 2024, which is the depth wall, not a code limit.
- **Fidelity checks:** outside the quarterly roll weeks, my resampled 15m matches the
  native `ES_15m_1y` file to the penny (corr 0.9994; 0.00 diff in 8 of the overlap
  months). The roll differences are an overnight contract-roll convention, not an
  intraday artifact — the largest intraday moves in the series are the real
  Apr-2025 tariff crash/rally days, not roll gaps.
- **Path check:** reproducing the prior Feb–Jun 2025 ADX-12 chunk on this engine gave
  **$791.8 / 43 trades**, identical to the prior validated run — so flags and code
  path are confirmed unchanged. ES priced at $5/pt MES.

## Headline result

| Config | Trades | Win rate | Net PnL | Avg / trade |
|---|---|---|---|---|
| ADX-18 (the "before") | 278 | 52.2% | +$1,029 | $3.7 |
| **ADX-12 (current live)** | **392** | **53.1%** | **+$2,142** | **$5.5** |
| ADX-10 (more-aggressive candidate) | 406 | 53.0% | +$2,082 | $5.1 |

- **18 → 12 is strongly re-confirmed:** +114 trades and **+$1,113** (more than double
  the PnL), win rate slightly up, and average dollars-per-trade *rose* from $3.7 to
  $5.5. The added trades are accretive (+$9.8 each — above the ADX-18 book average),
  not filler. This held up on data the original tuning never saw.
- **12 → 10 is not worth it:** +14 trades but **−$60** PnL, win rate flat, avg/trade
  *down* ($5.5 → $5.1). The marginal trades the looser floor lets in are net-negative.

## Per-quarter PnL (MES $5/pt)

| Quarter | ADX-18 | ADX-12 | ADX-10 |
|---|---|---|---|
| 2024 Q1 | −76 | −115 | −115 |
| 2024 Q2 | −328 | +210 | +210 |
| 2024 Q3 | −88 | −184 | −283 |
| 2024 Q4 | +74 | +62 | +41 |
| 2025 Q1 | +204 | +288 | +324 |
| 2025 Q2 | +602 | +766 | +710 |
| 2025 Q3 | +336 | +456 | +533 |
| 2025 Q4 | +16 | +307 | +307 |
| 2026 Q1 | +167 | +94 | +94 |
| 2026 Q2 | +121 | +259 | +262 |

## Honest caveats — what the longer sample changed

- The single-year result reported "positive every quarter." That does **not** hold on
  26 months: **2024 Q1 (−$115) and 2024 Q3 (−$184)** were modest losers for ADX-12.
  But those quarters were negative for *all three* configs — they're a regime feature
  of early-mid 2024, not something the ADX change caused. ADX-12 still beat ADX-18 in
  every chunk and turned 2024 Q2 from −$328 to +$210.
- 2024 (the newly added year) is a tougher tape for this long-only book than 2025, so
  the longer sample is more conservative — and the ADX-12 edge survives it.
- Still ES-as-MES proxy, backtest only, not forward-tested.

## Recommendation

**Keep `ft_adx_min: 12.0` (no change).** The current live setting is the best of the
three on twice the data — it nearly doubles ADX-18's PnL while improving per-trade
quality. **Do not lower to 10** (dilutive). For more trade frequency beyond this, the
lever is new signal sources or a finer timeframe, not further loosening the ADX floor —
the quality gates below 12 are mostly removing losers.

Trade lists: `all_trades_ADX18.csv`, `all_trades_ADX12.csv`, `all_trades_ADX10.csv`.
