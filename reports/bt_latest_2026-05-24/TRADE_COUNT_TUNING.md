# Increasing trade frequency — what to tune (2026-05-25)

Goal: more trades without giving back the edge. Live-faithful backtest, longs-only (shorts already disabled), ES 1m as MES proxy, Feb 2025 – Jan 16 2026, MES $5/pt.

## Data note

Longest 1-minute data on disk is `ES_1m_1y.parquet` (Dec 2024 – Jan 16 2026, ~13 months) — that's the max backtest span. No multi-year 1m data is present and there are no databento/polygon keys in this environment, so **I can't download more years from here.** To extend: with your IB Gateway running you can pull more on your Mac via `python -m backtest.run --symbol MES --bar 1m --data-source ib --start <date> --end <date>` (IB caps 1m history, so multi-year may be partial), or use a paid source (databento). Drop the file in `data/ib/` and I can backtest it.

## What suppresses trades — and the fix

Sensitivity test (one gate at a time):

| Lever | Trades vs base | PnL effect | Verdict |
|---|---|---|---|
| `ft_adx_min` 18→14 | +more | +PnL, WR flat | good |
| **`ft_adx_min` 18→12** | **+more** | **+PnL, WR flat** | **best** |
| `ft_adx_min` 18→10 | +more | +PnL (summer only) | aggressive, not full-validated |
| `ft_trend_cont_adx_min` 25→20 | +6 | PnL collapses (−$379) | reject — keep 25 |
| re-enable `OR_BREAK_LONG` | +5 | ~+$27 | marginal |
| re-enable shorts | +100 | −PnL | reject — shorts have no edge |

The binding constraint was the pullback ADX floor (`ft_adx_min`), not a lack of setups. ADX≥18 was over-filtering `EMA21_PB_LONG`.

## Validated result — `ft_adx_min: 12` (full span, longs-only)

| Config | Trades | WR | PnL |
|---|---|---|---|
| ADX≥18 (was) | 117 | 59.0% | +$1,249 |
| ADX≥14 | 154 | 59.1% | +$1,617 |
| **ADX≥12 (applied)** | **168** | **58.9%** | **+$1,977** |

- **+51 trades (+44%) and +$728 (+58%)**, win rate unchanged.
- Positive every quarter (Q1 +$80, Q2 +$700, Q3 +$627, Q4 +$317, 2026Q1 +$253).
- The 51 added trades averaged **+$14/trade — better than the existing book** (+$11), so they're accretive, not filler.
- Added volume flows into `EMA21_PB_LONG` (68→125 trades, still 58% WR). `TREND_CONT_LONG` unchanged (its own ADX≥25 floor stays).

## Applied

`config.yaml`: `ft_adx_min: 18.0 → 12.0`. (TREND_CONT keeps `ft_trend_cont_adx_min: 25` — lowering it was tested and is clearly bad.)

## Full progression of all changes (live-faithful, longs-only)

| Stage | Trades | WR | PnL |
|---|---|---|---|
| Original live baseline (shorts on, ADX 18) | 224 | 49.6% | +$341 |
| + opening & Friday blocks | 216 | 51.4% | +$687 |
| + shorts disabled | 117 | 59.0% | +$1,249 |
| **+ ADX-min 12 (current)** | **168** | **58.9%** | **+$1,977** |

## Caveats / next

- Backtest only (~13 months, one ES-as-MES proxy). Not forward-tested. Reversible config flag.
- `ft_adx_min: 10` tested higher in the summer window but isn't full-validated and lowers the trend filter further (regime risk) — worth a look only with more data.
- More trade frequency would mainly come from **new signal sources or a finer timeframe**, not more loosening — the quality gates beyond ADX are mostly removing losers.
