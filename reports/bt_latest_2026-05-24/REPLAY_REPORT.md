# MES Replay — latest code, win/loss (2026-05-24)

**Ask:** look at the RAG entries (MES bot) and replay them with the latest code to see win or loss.

## Key finding first: there were no RAG entries to replay

The RAG / learning stores hold **zero MES trade entries**:

- `data/llm_trades.db` — 0 rows (last touched Nov 2025)
- `rag_data/trades/trade_outcomes.db` — empty (0 bytes)
- `data/learning.db` — 36 `trade_events` + 18 `bucket_stats`, **all `bot='spy_options'`**, no MES
- The only MES RAG content is 115 daily `market_context_*.txt` narrations in `rag_data/docs_dynamic/` — daily context, not trades

This is the live confirmation of the **cold-start deadlock** documented in `docs/rag_authority_review.md` and `docs/expectancy_gate_proposal.md`: the static R:R floor rejected the MES setups, nothing filled, so the learning layer never recorded a single MES trade.

So "replay the RAG entries" was run as the agreed alternative: **replay the MES strategy through the latest code over the full window and score win/loss** (chosen method: full backtest, latest code, Feb 2025–Jan 2026).

## Method

- Engine: `backtest/run.py` — runs the **current live trading logic** (`es_fifteen_min` strategy + trading-manager gates).
- Window: **2025-02-01 → 2026-01-16** (1-minute primary data ends Jan 16 2026).
- Data: `data/ib/ES_1m_1y.parquet` + paired 15m/30m. ES and MES are the same index in points, so signal logic and win/loss are identical; **the engine prices PnL at $5/point (MES multiplier)**, so the dollar figures below are already MES-scale.
- Run in 3 four-month chunks (per-run time limit), then concatenated. Day-aligned boundaries; loss-streak / opening-range state resets intraday, so this is ~equivalent to one run. Verified: 0 duplicate trades, PnL arithmetic exact, every take-profit = win and every stop-loss = loss, trades spread across all 12 months.

## Result — latest code, 166 trades

| Metric | Latest code |
|---|---|
| Trades | 166 |
| Win rate | **50.6%** (84 W / 82 L) |
| Net PnL | **+$66.60** (MES, $5/pt) |
| Profit factor | 1.02 |
| Avg win / avg loss | +$47.8 / −$48.2 |

All trades are **LONG** — shorts are disabled in the current code.

### By signal type

| Signal | n | WR | Net PnL | Avg/trade |
|---|---|---|---|---|
| EMA9_PB_LONG | 13 | **76.9%** | +$348.80 | +$26.83 |
| OR_BREAK_LONG | 5 | 60.0% | +$91.75 | +$18.35 |
| EMA21_PB_LONG | 43 | 53.5% | +$265.55 | +$6.18 |
| TREND_CONT_LONG | 105 | 45.7% | **−$639.50** | −$6.09 |
| **ALL** | **166** | **50.6%** | **+$66.60** | +$0.40 |

### By exit reason

| Exit | n | WR | Net PnL |
|---|---|---|---|
| take_profit | 70 | 100% | +$3,765.75 |
| stop_loss | 76 | 0% | −$3,879.90 |
| flatten (max-hold/EOD) | 20 | 70% | +$180.75 |

## Read

- The book is **net positive but razor-thin** (+$66, PF 1.02). It's carried entirely by the pullback signals.
- **TREND_CONT_LONG is the bleeder**: 63% of all trades (105), sub-coin-flip 45.7% WR, −$640. It dominates volume and drags the whole system to breakeven.
- **EMA9_PB_LONG (77% WR) and OR_BREAK_LONG (60%)** are the cleanest edges, but thin samples (n=13, n=5).
- **EMA21_PB_LONG** is a modest positive (53.5%, +$266).

## Caveats

- Not a literal RAG replay — none existed. This is a full-strategy replay under current code.
- ES 1m used as the MES price series (identical index; PnL already at MES $5/pt).
- Chunk starts carry ~1 day of indicator warmup; effect on ~166 trades is negligible.
- A direct comparison to the old `reports/bt_baseline` (53 trades, 45.3% WR) is **apples-to-oranges**: that was an older "1-minute" strategy on sparse MES-native data. Treat the numbers above as the standalone latest-code result, not a clean before/after.
- `OR_BREAK_LONG` fired in the backtest though it's disabled in the live `config.yaml` — the backtest config may not mirror every live gate. Worth a follow-up if exact live-parity matters.

Files: `all_trades_latest_code.csv`, `summary_by_signal.csv`.
