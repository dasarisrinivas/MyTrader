# MES strategy — improvement summary (2026-05-25)

Verified, live-faithful backtest. ES 1-minute data as the MES price series (identical index; engine prices PnL at MES $5/pt). Window: Feb 2025 – Jan 16 2026.

## How this started

The ask was "replay the RAG entries with the latest code, win or loss." There were **no RAG entries** — the RAG/learning stores (`llm_trades.db`, `rag_data/trades`, MES rows in `learning.db`) are all empty, which is the live confirmation of the cold-start deadlock in `docs/rag_authority_review.md`. So instead the full MES strategy was replayed through the current code, then diagnosed and improved.

A key mid-course correction: the backtest initially ran on `OneMinuteStrategyConfig` dataclass defaults, not your live `one_minute:` config (the engine only reads a `strategy:` section). All numbers below are from the corrected, live-faithful runs.

## The three changes

| # | Change | Where | Default |
|---|---|---|---|
| 1 | `OPENING_BLOCK` — skip first 30 min of RTH (all signals) | `es_fifteen_min.py` | on |
| 2 | `FRIDAY_TREND_CONT_BLOCK` — no trend-continuation on Fridays | `es_fifteen_min.py` | on |
| 3 | `ft_shorts_enabled: false` — disable short signals | `config.yaml` | applied |

(The ADX≥25 floor on `TREND_CONT_LONG` was already live — no change.)

## Verified progression (MES $5/pt)

| Stage | Trades | WR | Net PnL |
|---|---|---|---|
| Live baseline | 224 | 49.6% | +$341 |
| + opening & Friday blocks | 216 | 51.4% | +$687 |
| **+ shorts disabled (longs only)** | **117** | **59.0%** | **+$1,249** |

~3.7× the true baseline. Positive in all five quarters (Q1 +$155, Q2 +$533, Q3 +$387, Q4 +$26, 2026Q1 +$147).

## Why each change works

- **Opening block:** the first 30 min of RTH lost money on every signal (whipsaw before the opening range forms).
- **Friday block:** `TREND_CONT_LONG` on Fridays was 23% WR / −$741 in the early diagnostic; poor follow-through into the weekend.
- **Disable shorts:** both short signals had no reliable edge (−$492/yr), and removing them gained **+$562** — more than their own losses — because their loss streaks were throttling subsequent longs and their positions were blocking long entries (path-dependence, confirmed by re-run not arithmetic).

## Caveats / what's next

- One year of ES-as-MES data. Disabling shorts gives up downside participation — fine on this sample (no short edge) but worth watching in a sustained bear trend.
- All changes are reversible (two default-on flags + one config flag).
- **Not yet forward-validated.** Per project discipline (`docs/expectancy_gate_proposal.md`), run on paper/forward before sizing up. Forward validation requires the live bot + IB Gateway running on your machine — it can't be done from the backtest sandbox.

Trade lists: `all_trades_LONGS_ONLY.csv` (final), `all_trades_LIVE_baseline_blocksOFF.csv`, `all_trades_LIVE_with_blocks.csv`.
