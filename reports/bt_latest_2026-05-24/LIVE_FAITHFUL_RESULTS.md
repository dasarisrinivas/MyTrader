# Live-faithful backtest — corrected before/after (2026-05-25)

## Important correction

My earlier runs (`all_trades_latest_code.csv`, the +$67 → +$1,170 figures) were **not run on your live config**. The backtest's `run_backtest` only reads a `strategy:` section, but your live settings live under `one_minute:` in config.yaml — so the backtest silently used `OneMinuteStrategyConfig` dataclass defaults, which differ from live in ~45 settable keys, including:

- `ft_trend_cont_adx_min`: dataclass 18 vs **live 25** (the ADX≥25 floor is already live — no change needed)
- `ft_or_break_long_enabled`: enabled in the dataclass run vs **disabled live**
- `ft_shorts_enabled`: off in dataclass vs **on live** (live trades shorts; the dataclass run had none)
- entry window 00:00–23:59 vs live 09:30–16:00, take-profit 1.0× vs 2.0×, overnight/ETH on, etc.

These runs inject the live `one_minute` config (dataclass fields via `--config`, getattr-only flags via class attrs). The baseline reproduces ~224 trades, matching the "224-trade backtest" referenced in config.yaml — a fidelity check.

## True before/after (live config, Feb 2025 – Jan 16 2026, MES $5/pt)

| | Before (live, blocks off) | After (live + 2 blocks) |
|---|---|---|
| Trades | 224 | 216 |
| Win rate | 49.6% | 51.4% |
| **Net PnL** | **+$341** | **+$687** |

The two blocks (OPENING_BLOCK + FRIDAY_TREND_CONT_BLOCK) still help — they roughly **double** annual PnL (+$345) with 8 fewer trades. Positive in 4 of 5 quarters after (vs 3 of 5 before).

### By signal (before → after)

| Signal | Before | After |
|---|---|---|
| EMA21_PB_LONG | 68 / 57.4% / +$783 | 71 / 57.7% / +$811 |
| EMA9_PB_LONG | 16 / 75.0% / +$363 | 17 / 70.6% / +$293 |
| TREND_CONT_LONG | 40 / 42.5% / **−$312** | 28 / 53.6% / **+$75** |
| EMA21_PB_SHORT | 81 / 43.2% / **−$227** | 81 / 43.2% / −$227 |
| OR_BREAK_SHORT | 19 / 42.1% / **−$266** | 19 / 42.1% / −$266 |

(Per-signal counts shift slightly because blocking a trade changes downstream state and signal selection — net is 224→216.)

## The real headline: shorts are the biggest drag

| Side | Before | After |
|---|---|---|
| LONG | 124 / +$834 | 116 / **+$1,179** |
| SHORT | 100 / **−$492** | 100 / **−$492** |

The long book makes +$1,179/yr after the blocks; the short book **loses −$492/yr** and neither of my two blocks touches it. `EMA21_PB_SHORT` (−$227) and `OR_BREAK_SHORT` (−$266) are the single biggest remaining opportunity — far larger than any residual long-side tweak. This was invisible in the earlier runs because the dataclass default disabled shorts.

## Recommendation

1. The two blocks are validated under live config — keep them (they're in code, default-on).
2. ADX≥25 floor is already live — nothing to do.
3. **Next target: the short side.** Diagnose `EMA21_PB_SHORT` / `OR_BREAK_SHORT` the same way (hour/DOW/ADX/regime) — fixing or trimming them is worth more than anything left on the long side.

Files: `all_trades_LIVE_baseline_blocksOFF.csv`, `all_trades_LIVE_with_blocks.csv`.
