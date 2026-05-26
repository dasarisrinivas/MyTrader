# Short-side diagnostic + the longs-only result (2026-05-25)

Live-faithful data. The short book loses −$492/yr (EMA21_PB_SHORT −$227 / 81 trades, OR_BREAK_SHORT −$266 / 19 trades).

## Why the shorts lose

`EMA21_PB_SHORT` (n=81) — losses concentrate in low trend strength and the mid-day chop:

| ADX | n | WR | PnL |
|---|---|---|---|
| 18–22 | 16 | 18.8% | −$281 |
| 22–27 | 15 | 46.7% | −$25 |
| 27+ | 50 | 50.0% | +$79 |

Only the high-ADX slice (≥27) is positive, and even then barely (+$79). Also bad: Fridays (−$209), 11 AM CT (−$354), ATR 6–8 (−$394).

`OR_BREAK_SHORT` (n=19) — too thin to rescue: net −$266, no stable positive pocket (the one "good" ADX bucket is 3 trades).

## The bigger effect: shorts damage the longs

Disabling shorts doesn't just remove their −$492 — it stops them triggering loss-streak throttles and occupying position slots that suppress good long entries. Verified by re-running with `ft_shorts_enabled: false`:

| Book (verified, live config, MES $5/pt) | Trades | WR | Net PnL |
|---|---|---|---|
| Live baseline (shorts on, no blocks) | 224 | 49.6% | +$341 |
| + 2 blocks (opening + Friday TREND_CONT) | 216 | 51.4% | +$687 |
| **+ disable shorts (longs only)** | **117** | **59.0%** | **+$1,249** |

Disabling shorts adds **+$562** (more than the arithmetic −$492, because of the state-damage effect) and halves trade count. Positive in **all five quarters** (Q1 +$155, Q2 +$533, Q3 +$387, Q4 +$26, 2026Q1 +$147) — Q4 flips from −$124 to positive.

## Recommendation

1. **Disable shorts: set `one_minute.ft_shorts_enabled: false` in config.yaml.** One-line, reversible. Single biggest lever found: +$687 → +$1,249, 59% WR, positive every quarter.
2. If you want to keep *some* short exposure, the only defensible slice is `EMA21_PB_SHORT` at ADX ≥ 27 (+$79) and drop `OR_BREAK_SHORT` — but that adds code complexity for ~$80 and is less robust than just turning shorts off. Not recommended over option 1.

### Caveats
- One year of ES-as-MES data. Disabling shorts gives up downside participation; in a sustained bear trend shorts could matter. On this sample they have no edge and actively harm.
- Reversible config flag — validate forward/paper before committing live, per project discipline.

Files: `all_trades_LONGS_ONLY.csv`.
