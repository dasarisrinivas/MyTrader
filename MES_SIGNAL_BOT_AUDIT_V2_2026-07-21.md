# MES Bot — Brutal Audit V2 (independent re-verification) — 2026-07-21

V2 re-runs the audit from scratch and **checks V1 against evidence, including
V1's own numbers.** Where V1 was wrong, it is corrected here. New analyses V1
skipped: real comparison vs simple systems, fake-precision audit, survivorship
audit, greenfield rebuild.

Method: live fills from `data/orders.db`; a deduped 33,997-bar clean 15m series
(2024-01→2026-07) rebuilt from `logs/decisions.jsonl` for edge testing;
`config.yaml` git history for survivorship.

---

## 0. Corrections to V1 (evidence beats the last audit)

| V1 claim | Truth | Verdict on V1 |
|---|---|---|
| "85 CORRUPTED_PNL rows excluded" | Only **8** rows tagged corrupt (~$17k each). The 85 were **null-PnL**, a different bucket. | **Wrong** — mislabeled. Right to exclude them, wrong reason/count. |
| "235 known-outcome trades, breakeven PF 0.96" | Clean n=**227**, WR **28.6%**, PF **0.964**, exp −$0.57. Correct in spirit. | Roughly right. |
| "no edge, no ruin, breakeven" | **Native IBKR fills alone (114): 25.4% WR, −$497.38, clearly losing.** The +PF is propped by 113 **reconstructed BACKFILL** rows (+$367). | **V1 understated it.** Real fills lose money; reconstruction bias flatters the aggregate. |
| Didn't compare to simple systems | A one-line RSI rule (PF 1.32) **beats** the whole bot (PF 1.05). | **V1's biggest miss.** |

V1's delete-list and surgery were correct. V1's *performance read was too kind* and it never asked whether the strategy is even on the right side of the edge.

---

## Phase 2/6 — Why it never trades (verified)

1148 live 15m bars → 13 raw signals → ~5 survive. Confirmed. Root causes, ranked by live-bar block frequency:

| Gate | Bars blocked | Real condition |
|---|---|---|
| F (stack + adx<25) | 1148 | EMA9>EMA21>EMA50 AND ADX≥25 |
| C (stack) | 1142 | same stack, re-checked |
| D (shorts_disabled/ema) | 1138 | 511 = shorts config-disabled |
| B (no OR cross) | 1128 | |
| A (ema21<=ema50) | 1093 | |

**Filters are correlated, not independent.** `C:stack` (671) and `F:stack` (664) test the *same* EMA condition already in A. "7 quality gates" ≈ 3 real conditions, gated 2–3×. Confidence also double-counted: base 0.70 → sentiment only ever subtracted → RAG add was dead (crashed 13/13, `session_range_pts` NameError). Architecture fought itself.

---

## Phase 3 — Live performance (corrected, real fills only)

| Population | n | WR | Net $ | Read |
|---|---|---|---|---|
| Native fills (STOP_LOSS/PROFIT_TARGET) | 114 | **25.4%** | **−497.38** | The only trustworthy set. Loses. |
| Reconstructed (BACKFILL) | 113 | 31.9% | +367.16 | After-the-fact, optimistic. Suspect. |
| Clean total | 227 | 28.6% | −130.22 | PF 0.96, exp −$0.57 |

**Sample size: barely adequate (114 native), regime-skewed** (Dec-Feb heavy, Mar-Jul near-zero). Not enough recent data to claim the current config works today. Verdict: **no demonstrated edge; the trustworthy subset loses.**

---

## Phase 9 — Bot vs simple systems (THE headline, V1 never ran it)

33,997 clean bars. Enter at close, exit +6 bars (bot's ~90min max hold), return in ATR units. No costs/intrabar (directional-edge test). Same bars for all:

| System | n | WR | mean R | PF |
|---|---|---|---|---|
| **RSI<30/>70 mean-reversion** | 4494 | 51.5% | **+0.236** | **1.32** |
| **BOT es_fifteen_min (its own signals)** | 829 | 51.7% | +0.038 | **1.05** |
| Momentum(6) | 33598 | 46.9% | −0.260 | 0.76 |
| EMA9/21 cross | 5198 | 41.0% | −1.105 | 0.36 |
| ORB breakout | 3355 | 40.0% | −0.775 | 0.42 |
| Donchian-20 | 6373 | 39.6% | −1.227 | 0.33 |

**Every trend/breakout rule is a structural loser on MES 15m** (PF 0.33–0.48). The bot's entire thesis — EMA21 pullback, OR breakout, trend continuation — is trend-following. It reaches PF 1.05 **only** by filtering ~98% of trend setups away. **60k lines of machinery to grind a losing edge up to breakeven.**

Meanwhile a **one-line RSI mean-reversion rule scores PF 1.32 on 5× the frequency.** The instrument mean-reverts intraday; the bot trades the opposite side.

**RSI edge robustness (not a fluke):**
- Horizons K=2/4/6/8/12 → PF 1.79/1.61/1.32/1.50/1.52 (always positive).
- Both sides: long RSI<30 PF 1.33, short RSI>70 PF 1.31 (symmetric = real mean-reversion).
- By year: 2025 PF 1.45 (n=3341), 2026 PF 1.10 (n=998, decaying), 2024 thin/negative (n=156).

Caveats (stated, not hidden): close-to-close, no commissions (~0.02R on MES, immaterial vs the gap), fixed-bar exit. The **1.32-vs-0.36 gap** is far too large to be a methodology artifact, and index-futures intraday mean-reversion is well documented.

---

## Phase 10 — Fake precision / magic numbers

- `config.yaml` = 1,351 lines; **120 `ft_` knobs for one strategy.**
- `counter_trend_penalty: 0.15 (was 0.10)`, `stocktwits_weight: 0.55 (was 0.4)`, `low_volume_confidence_penalty: 0.95 (was 0.90)` — none statistically justified; all hand-nudged.
- `min_confidence_for_trade: 0.40` with a comment explaining it catches "0.55–0.60" stacks — a threshold reverse-engineered to let specific past signals through.
- Confidence reported to 0.700 / 0.770 — 3 decimals on a number with no distribution behind it. Fake precision.

---

## Phase 11 — Survivorship / curve-fitting (guilty)

- **63 dated post-hoc tweaks** in config (`Fix #N`, `Lowered from X→Y`, `Relaxed from`, dated MAR 3 / MAR 9 / MAR 16 …) — parameters moved in reaction to individual losing days.
- Filters **disabled after a single bad week**: `ft_or_break_long_enabled: false # 0/5 WR, −$197`, `ft_trend_cont_short_enabled: false # 0/5 WR, −$304`. Killing a strategy on n=5 is noise-fitting, not evidence.
- The config is **optimized for the past, not the future** — textbook survivorship. Every loss spawned a new filter; the surviving config is the one that would have dodged history, which is exactly what fails forward.

---

## Phase 4 — Architecture scorecard

| Subsystem | Complexity | Live edge | Replay edge | Stat confidence | Verdict |
|---|---|---|---|---|---|
| es_fifteen_min strategy | 9 | breakeven→losing | PF 1.05 | low | **REBUILD** (wrong side of edge) |
| feature_engineer (EMA/RSI/ATR/ADX/MACD/PDH) | 4 | n/a (inputs) | needed | — | **KEEP** |
| RAG + LLM hybrid | 10 | crashed 13/13 | none | none | **DELETE ✅** |
| Multi-source sentiment | 8 | only subtracts | none | none | **DELETE ✅** |
| VX futures feed | 5 | none isolated | none | none | **DELETE ✅** |
| Manager veto (Q1–Q4) | 6 | 3/13 reject | none | none | **DELETE (for MES) ✅** |
| Risk mgr/gate/dyn_support/atr/validator | 7 | trade-only | n/a | — | **DELETE ✅** |
| live_trading_manager / ib_executor / components | 10 | trade-only | n/a | — | **DELETE ✅** |
| 1m strategies / scoring / entry/* | 8 | sunset | none | none | **DELETE ✅** |
| optimizer | 6 | none | curve-fit | none | **DELETE ✅** |
| 120-knob config | 9 | negative (survivorship) | — | none | **SIMPLIFY → ≤10** |

---

## Phase 8 & 13 — Would I build this today? **NO. REBUILD.**

The strategy trades filtered trend-following on an instrument that mean-reverts on 15m. Evidence says flip sides.

**Greenfield MES signal bot (≤2 strategies, ≤10 params, no LLM/RAG/sentiment/adaptive):**

```
Data: IB 15m MES bars (data feed only) + 30m regime EMA.
Strategy 1 (primary): RSI(14) mean-reversion.
    LONG  when RSI < 30  and close within 1.0*ATR of a prior swing low
    SHORT when RSI > 70  and close within 1.0*ATR of a prior swing high
    stop 1.0*ATR, target 1.0*ATR (K≈6-bar time-exit fallback)
Strategy 2 (optional): fade the OR — SHORT a failed break above OR_high,
    LONG a failed break below OR_low. (Same mean-reversion family.)
Regime filter (single): skip Strategy 1 when 30m ADX > 30 (strong trend =
    mean-reversion off). One filter, not seven.
Confidence: monotonic in |RSI-50| distance past threshold, 2 buckets
    (MEDIUM/HIGH). No engine, no decimals-of-decimals.
Params (10): rsi_len, rsi_lo, rsi_hi, atr_len, atr_stop_mult, atr_target_mult,
    swing_lookback, max_hold_bars, htf_adx_off, min_atr.
Output: BUY/SELL/HOLD + confidence, entry, stop, target, expected_R, regime,
    supporting/blocking evidence → logs/mes_signals.jsonl + Telegram.
Validation: nightly replay of emitted signals vs next-bar outcome; walk-forward
    2025→2026, keep only if out-of-sample PF > 1.15 net of costs.
```

**Why it should beat the incumbent over 12 months:** it trades the side of the edge the data actually shows (PF 1.32 vs the incumbent's 1.05), with 5× the signal frequency, 10 params instead of 120 (far less overfit surface), and a built-in validation loop so decay (2026 PF already down to 1.10) is caught, not ignored.

---

## Final verdict — KEEP / SIMPLIFY / DELETE / REBUILD

| Subsystem | Verdict |
|---|---|
| Execution stack (mgr/executor/components/reconcile/risk) | **DELETE** ✅ done |
| RAG / LLM / sentiment / VX / optimizer / manager-veto | **DELETE** ✅ done |
| 1m-era strategies | **DELETE** ✅ done |
| 120-knob config | **SIMPLIFY** → ≤10 params |
| feature_engineer, telegram, logging, signal_bot shell | **KEEP** |
| es_fifteen_min core thesis (trend-following) | **REBUILD** → RSI mean-reversion |

## Highest-ROI changes (Phase 14)

| # | Change | Hrs | Risk | Expected | Validation |
|---|---|---|---|---|---|
| 1 | Add RSI mean-reversion strategy alongside es_fifteen_min in signal_bot; log both, trade neither's orders | 4 | low | +0.20R/signal, 5× frequency | walk-forward 2025→2026 net of costs |
| 2 | Nightly signal-outcome scorer (emitted signal vs next-6-bar) | 3 | low | catches decay (2026 already 1.10) | self-measuring |
| 3 | Collapse config to ≤10 params; delete 63 post-hoc tweaks | 3 | med | removes overfit surface | re-run walk-forward, PF must hold |
| 4 | Retire es_fifteen_min if RSI wins out-of-sample 60 days | 1 | low | stop trading the losing side | 60-day live A/B in signal log |

## 90-day roadmap

- **Days 0–15:** ship RSI mean-reversion as a *second* signal stream in the signal bot (log-only). Nightly scorer live.
- **Days 15–45:** walk-forward RSI vs es_fifteen_min on 2025→2026, net of costs. Prune config to ≤10 params.
- **Days 45–90:** if RSI out-of-sample PF > 1.15 and beats incumbent in the live signal log, make it primary; demote es_fifteen_min. Keep only what the scorer proves.

## Caveman verdict

Bot fight the tape. Tape bounce, bot chase. 60k line to make loser look flat. One-line RSI beat whole machine. Delete done. Now flip side: trade bounce, not chase. Ten knob, not one-twenty. Measure every night. Keep only what pay.
