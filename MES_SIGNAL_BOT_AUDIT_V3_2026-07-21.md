# MES Bot — Independent Audit V3 (attacks V1 AND V2) — 2026-07-21

V3 distrusts the previous two audits — including their headline "RSI wins" —
and re-tests with methods V2 skipped: variance-ratio market-structure test,
cost-adjusted returns, a random-entry null, and **non-overlapping / out-of-sample**
significance. Result: **both prior audits were overconfident.** Corrections below.

Data: `data/orders.db` fills; a deduped 33,998-bar clean 15m series
(2024→2026) from `logs/decisions.jsonl`. Cost model: 0.65 pt round-trip
(~1 tick slippage/side + commission).

---

## Phase 1 — Verify every prior finding

| # | Prior claim | Verdict | Evidence |
|---|---|---|---|
| 1 | Native fills lose money | **VERIFIED** | 114 native fills: 25.4% WR, −$497 |
| 2 | PF inflated by reconstructed BACKFILL | **VERIFIED** | BACKFILL 113 rows +$367 vs native −$497 |
| 3 | Massive over-engineering | **VERIFIED** | ~61k lines deleted, bot works at 278 |
| 4 | Filters correlated / re-check same thing | **VERIFIED** | gates A/C/F all test EMA9>21>50 stack |
| 5 | Confidence double-counts | **VERIFIED** | base 0.70 → sentiment only subtracts → dead RAG add |
| 6 | RAG/LLM crashed in prod | **VERIFIED** | `session_range_pts` NameError, 13/13 live signals |
| 7 | Sentiment only reduced confidence | **VERIFIED** | log overlays all negative/zero |
| 8 | Too many correlated filters → near-zero trades | **VERIFIED** | 1148 live bars → 13 raw signals |
| 9 | **RSI significantly beats the strategy** | **PARTIALLY — OVERCLAIMED** | RSI beats the bot, but is **not** a proven forward edge (see Phase 3/10) |
| 10 | ~120 params, post-hoc tweaks | **VERIFIED** | 120 `ft_` knobs, 63 dated tweaks |
| 11 | Optimized for past losses (survivorship) | **VERIFIED** | filters killed on 0/5-WR weeks |

**The one correction that matters: finding #9.** V2 said RSI mean-reversion
"significantly outperforms." Independent re-test says the *direction* (reversion)
is structurally right, but the *specific RSI rule* is **not statistically proven
and died out-of-sample in 2026.** V2 mistook an in-sample, overlapping-window,
gross-of-cost result for a robust edge.

---

## Phase 2 — Performance, recomputed (real fills only)

| Population | n | WR | Net $ | Trust |
|---|---|---|---|---|
| Native (STOP_LOSS/PROFIT_TARGET) | 114 | 25.4% | **−497.38** | high |
| Reconstructed (BACKFILL) | 113 | 31.9% | +367.16 | low (after-the-fact) |
| Clean total | 227 | 28.6% | −130.22 | mixed |

**Reconstruction bias quantified:** BACKFILL rows swing the aggregate by
**+$864** vs native (from −$497 to +$367). Trusting only native fills, the
system is a clear loser. Sample is regime-skewed (Dec-Feb heavy, Mar-Jul thin) —
**not enough recent trustworthy data** to claim the config works today.

---

## Phase 5 — Market structure (what MES 15m actually is)

Variance-ratio test on 33,998 bars (VR<1 = mean-reverting, =1 random, >1 trending):

| q (bars) | VR |
|---|---|
| 2 | 0.409 |
| 4 | 0.286 |
| 6 | 0.228 |
| 12 | 0.182 |

**Decisively mean-reverting**, and more so at longer horizons. Lag-1
autocorrelation −0.59 (treat as inflated by bid/ask bounce + replay-series
stitching — the VR curve is the robust evidence, not that single lag).

**Conclusion: the bot's trend-following thesis (EMA21 pullback, OR breakout,
trend continuation) fights the instrument.** This is the strongest, most
method-independent finding in all three audits, and it does NOT depend on the
RSI backtest.

---

## Phase 3 / 4 / 10 — Does ANY strategy have a proven edge? (aggressive challenge)

Same bars, enter at close, K=6 exit, ATR-unit returns.

**Gross (V2's method):**

| System | n | WR | meanR | PF |
|---|---|---|---|---|
| RSI<30/>70 revert | 4495 | 51.5% | +0.236 | 1.32 |
| BOT es_fifteen_min | 829 | 51.7% | +0.038 | 1.05 |
| Trend rules (EMA/ORB/Donchian/Mom) | — | 40–47% | negative | 0.33–0.76 |

**Net of costs + significance (V3's method):**

| Test | RSI | BOT |
|---|---|---|
| Net meanR (K=6) | +0.135 | **−0.041** |
| Net PF | 1.17 | **0.95** |
| Naive t (overlapping) | 3.19 | −0.37 |
| **Non-overlapping t** | **1.84** | — |
| vs cost-matched random null | **z = +1.2** | — |

**Out-of-sample, non-overlapping, net:**

| Year | RSI n | PF | t |
|---|---|---|---|
| 2025 | 1157 | 1.27 | **2.71** (real) |
| **2026** | 301 | **0.93** | **−0.41 (DEAD)** |

**Verdicts:**
- **The bot has no edge net of costs.** PF 0.95, t≈0. Not different from a loser. **Do not trade it.**
- **The RSI rule beats the bot but is not a proven forward edge.** Full-sample non-overlapping t=1.84 (below 95%), only z=+1.2 over a cost-matched random null, and it **decayed to PF 0.93 in 2026 out-of-sample.** It worked in 2025 (t=2.71) then stopped. That is either regime change or 2025 was partly noise.
- The V2 bracket-exit number (PF 1.67, t=16) is **rejected** — close-only data can't see intrabar stop-outs, so it survivorship-inflates winners. Distrust it more, not less.

**Bottom line: no strategy in this repo has a demonstrated, cost-surviving,
out-of-sample edge on MES 15m today.** Mean-reversion is the right *research
direction* (VR<1); the specific rule is unproven. This is precisely why
**signal-only (log, score, trade nothing) is the correct posture** — we have no
edge worth risking capital on, so the job is to *find* one under measurement,
not to deploy one on faith.

---

## Phase 6 — Architecture scorecard (current state)

| Subsystem | Cmplx | Live value | Stat evidence | Verdict |
|---|---|---|---|---|
| signal_bot (new, 278 ln) | 2 | emits signals | n/a | **KEEP** |
| es_fifteen_min strategy | 9 | PF 0.95 net | none | **REBUILD** (wrong side) |
| feature_engineer | 4 | inputs | needed | **KEEP** |
| Execution stack / RAG / LLM / sentiment / VX / optimizer / manager-veto / risk | 7–10 | none | none | **DELETE** ✅ done |
| 1m-era strategies | 8 | none | none | **DELETE** ✅ done |
| 120-knob config | 9 | negative (overfit) | none | **SIMPLIFY** → ≤10 |

Signal path verified clean: no `placeOrder`/`Trade(`/`reconcile`/`ib_executor`
anywhere in `shree/signal_bot/` or `run_bot.py`.

---

## Phase 7 / 8 — Fake complexity + survivorship (verified guilty)

- 120 `ft_` params for one strategy; 1,351-line config.
- 63 dated post-hoc tweaks (`Fix #N`, `Lowered X→Y`, MAR-3/9/16 …) — reactions to individual days.
- Filters killed on n=5: `ft_or_break_long_enabled: false # 0/5 WR −$197`, `ft_trend_cont_short_enabled: false # 0/5 WR −$304`. Noise-fitting.
- Fake precision: confidence to 3 decimals (0.700/0.770) with no distribution; weights nudged (`0.10→0.15`, `0.4→0.55`) without validation.
- **Survivorship confirmed:** the config is the one that would have dodged past losses — the exact thing that fails forward.

---

## Phase 9 — Greenfield (≤2 strategies, ≤10 params, no LLM/RAG/sentiment)

```
Data: IB 15m MES (feed only) + 30m ADX for regime.
Strategy 1 (primary): mean-reversion — structurally supported (VR 0.18–0.41).
    LONG  RSI(14) < rsi_lo  and close ≥ lower_band (e.g. VWAP−k·ATR or prior swing low)
    SHORT RSI(14) > rsi_hi  and close ≤ upper_band
    exit: target = tgt·ATR, stop = stop·ATR, or max_hold bars.
Strategy 2 (optional): OR-fade — fade a failed break of OR_high/OR_low (also reversion).
Regime gate (ONE): disable Strategy 1 when 30m ADX > adx_off (strong trend).
Confidence: 2 buckets from |RSI−50| distance. No engine, no decimals.
Params (10): rsi_len, rsi_lo, rsi_hi, atr_len, stop_mult, tgt_mult,
    band_k, max_hold, adx_off, min_atr.
Output: BUY/SELL/HOLD + confidence, entry, stop, target, expected_R, regime,
    supporting/blocking evidence → logs/mes_signals.jsonl + Telegram.
```

**Honest caveat vs V2:** greenfield trades the structurally-correct *direction*,
but V3 shows even RSI is unproven forward. So greenfield ships **log-only** and
earns the right to trade via the Phase 11 validation gate — never on faith.

---

## Phase 11 — Permanent validation framework (the real deliverable)

Because no edge is proven, the bot's #1 job is to *measure* itself. Nightly job:

```
For each emitted signal in logs/mes_signals.jsonl older than max_hold:
    fetch actual next-K-bar outcome, compute realized R (net of a cost constant).
    append to logs/signal_scorecard.jsonl.
Roll up by: strategy · regime · confidence bucket · time-of-day · month.
Report per bucket: n, WR, PF, expectancy, rolling-60-trade PF.
ALERT when: rolling PF crosses below 1.0, or a bucket's t-stat degrades,
    or zero signals for N days.
Promotion rule: a strategy may go from log-only to "tradeable" ONLY after
    ≥100 non-overlapping signals at out-of-sample PF > 1.15 net of costs.
```

This turns the 2026-decay problem (which killed the RSI edge) into an automatic
alarm instead of a thing discovered by a manual audit a year later.

---

## Phase 12 — Roadmap

| # | Task | Effort | Risk | Expected | Validation | Commit |
|---|---|---|---|---|---|---|
| 1 | Nightly signal-scorer + scorecard.jsonl | 3h | low | self-measurement, decay alarm | replays own output | `feat(signal): nightly outcome scorer` |
| 2 | Add mean-reversion strategy as a 2nd **log-only** signal stream | 4h | low | trades correct side (VR<1) | Phase 11 gate | `feat(signal): mean-reversion stream (log-only)` |
| 3 | Collapse config to ≤10 params; delete 63 post-hoc tweaks | 3h | med | removes overfit surface | walk-forward PF must hold | `refactor(config): 120→10 params` |
| 4 | Promote a strategy only if OOS PF>1.15 over 100 non-overlap signals | — | low | trade only proven edge | scorecard gate | — |

### 30 / 60 / 90 day

- **30:** ship nightly scorer; mean-reversion added log-only; both streams scored side-by-side. Prune config.
- **60:** 100+ non-overlapping OOS signals per stream collected; compare net PF; kill whichever is < 1.0.
- **90:** if either stream clears PF>1.15 OOS, mark tradeable (still signal-only bot — "tradeable" = worth a human/paper acting on). If neither clears, **keep trading nothing** and keep searching. No edge → no deployment. That is the correct outcome, not a failure.

---

## Final Caveman Verdict

Three audit now. Each one too proud. V1 kind to bot. V2 kind to RSI. Truth colder:

- Bot lose money net of cost. PF 0.95. Dead. **DELETE the trend thesis.**
- MES 15m bounce, not run. Variance-ratio prove it (0.18–0.41). **Solid rock.**
- RSI point right way but not proven — die in 2026 out-of-sample (PF 0.93). **Not faith-worthy yet.**
- No strategy here earn real edge today. So: **trade nothing. Log everything. Score every night. Trade only what forward-prove itself.**

Signal-only not a downgrade. Signal-only is the honest posture when no edge proven. Machine job now = find edge under measurement, not pretend one exist.

KEEP: signal_bot, feature_engineer, logging.
SIMPLIFY: config → 10 knob.
DELETE: execution/RAG/LLM/sentiment/optimizer (done).
REBUILD: strategy → mean-reversion, log-only, earn trade-right by validation.
