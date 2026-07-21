# MES Bot Brutal Audit + Signal-Only Surgery — 2026-07-21

Account went Margin→Cash. MES can't trade. Bot converted to **signal-only**.
Same pass = brutal architecture audit. Evidence from logs + DBs, not opinion.

---

## Headline

- Deleted **~61,000 lines / 122 files**. New signal bot = **278 lines**.
- Live MES record (real fills, `data/orders.db`): **235 known-outcome trades, 31.1% win rate, profit factor 0.96, expectancy −$0.55/trade.** Net ≈ breakeven-to-losing.
- Root cause of "never trades": not one gate — a **7-stage funnel** where each stage is fine but the product is near-zero. 1148 live 15m bars → **13 raw signals** (2.2/week) → ~5 survive sentiment+manager veto.
- Biggest fake-complexity offenders deleted: RAG/LLM hybrid pipeline (crashed on EVERY signal with `session_range_pts is not defined` — dead all along), multi-source sentiment (only ever *cut* confidence), 20+ config filter knobs.

---

## Phase 2 — Log forensics (numbers)

`logs/decisions.jsonl` = 437,846 rows, but **99% is backtest replay** (all `es_fifteen_min`, spanning 2024-01→2026-07 with `logged_at` write-times in batches). Live evidence is in `live_trading*.log`.

**Live funnel (Jun 4 – Jul 21, 29 trading days):**

| Stage | Count | Note |
|---|---|---|
| 15m bars evaluated | 1148 | |
| Raw strategy signals (BUY/SELL) | 13 | 0.45/day |
| Sentiment overlay | applied to all | **only ever reduced conf** (−0.21 seen); never the deciding *lift* |
| Manager veto REJECT | 3 of 13 | `Q1_regime, Q2_strategy_fit, Q4_recent_pattern` |
| Hybrid RAG pipeline | **crashed 13/13** | `name 'session_range_pts' is not defined` — 100% dead, silently fell back to HOLD(0.15) |

**Raw-signal blocker ranking (why a bar is HOLD), from 1148 live bars:**

| Gate | Bars blocked | What it checks |
|---|---|---|
| F (stack/adx) | 1148 | EMA stack + ADX<25 |
| C (stack) | 1142 | EMA stack + close vs ema9 |
| D (shorts_disabled / ema) | 1138 | 511 of these = `shorts_disabled` |
| B (no_cross / no_OR) | 1128 | OR breakout cross |
| A (ema21<=ema50) | 1093 | pullback-long structure |
| E (no_cross) | 636 | |

Top single reasons: `B:no_cross` 683, `C:stack` 671, `F:stack` 664, `A:ema21<=ema50` 538, `D:shorts_disabled` 511, `F:adx<25` 389.

**Read:** the strategy is a narrow trend-pullback that only fires when EMA9>EMA21>EMA50 stacked, ADX≥25, price pulls to EMA21, AND not near PDH. In chop (most of Jun-Jul) those never co-occur → structural near-zero.

---

## Phase 9 — Statistical validation (real fills, `data/orders.db` trade_outcomes)

Excluded 85 rows tagged `CORRUPTED_PNL_CUMULATIVE_IBKR` (IBKR fed cumulative not per-trade PnL — a real data-integrity bug) + 85 null-PnL rows.

| Metric | Value |
|---|---|
| Known-outcome trades | 235 (73W / 162L) |
| Win rate | **31.1%** |
| Avg win / avg loss | +$54.15 / −$22.53 (payoff 2.40) |
| Breakeven WR at that payoff | 29.4% |
| Profit factor | **0.96** |
| Expectancy | **−$0.55 / trade** |
| Monthly net (clean) | Dec −$230, Jan +$146, Feb −$64, Mar–May ~0/unknown, Jul +$17 |

**Enough sample? Marginal.** 235 real trades clears minimum viability, but concentrated in Dec-Feb; Mar-Jul is near-zero activity so recent regime is unproven. The system sits *right on* its breakeven line (PF 0.96, WR 31% vs 29% needed) — no demonstrated edge, no demonstrated ruin either. It is a coin-flip with 2.4:1 payoff that barely doesn't pay for itself.

---

## Phase 3/4/5 — Architecture verdicts

| Subsystem | Edge? | Verdict | Evidence |
|---|---|---|---|
| es_fifteen_min strategy | UNKNOWN (breakeven) | **KEEP** | Only live signal source; PF 0.96 |
| feature_engineer (EMA/RSI/ATR/ADX/MACD/PDH) | YES (needed) | **KEEP** | Strategy inputs |
| RAG + LLM hybrid pipeline | NO | **DELETE ✅** | Crashed 13/13 live signals; silent HOLD fallback. Impressive, never worked. |
| Multi-source sentiment (Stocktwits/Reddit/Twitter/VIX) | NO | **DELETE ✅** | Reddit rate-limited, Twitter 0 samples, only ever *reduced* conf. Confidence double-count. |
| VX futures feed | UNKNOWN | **DELETE ✅** | Only fed sentiment multiplier; no isolated edge |
| Trading-manager veto (Q1-Q4 framework) | UNKNOWN | **DELETE ✅ (for MES)** | Rejected 3/13; adds a 2nd opaque gate on top of 7 strategy gates |
| Risk manager / risk_gate / dynamic_support / atr_module | trade-only | **DELETE ✅** | Sizing/stops/brackets — meaningless for signals |
| live_trading_manager (3,780 lines) | trade-only | **DELETE ✅** | Orchestrator for orders/fills/reconcile |
| ib_executor (2,974 lines) | trade-only | **DELETE ✅** | Order placement |
| 1m-era strategies (mes_one_minute, scoring, entry/*, range_reversion) | NO | **DELETE ✅** | Sunset Feb 2026, never removed |
| optimization/optimizer | NO | **DELETE ✅** | Curve-fit params, no walk-forward validation on file |
| monitoring/order_tracker, live_tracker | trade-only | **DELETE ✅** | |
| backtest/runner | broken | **DELETE ✅** | Imported nonexistent `shree.agents` |

---

## Phase 6 — Why it never traded (exact)

1. **Filters are stacked AND correlated.** Gates A/C/F all re-check the EMA9>EMA21>EMA50 stack (`C:stack` 671, `F:stack` 664 — same condition counted 3×). That's not 7 independent filters; it's ~3 real conditions triple-gated.
2. **ADX≥25 + stacked EMAs + EMA21 touch + not-near-PDH must all co-occur.** In a rangebound tape (Jun-Jul VIX ~17) those are near mutually exclusive.
3. **Confidence double-counts.** Base 0.70 from strategy, then sentiment overlay subtracts (never the reason to fire), then a broken RAG pipeline was supposed to add but crashed to HOLD(0.15). Architecture fought itself.
4. **A second veto layer** (trading-manager Q1-Q4) on top of the 7 strategy gates.
5. **shorts_disabled** blocked the D-family on 511 bars — half the tape was structurally un-tradeable by config.

---

## Phase 7 — Would I build this today? **NO.**

One narrow strategy carried a stack of RAG/LLM/sentiment/optimizer/veto machinery that (a) never demonstrated edge and (b) literally crashed in production undetected. The signal came from ~200 lines of EMA logic; ~60k lines were scaffolding.

**Better shape (what I built):** market data → indicators → one strategy → signal record → log + notify. 278 lines. Observable (every field logged), debuggable (pure `evaluate()` unit-tested offline), no broker coupling.

---

## Phase 8 — Signal-only bot (shipped, running live)

`shree/signal_bot/bot.py` (`MesSignalBot`). IB = **market-data feed only**. Per 15m bar → `logs/mes_signals.jsonl` + Telegram on BUY/SELL. Every record carries: `signal, confidence, entry, stop, target, expected_r, regime, adx, atr, htf_30m_trend, supporting_evidence, blocking_evidence`.

Verified live: pid running under launchd, emitting `HOLD | TRANSITIONAL ADX=22.4 ATR=5.46 htf=UP`. Zero order APIs on the object (test-enforced). Gold + SPY + trading-manager bots untouched; all 317 non-deleted tests green + 4 new signal-bot tests.

---

## Phase 10 — Final verdict

**A. Delete immediately (DONE):** execution stack, RAG/LLM, sentiment, VX feed, optimizer, 1m strategies, broken backtest runner, monitoring order-trackers. ~61k lines gone.

**B. Simplify:** the es_fifteen_min gate set — collapse A/C/F (they re-test the same EMA stack) into one stack check + one momentum check. ~40 config knobs → ~8. Kill the PDH-proximity + overnight-MACD micro-filters until one shows edge.

**C. Untouched:** es_fifteen_min core, feature_engineer, telegram, spy_options, gold, trading_manager.

**D. Biggest mistakes:** (1) shipping a RAG/LLM pipeline that crashed on 100% of live signals with nobody noticing — no signal-count alarm. (2) IBKR cumulative-PnL corruption silently poisoning outcomes. (3) confidence that subtracts more than it adds.

**E. Missed opportunities:** never measured per-gate edge (which filter actually improves WR?); never walk-forward validated; logged everything but *alerted* on nothing.

**F. One-month rebuild = what shipped today** + a nightly job that replays `mes_signals.jsonl` against next-bar outcomes to score each gate's marginal edge. Let evidence delete the next filter.

**G. Roadmap:**
- **CRITICAL (done):** remove order placement; ship signal-only bot; verify live.
- **HIGH:** signal-outcome scorer (did HOLD-vs-fire have been right?); alert on N-day zero-signal droughts.
- **MEDIUM:** collapse correlated gates; prune config knobs; re-validate shorts side.
- **LOW:** revisit whether a 2nd, uncorrelated strategy (e.g. mean-reversion for chop) lifts frequency without wrecking the 2.4 payoff.
