# SPY Options Bot — Full System Audit (2026-07-13)

**Method:** 52-agent review — 10 code auditors (each read its subsystem in full), adversarial verification of every critical/high finding (24/25 CONFIRMED), 5 independent backtests on 60d of 5-min data, and a completeness critic that re-verified 5 flagship findings directly against the live repo/DB. Read-only; no IB connection; no live orders.

**Headline:** The bot's two real problems are **(1) whole signal families that can never execute, and (2) live-account safety machinery that silently does not work.** The strategy/confidence layer is a *secondary* concern — the backtests show most of it is statistically inert (confidence score does not predict win rate; several "signals" are dead columns). **Do not tune the strategy until the safety + plumbing bugs are fixed.**

> Data caveat that governs every P&L statement below: only **3 of 216** signal rows have real fills; `outcome`/`pnl_pct` are a *simulated shadow-exit monitor*. n=24 "decided" outcomes underlies the win-rate figures — nothing survives strict multiple-testing correction. Treat WR numbers as *signal quality under an idealized exit*, not realized edge.

---

## PRIORITY-RANKED ROADMAP

### P0 — Live-capital safety (fix BEFORE the bot trades live again)
| # | Finding | File:line | Why it's P0 |
|---|---|---|---|
| 1 | **Reconnect orphans the live bracket** — reconnect never rebinds `LivePosition`'s Trade objects | [executor.py:195](shree/spy_options/executor.py:195) | After the nightly Gateway restart (or any blip) an open position's TP/SL is no longer tracked — unmanaged live risk |
| 2 | **Keepalive never runs** — `reqCurrentTime()` called from inside an async coroutine always raises, swallowed at DEBUG | [ib_client.py:123](shree/spy_options/ib_client.py:123) / [executor.py:179](shree/spy_options/executor.py:179) | The liveness check that's supposed to trigger reconnect has never once completed |
| 3 | **TM kill-switch fails OPEN** — executor waits 2.5s for the TM daemon's JSONL verdict; on timeout it **proceeds to trade**, and `_gate()` has no local posture check | [manager.py:2586](shree/spy_options/manager.py:2586) | POSTURE_KILLED / LOCKED / daily-loss stop are all disabled exactly when the daemon is down. Matches your prior MES incident pattern |
| 4 | **Rejected entry orders leak a position slot** — never marked closed | [executor.py:560](shree/spy_options/executor.py:560) | After enough rejects the bot silently hits max-positions and stops trading |
| 5 | **Delayed-data → phantom trades** (compound): chain snapshot has no delayed fallback → all quotes/Greeks read 0; liquidity filter treats 0-quote as *max liquid*; Greek gates *fail open* on 0.0 | [ib_client.py:488](shree/spy_options/ib_client.py:488) + [chain_builder.py:86](shree/spy_options/chain_builder.py:86) + [executor.py:268](shree/spy_options/executor.py:268) | On IB error 10089 the bot can place an order on a zero-Greek phantom contract, no alarm |
| 6 | **Data-feed outage suspends ALL risk management** incl. the 0DTE flatten, even when execution is healthy | [manager.py:567](shree/spy_options/manager.py:567) | An open position stops being managed on a data hiccup |
| 7 | **15:50 EOD flatten can never fire** — the 15:45 RTH stop gates it out first | [manager.py:348](shree/spy_options/manager.py:348) | The overnight-safety net is dead code |

### P1 — Restore function (signals that can't trade + correctness)
| # | Finding | File:line |
|---|---|---|
| 8 | **Bullish PC_RATIO never enriched → 100% of live bullish PC_RATIO calls untradeable** (PUT branch enriches, CALL branch doesn't) | [signal_engine.py:2054](shree/spy_options/signal_engine.py:2054) |
| 9 | **TREND_CONTINUATION (primary strategy) bypasses DynamicConfidence** — flat hardcoded confidence | [continuation.py:182](shree/spy_options/rules_v2/continuation.py:182) |
| 10 | **LONG_STRADDLE never enriched (Greeks hardcoded 0) AND unconditionally rejected by the quality gate** → 45/215 signals (21%) can never trade | [signal_engine.py:2334](shree/spy_options/signal_engine.py:2334) + [executor.py:251](shree/spy_options/executor.py:251) |
| 11 | **TM position-size decision never wired to executor** — small/normal/aggressive computed then dropped | [manager.py:2602](shree/spy_options/manager.py:2602) |
| 12 | **`require_green_edge` gate computed from a `target_pct` that ≠ the executor's real take-profit** | [edge_reality.py:444](shree/spy_options/edge_reality.py:444) |
| 13 | **Wilder RSI seed double-counts the boundary bar** (off-by-one) — every RSI-derived value biased | [technical_levels.py:163](shree/spy_options/technical_levels.py:163) |
| 14 | **Floor-trader pivots never populate in production** (bar feed never contains a prior day) | [technical_levels.py:341](shree/spy_options/technical_levels.py:341) |
| 15 | **Bars include the currently-forming 5-min bar** → regime/continuation/ORB can fire on a repainting bar | [ib_client.py:841](shree/spy_options/ib_client.py:841) |
| 16 | **PC_RATIO has two contradictory regime requirements**; only survives because the two classifiers disagree | [pc_ratio_alignment.py:57](shree/spy_options/rules_v2/pc_ratio_alignment.py:57) |
| 17 | **Structural-trend gate is vacuously True at live default `trend_pivot_count=1`** — silently disabled | [structure.py:166](shree/spy_options/rules_v2/structure.py:166) |
| 18 | **Throttle self-clears zone-lock/leg-cap on nearly every trending bar** (naive ratchet, not real swing pivots; `pivot_lookback_bars` defined-but-unused) | [throttle.py:104](shree/spy_options/rules_v2/throttle.py:104) |

### P2 — Evidence-backed tuning (each backed by a backtest below)
| # | Action | Evidence |
|---|---|---|
| 19 | **Fix or re-derive the theta gate** — `max_theta_burn_pct_per_hour=8.0` blocks **100% of 0DTE entries at every delta and every hour**; likely double-counts decay (`sig.theta` already reflects shrinking T, then ×session-fraction again) | greeks-strike backtest |
| 20 | **Re-formulate QQQ/IWM RS to a rolling 30-min window** (session-cumulative version the bot uses is *noise*, p≈0.77; rolling version has small real edge p=0.004–0.02) **and KILL the divergence veto — it's backwards** (BEARISH_NONCONFIRM → *positive* fwd returns, IC=−0.22) | cross-market backtest |
| 21 | **Delete dead features**: `flow_score`=0.0 in all 216 rows; `gex_bias`/`dark_pool_bias`=NEUTRAL in 214/216. Any gate on these is a no-op | db-outcomes |
| 22 | **Fix `intraday_pc_ratio` population** — it backs PC_RATIO_EXTREME (35% of all signals) yet is populated in only 3/216 rows | db-outcomes |
| 23 | **Remove the Block 16 RSI adjustment** — the team's own commit a7e7f51 (Jul 5) already found "no predictive edge (41–44% over 134 events)" yet it still applies ±0.02 live | confidence-engine + git log |
| 24 | **De-duplicate QQQ RS** — scored twice from two disagreeing sources (live `cross_asset.py` + 10-min-stale `sector_signals.py` `qqq_vs_spy_pct`) | cross-market |
| 25 | **Fix `net_pl_pct` ~100× unit bug** — reads −114% to −302% even on winners (`round_trip_cost_pct` treated as 100–313% of premium) | db-outcomes |
| 26 | **Cap/group the 21 additive confidence blocks** — ~8 measure the same momentum and co-fire → tier inflation. *Tempered:* db-outcomes shows confidence score doesn't predict WR (ρ=0.073, p=0.734), so this is cleanup, not an edge unlock | confidence-engine + market-leaders |

### P2 — Directional (CALL/PUT) symmetry
| # | Finding | File:line |
|---|---|---|
| 27 | PC_RATIO triggers **1.8 bearish / 0.5 bullish are not ratio-symmetric** → 63 PUT vs 14 CALL signals (4.5×); SPY's structural put-hedging demand fires the bearish trigger constantly | [spy_options.py:96](shree/config/spy_options.py:96) |
| 28 | PC_RATIO base-confidence formula is **not a true call/put mirror** | [signal_engine.py:1418](shree/spy_options/signal_engine.py:1418) |
| 29 | **VWAP-reversion exit → 0% WR for PUTs in TREND_DOWN vs 36% for CALLs in TREND_UP** (no fix proposed yet) | [manager.py:1719](shree/spy_options/manager.py:1719) |

> **Caveat:** the entire Apr–Jul 2026 sample was a **+17% one-directional bull quarter**. Puts fought the tape the whole time, so "bad put logic" and "bad quarter for puts" can't be cleanly separated. The put/call WR gap (73% vs 22%) is Fisher p=0.033 but **does not survive multiple-testing correction** (n=24). Symmetrize the *triggers/formula* on correctness grounds; do **not** treat the WR gap as an established edge.

### P3 — DO NOT DO (tested, no edge — save the effort)
- **Don't build a mega-cap option-flow pipeline** (NVDA/TSLA/MSFT/… unusual activity, GEX, block trades). All 8 leaders' price proxies fail to predict SPY at 15–30 min: **12/12 IC tests fail p<0.05**, no lead/lag beyond 1 bar, breadth-extreme event study n.s. — market-leaders backtest.
- **Don't add TLT / UVXY / sector-rotation as new confidence blocks.** Individually marginal, but the combined model **underperforms a do-nothing majority-class baseline in every CV fold** — cross-market backtest.
- **Don't loosen any rules_v2 gate.** Every ablation's expectancy 95% CI spans zero; removing the **throttle** or **grind-trend path** makes expectancy *worse* (they're protective, not over-filters) — filter-ablation backtest.
- **Don't chase 0DTE Δ0.30 on the "highest mean return" result** — its *median* is negative (38–47% WR); the mean is outlier-driven convexity.

---

## BACKTEST EVIDENCE (mission deliverable #11)

### 1. Signals-DB outcomes (n=24 decided, simulated exits)
- CALL 11W/4L (73%) vs PUT 2W/7L (22%); Fisher **p=0.033** but fails Bonferroni (~15–20 cuts). Not an established edge.
- **Confidence does NOT predict WR**: terciles 62.5/44.4/57.1% (non-monotonic), Spearman ρ=0.073 p=0.734.
- **`dynamic_confidence_delta` does not move signals toward wins** (win-mean 0.053 vs loss-mean 0.054).
- **Dead features**: `flow_score`=0.0 (all 216), `gex_bias`/`dark_pool_bias`=NEUTRAL (214/216), `intraday_pc_ratio` populated 3/216.
- **`net_pl_pct` unit bug** (~100×): −114% to −302% even on winners.
- TREND_DOWN 0W/4L; 1DTE 0W/4L (worst same-DTE cell). Puts ~16% more expensive/contract (p=0.036).

### 2. Cross-market confirmation
- QQQ RS **as computed** (session-cumulative) = noise (p=0.77–0.81). **Rolling 30-min** QQQ RS IC +0.035 (p=0.02), IWM +0.044 (p=0.004) — reformulate.
- **Divergence veto is backwards**: BEARISH_NONCONFIRM (bot blocks calls/boosts puts) → +0.046% fwd 30-min; faithful-replica IC=−0.22 (p=0.02, n=125). Suspend/flip it.
- Combined 10-feature GBM: 51.3% OOS vs 55.9% baseline — worse. Don't stack more blocks.

### 3. Market-leaders lead/lag — **null across the board** (12/12 tests p>0.05, no lead beyond 1 bar). Also surfaced the TM-fail-open (P0-#3) and throttle (P1-#18) code bugs.

### 4. Filter-ablation (faithful rules_v2 reimpl, 59 days) — **rules_v2 is NOT the over-filter** (55% pass, 5.4 trades/day; baseline PF 1.08, expectancy CI [−0.089,+0.166] spans zero). Throttle-off and grind-off both *worse*. `min_time_et=10:30` is dead (min_bars=22 dominates to ~11:20). **The 99% attrition is downstream: TM veto + executor quality gate + entry-trigger breach + daily caps.**

### 5. Strike/expiry/Greeks economics
- Spread cost is **small** (0.5–1.5% of premium), *not* the friction.
- Shorter DTE edge is **gamma convexity** (0DTE win 4.3× loss) — but win rate is *lower* at 0DTE; median return negative every cell.
- **Theta gate blocks 100% of 0DTE** at every delta/hour (funnels survivors to 2DTE + high-delta = the *lower*-EV half). `min_premium=$0.30` independently prunes the cheapest convex 0DTE. Gamma-bomb gate likely dead (time gate rejects first).

---

## MISSING DATA / ANALYTICS (deliverables #3–5)
**Absent entirely:** ADX, MACD, Bollinger Bands; true IV Percentile (only a VIX-52w-range proxy that uses the VIX index, not each contract's own IV); chain-derived implied/expected move; options-specific cumulative delta (the tape feed is built on the SPY *stock* ticker, not the option contracts). **Collected but unused:** vega (shown/stored, never gates anything). **Present & correct:** ATR, EMA9/21, RSI(5m) [modulo the off-by-one], VWAP ±1/2σ bands, max-pain (recomputed live).

## CROSS-MARKET VERDICT (deliverable #6)
Mostly **negative** — the honest result is that leaders/TLT/UVXY/sectors don't add edge at this horizon. The one keeper: **reformulate the QQQ/IWM RS you already stream to a rolling 30-min window**, and **fix/kill the divergence veto**.

---

## CONTRADICTIONS & CAVEATS the critic caught
- **$0.00 P&L on real fills is CONFIRMED** (critic queried the DB: rows 203/212 have entry==exit, `fill_pnl_usd=0.0`). One dimension mislabeled it REFUTED — disregard that; the corruption is real. *(Already addressed by the close-logging fix.)*
- The confidence-engine decorrelation push is **undercut by db-outcomes' null** (confidence doesn't predict WR). Do it as cleanup, not as an edge play.
- Most code-correctness fixes have **no P&L estimate** (deliverable #9 partially unmet by design) — they're justified by correctness/safety, not backtested gain. The items with real P&L/statistical backing are the P2 tuning + P3 don't-do list above.

---

## ALREADY FIXED INLINE (this session, worktree only — not yet deployed)
1. **Marketable entry pricing** ([executor.py:435](shree/spy_options/executor.py:435)) — fixes the passive mid+2¢ no-fill (753C). *Follow-ups from audit: make `entry_cross_frac/max` real config fields; add reprice/chase loop + stale-quote re-quote (P1-#12).*
2. **SPY learned-rejection asymmetry + thin-bucket safety valve** ([rules.py:751](shree/trading_manager/rules.py:751)) — the poison caused 77% of SPY rejects for a month.
3. **MES/SPY risk-track separation** (posture/streak/kill decoupled) — audit corroborates the coupling (P0-context) and the missing SPY health net.
4. **Close-logging real-fill P&L** — addresses the $0.00 fabricated P&L.

> These were made against the worktree; the audit read the **main** folder, so it still flags them as open. **Deployment is required** for them to take effect on the live bot.
