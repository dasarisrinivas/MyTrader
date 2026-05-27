# Architecture & Execution-Faithfulness Audit (2026-05-27)

Skeptical prop-firm review of the research → execution → replay pipeline. Premise:
complexity is guilty until proven innocent; any engine/replay mismatch is critical
until explained; live capital is at risk.

## Headline

**The "major warning sign" (engine +$2,142 / 53.1% vs replay −$4,794 / 34.2% on the
same trades) was caused entirely by bugs in MY standalone replay, not the engine.**
The backtest engine is internally sound — verified no phantom exits, no lookahead, and
**conservative (stop-first) same-bar fills via a direct unit test.** The real risk is
not the engine; it is that **the edge is statistically thin and slippage-fragile.**

---

## 1. Backtest fidelity audit

| Dimension | Finding | Verdict |
|---|---|---|
| Entry timing | Signal forms at bar T close; entry fills at **bar T+1 open + 1-tick slippage** (entry_price ≈ next-bar open). | OK, no lookahead |
| Bar-close vs intrabar | Exits evaluated on **15m bar** OHLC (no intrabar path). | Risk (see §below) |
| Fill model | Market entry at open+slip; stop fills at stop−slip; limit/TP fills at level. Commission $2.40 RT, 1-tick slippage. | Reasonable |
| **SL/TP ordering** | Bracket children inserted **SL before TP**; `process_bar` iterates in insertion order → **stop checked first**. **Unit-tested**: a bar spanning both SL and TP fills the **STOP** (−$34.9), OCO-cancels TP. | **Conservative ✓** |
| Same-bar entry lookahead | Exit children activated *after* the entry-bar order snapshot → only checked from **bar T+1**. | OK ✓ |
| Breakeven | None on normal trades. | OK |
| Trailing / partials | Broker supports TRAIL/PARTIAL but strategy doesn't use them; fixed 1 contract. | OK |
| Stop tightening | `_apply_rth_close_tighten` (tighten into RTH close) + drawdown-guard tighten. Real dynamic exits — *not* modeled by external replays. | OK (not lookahead) |
| Max-hold | `max_hold_minutes` (config 8 bars × 15 = 120 min) → market exit. | OK |
| Overnight/session | `run_15m_only` calls strategy **only on RTH bars**; fills processed on all bars. Live bot mirrors this (skips overnight entries). | OK, consistent |
| Lookahead risk | Strategy receives `history = features_df.iloc[:bar_idx+1]` — only data through current bar. | **None found ✓** |
| Data leakage / survivorship | Single continuous ES contract series (rolled), no symbol selection, no forward fill of signals. | None found |

**The one genuine fidelity limitation:** exits are resolved on **15m bars without
intrabar sequence**. The engine handles this *conservatively* (stop-first), which if
anything **understates** PnL on both-touched bars (it assumes the worst path). So the
+$2,142 is not inflated by this; it may be marginally pessimistic. (A 1m-primary
re-run would remove the assumption entirely and is the recommended confirmation.)

## 2. Replay consistency audit — root cause of the inversion

Five independent attempts to externally replay the 392 trades all produced ~34% WR /
negative PnL with **−0.75 PnL correlation** and **15% sign agreement** vs the engine —
and critically, only **22% agreement on *unambiguous* stop-loss trades**, where there
is nothing to disagree about. That validation failure proves the **replays are buggy**,
not the engine. Root causes identified:

1. **SL/TP anchoring** — engine anchors levels to the **signal-bar close**; my replays
   used the **fill price** (~0.5–4pt higher). On 6–8pt stops, that flips boundary cases.
2. **Entry-bar inclusion** — replays scanned the entry bar for exits; the engine starts
   at bar T+1.
3. **Unmodeled exits** — RTH-close tighten, drawdown-guard, max-hold market exits.
4. **Resolution granularity** — 1m first-touch vs the engine's 15m stop-first.

Direct falsification of the engine against raw bars: **0 phantom exits** (every exit
price was actually touched), **0 lookahead** (no exit before entry). Hand-verified
sample trades reconcile exactly with raw 15m data once correct anchoring is applied.

**Conclusion: the engine is the authority; the replays are invalid.**

## 3. Multi-run contamination audit

| Artifact | Provenance | Safe? |
|---|---|---|
| `all_trades_ADX12.csv` | Single controlled 3-chunk run, current config (shorts-off/ADX-12/blocks), one code version, ES_15m_multiyr data. | **CLEAN — use this** |
| `manager_decisions.jsonl` | Accumulated across *every* run (multiple configs, code versions, MOM_BRK experiments, paper+live+backtest). `override=True` on 98.5%, each signal logged ~4.6×, 0 real approvals. | **CONTAMINATED** |
| `orders.db` (orders table) | Mixed backtest/paper/live; $100k+ phantom aggregate sums. | **CONTAMINATED** |
| `decisions.jsonl` (133 MB) | Multi-run accumulation. | Contaminated |

Never compute population statistics from the contaminated logs. The "approved/rejected"
signal populations I sampled from `manager_decisions.jsonl` mixed configs/timeframes and
must not be used for win/loss conclusions.

## 4. Architecture consistency — research vs live divergence

Lifecycle: **data (ES 1m→15m resample) → engineer_features → es_fifteen_min.generate
(signals + blocks) → [gate] → fills → exits → logs**.

The single most important divergence: **the backtest gate ≠ the live gate.**
- **Backtest:** signals pass through `RiskManager` (risk_gate) only. No R:R≥2.0 mandate.
- **Live:** signals additionally pass through the `trading_manager` (R:R≥2.0 static
  floor + posture/streak logic). This is why live took **zero** trades (deadlock) while
  the backtest is fully invested. **The validated +$2,142 was never gated by the live
  manager.** (Fix applied: `TM_MIN_RR=1.2`.)

Other notes: indicator periods are hard-coded (not shared via config) so a 5m timeframe
silently changes the trend horizon; the manager runs as a separate launchd process with
its own state file (hidden statefulness — posture/streaks persist across restarts).

## 5. Source of truth

- **Authoritative:** the **backtest engine** (`run_15m_only` + `BrokerSimulator`) and
  its trade log `all_trades_ADX12.csv`. Verified sound.
- **Trustworthy metrics:** engine WR / PnL / per-trade realized_pnl; gate-toggle
  counterfactuals (they re-run the engine).
- **Unsafe artifacts:** every standalone replay/first-touch simulator I built this
  session; `manager_decisions.jsonl`, `orders.db`, `decisions.jsonl` for population stats.
- **Prior findings — STILL VALID** (engine-based): shorts-off, opening/Friday blocks,
  ADX 18→12, the 5m-prototype rejection, the gate counterfactual, and the
  momentum-breakout **rejection** (its backtest used the engine).
- **Prior findings — INVALIDATED** (built on buggy first-touch sims): the discovery
  *screen* PnL numbers, the cost-aware trigger screen PnL, the approved-signal −$1,376,
  and all the replay PnL (−$4,794 / −$3,477 / −$8,389). Discard these numbers. (The
  qualitative "momentum was the only positive archetype" still holds — but its rejection
  came from the engine, which is what matters.)

## 6. Statistical robustness (on the trusted engine book)

- **Edge is marginal:** mean **$5.46/trade**, t-stat **2.09**, 95% CI **[$0.34, $10.59]**.
  Profit factor 1.28. Statistically barely significant.
- **Concentrated:** 49% of profit from **3 of 27 months**; 19/27 months positive.
- **Period-dependent:** walk-forward segments +$30 / +$430 / +$981 / +$701 — nearly flat
  in 2024, carried by 2025.
- **Slippage-fragile (biggest risk):** +$2.5/trade → +$1,162; **+$5/trade → +$182**;
  +$7.5/trade → **negative**. The edge is thinner than 2–3 ticks of slippage.
- Max drawdown −$454 (21% of profit) — acceptable in isolation.

## 7. Verdicts

### Confirmed-valid findings
1. Engine has no lookahead, no phantom exits, conservative stop-first fills (unit-tested).
2. Disabling shorts, the opening & Friday blocks, and ADX 18→12 each improved the
   engine book (gate-toggle verified).
3. New orthogonal long signals (momentum breakout) fail in the engine.
4. The live no-trades bug = trading_manager R:R≥2.0 vs ~1.3 strategy (cold-start deadlock).
5. The book's edge is real but **thin** (t≈2.1) and **slippage-fragile**.

### Invalidated findings
1. Every standalone-replay PnL number this session (engine/replay "mismatch" was my bug).
2. Any win/loss conclusion drawn from `manager_decisions.jsonl` / `orders.db` populations.
3. The exploratory-screen absolute PnL figures (first-touch proxy).

### Top 5 technical risks
1. **15m exit resolution** — conservative today, but unconfirmed vs true 1m path; a
   1m-primary re-run is the clean confirmation. (Fragility: stop-first relies on an
   implicit child-insertion order.)
2. **Research ≠ live gate** — backtest never applied the manager; live behavior can
   differ materially from the validated book.
3. **Contaminated logs** used as if authoritative — easy to draw false conclusions.
4. **Hidden manager statefulness** (posture/streak/health) persists across restarts and
   can silently halt trading.
5. **Hard-coded indicator periods** — any timeframe change silently alters the strategy.

### Top 5 overfitting risks
1. Edge concentrated in 3 months / one strong year (2025); 2024 ≈ flat.
2. t-stat 2.09 — borderline; many gates were tuned on the same ~1 year of data.
3. A large stack of hand-added blocks/guards (opening, Friday, Monday, late-afternoon,
   medium-ATR, exhaustion, RSI/MACD overnight guards) — each fitted to past losers.
4. Tight 6–8pt stops on 15m bars — outcomes hypersensitive to fills/slippage.
5. SL/TP and ADX thresholds optimized against this exact dataset.

### Top 5 improvements most likely to raise *real* live expectancy
1. **Re-run the backtest at 1m-primary exit resolution** and re-validate — removes the
   only fidelity assumption and gives a slippage-honest PnL. Do this before sizing up.
2. **Stress the edge at realistic slippage (+2–3 ticks)** and only trade sizes/sessions
   where it survives; widen targets or filter to higher-ATR setups to lift $/trade above
   the slippage floor.
3. **Make the manager R:R floor expectancy-based** (win-rate aware), not a static 2.0 —
   aligns the live gate with the validated strategy and ends the deadlock cleanly.
4. **Cut complexity:** test removing the rarely-binding guards (exhaustion, medium-ATR)
   — fewer fitted knobs = less overfit, easier to trust.
5. **Forward/paper-validate** the longs-only RTH book for 1–2 months at minimum size
   before any scale-up; the backtest edge is too thin to trust un-forward-tested.

## Bottom line
The engine is trustworthy; my replays were not. But trustworthy ≠ profitable-live: the
validated edge is **thin (t≈2.1), concentrated, and slippage-fragile.** Treat +$2,142 as
an *optimistic-fills-aside* upper-ish estimate that **has not survived a slippage-honest,
1m-resolution, forward test.** Do not size up until it does.
