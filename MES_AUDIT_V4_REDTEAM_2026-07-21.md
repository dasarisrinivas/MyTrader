# MES Audit V4 — Red-Team of V3 (attacking my own strongest audit) — 2026-07-21

Objective: break V3. V3 was the confident one. It should be attacked hardest.
Result: **V3's two headline quantitative claims (MES 15m mean-reverts; RSI is
the right direction) do NOT survive.** They were built on a corrupt replay
series. V3's code/config/live-fill findings survive intact.

---

## The fatal data-integrity bug (V2 and V3 both missed it)

Everything quantitative in V2/V3 was computed from `logs/decisions.jsonl`,
deduped and **sorted by ISO-timestamp string**. That sort is wrong:

- Timestamps carry timezone offsets (`-06:00`/`-05:00`). String-sorting them is
  not chronological → the "series" jumped **backward 5–6 hours** thousands of
  times. 0% of V3's adjacency was guaranteed chronological.
- Dedup by string kept **33,998** rows; dedup by true UTC instant yields
  **27,512** — V3 carried ~6,500 duplicate bars in wrong positions.
- The file **mixes 1-minute-era and 15-minute-era** replay bars: after true
  sort, 5,093 gaps are 1 min, 15,467 are 15 min, 6,355 are 30 min, 25% are
  >20 min. V3 treated every row as a uniform 15-min step. It is not.

**Consequence:** every `close[i+K]−close[i]`, the variance ratio, the −0.59
autocorrelation, and the RSI/trend forward-return comparison in V2/V3 were
computed across scrambled, mixed-timeframe, duplicated bars. **Invalid as run.**

---

## What clean data actually shows (results flip with cleaning)

Rebuilt three ways. The numbers are not robust — they swing with the cleaning:

| Cleaning | bars | VR(6) | lag-1 AC | RSI net PF | Bot net PF |
|---|---|---|---|---|---|
| V3 (string-sort, mixed TF, dupes) | 33,998 | 0.23 | −0.59 | **1.32** | 1.05 (gross) |
| Strict contiguous 15m (≥40 unbroken) | 563 | **0.93** | **−0.095** | ~1 (n=6) | n/a |
| Session-grouped 15m (≥15/session) | 10,064 | 0.28 | −0.44 | **0.80** (t −3.1) | 0.55 |

- The **strictest, only internally-consistent** subset (563 bars, truly unbroken
  15-min chains) shows **VR≈0.9 (random walk)** and **normal lag-1 AC −0.095**.
- Looser subsets show VR<1 **but carry a −0.44 lag-1 signature** — abnormal for
  real 15m index futures (~−0.05 to −0.10 expected) = residual artifact from
  stitching different backtest vintages of the same session.
- RSI net PF swings **1.32 → 0.80** (from winning to *losing*, t=−3.1) purely
  on cleaning choice.

**When a conclusion inverts based on defensible data-cleaning decisions, it is
not a finding — it is noise.**

---

## Phase 1 — V3 conclusions re-graded

| V3 conclusion | Red-team verdict | Why |
|---|---|---|
| Native fills lose (−$497, 25% WR) | **VERIFIED** | From `orders.db`, independent of replay series |
| PF inflated by BACKFILL reconstruction | **VERIFIED** | Native vs backfill split is real |
| Over-engineering / ~61k lines deletable | **VERIFIED** | Code + import graph |
| Filters correlated (A/C/F same stack) | **VERIFIED** | Strategy source |
| Confidence double-counts | **VERIFIED** | Live logs |
| RAG/LLM crashed 13/13 in prod | **VERIFIED** | Live logs, `session_range_pts` |
| Sentiment only reduced confidence | **VERIFIED** | Live logs |
| ~120 params, 63 post-hoc tweaks, survivorship | **VERIFIED** | `config.yaml` + git |
| Bot has no positive edge | **LIKELY** (see caveat) | Loses in every cleaning; but native n=114 is small |
| **MES 15m is structurally mean-reverting** | **INCORRECT / UNSUPPORTED** | Strict-clean data = random walk (VR 0.93); result flips with cleaning |
| **RSI is the right research direction** | **UNSUPPORTED** | PF 1.32→0.80 across cleanings; loses on clean data; dead OOS 2026 |
| No strategy has proven edge → signal-only correct | **VERIFIED (strengthened)** | If anything, data is too broken to claim *any* edge |

**Net: V3's architecture/forensics = solid. V3's market-structure and RSI
statistics = broken.** V3 was "another convincing narrative built on incomplete
[corrupt] evidence" — exactly what the red-team brief feared.

---

## Phase 2 — Statistical attacks (all land)

- **Overlapping windows:** V3's naive t-stats (3.19, 16.29) used 6-bar-overlapping observations → independence violated → t inflated. Non-overlapping already dropped RSI to t=1.84; clean data drops it to negative.
- **Multiple testing:** ~6 systems × 5 horizons × 3 years ≈ 90 tests. No Bonferroni. The lone "winner" (RSI 2025 t=2.71) is cherry-picked; corrected threshold (~t>3.3 for 90 tests at 0.05) rejects it.
- **Replay/vintage bias:** the −0.44 residual autocorr is the fingerprint of stitching different backtest runs of the same dates. Not market behavior.
- **Small sample:** "bot has no edge" rests on **114 native fills** — Wilson 95% CI on 25.4% WR is roughly 18–34%. Enough to say "not proven profitable," **not** enough to say "proven loser." V3 slid from the former to the latter.
- **Look-ahead:** unquantifiable — the replay's RSI/ATR/EMA were logged by the strategy; whether they used only-past bars in every replay path is unverified.

---

## Phase 3 — Market structure: verdict downgraded to UNKNOWN

V3: "structurally mean-reverting." Red-team: **cannot be concluded from this
data.** Strict-clean → random walk; loose-clean → artifact-driven reversion.
The variance-ratio test is fine in principle but was fed a scrambled series.
RTH/ETH split, per-year, per-vol-regime — none can be trusted until the input
series is a real, contiguous, single-timeframe price feed, which
`decisions.jsonl` is not. **MES 15m structure is currently UNMEASURED.**

---

## Phase 4 — RSI conclusion: rejected as stated

RSI is not shown to be "the right direction." On the cleanest data it loses
(PF 0.80). It is likely a **proxy for short-horizon bounce** that is (a) partly
microstructure (uncapturable net of spread) and (b) unstable across cleanings
and years. z-score / Bollinger / VWAP-deviation would just be re-parameterized
versions of the same bounce and inherit the same instability. **No mean-reversion
variant is validated.** Designing a "stronger" one now would be fitting noise.

---

## Phase 5 — "No edge exists": the one to be careful with

Separated, not combined:
- **Entry quality:** unmeasured on clean data (too few clean signals).
- **Exit quality:** V3's stop/target tests used close-only data → cannot see
  intrabar stops → invalid. Exit edge is **unknown**.
- **Risk/sizing:** irrelevant to signal-only.
- **Execution:** native fills lose, but 114 trades can't separate bad-edge from
  bad-execution or variance.

Correct statement: **"No edge has been demonstrated"** — NOT "no edge exists."
Absence of evidence here is genuinely absence of *usable data*, not proof of a
negative. V3 occasionally overstated this. The safe posture is identical either
way: trade nothing, collect clean data, measure.

---

## Phase 7 — Architecture (unchanged, evidence independent of the data bug)

| Subsystem | Verdict |
|---|---|
| Execution/RAG/LLM/sentiment/VX/optimizer/manager-veto/risk | **DELETE** ✅ done |
| 1m-era strategies | **DELETE** ✅ done |
| signal_bot, feature_engineer, logging | **KEEP** |
| es_fifteen_min | **UNKNOWN** (no valid edge test exists yet — was REBUILD, now downgraded to UNKNOWN) |
| 120-knob config | **SIMPLIFY** |
| RSI greenfield strategy | **DO NOT BUILD YET** (unvalidated) |

---

## Phase 8 — Hidden assumptions nobody questioned

1. **That `decisions.jsonl` is a usable price series.** It is not — this is the whole V4 finding.
2. **Why 15-minute bars?** Never justified. Structure/edge may live at 1m, 5m, or daily.
3. **Why MES / why RTH-anchored OR?** Inherited, never tested.
4. **Why ATR-normalized returns?** Assumes ATR is the right risk unit; untested vs raw ticks or realized vol.
5. **Why confidence at all?** No calibration curve was ever produced; confidence is decoration.
6. **Why backtest on replay logs instead of raw historical OHLCV?** The proper input (tick or 1-min OHLCV from IB) exists and was bypassed.

---

## Phase 9 — Research to run BEFORE writing any strategy (ranked by EV)

| # | Hypothesis | Data | Test | Success | Effort | EV |
|---|---|---|---|---|---|---|
| 1 | A clean, single-timeframe MES 15m RTH OHLCV series changes all conclusions | raw IB 1-min→15m resample, 2+ yrs | rebuild VR/autocorr on real bars | contiguous, lag-1 AC in [−0.15,0] | 1d | **highest** — everything depends on it |
| 2 | MES 15m is random-walk (no linear intraday edge) | data #1 | VR + Ljung-Box on RTH-only | fail to reject RW → stop hunting linear rules | 0.5d | high (kills dead ends) |
| 3 | Edge lives at a different horizon | 1m/5m/daily OHLCV | VR + simple rules per TF, walk-forward, Bonferroni | OOS PF>1.15 net at any TF | 2d | high |
| 4 | Overnight-gap / open-drive has structure RTH lacks | daily O/H/L/C + prior close | gap-fill / open-range stats | monotonic OOS edge | 1d | med-high |
| 5 | Volatility (not direction) is the forecastable quantity | 15m realized vol | HAR-RV forecast R² OOS | R²>0 OOS | 1.5d | med |
| 6 | Exit rules dominate entry (test with intrabar data) | tick/1m OHLC | fixed entry × exit grid, intrabar-accurate | exit choice moves PF sign | 1d | med |
| 7 | Day-of-week / time-of-day session effects | timestamped OHLCV | bucketed expectancy, multiple-test corrected | surviving bucket after correction | 0.5d | med |
| 8 | Cross-asset lead (ES/NQ/VIX → MES) | synchronized feeds | lagged regression OOS | OOS predictive R²>0 | 2d | med |
| 9 | The 114 native fills reflect execution, not edge | fills + intended signal levels | slippage vs signal-price study | slippage explains loss | 0.5d | low-med |
| 10 | Regime-conditional structure (trend days vs chop days) | data #1 + ADX | VR conditioned on opening ADX | VR differs by regime, OOS-stable | 1d | low-med |

**#1 gates all others.** No strategy work until a clean price series exists.

---

## Phase 10 — Final verdict (direct answers)

1. **Is V3 technically sound?** Partly. Its forensic/architecture half is sound; its quantitative half is **not** — built on a corrupt series.
2. **Strongest V3 conclusions:** over-engineering, RAG crash, correlated filters, survivorship, native-fills-lose. All independent of the data bug. Trust them.
3. **Weakest:** "MES 15m mean-reverts" and "RSI is the right direction." Both flip with cleaning.
4. **Unsupported:** any positive statement that MES 15m has a specific tradeable structure or that RSI has edge.
5. **Trust V3 enough to rebuild the bot?** For **deletion** — yes (already done). For **choosing a new strategy** — **no.** V3 does not license building an RSI bot.
6. **Additional evidence required:** a clean, contiguous, single-timeframe MES OHLCV series (research #1), then walk-forward with multiple-testing correction and intrabar-accurate exits.
7. **Bet my own money today on:**
   - Current strategy — **No.** Loses net in every cleaning.
   - RSI strategy — **No.** Unvalidated; loses on clean data; dead OOS.
   - Greenfield — **No.** Not yet built on trustworthy data.
   - **Nothing — Yes.** Correct action: signal-only bot logging + a real validation dataset. Trade only what forward-proves out net of costs, multiple-testing corrected.

---

## Caveman verdict

Attack V3. V3 bleed. The bone pile (`decisions.jsonl`) was rotten — mixed 1-min
and 15-min bone, sorted by wrong stick, double-counted. Every "mean revert"
number and every "RSI win" number come from rotten pile. Clean the pile hard →
VR say random walk, RSI say lose. Number flip when pile cleaned different way.
Flip number = no truth, only noise.

Still true (from other piles, not the rotten one): bot lose real money, code
bloated, RAG dead, config curve-fit. Those solid.

Not true anymore: "MES bounce." "RSI right way." **UNKNOWN both.** Cannot know
until get real clean price bar from IB, not replay log.

Bet money today? Trade **nothing**. Not bot, not RSI, not greenfield. First get
clean data. Then measure. Then maybe trade. Signal-only bot already right answer
— now for the right reason: not "RSI better," but "we do not yet know anything,
so risk nothing and go measure."
