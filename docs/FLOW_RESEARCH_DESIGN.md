# SPY Options — Real Flow Research Layer (design only)

**Status:** DESIGN. Observation system. No trading. No strategy. No TC change.
No governor change. No production gating. Nothing here connects to live orders.

**Purpose:** prove or disprove that real OPRA options flow carries forward-
predictive information that price-derived features do not.

**Date:** 2026-07-23

---

## 0. Why this exists (one paragraph)

Six audits proved every current feature is price-derived → predicts nothing
(autocorrelation of noise; forward returns flat every horizon; barrier-win
≤ null). `flow_score` = 2 distinct values / 657 rows (dead). "GEX" =
yfinance end-of-day OI proxy (cosmetic). Real options flow — actual executed
prints + aggressor side — is the ONLY axis orthogonal to price. This layer
observes that axis and tests it with the same rigor that killed everything else.
Null result is an acceptable, expected outcome. We are buying a *test*, not an
edge.

---

## 1. Data model — `spy_flow_prints`

One row per executed option print on SPY (and optionally SPX). Raw tape.
Immutable. No signals, no scores — measurements only.

```sql
CREATE TABLE spy_flow_prints (
    id              INTEGER PRIMARY KEY,
    -- identity / time
    ts_utc          TEXT    NOT NULL,   -- print exchange timestamp (ISO, ms)
    ts_et           TEXT    NOT NULL,   -- same instant, ET (session bucketing)
    session_date    TEXT    NOT NULL,   -- YYYY-MM-DD ET (fast day filter)
    -- underlying context at print time
    underlying_px   REAL,               -- SPY mid at/near print ts
    -- contract identity
    root            TEXT    NOT NULL,   -- 'SPY' | 'SPX'
    expiry          TEXT    NOT NULL,   -- YYYY-MM-DD
    dte             INTEGER,            -- trading-DTE (NYSE calendar), not cal
    strike          REAL    NOT NULL,
    right           TEXT    NOT NULL,   -- 'C' | 'P'
    -- the print
    trade_px        REAL    NOT NULL,   -- option trade price
    size            INTEGER NOT NULL,   -- contracts
    premium         REAL    NOT NULL,   -- trade_px * size * 100 (dollars)
    exchange        TEXT,               -- OPRA participant code
    condition_codes TEXT,               -- raw OPRA trade condition list (JSON)
    -- NBBO at print (for classification; from quote feed, Lee-Ready)
    bid             REAL,
    ask             REAL,
    -- derived classification (computed once at ingest, never trusted blindly)
    aggressor       TEXT,               -- 'BUY' | 'SELL' | 'MID' | 'UNKNOWN'
    aggressor_src   TEXT,               -- 'QUOTE' | 'TICK' | 'ISO_COND' | 'NONE'
    is_sweep        INTEGER DEFAULT 0,  -- multi-exchange burst, same contract/side
    is_block        INTEGER DEFAULT 0,  -- size >= block threshold
    oc_estimate     TEXT,               -- 'OPEN' | 'CLOSE' | 'UNKNOWN' (vs prior OI)
    -- greeks at print (from vendor or self-computed BS)
    delta           REAL,
    gamma           REAL,
    iv              REAL,
    greeks_src      TEXT,               -- 'VENDOR' | 'COMPUTED'
    -- provenance
    data_source     TEXT,              -- 'polygon' | 'thetadata' | 'databento'
    ingested_at     TEXT
);
CREATE INDEX ix_flow_session ON spy_flow_prints(session_date, ts_et);
CREATE INDEX ix_flow_contract ON spy_flow_prints(session_date, expiry, strike, right);
```

### Field justification

| Field | Why it exists |
|---|---|
| `ts_utc` / `ts_et` | Exact instant to align flow to a signal's timestamp and to bucket by session clock (open/mid/close behave differently). |
| `session_date` | Cheap partition key for per-day aggregation and the validation battery. |
| `underlying_px` | Anchor. Every flow measure must be judged *relative to spot* (ATM vs wing). Also lets us bucket by moneyness. |
| `root` | Separate SPY (ETF, 1-lot=100 shares) from SPX (cash, ×10 notional). Never mix premiums. |
| `expiry` / `dte` | Flow concentration by expiry is a core measure (0DTE vs weekly vs monthly are different actors). Trading-DTE matches bot convention. |
| `strike` / `right` | Contract identity; needed for moneyness, walls, repeated-strike detection. |
| `trade_px` / `size` | The raw print. Everything derives from these two. |
| `premium` | Dollar-weighted aggregation unit. Net premium imbalance is the headline measure. |
| `exchange` | Sweep detection needs multi-venue prints of the same contract in a tight window. |
| `condition_codes` | OPRA flags: ISO (intermarket sweep), late, spread-leg, auction. Filters out non-directional prints (multi-leg legs, auctions) that would poison flow. |
| `bid` / `ask` | Prevailing NBBO — the ONLY way to classify aggressor honestly (at-ask buy / at-bid sell). Without quotes, aggressor is a guess. |
| `aggressor` | The whole point. Informed direction of the print. |
| `aggressor_src` | Honesty tag. QUOTE = trustworthy; TICK/NONE = weak. Lets validation weight or drop low-confidence classifications. |
| `is_sweep` | Sweeps = urgency (taker crossing multiple venues). Classic "informed" proxy. Flag, not gospel. |
| `is_block` | Large single prints = institutional. Size threshold, root-specific. |
| `oc_estimate` | Opening vs closing changes meaning (new position vs unwind). Estimated by comparing size to OI delta — weak, hence "estimate". |
| `delta` | Delta-weighted flow = directional exposure actually transacted (10 ATM ≠ 10 deep OTM). |
| `gamma` | For a *real* dealer-gamma reconstruction later (replaces the fake OI proxy) — but stored raw now, not modeled. |
| `iv` | Distinguishes vol-buying (straddle/hedge) from directional bets; guards against reading a vol trade as a direction bet. |
| `greeks_src` | Vendor greeks vs self-computed BS have different error. Tag it. |
| `data_source` / `ingested_at` | Provenance + reproducibility across vendors/replays. |

---

## 2. Flow features (measurements, NOT signals)

Computed per **snapshot** = one (session, clock-time) evaluation over a trailing
window (e.g. last 5 min, last 30 min, cumulative-since-open). No thresholds,
no BUY/SELL, no scores. Just numbers written next to context.

### Tier A — likely useful (orthogonal to price by construction)
| Measure | Definition |
|---|---|
| `net_call_prem` | Σ premium of BUY-aggressor calls − SELL-aggressor calls, window. |
| `net_put_prem` | Same for puts. |
| `pc_prem_imbalance` | (net_call_prem − net_put_prem) / total_prem. Directional pressure. |
| `dw_flow` | Δ-weighted signed flow: Σ(aggressor_sign · delta · size). Actual directional exposure bought. |
| `sweep_intensity` | sweep premium / total premium, window. Urgency of takers. |
| `block_prem` | Σ premium where is_block=1. Institutional footprint. |
| `oc_open_ratio` | opening-estimate premium / total. New risk vs unwind. |

### Tier B — context / conditioning (interpret with care)
| Measure | Definition |
|---|---|
| `expiry_concentration` | Herfindahl of premium across expiries. 0DTE-dominated ≠ swing flow. |
| `strike_repetition` | Max premium share on a single strike/expiry. Repeated hits = conviction or a wall. |
| `atm_vs_wing` | Premium share within ±0.5% of spot vs outside. Hedging vs speculation. |
| `iv_weighted_side` | Are aggressive buys concentrated in high-IV (vol bet) or low (directional)? |

### Explicitly NOISE — measure but distrust
- Raw print count (dominated by tiny retail 1-lots).
- Total volume without aggressor side (no direction = no information).
- Mid-price prints (`aggressor='MID'`) — ambiguous, exclude from net measures.
- Multi-leg spread legs (condition-coded) — a call print that's really one leg
  of a spread is not a directional call buy. **Must be filtered by condition_codes.**
- Auction/late prints — not real-time intent.
- SELL-to-open vs SELL-to-close indistinguishable without OI → treat sells as
  weaker signal than buys.

**Separation rule:** every net measure uses only `aggressor IN ('BUY','SELL')`
AND `aggressor_src='QUOTE'` AND condition_codes clean of spread/auction/late.
Everything else is diagnostic only. This is where most fake "flow" products lie.

---

## 3. Integration (production untouched)

Two new tables. Zero writes to `spy_signals`, executor, manager, governor.

```sql
-- raw tape (section 1)
spy_flow_prints

-- computed snapshots joined to context
CREATE TABLE shadow_flow (
    id              INTEGER PRIMARY KEY,
    snapshot_kind   TEXT NOT NULL,   -- 'SIGNAL' | 'REJECTED' | 'INTERVAL'
    signal_id       INTEGER,         -- FK spy_signals.id when kind in (SIGNAL,REJECTED)
    session_date    TEXT NOT NULL,
    ts_et           TEXT NOT NULL,
    window_s        INTEGER NOT NULL, -- trailing window for the measures
    -- Tier A
    net_call_prem REAL, net_put_prem REAL, pc_prem_imbalance REAL,
    dw_flow REAL, sweep_intensity REAL, block_prem REAL, oc_open_ratio REAL,
    -- Tier B
    expiry_concentration REAL, strike_repetition REAL,
    atm_vs_wing REAL, iv_weighted_side REAL,
    -- bookkeeping
    n_prints INTEGER, n_prints_used INTEGER,  -- transparency: how many survived filters
    computed_at TEXT
);
CREATE INDEX ix_shadow_sig ON shadow_flow(signal_id);
CREATE INDEX ix_shadow_kind ON shadow_flow(snapshot_kind, session_date);
```

Attach snapshots at three triggers (all observation-only):
1. **Every dispatched signal** — join on `spy_signals.id` (existing outcomes give
   forward return + real P&L for free).
2. **Every rejected/blocked signal** — reuse existing `blocked_signals.jsonl` +
   `blocked_gate` plumbing already in `analytics_db.py`. Rejected set is the
   control group.
3. **Fixed intervals** — every N minutes regardless of signal (e.g. 09:45,
   10:15, ..., 15:45). Unconditional sample; removes signal-selection bias.

Producer is a **separate process** (`scripts/flow_research_ingest.py`,
`scripts/flow_snapshotter.py`) reading its own vendor connection and its own DB.
It does not import the executor or manager. Bot cannot see these tables.

---

## 4. Validation framework (same bar as prior audits)

Core question: **does flow predict anything price does not?**

Run over BOTH the retro-attached 657 historical signals AND new interval
snapshots. Battery per Tier-A measure:

| Test | Method | Null / pass bar |
|---|---|---|
| Forward return | SPY return at +1/+5/+15/+30 min after snapshot, split by measure sign/quintile | monotone lift across quintiles, not flat |
| Option P&L proxy | Simulated ATM 1-DTE option P&L over same horizons (executor-real fill model) | positive expectancy in top quintile |
| MFE / MAE | Max favorable / adverse excursion, ±0.5% barriers, 90-min stop | MFE>MAE asymmetry beyond null |
| Barrier win | P(+0.5% before −0.5%) top vs bottom quintile | top-quintile > 50%, spread vs bottom significant |
| Information lift | Does measure improve forward-return prediction *after* residualizing on price features (RSI/VWAP/ORB/ADX)? | partial-correlation / added-R² > 0, Wilson-lower-bounded |
| Feature correlation | Pearson/Spearman of each flow measure vs each existing feature | LOW correlation required — else it's a price restatement, not new info |
| Direction-confound guard | Regress out contemporaneous SPY move (the CALL_SWEEP=beta trap) | lift must survive; beta ≠ alpha |

Controls:
- **Signal vs rejected vs interval** three-way — does flow separate winners in
  the unconditional interval set (not just the selected signal set)?
- **Aggressor-source split** — QUOTE-classified only vs all. If "edge" only
  appears in weakly-classified prints, it's noise.
- **Randomization** — shuffle flow labels across timestamps; battery must go flat.
  If shuffled flow still "predicts," the test is broken.

Decision logic: a measure is INTERESTING only if it shows forward lift AND low
correlation with existing features AND survives direction-confound residualization
AND survives the shuffle. Anything less = null, documented, dropped.

---

## 5. Minimum sample (pre-registered, before any strategy talk)

| Gate | Requirement |
|---|---|
| Historical replay window | ≥ 40 trading sessions of OPRA replay (retro-attach to existing 657 signals covers Apr–Jul immediately). |
| Interval snapshots | ≥ 1,500 unconditional interval observations (≈40 sessions × ~8/day, well past CLT). |
| Per-measure signal cell | ≥ 200 observations in each outcome quintile before any lift is quoted. |
| Live forward-confirm | ≥ 20 *new* live sessions after historical passes — historical replay can hide look-ahead; live is the honest test. |
| Confidence bar | Wilson lower bound on any win-rate claim; lift significant at the quintile spread, not the point estimate; must survive shuffle + direction-confound. |
| Promotion to "strategy consideration" | ALL above pass. Then and only then is a strategy scoped — as a separate proposal, governor-gated, shadow-first. Not in this layer. |

If historical replay is flat → **stop before paying for real-time.** That is the
cheapest possible kill.

---

## 6. Cost control

**Principle: prove on historical replay before paying one cent for real-time.**
The bot already has 657 dated signals with outcomes; historical OPRA for that
same window retro-attaches flow and runs the full battery with zero live
subscription.

### Data requirements
- Per-print OPRA trades for SPY (± SPX) options.
- Time-synced NBBO quotes (mandatory for honest aggressor classification).
- Greeks: vendor-supplied OR self-computed Black-Scholes (we already compute
  greeks elsewhere — self-compute is fine, tag `greeks_src='COMPUTED'`).

### Vendor options (verify current pricing at purchase; figures approximate)
| Vendor | Fit | Approx cost | Replay/history |
|---|---|---|---|
| **ThetaData** | Options-specialized; trades + NBBO + greeks; strong historical | ~$80–160/mo | Yes — historical tick download. **Best POC fit.** |
| **Polygon Options** | Advanced tier for real-time trades + quotes WS; flat files for history | Starter ~$29 (delayed), Advanced ~$199/mo | Flat files (S3) on higher tiers |
| **Databento** | Raw OPRA, pay-per-GB, precise replay | usage-based | Yes — historical replay, granular |
| **CBOE DataShop** | Historical flat files | per-dataset one-off | History only, no real-time |
| Retail alert feeds (Unusual Whales, Cheddar, FlowAlgo) | processed alerts | $50–250/mo | **DO NOT BUY** — lagged, survivorship, no raw tape |

### Smallest proof-of-concept (do this FIRST)
1. Buy **historical** OPRA replay for ~10 sessions from ThetaData (or Databento
   pay-per-GB) — smallest spend, likely < $100 one-off / one month.
2. Reconstruct prints + NBBO for SPY only, 0–2 DTE only (where day-trade flow
   lives). Ignore SPX, ignore far expiries — cuts data volume hard.
3. Retro-attach to the existing signals in that window + generate interval
   snapshots. Run the battery.
4. Read the correlation + information-lift + shuffle results.
   - Flat → **DO NOT PROCEED**, kill, report null. Total spend ≈ one month/one-off.
   - Signal of lift → expand to 40-session replay, then subscribe real-time
     ONLY for the ≥20-session live forward-confirm.

No real-time subscription until historical shows lift. No annual commit ever
during research.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| **Aggressor misclassification** | Quote-synced Lee-Ready; `aggressor_src` honesty tag; validation restricted to QUOTE-classified; shuffle test catches a broken classifier. |
| **Spread legs read as directional** | Filter `condition_codes` (spread/auction/late) out of net measures. This is the #1 way retail flow products deceive. |
| **Look-ahead in historical replay** | Snapshot uses only prints with `ts <= snapshot_ts`; live forward-confirm gate (≥20 sessions) is mandatory before any claim. |
| **Direction confounding (beta not alpha)** | Residualize on contemporaneous SPY move — same guard that flagged CALL_SWEEP. |
| **SPX/SPY notional mixing** | `root` separation; never pool premiums. |
| **Selection bias (only signal times)** | Unconditional interval snapshots as the primary unbiased sample. |
| **Data cost creep** | Historical-first; SPY 0–2DTE only; kill on flat before real-time. |
| **Scope creep into trading** | This layer writes only its own two tables; no import of executor/manager/governor; hard architectural wall. |
| **Vendor greeks error** | Prefer self-computed BS, tagged; don't trust vendor IV blindly. |
| **Null fatigue → forcing it** | Pre-registered bars (section 4/5); flat result is a valid, logged outcome, not a failure to overcome. |

---

## Final verdict

**READY FOR FLOW RESEARCH.**

Conditions binding the go:
- Historical replay POC FIRST (≈10 sessions, < ~$100). No real-time until lift shows.
- Observation only — two new tables, zero production/executor/governor/TC changes.
- Full validation battery + shuffle + direction-confound guard, pre-registered bars.
- Flat historical result = **DO NOT PROCEED**, documented null, cheapest kill.
- Strategy is a *separate* future proposal, only after all sample gates pass.
