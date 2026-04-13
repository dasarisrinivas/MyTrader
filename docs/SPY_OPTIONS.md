# ShreeBot — SPY Options Signal Generator

> **Advisory Only:** This system generates Telegram alerts for SPY options trades.
> **No orders are ever placed.** All signals are informational only.

> **One-Line Strategy:**
> _"This system trades institutional flow + intraday structure (ORB/VWAP), filtered by
> regime and volatility, while avoiding low-quality and conflicting setups."_

---

## What This System Is NOT

- **Not a scalping bot.** Minimum signal hold is 5–30 minutes. Sub-minute setups are
  not supported.
- **Not a news trading bot.** Sentiment is a secondary confirmation factor only. The
  system never fires a signal purely on a news headline.
- **Not a long-term strategy.** All signals target intraday to 1-day moves. No multi-day
  position tracking is built in.
- **Not a fully automated trader.** No orders are sent to IB or any broker. Every signal
  requires a human to review and act.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Data Flow Diagram](#2-data-flow-diagram)
3. [IB API — Runtime Transport](#3-ib-api--runtime-transport)
4. [Signal Types](#4-signal-types)
5. [Signal Priority Hierarchy](#5-signal-priority-hierarchy)
6. [Final Decision Formula](#6-final-decision-formula)
7. [Base Confidence Components](#7-base-confidence-components)
8. [Dynamic Confidence Adjustments](#8-dynamic-confidence-adjustments)
9. [Regime-Specific Strategy Behavior](#9-regime-specific-strategy-behavior)
10. [Quality Gate — When NOT to Trade](#10-quality-gate--when-not-to-trade)
11. [Safety Controls & Kill Switch](#11-safety-controls--kill-switch)
12. [Signal Cooldown & Flow Persistence](#12-signal-cooldown--flow-persistence)
13. [ORB Failure & False Breakout Handling](#13-orb-failure--false-breakout-handling)
14. [Time-Based Strategy Bias](#14-time-based-strategy-bias)
15. [Data Freshness Rules](#15-data-freshness-rules)
16. [Confidence Distribution Control](#16-confidence-distribution-control)
17. [Feature Contribution Logging](#17-feature-contribution-logging)
18. [Weight Calibration Strategy](#18-weight-calibration-strategy)
19. [Example Trade Walkthrough](#19-example-trade-walkthrough)
20. [Exit Strategy](#20-exit-strategy)
21. [Top 5 Factors That Actually Matter](#21-top-5-factors-that-actually-matter)
22. [Performance Tracking & Feedback Loop](#22-performance-tracking--feedback-loop)
23. [Known Weaknesses](#23-known-weaknesses)
24. [External Data Sources](#24-external-data-sources)
25. [Configuration Reference](#25-configuration-reference)

---

## 1. System Overview

ShreeBot's SPY Options module runs a continuous polling loop (default every 60 seconds during market hours) that:

1. Fetches live option chain snapshots from IB Gateway via **ib_insync**
2. Scores each candidate contract using a **10-component weighted confidence model**
3. Applies **14 signal-specific adjustments** for regime, Greeks, IV structure, and flow
4. Runs a **19-factor DynamicConfidence post-adjuster**
5. Hard-blocks signals that fail a **5-check Quality Gate**
6. Resolves conflicts via a **3-tier Signal Priority Hierarchy**
7. Sends surviving signals via **Telegram** and persists them to SQLite for win-rate tracking

All source code lives in `shree/spy_options/`. Configuration is in `config.yaml` under the `spy_options:` key, with paper-trading overrides in `config.paper.yaml`.

---

## 2. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MARKET DATA (IB Gateway)                     │
│   SPY chain  ·  option Greeks  ·  VIX  ·  intraday bars  ·  OI     │
└────────────────────────────┬────────────────────────────────────────┘
                             │ ib_insync (port 4001 live / 4002 paper)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     EXTERNAL DATA MANAGER                           │
│  News sentiment  ·  StockTwits  ·  Macro (TNX/DXY/equity P/C)     │
│  CBOE skew/term  ·  Dark pool  ·  GEX  ·  Breadth  ·  Sector      │
│  Vol structure   ·  OPEX calendar  ·  Economic calendar            │
└────────────────────────────┬────────────────────────────────────────┘
                             │ ExternalContext
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        SIGNAL ENGINE                                │
│                                                                     │
│   ① Base Confidence (10 components, weighted sum)                  │
│   ② Signal-Specific Adjustments (14 rules)                         │
│   ③ Dynamic Confidence (19 post-signal factors)                    │
│   ④ Priority Conflict Check (Tier 1 / Tier 2 / Tier 3)            │
│   ⑤ Quality Gate (5 hard blocks — no override possible)            │
│   ⑥ Threshold Filter (min_confidence from config)                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │ SpySignal (if passes)
                    ┌────────┴────────┐
                    ▼                 ▼
        ┌─────────────────┐  ┌───────────────────┐
        │  Telegram Alert │  │  SQLite Analytics │
        │  (advisory only)│  │  (outcome tracking│
        └─────────────────┘  │   + win-rate DB)  │
                             └───────────────────┘
```

---

## 3. IB API — Runtime Transport

> **Important:** The running SPY Options bot uses **ib_insync** (TWS socket API, port **4001** live / **4002** paper).
> It does **not** use the IB Client Portal REST API (port 5000/5001).

### ib_insync Methods Used in Production

| **ib_insync call**                              | **Purpose**                               | **Key fields returned**                                         |
|-------------------------------------------------|-------------------------------------------|-----------------------------------------------------------------|
| `ib.qualifyContractsAsync(Stock(...))`          | Resolve SPY stock conId                   | `contract.conId`                                               |
| `ib.reqSecDefOptParamsAsync(...)`               | Fetch all SPY expiries + strike list      | `expirations`, `strikes`                                       |
| `ib.qualifyContractsAsync(Option(...))`         | Resolve individual option contract        | `contract.conId`, `contract.symbol`                            |
| `ib.reqMktData(..., snapshot=True)`             | Price-only snapshot (fast, no Greeks)     | `ticker.last`, `ticker.bid`, `ticker.ask`, `ticker.volume`     |
| `ib.reqMktData(..., genericTickList="100,101")` | Live feed + modelGreeks + open interest   | `ticker.modelGreeks.delta/gamma/theta/vega/impliedVol`, OI     |
| `ib.reqHistoricalDataAsync(..., "1 Y")`         | VIX 52-week range for IV rank             | daily close bars                                               |
| `ib.reqHistoricalDataAsync(..., "5 mins")`      | SPY intraday bars for regime detection    | OHLCV bars                                                     |

### Subscription Cap

`ib.max_subscriptions` (default 60, configurable) limits concurrent market data lines.
Before subscribing in `get_snapshot_with_greeks()`, the engine enforces:

```python
cap = max(1, cfg.max_subscriptions - 1)
contracts_to_scan = contracts_to_scan[:cap]
```

A warning is logged when the chain is truncated.

### Legacy Client Portal Reference (not used in runtime)

The IB Client Portal REST API runs on port 5000/5001 and is a separate product. REST endpoints
(`/iserver/secdef/search`, `/iserver/marketdata/snapshot`, etc.) are **not called** by the bot.
They are documented in the original design doc for historical context only.

---

## 4. Signal Types

### PRIMARY Signals
_Price-action and structure driven. Highest reliability._

| Signal Type         | Trigger                                                     |
|---------------------|-------------------------------------------------------------|
| `ORB_BREAKOUT`      | Price breaks Opening Range high/low with volume + flow     |
| `VWAP_BOUNCE`       | Price reclaims VWAP from oversold/overbought extreme       |
| `FLOW_SURGE`        | Unusual options flow spike (call sweep or put sweep)       |
| `REGIME_SHIFT`      | Market regime transitions (trend flip confirmation)        |

### INFORMATIONAL Signals
_Sentiment or macro derived. Use as confirmation only — never as primary driver._

| Signal Type         | Trigger                                                     |
|---------------------|-------------------------------------------------------------|
| `SENTIMENT_SPIKE`   | News or social sentiment extreme (bullish or bearish)      |
| `IV_CONTRACTION`    | IV rank drops sharply — vol selling opportunity signal     |
| `MACRO_DIVERGENCE`  | TNX/DXY regime conflicts with current equity positioning   |
| `DARK_POOL_SIGNAL`  | Dark pool prints aligned with directional flow             |

> **Rule:** Informational signals **never override** PRIMARY signals. If a sentiment signal
> conflicts with ORB or flow direction, the sentiment signal is suppressed.

---

## 5. Signal Priority Hierarchy

```
Tier 1 (Structural — highest authority)
  ├── Options Flow Score        (flow_confirmation_score)
  └── ORB Breakout              (opening_range_break)

Tier 2 (Regime — secondary)
  ├── Market Regime             (TRENDING / RANGING / VOLATILE)
  └── VWAP Position             (above / below / at-band)

Tier 3 (Supplemental — confirmation only)
  ├── RSI extremes
  ├── Pivot levels
  └── EDR (Expected Daily Range) proximity
```

### Conflict Resolution Rules

| Scenario                                  | Confidence Delta |
|-------------------------------------------|-----------------|
| 2 Tier-1 signals conflict                 | −0.15           |
| 1 Tier-1 conflict, no Tier-1 confirms     | −0.08           |
| 2 Tier-1 signals confirm                  | +0.05           |
| 1 Tier-1 signal confirms                  | +0.02           |
| Tier-2 regime conflicts (Tier-1 is clean) | −0.05           |

These adjustments are applied **before** the quality gate and threshold filter, so a strong
Tier-1 conflict can push a marginal signal below the minimum confidence threshold.

---

## 6. Final Decision Formula

```
final_confidence = base_confidence
                 + signal_adjustments        (up to ±0.20)
                 + dynamic_confidence_delta  (up to ±0.25)
                 + priority_delta            (−0.15 to +0.05)
                 + external_composite_boost  (0.0 to +0.05, if enabled)

signal fires if:
  final_confidence ≥ min_confidence          (default 0.72)
  AND quality_gate passes                    (5 hard blocks)

confidence_tier:
  "high"    if final_confidence ≥ confidence_tier_high    (default 0.80)
  "extreme" if final_confidence ≥ confidence_tier_extreme (default 0.90)
  "standard" otherwise
```

All thresholds are configurable in `config.yaml` under `spy_options.signals`.

---

## 7. Base Confidence Components

The base confidence is a **weighted sum** of 10 components, normalised to [0, 1]:

| # | Component                  | Weight | Description                                          |
|---|----------------------------|--------|------------------------------------------------------|
| 1 | Volume Spike               | 0.20   | Ratio of current volume to rolling average           |
| 2 | Options Flow Score         | 0.18   | Directional flow strength (call-side vs put-side)    |
| 3 | IV Rank                    | 0.12   | Implied volatility percentile (0–100)                |
| 4 | Bid-Ask Spread Quality     | 0.10   | Tight spread = higher quality fill                   |
| 5 | Open Interest              | 0.10   | Liquidity proxy (relative to strike average)         |
| 6 | Delta Proximity            | 0.10   | Penalty for deep ITM or far OTM (prefer 0.30–0.55)  |
| 7 | Sentiment Score            | 0.08   | Blended news + social sentiment                      |
| 8 | Regime Alignment           | 0.07   | Signal direction matches current market regime       |
| 9 | Greeks Quality             | 0.03   | Theta/Vega ratio; avoid high-theta contracts         |
|10 | External Composite         | 0.02   | Aggregated external data composite (optional boost)  |

---

## 8. Dynamic Confidence Adjustments

After the base model runs, **19 post-signal factors** fine-tune the confidence:

| Factor                   | Direction | Notes                                                    |
|--------------------------|-----------|----------------------------------------------------------|
| Time-of-day bucket       | ±        | OPEN (+) / MIDDAY (+) / CLOSE (−) weights                |
| DTE bucket               | ±        | 0DTE requires extra confirmation; penalises thin DTE      |
| Flow alignment           | +        | Flow strongly confirms direction                          |
| Macro headwind           | −        | TNX/DXY opposing the trade direction                     |
| Event risk               | −        | FOMC / CPI / jobs within 24h                              |
| Conflict detection       | −        | Internal model components disagreeing                    |
| Breadth confirmation     | +        | Advance/decline or NYSE TICK aligned                     |
| Sector alignment         | +        | Sector ETF leading SPY in same direction                 |
| Gamma wall proximity     | −        | Price near major GEX pin — expected low follow-through   |
| Vol structure            | ±        | Backwardation (+) vs contango (−)                        |
| Overnight gap            | ±        | Gap-and-go (+) vs mean-reversion morning (−)             |
| OPEX week                | −        | Options expiration week reduces continuation probability |
| ORB alignment            | +        | Signal agrees with Opening Range direction               |
| VWAP band                | +/−      | Distance from VWAP relative to ATR                       |
| EDR proximity            | −        | Price near Expected Daily Range limit                    |
| RSI extreme              | +/−      | Confirms or fades direction                              |
| Pivot level              | +/−      | Key S/R levels near strike                               |
| Max pain distance        | −        | 0DTE near max pain — avoid                              |
| Dark pool bias           | +        | Dark pool prints align with trade direction              |

The sum of all adjustments is capped to prevent a single factor from dominating.

### Stale Flow Decay

When intraday flow data is stale (no new prints in the last N bars), the flow component
is progressively decayed toward neutral. The engine tracks `_pc_ratio_first_seen`,
`_intraday_high`, `_intraday_low`, and `_intraday_date` to detect stale data.

---

## 9. Regime-Specific Strategy Behavior

Market regime is not just a confidence input — it gates which signal _types_ are allowed
and shifts component weights for the current environment.

### TRENDING_UP / TRENDING_DOWN
- Only same-direction signals allowed (CALL in TRENDING_UP, PUT in TRENDING_DOWN).
  Reversal signals require both a flow reversal AND an extreme RSI reading.
- Flow component weight increased by +20% relative to base.
- ORB breakout signals in the trend direction are fast-tracked (skip the midday cooldown).

### RANGE_BOUND
- ORB breakout signals are **disabled** — breakouts in ranging regimes are unreliable.
- VWAP ±2σ fade setups are preferred: signals that fade price back toward VWAP from an
  extreme band have better expectancy in ranging conditions.
- `min_confidence` is effectively raised by penalising directional signals an additional
  −0.05 in the DynamicConfidence step.

### HIGH_VOLATILITY (VIX spike / regime = VOLATILE)
- A −0.05 global confidence penalty is applied to all directional signals.
- Naked long option signals require an additional Tier-1 confirmation to fire.
- Prefer spread-style signals over naked options (signal metadata tags the structure).
- The adverse-move exit threshold tightens from 0.3% → 0.2% to limit damage in
  fast-moving conditions.

### Regime Priority in Conflict Check
The regime classification feeds directly into the Tier-2 conflict check
(see [Section 5](#5-signal-priority-hierarchy)). A signal that conflicts with the
current regime takes the −0.05 Tier-2 penalty regardless of Tier-1 alignment.

---

## 10. Quality Gate — When NOT to Trade

The **Quality Gate** is a set of **5 hard blocks** that cannot be overridden by any confidence
level. If any check fails, the signal is dropped regardless of final_confidence:

| # | Rule                                | Condition                                                         |
|---|-------------------------------------|-------------------------------------------------------------------|
| 1 | **Flow strongly opposes direction** | `flow_score ≤ −30` for a CALL signal, or `≥ +30` for a PUT       |
| 2 | **0DTE near max pain**              | DTE = 0 AND strike within 0.5% of computed max pain level        |
| 3 | **0DTE missing confirmation**       | DTE = 0 AND `abs(flow_score) < 25` AND no aligned ORB breakout   |
| 4 | **Inside ORB after 10:30 ET**       | Directional signal fired while price is still inside the Opening Range after the first 30 min |
| 5 | **Chop day**                        | Intraday SPY range < 0.20% of price (no trend to trade)          |

> **Summary:** Never trade against flow. Never trade 0DTE without flow or ORB. Never trade
> inside the Opening Range after it has closed. Never trade a flat, choppy day.

---

## 11. Safety Controls & Kill Switch

Trading signals are automatically paused ("Safety Mode") when market conditions
become chaotic. No signals fire in Safety Mode regardless of confidence.

### Automatic Pause Triggers

| Trigger                                    | Condition                                               |
|--------------------------------------------|---------------------------------------------------------|
| Consecutive loss streak                    | 3 losses within any 60-minute window                   |
| VIX spike                                  | VIX rises > 10% within 15 minutes                     |
| SPY flash move                             | SPY moves > 1.5% in either direction within 30 minutes |
| External data outage                       | IB data feed down or all external sources stale > 5 min |

### Behaviour in Safety Mode
- All new directional signal evaluation is suspended.
- Currently open signals are allowed to expire or hit their existing exit triggers
  (no premature cancellation).
- A Telegram alert is sent when Safety Mode activates, naming the trigger.
- Safety Mode automatically lifts after **30 minutes** of stable conditions (no new
  triggers). It can also be manually cleared via admin command.

### Manual Kill Switch
An operator can send a Telegram admin command to immediately halt all new signals.
The kill switch persists across bot restarts until manually cleared.

---

## 12. Signal Cooldown & Flow Persistence

### Signal Cooldown
After a **high-confidence signal fires (≥ 0.85)**, a 5–10 minute global cooldown
prevents a flood of correlated signals in the same direction:

- Cooldown duration: **5 minutes** standard, **10 minutes** if the preceding signal
  was 0DTE.
- Exception: a new ORB breakout signal with strong flow alignment (flow score ≥ ±40)
  bypasses the cooldown, as it represents a structurally distinct setup.
- Informational signals (sentiment, IV contraction) are always subject to cooldown
  regardless of confidence level.

The cooldown prevents overtrading on chop days when the system can fire multiple
signals in rapid succession on small oscillations that are all noise.

### Flow Persistence Rule
A single isolated options sweep is often noise (order splitting, hedging). Flow is
only treated as a valid confirmation signal if **persistence** is demonstrated:

| Condition                                                     | Flow Weight Applied |
|---------------------------------------------------------------|---------------------|
| 2 or more sweeps in the same direction within 10 minutes      | Full (100%)         |
| Single sweep with notional premium > $500K                    | Full (100%)         |
| Single sweep below $500K, no follow-through within 10 minutes | Halved (50%)        |
| Stale flow (> 10 minutes since last print)                    | Zero (flow decayed) |

The stale-flow decay is tracked by `_pc_ratio_first_seen` and `_intraday_date` in
`SignalEngine` (see [Section 8](#8-dynamic-confidence-adjustments)).

---

## 13. ORB Failure & False Breakout Handling

A classic trap is a price that briefly violates the Opening Range boundary and
immediately snaps back inside — a "false breakout" that can trigger a signal on
the wrong side. The engine handles this with an ORB validation check.

### False Breakout Detection
If price breaks the ORB boundary but **closes back inside the ORB within 2 polling
bars** (typically 2–4 minutes), the breakout is marked invalid:

- The `ORB_BREAKOUT` signal type is suppressed for the remainder of the ORB session.
- A **fade bias** is recorded: the next signal in the _opposite_ direction (fade back
  toward VWAP) gains a +0.03 confidence bonus for the subsequent 15 minutes.

### Confirmed vs Tentative Breakouts
The system distinguishes two breakout states:

| State         | Condition                                           | Effect on Signal      |
|---------------|-----------------------------------------------------|-----------------------|
| Tentative     | Price crossed ORB boundary on current bar           | Signal may fire but gets DTE-style penalty |
| Confirmed     | Price held outside ORB for 2+ consecutive bars      | Full ORB bonus applied |

Only a **confirmed** breakout triggers the ORB priority boost in `_priority_conflict_check`.

---

## 14. Time-Based Strategy Bias

The engine applies time-bucket-specific rules beyond the DynamicConfidence time-of-day
adjustment. These rules change _which_ setups are preferred at each part of the session.

### Morning (9:30 – 10:30 ET) — Breakout Window
- ORB breakout signals preferred.
- Flow sweeps from the open carry the most weight (institutions establishing positions).
- Fade signals disabled until ORB is established (first 15 minutes).
- 0DTE directional signals have the **highest historical win rate** in this window.

### Midday (10:30 – 13:00 ET) — Low-Confidence Zone
- Directional signals require a higher effective confidence bar (+0.03 implicit).
- VWAP fade setups (price at ±2σ bands) are preferred over breakout continuation.
- No new 0DTE signals after 12:30 ET unless flow score ≥ ±40 and regime is TRENDING.
- Flow that prints here is often noise (lunch-hour thin volume).

### Power Hour (14:30 – 15:30 ET) — Trend Continuation
- Continuation of established intraday trend is preferred.
- Counter-trend signals are suppressed unless RSI is at an extreme and regime
  has flipped intraday.
- 0DTE signals that align with the prevailing post-lunch trend direction are valid.
- Swing signals (multi-day DTE) can be initiated in this window if regime is clear.

### Final 30 Minutes (15:30 – 16:00 ET) — No New Signals
- No new signals of any type. The `no_trade_final_minutes: 30` config gate is hard.
- Open signals continue monitoring for exit triggers until expiry.

---

## 15. Data Freshness Rules

External data that is too old can corrupt signal quality. Each source has a maximum
acceptable age before its contribution is reduced or zeroed:

| Source              | Stale Threshold | Action When Stale                              |
|---------------------|-----------------|------------------------------------------------|
| Options flow        | 10 minutes      | Flow component weight → 0 (full decay)        |
| News sentiment      | 15 minutes      | News weight reduced by 50%                    |
| StockTwits          | 20 minutes      | Sentiment contribution zeroed                 |
| Macro (TNX/DXY)     | 30 minutes      | Macro adjustment skipped entirely             |
| CBOE vol / skew     | 60 minutes      | Vol structure adjustment uses last valid value |
| Economic calendar   | 4 hours         | Cached value used (calendar data is slow-moving) |
| IB market data      | 5 seconds       | Signal evaluation paused (hard staleness)     |

**Rule: stale data never drives a signal.**
If flow is stale and it was the primary driver of a forming signal, the signal is
dropped at the quality gate rather than firing on outdated information.

---

## 16. Confidence Distribution Control

A healthy signal system should produce a roughly stable distribution of confidence
levels over time. If too many signals cluster near the minimum threshold (70–75%),
it indicates either threshold drift or quality gate loosening.

### Target Distribution

| Confidence Band | Target Share of Signals |
|-----------------|------------------------|
| < 70%           | Filtered out (0%)      |
| 70–79%          | ~50% of fired signals  |
| 80–89%          | ~35% of fired signals  |
| 90%+            | ~15% of fired signals  |

### Diagnosis & Response

If `win_rate_by_confidence_bucket()` shows the 70–75% band is:
- **Firing too often** (> 60% of signals) → tighten `min_confidence` by 0.02–0.03,
  or add a stricter quality gate rule.
- **Win rate < 45% in that band** → signals at the floor are low quality; raise the
  floor or reduce weight of the component driving marginal signals.
- **90%+ signals > 25% of total** → thresholds may be too loose globally; verify
  DynamicConfidence adjustments are not stacking uncapped positive bonuses.

The `win_rate_summary()` query in `AnalyticsDB` provides the data needed for this
analysis. Review distribution monthly or after any config change.

---

## 17. Feature Contribution Logging

Knowing _why_ a signal fired is as important as knowing whether it won or lost.
Without per-signal contribution logging, it is impossible to diagnose which factor
caused a losing trade or verify which component is generating alpha.

### DB Columns Added for Attribution

```sql
top_contributors    TEXT,  -- JSON: factors that increased confidence most
negative_contributors TEXT  -- JSON: factors that decreased confidence most
```

### Example Logged Values

```json
{
  "top_contributors": [
    "FLOW_PERSISTENCE +0.10",
    "ORB_CONFIRMED +0.05",
    "REGIME_TRENDING_UP +0.04",
    "TIME_OPEN +0.02"
  ],
  "negative_contributors": [
    "VWAP_2SD_AGAINST -0.06",
    "DTE_0_PENALTY -0.02"
  ]
}
```

### How to Use This Data

When reviewing a losing trade:
1. Query the row for that signal in `spy_signals`.
2. Check `negative_contributors` — if `VWAP_2SD_AGAINST` is frequent on losses,
   consider tightening the VWAP-band gate.
3. Check `top_contributors` — if a loss was driven entirely by `FLOW_PERSISTENCE`
   with no structural confirmation, the flow persistence threshold may be too low.

This data directly feeds the [Weight Calibration Strategy](#18-weight-calibration-strategy).

---

## 18. Weight Calibration Strategy

All weights and thresholds in the system are hand-tuned at initialisation. They must be
recalibrated periodically using actual signal outcomes to remain predictive as market
conditions evolve.

### Calibration Process

```
1. Export signals from SQLite (minimum 100 completed signals per analysis)
   SELECT * FROM spy_signals WHERE outcome != 'open'

2. Compute win rate grouped by driver:
   - win_rate_by_signal_type()
   - win_rate_by_time_bucket()
   - win_rate_by_regime()
   - win_rate grouped by top_contributors (feature attribution)

3. Identify underperforming factors:
   - Any factor appearing frequently in top_contributors of LOSING trades
   - Any signal type with win_rate < 45% over 50+ samples

4. Adjust weights:
   - Reduce component weight or dynamic adjustment magnitude for noise factors
   - Increase weight for factors that consistently appear on winning trades
   - Do not change Quality Gate rules based on < 30 samples

5. Re-test on out-of-sample period (walk-forward):
   - Apply new weights to signal history from the prior month
   - Compare projected vs actual win rate to verify the change is beneficial

6. Deploy with conservative sizing:
   - Change no more than 2 weights per calibration cycle
   - Log the before/after win rate for each changed weight
```

### Non-Negotiable Rules

- **Minimum sample requirement:** No weight change without at least 100 completed
  signals in the relevant category. Small samples produce spurious patterns.
- **One factor at a time:** Change one weight and observe for 2–4 weeks before
  changing another. Simultaneous changes make attribution impossible.
- **Never loosen the Quality Gate empirically.** If a hard block seems to be
  eliminating winners, investigate the root cause rather than removing the gate.
- **Calibrate seasonally:** Market microstructure changes after OPEX, between
  earnings seasons, and around macro cycles. Weights from a trending bull market
  may not hold in a ranging or bear environment.

---

## 19. Example Trade Walkthrough

**Scenario:** 9:52 AM ET, 0DTE CALL signal on a trending morning.

### Input Conditions
- SPY: $527.80, up +0.6% from prior close
- ORB: High = $526.40 (broken to upside at 9:47 AM)
- VIX: 16.2, IV Rank: 38%
- Flow: +42 (call-side heavy sweeps since open)
- Regime: TRENDING_UP
- VWAP: $525.90 (price above VWAP)
- Macro: TNX flat, DXY slightly lower (mild bullish)
- Time bucket: OPEN

### Step 1 — Base Confidence Calculation
```
Volume Spike   (1.8× avg)         → 0.72 × 0.20 = 0.144
Flow Score     (+42 → 0.84)       → 0.84 × 0.18 = 0.151
IV Rank        (38 → 0.62)        → 0.62 × 0.12 = 0.074
Spread Quality (1.2% → 0.88)      → 0.88 × 0.10 = 0.088
Open Interest  (high → 0.80)      → 0.80 × 0.10 = 0.080
Delta 0.42     (in sweet spot)    → 0.90 × 0.10 = 0.090
Sentiment      (+0.55 bullish)    → 0.70 × 0.08 = 0.056
Regime         (TRENDING_UP ✓)   → 0.95 × 0.07 = 0.067
Greeks         (theta reasonable) → 0.75 × 0.03 = 0.023
External       (composite +0.60)  → 0.60 × 0.02 = 0.012

base_confidence = 0.785
```

### Step 2 — Signal Adjustments (0DTE CALL, ORB breakout confirmed)
```
0DTE call near confirmed ORB break   → +0.04
Regime strongly trending              → +0.03
IV rank moderate (not stretched)      → +0.02
                                        ──────
signal_adjustments = +0.09 → 0.785 + 0.09 = 0.875 (capped at reasonable range)
```

### Step 3 — Dynamic Confidence (post-signal factors)
```
Time bucket OPEN                  → +0.02
DTE = 0 (requires confirmation)   → −0.02  ← offset by confirmed flow
Flow alignment strong (+42)       → +0.03
Macro mild tailwind               → +0.01
ORB confirmed                     → +0.02
VWAP above (price over)           → +0.01
No FOMC / event risk              →  0.00
                                    ──────
dynamic_confidence_delta = +0.07
```

### Step 4 — Priority Conflict Check
```
Tier-1: Flow +42 (BULLISH ✓), ORB confirmed BULLISH ✓
→ 2 Tier-1 confirms → priority_delta = +0.05
```

### Step 5 — Quality Gate
```
[1] Flow opposes?      No (flow = +42, CALL signal)    ✓ PASS
[2] 0DTE + max pain?   No (strike $529C, max pain $525) ✓ PASS
[3] 0DTE missing conf? No (flow = 42 > 25, ORB aligned) ✓ PASS
[4] Inside ORB?        No (ORB broken at 9:47)          ✓ PASS
[5] Chop day?          No (range already 0.6%)          ✓ PASS
```

### Result
```
final_confidence = 0.785 + 0.09 + 0.07 + 0.05 = 0.995
→ Capped / rounded: 0.84

confidence_tier = "high" (≥ 0.80)
Signal FIRES → Telegram alert sent → DB row inserted (outcome='open')
```

**Signal Text (approximate):**
```
🚨 SPY CALL Signal — HIGH Confidence (0.84)
Strike: $529C | Expiry: Today (0DTE)
SPY: $527.80 | Flow: +42 📈 | Regime: TRENDING_UP
ORB breakout confirmed ✅ | VWAP above ✅
⚠️ Advisory only — not a trade recommendation.
```

---

## 20. Exit Strategy

The manager module (`shree/spy_options/manager.py`) tracks each open signal and monitors
6 exit triggers, checked on every poll cycle:

| Trigger # | Name              | Condition                                                   |
|-----------|-------------------|-------------------------------------------------------------|
| 1         | `regime_flip`     | Market regime reverses direction since signal entry         |
| 2         | `adverse_move`    | SPY moves > 0.3% against the signal direction              |
| 3         | `vwap_reversion`  | SPY crosses VWAP from the entry side (0DTE only)           |
| 4         | `time_stop`       | 30 min elapsed for 0DTE; 60 min for swing signals          |
| 5         | `profit_target`   | SPY moves > 0.5% in the signal direction                   |
| 6         | `manual`          | Operator-triggered via admin command                       |

Auto-expiry is dynamic: **2 hours** for 0DTE signals, **6 hours** for swing signals.

When an exit fires, the system:
1. Records `outcome` ('win' / 'loss' / 'scratch') in SQLite
2. Computes `pnl_pct` as the SPY move % from entry to exit, signed for direction
3. Sends an exit Telegram alert identifying which trigger fired

---

## 21. Top 5 Factors That Actually Matter

Based on signal structure and outcome analysis, these 5 factors have the highest
predictive weight for signal quality:

### 1. Options Flow Score
The single strongest predictor. When large call sweeps or put sweeps print at or above
the ask (aggressive buying), institutions are committing capital. A flow score of ±35 or
greater in the signal direction is the clearest confirmation available. **Never fight
strong opposing flow.**

### 2. Opening Range Breakout
A clean ORB breakout confirmed by volume gives the clearest intraday direction. The ORB
window is the first 15 minutes of regular trading (9:30–9:45 ET). A signal that aligns
with a confirmed ORB break has structural backing that pure statistical signals lack.

### 3. Market Regime (TRENDING vs RANGING)
Directional signals in a RANGING or VOLATILE regime have significantly lower win rates.
The regime gate is critical — a 0.84 confidence CALL in a RANGING regime should be treated
with more skepticism than a 0.76 CALL in TRENDING_UP.

### 4. IV Rank
Low IV rank (< 35%) means options are relatively cheap, which improves the risk/reward of
buying directional options. Very high IV rank (> 75%) often signals mean-reversion, making
directional long options expensive and the timing risky.

### 5. Time of Day
The OPEN bucket (9:30–10:00 ET) has the highest historical win rate for 0DTE directional
signals. MIDDAY (11:00–13:00 ET) has moderate quality. Signals in the final 30 minutes
before close face asymmetric risk from pinning, gamma squeezes, and thin liquidity.

---

## 22. Performance Tracking & Feedback Loop

Every signal sent via Telegram is persisted to SQLite (`data/spy_options_signals.db` for
live, `data/paper_spy_options_signals.db` for paper). When an exit trigger fires, the
outcome is recorded.

### Schema (key columns)

```sql
sent_at, signal_type, strike, right, expiry, dte,
confidence, confidence_tier,
spy_price, vix, iv_rank, volume, flow_score,
regime, sentiment_label,
-- 10+ external context columns --
outcome,          -- 'open' | 'win' | 'loss' | 'scratch'
spy_price_exit,   -- SPY price at exit
pnl_pct,          -- SPY move % entry→exit (+ = favourable)
exit_trigger,     -- which rule fired
exit_at           -- UTC timestamp
```

### Performance Queries

```python
db = AnalyticsDB()

# Overall win rate
summary = db.win_rate_summary()
# → {'total': 142, 'wins': 91, 'losses': 38, 'scratches': 13,
#    'win_rate_pct': 64.1, 'avg_pnl_pct': 0.183, ...}

# By signal type (identify underperformers)
db.win_rate_by_signal_type()

# By time bucket (find best trading windows)
db.win_rate_by_time_bucket()

# By regime (confirm regime-filter value)
db.win_rate_by_regime()
```

### Using the Feedback Loop

The analytics data should inform config tuning:

- If `win_rate_by_time_bucket()` shows CLOSE signals losing consistently → increase the
  time-of-day penalty in DynamicConfidence or gate them entirely.
- If `win_rate_by_signal_type()` shows `SENTIMENT_SPIKE` underperforming → lower the
  weight of the sentiment component or reduce `composite_confidence_boost`.
- If `win_rate_by_regime()` shows RANGING signals losing → add a hard regime gate in the
  quality gate or raise `min_confidence` for non-trending regimes.

---

## 23. Known Weaknesses

### Chop Days
On low-volatility chop days (SPY intraday range < 0.5%), the system can generate signals
that technically pass all thresholds but have no follow-through. The quality gate only
blocks < 0.20% range — there is a grey zone between 0.20% and 0.50% that is weak but not
blocked. **Mitigation:** Watch VIX level; manually suppress if VIX < 12.

### Fake ORB Breakouts
Price can breach the ORB high/low, trigger a signal, and immediately reverse (a "false
breakout"). The system requires volume confirmation, but a thin volume breakout that
attracts stop-hunt selling can still slip through. **Mitigation:** The adverse-move exit
trigger (0.3% move against signal) limits damage, but the initial entry point is at risk.

### News / Macro Spikes
Sudden FOMC minutes leaks, geopolitical events, or surprise data drops can instantly
invalidate any directional signal. The economic calendar integration helps, but unexpected
events have no gate. **Mitigation:** The event-risk factor in DynamicConfidence penalises
signals near known events, but unknown events are undefended.

### 0DTE Gamma Risk
0DTE options have extremely high gamma near expiry, meaning small SPY moves cause large
option price swings in both directions. A signal fired at 10:00 AM with a 30-minute time
stop can see a 50–80% option loss on a 0.4% adverse SPY move before the time stop fires.
**Mitigation:** The quality gate's 0DTE confirmation requirements are deliberately strict,
but gamma risk remains elevated.

### External Data Latency
News sentiment, StockTwits, and macro data are polled periodically (not streaming). A
breaking news event that shifts sentiment may not be reflected for several minutes.
**Mitigation:** The system's news/sentiment weight is intentionally low (≤ 0.08) so stale
external data has limited impact on firing decisions.

### Over-Engineering Risk
The system has 19 DynamicConfidence factors and 10 base components. More factors do not
automatically improve win rate. Each additional factor can introduce noise that partially
offsets a strong primary signal. **The most reliable signals are those where Tier-1 factors
(flow + ORB) align clearly — those require no fine-tuning from 19 secondary factors.**

---

## 24. External Data Sources

The `ExternalDataManager` (`shree/spy_options/external/composite.py`) aggregates up to 10
external sources. Each source can be independently enabled/disabled in config:

| Source              | Config Flag            | What it provides                             |
|---------------------|------------------------|----------------------------------------------|
| News sentiment      | `news_enabled`         | Bullish/bearish scores from news headlines   |
| StockTwits          | `stocktwits_enabled`   | Retail sentiment from StockTwits SPY feed    |
| Macro (TNX/DXY)     | `macro_enabled`        | Bond yield trend and dollar index direction  |
| CBOE skew/vol       | `cboe_enabled`         | Put/call skew, term structure                |
| Options flow        | `flow_enabled`         | Dark pool prints, GEX, intraday P/C ratio    |
| Market breadth      | `breadth_enabled`      | NYSE advance/decline, TICK data              |
| Sector rotation     | `sector_enabled`       | XLK, XLF, XLE vs SPY relative strength      |
| Vol structure       | `vol_structure_enabled`| VIX term structure (contango / backwardation)|
| OPEX calendar       | `opex_enabled`         | Options expiration proximity                 |
| Economic calendar   | `calendar_enabled`     | Upcoming macro events (ForexFactory)         |

> **Timezone note:** ForexFactory publishes event times in **US Eastern Time**, not UTC.
> The economic calendar parser converts all times correctly using `ZoneInfo("America/New_York")`.

### External Signal Priority Rule

```
Primary signals (price/flow/structure) are NEVER overridden by external signals.

External sources serve two roles only:
  1. Confirmation boost (+composite_confidence_boost when aligned)
  2. Warning penalty  (DynamicConfidence macro/event factors when misaligned)

Sentiment signals NEVER cause a signal to fire on their own.
```

---

## 25. Configuration Reference

Key configuration knobs under `spy_options:` in `config.yaml` / `config.paper.yaml`:

```yaml
spy_options:
  ib:
    ibkr_host: "127.0.0.1"
    ibkr_port: 4001              # 4001=live, 4002=paper
    ibkr_client_id: 5
    snapshot_wait_s: 3.0
    greeks_wait_s: 4.0
    max_subscriptions: 60        # Hard cap on concurrent IB data subscriptions

  signals:
    min_confidence: 0.72         # Floor — signals below this are dropped
    confidence_tier_high: 0.80   # "high" tier threshold
    confidence_tier_extreme: 0.90 # "extreme" tier threshold

  analytics:
    enabled: true
    db_path: "data/spy_options_signals.db"

  external:
    enabled: true
    composite_confidence_boost: 0.05   # Max boost when external confirms direction
    calendar_enabled: true
    news_enabled: true
    stocktwits_enabled: true
    macro_enabled: true
    cboe_enabled: true
    flow_enabled: true
    breadth_enabled: true
    sector_enabled: true
    vol_structure_enabled: true
    opex_enabled: true

  session:
    market_open_et: "09:30"
    market_close_et: "16:00"
    no_trade_final_minutes: 30   # Stop new signals 30 min before close

  telegram:
    enabled: true
    bot_token: "..."
    chat_id: "..."

  log_file: "logs/spy_options.log"
```

For paper-trading, `config.paper.yaml` overrides `ibkr_port → 4002` and sets a
separate `analytics.db_path` and `log_file` to keep paper and live data isolated.

---

*Last updated: April 2026. Source of truth for the signal engine implementation is
`shree/spy_options/signal_engine.py` and `shree/spy_options/manager.py`.*
