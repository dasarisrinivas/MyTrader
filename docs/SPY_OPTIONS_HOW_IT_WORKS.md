# SPY Options Bot — How It Works

## The Basic Idea

Every 60 seconds during market hours (9:35–3:45 ET), the bot:
1. Fetches SPY price, VIX, 5-minute bars, and option chain data from IB Gateway
2. Classifies the market regime and scores sentiment
3. Fetches Greeks (delta, gamma, theta, vega, IV) for every tracked contract
4. Refreshes external signals (news, macro, social, economic calendar)
5. Runs a weighted confidence model across all sources to score signals
6. Sends Telegram alerts only — no trades placed, ever

---

## Poll Cycle (What Happens Every 60 Seconds)

```
1. Fetch SPY price          (IB snapshot)
2. Fetch VIX                (IB snapshot)
3. Fetch SPY 5-min bars     (IB historical — for regime + sentiment)
4. Classify market regime   (EMA, ATR, VWAP — pure math)
5. Score IB sentiment       (VIX trend + VWAP position + EMA slope)
6. Compute IV rank          (VIX vs 52-week range, cached at startup)
7. Refresh external signals (TTL-gated — most sources don't re-fetch every poll)
   a. Economic calendar     (Forex Factory JSON — once per day)
   b. News RSS sentiment    (VADER scoring — every 10 min)
   c. StockTwits retail     (free API — every 10 min)
   d. Reddit WSB/investing  (asyncpraw — every 15 min, disabled by default)
   e. Macro signals         (yfinance: 10Y yield, DXY, oil, gold — daily)
   f. CBOE P/C ratio        (daily CSV — once per day)
8. For each of 2 expiries:
   a. Resolve option conids (cached after day 1)
   b. Fetch price + Greeks  (snapshot=False, explicit cancel after read)
   c. Apply liquidity filter (OI, spread, volume)
   d. Run signal engine     (7 signal types, weighted confidence)
   e. Apply event risk modifier (penalise directional, boost straddle)
9. Deduplicate signals
10. Send Telegram alerts
11. Persist to SQLite analytics DB
```

**Total time per poll: ~14–19 seconds** (well within the 60s window).

---

## Market Regime Detection

Before any signal fires, the bot classifies the current market environment using 5-minute SPY bars:

| Regime | What it means | Signals favoured |
|---|---|---|
| **TREND_UP** | EMA9 > EMA21, positive slope, SPY above VWAP | Call sweeps, Bull Call Spreads |
| **TREND_DOWN** | EMA9 < EMA21, negative slope, SPY below VWAP | Put sweeps, Bear Put Spreads |
| **RANGE_BOUND** | No clear trend (default) | Premium selling (Iron Condor) |
| **HIGH_VOL** | VIX > 26 or ATR expanded 2× median | Spreads over naked longs |
| **LOW_VOL** | VIX < 12 and ATR compressed 0.5× median | Long premium (cheap options) |
| **NEWS_DRIVEN** | ATR blow-up > 2.5× recent median | Extreme caution — noted in alert |

---

## IB Sentiment Score

A single score from **-100 (bearish) to +100 (bullish)** computed from IB data only — no Twitter, no Reddit, no external APIs:

| Component | Weight | How it's calculated |
|---|---|---|
| VIX trend | 40% | Rising VIX > 5% above recent avg = -40; falling = +40 |
| SPY vs VWAP | 35% | Distance above/below VWAP, normalized by ATR |
| EMA slope | 25% | Rate of EMA-9 change, normalized by ATR |

Labels: **BULLISH** (>+25), **NEUTRAL** (-25 to +25), **BEARISH** (<-25)

---

## External Signals

Six free, publicly available data sources add context on top of IB data:

### 1. Economic Calendar (Forex Factory JSON)
- Fetched once per day from `cdn-nfs.faireconomy.media/ff_calendar_thisweek.json` — no API key
- Tracks US High-impact events (FOMC, CPI, NFP, GDP, etc.)
- **Event risk window**: ±30 minutes around any High-impact event triggers `event_risk=True`
- **Effect on signals**:
  - Directional signals (CALL/PUT SWEEP, spreads, P/C Ratio) → confidence **−30%** within window
  - LONG STRADDLE → confidence **+10%** within window (big move expected)

### 2. News RSS Sentiment (VADER)
- Feeds: Yahoo Finance, MarketWatch, CNBC, Reuters
- Only headlines/summaries mentioning SPY, S&P, Fed, inflation, yields, etc. are scored
- VADER compound score averaged across all relevant articles (−1.0 to +1.0)
- Refreshed every **10 minutes**, no API key required

### 3. StockTwits Retail Sentiment
- Free public API: `api.stocktwits.com/api/2/streams/symbol/SPY.json` — no authentication
- Counts Bullish/Bearish sentiment tags on the last 30 messages
- Returns `bullish_pct`, `bearish_pct`, and normalised score (−1.0 to +1.0)
- Refreshed every **10 minutes**

### 4. Reddit Sentiment (asyncpraw)
- Subreddits: r/wallstreetbets, r/investing, r/stocks
- Filters posts mentioning SPY, SPX, 0DTE, options, etc.
- VADER-scores post titles
- **Disabled by default** — requires Reddit app credentials (free to create)
- Refreshed every **15 minutes** when enabled

### 5. Macro Signals (yfinance)
- Tickers: `^TNX` (10Y yield), `DX-Y.NYB` (DXY), `CL=F` (crude oil), `GC=F` (gold)
- Trend = last close vs 5-day average: RISING / FALLING / FLAT
- Refreshed **once per day** (EOD data)
- Computes `spy_headwind` score: rising yields + rising dollar = headwind for equities

### 6. CBOE Equity Put/Call Ratio
- Daily CSV from `cdn.cboe.com` — no authentication
- **Contrarian interpretation**: high P/C (> 1.2) = too much bearish hedging → slight bullish bias; low P/C (< 0.7) = complacency → slight bearish bias
- Refreshed **once per day**

---

## External Composite Score

All six sources are combined into a single **composite score** (−1.0 to +1.0):

| Source | Default Weight | Note |
|---|---|---|
| News RSS sentiment | 30% | Always available |
| StockTwits retail | 20% | Available unless rate-limited |
| Macro headwind | 20% | Daily refresh |
| CBOE P/C contrarian | 20% | Daily refresh |
| Reddit | 10% | Only when enabled |

Weights are **dynamically normalised** — if a source is unavailable, its weight is redistributed to the others. A score near zero is shown if fewer than 2 sources are available.

**Impact on signal confidence**: ±5% max. Directional alignment boosts; misalignment subtracts.

---

## Volume Spike Detection

IB provides cumulative day volume. The bot converts that to per-poll deltas:

```
Poll 1: volume = 5,000
Poll 2: volume = 5,800  →  delta = 800 contracts this minute
Poll 3: volume = 6,100  →  delta = 300
Poll 4: volume = 9,500  →  delta = 3,400  ← SPIKE (3,400 vs avg ~550)
```

A spike fires if: `this_delta >= 4× rolling_average` AND `delta >= 300 contracts`

**Repeat sweep tracking**: If the same strike fires 2× within 15 minutes → flow score 0.5. If 3×+ → flow score 1.0. This is factored into confidence.

---

## Liquidity Filters (Pre-Signal Gate)

Before any contract can generate a signal, it must pass all three:

| Filter | Threshold | Why |
|---|---|---|
| Open Interest | > 1,000 contracts | Avoid illiquid strikes |
| Bid/Ask spread | < 8% of mid price | Avoid wide, untradeable spreads |
| Daily volume | > 500 contracts | Minimum activity required |

Contracts failing this are silently dropped — the signal engine never sees them.

---

## Greeks

Every tracked contract gets live Greeks from IB via `modelGreeks` (not estimated):

| Greek | Role in scoring |
|---|---|
| **Delta** | Ideal range: 0.30–0.60 for calls, -0.60 to -0.30 for puts. Outside = confidence penalty |
| **Gamma** | Ideal: 0.005–0.08. Rewards contracts sensitive to intraday moves |
| **Theta** | Penalises high decay (< -0.15/day) for directional long plays |
| **Vega** | Included in alert for context |
| **IV** | Option-level implied vol shown in alert |
| **IV Rank** | VIX position in its 52-week range (0=low, 100=high). Drives debit vs credit bias |

---

## The 7 Signal Types

| Signal | What triggers it | Best regime |
|---|---|---|
| **CALL SWEEP** | Volume spike on calls + bid-side pressure | TREND_UP |
| **PUT SWEEP** | Volume spike on puts + ask-side pressure | TREND_DOWN |
| **BULL CALL SPREAD** | Call sweep + IV rank < 30 (cheap options) | TREND_UP + LOW_VOL |
| **BEAR PUT SPREAD** | Put sweep + IV rank < 30 | TREND_DOWN + LOW_VOL |
| **LONG STRADDLE** | Both call AND put spike simultaneously | Any (big move expected) |
| **HIGH IV ALERT** | IV rank > 70 or VIX > 26 | HIGH_VOL / RANGE_BOUND |
| **P/C RATIO EXTREME** | Chain-level put/call ratio > 1.8 or < 0.5 | Any |

---

## Weighted Confidence Model

Each signal is scored across 10 components, then adjusted by external signals:

| Component | Weight | Detail |
|---|---|---|
| Volume spike strength | 25% | How many × the rolling average |
| Bid/ask imbalance | 15% | Aggressor at bid vs ask |
| Delta quality | 10% | Is delta in the ideal 0.30–0.60 range? |
| Gamma quality | 10% | Is gamma in the ATM sweet spot? |
| Theta penalty | 10% | Heavy decay hurts confidence for longs |
| IV regime alignment | 10% | Low IV rank → debit good; high → credit good |
| Sentiment alignment | 10% | Bullish IB sentiment boosts calls, bearish boosts puts |
| Open interest strength | 5% | More OI = more confidence |
| Flow score (repeat sweeps) | 5% | 0.5 for 2 hits, 1.0 for 3+ hits in 15 min |
| **External composite** | ±5% | News + macro + social alignment (additive adjustment) |

**Confidence tiers:**

| Score | Tier | Meaning |
|---|---|---|
| 70–79% | MEDIUM | Worth watching |
| 80–89% | ★ HIGH | Strong signal |
| 90%+ | ★★ EXTREME ★★ | Multiple confirming factors |

Signals below **70%** are dropped.

**Event risk overrides:**
- Directional signals within ±30 min of a US High-impact event: confidence × 0.70 (−30%)
- LONG STRADDLE within the same window: confidence × 1.10 (+10%)

---

## Deduplication

Same signal type + strike + expiry suppressed for **90 minutes** after sending. No spam.

---

## Sample Telegram Message

```
🔥 SPY OPTIONS — CALL SWEEP

📌 565C  exp APR26
💰 SPY: $562.40
📊 VIX: 14.2 (LOW IV)
🌡 IV Rank: 24/100
🌍 Regime: TREND_UP
📐 Greeks: Δ +0.423 | Γ 0.0312 | Θ -0.071 | V 0.182 | IV 19.4%
📦 Volume: 12,450
🔥 Spike: 6.2× rolling avg
🏦 Open Interest: 8,300
↔ Bid/Ask size: 850 / 120
💵 Bid/Ask: $1.85 / $1.92
↔ Spread: 3.6%
🧭 Sentiment: +62 (BULLISH)
🌐 Ext Sentiment: +0.31 (BULLISH)
  📰 bullish news (+0.182)
  📱 StockTwits: +0.45
  📉 Macro: TNX FLAT | DXY FALLING ↓
  ⚖️ CBOE Equity P/C: 0.83
📅 Next event: ISM Manufacturing (47 min)
🎯 Confidence: 86% [★ HIGH]
🕐 10:42 ET

🧠 Analysis:
  • Call volume spike: 12,450 contracts at 565C
  • Spike: 6.2× rolling avg
  • Bid/Ask ratio 7.1× — aggressive buyer at ask
  • Low IV rank (24) — debit strategies are cheap
  • Regime: TREND_UP  Sentiment: BULLISH (+62)
  • Delta +0.423  Gamma 0.0312  IV 19.4%

💡 Signal Idea:
  Call Sweep: 565C exp APR26
  Bull Call Spread: Buy 565C / Sell 570C exp APR26
  Risk: Exit if SPY loses VWAP ($559.80) or VIX spikes

⚠️ For informational purposes only. Not financial advice.
Options carry significant risk of loss.
#SPY #Options #ShreeBot
```

---

## Analytics Database

Every signal sent is stored in `data/spy_options_signals.db` (SQLite):

| Field | Purpose |
|---|---|
| signal_type, strike, right, expiry | What was signalled |
| confidence, confidence_tier | Score and tier |
| delta, gamma, theta, vega, impl_vol | Greeks at signal time |
| iv_rank | IV rank at signal time |
| regime, sentiment_score | IB market context |
| flow_score | Repeat sweep score |
| reasoning | Full analysis bullets |

---

## Key Parameters

| Parameter | Value | Config key |
|---|---|---|
| Strike window | ±4% of current SPY price | `chain.strike_pct_range` |
| Min session volume | 500 contracts | `signals.min_volume_for_signal` |
| Min sweep size | 300 contracts in one poll | `signals.sweep_poll_volume_threshold` |
| Min open interest | 1,000 contracts | `chain.liquidity_min_oi` |
| Max bid/ask spread | 8% of mid price | `chain.liquidity_max_spread_pct` |
| Spike multiplier | 4× rolling average | `signals.volume_spike_mult` |
| Expiries tracked | Current + next monthly | `chain.num_expiries` |
| Poll interval | Every 60 seconds | `session.poll_interval_s` |
| RTH session | 9:35 AM – 3:45 PM ET, weekdays | `session.rth_start_et` / `rth_stop_et` |
| Dedup window | 90 minutes per signal | `signals.dedup_window_minutes` |
| Repeat sweep window | 15 minutes | `signals.sweep_window_minutes` |
| Min confidence | 70% | `signals.min_confidence` |
| Greeks wait | 4 seconds after subscription | `ib.greeks_wait_s` |
| VIX 52w range | Fetched once at startup | — |
| Event risk window | ±30 minutes | `external.event_risk_window_minutes` |
| News TTL | 10 minutes | `external.news_ttl_minutes` |
| StockTwits TTL | 10 minutes | `external.stocktwits_ttl_minutes` |
| Reddit | Disabled by default | `external.reddit_enabled` |
| Macro refresh | Daily (yfinance EOD) | `external.macro_enabled` |
| Ext confidence boost | ±5% max | `external.composite_confidence_boost` |

---

## Data Sources Summary

| Source | Data | Requires | Refresh |
|---|---|---|---|
| IB Gateway (ib_insync) | SPY price, VIX, bars, Greeks | IB account + Gateway | Every 60s |
| Forex Factory JSON | Economic calendar | Nothing | Daily |
| Yahoo/CNBC/Reuters RSS | News headlines | Nothing | 10 min |
| StockTwits public API | Retail bullish/bearish % | Nothing | 10 min |
| yfinance | 10Y yield, DXY, oil, gold | Nothing | Daily |
| CBOE public CSV | Equity P/C ratio | Nothing | Daily |
| Reddit (asyncpraw) | WSB/investing sentiment | Reddit app credentials | 15 min (opt-in) |
