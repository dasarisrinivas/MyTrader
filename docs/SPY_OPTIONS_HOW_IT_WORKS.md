# SPY Options Bot — How It Works

## The Basic Idea

Every 60 seconds during market hours (9:35–3:45 ET), the bot:
1. Fetches SPY price, VIX, 5-minute bars, and option chain data from IB Gateway
2. Classifies the market regime and scores IB-based sentiment
3. Computes intraday technical levels (ORB, VWAP bands, pivot points, EDR, RSI)
4. Refreshes external signals (news, options flow, macro, social, economic calendar)
5. Fetches Greeks (delta, gamma, theta, vega, IV) for every tracked contract
6. Computes max pain strike from live chain OI (0DTE pinning target)
7. Runs a weighted confidence model with dynamic time-of-day / DTE adjustments
8. Sends Telegram entry alerts — **no trades placed, ever**
9. Monitors active signals and sends EXIT alerts when the thesis breaks down

---

## Poll Cycle (Every 60 Seconds)

```
1.  Fetch SPY price          (IB streaming — persistent subscription, not re-fetched)
2.  Fetch VIX                (IB snapshot)
3.  Fetch SPY 5-min bars     (IB historical — regime + sentiment + tech-level inputs)
4.  Classify market regime   (EMA9/21, ATR14, VWAP — pure math on bar data)
5.  Score IB sentiment       (VIX trend + VWAP position + EMA slope)
6.  Compute IV rank          (VIX vs 52-week range, cached at startup)
7.  Compute technical levels (pure Python, zero network calls — every poll)
    a. Opening Range Breakout (ORB)   — 30-min session high/low; breakout/breakdown status
    b. VWAP std-dev bands             — ±1σ / ±2σ volume-weighted extension zones
    c. Daily pivot points             — PP, R1, R2, S1, S2 (Floor-Trader, prior session)
    d. Expected Daily Range (EDR)     — VIX-implied 1-σ range; exhaustion % consumed
    e. RSI (5-min) + divergence       — overbought/oversold; bearish/bullish divergence
8.  Refresh external signals (TTL-gated — most sources don't refetch every 60s)
    a. Economic calendar     (Forex Factory JSON — daily refresh)
    b. News RSS sentiment    (Yahoo/MarketWatch/CNBC/Reuters + VADER — 10-min TTL)
    c. StockTwits retail     (free public API — 10-min TTL)
    d. Reddit enhanced       (asyncpraw: body+comments+karma — 15-min TTL, opt-in)
    e. Macro signals         (yfinance: TNX, DXY, VIX, oil, gold — daily + 15-min intraday)
    f. CBOE P/C ratio        (daily CSV — daily refresh)
    g. Options flow confirm  (yfinance options chain → unusual activity, GEX, dark pool)
9.  For each of 2 expiries:
    a. Resolve option conids (cached after day 1)
    b. Fetch price + Greeks  (streaming subscription, explicit cancel after read)
    c. Apply liquidity filter (OI ≥ 100, spread < 8%, volume ≥ 50)
    d. Compute max pain      (from IB chain OI — first expiry only; 0DTE pin target)
    e. Run signal engine     (8 signal types, weighted confidence model)
    f. Apply event risk      (−30% directional near High-impact event, +10% straddle)
    g. Apply dynamic conf.   (time-of-day, DTE, flow, ORB, VWAP bands, EDR, RSI, pivots)
10. Deduplicate signals      (90-min window per strike/type/expiry)
11. Send Telegram entry alerts
12. Persist to SQLite analytics DB
13. Check active signals → send EXIT alerts if price/regime conditions reverse
```

---

## Connection Stability

| Feature | Detail |
|---|---|
| **Keepalive ping** | `reqCurrentTime()` every 30s prevents idle disconnect |
| **Auto-reconnect** | 5 retries with exponential backoff (5–25s) on disconnect |
| **Streaming SPY price** | Persistent `snapshot=False` subscription — not re-fetched each poll |
| **Connection guard** | `get_spy_price()` checks `isConnected()` and triggers reconnect if stale |
| **Error 10089 fallback** | Missing market data subscription → retry with delayed data |
| **Chain selection** | IB returns duplicate SMART entries; bot picks the one with the most expirations |

---

## Market Regime Detection

Five-minute SPY bars → one of six regimes. Priority order: NEWS_DRIVEN → HIGH_VOL → LOW_VOL → TREND_UP → TREND_DOWN → RANGE_BOUND.

| Regime | Condition | Signals favoured |
|---|---|---|
| **TREND_UP** | EMA9 > EMA21, positive slope, SPY above VWAP | Call sweeps, Bull Call Spreads |
| **TREND_DOWN** | EMA9 < EMA21, negative slope, SPY below VWAP | Put sweeps, Bear Put Spreads |
| **RANGE_BOUND** | No clear trend (default) | Iron Condor, premium selling |
| **HIGH_VOL** | VIX > 26 or ATR expanded 2× median | Spreads over naked longs |
| **LOW_VOL** | VIX < 12 and ATR compressed 0.5× median | Long premium (cheap options) |
| **NEWS_DRIVEN** | ATR blow-up > 2.5× recent median | Caution — noted in alert |

---

## IB Sentiment Score (−100 to +100)

Computed from IB data only — no external APIs involved:

| Component | Weight | Calculation |
|---|---|---|
| VIX trend | 40% | Rising VIX > 5% above recent avg = −40; falling = +40 |
| SPY vs VWAP | 35% | Distance above/below VWAP, normalised by ATR |
| EMA slope | 25% | Rate of EMA-9 change, normalised by ATR |

Labels: **BULLISH** (> +25), **NEUTRAL** (−25 to +25), **BEARISH** (< −25)

---

## External Signals (11 Sources)

### 1. Economic Calendar — Forex Factory JSON
- URL: `cdn-nfs.faireconomy.media/ff_calendar_thisweek.json` — no API key
- Tracks US High-impact events: FOMC, CPI, NFP, GDP, Jobless Claims, Fed speeches
- **Event risk window**: ±30 min → `event_risk=True`
- **Effect**: directional signals −30% confidence; LONG STRADDLE +10%

### 2. News RSS Sentiment — VADER
- Feeds: Yahoo Finance, MarketWatch, CNBC, Reuters
- Only headlines/summaries mentioning SPY, S&P, Fed, inflation, yields, etc.
- VADER compound score averaged across all relevant articles (−1.0 to +1.0)
- 10-minute TTL — no API key

### 3. StockTwits Retail Sentiment
- `api.stocktwits.com/api/2/streams/symbol/SPY.json` — free, no auth
- Counts Bullish/Bearish tags on recent messages → `bullish_pct`, `bearish_pct`
- Score: (bull − bear) / total → −1.0 to +1.0
- 10-minute TTL

### 4. Reddit Enhanced Sentiment (opt-in)
- Subreddits: r/wallstreetbets, r/options, r/stocks, r/investing, r/Daytrading, r/thetagang
- Scores **post body + top 5 comments** (not just title)
- **SPY-specific keyword boost**: breakout (+0.15), melt-up (+0.20), squeeze (+0.12), rug pull (−0.20), rejection (−0.12), gamma squeeze (+0.18), put wall (−0.12), call wall (+0.12) … 30+ terms
- **Karma weighting**: log10(upvotes) / 4 — high-karma posts carry more weight
- **Engagement weighting**: log10(comments) / 3
- **Bot/spam filter**: accounts with karma < 10 or bot-like usernames dropped
- **Sarcasm detection**: `/s`, 🙄, `lmao` patterns flip the sign
- **Contrarian logic**: score > +70 with 80%+ bullish posts → `CONTRARIAN_BEARISH`; score < −70 → `CONTRARIAN_BULLISH`
- Disabled by default — requires Reddit app credentials (free at reddit.com/prefs/apps)
- 15-minute TTL

### 5. Macro Signals — yfinance (Two Refresh Tiers)

**Daily** (once per day): Base levels + trend for TNX, DXY, oil, gold, VIX

**Intraday** (every 15 minutes): Current level vs today's open for TNX, DXY, VIX, SPY

| Signal | Macro Meaning | Effect on SPY |
|---|---|---|
| TNX rising + DXY rising + VIX rising | Broad risk-off | Headwind — penalise calls |
| TNX falling + DXY falling + VIX falling | Risk-on | Tailwind — boost calls |
| Strong breadth (SPY > open, VIX falling) | Momentum day | Boost directional |

Composite `spy_headwind` score: −1.0 (strong headwind) to +1.0 (tailwind)

Labels: STRONG_HEADWIND / HEADWIND / NEUTRAL / TAILWIND / STRONG_TAILWIND

### 6. CBOE Equity Put/Call Ratio
- Daily CSV from `cdn.cboe.com` — no auth
- **Contrarian**: P/C > 1.2 = too bearish → slight bullish bias; P/C < 0.7 = complacency → slight bearish bias

### 7. Options Flow Confirmation (ExternalFlowConfirmation)
Primary source: **yfinance options chain** — always available, no auth required.

For each near-term SPY expiry, every option contract with unusual activity is scored:

```
contribution = base × dte_mult × prem_mult × bias_mult × voi_mult
                    × delta_mult × type_mult × time_decay

base:       +10 BULLISH, −10 BEARISH
dte_mult:    2.0 (0DTE) | 1.5 (1DTE) | 1.0 (2-7DTE) | 0.5 (8+ DTE)
prem_mult:   log10(premium / $100K) + 1  [capped 0.5–3.0]
bias_mult:   1.3 (ASK exec) | 1.0 (MID) | 0.7 (BID exec)
voi_mult:    1.5 (vol/OI > 3×) | 1.2 (vol/OI > 1×)
delta_mult:  1.0 (ATM: |Δ| 0.25–0.75) | 0.6 (far OTM)
type_mult:   1.2 (sweep or block)
time_decay:  exp(−age_min / 120)     half-life ~83 min
```

Aggregate score normalised to **−100 to +100**.

**Gamma Exposure (GEX)**:
```
net_gex = Σ(call_gamma × call_OI) − Σ(put_gamma × put_OI)
Positive → dealers long gamma → dampening → RANGE signal
Negative → dealers short gamma → amplifying → TREND signal
```
Labels: `SUPPORTIVE_UPSIDE` / `SUPPORTIVE_DOWNSIDE` / `NEUTRAL`

**Dark pool bias**: Attempted from Alpha Query free endpoint; falls back to large-block premium skew (institutional $1M+ trades). Labels: `ACCUMULATION` / `DISTRIBUTION` / `NEUTRAL`

**Barchart scrape**: Attempted with proper User-Agent headers; returns empty list on failure (anti-bot, rate limit) — yfinance is always the fallback.

**Gamma Walls** (computed from the same yfinance chain, no additional cost):

| Level | Definition | Effect |
|---|---|---|
| Call Wall | Strike with highest cumulative call OI | Resistance; penalises call signals by −5% |
| Put Wall | Strike with highest cumulative put OI | Support floor; penalises put signals by −5% |
| Gamma Flip | Strike where net GEX changes sign | Key transition level — noted in signal reasoning |
| `at_call_wall` | SPY within 0.3% of call wall | Active resistance flag |
| `at_put_wall` | SPY within 0.3% of put wall | Active support flag |

### 8. Market Breadth — Sector ETF Participation Proxy
Uses yfinance to download 11 SPDR sector ETFs (XLK, XLF, XLE, XLI, XLV, XLB, XLU, XLRE, XLP, XLY, XLC) on a 5-min intraday bar:

- **Breadth ratio**: fraction of sectors currently above their day-open (0.0–1.0)
- **Labels**: STRONG (≥75%), MODERATE (≥55%), NEUTRAL, MODERATE_WEAK (≤45%), WEAK (≤25%)
- **Up/Down volume proxy**: SPY 5-min bars where close > prev_close = up-volume; opposite = down-volume
- **NYSE TICK**: attempted via `^TICK` (yfinance — not always available; graceful fallback)
- 10-minute TTL; no API key

**Confidence effect**: STRONG breadth → +3% for calls; WEAK breadth → +3% for puts; opposite direction → −5%

### 9. Sector Leadership — Tech & Momentum Proxy
Sector ETFs measured vs day-open: XLK, XLF, SMH (semis), IWM (small cap), QQQ (Nasdaq), XLE, XLI.

| Metric | Source | Meaning |
|---|---|---|
| `sector_label` | ETF bull/bear count | BULL_SWEEP / BULL_LEANING / MIXED / BEAR_LEANING / BEAR_SWEEP |
| `qqq_vs_spy_pct` | QQQ intraday % − SPY % | Nasdaq leading/lagging |
| `iwm_vs_spy_pct` | IWM intraday % − SPY % | Small-cap risk appetite |
| `es_premium` | (ES=F / 10) − SPY | Futures premium (positive = futures leading) |
| `gap_pct` | Today's open vs prior close | Opening gap context |
| `above_overnight_high` | SPY > today's range high | Breakout flag |
| `usdjpy_trend` | JPY=X inverted daily change | RISK_ON / RISK_OFF / NEUTRAL |

10-minute intraday TTL; daily TTL for prior close + USDJPY.

**Confidence effect**: BULL_SWEEP → +3% for calls; BEAR_SWEEP → +3% for puts. QQQ leading adds +2% for confirmed calls. Breaking above overnight high → +3% for calls; below overnight low → +3% for puts.

### 10. Volatility Term Structure — VIX/VXV + VVIX
Free via yfinance — 15-minute TTL.

| Ticker | Measures |
|---|---|
| `^VIX` | 30-day implied vol |
| `^VXV` | 93-day implied vol |
| `^VVIX` | Volatility of VIX (vol-of-vol) |

**VIX/VXV ratio** → term structure classification:

| Ratio | Label | Meaning |
|---|---|---|
| < 0.85 | STEEP_CONTANGO | Calm market; vol term structure is normal |
| 0.85–0.95 | CONTANGO | Normal; near-term vol lower than longer-dated |
| 0.95–1.00 | FLAT | Compressed structure |
| 1.00–1.10 | BACKWARDATION | Fear spike; near-term vol > 93-day |
| > 1.10 | STEEP_BACKWARDATION | Panic; acute short-term fear |

**VVIX elevated** (> 115): vol-of-vol is high → reduce conviction on any directional signal (−3%).

**Confidence effect**: BACKWARDATION → −5% for calls; CONTANGO → +2% for calls.

### 11. OPEX Calendar — Dealer Gamma Environment
Pure Python date math — zero network calls, zero latency.

| Field | Definition |
|---|---|
| `next_opex` | Date of next monthly OPEX (3rd Friday of month) |
| `days_to_opex` | Calendar days to next OPEX |
| `is_opex_week` | True if 0–4 days away |
| `is_opex_day` | True if today is OPEX Friday |
| `is_triple_witching` | OPEX is in March, June, September, or December |
| `gamma_environment` | PINNING / EXPANSIVE / NEUTRAL |

**Gamma environment logic**:
- **PINNING**: opex week + dealers net long gamma (positive net GEX) → dealers sell rallies/buy dips → price tends to pin
- **EXPANSIVE**: dealers net short gamma OR far from OPEX → moves accelerate → momentum works better
- **NEUTRAL**: otherwise

**Confidence effect**: PINNING → −3% for directional signals; EXPANSIVE → +2% for directional signals.

---

## External Composite Score

All weighted sources combined into a single score (−1.0 to +1.0), dynamically normalised:

| Source | Default Weight | Note |
|---|---|---|
| Options flow confirmation | 20% | yfinance always available |
| News RSS sentiment | 20% | No API key |
| Macro headwind | 20% | Daily + 15-min intraday |
| StockTwits retail | 15% | Free API |
| CBOE P/C contrarian | 15% | Daily |
| Reddit enhanced | 10% | Only when credentials provided |

Weights are **dynamically normalised** when sources are unavailable. Score shown as zero if fewer than 2 sources have data.

**Base confidence impact**: ±5% max from composite alignment alone.

The following sources are **not included in the composite score** but directly adjust confidence in the Dynamic Confidence Engine:

| Source | Confidence Adjustments |
|---|---|
| Market breadth | ±3–5% based on breadth_ratio and signal direction |
| Sector leadership | ±3–5% based on sector_label alignment; +2% if QQQ leads |
| Gamma walls | −5% if at resistance/support wall; +3% if supportive side |
| Volatility term structure | −5% (backwardation), +2% (contango); −3% if VVIX elevated |
| Overnight context | +3% breaking range high/low; −2% trading against range |
| OPEX gamma environment | −3% (PINNING), +2% (EXPANSIVE) |
| **ORB breakout** | ±5% confirmed breakout/breakdown direction |
| **VWAP bands** | ±2–6% based on extension zone (±2σ = strongest) |
| **EDR exhaustion** | −2% to −8% when VIX-implied range is consumed |
| **RSI divergence** | ±5% when price/RSI diverge; ±2% for overbought/oversold |
| **Pivot proximity** | ±3–4% at R1/R2/S1/S2; −2% at PP |
| **Max pain** | −3% to −6% when near pin target (0DTE strongest) |

---

## Intraday Technical Levels (TechnicalLevelsTracker)

Pure Python computation from the 5-minute IB bars — zero network calls, zero latency. Computed every poll cycle and injected into `ExternalContext` before signals are evaluated.

### 1. Opening Range Breakout (ORB)

The single most-watched SPY intraday reference. Institutional desks and systematic traders use the first 30-minute range (9:30–10:00 ET) as the primary directional filter.

| Field | Description |
|---|---|
| `orb_high` / `orb_low` | High/low of the 9:30–10:00 ET session |
| `orb_established` | True once the 10:00 ET build window closes |
| `orb_status` | `BUILDING` / `INSIDE` / `ABOVE_ORB` / `BELOW_ORB` |
| `orb_breakout_confirmed` | Price closed outside range for ≥1 bar |
| `orb_width_pct` | Range width as % of SPY price (tight < 0.20% = reliable) |

**Confidence effect**: Confirmed breakout above ORB → +5% for calls, −5% for puts. Confirmed breakdown → +5% for puts, −5% for calls. Price stuck inside ORB after 10:00 ET → −3% for all directional signals.

### 2. VWAP Standard Deviation Bands

Volume-weighted standard deviation computed from the full session's bar data. SPY rarely sustains above ±2σ — these are fade zones, not continuation zones.

| Band Position | Meaning | Effect |
|---|---|---|
| `ABOVE_2SD` | Stretched above normal range | +5% puts / −6% calls |
| `ABOVE_1SD` | Elevated | +2% puts / −2% calls |
| `INSIDE_1SD` | Normal range | No adjustment |
| `BELOW_1SD` | Depressed | +2% calls / −2% puts |
| `BELOW_2SD` | Stretched below normal range | +5% calls / −6% puts |

### 3. Daily Pivot Points (Floor-Trader Formula)

Computed from the prior session's high, low, and close. SPY exhibits measurable mean-reversion at these levels intraday.

```
PP = (prev_high + prev_low + prev_close) / 3
R1 = 2×PP − prev_low       R2 = PP + (prev_high − prev_low)
S1 = 2×PP − prev_high      S2 = PP − (prev_high − prev_low)
```

**Proximity threshold**: 0.3% of SPY price or $1.50, whichever is larger.

**Confidence effect**:
- At R1/R2 (resistance) + call signal → −4% (buying into resistance)
- At R1/R2 + put signal → +3% (natural ceiling)
- At S1/S2 (support) + put signal → −4% (shorting at support)
- At S1/S2 + call signal → +3% (natural floor)
- At PP exactly → −2% (direction undecided)

### 4. Expected Daily Range (EDR) Exhaustion

VIX implies a 1-σ expected intraday range for SPY. When most of that range is already consumed, fade signals are more likely than continuation signals. This is the most common 0DTE afternoon over-trade mistake.

```
EDR (points) = VIX / √252 / 100 × SPY_price
EDR used %   = max(high − open, open − low) / EDR × 100
```

**Confidence effect**:

| EDR consumed | Effect on directional signals |
|---|---|
| ≥ 120% (extreme extension) | −8% |
| ≥ 85% (exhausted) | −5% |
| ≥ 60% (getting stretched) | −2% |
| < 60% | No adjustment |

### 5. RSI (5-min) with Divergence Detection

14-period Wilder RSI computed on the 5-minute close series. Divergence detection compares price direction vs RSI direction over the last 8 bars.

**Divergence rules**:
- **Bearish divergence**: price making higher highs but RSI not following — only flagged when RSI is still elevated (> 55 now, > 60 at lookback start)
- **Bullish divergence**: price making lower lows but RSI recovering — only flagged when RSI is still depressed (< 45 now, < 40 at lookback start)

**Confidence effect**:

| Condition | Effect |
|---|---|
| Bearish divergence + put signal | +5% |
| Bearish divergence + call signal | −5% |
| Bullish divergence + call signal | +5% |
| Bullish divergence + put signal | −5% |
| RSI overbought (≥70) + put | +2% |
| RSI oversold (≤30) + call | +2% |

### 6. Max Pain Strike

Computed from the live IB options chain open interest every poll cycle (first available expiry). Max pain is the closing price where total option-buyer losses are maximised — the pinning target for market makers.

```
pain(K) = Σ_calls(max(S − K, 0) × OI[S]) + Σ_puts(max(K − S, 0) × OI[S])
max_pain = K that minimises pain(K)
```

**Near max pain**: SPY within $1.50 of the max pain strike.

**Confidence effect**:

| Condition | Effect |
|---|---|
| Near max pain + 0DTE directional | −6% |
| Near max pain + 1–2 DTE directional | −3% |

---

## Dynamic Confidence Engine

After the base confidence model (10 components) and event-risk modifier, a second layer of adjustments is applied that adapts to:

> **Adjustment blocks 1–12** cover time-of-day, DTE, flow alignment, macro environment, event risk, conflict detection, market breadth, sector leadership, gamma walls, volatility term structure, overnight context, and OPEX gamma environment — all described below.
>
> **Adjustment blocks 13–18** are the new intraday technical level adjustments added in the April 2026 update.

### Time of Day (ET)

| Bucket | Hours | Adjustment |
|---|---|---|
| **OPEN** | 9:30–10:30 | +10% (breakout/momentum hour) |
| **MIDDAY** | 10:30–14:00 | −5% (chop zone) |
| **PRE_POWER** | 14:00–15:00 | neutral |
| **POWER_HOUR** | 15:00–16:00 | +5% (directional moves solidify) |

### DTE Rules

| DTE | Rule | Effect |
|---|---|---|
| 0 (0DTE) | Requires flow score ≥ ±25 | Unconfirmed: −8%; Confirmed: +2% |
| 1 | Standard | No adjustment |
| 2–7 | Swing | +2% when moderate flow aligns |
| 8+ | Standard | No adjustment |

### Flow Alignment

| Flow strength | Signal aligns | Signal conflicts |
|---|---|---|
| Strong (≥ ±40) | +10% | −10% |
| Moderate (≥ ±20) | +5% | −5% |
| Weak (≥ ±10) | +3% | −3% |

### Macro Environment

| Condition | Effect |
|---|---|
| Strong headwind (< −0.40) + call signal | −8% |
| Strong tailwind (> +0.30) + call signal | +5% |
| High VIX (> 30) + call signal | −5% |
| Low VIX (< 14) + put signal | −3% |

### Conflict Detection

If **2 or more** of the following four sources oppose the signal direction → confidence −8% with `conflict_detected=True`:

1. **Flow score** (opposing flow > ±20)
2. **IB sentiment** (opposing sentiment > ±20)
3. **Macro headwind** (opposing macro > ±0.20)
4. **Market regime** (TREND_DOWN opposes calls; TREND_UP opposes puts)

### Market Breadth Adjustment

| Condition | Effect |
|---|---|
| breadth_ratio ≥ 70% + call signal | +3% |
| breadth_ratio ≤ 30% + put signal | +3% |
| breadth_ratio ≤ 30% + call signal | −5% |
| breadth_ratio ≥ 70% + put signal | −5% |

### Sector Leadership Adjustment

| Condition | Effect |
|---|---|
| BULL_SWEEP / BULL_LEANING + call | +3% |
| BEAR_SWEEP / BEAR_LEANING + put | +3% |
| Opposite-direction sweep | −3% |
| QQQ leading (> +0.20% vs SPY) + confirmed call | +2% additional |

### Gamma Wall Adjustment

| Condition | Effect |
|---|---|
| SPY at call wall + call signal | −5% (resistance) |
| SPY at call wall + put signal | +3% (natural ceiling) |
| SPY at put wall + put signal | −5% (support absorbs puts) |
| SPY at put wall + call signal | +3% (floor support) |

### Volatility Term Structure Adjustment

| Condition | Effect |
|---|---|
| BACKWARDATION + call signal | −5% |
| STEEP_BACKWARDATION + put signal | +3% |
| CONTANGO + call signal | +2% |
| VVIX elevated (> 115) | −3% on any directional |

### Overnight Context Adjustment

| Condition | Effect |
|---|---|
| SPY above overnight range high + call | +3% (breakout momentum) |
| SPY below overnight range low + put | +3% (breakdown momentum) |
| SPY above high + put signal | −2% (against momentum) |
| SPY below low + call signal | −2% (against momentum) |

### OPEX Gamma Environment Adjustment

| Environment | Effect |
|---|---|
| PINNING (opex week + positive GEX) | −3% for directional signals |
| EXPANSIVE (short gamma or far from OPEX) | +2% for directional signals |

### Opening Range Breakout Adjustment (Block 13)

| Condition | Effect |
|---|---|
| ABOVE_ORB confirmed + call | +5% (breakout direction) |
| ABOVE_ORB confirmed + put | −5% (fading breakout) |
| BELOW_ORB confirmed + put | +5% |
| BELOW_ORB confirmed + call | −5% |
| INSIDE ORB (after 10:00 ET) + directional | −3% (range-bound) |

### VWAP Band Exhaustion Adjustment (Block 14)

| Condition | Effect |
|---|---|
| ABOVE_2SD + put (fade) | +5% |
| ABOVE_2SD + call (chase) | −6% |
| BELOW_2SD + call (fade) | +5% |
| BELOW_2SD + put (chase) | −6% |
| ABOVE_1SD / BELOW_1SD | ±2% (mild) |

### EDR Exhaustion Adjustment (Block 15)

| EDR consumed | Effect |
|---|---|
| ≥ 120% | −8% directional |
| ≥ 85% | −5% directional |
| ≥ 60% | −2% directional |

### RSI Divergence Adjustment (Block 16)

| Condition | Effect |
|---|---|
| Bearish divergence + put | +5% |
| Bearish divergence + call | −5% |
| Bullish divergence + call | +5% |
| Bullish divergence + put | −5% |
| RSI overbought + put (no divergence) | +2% |
| RSI oversold + call (no divergence) | +2% |

### Pivot Point Proximity Adjustment (Block 17)

| Condition | Effect |
|---|---|
| At R1/R2 + put | +3% (natural ceiling) |
| At R1/R2 + call | −4% (resistance) |
| At S1/S2 + call | +3% (natural floor) |
| At S1/S2 + put | −4% (support) |
| At PP + directional | −2% (indecision) |

### Max Pain Proximity Adjustment (Block 18)

| Condition | Effect |
|---|---|
| Near max pain + 0DTE directional | −6% (pin risk) |
| Near max pain + 1–2 DTE directional | −3% |

### Net Adjustment Tiers

| Net delta | Label |
|---|---|
| ≥ +10% | Exceptional alignment |
| +5% to +10% | Strong alignment |
| +3% to +5% | Moderate alignment |
| ±3% | Weak alignment |
| −10% to −20% | Severe penalty |

---

## Volume Spike Detection

IB provides cumulative day volume. The bot tracks per-poll deltas:

```
Poll 1: volume = 5,000
Poll 2: volume = 5,800  →  delta = 800 contracts this minute
Poll 3: volume = 6,100  →  delta = 300
Poll 4: volume = 9,500  →  delta = 3,400  ← SPIKE (3,400 vs avg ~550 = 6.2×)
```

A spike fires if: `delta ≥ 4× rolling_average` AND `delta ≥ 300 contracts`

**Repeat sweep tracking**: same strike fires 2× within 15 min → flow score 0.5; 3×+ → 1.0.

---

## Liquidity Filters (Pre-Signal Gate)

| Filter | Threshold | Why |
|---|---|---|
| Open Interest | ≥ 100 contracts | Avoid truly illiquid strikes |
| Bid/Ask spread | < 8% of mid price | Avoid untradeable spreads |
| Daily volume | ≥ 50 contracts | Minimum activity |

> Thresholds relaxed from OI ≥ 1,000 / vol ≥ 500 after analysis showed the original values filtered 28 of 30 options during low-volume periods.

---

## Greeks (Live from IB modelGreeks)

| Greek | Role in confidence scoring |
|---|---|
| **Delta** | Ideal: 0.30–0.60 calls, −0.60 to −0.30 puts. Outside = −confidence |
| **Gamma** | Ideal: 0.005–0.08. Rewards ATM sensitivity |
| **Theta** | Penalises rapid decay (< −0.15/day) for directional longs |
| **Vega** | Context only — shown in alert |
| **IV** | Option-level implied vol |
| **IV Rank** | VIX vs 52-week range (0=cheap, 100=expensive) |

---

## The 8 Signal Types

| Signal | Trigger | Best regime |
|---|---|---|
| **CALL SWEEP** | Volume spike on calls + bid-side pressure | TREND_UP |
| **PUT SWEEP** | Volume spike on puts + ask-side pressure | TREND_DOWN |
| **BULL CALL SPREAD** | Call sweep + IV rank < 30 | TREND_UP + LOW_VOL |
| **BEAR PUT SPREAD** | Put sweep + IV rank < 30 | TREND_DOWN + LOW_VOL |
| **LONG STRADDLE** | Both call AND put spike simultaneously | Any (big move expected) |
| **HIGH IV ALERT** | IV rank > 70 or VIX > 26 | HIGH_VOL / RANGE_BOUND |
| **P/C RATIO EXTREME** | Chain put/call ratio > 1.8 or < 0.5 | Any |
| **ORB BREAKOUT** | SPY closes outside 30-min opening range for ≥1 bar | TREND_UP / TREND_DOWN |

**ORB BREAKOUT** fires once the 10:00 ET build window closes and price has confirmed a break above (→ call) or below (→ put) the range. Base confidence 72%, boosted by regime/sentiment/flow alignment and a tight ORB width. Monitored for EXIT alerts like all other directional signals.

P/C confidence formula (tuned for 70% min threshold):
- Bearish (P/C > 1.8): `base = 0.70 + (pc − 1.8) × 0.10` (base capped 0.90)
- Bullish (P/C < 0.5): `base = 0.70 + (0.5 − pc) × 0.15` (base capped 0.90)
- Sentiment adjustment: opposing sentiment penalises (up to −10%), aligned boosts (+8%)
- Regime gate: TREND_UP opposes puts (−12%), TREND_DOWN opposes calls (−12%)
- Final: `min(0.95, base + sent_adj + regime_adj)` — hard cap prevents 100% signals

---

## Weighted Confidence Model

**Base confidence** (10 components):

| Component | Weight | Detail |
|---|---|---|
| Volume spike strength | 25% | How many × the rolling average |
| Bid/ask imbalance | 15% | Aggressor at bid vs ask |
| Delta quality | 10% | Ideal 0.30–0.60 range |
| Gamma quality | 10% | ATM sweet spot |
| Theta penalty | 10% | Heavy decay hurts confidence |
| IV regime alignment | 10% | Low IV → debit good; high → credit good |
| Sentiment alignment | 10% | IB sentiment agrees with direction |
| Open interest strength | 5% | More OI = more confidence |
| Flow score (repeat sweeps) | 5% | 0.5 for 2× hits, 1.0 for 3×+ |
| External composite | ±5% | All external sources aligned |

Then **event risk** modifier → **dynamic confidence** adjustment → **threshold filter** at 70%.

> **Hard cap: 95%.** No signal can ever reach 100% confidence. Both the base PC_RATIO formula (capped at 0.90) and the dynamic confidence finaliser (`min(0.95, …)`) enforce this ceiling. This preserves uncertainty and prevents over-conviction from stacking additive adjustments.

**Confidence tiers:**

| Score | Tier | Meaning |
|---|---|---|
| 70–79% | MEDIUM | Worth watching |
| 80–89% | ★ HIGH | Strong signal |
| 90–95% | ★★ EXTREME ★★ | Multiple confirming factors (hard cap at 95%) |

---

## Exit Alert System

After sending a directional entry signal, the bot monitors market conditions every poll cycle.

**Monitored signals**: CALL_SWEEP, PUT_SWEEP, BULL_CALL_SPREAD, BEAR_PUT_SPREAD, PC_RATIO_EXTREME, ORB_BREAKOUT

**Three independent triggers** (any one fires the EXIT alert):

| Trigger | Condition |
|---|---|
| **Price adverse ≥ 0.5%** | SPY moved against signal direction |
| **Regime flip** | TREND_UP → TREND_DOWN (for bullish), or vice-versa |
| **Urgent adverse ≥ 1.0%** | Large adverse move — escalation |

**Lifecycle**: one EXIT per signal; auto-expire after 6 hours; reset daily at midnight ET.

---

## Sample Telegram Messages

### Entry Alert
```
🔥 SPY OPTIONS — CALL SWEEP

📌 SPY Apr 17, 2026 565C  (16 DTE)
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
🌐 Ext Composite: +0.31 (BULLISH)
  🟢 Flow: +38  GEX: SUPPORTIVE UPSIDE
  🏦↑ Dark Pool: ACCUMULATION
  ⚖️ Intraday P/C: 0.76
  🟢 Macro: TAILWIND | TNX FLAT | DXY FALLING
  📰+ News: +0.182
  📱 StockTwits: +0.45
  ⚖️ CBOE Equity P/C: 0.83
⚙️ Dyn adj: +7% [OPEN | 0DTE_CONFIRMED]
📅 Next event: ISM Manufacturing (47 min)
🎯 Confidence: 86% [★ HIGH]
🕐 10:42 ET

🧠 Analysis:
  • Call volume spike: 12,450 contracts at 565C
  • Spike: 6.2× rolling avg
  • Bid/Ask ratio 7.1× — aggressive buyer at ask
  • Low IV rank (24) — debit strategies are cheap
  • Regime: TREND_UP  Sentiment: BULLISH (+62)
  • 📊 Dynamic adj: +7% [OPEN | 0DTE_CONFIRMED | STRONG_ALIGNMENT]

💡 Signal Idea:
  Call Sweep: 565C exp APR26
  Bull Call Spread: Buy 565C / Sell 570C exp APR26
  Risk: Exit if SPY loses VWAP ($559.80) or VIX spikes

⚠️ Exit trigger: SPY ≤ ~$559.59 (−0.5% from $562.40)
🔴 Urgent exit: SPY ≤ ~$556.78 (−1.0%)

⚠️ For informational purposes only. Not financial advice.
Options carry significant risk of loss.
#SPY #Options #ShreeBot
```

### Exit Alert
```
🚨 EXIT ALERT — CALL_SWEEP 🚨

📌 SPY Apr 17, 2026 565C  (16 DTE)

Original signal was BULLISH
🔹 Entry SPY: $562.40
📉 Current SPY: $559.50  (-2.90, -0.52%)

⚠️ Exit Reason(s):
  • SPY dropped 0.52% since entry ($562.40 → $559.50)
  • Regime flipped: TREND_UP → TREND_DOWN

🌍 Current Regime: TREND_DOWN
🕐 11:15 ET

⚠️ Consider closing or hedging this position.
Not financial advice.
#SPY #Options #EXIT #ShreeBot
```

---

## Analytics Database

Every signal is stored in `data/spy_options_signals.db` (SQLite):

| Field Group | Fields |
|---|---|
| Signal identity | signal_type, strike, right, expiry, expiry_date, dte |
| Confidence | confidence, confidence_tier, dynamic_confidence_delta, confidence_time_bucket, confidence_dte_rule, conflict_detected |
| Greeks | delta, gamma, theta, vega, impl_vol, iv_rank |
| Market context | regime, sentiment_score, sentiment_label, vix, spy_price |
| Options flow | flow_score (repeat sweeps), flow_confirmation_score, dark_pool_bias, gex_bias, intraday_pc_ratio |
| External composite | external_composite, news_score, retail_score |
| Macro | macro_headwind, macro_label, tnx_trend, dxy_trend, equity_pc |
| Metadata | reasoning (JSON), suggested_trade, sent_at |

Schema migrations applied automatically on startup — existing databases are upgraded without data loss.

---

## Key Parameters

| Parameter | Default | Config key |
|---|---|---|
| Strike window | ±4% of SPY price | `chain.strike_pct_range` |
| Min session volume | 500 contracts | `signals.min_volume_for_signal` |
| Min sweep poll volume | 300 contracts | `signals.sweep_poll_volume_threshold` |
| Min open interest | 100 contracts | `chain.liquidity_min_oi` |
| Max bid/ask spread | 8% of mid | `chain.liquidity_max_spread_pct` |
| Min daily volume | 50 contracts | `chain.liquidity_min_volume` |
| Spike multiplier | 4× rolling avg | `signals.volume_spike_mult` |
| Expiries tracked | 2 (current + next) | `chain.num_expiries` |
| Poll interval | 60 seconds | `session.poll_interval_s` |
| RTH session | 9:35–15:45 ET | `session.rth_start_et` / `rth_stop_et` |
| Dedup window | 90 minutes | `signals.dedup_window_minutes` |
| Repeat sweep window | 15 minutes | `signals.sweep_window_minutes` |
| Min confidence | 70% | `signals.min_confidence` |
| Greeks wait | 4 seconds | `ib.greeks_wait_s` |
| Event risk window | ±30 minutes | `external.event_risk_window_minutes` |
| News TTL | 10 minutes | `external.news_ttl_minutes` |
| StockTwits TTL | 10 minutes | `external.stocktwits_ttl_minutes` |
| Reddit | Disabled | `external.reddit_enabled` |
| Macro intraday TTL | 15 minutes | — (hardcoded in MacroSignals) |
| Flow TTL | 10 minutes | `external.flow_ttl_minutes` |
| Dark pool | Enabled | `external.flow_dark_pool_enabled` |
| Ext confidence boost | ±5% max | `external.composite_confidence_boost` |
| 0DTE min flow score | 25 | DynamicConfidence default |
| Conflict penalty | −8% | DynamicConfidence default |
| Exit price threshold | 0.5% adverse | manager hardcoded |
| Exit urgent threshold | 1.0% adverse | manager hardcoded |
| Signal max age | 6 hours | manager hardcoded |
| ORB build window | 9:30–10:00 ET (30 min) | TechnicalLevelsTracker hardcoded |
| ORB breakout confirm | ≥1 bar closed outside range | TechnicalLevelsTracker hardcoded |
| VWAP band SD | volume-weighted 1σ / 2σ | TechnicalLevelsTracker hardcoded |
| Pivot proximity threshold | 0.3% or $1.50 (whichever larger) | TechnicalLevelsTracker hardcoded |
| EDR exhaustion threshold | 85% of VIX-implied range | TechnicalLevelsTracker hardcoded |
| RSI period (5-min) | 14 | TechnicalLevelsTracker hardcoded |
| RSI divergence lookback | 8 bars (40 min) | TechnicalLevelsTracker hardcoded |
| Max pain proximity | $1.50 | manager hardcoded |

---

## Data Sources Summary

| Source | Data | Requires | Refresh |
|---|---|---|---|
| IB Gateway (ib_insync) | SPY price, VIX, bars, Greeks | IB account + Gateway | Every 60s |
| TechnicalLevelsTracker | ORB, VWAP bands, pivots, EDR, RSI, max pain | Nothing (pure Python) | Every 60s (instant) |
| yfinance options chain | SPY options flow, GEX, gamma walls, unusual activity | Nothing | 10 min |
| yfinance sector ETFs | 11-sector breadth ratio, up/down vol proxy | Nothing | 10 min |
| yfinance sector leaders | XLK/XLF/SMH/IWM/QQQ vs open, gap, overnight range, ES premium | Nothing | 10 min |
| yfinance vol structure | ^VIX, ^VXV, ^VVIX — term structure and vol-of-vol | Nothing | 15 min |
| yfinance macro | 10Y yield, DXY, oil, gold, VIX | Nothing | Daily + 15 min |
| OPEX calendar | Monthly/quarterly OPEX dates, gamma environment | Nothing (pure Python) | Per poll (instant) |
| Forex Factory JSON | Economic calendar | Nothing | Daily |
| Yahoo/CNBC/Reuters RSS | News headlines | Nothing | 10 min |
| StockTwits public API | Retail bullish/bearish % | Nothing | 10 min |
| CBOE public CSV | Equity P/C ratio | Nothing | Daily |
| Alpha Query | Dark pool index | Nothing | 10 min (best-effort) |
| Barchart scrape | Unusual options activity | Nothing | 10 min (best-effort) |
| Reddit (asyncpraw) | WSB/investing sentiment | Reddit app credentials | 15 min (opt-in) |
