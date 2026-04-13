# SPY Options Bot — How It Works (Implementation-Accurate)# SPY Options Bot — How It Works (Implementation-Accurate)



> **Signal-only** — generates Telegram alerts. No orders are ever placed.## The Basic Idea

> Entry point: `run_spy_options.py` → `SpyOptionsManager` (IB client_id=5).

Every 60 seconds during market hours (9:35–3:45 ET), the bot:

---1. Fetches SPY price, VIX, 5-minute bars, and option chain data from IB Gateway

2. Classifies the market regime and scores IB-based sentiment

## 1. System Overview3. Computes intraday technical levels (ORB, VWAP bands, pivot points, EDR, RSI)

4. Refreshes external signals (news, options flow, macro, social, economic calendar)

Every **60 seconds** during RTH (9:35–15:45 ET), the bot runs a 7-stage pipeline:5. Fetches Greeks (delta, gamma, theta, vega, IV) for every tracked contract

6. Computes max pain strike from live chain OI (0DTE pinning target)

1. **Market Data** — Fetch SPY price (streaming), VIX (snapshot), 5-min bars from IB Gateway7. Runs a weighted confidence model with dynamic time-of-day / DTE adjustments

2. **Regime + Sentiment** — Classify market regime (6 states) and score IB-based sentiment (−100 to +100)8. Sends Telegram entry alerts — **no trades placed, ever**

3. **Technical Levels** — Compute ORB, VWAP bands, pivots, EDR, RSI, max pain (pure Python, zero network)9. Monitors active signals and sends EXIT alerts when the thesis breaks down

4. **External Signals** — Refresh 10 TTL-gated external sources (news, flow, macro, social, calendar, OPEX)

5. **Chain + Greeks** — Build option chain for 2 near-term expiries; fetch delta/gamma/theta/vega/IV via `modelGreeks`---

6. **Signal Engine** — Run 8 signal rules → 10-component weighted confidence → signal-specific adjustments → 19-factor dynamic confidence → Tier-1/2/3 priority hierarchy → quality gate → threshold filter at 70%

7. **Dispatch** — Dedup (90-min window) → direction throttle (3 per direction per 90 min) → daily cap (10/day) → Telegram + SQLite## Poll Cycle (Every 60 Seconds)



Active signals are monitored every poll cycle for **exit conditions** (6 independent triggers).```

1.  Fetch SPY price          (IB streaming — persistent subscription, not re-fetched)

---2.  Fetch VIX                (IB snapshot)

3.  Fetch SPY 5-min bars     (IB historical — regime + sentiment + tech-level inputs)

## 2. Data Flow Diagram4.  Classify market regime   (EMA9/21, ATR14, VWAP — pure math on bar data)

5.  Score IB sentiment       (VIX trend + VWAP position + EMA slope)

```6.  Compute IV rank          (VIX vs 52-week range, cached at startup)

                        ┌──────────────┐7.  Compute technical levels (pure Python, zero network calls — every poll)

                        │  IB Gateway  │    a. Opening Range Breakout (ORB)   — 30-min session high/low; breakout/breakdown status

                        │  (ib_insync) │    b. VWAP std-dev bands             — ±1σ / ±2σ volume-weighted extension zones

                        └──────┬───────┘    c. Daily pivot points             — PP, R1, R2, S1, S2 (Floor-Trader, prior session)

                               │    d. Expected Daily Range (EDR)     — VIX-implied 1-σ range; exhaustion % consumed

              ┌────────────────┼────────────────┐    e. RSI (5-min) + divergence       — overbought/oversold; bearish/bullish divergence

              ▼                ▼                ▼8.  Refresh external signals (TTL-gated — most sources don't refetch every 60s)

        SPY price/VIX    5-min bars      Option chain    a. Economic calendar     (Forex Factory JSON — daily refresh)

        (streaming)      (historical)    (Greeks via modelGreeks)    b. News RSS sentiment    (Yahoo/MarketWatch/CNBC/Reuters + VADER — 10-min TTL)

              │                │                │    c. StockTwits retail     (free public API — 10-min TTL)

              ▼                ▼                ▼    d. Reddit enhanced       (asyncpraw: body+comments+karma — 15-min TTL, opt-in)

     ┌────────────┐   ┌──────────────┐   ┌──────────────┐    e. Macro signals         (yfinance: TNX, DXY, VIX, oil, gold — daily + 15-min intraday)

     │ IV Rank    │   │ Regime       │   │ Liquidity    │    f. CBOE P/C ratio        (daily CSV — daily refresh)

     │ (VIX 52w)  │   │ Detector     │   │ Filter       │    g. Options flow confirm  (yfinance options chain → unusual activity, GEX, dark pool)

     └─────┬──────┘   │ + Sentiment  │   │ (OI/spread/  │9.  For each of 2 expiries:

           │          │ + Tech Levels│   │  volume)     │    a. Resolve option conids (cached after day 1)

           │          └──────┬───────┘   └──────┬───────┘    b. Fetch price + Greeks  (streaming subscription, explicit cancel after read)

           │                 │                  │    c. Apply liquidity filter (OI ≥ 100, spread < 8%, volume ≥ 50)

           ▼                 ▼                  ▼    d. Compute max pain      (from IB chain OI — first expiry only; 0DTE pin target)

     ┌─────────────────────────────────────────────────┐    e. Run signal engine     (8 signal types, weighted confidence model)

     │              Signal Engine (5 stages)           │    f. Apply event risk      (−30% directional near High-impact event, +10% straddle)

     │                                                 │    g. Apply dynamic conf.   (time-of-day, DTE, flow, ORB, VWAP bands, EDR, RSI, pivots)

     │  Stage 1: 8 signal rules (sweep, ORB, P/C, IV) │10. Deduplicate signals      (90-min window per strike/type/expiry)

     │  Stage 2: Event risk modifier (±30%)            │11. Send Telegram entry alerts

     │  Stage 3: 19-factor dynamic confidence          │12. Persist to SQLite analytics DB

     │  Stage 4: Tier-1/2/3 priority hierarchy         │13. Check active signals → send EXIT alerts if price/regime conditions reverse

     │  Stage 5: Quality gate (5 hard blocks)          │```

     └──────────────────┬──────────────────────────────┘

                        │---

          ┌─────────────┼──────────────┐

          ▼             ▼              ▼## Connection Stability

   ┌────────────┐ ┌──────────┐  ┌────────────┐

   │  Telegram  │ │  SQLite  │  │ Exit       │| Feature | Detail |

   │  Alerts    │ │ Analytics│  │ Monitor    │|---|---|

   └────────────┘ └──────────┘  │ (6 triggers│| **Keepalive ping** | `reqCurrentTime()` every 30s prevents idle disconnect |

                                │  per poll) │| **Auto-reconnect** | 5 retries with exponential backoff (5–25s) on disconnect |

                                └────────────┘| **Streaming SPY price** | Persistent `snapshot=False` subscription — not re-fetched each poll |

```| **Connection guard** | `get_spy_price()` checks `isConnected()` and triggers reconnect if stale |

| **Error 10089 fallback** | Missing market data subscription → retry with delayed data |

**External Data** (TTL-gated, injected into `ExternalContext` before Stage 1):| **Chain selection** | IB returns duplicate SMART entries; bot picks the one with the most expirations |



```---

  yfinance ──► Options flow / GEX / dark pool / gamma walls  (10-min TTL)

           ──► Sector ETFs (breadth + leadership)             (10-min TTL)## Market Regime Detection

           ──► Macro (TNX, DXY, VIX, oil, gold)              (daily + 15-min)

           ──► Vol structure (VIX/VXV, VVIX)                  (15-min TTL)Five-minute SPY bars → one of six regimes. Priority order: NEWS_DRIVEN → HIGH_VOL → LOW_VOL → TREND_UP → TREND_DOWN → RANGE_BOUND.

  RSS+VADER ─► News sentiment                                 (10-min TTL)

  StockTwits ► Retail sentiment                               (10-min TTL)| Regime | Condition | Signals favoured |

  Reddit ────► Enhanced sentiment (opt-in, asyncpraw)         (15-min TTL)|---|---|---|

  CBOE CSV ──► Equity P/C ratio                               (daily)| **TREND_UP** | EMA9 > EMA21, positive slope, SPY above VWAP | Call sweeps, Bull Call Spreads |

  ForexFactory► Economic calendar                             (daily)| **TREND_DOWN** | EMA9 < EMA21, negative slope, SPY below VWAP | Put sweeps, Bear Put Spreads |

  OPEX ──────► Pure Python date math                          (per poll, instant)| **RANGE_BOUND** | No clear trend (default) | Iron Condor, premium selling |

```| **HIGH_VOL** | VIX > 26 or ATR expanded 2× median | Spreads over naked longs |

| **LOW_VOL** | VIX < 12 and ATR compressed 0.5× median | Long premium (cheap options) |

---| **NEWS_DRIVEN** | ATR blow-up > 2.5× recent median | Caution — noted in alert |



## 3. IB API / Runtime Transport---



The bot connects to **IB Gateway via `ib_insync`** (socket API, port 4001 live / 4002 paper). Client Portal / REST API is **not used**.## IB Sentiment Score (−100 to +100)



| Method | Purpose | Frequency |Computed from IB data only — no external APIs involved:

|---|---|---|

| `reqMktData(snapshot=False)` | SPY price — persistent streaming subscription | Once at startup || Component | Weight | Calculation |

| `reqMktData(snapshot=True)` | VIX level | Every poll ||---|---|---|

| `reqHistoricalData()` | SPY 5-min bars (regime, sentiment, tech levels) | Every poll || VIX trend | 40% | Rising VIX > 5% above recent avg = −40; falling = +40 |

| `reqSecDefOptParams()` | Option chain strikes + expiries | Once per day per expiry || SPY vs VWAP | 35% | Distance above/below VWAP, normalised by ATR |

| `reqMktData(snapshot=False)` + cancel | Option Greeks (`modelGreeks`) — subscribe, wait 4s, read, cancel | Per option per poll || EMA slope | 25% | Rate of EMA-9 change, normalised by ATR |

| `reqCurrentTime()` | Keepalive ping (prevents idle disconnect) | Every 30s |

Labels: **BULLISH** (> +25), **NEUTRAL** (−25 to +25), **BEARISH** (< −25)

**Subscription cap**: IB allows ~100 concurrent market data lines. The bot caps at `max_subscriptions` (default 60). When the cap is reached, it processes contracts in batches — subscribe → read → cancel before the next batch.

---

**Connection resilience**: 5 retries with exponential backoff (5–25s). Error 10089 (missing data subscription) falls back to delayed data. `get_spy_price()` checks `isConnected()` and triggers reconnect if stale.

## External Signals (11 Sources)

---

### 1. Economic Calendar — Forex Factory JSON

## 4. Signal Types- URL: `cdn-nfs.faireconomy.media/ff_calendar_thisweek.json` — no API key

- Tracks US High-impact events: FOMC, CPI, NFP, GDP, Jobless Claims, Fed speeches

### PRIMARY (directional, actionable)- **Event risk window**: ±30 min → `event_risk=True`

- **Effect**: directional signals −30% confidence; LONG STRADDLE +10%

| Signal | Trigger | Best Regime |

|---|---|---|### 2. News RSS Sentiment — VADER

| **CALL_SWEEP** | Volume spike on calls + bid-side pressure | TREND_UP |- Feeds: Yahoo Finance, MarketWatch, CNBC, Reuters

| **PUT_SWEEP** | Volume spike on puts + ask-side pressure | TREND_DOWN |- Only headlines/summaries mentioning SPY, S&P, Fed, inflation, yields, etc.

| **BULL_CALL_SPREAD** | Call sweep + IV rank < 30 (cheap debit) | TREND_UP + LOW_VOL |- VADER compound score averaged across all relevant articles (−1.0 to +1.0)

| **BEAR_PUT_SPREAD** | Put sweep + IV rank < 30 | TREND_DOWN + LOW_VOL |- 10-minute TTL — no API key

| **ORB_BREAKOUT** | Confirmed close outside 30-min opening range | TREND_UP / TREND_DOWN |

| **LONG_STRADDLE** | Both call AND put spike simultaneously | Any (big move expected) |### 3. StockTwits Retail Sentiment

- `api.stocktwits.com/api/2/streams/symbol/SPY.json` — free, no auth

### INFORMATIONAL (environment alerts — modifiers only)- Counts Bullish/Bearish tags on recent messages → `bullish_pct`, `bearish_pct`

- Score: (bull − bear) / total → −1.0 to +1.0

| Signal | Trigger | Note |- 10-minute TTL

|---|---|---|

| **HIGH_IV_ALERT** | IV rank > 70 or VIX > 26 | Not directional — signals credit structure preference |### 4. Reddit Enhanced Sentiment (opt-in)

| **PC_RATIO_EXTREME** | Chain P/C > 1.8 or < 0.5 | Secondary confirmer only. **Never overrides price/flow.** |- Subreddits: r/wallstreetbets, r/options, r/stocks, r/investing, r/Daytrading, r/thetagang

- Scores **post body + top 5 comments** (not just title)

**Key rule**: Sentiment (StockTwits, Reddit, News) is **never Tier 1** — it only adjusts within ±5% and cannot override price action or flow.- **SPY-specific keyword boost**: breakout (+0.15), melt-up (+0.20), squeeze (+0.12), rug pull (−0.20), rejection (−0.12), gamma squeeze (+0.18), put wall (−0.12), call wall (+0.12) … 30+ terms

- **Karma weighting**: log10(upvotes) / 4 — high-karma posts carry more weight

---- **Engagement weighting**: log10(comments) / 3

- **Bot/spam filter**: accounts with karma < 10 or bot-like usernames dropped

## 5. Signal Priority Hierarchy- **Sarcasm detection**: `/s`, 🙄, `lmao` patterns flip the sign

- **Contrarian logic**: score > +70 with 80%+ bullish posts → `CONTRARIAN_BEARISH`; score < −70 → `CONTRARIAN_BULLISH`

| Tier | Factors | Authority |- Disabled by default — requires Reddit app credentials (free at reddit.com/prefs/apps)

|---|---|---|- 15-minute TTL

| **Tier 1** (highest) | Options flow score (≥ ±30), ORB breakout (confirmed) | Can block a signal outright |

| **Tier 2** (strong) | Market regime (TREND_UP/DOWN), VWAP band position (±2σ) | Heavily adjusts confidence |### 5. Macro Signals — yfinance (Two Refresh Tiers)

| **Tier 3** (secondary) | RSI divergence, pivot proximity, EDR exhaustion | Fine-tunes; cannot override Tier 1 |

**Daily** (once per day): Base levels + trend for TNX, DXY, oil, gold, VIX

**Conflict resolution table**:

**Intraday** (every 15 minutes): Current level vs today's open for TNX, DXY, VIX, SPY

| Condition | Delta |

|---|---|| Signal | Macro Meaning | Effect on SPY |

| 2 Tier-1 factors oppose signal direction | **−15%** ||---|---|---|

| 1 Tier-1 opposes, 0 Tier-1 confirms | **−8%** || TNX rising + DXY rising + VIX rising | Broad risk-off | Headwind — penalise calls |

| 1 Tier-1 confirms, 0 Tier-1 opposes | **+2%** || TNX falling + DXY falling + VIX falling | Risk-on | Tailwind — boost calls |

| 2 Tier-1 factors confirm direction | **+5%** || Strong breadth (SPY > open, VIX falling) | Momentum day | Boost directional |

| Tier-2 regime opposes (no Tier-1 penalty) | **−5%** |

Composite `spy_headwind` score: −1.0 (strong headwind) to +1.0 (tailwind)

---

Labels: STRONG_HEADWIND / HEADWIND / NEUTRAL / TAILWIND / STRONG_TAILWIND

## 6. Final Decision Formula

### 6. CBOE Equity Put/Call Ratio

```- Daily CSV from `cdn.cboe.com` — no auth

final_confidence = min(0.95, max(0.0,- **Contrarian**: P/C > 1.2 = too bearish → slight bullish bias; P/C < 0.7 = complacency → slight bearish bias

    base_confidence                    # 10-component weighted model (0.0–1.0)

    + signal_specific_adjustments      # stale-decay, move-exhaustion, RSI-div,### 7. Options Flow Confirmation (ExternalFlowConfirmation)

                                       # DTE-adj, vol-gate, macro-vel, TICK,Primary source: **yfinance options chain** — always available, no auth required.

                                       # price-structure, trap-detect, sentiment,

                                       # external-composite, VWAP, regime,For each near-term SPY expiry, every option contract with unusual activity is scored:

                                       # multi-factor (flow/breadth/GEX/DP/QQQ/IWM)

    + dynamic_confidence_delta         # 19-factor DynamicConfidence adjuster```

    + priority_conflict_adjustment     # Tier-1/2/3 hierarchy (−0.15 to +0.05)contribution = base × dte_mult × prem_mult × bias_mult × voi_mult

    + external_composite_boost         # ±composite_confidence_boost (default ±5%)                    × delta_mult × type_mult × time_decay

))

```base:       +10 BULLISH, −10 BEARISH

dte_mult:    2.0 (0DTE) | 1.5 (1DTE) | 1.0 (2-7DTE) | 0.5 (8+ DTE)

Then **quality gate** (hard blocks — no confidence override) → **TOD ceiling** → **threshold filter** at 70%.prem_mult:   log10(premium / $100K) + 1  [capped 0.5–3.0]

bias_mult:   1.3 (ASK exec) | 1.0 (MID) | 0.7 (BID exec)

**Time-of-day hard caps** (PC_RATIO_EXTREME only):voi_mult:    1.5 (vol/OI > 3×) | 1.2 (vol/OI > 1×)

delta_mult:  1.0 (ATM: |Δ| 0.25–0.75) | 0.6 (far OTM)

| Bucket | Hours (ET) | Max confidence |type_mult:   1.2 (sweep or block)

|---|---|---|time_decay:  exp(−age_min / 120)     half-life ~83 min

| PRE_MARKET | before 9:30 | 0.75 |```

| OPEN | 9:30–10:00 | 0.82 |

| PRIME | 10:00–11:30 | 0.95 |Aggregate score normalised to **−100 to +100**.

| LUNCH | 11:30–13:30 | 0.80 |

| AFTERNOON | 13:30–15:00 | 0.88 |**Gamma Exposure (GEX)**:

| CLOSE | 15:00–16:00 | 0.93 |```

net_gex = Σ(call_gamma × call_OI) − Σ(put_gamma × put_OI)

**Opening period cap**: ALL directional signals capped at 0.82 during 9:30–10:00 ET (market-maker positioning noise).Positive → dealers long gamma → dampening → RANGE signal

Negative → dealers short gamma → amplifying → TREND signal

**Confidence tiers** (from `SpyOptionsSignalConfig`):```

Labels: `SUPPORTIVE_UPSIDE` / `SUPPORTIVE_DOWNSIDE` / `NEUTRAL`

| Score | Tier |

|---|---|**Dark pool bias**: Attempted from Alpha Query free endpoint; falls back to large-block premium skew (institutional $1M+ trades). Labels: `ACCUMULATION` / `DISTRIBUTION` / `NEUTRAL`

| 70–79% | MEDIUM — worth watching |

| 80–89% | ★ HIGH — strong signal |**Barchart scrape**: Attempted with proper User-Agent headers; returns empty list on failure (anti-bot, rate limit) — yfinance is always the fallback.

| 90–95% | ★★ EXTREME ★★ — hard cap at 95% |

**Gamma Walls** (computed from the same yfinance chain, no additional cost):

---

| Level | Definition | Effect |

## 7. Base Confidence Components|---|---|---|

| Call Wall | Strike with highest cumulative call OI | Resistance; penalises call signals by −5% |

10 weighted factors produce the raw base confidence (0.0–1.0):| Put Wall | Strike with highest cumulative put OI | Support floor; penalises put signals by −5% |

| Gamma Flip | Strike where net GEX changes sign | Key transition level — noted in signal reasoning |

| # | Component | Weight | Detail || `at_call_wall` | SPY within 0.3% of call wall | Active resistance flag |

|---|---|---|---|| `at_put_wall` | SPY within 0.3% of put wall | Active support flag |

| 1 | Volume spike strength | 25% | `(spike_mult − threshold) / 10`, normalised 0–1 |

| 2 | Bid/ask imbalance | 15% | Aggressor at bid (calls) or ask (puts) vs threshold |### 8. Market Breadth — Sector ETF Participation Proxy

| 3 | Delta quality | 10% | 1.0 if calls 0.30–0.60 or puts −0.60 to −0.30; else 0.3 |Uses yfinance to download 11 SPDR sector ETFs (XLK, XLF, XLE, XLI, XLV, XLB, XLU, XLRE, XLP, XLY, XLC) on a 5-min intraday bar:

| 4 | Gamma quality | 10% | 1.0 if 0.005–0.08 (ATM sweet spot); else 0.3 |

| 5 | Theta penalty | 10% | 0.2 if < −0.15/day; 0.6 if < −0.08; else 1.0 |- **Breadth ratio**: fraction of sectors currently above their day-open (0.0–1.0)

| 6 | IV regime alignment | 10% | Low IV → debit good (rank/100); high → credit good (1−rank/100) |- **Labels**: STRONG (≥75%), MODERATE (≥55%), NEUTRAL, MODERATE_WEAK (≤45%), WEAK (≤25%)

| 7 | Sentiment alignment | 10% | IB sentiment score normalised to signal direction |- **Up/Down volume proxy**: SPY 5-min bars where close > prev_close = up-volume; opposite = down-volume

| 8 | Open interest strength | 5% | `min(1.0, OI / 10,000)` |- **NYSE TICK**: attempted via `^TICK` (yfinance — not always available; graceful fallback)

| 9 | Flow score (repeat sweeps) | 5% | 0.5 for 2× same-strike hits in 15 min; 1.0 for 3×+ |- 10-minute TTL; no API key

| 10 | External composite | ±5% | `composite_score × composite_confidence_boost` directional |

**Confidence effect**: STRONG breadth → +3% for calls; WEAK breadth → +3% for puts; opposite direction → −5%

---

### 9. Sector Leadership — Tech & Momentum Proxy

## 8. Dynamic Confidence AdjustmentsSector ETFs measured vs day-open: XLK, XLF, SMH (semis), IWM (small cap), QQQ (Nasdaq), XLE, XLI.



19 additive factors applied after base confidence and event-risk modifier:| Metric | Source | Meaning |

|---|---|---|

| # | Factor | Conditions → Delta || `sector_label` | ETF bull/bear count | BULL_SWEEP / BULL_LEANING / MIXED / BEAR_LEANING / BEAR_SWEEP |

|---|---|---|| `qqq_vs_spy_pct` | QQQ intraday % − SPY % | Nasdaq leading/lagging |

| 1 | **Time of day** | OPEN (pre-10:15 unconfirmed −4%, confirmed +5%; post-10:15 +10%), MIDDAY −5%, POWER_HOUR +5% || `iwm_vs_spy_pct` | IWM intraday % − SPY % | Small-cap risk appetite |

| 2 | **DTE rules** | 0DTE unconfirmed −8%, 0DTE confirmed +2%, swing +2% if flow moderate || `es_premium` | (ES=F / 10) − SPY | Futures premium (positive = futures leading) |

| 3 | **Flow alignment** | Strong ±40: +10%/−10%. Moderate ±20: +5%/−5%. Weak ±10: +3%/−3% || `gap_pct` | Today's open vs prior close | Opening gap context |

| 4 | **Macro environment** | Headwind < −0.40: calls −8%, puts +5%. Tailwind > +0.30: calls +5%, puts −5%. VIX > 30 calls −5%. VIX < 14 puts −3% || `above_overnight_high` | SPY > today's range high | Breakout flag |

| 5 | **Event risk** | Within 30 min of high-impact event: directional −5% (on top of −30% from event modifier) || `usdjpy_trend` | JPY=X inverted daily change | RISK_ON / RISK_OFF / NEUTRAL |

| 6 | **Conflict detection** | ≥2 of {flow, sentiment, macro, regime} oppose direction → −8% |

| 7 | **Market breadth** | ≥70% sectors up + call: +3%. ≤30% + put: +3%. Opposite: −5% |10-minute intraday TTL; daily TTL for prior close + USDJPY.

| 8 | **Sector leadership** | BULL_SWEEP + call: +3%. QQQ leading + call: +2%. QQQ green + put: −8% (rotation, not selloff). IWM strong + put: −4% |

| 9 | **Gamma walls** | At call wall + call: −5%. At call wall + put: +3%. At put wall + put: −5%. At put wall + call: +3% |**Confidence effect**: BULL_SWEEP → +3% for calls; BEAR_SWEEP → +3% for puts. QQQ leading adds +2% for confirmed calls. Breaking above overnight high → +3% for calls; below overnight low → +3% for puts.

| 10 | **Vol term structure** | Backwardation + call: −5%. Steep backwardation + put: +3%. Contango + call: +2%. VVIX > 115: −3% any directional |

| 11 | **Overnight context** | Above overnight high + call: +3%. Below overnight low + put: +3%. Against: −2% |### 10. Volatility Term Structure — VIX/VXV + VVIX

| 12 | **OPEX gamma environment** | PINNING: −3% directional. EXPANSIVE: +2% directional |Free via yfinance — 15-minute TTL.

| 13 | **ORB breakout** | Confirmed above + call: +5%. Confirmed below + put: +5%. Inside after 10:00: −3% |

| 14 | **VWAP band exhaustion** | ±2σ fade: +5%. ±2σ chase: −6%. ±1σ: ±2% || Ticker | Measures |

| 15 | **EDR exhaustion** | ≥120%: −8%. ≥85%: −5%. ≥60%: −2% ||---|---|

| 16 | **RSI divergence** | Bearish div + put: +5%. Bullish div + call: +5%. Against: −5%. OB/OS only: ±2% || `^VIX` | 30-day implied vol |

| 17 | **Pivot proximity** | At R1/R2 + call: −4%, + put: +3%. At S1/S2 + put: −4%, + call: +3%. At PP: −2% || `^VXV` | 93-day implied vol |

| 18 | **Max pain proximity** | 0DTE near max pain: −6%. 1–2 DTE: −3% || `^VVIX` | Volatility of VIX (vol-of-vol) |

| 19 | **Dark pool bias** | DISTRIBUTION + put: +3%, + call: −3%. ACCUMULATION + call: +3%, + put: −3% |

**VIX/VXV ratio** → term structure classification:

**Stale-flow decay** (PC_RATIO signals only): if extreme P/C persists without price follow-through (>0.15% move), confidence decays: 15–30 min → −2%, 30–45 min → −4%, 45–60 min → −7%, 60+ min → −10%.

| Ratio | Label | Meaning |

---|---|---|---|

| < 0.85 | STEEP_CONTANGO | Calm market; vol term structure is normal |

## 9. Quality Gate| 0.85–0.95 | CONTANGO | Normal; near-term vol lower than longer-dated |

| 0.95–1.00 | FLAT | Compressed structure |

**Hard blocks** — confidence cannot override these. Signal is dropped regardless of score.| 1.00–1.10 | BACKWARDATION | Fear spike; near-term vol > 93-day |

| > 1.10 | STEEP_BACKWARDATION | Panic; acute short-term fear |

| # | Block | Exact Condition |

|---|---|---|**VVIX elevated** (> 115): vol-of-vol is high → reduce conviction on any directional signal (−3%).

| 1 | Flow opposes direction | Flow ≤ −30 for CALL, or ≥ +30 for PUT |

| 2 | 0DTE near max pain | DTE=0 AND SPY within $1.50 of max pain strike |**Confidence effect**: BACKWARDATION → −5% for calls; CONTANGO → +2% for calls.

| 3 | 0DTE missing confirmation | DTE=0 AND flow < ±25 AND no aligned ORB breakout |

| 4 | Inside ORB after 10:30 ET | Directional signal AND ORB established AND price INSIDE range |### 11. OPEX Calendar — Dealer Gamma Environment

| 5 | Chop day | Directional AND SPY intraday range < 0.20% |Pure Python date math — zero network calls, zero latency.



**Summary rule**: Never trade against flow. If options flow strongly opposes your signal direction, the signal is dead on arrival.| Field | Definition |

|---|---|

**Cross-signal conflict filter**: If both CALL and PUT signals pass threshold in the same cycle, only the higher-confidence direction survives. Applied both within-expiry and cross-expiry.| `next_opex` | Date of next monthly OPEX (3rd Friday of month) |

| `days_to_opex` | Calendar days to next OPEX |

---| `is_opex_week` | True if 0–4 days away |

| `is_opex_day` | True if today is OPEX Friday |

## 10. Example Trade Walkthrough| `is_triple_witching` | OPEX is in March, June, September, or December |

| `gamma_environment` | PINNING / EXPANSIVE / NEUTRAL |

**9:52 AM ET** — OPEN bucket, 0DTE SPY 545C.

**Gamma environment logic**:

### Stage 1: Base confidence (10 components)- **PINNING**: opex week + dealers net long gamma (positive net GEX) → dealers sell rallies/buy dips → price tends to pin

- **EXPANSIVE**: dealers net short gamma OR far from OPEX → moves accelerate → momentum works better

| Component | Raw | Weighted |- **NEUTRAL**: otherwise

|---|---|---|

| Volume spike: 7× avg | 0.30 | 0.25 × 1.0 |**Confidence effect**: PINNING → −3% for directional signals; EXPANSIVE → +2% for directional signals.

| Bid/ask imbalance: 4.2× | 0.15 | — |

| Delta: 0.44 (ideal) | 0.10 | — |---

| Gamma: 0.025 (in range) | 0.10 | — |

| Theta: −0.05 (mild) | 0.10 | — |## External Composite Score

| IV rank: 38% | 0.062 | — |

| Sentiment: +45 bullish | 0.072 | — |All weighted sources combined into a single score (−1.0 to +1.0), dynamically normalised:

| OI: 8,500 | 0.043 | — |

| Flow: 0.62 (repeat) | 0.031 | — || Source | Default Weight | Note |

| Ext composite: +0.35 | +0.018 | — ||---|---|---|

| **base_confidence** | | **0.726** || Options flow confirmation | 20% | yfinance always available |

| News RSS sentiment | 20% | No API key |

### Stage 2: Signal-specific adjustments| Macro headwind | 20% | Daily + 15-min intraday |

| StockTwits retail | 15% | Free API |

| Adjustment | Delta || CBOE P/C contrarian | 15% | Daily |

|---|---|| Reddit enhanced | 10% | Only when credentials provided |

| ORB confirmed ABOVE_ORB | +0.03 |

| VWAP: ABOVE_1SD (caution) | −0.02 |Weights are **dynamically normalised** when sources are unavailable. Score shown as zero if fewer than 2 sources have data.

| Regime: TREND_UP | +0.05 |

| **After adjustments** | **0.786** |**Base confidence impact**: ±5% max from composite alignment alone.



### Stage 3: Dynamic confidence (19 factors)The following sources are **not included in the composite score** but directly adjust confidence in the Dynamic Confidence Engine:



| Factor | Delta || Source | Confidence Adjustments |

|---|---||---|---|

| Time of day (OPEN, confirmed) | +0.037 || Market breadth | ±3–5% based on breadth_ratio and signal direction |

| Flow alignment (STRONG +63) | +0.10 || Sector leadership | ±3–5% based on sector_label alignment; +2% if QQQ leads |

| Macro (mild tailwind) | +0.05 || Gamma walls | −5% if at resistance/support wall; +3% if supportive side |

| ORB confirmed above | +0.05 || Volatility term structure | −5% (backwardation), +2% (contango); −3% if VVIX elevated |

| **final_confidence** | **0.84** || Overnight context | +3% breaking range high/low; −2% trading against range |

| OPEX gamma environment | −3% (PINNING), +2% (EXPANSIVE) |

### Stage 4–5: Priority check + Quality gate| **ORB breakout** | ±5% confirmed breakout/breakdown direction |

| **VWAP bands** | ±2–6% based on extension zone (±2σ = strongest) |

- Tier-1: flow confirms (+63), ORB confirms → +5% (but already counted in dynamic)| **EDR exhaustion** | −2% to −8% when VIX-implied range is consumed |

- Quality gate: no blocks triggered| **RSI divergence** | ±5% when price/RSI diverge; ±2% for overbought/oversold |

- **Result: 84% → HIGH tier → DISPATCHED ✓**| **Pivot proximity** | ±3–4% at R1/R2/S1/S2; −2% at PP |

- Signal: `CALL_SWEEP 545C exp APR26 confidence=84%`| **Max pain** | −3% to −6% when near pin target (0DTE strongest) |



------



## 11. Exit Strategy## Intraday Technical Levels (TechnicalLevelsTracker)



After sending a directional entry alert, the bot monitors every poll cycle. **One EXIT per signal**; auto-expire after 2h (0DTE) or 6h (swing).Pure Python computation from the 5-minute IB bars — zero network calls, zero latency. Computed every poll cycle and injected into `ExternalContext` before signals are evaluated.



**6 independent triggers** (any one fires the EXIT alert):### 1. Opening Range Breakout (ORB)



| # | Trigger | Condition |The single most-watched SPY intraday reference. Institutional desks and systematic traders use the first 30-minute range (9:30–10:00 ET) as the primary directional filter.

|---|---|---|

| 1 | Price adverse | SPY moved ≥ 0.5% against signal direction || Field | Description |

| 2 | Regime flip | TREND_UP → TREND_DOWN (for bullish), or vice-versa ||---|---|

| 3 | Urgent adverse | SPY moved ≥ 1.0% against — escalation || `orb_high` / `orb_low` | High/low of the 9:30–10:00 ET session |

| 4 | Time stop | 0DTE: 30 min. Swing: 60 min || `orb_established` | True once the 10:00 ET build window closes |

| 5 | Profit target | SPY moved ≥ 0.5% in signal direction — "consider taking profits" || `orb_status` | `BUILDING` / `INSIDE` / `ABOVE_ORB` / `BELOW_ORB` |

| 6 | VWAP reversion | SPY crossed back through VWAP from entry side (thesis weakened) || `orb_breakout_confirmed` | Price closed outside range for ≥1 bar |

| `orb_width_pct` | Range width as % of SPY price (tight < 0.20% = reliable) |

**Monitored signal types**: CALL_SWEEP, PUT_SWEEP, BULL_CALL_SPREAD, BEAR_PUT_SPREAD, PC_RATIO_EXTREME, ORB_BREAKOUT.

**Confidence effect**: Confirmed breakout above ORB → +5% for calls, −5% for puts. Confirmed breakdown → +5% for puts, −5% for calls. Price stuck inside ORB after 10:00 ET → −3% for all directional signals.

---

### 2. VWAP Standard Deviation Bands

## 12. Top 5 Factors That Actually Matter

Volume-weighted standard deviation computed from the full session's bar data. SPY rarely sustains above ±2σ — these are fade zones, not continuation zones.

Based on the confidence model weights and quality gate design, these are the factors with the most real impact on whether a signal fires and at what confidence:

| Band Position | Meaning | Effect |

1. **Options flow** (Tier-1) — strong flow ≥ ±40 adds +10%, strong opposing flow is a hard block. Repeat sweeps (3× in 15 min) add +5% base. This is the single most important input.|---|---|---|

| `ABOVE_2SD` | Stretched above normal range | +5% puts / −6% calls |

2. **ORB breakout** (Tier-1) — confirmed breakout adds +5% from dynamic confidence, +5% from signal-specific, and gives Tier-1 confirmation. Inside ORB after 10:30 is a hard block for directional signals.| `ABOVE_1SD` | Elevated | +2% puts / −2% calls |

| `INSIDE_1SD` | Normal range | No adjustment |

3. **Market regime** — TREND_UP/DOWN alignment adds +5% signal-specific. Regime flip is an exit trigger. Regime opposing is a Tier-2 −5% penalty and counts toward the 4-factor conflict detection.| `BELOW_1SD` | Depressed | +2% calls / −2% puts |

| `BELOW_2SD` | Stretched below normal range | +5% calls / −6% puts |

4. **IV rank + VIX level** — Determines spread vs naked signal type (IV < 30 → debit spreads). VIX > 30 penalises calls −5%. VIX < 14 penalises puts −3%. IV regime alignment is 10% of base confidence.

### 3. Daily Pivot Points (Floor-Trader Formula)

5. **Time of day** — OPEN bucket gives +5–10% but is capped at 0.82 for the first 30 min. MIDDAY penalises −5%. POWER_HOUR boosts +5%. PC_RATIO signals have per-bucket hard caps (LUNCH = 0.80 max). The best signals fire during the 10:00–11:30 PRIME window.

Computed from the prior session's high, low, and close. SPY exhibits measurable mean-reversion at these levels intraday.

---

```

## 13. Performance Tracking & Feedback LoopPP = (prev_high + prev_low + prev_close) / 3

R1 = 2×PP − prev_low       R2 = PP + (prev_high − prev_low)

Every signal is persisted to `data/spy_options_signals.db` (SQLite, auto-migrating schema).S1 = 2×PP − prev_high      S2 = PP − (prev_high − prev_low)

```

**Schema** (key fields): signal_type, strike, right, expiry, dte, confidence, confidence_tier, spy_price, vix, iv_rank, regime, sentiment_score, flow_score, flow_confirmation_score, dark_pool_bias, gex_bias, external_composite, macro_headwind, dynamic_confidence_delta, confidence_time_bucket, conflict_detected, reasoning (JSON), outcome, spy_price_exit, pnl_pct, exit_trigger.

**Proximity threshold**: 0.3% of SPY price or $1.50, whichever is larger.

**Outcome tracking**: When an exit trigger fires, `record_outcome()` writes:

- `outcome`: win (>+0.1% favourable), loss (<−0.1%), or scratch**Confidence effect**:

- `pnl_pct`: SPY move from entry to exit in signal direction (+ = favourable)- At R1/R2 (resistance) + call signal → −4% (buying into resistance)

- `exit_trigger`: time_stop, profit_target, adverse_move, regime_flip, vwap_reversion, manual- At R1/R2 + put signal → +3% (natural ceiling)

- At S1/S2 (support) + put signal → −4% (shorting at support)

**4 query methods** for tuning:- At S1/S2 + call signal → +3% (natural floor)

- At PP exactly → −2% (direction undecided)

| Method | Breaks down by | Use case |

|---|---|---|### 4. Expected Daily Range (EDR) Exhaustion

| `win_rate_summary()` | Overall | Total wins/losses/scratches, avg P&L, best/worst |

| `win_rate_by_signal_type()` | Signal type | Disable underperforming signal types |VIX implies a 1-σ expected intraday range for SPY. When most of that range is already consumed, fade signals are more likely than continuation signals. This is the most common 0DTE afternoon over-trade mistake.

| `win_rate_by_time_bucket()` | OPEN/MIDDAY/etc | Adjust time-of-day multipliers |

| `win_rate_by_regime()` | TREND_UP/DOWN/etc | Tune regime-based adjustments |```

EDR (points) = VIX / √252 / 100 × SPY_price

---EDR used %   = max(high − open, open − low) / EDR × 100

```

## 14. Known Weaknesses

**Confidence effect**:

| # | Weakness | Why it's hard to fix |

|---|---|---|| EDR consumed | Effect on directional signals |

| 1 | **Chop days** (SPY range < 0.20%) | Quality gate blocks directional signals, but the bot can't predict chop in advance — it only detects it after the range is established ||---|---|

| 2 | **Fake ORB breakouts** | ORB requires ≥1 bar close outside range, but false breakouts on low volume still slip through; no volume confirmation on ORB itself || ≥ 120% (extreme extension) | −8% |

| 3 | **News spike whipsaws** | Event risk modifier applies −30% to directional signals near events, but the market can gap through any level. No real-time headline parsing — only Forex Factory calendar || ≥ 85% (exhausted) | −5% |

| 4 | **0DTE gamma risk** | 0DTE options have extreme gamma — delta can flip 0.30 → 0.80 in minutes. Quality gate blocks near max pain, but fast moves can still trap || ≥ 60% (getting stretched) | −2% |

| 5 | **External data latency** | yfinance options chain is 15-min delayed. Dark pool data from Alpha Query is best-effort. Barchart scrape usually fails (anti-bot). Flow signals are already stale by the time they're processed || < 60% | No adjustment |

| 6 | **Over-engineering risk** | 19 dynamic confidence factors + signal-specific adjustments + priority hierarchy means a lot of parameters to tune. Additive adjustments can cancel each other out or produce unexpected edge cases |

### 5. RSI (5-min) with Divergence Detection

---

14-period Wilder RSI computed on the 5-minute close series. Divergence detection compares price direction vs RSI direction over the last 8 bars.

## 15. External Data Sources

**Divergence rules**:

10 sources (11 counting Reddit separately). Each can be individually enabled/disabled via config flags.- **Bearish divergence**: price making higher highs but RSI not following — only flagged when RSI is still elevated (> 55 now, > 60 at lookback start)

- **Bullish divergence**: price making lower lows but RSI recovering — only flagged when RSI is still depressed (< 45 now, < 40 at lookback start)

| # | Source | Data | Config flag | TTL | Requires |

|---|---|---|---|---|---|**Confidence effect**:

| 1 | **Economic calendar** | FOMC, CPI, NFP, Fed events | `calendar_enabled` | Daily | Nothing (Forex Factory JSON) |

| 2 | **News RSS** | Headlines + VADER sentiment | `news_enabled` | 10 min | Nothing || Condition | Effect |

| 3 | **StockTwits** | Retail bull/bear % | `stocktwits_enabled` | 10 min | Nothing ||---|---|

| 4 | **Reddit** | WSB/options body+comments+karma | `reddit_enabled` | 15 min | Reddit app credentials (opt-in) || Bearish divergence + put signal | +5% |

| 5 | **Macro signals** | TNX, DXY, VIX, oil, gold | `macro_enabled` | Daily + 15 min | Nothing (yfinance) || Bearish divergence + call signal | −5% |

| 6 | **CBOE P/C ratio** | Equity put/call daily | `cboe_enabled` | Daily | Nothing (CSV) || Bullish divergence + call signal | +5% |

| 7 | **Options flow** | Unusual activity, GEX, dark pool, gamma walls | `flow_enabled` | 10 min | Nothing (yfinance chain) || Bullish divergence + put signal | −5% |

| 8 | **Market breadth** | 11 sector ETFs above-open ratio | `breadth_enabled` | 10 min | Nothing (yfinance) || RSI overbought (≥70) + put | +2% |

| 9 | **Sector leadership** | XLK/XLF/SMH/IWM/QQQ vs open, gap, overnight | `sector_enabled` | 10 min | Nothing (yfinance) || RSI oversold (≤30) + call | +2% |

| 10 | **Vol structure** | VIX/VXV ratio, VVIX (vol-of-vol) | `vol_structure_enabled` | 15 min | Nothing (yfinance) |

| — | **OPEX calendar** | Monthly OPEX dates, gamma environment | `opex_enabled` | Per poll | Nothing (pure Python) |### 6. Max Pain Strike



**External priority rule**: Sources 1–6 feed the **composite score** (weighted average, ±5% max impact). Sources 7–10 + OPEX + technical levels (ORB/VWAP/pivots/EDR/RSI/max pain) adjust confidence **directly** in the dynamic confidence engine — they are not averaged into the composite.Computed from the live IB options chain open interest every poll cycle (first available expiry). Max pain is the closing price where total option-buyer losses are maximised — the pinning target for market makers.



**Composite weights** (dynamically normalised when sources unavailable):```

pain(K) = Σ_calls(max(S − K, 0) × OI[S]) + Σ_puts(max(K − S, 0) × OI[S])

| Source | Weight |max_pain = K that minimises pain(K)

|---|---|```

| Options flow confirmation | 20% |

| News RSS sentiment | 20% |**Near max pain**: SPY within $1.50 of the max pain strike.

| Macro headwind | 20% |

| StockTwits retail | 15% |**Confidence effect**:

| CBOE P/C contrarian | 15% |

| Reddit enhanced | 10% || Condition | Effect |

|---|---|

---| Near max pain + 0DTE directional | −6% |

| Near max pain + 1–2 DTE directional | −3% |

## 16. Configuration Reference

---

Full annotated YAML block (`config.yaml` → `spy_options:` section):

## Dynamic Confidence Engine

```yaml

spy_options:After the base confidence model (10 components) and event-risk modifier, a second layer of adjustments is applied that adapts to:

  enabled: false                          # Master kill switch

> **Adjustment blocks 1–12** cover time-of-day, DTE, flow alignment, macro environment, event risk, conflict detection, market breadth, sector leadership, gamma walls, volatility term structure, overnight context, and OPEX gamma environment — all described below.

  ib:>

    ibkr_host: "127.0.0.1"> **Adjustment blocks 13–18** are the new intraday technical level adjustments added in the April 2026 update.

    ibkr_port: 4001                       # 4001=live, 4002=paper

    ibkr_client_id: 5                     # Unique — MES=1, VX=2, Gold=3### Time of Day (ET)

    snapshot_wait_s: 3.0                  # Wait for price snapshot

    greeks_wait_s: 4.0                    # Extra wait for modelGreeks| Bucket | Hours | Adjustment |

    max_subscriptions: 60                 # IB line cap (~100 max)|---|---|---|

| **OPEN** | 9:30–10:30 | +10% (breakout/momentum hour) |

  chain:| **MIDDAY** | 10:30–14:00 | −5% (chop zone) |

    strike_pct_range: 0.04                # ±4% of SPY price| **PRE_POWER** | 14:00–15:00 | neutral |

    max_strikes_per_expiry: 30            # Calls + puts combined| **POWER_HOUR** | 15:00–16:00 | +5% (directional moves solidify) |

    num_expiries: 2                       # Current + next expiry

    exchange: "SMART"### DTE Rules

    conid_resolve_delay_s: 0.1

    liquidity_min_oi: 1000                # Min open interest| DTE | Rule | Effect |

    liquidity_max_spread_pct: 8.0         # Max bid/ask spread %|---|---|---|

    liquidity_min_volume: 500             # Min daily volume| 0 (0DTE) | Requires flow score ≥ ±25 | Unconfirmed: −8%; Confirmed: +2% |

| 1 | Standard | No adjustment |

  signals:| 2–7 | Swing | +2% when moderate flow aligns |

    volume_spike_mult: 4.0                # Spike = increment ≥ 4× rolling avg| 8+ | Standard | No adjustment |

    min_volume_for_signal: 500            # Ignore below this session volume

    sweep_poll_volume_threshold: 300      # Min contracts per poll for sweep### Flow Alignment

    vix_low: 16.0                         # Below → debit spreads preferred

    vix_high: 26.0                        # Above → premium selling| Flow strength | Signal aligns | Signal conflicts |

    pc_ratio_bearish: 1.8                 # P/C > 1.8 → bearish extreme|---|---|---|

    pc_ratio_bullish: 0.5                 # P/C < 0.5 → bullish extreme| Strong (≥ ±40) | +10% | −10% |

    bid_ask_imbalance_threshold: 3.0| Moderate (≥ ±20) | +5% | −5% |

    straddle_spike_mult: 3.0| Weak (≥ ±10) | +3% | −3% |

    min_confidence: 0.70                  # Drop below this

    confidence_tier_high: 0.80            # ★ HIGH starts here### Macro Environment

    confidence_tier_extreme: 0.90         # ★★ EXTREME starts here

    pc_ratio_min_denom_volume: 200        # Min volume for P/C ratio denom| Condition | Effect |

    pc_ratio_flip_cooldown_minutes: 30    # Suppress opposite direction|---|---|

    dedup_window_minutes: 90              # Same signal suppression window| Strong headwind (< −0.40) + call signal | −8% |

    max_signals_per_day: 10               # Daily cap| Strong tailwind (> +0.30) + call signal | +5% |

    sweep_window_minutes: 15              # Repeat sweep tracking window| High VIX (> 30) + call signal | −5% |

| Low VIX (< 14) + put signal | −3% |

  session:

    rth_only: true### Conflict Detection

    rth_start_et: "09:35"                 # Skip first 5 min (MM noise)

    rth_stop_et: "15:45"                  # Stop 15 min before closeIf **2 or more** of the following four sources oppose the signal direction → confidence −8% with `conflict_detected=True`:

    poll_interval_s: 60                   # Main loop interval

1. **Flow score** (opposing flow > ±20)

  analytics:2. **IB sentiment** (opposing sentiment > ±20)

    enabled: true3. **Macro headwind** (opposing macro > ±0.20)

    db_path: "data/spy_options_signals.db"4. **Market regime** (TREND_DOWN opposes calls; TREND_UP opposes puts)



  external:### Market Breadth Adjustment

    enabled: true

    calendar_enabled: true| Condition | Effect |

    event_risk_window_minutes: 30|---|---|

    news_enabled: true| breadth_ratio ≥ 70% + call signal | +3% |

    news_ttl_minutes: 10.0| breadth_ratio ≤ 30% + put signal | +3% |

    stocktwits_enabled: true| breadth_ratio ≤ 30% + call signal | −5% |

    stocktwits_ttl_minutes: 10.0| breadth_ratio ≥ 70% + put signal | −5% |

    reddit_enabled: false                 # Opt-in — needs credentials

    reddit_client_id: ""### Sector Leadership Adjustment

    reddit_client_secret: ""

    reddit_ttl_minutes: 15.0| Condition | Effect |

    macro_enabled: true|---|---|

    cboe_enabled: true| BULL_SWEEP / BULL_LEANING + call | +3% |

    flow_enabled: true| BEAR_SWEEP / BEAR_LEANING + put | +3% |

    flow_ttl_minutes: 10.0| Opposite-direction sweep | −3% |

    flow_barchart_enabled: true           # Best-effort scrape| QQQ leading (> +0.20% vs SPY) + confirmed call | +2% additional |

    flow_dark_pool_enabled: true          # Alpha Query + premium skew

    breadth_enabled: true### Gamma Wall Adjustment

    breadth_ttl_minutes: 10.0

    sector_enabled: true| Condition | Effect |

    sector_ttl_minutes: 10.0|---|---|

    vol_structure_enabled: true| SPY at call wall + call signal | −5% (resistance) |

    vol_structure_ttl_minutes: 15.0| SPY at call wall + put signal | +3% (natural ceiling) |

    opex_enabled: true| SPY at put wall + put signal | −5% (support absorbs puts) |

    composite_confidence_boost: 0.05      # Max ±5% from external composite| SPY at put wall + call signal | +3% (floor support) |



  log_file: "logs/spy_options.log"### Volatility Term Structure Adjustment

```

| Condition | Effect |

**Hardcoded values** (not configurable — change in source):|---|---|

| BACKWARDATION + call signal | −5% |

| Parameter | Value | Location || STEEP_BACKWARDATION + put signal | +3% |

|---|---|---|| CONTANGO + call signal | +2% |

| Exit price threshold | 0.5% adverse | `manager.py` || VVIX elevated (> 115) | −3% on any directional |

| Exit urgent threshold | 1.0% adverse | `manager.py` |

| Exit profit target | 0.5% favourable | `manager.py` |### Overnight Context Adjustment

| Time stop: 0DTE | 30 min | `manager.py` |

| Time stop: swing | 60 min | `manager.py` || Condition | Effect |

| Signal max age: 0DTE | 2 hours | `manager.py` ||---|---|

| Signal max age: swing | 6 hours | `manager.py` || SPY above overnight range high + call | +3% (breakout momentum) |

| ORB build window | 9:30–10:00 ET | `technical_levels.py` || SPY below overnight range low + put | +3% (breakdown momentum) |

| Pivot proximity | 0.3% or $1.50 | `technical_levels.py` || SPY above high + put signal | −2% (against momentum) |

| EDR exhaustion threshold | 85% | `dynamic_confidence.py` || SPY below low + call signal | −2% (against momentum) |

| RSI period (5-min) | 14 | `technical_levels.py` |

| RSI divergence lookback | 8 bars (40 min) | `technical_levels.py` |### OPEX Gamma Environment Adjustment

| Max pain proximity | $1.50 | `manager.py` |

| 0DTE min flow score | ±25 | `signal_engine.py` || Environment | Effect |

| Conflict penalty | −8% | `dynamic_confidence.py` ||---|---|

| Same-direction throttle | 3 per 90 min | `manager.py` || PINNING (opex week + positive GEX) | −3% for directional signals |

| Opening period cap | 0.82 (9:30–10:00) | `signal_engine.py` || EXPANSIVE (short gamma or far from OPEX) | +2% for directional signals |

| Keepalive interval | 30s | `ib_client.py` |

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

## Telegram Alert Format

Alerts are intentionally rich but structured. Each **entry alert** includes:

- Contract and market snapshot (SPY, VIX, IV rank, regime)
- Liquidity + Greek context (bid/ask, spread, OI, delta/gamma/theta/vega/IV)
- Confidence context (tier, dynamic adjustment, key reasoning bullets)
- Suggested trade framing + explicit SPY-based exit trigger levels

Each **exit alert** includes:

- Original signal direction and entry SPY
- Current SPY move in points and %
- Trigger reason(s): adverse move, regime flip, or urgent adverse threshold

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
