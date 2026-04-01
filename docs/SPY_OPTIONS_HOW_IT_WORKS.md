# SPY Options Bot — How It Works

## The Basic Idea

Every 60 seconds during market hours (9:35–3:45 ET), the bot:
1. Fetches SPY price, VIX, 5-minute bars, and option chain data from IB Gateway
2. Classifies the market regime and scores sentiment
3. Fetches Greeks (delta, gamma, theta, vega, IV) for every tracked contract
4. Runs a 10-component weighted confidence model to score signals
5. Sends Telegram alerts only — no trades placed, ever

---

## Poll Cycle (What Happens Every 60 Seconds)

```
1. Fetch SPY price          (IB snapshot)
2. Fetch VIX                (IB snapshot)
3. Fetch SPY 5-min bars     (IB historical — for regime + sentiment)
4. Classify market regime   (EMA, ATR, VWAP — pure math)
5. Score sentiment          (VIX trend + VWAP position + EMA slope)
6. Compute IV rank          (VIX vs 52-week range, cached at startup)
7. For each of 2 expiries:
   a. Resolve option conids (cached after day 1)
   b. Fetch price + Greeks  (snapshot=False, explicit cancel after read)
   c. Apply liquidity filter (OI, spread, volume)
   d. Run signal engine     (7 signal types, weighted confidence)
8. Deduplicate signals
9. Send Telegram alerts
10. Persist to SQLite analytics DB
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

## Sentiment Score

A single score from **-100 (bearish) to +100 (bullish)** computed from IB data only — no Twitter, no Reddit, no external APIs:

| Component | Weight | How it's calculated |
|---|---|---|
| VIX trend | 40% | Rising VIX > 5% above recent avg = -40; falling = +40 |
| SPY vs VWAP | 35% | Distance above/below VWAP, normalized by ATR |
| EMA slope | 25% | Rate of EMA-9 change, normalized by ATR |

Labels: **BULLISH** (>+25), **NEUTRAL** (-25 to +25), **BEARISH** (<-25)

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

Replaces the old simple scoring. Each signal is scored across 10 components:

| Component | Weight | Detail |
|---|---|---|
| Volume spike strength | 25% | How many × the rolling average |
| Bid/ask imbalance | 15% | Aggressor at bid vs ask |
| Delta quality | 10% | Is delta in the ideal 0.30–0.60 range? |
| Gamma quality | 10% | Is gamma in the ATM sweet spot? |
| Theta penalty | 10% | Heavy decay hurts confidence for longs |
| IV regime alignment | 10% | Low IV rank → debit good; high → credit good |
| Sentiment alignment | 10% | Bullish sentiment boosts calls, bearish boosts puts |
| Open interest strength | 5% | More OI = more confidence |
| Flow score (repeat sweeps) | 5% | 0.5 for 2 hits, 1.0 for 3+ hits in 15 min |

**Confidence tiers:**

| Score | Tier | Meaning |
|---|---|---|
| 70–79% | MEDIUM | Worth watching |
| 80–89% | ★ HIGH | Strong signal |
| 90%+ | ★★ EXTREME ★★ | Multiple confirming factors |

Signals below **70%** are dropped (was 55% in the old version).

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
| regime, sentiment_score | Market context |
| flow_score | Repeat sweep score |
| reasoning | Full analysis bullets |

Use this to track signal win rate, regime performance, and confidence model accuracy over time.

---

## Key Parameters

| Parameter | Value |
|---|---|
| Strike window | ±4% of current SPY price (ATM zone only) |
| Min session volume | 500 contracts |
| Min sweep size | 300 contracts in one 60s poll |
| Min open interest | 1,000 contracts |
| Max bid/ask spread | 8% of mid price |
| Spike multiplier | 4× rolling average |
| Expiries tracked | Current + next monthly expiry |
| Poll interval | Every 60 seconds |
| RTH session | 9:35 AM – 3:45 PM ET, weekdays only |
| Dedup window | 90 minutes per signal |
| Repeat sweep window | 15 minutes |
| Min confidence | 70% (weighted model) |
| Greeks wait | 4 seconds after subscription (IB model lag) |
| VIX 52w range | Fetched once at startup, cached for IV rank |
