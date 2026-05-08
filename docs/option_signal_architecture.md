OptionsEdge — Signal Generation Architecture
Principal Reference Document · Version 3.0
Full system design, decision logic, and implementation specification. No code required to read; sufficient to rebuild from.

Table of Contents
Philosophy & Design Goals
System Architecture — The Seven Layers
Data Model — What the System Knows About Each Ticker
Layer 1 — Price Level Classification
Layer 2 — Pattern Recognition
Layer 3 — Multi-Timeframe Alignment
Layer 4 — Session Phase & Volatility Regime
Layer 5 — Greeks & Options Intelligence
Layer 6 — Confluence Scoring Engine
Layer 7 — Risk Filters & Portfolio Controls
Edge Reality System (Institutional Audit Additions)
Position Sizing Engine
Signal Output — Final Presentation
Signal Log & Net P&L Accounting
End-to-End Flow — Step by Step
Worked Examples
Design Principles & Honest Limitations
Implementation Specification
1. Philosophy & Design Goals
OptionsEdge is not a black-box screener. It is a structured decision-support system that compresses the institutional pre-trade checklist into a real-time UI. Every signal the system generates is the same analysis a professional derivatives desk would perform before entering a 0DTE position — except it happens in seconds rather than minutes, and it is displayed in a way that forces the trader to see and acknowledge every risk factor before pressing the button.

The core problem it solves: Most retail 0DTE traders have intuition but no process. They see a pattern and take a trade without asking: Is the volatility regime right for this pattern? Does my target beat my transaction costs? Am I correlated to three other positions I'm already holding? OptionsEdge answers all of these before the trader ever opens an order ticket.

What it explicitly does not do:

Execute trades automatically
Guarantee any particular win rate
Replace judgment in fast-moving markets
Claim its confidence score is a probability — it is a relative ranking
Target user: Experienced options traders who understand what delta, gamma, and IV crush mean, who trade 0DTE or same-day options in liquid US equities and ETFs, and who want a rigorous pre-trade framework rather than a trade alert service.

2. System Architecture — The Seven Layers
Every signal passes through seven sequential layers. Each layer either blocks a signal (STAY OUT) or adjusts the score of a directional signal (CALL/PUT). A signal only earns Grade A or A+ if it passes all seven cleanly.

RAW PRICE & OPTIONS DATA
          │
          ▼
┌─────────────────────────────────────────────────────┐
│  LAYER 1: Price Level Classification                 │
│  Map current price to 16 key levels.                │
│  Identify which levels are "in play" (within ±0.4%) │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│  LAYER 2: Pattern Recognition                        │
│  Match price behavior to 1 of 13 named patterns.    │
│  Assign signal direction: CALL / PUT / STAY OUT      │
│  Assign base win rate (70–82%)                       │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│  LAYER 3: Multi-Timeframe Alignment                  │
│  Score Daily + 4H + 1H trend direction.             │
│  Counter-trend = 0.55× score multiplier             │
│  Full alignment = 1.00× (no penalty)                │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│  LAYER 4: Session Phase & Volatility Regime          │
│  6 session windows, each with a quality multiplier. │
│  4 vol regimes determine which patterns are valid.  │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│  LAYER 5: Greeks & Options Intelligence              │
│  IV penalty, DTE optimization, spread recommendation│
│  Skew awareness, IV-adjusted stop, dollar gamma     │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│  LAYER 6: Confluence Scoring Engine                  │
│  Multiplies all factors into a 0–100 score.         │
│  Adds bonus points for leading indicators.          │
│  Maps score to Grade: A+ / A / B / C / D            │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│  LAYER 7: Risk Filters & Portfolio Controls          │
│  Event calendar, portfolio Greeks, daily loss limit, │
│  correlation check, worst-case tail risk stress     │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
          FINAL SIGNAL OUTPUT
    ┌─────────────────────────────────┐
    │  Direction: CALL / PUT / STAY OUT│
    │  Grade: A+ / A / B / C / D       │
    │  Score: 0–100 (ranking only)     │
    │  Structure: Spread or Naked      │
    │  DTE recommendation              │
    │  IV-adjusted stop %              │
    │  Regime-conditioned win rate     │
    │  Break-even win rate             │
    │  Net edge margin                 │
    │  Position size (contracts)       │
    └─────────────────────────────────┘

3. Data Model — What the System Knows About Each Ticker
Every ticker in the watchlist carries a complete data object. This is populated from live market data feeds (or simulated data in a demo environment). Understanding this data model is the foundation for rebuilding the system.

3.1 Price Levels
Sixteen price levels are tracked per ticker. Each one represents a zone where algorithmic participants have historical interest — either as buyers, sellers, or stop hunters.

Field	Label	What It Represents	Why It Matters
pdh	PDH	Previous day's high	Algos defended this overnight; breakout or rejection is the signal
pdl	PDL	Previous day's low	Proven demand zone; breakdown or bounce is the signal
pdc	PDC	Previous day's close	Overnight settlement price; VWAP magnet, convergence target
pmh	PMH	Pre-market high	Smart money positioned here before open; often first resistance
pml	PML	Pre-market low	Pre-market support; often tested in first hour
orh	ORH	Opening range high (first 30 min)	Defines the "battlefield" for the day; breakout = trend day
orl	ORL	Opening range low (first 30 min)	Mirror of ORH; breakdown = bearish trend day
vwap	VWAP	Volume-weighted average price (live)	Institutional fair value. Above = bullish; below = bearish
poc	POC	Point of control (highest volume price)	Price magnet; tends to be revisited during the session
weeklyHigh	Wk Hi	Prior week's high	Macro structure ceiling
weeklyLow	Wk Lo	Prior week's low	Macro structure floor
fib236	0.236	Fibonacci retracement level	Shallow pullback entry zone
fib382	0.382	Fibonacci retracement level	First major pullback entry zone
fib500	0.500	Fibonacci retracement level	Mid-point of prior range; often acts as pivot
fib618	0.618	Fibonacci retracement level	Golden ratio; strongest pullback entry zone
fib786	0.786	Fibonacci retracement level	Deep pullback; last line before trend invalidation
Fibonacci levels are calculated from the PDH→PDL range:

range       = PDH − PDL
fib236      = PDH − (range × 0.236)
fib382      = PDH − (range × 0.382)
fib500      = PDH − (range × 0.500)
fib618      = PDH − (range × 0.618)
fib786      = PDH − (range × 0.786)

Proximity detection: Any level within ±0.4% of the current price is flagged as "in play." Only in-play levels can trigger patterns. This prevents the system from claiming a PDH Reclaim when price is 2% below PDH.

3.2 Options Greeks
Field	Description	Trading Relevance
delta	Rate of option price change per $1 move in underlying	Used as a rough ITM probability proxy (~50Δ = near ATM)
theta	Daily premium decay rate (negative = you lose this daily)	The core cost of holding a long option position
gamma	Rate of delta change per $1 move in underlying	Accelerates near expiry; the source of 0DTE explosiveness AND danger
vega	Option price change per 1% change in IV	IV crush exposure; higher vega = more hurt when IV falls
iv	Implied volatility as a percentage	Market's priced-in expected move; high IV = overpriced premium
hv	Historical (realized) volatility	Actual recent moves; comparison against IV reveals over/underpricing
expectedMove	±$ move implied by options pricing	Realistic target cap — don't target beyond the market's expected range
maxPain	Strike price where market makers lose the least	Price gravitates here into expiry (Friday phenomenon on weekly options)
premium	ATM option price per share	Cost basis for sizing calculations
bidAsk	Width of bid-ask spread	Direct transaction cost; wide spread = money lost on every entry/exit
3.3 Multi-Timeframe Analysis
Three analysis timeframes feed the scoring engine. Each has its own score (0–100) and provides different data:

15-Minute (structural bias):

VWAP relation: above or below
EMA 9/21/50 alignment: bullish / bearish / mixed — confirming indicator [LAG]
Market structure: higher highs/lower lows, ranging — leading indicator [LEADING]
RSI 15: momentum level — lagging indicator [LAG]
5-Minute (setup quality):

RSI level — lagging [LAG]
Bollinger Band state: at upper band, at lower band, mid-band — confirming [LAG]
MACD signal: bullish/bearish crossover — lagging [LAG]
Nearest price level: which key level is price closest to — leading [LEADING]
Divergence: hidden bullish/bearish (continuation), regular bullish/bearish (reversal) — leading [LEADING]
1-Minute (trigger readiness):

Candle pattern: engulfing, pin bar, doji, inside bar — leading [LEADING]
Volume state: surge above average, below average — leading [LEADING]
Break of Structure (BoS): confirmed or not — leading [LEADING]
ATR: average true range, used for stop sizing
Critical design note on indicators: Leading indicators (price action, volume, BoS, market structure, divergence) carry more weight in the scoring engine than lagging indicators (RSI, MACD, EMA). The UI explicitly labels each indicator as [LEADING] or [LAG] so the trader never double-counts confirmation from lagging signals.

RSI, MACD, and EMA are all momentum indicators that reflect the same underlying data (price history). Having all three agree does not provide three independent sources of evidence — it provides one. This redundancy is intentional as a confirmation tool, but it is explicitly labeled so the trader understands they are not seeing three separate signals.

3.4 Gap Data
Field	Description
pct	Gap size as a percentage of prior close
type	up / down / flat
Gaps above 1% are noted as "likely to fill" — a reference to the statistical tendency for gap-fill moves during the first 2 hours of trading.

4. Layer 1 — Price Level Classification
Before any signal is generated, the system maps the current price to its nearest key levels and computes the percentage distance to each:

distance_to_level = (current_price − level_price) / level_price × 100

A level is "in play" if |distance_to_level| ≤ 0.4%.

The level map is displayed visually in the Key Levels panel, sorted from highest to lowest price, with color coding:

Yellow: PDH, PDL (primary session levels)
Purple: PMH, PML (pre-market levels)
Orange: ORH, ORL (opening range)
Cyan: VWAP (dynamic fair value)
Violet: POC (volume-weighted anchor)
Blue: Fibonacci retracements
Indigo: Weekly High/Low (macro structure)
5. Layer 2 — Pattern Recognition
5.1 The Thirteen Patterns
The system recognizes 13 named price patterns. Each pattern maps to a signal direction (CALL/PUT/STAY OUT) and carries a baseline historical win rate based on observed 0DTE performance across normal market conditions. These win rates are regime-dependent — see Layer 4 and Section 11 for how they are adjusted in real time.

Pattern	Signal	Base Win Rate	Setup Category
PDH Reclaim	CALL	82%	Breakout continuation
PDH Rejection	PUT	78%	Resistance rejection
PDL Bounce	CALL	76%	Support bounce
PDL Breakdown	PUT	80%	Support failure
ORB Long	CALL	79%	Opening range breakout
ORB Short	PUT	76%	Opening range breakdown
VWAP Reclaim	CALL	72%	Mean reversion long
VWAP Rejection	PUT	70%	Mean reversion short
PMH Resistance	PUT	74%	Pre-market level rejection
PML Support	CALL	72%	Pre-market level bounce
Gap Fill Long	CALL	74%	Gap fill bullish
Gap Fill Short	PUT	71%	Gap fill bearish
Inside Day — No Edge	STAY OUT	28%	No directional edge
Important: These are historical averages across all regimes. In a CHOPPY regime, the effective win rate for breakout patterns (PDH Reclaim, ORB Long, PDL Breakdown) drops to 49–60%. Never use the base rate as your expected probability in the current session. Use the regime-adjusted win rate (Layer 4 + Section 11).

5.2 Pattern Entry Logic
Breakout patterns (PDH Reclaim, ORB Long, ORB Short, PDL Breakdown):

Require a confirmed candle CLOSE beyond the level — not just a wick
Volume must be above the 20-period average on the break candle
1-minute Break of Structure (BoS) confirmation required
Entry: first 5-minute candle close beyond the level, or 1-minute pullback retest
Rejection patterns (PDH Rejection, VWAP Rejection, PMH Resistance):

Require a bearish reversal candle at the level (engulfing, pin bar, shooting star)
RSI divergence on 5-minute preferred
Confirmation: close back below the level
Mean reversion patterns (VWAP Reclaim, PDL Bounce, PML Support):

Price must have been below the level, then show a strong reclaim candle
Volume surge on the reclaim candle
At least 1 of 3 trend timeframes not aggressively counter-trend
STAY OUT (Inside Day — No Edge):

Price is contained within prior day's PDH and PDL
No directional edge — waiting for a break of either extreme
System still shows the ticker but locks the direction to STAY OUT
6. Layer 3 — Multi-Timeframe Alignment
6.1 Alignment Score
Three trend timeframes are assessed for directional agreement with the signal:

alignment_score = count of (Daily, 4H, 1H) that match signal direction
For CALL signals: match = "Bullish"
For PUT signals:  match = "Bearish"
Neutral counts as non-matching.

6.2 Alignment Multiplier Table
Score	Meaning	Multiplier Applied to Raw Score
3/3	Full alignment — all timeframes agree	× 1.00
2/3	Partial alignment — one timeframe neutral or opposing	× 0.92
1/3	Weak alignment — mostly counter-trend	× 0.78
0/3	Fully counter-trend — all three timeframes oppose the signal	× 0.55
A fully counter-trend signal loses 45% of its base score before any other calculation runs. These signals still appear in the watchlist (as Grade C or D) — the system never hides a signal, but it makes the quality degradation visible and explicit. A trader taking a 0/3 alignment trade understands they are fighting the broader trend.

7. Layer 4 — Session Phase & Volatility Regime
7.1 Session Phase Engine
The trading day is divided into six windows, each with fundamentally different characteristics. Time is calculated in minutes since midnight (ET timezone):

Phase Name	ET Window	Minutes (midnight)	Tradeable	Quality Multiplier	Rationale
Pre-Market	Before 9:30 AM	< 570	No	0.40	Thin liquidity; levels unreliable
Opening Danger Zone	9:30–9:45 AM	570–584	No	0.50	HFT whipsaw; false signal factory
Prime Time	9:45–11:30 AM	585–689	Yes	1.00	Maximum institutional order flow; cleanest setups
Lunch Chop	11:30 AM–1:00 PM	690–779	No	0.55	Algos go offline; wide spreads; low conviction
Afternoon	1:00–3:00 PM	780–899	Yes	0.85	Directional but reduced momentum; confirmed moves only
Power Hour	3:00–4:00 PM	900–959	Yes	0.70	Extreme 0DTE gamma acceleration; reduce size 50%
After Hours	After 4:00 PM	≥ 960	No	0.00	Market closed; all grades show D
The quality multiplier directly multiplies the raw confluence score. The same perfect setup scores 45% lower at lunch than it does during prime time. This is intentional — the setup quality is meaningless if the market environment is hostile to that type of move.

7.2 Hourly Theta Acceleration
Daily theta is not experienced evenly across the session. 0DTE options decay slowly in the morning and accelerate violently toward close. The system displays $/hr of theta decay, not just the flat daily rate:

Session	Fraction of Daily Theta Burned
Pre-Market	1%
Opening Danger	5%
Prime Time	9%
Lunch Chop	9%
Afternoon	14%
Power Hour	38% — more than a third of daily theta in 60 minutes
After Hours	0%
Formula:

hourlyTheta = dailyTheta × sessionFraction
displayed as: $|hourlyTheta × 100| per contract per hour

This tells a trader that a $0.45/day theta on a TSLA put becomes $0.17/hr during power hour — the position is losing value extremely quickly simply from time passing.

7.3 Volatility Regime Detection
The system classifies the day into one of four volatility regimes using three signals: VIX level, the proportion of STAY OUT signals in the watchlist, and the average gap size with directional concentration.

Detection logic (in priority order):

VIX > 26 → HIGH VOL (regardless of anything else)
3 or more STAY OUT signals in watchlist → CHOPPY
Average gap % > 0.4% AND 3+ same-direction signals → TREND DAY
Default → RANGE DAY
Regime behavior table:

Regime	Best Patterns	Position Sizing	Key Risk
TREND DAY	ORB Long/Short, PDH Reclaim, PDL Breakdown	Full size	Missing the move by waiting too long
RANGE DAY	PDH Rejection, PDL Bounce, VWAP Reclaim/Rejection	Normal	False breakouts triggering on PDH/PDL that fail
CHOPPY	A+ only — PDH/PDL extremes only	Reduce 50%	Nearly all signals are false positives
HIGH VOL	Spreads only — no naked debit	Spreads only	Gamma explosion; IV crush post-event
8. Layer 5 — Greeks & Options Intelligence
This layer translates raw options data into actionable intelligence. It answers: Should you buy a spread or naked? What DTE? What stop? What is your actual edge after volatility pricing?

8.1 IV Penalty for Debit Buyers
High implied volatility means you are overpaying for premium. Even if the directional call is correct, IV crush can eliminate your profit or turn a winning direction into a losing trade. The IV penalty reduces the score before it is displayed:

IV Level	Penalty Multiplier	Interpretation
IV > 70%	× 0.78	Severely overpriced — 22% confidence reduction
IV 56–70%	× 0.88	Moderately overpriced — significant vega risk
IV 41–55%	× 0.95	Slightly elevated — modest penalty
IV ≤ 40%	× 1.00	Fair or cheap premium — no penalty
8.2 IV Rank vs. IV Level
The system uses IV Rank (IVR) rather than raw IV level. IVR contextualizes current IV against the past 52 weeks:

IVR = (current IV − 52-week IV low) / (52-week IV high − 52-week IV low) × 100

An IVR of 85 means current IV is in the 85th percentile of the past year — very elevated. An IVR of 15 means options are cheap by recent standards.

8.3 DTE Optimization Engine
Which expiration to use is not fixed at 0DTE. The system recommends based on IVR and session phase:

Condition	Recommended DTE	Risk Level	Reason
Power Hour (after 3 PM)	1–2 DTE	DANGER	0DTE exponential theta kill after 3 PM
IVR > 65	1–2 DTE	WARN	Severely overpriced premium; 1-2 DTE cuts vega exposure ~40%
Afternoon + IVR > 40	0–1 DTE	WARN	Mid-session elevated IV; conditional
Prime Time, normal IV	0 DTE	OK	Optimal conditions for 0DTE leverage
All other conditions	0–1 DTE	OK	Standard; use 1 DTE if target > 1 ATR away
8.4 IV-Adjusted Stop Loss
A fixed -20% stop is wrong. High IV means options move faster per point in the underlying. Using a fixed percentage stop when IV is high means your stop is too wide in dollar terms; when IV is low, it may be too tight. The system adjusts:

IVR Level	Stop %	Dollar Impact
IVR > 65	-15%	High-IV options move violently; tight % protects dollar risk
IVR 41–65	-20%	Standard environment
IVR ≤ 40	-25%	Cheap premium; wider % = similar dollar risk per contract
8.5 Spread vs. Naked Recommendation
When premium is overpriced (IVR ≥ 45), buying naked options exposes the trader to IV crush even on a correct directional call. The system recommends a debit spread:

Spread trigger: IVR ≥ 45
Spread width:   IVR > 65 → 2-wide  |  IVR 45–65 → 1-wide
CALL signal → Bull Call Spread (buy ATM, sell N-wide OTM call)
PUT signal  → Bear Put Spread  (buy ATM, sell N-wide OTM put)
Max profit = (width − debit paid) × 100
Max loss   = debit paid × 100
Vega reduction vs. naked: ~60%

At IVR ≥ 45, the system shows only spread structures — never naked debit. Naked is shown only when IVR < 45.

8.6 Bid-Ask Liquidity Penalty
The bid-ask spread is a direct transaction cost paid twice (entry and exit). Wide spreads are penalized in the confluence score and flagged with a warning tier:

Bid-Ask as % of Premium	Penalty	Warning Level
> 10%	× 0.88 on score	DANGER
6–10%	× 0.94 on score	WARN
< 6%	× 1.00	OK
Round-trip cost displayed: bid-ask spread × 100 = dollar cost per contract for entry + exit combined.

8.7 Skew Awareness
Options markets are not symmetrically priced. The system flags two common skew distortions that silently inflate the cost of entry:

Index put skew (SPY, QQQ): Put options on major indices cost 3–5 extra implied volatility points compared to equivalent calls. This is because institutional portfolio managers perpetually buy index puts for protection, creating structural demand. The skew is not directional alpha — it is a hidden cost that makes SPY/QQQ puts more expensive than they appear.

High-beta call skew (TSLA, NVDA): Retail lottery demand inflates upside call premium on high-momentum names. A TSLA 5% OTM call trades at a higher implied volatility than the equivalent put — the same distance, the other direction. Comparing to the same-delta put reveals the actual skew cost.

8.8 Dollar Gamma
Raw gamma (e.g., 0.045) is dimensionless and often misread. Dollar gamma translates it into how much your position's effective delta changes per $1 move in the underlying, per contract:

dollarGamma = gamma × 100   (100 shares per contract)

Example: TSLA gamma 0.065 → dollar gamma $6.50. A $1 move in TSLA changes your effective delta by $6.50 per contract. At $175, a 1% move ($1.75) changes delta by $11.38. Near expiry, this compounds rapidly — see Section 11.4 on gamma convexity acceleration.

9. Layer 6 — Confluence Scoring Engine
This is the mathematical core of the system. It synthesizes all upstream inputs into a single 0–100 score, then maps that score to a letter grade.

9.1 The Algorithm
Input prerequisites:

Pattern base win rate (from Layer 2)
Alignment multiplier (from Layer 3)
Session quality multiplier (from Layer 4)
IV penalty (from Layer 5)
Liquidity penalty (from Layer 5)
Event penalty (from Layer 7)
Leading indicator bonus points (divergence, volume, BoS)
Formula:

raw_score = (baseWinRate
             × alignmentMultiplier
             × sessionQualityMultiplier
             × ivPenalty
             × liquidityPenalty
             × eventPenalty)
            + divergenceBonus
            + volumeBonus
            + bosBonus
final_score = clamp(round(raw_score), minimum=8, maximum=97)

Component breakdown:

Component	Range	Source
baseWinRate	28–82	Pattern lookup table
alignmentMultiplier	0.55–1.00	0/1/2/3 timeframe alignment
sessionQualityMultiplier	0.00–1.00	Session phase
ivPenalty	0.78–1.00	Current IV level
liquidityPenalty	0.88–1.00	Bid-ask % of premium
eventPenalty	0.50–1.00	Event calendar impact
divergenceBonus	+3 or +6	Regular +3, Hidden +6
volumeBonus	+5	Volume surge on trigger candle
bosBonus	+4	Break of Structure confirmed
STAY OUT exception: Signals that are STAY OUT do not go through the full multiplier chain. Their score is capped at min(35, analysis15.score / 2).

9.2 Grade Thresholds
Score	Grade	Interpretation
88–97	A+	Trade with full position size
78–87	A	Trade with full position size
68–77	B	Trade with 50–75% position size
55–67	C	Consider skipping; only in TREND DAY
8–54	D	Do not trade
9.3 What the Score Is and Is Not
The confluence score is a relative ranking tool, not a calibrated probability.

A score of 91 does not mean a 91% chance of success. It means this setup ranks near the top of the quality distribution given the current conditions. A score of 91 in a CHOPPY regime is a worse trade than a score of 78 in a TREND DAY because the underlying patterns are degraded by regime. Always read the score alongside the regime-adjusted win rate (Section 11.1), not in isolation.

9.4 Worked Example — TSLA PDL Breakdown
Conditions: Prime Time, IV Rank 85, bid-ask 8.8%, full bearish alignment, FOMC day

Base win rate (PDL Breakdown):          80
× alignmentMultiplier (3/3 bearish):   × 1.00
× sessionQualityMult (prime-time):     × 1.00
× ivPenalty (IV = 72):                 × 0.78
× liquidityPenalty (bid-ask 8.8%):     × 0.88
× eventPenalty (FOMC day):             × 0.85
= subtotal:                             46.8
+ bosBonus (BoS confirmed):            + 4
+ volumeBonus (surge):                 + 5
+ divergenceBonus (Hidden Bearish):    + 6
= RAW SCORE:                           61.8 → Grade C
Regime-adjusted win rate (RANGE DAY): 80 × 0.82 = 66%
Break-even win rate (35% target, 15% stop, 17.6% round-trip cost): 61.5%
Net edge: 66% − 61.5% = +4.5% (thin — limit orders critical)
DTE recommendation: 1–2 DTE (IVR 85 → overpriced)
Structure recommendation: Bear Put Spread (IVR ≥ 45)

The high IV, wide spread, and FOMC penalty reduce what would be a strong directional setup to a Grade C. The spread recommendation appears because IVR > 45. The edge is real (+4.5%) but thin — one bad fill destroys it.

10. Layer 7 — Risk Filters & Portfolio Controls
10.1 Event Calendar System
Market events are classified by type and severity. Events are not optional — they are the highest-priority warnings in the system and can override an A+ signal.

Event types: FOMC, CPI, Earnings (ticker-specific), OPEX (options expiration), NFP (Non-Farm Payrolls), Quad Witch (quarterly expiration)

Each event carries:

Timing (e.g., "2:00 PM ET")
IV impact classification: "CRUSHES IV" / "EXPANDS IV" / "MIXED"
Explicit trader instruction (not just a warning badge)
Severity: HIGH or MEDIUM
Event impact on scoring:

Ticker-specific HIGH event (e.g., NVDA earnings day): eventPenalty = 0.50
Market-wide HIGH event (e.g., FOMC):                  eventPenalty = 0.85
MEDIUM event:                                          eventPenalty = 0.92
No event:                                              eventPenalty = 1.00

Events appear in three places simultaneously:

Top bar — flashing pill showing count and types
Full-width red/amber warning banners below navigation
Portfolio Risk Monitor — event calendar section with per-event IV impact guidance
The key principle: An earnings event on NVDA does not just add a warning badge — it actively degrades the NVDA signal score by 50% and shows an explicit instruction (e.g., "This event OVERRIDES any NVDA CALL signal — IV will crush 40–60% at close.").

10.2 Portfolio Greeks Aggregation
The system aggregates Greeks across all active signals to show the portfolio-level exposure, not just individual position risk:

For each active ticker (signal ≠ STAY OUT):
  direction = CALL → +1, PUT → −1
netDelta = Σ (delta × direction)
netTheta = Σ theta           (all negative — all long options decay)
netVega  = Σ vega            (all positive — all long options expand with IV)
netGamma = Σ gamma           (all positive — all long options benefit from realized vol)

Warning thresholds — when these trigger, the cell turns amber:

Metric	Threshold	Risk Being Flagged
netDelta	
netTheta × 100	> $150/day	Excessive daily time decay cost across all positions
netVega	> 1.5	Heavy IV crush exposure if volatility collapses post-event
netGamma	> 0.15	Explosive combined gamma; position delta unstable near expiry
10.3 Correlation Warning
Holding three or more same-direction signals triggers a correlation warning:

correlatedCalls = count of tickers with signal = CALL
correlatedPuts  = count of tickers with signal = PUT
Warning triggers at: correlatedCalls ≥ 3  OR  correlatedPuts ≥ 3

Why this matters: SPY CALL + QQQ CALL + NVDA CALL is not three independent trades. All three are long beta, all three go down together if the market sells off. The system treats this as one macro bet expressed in three instruments — which is exactly what it is.

10.4 Daily Loss Limit
The system tracks intraday losses as a percentage of account and applies progressive warnings:

dailyBudgetPct = (dailyLosses / dailyLossLimit) × 100
< 50% used  → Green  (normal trading)
50–75% used → Amber  (slow down; be selective)
> 75% used  → Red    (consider stopping)
≥ 100% used → Blinking DAILY LIMIT HIT alert in top bar

The daily loss limit is user-configurable (1%, 2%, 3%, or 5% of account). The recommended professional standard is 2–3%.

11. Edge Reality System (Institutional Audit Additions)
The following three systems were added after an institutional quant/derivatives trading audit of the original architecture. They address the most important failure modes identified: regime-dependent win rates being misread as static probabilities, transaction costs eliminating edge in non-ideal conditions, gamma explosion risk near close, and correlated position blow-up.

11.1 Regime-Conditioned Win Rates
Historical win rates (82% PDH Reclaim, 80% PDL Breakdown, etc.) are derived from all market conditions averaged together. In practice, these patterns perform vastly differently by regime.

The key insight: If you take a PDH Reclaim in a CHOPPY regime, you are not trading a 82% setup. You are trading a 49% setup that looks like an 82% setup.

Regime win rate multipliers:

Regime	Multiplier	Rationale
TREND DAY	× 1.00	Breakout patterns at full strength — institutions are buying the breakout
RANGE DAY	× 0.82	False breakouts common; mean-reversion works but momentum fails
CHOPPY	× 0.60	40% edge erosion — structural noise dominates signal
HIGH VOL	× 0.72	IV overpricing erodes debit buyer edge; direction correct but premium wrong
Calculation:

regimeAdjustedWinRate = round(baseWinRate × regimeMultiplier)
Examples:
PDH Reclaim in TREND DAY:  82 × 1.00 = 82%
PDH Reclaim in RANGE DAY:  82 × 0.82 = 67%
PDH Reclaim in CHOPPY:     82 × 0.60 = 49%
PDH Reclaim in HIGH VOL:   82 × 0.72 = 59%

The dashboard displays both the historical (all-regime) win rate grayed out as reference, and the current regime-adjusted win rate in bold — the number you actually use to evaluate the trade.

11.2 Break-Even Win Rate Calculator
Every trade has a minimum win rate below which you lose money even with correct directional calls — because transaction costs eat the profit on small wins.

Formula (derived from expectancy equation):

netWin  = targetPct − roundTripCostPct
netLoss = stopPct   + roundTripCostPct
breakevenWinRate = netLoss / (netWin + netLoss)

Where roundTripCostPct = (bidAsk / premium) × 200 (both legs of the round trip as a percentage of premium).

Example — TSLA put with 8.8% round-trip cost, 35% target, 15% stop:

netWin  = 35 − 17.6 = 17.4%
netLoss = 15 + 17.6 = 32.6%
breakeven = 32.6 / (17.4 + 32.6) = 65.2%

If your regime-adjusted win rate on this setup is 65% and break-even is 65.2%, you have no edge. The spread is consuming it entirely. The system flags this explicitly as "No edge in RANGE DAY after costs."

11.3 Net Edge Margin
The dashboard displays the gap between regime-adjusted win rate and break-even win rate as your net edge:

edgeMargin = regimeAdjustedWinRate − breakevenWinRate
> +5%:  Green  — meaningful edge exists; trade normally
0–5%:   Amber  — thin edge; one bad fill eliminates profitability; limit orders critical
≤ 0%:   Red    — no edge in this regime after costs; skip this trade

This single number answers the most important pre-trade question: Do I actually have an edge here, or am I just paying transaction costs to gamble?

11.4 Gamma Convexity Acceleration
Static gamma is a point-in-time estimate that becomes dangerously underestimated near 0DTE expiry. As time to expiry shrinks, gamma accelerates — options increasingly behave like binary bets rather than continuous functions.

Empirical gamma acceleration near close:

Time to Close	Gamma Multiplier	Meaning
> 120 minutes	× 1.0	Static gamma is approximately accurate
60–120 minutes	× 1.5	Effective gamma 50% higher than stated
30–60 minutes	× 2.5	Effective gamma 150% higher than stated
< 30 minutes	× 4.0	Effective gamma 300% higher — binary-like behavior
Displayed calculation:

effectiveGamma = staticGamma × gammaAccelMultiplier
deltaShiftPerAdverseMove = effectiveGamma × 100 × adverseMoveSize

Example: TSLA static gamma 0.065, 45 minutes to close (multiplier × 2.5):

Effective gamma: 0.065 × 2.5 = 0.163
A $2.04 adverse move (30% of $6.80 expected move): 0.163 × 100 × $2.04 = $33.25 delta shift per contract
This means a position that appeared to have a $6.50 delta shift per $1 move now has a $16.30 delta shift per $1 move. The position is behaving like a much larger directional bet than the trader sized it for. This is the "0DTE gamma bomb" that destroys accounts during power hour.

The warning activates automatically during the Afternoon and Power Hour sessions and displays the effective gamma alongside the multiplier and the concrete dollar delta shift for a representative adverse move.

11.5 Worst-Case Correlated Blow-Up
The portfolio tail risk stress test answers: If everything goes wrong at exactly the same time, how much do I lose?

This scenario — FOMC surprise, flash crash, massive gap-open event — causes all correlated positions to hit their stops simultaneously. Individual position sizing may be correct (1% risk each), but combined they represent a single correlated event.

Calculation:

For each active position:
  riskPerContract = premium × 100 × (ivAdjustedStop% / 100)
  maxContracts    = floor((accountSize × riskPct%) / riskPerContract)
  positionLoss    = maxContracts × riskPerContract
worstCaseStress = Σ positionLoss across all active tickers

Displayed as:

Total dollar loss in the simultaneous stop-out scenario
As a percentage of total account
With a warning if it exceeds 5% of account in a single event
Example: 4 active signals, each sized at 1% risk on a $25,000 account:

Theoretical individual risk: 4 × $250 = $1,000 (4% of account)
But if all positions stop simultaneously on the same macro event, the actual worst case may be higher due to slippage and widened spreads in a fast market
The stress figure is displayed in the Portfolio Risk Monitor so the trader can decide whether to reduce the number of concurrent positions before a risk event
12. Position Sizing Engine
Position sizing is calculated from three user inputs: account size, maximum risk percentage per trade, and the IV-adjusted stop loss.

Formula:

maxRiskDollars   = accountSize × (riskPct / 100)
premPerContract  = premium × 100
riskPerContract  = premPerContract × (ivAdjustedStopPct / 100)
maxContracts     = floor(maxRiskDollars / riskPerContract)
totalPosition    = maxContracts × premPerContract

Example — SPY CALL with $25,000 account, 1% risk, IVR 24 (stop -25%):

maxRiskDollars  = $25,000 × 0.01 = $250
premPerContract = $2.10 × 100   = $210
riskPerContract = $210 × 0.25   = $52.50
maxContracts    = floor($250 / $52.50) = 4 contracts
totalPosition   = 4 × $210 = $840 deployed

Example — TSLA PUT with $25,000 account, 1% risk, IVR 85 (stop -15%):

maxRiskDollars  = $250
premPerContract = $3.50 × 100 = $350
riskPerContract = $350 × 0.15 = $52.50
maxContracts    = floor($250 / $52.50) = 4 contracts
totalPosition   = 4 × $350 = $1,400 deployed

Note: both examples produce the same $250 dollar risk because the IV-adjusted stop calibrates the percentage to achieve consistent dollar exposure regardless of IV regime.

The system also displays the round-trip transaction cost per contract alongside the sizing output, so the trader can see the full cost picture before committing.

13. Signal Output — Final Presentation
After all seven layers and the edge reality system, the final signal is presented in a four-column card:

Column	Contents
Col 1: Active Signal	Signal direction (CALL/PUT/STAY OUT), pattern name, base win rate, confluence score + progress bar, score formula breakdown, Edge Reality Check (regime WR, historical WR, break-even WR, net edge margin), false precision disclaimer
Col 2: Greeks	Delta, theta (with hourly rate), gamma (with dollar gamma), vega, IV vs HV comparison, expected move, max pain, gamma convexity spike warning (Afternoon/Power Hour only)
Col 3: Structure & Costs	Spread vs. naked recommendation with exact strikes, DTE recommendation, IV-adjusted stop %, bid-ask round-trip cost, skew warning
Col 4: Entry Checklist	5-step execution protocol, IV-adjusted stop, position size (contracts), total position value, per-contract transaction cost
Entry Checklist (5 steps):

Wait for 1-minute candle to CLOSE above/below the trigger level — never enter on a wick
Place a LIMIT order at mid-price (bid + ask) ÷ 2
Never use market orders — you pay the full bid-ask immediately
Cancel if unfilled within 30 seconds
Invalidate the setup if price moves more than 0.5% past the trigger before fill
Fill realism note: For tickers with bid-ask spreads above 8% (TSLA, AMZN, AAPL in active vol), mid-price fills on 0DTE options are not always achievable in fast-moving conditions. In power hour or during news events, plan for fills closer to the ask on entry and closer to the bid on exit — effectively paying the full spread on both legs rather than half.

14. Signal Log & Net P&L Accounting
The Signal Log page tracks all historical trades and produces performance metrics on both a gross and net basis. The distinction between the two is the most important number for evaluating a strategy's real-world viability.

14.1 Round-Trip Transaction Cost Model
Every trade pays the bid-ask spread twice — once to enter, once to exit. The system models this per ticker based on realistic spread conditions:

Ticker	Round-Trip Cost %	Dollar Cost Example
SPY	4.8%	$0.05 × 2 = $10 on $210 premium
QQQ	5.9%	$0.07 × 2 = $14 on $240 premium
TSLA	8.8%	$0.22 × 2 = $44 on $350 premium
NVDA	7.2%	$0.28 × 2 = $56 on $820 premium
AAPL	9.5%	$0.12 × 2 = $24 on $145 premium
AMZN	10.3%	$0.10 × 2 = $20 on $130 premium
Net P&L formula:

netPl = grossPl − roundTripCostPct

This applies on both wins and losses. A +35% gross win on TSLA yields a +26.2% net win. A -20% gross loss on TSLA is a -28.8% net loss. The spread is paid regardless of outcome.

14.2 Performance Metrics (Gross vs. Net)
The log calculates and displays all metrics on both a gross and net basis side-by-side:

Metric	Definition
Win Rate (gross)	% of trades where gross P&L > 0
Win Rate (net)	% of trades where netPl > 0 (may be lower — some gross winners become net losers)
Profit Factor (gross)	Σ(grossWins) / Σ(
Profit Factor (net)	Σ(netWins) / Σ(
Avg Win	Average net P&L on winning trades
Avg Loss	Average net P&L on losing trades
MAE	Max Adverse Excursion — how far against you the trade went before winning
MFE	Max Favorable Excursion — peak unrealized gain on a position
Hold Time	Minutes from entry to exit
14.3 Equity Curve
The chart displays two lines simultaneously:

Dashed line: Gross cumulative P&L (what the strategy claims)
Solid line: Net cumulative P&L (what you actually keep)
The gap between the two lines at any point equals the total transaction costs paid to that point in the sample. This gap is the strategy tax — and for strategies with many trades or wide spreads, it is often the difference between a profitable system and a losing one.

14.4 Exit Reason Analysis
Each trade logs its exit reason:

Target hit: Reached the defined profit target (35–50% of premium)
Stop hit: Reached the IV-adjusted stop loss
Time exit: Closed before expiry for time/theta reasons
This breakdown reveals whether losses are coming from the stop being too tight, targets being too ambitious, or time decay eating positions held too long.

15. End-to-End Flow — Step by Step
This is the complete operational sequence from market open to trade logged:

Step 1 — Pre-Market (Before 9:30 AM ET)

Session phase = "pre-market" → qualityMult = 0.40
All signals displayed but scored at 40% — no trades
Trader reviews key levels, identifies watchlist setups for the day
Event calendar reviewed — any FOMC, CPI, earnings?
Step 2 — Market Open (9:30–9:45 AM ET)

Session phase = "opening-danger" → qualityMult = 0.50
Banner: "DO NOT TRADE — HFT whipsaw in first 15 minutes"
Opening range begins forming (ORH and ORL being established)
Step 3 — Prime Time Begins (9:45 AM)

Session phase = "prime-time" → qualityMult = 1.00
Full scoring enabled
Regime detection runs: VIX + watchlist signals + average gap → TREND DAY / RANGE DAY / CHOPPY / HIGH VOL
Watchlist updates every 3 seconds with live score recalculations
Step 4 — Pattern Detection (continuous, per ticker)

Price moves relative to key levels
If price approaches within ±0.4% of a key level → level flagged as "in play"
Pattern classifier checks for matching candle behavior, volume, BoS
Signal direction assigned (CALL/PUT/STAY OUT)
VWAP level updates with live volume
Step 5 — Trader Selects a Ticker

Layer 1: Distance to all 16 price levels calculated and displayed
Layer 2: Pattern name and rationale shown
Layer 3: Alignment score computed (0–3) with multiplier
Layer 4: Session quality and regime multiplier applied
Layer 5: Greeks panel populated:
DTE recommendation computed
Spread vs. naked decision made
Skew warning checked
Dollar gamma computed
IV-adjusted stop set
Layer 6: computeStrength() → score + grade
Layer 7: All risk filters applied:
Event calendar → override/penalty if applicable
Portfolio Greeks aggregated
Correlation count checked
Daily loss budget checked
Edge Reality System:
Regime-adjusted win rate displayed
Break-even win rate calculated
Net edge margin shown (green/amber/red)
Step 6 — Trade Decision

Trader reads the signal card:
Grade A/A+: Trade with full size if edge margin is green
Grade B: Trade with 50–75% size; require edge margin ≥ +5%
Grade C: Skip unless TREND DAY with green edge margin
Grade D: Skip
Trader follows the entry checklist (5 steps)
Position sized from the position sizing panel
Step 7 — Active Position

Trader monitors with the dashboard
Power Hour entry: gamma convexity spike warning activates
If a second signal fires in the same direction: correlation check activates
Theta clock ticking — hourly theta display shows dollar/hr decay rate
Step 8 — Trade Exit

Target hit: close at limit 5–10 cents below the ask
Stop hit: close immediately, market order acceptable in liquid names
Time exit: any 0DTE position should be closed by 3:45 PM at the latest
Step 9 — Signal Log Entry

Gross P&L recorded (%)
Net P&L calculated (gross minus round-trip cost)
Exit reason, MAE, MFE, hold time logged
Equity curve (gross vs. net) updated
Running performance metrics recalculated
16. Worked Examples
Example A — High Quality Setup (Grade A+)
Ticker: QQQ
Time: 10:08 AM (Prime Time)
Regime: TREND DAY
Pattern: ORB Long — ORH @ $444.60 broken with volume surge

Layer 1: ORH in play at $444.60 (price at $445.12 = +0.12% above)
Layer 2: ORB Long → base win rate 79%
Layer 3: Daily Bullish, 4H Bullish, 1H Bullish → 3/3 alignment → × 1.00
Layer 4: Prime Time → × 1.00 | TREND DAY → mult 1.00
Layer 5: IV 22, IVR 32 → no iv penalty | bid-ask 2.9% → no liquidity penalty
         DTE: 0 DTE (prime-time + low IV) | Structure: Naked debit (IVR < 45)
Layer 6: 79 × 1.00 × 1.00 × 1.00 × 1.00 × 1.00 = 79 base
         + volBonus (surge) +5
         + bosBonus (BoS confirmed) +4
         = 88 → Grade A+
Edge Reality:
  Regime WR: 79 × 1.00 = 79%
  Round-trip cost: (0.07 / 2.40) × 200 = 5.8%
  Break-even WR: (20 + 5.8) / (29.2 + 25.8) = 46.9%
  Net edge: 79 − 46.9 = +32.1% (strong green)
Position (1% risk, $25,000 account):
  Premium $2.40/shr → $240/contract
  Stop -25% (IVR 32) → $60 risk/contract
  Max contracts: floor(250 / 60) = 4 contracts
  Total position: $960

Verdict: Trade. Grade A+, full edge, prime-time, trend day, full alignment.

Example B — Marginal Setup (Grade C)
Ticker: TSLA
Time: 1:45 PM (Afternoon)
Regime: RANGE DAY
Pattern: PDL Breakdown — PDL $176.40 broken

Layer 2: PDL Breakdown → base win rate 80%
Layer 3: 3/3 bearish alignment → × 1.00
Layer 4: Afternoon → × 0.85 | RANGE DAY → regime mult 0.82
Layer 5: IV 72, IVR 85 → ivPenalty × 0.78 | bid-ask 8.8% → liquidityPenalty × 0.88
         eventPenalty: FOMC day → × 0.85
Layer 6: 80 × 1.00 × 0.85 × 0.78 × 0.88 × 0.85 = 35.1 base
         + bosBonus +4 + volBonus +5 + divBonus (Hidden) +6
         = 50.1 → 50 → Grade D
Edge Reality:
  Regime WR: 80 × 0.82 = 66%
  Round-trip: (0.22 / 3.50) × 200 = 12.6%
  Break-even WR: (15 + 12.6) / (22.4 + 27.6) = 55.2%
  Net edge: 66 − 55.2 = +10.8%
  → Edge exists but Grade is D (session + event penalties too high)
Verdict: Skip. FOMC + afternoon + high IV + wide spread combine to make
this untradeable despite the directional edge and strong candle setup.
Structure recommendation: Bear Put Spread (IVR 85 → 2-wide)

17. Design Principles & Honest Limitations
17.1 Design Principles
1. Never hide why a signal is bad. Every Grade D signal remains visible in the watchlist with its score and the factors that degraded it. This teaches the trader what poor-quality conditions look like, rather than hiding them and creating false confidence in the remaining signals.

2. Leading indicators drive the edge; lagging indicators confirm. The [LEADING] and [LAG] labels in every analysis panel are structural design choices. RSI, MACD, and EMA are confirmation tools — not independent signal generators. Three lagging indicators agreeing is not three signals; it is one signal with repetition.

3. Net P&L, always. The log never shows only gross performance. Transaction costs are displayed per trade, per ticker, and summarized across the sample. The equity curve shows both the gross (theoretical) and net (actual) line simultaneously so the cost drag is always visible.

4. Event risk overrides everything. An FOMC day or earnings event degrades all affected signals automatically in the score formula. The system provides explicit instructions (not just badges) for what to do before each event, because "be careful" is not actionable guidance.

5. Risk scales with volatility. The IV-adjusted stop, DTE recommendation, and spread recommendation all adapt to the current IV environment. A fixed -20% stop applied uniformly is not a risk management system — it is a fixed rule applied to a variable market.

6. The score is a ranking, not a probability. A 91/100 score does not mean 91% probability of success. It means this setup ranks near the top of all possible setups given current conditions. The actual expected win rate in the current regime is shown separately and explicitly.

7. No silent fallbacks. When the market is closed, all scores go to zero and all grades show D. When a pattern has no edge (Inside Day), the score floor is 35. The system never invents optimism.

17.2 Honest Limitations
Win rates are historically-derived estimates, not guaranteed outcomes. The base win rates (72–82%) are derived from observed 0DTE behavior under "normal" conditions. They have not been walk-forward validated against a live trading sample. Regime-adjusted win rates are estimates, not calibrated probabilities. Trade the system for months before trusting any individual win rate number.

Patterns are behavioral, not mechanical. "PDH Reclaim" is a description of price behavior around a key level, not an algorithm that automatically produces a signal when conditions are met. The system assists identification; it does not replace chart reading.

The scoring formula is a weighted heuristic, not a machine learning model. The multipliers (0.78 for IV > 70, 0.55 for 0/3 alignment, etc.) are chosen based on trading experience and logic, not derived from a statistically rigorous regression on trade outcomes. They rank setups correctly more often than not, but they are not optimal in a mathematical sense.

Market microstructure is not modeled. The system does not have access to Level 2 order flow, dark pool prints, options market maker positioning, or tape reading data. Setups that look perfect on chart analysis can fail immediately due to an invisible wall of sell orders at a key level that the system cannot see.

Gamma convexity acceleration is an approximation. The multipliers (×1.5, ×2.5, ×4.0) are practical estimates of how gamma accelerates near expiry for near-ATM options. The actual rate depends on the exact moneyness and the shape of the volatility surface at that moment. Use these numbers as magnitude guidance, not precise values.

Fill realism in fast markets. The entry checklist instructs mid-price limit orders. In practice, for wide-spread tickers (TSLA, NVDA) during news-driven moves or power hour, mid-price fills may not be achievable. Budget for fills closer to the ask on entry and bid on exit during adverse conditions.

18. Implementation Specification
This section provides enough detail to rebuild the system from scratch.

18.1 Core State
The application maintains four categories of state:

Market state (global, updated every 3 seconds):

VIX level
Put/Call ratio
$TICK reading
Advance/Decline line
Ticker state (per ticker, updated every 3 seconds):

Current price (±0.35% random walk in demo mode; live feed in production)
Change percentage
All price levels (VWAP drifts ±0.06% each update)
Options Greeks
Gap data
Multi-timeframe analysis scores
Trend alignment
UI state (user-controlled):

Selected ticker
Account size
Risk percentage per trade
Daily loss limit
Computed state (derived from above):

Session phase (from current time)
Volatility regime (from VIX + ticker signals)
Confluence score per ticker (from all multipliers)
Grade per ticker (from score)
Portfolio Greeks aggregate
Regime-adjusted win rate per ticker
Break-even win rate per ticker
Net edge margin per ticker
Worst-case stress (from all active positions)
Gamma acceleration multiplier (from time to close)
18.2 Computation Order
Computations must run in this order because later steps depend on earlier results:

1. getSessionPhase(currentTime) → sessionPhase
2. detectRegime(vix, tickers) → regime
3. For each ticker:
   a. computeStrength(ticker, sessionPhase) → rawScore
   b. computeGrade(rawScore) → grade
4. computePortfolioGreeks(tickers) → portGreeks
5. For selected ticker:
   a. getBidAskWarning(bidAsk, premium) → bidAskWarn
   b. getSpreadRec(ticker) → spreadRec
   c. computeDTERec(ivRank, sessionPhase) → dteRec
   d. getSkewWarning(symbol, signal) → skewWarn
   e. computeIVStop(ivRank) → ivStop
   f. computeRegimeAdjustedWinRate(pattern, regime) → regimeWR
   g. computeBreakevenWR(target=35, stop=ivStop.pct, roundTrip) → breakevenWR
   h. edgeMargin = regimeWR − breakevenWR
   i. computeWorstCaseStress(tickers, riskPct, accountSize) → worstCaseStress
   j. gammaAccelMult from minsToClose → effectiveGamma
   k. Position sizing from maxRisk / riskPerContract → maxContracts

18.3 Key Formulas Reference
// Session phase (minutes since midnight, ET)
t < 570  → pre-market
t < 585  → opening-danger
t < 690  → prime-time
t < 780  → lunch-chop
t < 900  → afternoon
t < 960  → power-hour
else     → after-hours
// Confluence score
raw = (baseWinRate × alignMult × sessMult × ivPenalty × liquidityPenalty × eventPenalty)
      + divergenceBonus + volumeBonus + bosBonus
score = clamp(round(raw), 8, 97)
// Grade
score ≥ 88 → A+  |  ≥ 78 → A  |  ≥ 68 → B  |  ≥ 55 → C  |  < 55 → D
// Regime-adjusted win rate
regimeWR = round(PATTERN_WIN_RATE[pattern] × REGIME_WIN_RATE_MULT[regime])
// Break-even win rate
netWin  = max(0.1, targetPct − roundTripCostPct)
netLoss = stopPct + roundTripCostPct
breakevenWR = netLoss / (netWin + netLoss) × 100
// Round-trip cost
roundTripCostPct = (bidAsk / premium) × 200
// IV-adjusted stop
ivRank > 65 → 15% stop
ivRank > 40 → 20% stop
else        → 25% stop
// Position sizing
maxContracts = floor((accountSize × riskPct%) / (premium × 100 × stopPct / 100))
// Gamma acceleration
minsToClose < 30  → ×4.0
minsToClose < 60  → ×2.5
minsToClose < 120 → ×1.5
else              → ×1.0
// Portfolio Greeks
netDelta = Σ (delta × direction)      where direction: CALL=+1, PUT=−1
netTheta = Σ theta
netVega  = Σ vega
netGamma = Σ gamma
// Worst-case stress
For each active ticker:
  contracts = floor(accountRisk / (prem × 100 × stop%))
  loss      = contracts × prem × 100 × stop%
worstCase = Σ loss
// Hourly theta
hourlyTheta = dailyTheta × sessionFraction
  prime-time  → 0.09
  afternoon   → 0.14
  power-hour  → 0.38
// Net P&L
netPl = grossPl − ROUND_TRIP_COST_PCT[ticker]

18.4 Technology Stack (Reference Implementation)
Layer	Technology	Notes
Framework	React 18 + TypeScript	Strict typing for all data models
Build	Vite	Hot module replacement for live updates
Routing	Wouter	Lightweight SPA router (3 pages)
UI Components	shadcn/ui	Card, Badge, Progress, Table, Select
Charts	Recharts	LineChart (equity curve), BarChart (P&L by pattern)
Icons	Lucide React	Consistent icon library
Styling	Tailwind CSS v4 + CSS variables	Dark navy theme
State	React useState + useEffect	No external state library needed
Data refresh	setInterval 3000ms	Simulates real-time in demo; replace with WebSocket in production
Monorepo	pnpm workspace	Shared tooling, isolated artifact packages
18.5 Production Integration Points
To convert this from a demo system to a live one, replace these data sources:

Demo (current)	Production replacement
Static INITIAL_TICKERS array	Real-time options feed (CBOE LiveVol, Tradier, Tastytrade API)
Math.random() price drift	WebSocket quote stream
Hardcoded MARKET object	Live VIX, P/C ratio, $TICK from market data provider
Simulated Greeks	Live options chain Greeks from brokerage API
TODAY_EVENTS hardcoded array	Economic calendar API (Benzinga, MarketWatch Events API)
Session phase from wall clock	Same logic applies; ensure ET timezone conversion
Signal log buildHistory()	Real trade database (PostgreSQL recommended)
