# SPY Weekly Options Bot

Sells credit spreads on SPY every week to collect theta premium.
Runs automatically during NYSE market hours via IBKR TWS API.

> **Account:** $5,000 | **Mode:** Live | **Max position:** 1 spread
> **Backtest (1 yr, $5k, delta 0.25, credit spreads):** +27.81% return · Sharpe 1.28 · Max DD -13.0%

---

## ⚠️ Critical Risks — Read Before Running Live

### 1. Max Loss Is Defined — But It's Not Zero
Each trade is a **credit spread** (not a naked option). Your max loss per trade is:

```
Max Loss = (Spread Width − Net Credit) × 100 × contracts
         = ($10.00 − $0.80) × 100 × 1 = $920 worst case
```

This is capped. Unlike a naked put, SPY cannot drop past the long hedge leg.
But the 2× premium stop should exit at roughly $160 (2 × $0.80 × 100) long before max loss is reached.

### 2. Assignment Risk (Still Possible)
Selling a put spread means you may still be assigned on the short leg if it expires ITM.
The long leg offsets your obligation but the broker still processes the assignment first.
**The bot always closes positions by Thursday 3:45 PM ET** — this eliminates assignment risk
for the short leg in normal operation. A bot crash Thursday = close manually.

### 3. Gap Risk
Stops are checked every 5 minutes (every 60 seconds on Thursdays). A flash crash can move
SPY through the stop price between checks. The 2× stop is a target, not a guaranteed fill.

### 4. Bear Regime Call Spreads
When SPY is below its 200-day SMA, the bot sells **call spreads** only. Call spreads lose
if SPY rallies sharply. A strong bear-market bounce (like March 2020 recovery) can trigger
the loss stop on call spreads.

### 5. Correlation With MES Bot
Both bots are short volatility. In a VIX > 30 event the MES bot may also be under pressure.
The SPY bot halts new entries at VIX > 30 and triggers the VIX spike guard at VIX > 25% above
its 5-day average — but an open position can still be caught in a spike.

### 6. Ex-Dividend Dates
Short call positions are vulnerable to early assignment the day before SPY's ex-dividend date.
The bot blocks all entries 1 day before each SPY ex-div date. Verify exact dates each quarter
at etf.com and update `_EVENT_DATES` in `signal_engine.py` if they shift.

---

## How A Trade Happens (Step by Step)

### Step 1 — Cycle Starts (9:35 AM ET, Mon–Tue)

The bot wakes up after market open settles. Wednesday entries are blocked (only 2 DTE to
Friday expiry = dangerously high gamma). The cycle runs one check sequence:

```
9:35 AM ET → Run entry filters → Select strike → Build spread → Place order → Monitor
```

---

### Step 2 — Entry Filters (11 checks, first failure = skip day)

| # | Check | Rule | What Happens on Fail |
|---|---|---|---|
| 1 | **Event risk** | No FOMC / CPI / NFP / SPY ex-div today or tomorrow | Skip day, log reason |
| 2 | **Guard cooldown** | No active VIX spike or large-move pause | Skip until cooldown expires |
| 3 | **VIX gate** | 12.0 ≤ VIX ≤ 30.0 | Skip: premium too thin (<12) or panic regime (>30) |
| 4 | **VIX spike guard** | VIX ≤ 5-day avg × 1.25 | If breached → pause 3 calendar days, write to guard_state.json |
| 5 | **IV Rank** | VIX ≥ 20th percentile of its 52-week range | Skip: premium cheap relative to recent history |
| 6 | **Large move guard** | SPY open ≤ prior close × 1.02 (both directions) | If breached → pause 2 calendar days, write to guard_state.json |
| 7 | **SPY trend** | SPY vs 20-day SMA → determines which side to sell | No fail; sets direction (put/call/strangle) |
| 8 | **Market regime** | SPY vs 200-day SMA → bear regime = calls only | Restricts strategy; no puts in confirmed downtrend |
| 9 | **Support / Resistance** | No puts within 5% of 52w low; no calls within 3% of 52w high | Skip that leg |
| 10 | **Skew** | Skip puts if call IV > put IV by > 3% | Inverted skew = institutional upside hedging |
| 11 | **PDT limit** | < 3 round-trips in rolling 5-trading-day window | Skip: regulatory limit for accounts < $25k |

---

### Step 3 — Strike Selection

Once entry is approved, scan the live SPY option chain and apply:

| Filter | Rule |
|---|---|
| **Delta** | 0.18 – 0.32 (target 0.25) on the short leg |
| **DTE** | 5 – 7 calendar days (nearest Friday expiry) |
| **Expected move** | Strike must be ≥ 1σ OTM (σ = SPY × VIX/100 × √(DTE/365)) |
| **Theta/Delta ratio** | \|θ\|/\|Δ\| ≥ 0.08 — decay must justify the directional risk |
| **Volume** | ≥ 200 contracts traded today |
| **Open interest** | ≥ 500 contracts |
| **Bid/ask spread** | ≤ 15% of mid price and ≤ $0.10 absolute |

Among all passing strikes → select closest to delta 0.25, ranked by theta/delta efficiency.

---

### Step 4 — Credit Spread Pricing

The bot does **not** sell a naked option. It builds a spread:

**Bull Put Spread (uptrend — most common):**
```
Sell SPY 565 Put  (delta ~0.25) → receive $0.95 premium
Buy  SPY 555 Put  (10 strikes lower, the hedge) → pay $0.15 premium
                              ─────────────────────────────────────
Net credit received = $0.80/share = $80 per spread (1 contract = 100 shares)
Max loss = ($10.00 - $0.80) × 100 = $920 (if both legs expire deep ITM)
```

**Bear Call Spread (downtrend / bear regime):**
```
Sell SPY 580 Call (delta ~0.25) → receive $0.90 premium
Buy  SPY 590 Call (10 strikes higher, the hedge) → pay $0.20 premium
                              ─────────────────────────────────────
Net credit received = $0.70/share = $70 per spread
Max loss = ($10.00 - $0.70) × 100 = $930
```

**Floor check:** If net credit < $0.60, skip the trade. Low credit = not worth the margin.

---

### Step 5 — Order Placement

The order is a single **BAG (combo)** order sent to IBKR — both legs fill atomically. No leg risk.

```
Order type: Limit (SELL), price = net credit (e.g. $0.80)
TIF: DAY
If unfilled after 60 seconds → adjust limit by $0.01 toward market
Retry up to 3 times → cancel if still unfilled
```

Commission: $0.65 × 4 (2 legs × 2 sides) = $2.60 per spread round-trip.

---

### Step 6 — Position Monitoring

After fill, the bot monitors every **5 minutes** (every **60 seconds on Thursdays**).

| Exit Trigger | Condition | Action |
|---|---|---|
| **Profit target** | Cost to close ≤ 50% of net credit received | Buy to close → lock profit |
| **Loss stop** | Cost to close ≥ 2× net credit received | Buy to close → limit loss |
| **Delta stop** | Short leg \|delta\| > 0.50 (deep ITM) | Buy to close → assignment risk rising |
| **Thursday EOD** | 3:45 PM ET Thursday, any open position | Force close — never hold through Friday expiry |
| **Emergency gamma** | SPY moves >1.5% in any single 5-min bar on Thursday | Immediate close |

**Example trade lifecycle:**
```
Monday 9:35 AM  → Buy SPY 565/555 Put Spread, credit $0.80, target close at $0.40
Wednesday 2 PM  → Spread now worth $0.38 → profit target hit → close for $42 profit
                  ($0.80 - $0.38) × 100 - $2.60 commissions = $39.40 net
```

---

## Strategy Summary

| SPY vs SMA20 | SPY vs SMA200 | PDT Slots | Trade |
|---|---|---|---|
| Above (uptrend) | Bull market | Any | Bull put spread (sell put + buy put 10 lower) |
| Below (downtrend) | Bull market | Any | Bear call spread (sell call + buy call 10 higher) |
| Neutral | Bull market | ≥ 2 | Iron condor (both put spread + call spread) |
| Neutral | Bull market | 1 | Bull put spread only |
| Any | Bear market (below SMA200) | Any | Bear call spread only — no put spreads |

---

## Key Parameters

| Parameter | Value | Where Set |
|---|---|---|
| Max contracts | **1** | `MAX_CONTRACTS = 1` |
| Short leg delta target | **0.25** | `TARGET_DELTA_PUT/CALL = 0.25` |
| Delta acceptance range | **0.18 – 0.32** | `DELTA_TOLERANCE = 0.07` |
| Spread width | **$10** | `SPREAD_WIDTH = 10.0` |
| Min net credit | **$0.60** | `MIN_NET_CREDIT = 0.60` |
| Profit target | **50%** of credit | `PROFIT_TARGET_PCT = 0.50` |
| Loss stop | **2×** credit | `MAX_LOSS_MULTIPLE = 2.0` |
| Delta stop | **0.50** | `DELTA_STOP = 0.50` |
| VIX floor | **12.0** | `MIN_VIX = 12.0` |
| VIX ceiling | **30.0** | `MAX_VIX = 30.0` |
| IV Rank minimum | **20th percentile** | `MIN_IV_RANK = 0.20` |
| VIX spike guard | VIX > 5-day avg × **1.25** → pause **3 days** | `VIX_SPIKE_MULTIPLIER / SKIP_DAYS` |
| Large move guard | SPY gap > **2%** → pause **2 days** | `LARGE_MOVE_PCT / SKIP_DAYS` |
| Max account risk/trade | **5%** of NLV | `MAX_ACCOUNT_RISK_PCT = 0.05` |
| Daily loss limit | **3%** of account | `DAILY_LOSS_LIMIT_PCT = 0.03` |
| PDT max trades | **3** in 5-day window | `MAX_WEEKLY_TRADES = 3` |
| Trend SMA | **20-day** | `TREND_SMA_DAYS = 20` |
| Regime SMA | **200-day** | hardcoded in signal engine |
| Entry days | **Mon, Tue** only | `ENTRY_DAYS = (0, 1)` |
| Entry time | **9:35 – 15:30 ET** | `MARKET_OPEN/CLOSE_MINUTE` |
| Thursday force-close | **15:45 ET** | `EOD_CLOSE_HOUR/MINUTE` |
| Monitoring interval | **5 min** (60s Thursday) | `POLL_INTERVAL = 300` |
| IBKR port | **4001** (live) | `IBKR_LIVE_PORT = 4001` |
| Client ID | **20** | `CLIENT_ID = 20` |

---

## Backtest Results

**1-year baseline ($5k, delta 0.25, credit spreads):**

| Metric | Value |
|---|---|
| Total Return | **+27.81%** |
| Sharpe Ratio | **1.28** |
| Win Rate | **83.6%** (56/67 trades) |
| Max Drawdown | **-13.0%** |
| Avg P&L / Trade | **+$20.75** |
| Trades / Year | **67** |

**Crash period stress tests ($5k, credit spreads):**

| Period | Return | Max DD | Win Rate | Note |
|---|---|---|---|---|
| 2018 Q4 Crash | **-43.4%** | -46.2% | 60.9% | Fed rate hike + trade war, VIX spiked above 30 most days |
| 2020 COVID Crash | **-10.1%** | -20.8% | 80.0% | Spread cap limited losses vs naked; -34% SPY in 33 days |
| 2022 Bear Market | **-39.1%** | -40.1% | 67.2% | Regime filter forced calls-only; slow grind hurt call spreads |

**Key insight:** The 2020 result (-10.1%) improved significantly from prior naked-option version (-23%)
because the spread's $10 width caps the max loss per contract. 2022 is worse because the SMA200
regime filter switches to calls-only during the full-year bear grind — a known trade-off.

**Limitations:** Greeks reconstructed via Black-Scholes (VIX as IV proxy). Slippage modeled as 1–2%
of premium. Does not model assignment, early exercise, or dividend events.

---

## Margin Requirements

Credit spreads require **defined-risk margin** — much lower than naked options.

| Trade | Approximate Margin |
|---|---|
| Bull put spread (10-wide) | **~$1,000** (spread width − credit × 100) |
| Bear call spread (10-wide) | **~$1,000** |
| Iron condor (both) | **~$1,000** (IBKR takes the larger leg only) |

At $5k account this is 20% of capital per spread — manageable. IBKR rejects the order if
margin is insufficient and the bot logs the skip. Monitor available margin via TWS.

---

## What To Do If The Bot Crashes

| Situation | Action |
|---|---|
| Crashes, no open position | Restart normally — no urgency |
| Crashes Mon–Tue, open position | Restart; position reloads from JSON and monitoring resumes |
| Crashes Thursday, open position | **Close manually in TWS immediately** — assignment risk |
| Can't restart before 3:45 PM Thursday | Close manually. Do not wait. |
| Crashes during order entry | Check TWS — order may or may not have filled. Reconcile first. |

---

## Monitoring

**Telegram alerts fire on:**
- Trade opened (strikes, net credit, Greeks, PDT count, spread type)
- Trade closed (reason, P&L, running total)
- Guard triggered (VIX spike or large move — with resume date)
- PDT warning at 2/3 trades used
- Bot crash / reconnection

**Log file:** `logs/spy_options_bot.log`

**Daily checks (2 minutes):**
```
1. Check Telegram — any alerts overnight?
2. tail -f logs/spy_options_bot.log | grep "ERROR\|WARN\|TRADE\|GUARD"
3. Monday: confirm PDT count reset correctly
4. Each week: verify no SPY ex-dividend date falls this week
```

---

## Performance Tracking

| Metric | Target | Action if missed |
|---|---|---|
| Win rate (rolling 20 trades) | ≥ 65% | Audit filter settings — do not adjust blindly |
| Avg net credit collected | ≥ $0.60 / spread | IV Rank filter may be blocking too aggressively |
| Avg fill vs mid | ≤ $0.03 slippage | Widen liquidity filters or adjust limit retry step |
| Thursday force-close rate | ≤ 30% of trades | Profit target may be too tight |
| Guard cooldown days / month | ≤ 6 | Normal; if > 10 days/month check VIX regime |

If win rate drops below 60% over 20 consecutive trades — **pause and audit, do not tune parameters**.

---

## Scale-Up Rules

Do not increase contracts until both conditions hold:

```
1. Account grown to ≥ $10,000 (organic, not deposits)
2. Live win rate ≥ 65% over at least 20 trades (~3 months live data)
```

At $10k → 2 contracts. At $15k → 3 contracts.
Never increase mid-month based on a winning streak.

---

## Start / Stop

```bash
./start_spy_bot.sh          # Start (live trading, port 4001, clientId 20)
./stop_spy_bot.sh           # Graceful stop — waits for open order to resolve
./stop_spy_bot.sh --force   # Immediate kill — use only if no open position
```

**Never force-stop with an open position on Thursday.**

Logs → `logs/spy_options_bot.log`
PID file → `logs/spy_options_bot.pid`

---

## Files

```
spy_options_bot/
  main.py             ← entry point, main loop, --dry-run / --once / --reset-pdt flags
  config.py           ← all constants and env-var overrides
  signal_engine.py    ← 11 entry filters (event, guards, VIX, IV rank, trend, regime, S/R, skew, PDT)
  guard_tracker.py    ← JSON-persisted VIX spike + large-move cooldowns (guard_state.json)
  option_chain.py     ← strike selection + fetch_hedge_leg() for spread pricing
  order_manager.py    ← BAG/ComboLeg spread orders, position tracking, stop monitoring
  pdt_tracker.py      ← rolling 5-day PDT compliance, strangle-aware (2 slots)
  ibkr_connection.py  ← ib_insync wrapper, exponential backoff reconnect
  notifier.py         ← Telegram alerts (optional)
  backtest/
    backtest_engine.py   ← full spread simulation (BS pricing, slippage, credit floor)
    run_backtest.py      ← run 1-year backtest, outputs HTML report + trades.csv
    run_crash_tests.py   ← 2018/2020/2022 crash period comparison table
    options_simulator.py ← Black-Scholes, find_strike_for_delta, vix_to_sigma
    data_downloader.py   ← yfinance SPY + VIX downloader with file caching
```

---

## Related Bots

| Bot | clientId | Port | Asset |
|---|---|---|---|
| MES Futures Bot | 11 | 4001 | /MES (Micro E-mini S&P futures) |
| SPY Options Bot | 20 | 4001 | SPY weekly credit spreads |
| Backtest module | — | — | Offline only — never connects live |

**Combined risk:** Both bots lose in sharp unexpected moves. They are not hedges.
In a VIX > 30 event, the options bot halts new entries but an open spread position
can still reach max loss. The MES bot may also be under simultaneous stress.

---

## Tax Notes

All trades held < 1 week → **ordinary income** (short-term). SPY ETF options are **not**
Section 1256 contracts (SPX index options get 60/40 treatment — SPY does not). Consult a
tax advisor. Export trade log via IBKR Flex Query at year-end.
