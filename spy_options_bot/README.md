# SPY Weekly Options Bot

Sells short-dated SPY options every week to collect theta premium.
Runs automatically during NYSE market hours via IBKR TWS API.

> **Account:** $5,000 | **Mode:** Live | **Max position:** 1 contract
> **Backtest (1 yr, $5k, delta 0.16):** +6.4% return · Sharpe 0.15 · Max DD -12.9%

---

## ⚠️ Critical Risks — Read Before Running Live

These are not generic disclaimers. They are specific failure modes for this
exact strategy that will cost real money if not understood.

### 1. Assignment Risk (Most Important)
Selling a put means you may be **forced to buy 100 SPY shares** if the option
expires in-the-money. At ~$560/share, that's a $56,000 obligation on a $5,000
account — triggering a margin call and forced liquidation by IBKR.

**This bot mitigates assignment risk by:**
- Closing all positions by Thursday 3:45 PM ET (never holds through Friday expiry)
- Applying a 2× premium loss stop well before deep ITM territory
- Delta stop at 0.50 — exits before probability of assignment gets dangerous

**You must still:** Ensure your IBKR account has margin enabled and understand
that a bot crash on Thursday could leave a position open through expiry.
If the bot crashes Thursday and you can't restart it, **close the position manually.**

### 2. Gap Risk
The bot's stops are checked every 5 minutes (60 seconds on Thursdays). An
overnight gap or a halt + resume can move SPY through your stop price without
triggering an exit. The 2× premium stop is a *target* price, not a guaranteed
fill price. In a flash crash, you may stop out at 3× or 4× premium.

### 3. Naked Call Risk
When SPY is below its 20-day SMA, the bot sells calls. Unlike puts (which have
a floor at zero), calls have theoretically unlimited loss if SPY gaps up sharply.
The delta stop (0.50) and 2× premium stop limit this in practice, but a
surprise takeover bid or macro shock overnight can gap through both stops.

### 4. Correlation With MES Bot
Both bots are **short volatility** — they lose money when markets move
sharply in either direction. In a crisis (VIX > 30), the MES bot may also
be under pressure simultaneously. Do not assume the two bots are uncorrelated.

### 5. Ex-Dividend Dates
SPY pays quarterly dividends (~$1.50/quarter). Short call positions are
vulnerable to **early assignment** the day before the ex-dividend date as
call holders exercise to capture the dividend.

The bot **blocks all entries** 1 day before each SPY ex-dividend date.
SPY ex-dividend dates (approximate): mid-March, mid-June, mid-September,
mid-December. Verify exact dates at etf.com each quarter and update
`_EVENT_DATES` in `signal_engine.py` if they shift.

---

## What It Does

1. **Monday–Tuesday at 9:35 AM ET** — evaluates market conditions (Wed blocked: gamma risk)
2. Runs 9 filters in sequence — first failure stops evaluation immediately
3. If all pass → scans option chain → selects best contract → sells 1 contract
4. Collects premium upfront (credited to account same day)
5. Monitors position every 5 min (every 60s on Thursdays)
6. Closes automatically when profit target or stop is hit
7. **Thursday 3:45 PM ET** — force-closes any open position regardless of P&L

---

## Strategy

Sell premium on the side of the market that is less likely to be reached.

| SPY vs 20-day SMA | SPY vs 200-day SMA | PDT Slots | Action |
|---|---|---|---|
| Above SMA20 (uptrend) | Above SMA200 (bull) | Any | Sell put ~0.25 delta |
| Below SMA20 (downtrend) | Above SMA200 (bull) | Any | Sell call ~0.25 delta |
| Within 0.5% of SMA20 (neutral) | Above SMA200 (bull) | ≥ 2 | Sell strangle (put + call) |
| Within 0.5% of SMA20 (neutral) | Above SMA200 (bull) | 1 | Sell put (default to safer leg) |
| Any trend | Below SMA200 (bear) | Any | Sell call only — no puts in bear market |

**Why the 200-day SMA regime filter?**
In a bear market (SPY below SMA200), selling puts is directionally wrong — you're
selling downside protection into a confirmed downtrend. The regime filter blocks
puts entirely and restricts to calls (or no trade if trend is also down).

**What is selling a put?**
You receive cash (premium) upfront. You profit if SPY closes *above* the
strike price at expiry. Your maximum profit is the premium received. Your
risk is if SPY falls sharply below the strike — mitigated by the 2× stop.

**What is a strangle?**
Selling both a put and a call simultaneously. You collect premium on both
sides and profit if SPY stays in the range between the two strikes. Counts
as 2 PDT round-trips (open + close each leg).

---

## Entry Filters — All 9 Must Pass

Checked every cycle in this order. First failure stops evaluation immediately.

| # | Filter | Rule | Why |
|---|---|---|---|
| 1 | **Event Risk** | No entry 1 calendar day before FOMC, CPI, NFP, SPY ex-div | Binary events cause IV spikes that invalidate any premium model |
| 2 | **VIX Gate** | 12 ≤ VIX ≤ 30 at 9:35 AM (intraday snapshot) | <12 = premium too thin; >30 = panic regime, realized vol exceeds implied |
| 3 | **IV Rank** | VIX in top 20% of its own 1-year range | Only sell when premium is historically elevated vs recent baseline |
| 4 | **SPY Trend** | SPY price vs 20-day SMA | Determines which side to sell — avoids fighting the trend |
| 5 | **Market Regime** | SPY vs 200-day SMA — bear regime restricts to calls only | Prevents selling puts into a confirmed downtrend |
| 6 | **Support / Resistance** | No puts within 5% of 52-week low; no calls within 3% of 52-week high | Avoids selling into major technical levels where reversals accelerate |
| 7 | **Skew** | Skip puts if call IV > put IV by meaningful margin | Inverted put/call skew signals institutional hedging of upside — respect it |
| 8 | **PDT Limit** | < 3 round-trips in rolling 5-trading-day window | FINRA Pattern Day Trader rule — violation risks account restriction |
| 9 | **Time / Day** | Monday–Tuesday, 9:35 AM–3:30 PM ET only | Wednesday blocked (2 DTE = high gamma risk); needs ≥ 3 days of theta decay |

**Filter 3 — IV Rank vs VIX Gate:**
Filter 2 checks absolute level. Filter 3 checks relative level vs the past year.
Both must pass. Example: VIX=18 passes Filter 2 but if VIX has been 25–35 all year,
18 is in the bottom 20% of its range — Filter 3 blocks entry because premium is
cheap relative to recent history.

**Filter 8 — PDT explained:**
A round-trip = 1 open + 1 close. A strangle = 2 round-trips.
The 5-day window is rolling (not calendar week).

---

## Strike Selection

Once entry is approved, the bot scans the SPY option chain and applies these
filters to find the best contract. All must pass:

| Filter | Rule | Why |
|---|---|---|
| **Delta** | 0.18–0.32 (target 0.25) | ~25% probability of expiring ITM |
| **Expected Move** | Strike ≥ 1σ OTM (σ = SPY × VIX/100 × √(DTE/365)) | Never sell inside the market's own priced move |
| **Theta/Delta ratio** | \|θ\|/\|Δ\| ≥ 0.08 | Decay rate must justify directional risk taken |
| **Volume** | ≥ 200 contracts today | Active market — ensures fills |
| **Open Interest** | ≥ 500 contracts | Established strike — ensures exit liquidity |
| **Bid/Ask Spread** | ≤ 15% of mid price | Wide spreads erode premium before the trade starts |

**Best contract selection:** Among all strikes passing every filter, select the
one closest to 0.25 delta, then ranked by theta/delta efficiency (highest first).

**Order type:** Limit order at mid-price. If unfilled after 60 seconds, adjust
by $0.01 toward market. Repeat up to 3 times, then cancel if still unfilled.

---

## Exit Rules

Checked every **5 minutes** during market hours (every **60 seconds on Thursdays**).

| Trigger | Condition | Action |
|---|---|---|
| **Profit target** | Current cost to close ≤ 50% of premium collected | Buy to close — lock in profit |
| **Loss stop** | Current cost to close ≥ 2× premium collected | Buy to close — limit loss |
| **Delta stop** | \|delta\| > 0.50 (deep ITM) | Buy to close — assignment risk rising |
| **Thursday EOD** | 3:45 PM ET Thursday, any open position | Buy to close — never hold through expiry |
| **Emergency gamma** | SPY moves >1.5% in any single 5-min bar on Thursday | Buy to close immediately |

**Why 50% profit target?**
Closing at 50% of max profit captures most available theta while cutting time-in-trade
by ~60%. Less time in trade = less exposure to adverse moves.

---

## Backtest Results

**1-year baseline ($5k, delta 0.25 live / 0.16 backtest):**

| Metric | Value |
|---|---|
| Total Return | **+6.4%** |
| Sharpe Ratio | **0.15** |
| Win Rate | **82.2%** (60/73 trades) |
| Max Drawdown | **-12.9%** (= -$645 on $5k) |
| Avg P&L / Trade | **+$4.41** |
| Trades / Year | **73** |

**Crash period stress tests ($5k):**

| Period | Return | Max DD | Win Rate | Note |
|---|---|---|---|---|
| 2018 Q4 Crash | **-42.5%** | -44.0% | 66.7% | Fed rate hike + trade war selloff |
| 2020 COVID Crash | **-23.1%** | -31.0% | 82.8% | -34% SPY drop in 33 days |
| 2022 Bear Market | **-19.8%** | -23.2% | 70.0% | Full-year Fed tightening cycle |

**What crash tests tell you:** Short premium selling loses money in every crash
scenario even with all filters active. The 1-contract hard limit is your real
crash protection — your maximum loss per trade is bounded by 2× the premium
collected (~$100–200 per trade), not by SPY's move.

**Backtest limitations:**
- Greeks reconstructed via Black-Scholes (VIX as IV proxy) — not actual historical quotes
- Slippage modeled as % of premium (1% entry, 1–2% exit) — real fills may be wider in fast markets
- Does not model assignment risk, early exercise, or dividend events
- Past performance does not guarantee future results

---

## Risk Limits

| Limit | Value | Rationale |
|---|---|---|
| Max contracts | **1** | Hard ceiling — never changes regardless of account growth |
| Max account risk | **5%** ($250) | Caps max loss per trade at $250 on $5k |
| VIX floor | **12.0** | Below = premium too thin to justify risk |
| VIX ceiling | **30.0** | Above = panic regime, stops get run |
| PDT trades/week | **3** | Regulatory maximum for accounts < $25k |
| Max loss per trade | **2× premium** | Defined before entry, stored in position log |
| Daily loss limit | **3% of account** | If account drops $150 in a day, pause new entries |

---

## Margin Requirements

Selling naked puts and calls requires a **margin account** with options Level 3+
at IBKR. Approximate margin requirements per 1 SPY contract:

- **Short put:** ~$1,500–2,500 (varies with strike and volatility)
- **Short call:** ~$1,500–2,500
- **Strangle:** ~$2,500–4,000 (IBKR uses the larger leg + premium of smaller leg)

At $5k account size, a strangle may consume 50–80% of account margin. IBKR will
reject the order if margin is insufficient — the bot logs this and skips the trade.
Monitor available margin via TWS.

---

## What To Do If The Bot Crashes

| Situation | Action |
|---|---|
| **Crashes with no open position** | Restart normally — no urgency |
| **Crashes Mon–Tue with open position** | Restart bot; it reloads position from JSON and resumes monitoring |
| **Crashes Thursday with open position** | **Close manually in TWS immediately** — do not wait for restart |
| **Can't restart before market close Thursday** | Close manually. This is the assignment risk scenario. |
| **Crashes during order placement** | Check TWS manually — order may or may not have filled. Reconcile before restarting. |

---

## Monitoring

**Telegram alerts fire on:**
- Trade opened (strike, premium, Greeks, PDT count)
- Trade closed (reason, P&L, running total)
- PDT warning at 2/3 trades used
- Any risk stop triggered
- Bot crash / reconnection

**Log file:** `logs/spy_options_bot.log`

**Key things to check daily (takes 2 minutes):**
```
1. Check Telegram — any overnight alerts?
2. tail -f logs/spy_options_bot.log | grep "ERROR\|WARN\|TRADE"
3. Monday morning: confirm PDT count reset correctly
4. Every trade week: verify no SPY ex-dividend date falls in the week
```

---

## Performance Tracking

Track these weekly alongside the bot log:

| Metric | Target | Action if missed |
|---|---|---|
| Win rate (rolling 20 trades) | ≥ 65% | Review if filters are firing correctly |
| Avg premium collected | ≥ $0.50/contract | VIX may be too low — check IV rank filter |
| Avg fill vs mid-price | ≤ $0.03 slippage | Liquidity filters may need tightening |
| Thursday force-close rate | ≤ 30% of trades | Profit target may be too tight |

If win rate drops below 60% over 20 consecutive trades, **pause the bot and audit**.
Do not adjust parameters — diagnose first.

---

## Scale-Up Rules

Do not increase position size until **both** conditions are met:

```
1. Account has grown to ≥ $10,000 (organic growth, not deposits)
2. Live trading win rate ≥ 65% over at least 20 trades (~3 months)
```

At $10k → 2 contracts. At $15k → 3 contracts. Linear from there.
Never increase contracts mid-month based on a winning streak.

---

## Tax Notes

All trades are short-term (held < 1 week). All P&L is **ordinary income**, not
capital gains. SPY ETF options are **not** Section 1256 contracts (that treatment
applies to SPX index options only) — no 60/40 tax treatment. Consult a tax advisor.
Export trade log from IBKR Flex Query at year-end for reporting.

---

## Start / Stop

```bash
./start_spy_bot.sh          # Start (live trading)
./stop_spy_bot.sh           # Graceful stop — waits for open order to resolve
./stop_spy_bot.sh --force   # Immediate stop — use only if no open position
```

**Never force-stop with an open position.** Use graceful stop only.

Logs → `logs/spy_options_bot.log`

---

## Files

```
spy_options_bot/
  main.py           ← entry point, main loop, CLI flags (--dry-run, --once, --reset-pdt)
  config.py         ← all settings, env var overrides
  signal_engine.py  ← 9 entry filters (event risk, VIX, IV rank, trend, regime, S/R, skew, PDT, time)
  option_chain.py   ← strike selection (delta, expected move, theta/delta, liquidity)
  order_manager.py  ← places/tracks orders, position log JSON, entry premium persistence
  risk_manager.py   ← exit monitoring (profit target, stops, Thursday EOD, emergency gamma)
  pdt_tracker.py    ← rolling 5-day PDT compliance, strangle-aware (2 slots), JSON persistence
  ibkr_connection.py← ib_insync wrapper, exponential backoff reconnect (10 retries)
  logger.py         ← Loguru, ET timestamps, rotating file handler
  notifier.py       ← Telegram alerts (optional — bot runs fine without it)
  backtest/         ← historical simulation (run separately, never touches live account)
```

---

## Related Bots

This bot runs alongside the **MES Futures Bot** on the same IBKR account.

| Bot | clientId | Port | Asset |
|---|---|---|---|
| MES Futures Bot | 11 | 4001 | /MES (E-mini S&P futures) |
| SPY Options Bot | 20 | 4001 | SPY weekly options |
| Backtest module | 3 | — | Offline only — never connects live |

**Combined risk note:** Both bots lose money during sharp, unexpected moves.
They are not hedges for each other. In a VIX > 30 event, the options bot stops
trading but an existing MES position may be under stress simultaneously.
