# SPY Weekly Options Bot

Sells short-dated SPY options every week to collect premium. Runs automatically during market hours.

---

## What It Does

1. **Monday / Tuesday / Wednesday at 9:35 AM ET** — looks for a trade
2. Runs 8 filters to decide if conditions are good enough to enter
3. If all filters pass → sells 1 options contract and collects premium upfront
4. Monitors the position throughout the week
5. Closes the trade automatically when profit target or stop loss is hit
6. **Thursday by 3:45 PM ET** — force-closes any remaining position

---

## Strategy

**Sell a short put** (most common) or **strangle** (put + call together)

| Condition | Action |
|-----------|--------|
| SPY above 20-day average | Sell a put (bullish bias) |
| SPY below 20-day average | Sell a call (bearish bias) |
| SPY near 20-day average + 2 PDT slots left | Sell a strangle (both sides) |

**What is "selling a put"?**
You collect premium ($) upfront and profit if SPY stays above the strike price by expiry.

---

## Entry Filters — All 8 Must Pass

Checked in this order every cycle. First failure stops evaluation.

| # | Filter | Rule | Why |
|---|--------|------|-----|
| 1 | **Event Risk** | Skip entry 1 day before FOMC, CPI, NFP | These events cause unpredictable vol spikes |
| 2 | **VIX Gate** | VIX must be between 12 and 30 | Below 12 = premium worthless; above 30 = panic risk |
| 3 | **IV Rank** | VIX must be in top 20% of its 1-year range | Only sell when premium is historically elevated |
| 4 | **SPY Trend** | Compare SPY price vs 20-day SMA | Determines which side to sell (put or call) |
| 5 | **Support / Resistance** | No puts within 5% of 52-week low; no calls within 3% of 52-week high | Avoids selling into major price levels |
| 6 | **Skew** | Skip puts if call IV significantly exceeds put IV | Inverted skew = market pricing in upside risk |
| 7 | **PDT Limit** | Max 3 round-trips per rolling 5-trading-day window | Regulatory compliance (Pattern Day Trader rule) |
| 8 | **Time / Day** | Monday–Wednesday, after 9:35 AM ET only | Weekly options need 2–4 days to decay |

---

## Strike Selection Filters

Once entry is approved, the bot scans the option chain and applies these filters to find the best contract:

| Filter | Rule | Why |
|--------|------|-----|
| **Delta** | Target ±0.25 delta (±0.07 tolerance) | ~25% probability of expiring worthless |
| **Expected Move** | Strike must be ≥ 1 standard deviation OTM | Avoids strikes inside the market's priced move |
| **Theta / Delta ratio** | \|theta\| / \|delta\| ≥ 0.08 | Ensures meaningful decay relative to directional risk |
| **Volume** | ≥ 200 contracts traded today | Confirms active market participation |
| **Open Interest** | ≥ 500 contracts | Confirms liquid, established strike |
| **Spread** | Bid/ask spread ≤ 15% of mid price | Avoids wide spreads that eat into premium |

Best contract = closest to 0.25 delta, ranked by theta/delta efficiency.

---

## Exit Rules (checked every 5 minutes)

| Exit Trigger | Condition |
|--------------|-----------|
| Profit target | Position gained 50% of max profit → close |
| Loss stop | Loss reached 2× premium collected → close |
| Delta stop | Option gone deep in-the-money (delta > 0.50) → close |
| Thursday EOD | 3:45 PM ET Thursday → force close no matter what |
| Emergency | SPY moves >1.5% in 5 minutes on Thursday → close immediately |

---

## Backtest Results (1 year, $5k account)

| Metric | Result |
|--------|--------|
| Total Return | +46.6% |
| Win Rate | 83.3% |
| Sharpe Ratio | 1.59 |
| Profit Factor | 1.70 |
| Max Drawdown | -19.6% |
| Avg P&L / Trade | +$27.73 |
| Trades / Year | 84 |

---

## Risk Limits

- **1 contract max** — hard ceiling, never changes
- **VIX > 30** — no new trades (panic spike protection)
- **3 trades/week max** — PDT compliance
- **5% of account** max at risk at any time

---

## Start / Stop

```bash
./start_spy_bot.sh        # Start live trading
./stop_spy_bot.sh         # Graceful stop
./stop_spy_bot.sh --force # Immediate stop
```

Logs → `logs/spy_options_bot.log`

---

## Files

```
spy_options_bot/
  main.py          ← entry point / main loop
  config.py        ← all settings
  signal_engine.py ← 8 entry filters (event risk, IV rank, S/R, skew, trend)
  option_chain.py  ← strike selection (delta, expected move, theta/delta, liquidity)
  order_manager.py ← places and tracks orders
  risk_manager.py  ← monitors exits
  pdt_tracker.py   ← counts weekly trades
  notifier.py      ← Telegram alerts (optional)
  backtest/        ← historical simulation tools
```
