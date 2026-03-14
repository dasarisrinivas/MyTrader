# SPY Weekly Options Bot

Sells short-dated SPY options every week to collect premium. Runs automatically during market hours.

---

## What It Does

1. **Monday / Tuesday / Wednesday at 9:35 AM ET** — looks for a trade
2. Checks market conditions (VIX level, SPY trend)
3. If conditions are good → sells 1 options contract and collects premium upfront
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
| 2+ PDT slots available | Sell a strangle (both sides) |

**What is "selling a put"?**
You collect premium ($) upfront and profit if SPY stays above the strike price by expiry.

---

## Entry Filters (all must pass)

| Filter | Rule |
|--------|------|
| VIX (fear index) | Must be between 12 and 30 |
| SPY trend | Checked against 20-day moving average |
| Day of week | Monday, Tuesday, or Wednesday only |
| Time | After 9:35 AM ET |
| PDT limit | Max 3 trades per rolling week |

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

## Risk Limits

- **1 contract max** — hard ceiling, never changes
- **VIX > 30** — no new trades (panic spike protection)
- **3 trades/week max** — PDT compliance (avoids pattern day trader rule)
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
  signal_engine.py ← decides what to trade
  option_chain.py  ← finds the right strike
  order_manager.py ← places and tracks orders
  risk_manager.py  ← monitors exits
  pdt_tracker.py   ← counts weekly trades
  notifier.py      ← Telegram alerts (optional)
  backtest/        ← historical simulation tools
```
