# Quick Reference: SPY Futures Trading Bot Safety Features

## Critical Configuration Checklist

### ✅ Before Going Live

1. **Position Sizing (CRITICAL)**
   ```yaml
   trading:
     position_sizing_method: "fixed_fraction"  # NOT "kelly"
     risk_per_trade_pct: 0.005  # 0.5% per trade
   ```

2. **LLM Mode (CRITICAL)**
   ```yaml
   llm:
     override_mode: false  # MUST be false - commentary only
     use_background_thread: true
   ```

3. **Indicator Warm-Up (CRITICAL)**
   ```yaml
   trading:
     min_bars_for_signals: 200  # INCREASED from 15
   ```

4. **Safety Stops**
   ```yaml
   trading:
     disaster_stop_pct: 0.007  # 0.7% emergency stop
     max_trade_duration_minutes: 60
     trade_cooldown_minutes: 5
   ```

## Safety Features at a Glance

| Feature | When | Action | Config |
|---------|------|--------|--------|
| **Disaster Stop** | Every loop | Force close if >0.7% loss | `disaster_stop_pct: 0.007` |
| **Time Exit** | Every loop | Close if open >60 min | `max_trade_duration_minutes: 60` |
| **Trade Cooldown** | After trade | Wait 5 min before next | `trade_cooldown_minutes: 5` |
| **Regime Filter** | Before trade | Block if conditions bad | `min_atr_threshold: 0.5` |
| **Latency Guard** | Every loop | Warn if >3 sec | `max_loop_latency_seconds: 3.0` |

## What Changed (TL;DR)

### Before → After

1. **Position Sizing**
   - ❌ Kelly Criterion (aggressive)
   - ✅ Fixed 0.5% per trade (safe)

2. **Indicator Warm-Up**
   - ❌ 15 bars (unstable)
   - ✅ 200 bars (stable)

3. **LLM Role**
   - ❌ Can override trades (risky)
   - ✅ Commentary only (safe)

4. **Parameter Optimization**
   - ❌ Real-time loop (slow)
   - ✅ Daily batch (fast)

5. **Stop Orders**
   - ❌ STOP-MARKET (slippage)
   - ✅ STOP-LIMIT (better fills)

6. **Safety Checks**
   - ❌ Only bracket orders
   - ✅ Multiple independent checks

## Quick Start Commands

### Backtest
```bash
python main.py backtest --config config.yaml --data data/historical_spy_es.parquet
```

### Live Trading
```bash
python main.py live --config config.yaml
```

### Daily Optimization (Optional)
```bash
python -c "
from mytrader.optimization.daily_optimizer import daily_optimization
from mytrader.strategies.rsi_macd_sentiment import RsiMacdSentimentStrategy
from mytrader.strategies.momentum_reversal import MomentumReversalStrategy
import pandas as pd

strategies = [RsiMacdSentimentStrategy(), MomentumReversalStrategy()]
data = pd.read_parquet('data/historical_spy_es.parquet')
param_grid = {
    'rsi_period': [14, 21, 28],
    'macd_fast': [12, 16, 20],
}
daily_optimization(strategies, data, param_grid)
"
```

## Trading Loop Flow

```
┌─────────────────────────────────────────┐
│ 1. Get Current Price                    │
│ 2. Build Price History                  │
│ 3. Check Minimum Bars (200)             │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ 4. Market Regime Filter                 │
│    - ATR threshold                       │
│    - Trading hours                       │
│    - Economic events                     │
│    → HOLD if not tradable               │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ 5. Check Existing Position              │
│    a. Disaster Stop (>0.7% loss)        │
│    b. Time-Based Exit (>60 min)         │
│    c. Normal exits (SL/TP/signal)       │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ 6. Check Trade Cooldown (5 min)         │
│    → HOLD if in cooldown                │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ 7. Generate Quant Signal                │
│    - Trend, Breakout, Mean Reversion    │
│    - LLM commentary (background)        │
│    - Quant decides, LLM logs only       │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ 8. Calculate Position Size              │
│    - Fixed Fractional (0.5%)            │
│    - Based on stop distance             │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ 9. Place Order with STOP-LIMIT          │
│    - Entry: LIMIT order                 │
│    - Stop: STOP-LIMIT (2-tick offset)   │
│    - Target: LIMIT order                │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ 10. Track Entry Time & Apply Cooldown   │
│ 11. Check Loop Latency (<3 sec)         │
│ 12. Log Everything                       │
└─────────────────────────────────────────┘
```

## Log Messages to Watch For

### ✅ Good
```
✅ History complete with 200 bars
✅ Market regime suitable for trading
✅ Position closed: Take profit hit
✓ Loop completed in 1.23s
```

### ⚠️ Warning
```
⚠️ Market regime not tradable: ATR too low
⚠️ Loop latency high: 3.5s
⏸️ Trade cooldown active: 3.2 minutes remaining
```

### 🚨 Alert
```
🚨 DISASTER STOP TRIGGERED! Position moved 0.8% against us
⏰ TIME-BASED EXIT triggered: Trade open for 65 minutes
```

## Monitoring Checklist

Daily:
- [ ] Check disaster stops triggered (should be rare)
- [ ] Review time-based exits (should be occasional)
- [ ] Check regime filter blocks (should be common)
- [ ] Monitor loop latency (should be <2 sec)
- [ ] Review LLM commentary quality

Weekly:
- [ ] Run daily optimization
- [ ] Review win rate and profit factor
- [ ] Analyze regime filter effectiveness
- [ ] Check for any repeated errors
- [ ] Update parameter grid if needed

## Emergency Procedures

### If Disaster Stop Triggers Frequently
1. Increase `disaster_stop_pct` from 0.7% to 1%
2. Review stop-loss settings (may be too tight)
3. Check for high volatility regime
4. Consider reducing position size

### If Too Many Cooldown Blocks
1. Reduce `trade_cooldown_minutes` from 5 to 3
2. Review strategy for over-trading
3. Check if regime filter is working

### If Loop Latency High
1. Disable LLM temporarily
2. Check network connection
3. Review feature engineering code
4. Consider upgrading hardware

### If Regime Filter Blocks Everything
1. Lower `min_atr_threshold` from 0.5 to 0.3
2. Expand trading hours window
3. Review recent market conditions
4. Adjust thresholds in config

## Support

- **Documentation:** See REFACTORING_SUMMARY.md for details
- **Logs:** Check `logs/` directory
- **Config:** Review `config.yaml` settings
- **Backtest:** Test changes in backtest mode first

## Key Files

```
mytrader/
├── strategies/
│   ├── market_regime_filter.py    ← Regime checking
│   ├── weighted_voting.py         ← Weighted signals (future)
│   └── multi_strategy.py          ← Main strategy
├── optimization/
│   └── daily_optimizer.py         ← Batch optimization
├── llm/
│   └── background_worker.py       ← Non-blocking LLM
├── risk/
│   └── manager.py                 ← Position sizing
└── execution/
    └── ib_executor.py             ← Order execution

main.py                             ← Trading loop
config.example.yaml                 ← Example config
REFACTORING_SUMMARY.md              ← Full documentation
```

---

**Remember:** Safety first! Test in backtest, then paper trading, before going live.
