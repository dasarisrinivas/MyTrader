# Backtesting Guide

This document describes how to run historical backtests using the exact same
strategy, risk, and execution logic as the live trading bot.

## Quick Start

```bash
# Basic 2-year backtest (auto-selects best data source)
python -m backtest.run --symbol MES --start 2024-01-09 --end 2026-01-09

# With Databento data (recommended for long backtests)
export DATABENTO_API_KEY="your-key"
python -m backtest.run --symbol MES --start 2024-01-09 --end 2026-01-09 --data-source databento

# With Polygon data (alternative)
export POLYGON_API_KEY="your-key"
python -m backtest.run --symbol MES --start 2024-01-09 --end 2026-01-09 --data-source polygon

# Using IB for short-range data (<30 days)
python -m backtest.run --symbol MES --start 2026-01-04 --end 2026-01-09 --data-source ib

# Proxy mode (SPY instead of ES/MES) - limited to 8 days for 1m data
python -m backtest.run --symbol MES --start 2024-01-01 --end 2026-01-01 --proxy-mode

# Using local data file
python -m backtest.run --symbol MES --data-source file --data-file data/es_historical.csv
```

## Data Source Priority

When `--data-source auto` (default), the system tries sources in order:

1. **Databento** - Best quality, continuous futures, 2+ years history
2. **Polygon** - Good quality, lower cost than Databento
3. **Interactive Brokers** - Only has recent data (IB doesn't keep expired contracts)
4. **SPY Proxy** - Fallback, limited to 8 days for 1-minute data

## Architecture

The backtest framework reuses the **exact same decision-making logic** as the live bot:

```
                      ┌─────────────────────────────────────────────────────────┐
                      │               BACKTEST ENGINE                           │
                      └─────────────────────────────────────────────────────────┘
                                            │
           ┌────────────────────────────────┼────────────────────────────────┐
           │                                │                                │
           ▼                                ▼                                ▼
  ┌─────────────────┐            ┌─────────────────┐            ┌─────────────────┐
  │   Data Layer    │            │  Strategy Layer │            │   Risk Layer    │
  └─────────────────┘            └─────────────────┘            └─────────────────┘
  │ Databento       │            │ MesOneMinute    │            │ RiskManager     │
  │ Polygon         │───────────▶│ TrendStrategy   │───────────▶│ RiskGate        │
  │ IBHistorical    │            │   .generate()   │            │   .evaluate()   │
  │ FallbackProxy   │            │ (same as live)  │            │ (same as live)  │
  └─────────────────┘            └─────────────────┘            └─────────────────┘
                                            │
                                            ▼
                              ┌─────────────────────────┐
                              │    BrokerSimulator      │
                              │  ─────────────────────  │
                              │  • Market/Limit/Stop    │
                              │  • Bracket orders       │
                              │  • Slippage model       │
                              │  • Commission tracking  │
                              └─────────────────────────┘
                                            │
                                            ▼
                              ┌─────────────────────────┐
                              │   Analysis & Reports    │
                              │  ─────────────────────  │
                              │  • Sharpe, CAGR, MDD    │
                              │  • Regime analysis      │
                              │  • TCO contribution     │
                              │  • HTML/MD reports      │
                              └─────────────────────────┘
```

## Data Sources

### Option 1: Databento (Recommended for 2+ Year Backtests)

Databento provides high-quality continuous futures data with proper roll adjustment.

**Setup:**
1. Create account at https://databento.com
2. Get API key from dashboard
3. Set environment variable:
   ```bash
   export DATABENTO_API_KEY="db-xxxxx-xxxxx"
   ```

**Usage:**
```bash
python -m backtest.run --symbol MES --start 2024-01-09 --end 2026-01-09 --data-source databento
```

**Cost:** ~$0.01-0.05 per day of 1-minute data (varies by dataset)

### Option 2: Polygon.io

Polygon provides continuous futures data at lower cost.

**Setup:**
1. Create account at https://polygon.io
2. Get API key (free tier available)
3. Set environment variable:
   ```bash
   export POLYGON_API_KEY="xxxxx"
   ```

**Usage:**
```bash
python -m backtest.run --symbol MES --start 2024-01-09 --end 2026-01-09 --data-source polygon
```

### Option 3: Interactive Brokers

IB is best for **short-range backtests** (<30 days) on current/recent contracts.

⚠️ **Limitation:** IB does not keep historical contract definitions for expired futures.
You cannot get 2-year historical data from IB for futures like MES.

**Setup:**
1. Ensure TWS/IB Gateway is running (port 4002 for gateway, 7497 for paper)
2. Market data subscription required for ES/MES

**Usage:**
```bash
python -m backtest.run --symbol MES --start 2026-01-04 --end 2026-01-09 --data-source ib
```

### Option 4: SPY Proxy (Fallback)

When no premium data source is available, SPY ETF data from yfinance is used as a proxy.

⚠️ **Caveats:**
- Limited to 8 days of 1-minute data (yfinance restriction)
- Automatically falls back to daily data for longer periods
- Results labeled as "proxy_spy" to distinguish from real futures data
- No overnight/Globex session data
- Volatility characteristics differ from futures

**Usage:**
```bash
python -m backtest.run --symbol MES --proxy-mode
```

### Option 5: Local Files

For pre-downloaded data:

```bash
python -m backtest.run --data-source file --data-file data/es_historical.parquet
```

Required columns: `timestamp`, `open`, `high`, `low`, `close`, `volume`

## Continuous Futures

The backtest handles contract rolling automatically:

```yaml
# In configs/backtest.yaml
roll:
  method: "volume_oi_crossover"  # Switch when volume/OI shifts
  days_before_expiry: 5          # Or roll N days before expiry
  adjustment: "back_adjust"      # Adjust historical prices
```

Roll methods:
- `volume_oi_crossover`: Roll when front month volume < back month (most realistic)
- `days_before_expiry`: Roll N days before expiry (simpler)

Adjustment methods:
- `back_adjust`: Adjust all historical prices (preserves returns, changes absolute levels)
- `ratio_adjust`: Ratio-based adjustment (preserves percentage moves)
- `none`: No adjustment (may have gaps at rolls)

## Configuration

The full configuration file at `configs/backtest.yaml` includes:

| Section | Description |
|---------|-------------|
| `data` | Symbol, bar sizes, roll configuration |
| `backtest` | Period, capital, costs |
| `strategy` | EMA/RSI/ATR periods (matches `OneMinuteStrategyConfig`) |
| `trading` | Max contracts, tick size, bracket defaults |
| `risk_gate` | Daily limits, lockout rules (matches `RiskGateConfig`) |
| `optimizer` | TCO settings for TP extension |
| `analysis` | Regime bucketing, benchmarks |
| `output` | Report formats, trace mode |

### Key Parameters

```yaml
strategy:
  ema_fast: 9
  ema_slow: 21
  rsi_period: 14
  atr_multiplier_sl: 1.5    # Stop = 1.5x ATR
  atr_multiplier_tp: 2.5    # Target = 2.5x ATR

risk_gate:
  max_daily_loss: 1500.0
  max_consecutive_losses: 4
  consecutive_loss_lockout_minutes: 60
```

## Output

Results are saved to `reports/` (configurable):

```
reports/
├── backtest_MES_2024-01-09_2026-01-09_20260109_143022.html   # Interactive report
├── backtest_MES_2024-01-09_2026-01-09_20260109_143022.json   # Raw metrics
├── backtest_MES_2024-01-09_2026-01-09_20260109_143022_trades.csv
└── backtest_MES_2024-01-09_2026-01-09_20260109_143022_equity.csv
```

### Metrics Computed

**Standard Metrics:**
- Total Return, P&L
- Sharpe Ratio (annualized)
- Maximum Drawdown
- Win Rate, Profit Factor
- Average Win/Loss

**Strategy-Specific:**
- ATR regime performance (low/medium/high vol)
- VIX regime performance
- RTH vs ETH performance
- TCO (TrendContinuationOptimizer) contribution
- Signal type breakdown (pullback vs breakout)

**Diagnostics:**
- Block reason histogram (why trades were rejected)
- Consecutive loss sequences
- Time-of-day performance

## Avoiding Lookahead Bias

The engine enforces strict no-lookahead rules:

1. **Feature calculation**: All indicators (EMA, RSI, ATR) are calculated using data up to and including the current bar only
2. **MTF alignment**: 5-minute features use only completed 5m bars at the time of each 1m decision
3. **Signal generation**: `strategy.generate()` receives the same DataFrame view as live
4. **Order execution**: Fills occur at the next bar's price, not current bar

To verify no lookahead:
```python
# In backtest engine, we assert:
assert all(features.index <= current_bar.name)
```

## Decision Trace Mode

For debugging and live comparison:

```bash
python -m backtest.run --symbol MES --start 2024-01-09 --end 2026-01-09 --trace
```

This outputs `logs/backtest_decisions.csv`:

```csv
timestamp,bar_close,ema_fast,ema_slow,rsi,atr,signal_type,signal_direction,risk_gate_result,order_action
2024-01-10 10:30:00,4800.25,4799.50,4795.00,45.2,3.5,pullback,long,allowed,submit_bracket
2024-01-10 10:31:00,4801.00,4800.00,4795.50,47.8,3.4,none,,allowed,hold
...
```

Compare this with live decision logs to validate backtest fidelity.

## Reproducing Results

For reproducibility:

```bash
# Save the exact command
python -m backtest.run \
  --symbol MES \
  --start 2024-01-09 \
  --end 2026-01-09 \
  --config configs/backtest.yaml \
  --slippage 1.0 \
  --commission 2.40 \
  > reports/run_log.txt 2>&1

# The output includes git commit hash and config checksum
```

Results should be deterministic given:
- Same data (cached in `data/raw/`)
- Same config
- Same codebase version

## Caveats

### Data Quality
- IB historical data may have gaps during outages
- Volume data for MES is less liquid than ES
- Pre-market/post-market data may be sparse

### Execution Realism
- Slippage model is simplified (constant tick slippage)
- No queue position modeling for limit orders
- Partial fills assumed at end of bar

### Strategy Limitations
- No market impact modeling
- Assumes sufficient liquidity
- Does not simulate order book dynamics

### SPY Proxy Mode
- No overnight session
- Different tick size/volatility
- Results clearly labeled but may not transfer to futures

## Troubleshooting

### "No data available"
```bash
# Check IB connection
python scripts/check_ib_status.sh

# Use proxy mode as fallback
python -m backtest.run --proxy-mode
```

### "Too many requests"
```bash
# IB pacing limits - the downloader handles this automatically
# If persisting, wait 10 minutes and retry
```

### Memory issues with 2-year 1m data
```bash
# ~500k bars for 2 years of 1m data
# Recommended: 16GB RAM
# Or use 5m bars for longer periods:
python -m backtest.run --bar 5m
```

## Example Analysis

```python
# Load results programmatically
import json
import pandas as pd

with open("reports/backtest_MES_2024-01-09_2026-01-09_metrics.json") as f:
    metrics = json.load(f)

trades = pd.read_csv("reports/backtest_MES_2024-01-09_2026-01-09_trades.csv")

# Filter by regime
high_vol_trades = trades[trades["atr"] > 6]
print(f"High vol win rate: {high_vol_trades['pnl'].gt(0).mean():.1%}")
```

## See Also

- [Strategy Documentation](./strategies.md) - Signal generation logic
- [Risk Gate Documentation](./risk.md) - Risk guardrails
- [Live Trading Setup](./live_trading.md) - Production deployment
