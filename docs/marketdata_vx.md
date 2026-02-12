# VX Futures Feed - Real-Time VIX Market Data

## Overview

The VX Futures Feed provides real-time VIX (CBOE Volatility Index) futures prices from Interactive Brokers to use as a volatility/risk-off factor in MES/ES trading decisions. When VIX is elevated, the system automatically reduces position sizes to be more conservative.

## VX vs VIX - What's the Difference?

| Term | Description | Source |
|------|-------------|--------|
| **VIX** | CBOE Volatility Index (spot) | Calculated from S&P 500 options, not directly tradable |
| **VX** | VIX Futures | Traded on CBOE Futures Exchange (CFE), forward-looking |

**Why VX is better for trading:**
1. **Real-time streaming** - Direct from IBKR vs polling Yahoo Finance
2. **Forward-looking** - VX prices in expected future volatility
3. **Market-based** - Actual money at stake, more reliable signal
4. **Lower latency** - Critical for intraday trading decisions

## How VX Integrates Into the Trading System

The VX feed is integrated at **three levels**:

### 1. Confidence Scaling (SignalProcessor)

The feed calculates a **volatility multiplier** applied to signal confidence:

| VX Level | Multiplier | Description |
|----------|------------|-------------|
| VX ≥ 30  | 0.4x       | **Extreme fear** - Crisis/panic mode, significantly reduce positions |
| VX ≥ 20  | 0.7x       | **Elevated** - Uncertainty in markets, be cautious |
| VX < 20  | 1.0x       | **Normal** - Standard position sizing |

```
Base Signal: BUY with 0.75 confidence
Current VX: 25.50 (elevated)
VX Multiplier: 0.7

Final Confidence: 0.75 × 0.7 = 0.525
```

### 2. Volatility Regime (HybridRAGPipeline)

VX is injected into the features DataFrame and used by the Rule Engine to **override volatility regime**:

- If VX ≥ 25 and ATR-based regime is MEDIUM → Upgraded to **HIGH**
- If VX ≥ 20 and ATR-based regime is LOW → Upgraded to **MEDIUM**

This catches market-wide fear even when MES/ES ATR appears normal.

### 3. Sentiment Aggregation (SentimentAggregator)

The `get_vix_sentiment()` function uses VX in this priority:
1. **VX Futures Feed** (real-time IBKR data) - most accurate
2. **Cache** (if VX feed unavailable and cache fresh)
3. **Yahoo Finance** (fallback for spot VIX)

VIX sentiment score mapping:
- VIX < 12: +0.3 (complacency, watch for reversal)
- VIX 12-15: +0.5 (bullish, low fear)
- VIX 15-20: 0.0 (neutral)
- VIX 20-25: -0.3 (elevated fear)
- VIX 25-30: -0.5 (high fear)
- VIX > 30: -0.6 (extreme fear, contrarian opportunities)

## Requirements

### IBKR Market Data Subscription

You need the **CFE Enhanced (NP,L1)** market data subscription from Interactive Brokers to receive live VX futures data.

To check/add this subscription:
1. Log into IBKR Account Management
2. Go to Settings → User Settings → Market Data Subscriptions
3. Search for "CFE" and enable "CFE Enhanced (NP,L1)"

**Note:** Without this subscription, the feed will receive delayed data or fail to connect.

### IB Gateway/TWS Configuration

Ensure your IB Gateway or TWS is configured:
1. Enable "ActiveX and Socket Clients"
2. Add 127.0.0.1 to Trusted IPs
3. Use port 7497 for paper trading or 7496 for live trading

## Configuration

Add the following to your `config.yaml`:

```yaml
vix_feed:
  enabled: true  # Enable VX futures feed from IBKR
  
  # Connection settings (uses separate client_id to avoid conflicts)
  ib_host: "127.0.0.1"
  ib_port: 7497           # 7497=paper, 7496=live
  client_id: 71           # Unique client ID for VX feed
  market_data_type: 1     # 1=live, 3=delayed
  
  # Stale data handling
  stale_seconds: 120               # Data considered stale after 2 minutes
  conservative_on_stale: false     # If true, use 0.5 multiplier when stale
  
  # Volatility multiplier thresholds
  thresholds:
    extreme: 30.0          # VX >= 30: 40% of normal size
    elevated: 20.0         # VX >= 20: 70% of normal size
  
  # Reconnection settings
  max_retries: 5
  base_delay: 1.0
  max_delay: 60.0
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `enabled` | `true` | Enable/disable VX feed |
| `ib_host` | `127.0.0.1` | IB Gateway host |
| `ib_port` | `7497` | IB Gateway port (7497=paper, 7496=live) |
| `client_id` | `71` | Unique client ID (avoid conflicts with main connection) |
| `market_data_type` | `1` | 1=live, 3=delayed |
| `stale_seconds` | `120` | Seconds before data is considered stale |
| `conservative_on_stale` | `false` | Use 0.5 multiplier when data is stale |
| `thresholds.extreme` | `30.0` | VX level for extreme fear (0.4x) |
| `thresholds.elevated` | `20.0` | VX level for elevated volatility (0.7x) |

## Usage

### Automatic Integration

When enabled, the VX feed is automatically integrated into the signal processing pipeline. You'll see log messages like:

```
📈 VX FUTURES FEED INITIALIZATION
======================================================================
   IBKR Connection: 127.0.0.1:7497 (client_id=71)
   Market Data Type: Live
   Stale Threshold: 120s
   Volatility Thresholds:
      VX >= 30.0: 0.4x multiplier (extreme fear)
      VX >= 20.0: 0.7x multiplier (elevated)
      VX < 20.0: 1.0x multiplier (normal)
   ✅ VX Feed background thread started
======================================================================
```

When VX affects a trade decision:

```
📈 VX Volatility Adjustment: VX=25.50 → 0.7x multiplier
📈 VX Volatility Scaling: 0.750 × 0.7 = 0.525
```

### Programmatic Usage

```python
from shree.data.vx_futures_feed import VxFuturesFeed, init_vx_feed, get_vx_feed

# Initialize (typically done automatically by SignalProcessor)
vx_feed = init_vx_feed(
    host="127.0.0.1",
    port=7497,
    client_id=71,
)
vx_feed.start_in_background()

# Get current state
latest = vx_feed.get_latest()
# Returns: {
#     "price": 22.50,
#     "last_update": "2026-01-09T10:30:00",
#     "contract": "VXH6",
#     "is_stale": False,
#     "multiplier": 0.7,
#     "error_count": 0,
#     "connected": True
# }

# Get just the multiplier
multiplier = vx_feed.get_volatility_multiplier()  # 0.7

# Get just the price
price = vx_feed.get_vx_price()  # 22.50

# Check if data is stale
is_stale = vx_feed.is_stale()  # False

# Shutdown
vx_feed.stop_background()
```

### Using the Global Singleton

```python
from shree.data.vx_futures_feed import get_vx_feed, shutdown_vx_feed

# Get the global instance (after initialization)
vx_feed = get_vx_feed()
if vx_feed:
    multiplier = vx_feed.get_volatility_multiplier()

# Shutdown on application exit
shutdown_vx_feed()
```

## Contract Selection

The feed automatically selects the **front-month VX contract**:

1. Queries all VX futures contracts via `reqContractDetails`
2. Filters for non-expired contracts (`lastTradeDateOrContractMonth >= today`)
3. Sorts by expiration date
4. Selects the nearest expiry (front month)

Contract rolls happen automatically when a new front month becomes available.

## Troubleshooting

### "VX Feed: No VX contracts found"

- Check that IB Gateway/TWS is running and logged in
- Verify CFE market data subscription is active
- Ensure you're using the correct port (7497 paper / 7496 live)

### "VX Feed connection timeout"

- IB Gateway may be in a bad state - restart it
- Check that no other applications are using the same client_id
- Verify network connectivity to IB Gateway

### Data is always stale

- CFE Enhanced subscription may not be active
- Market may be closed (VX trades ~23 hours on weekdays)
- Try setting `market_data_type: 3` for delayed data

### Multiplier always returns 1.0

- VX feed may not be connected (check logs for errors)
- VIX may genuinely be below 20
- Data may be stale (check `is_stale()`)

## Monitoring

### Log Messages

Watch for these log messages:

| Message | Meaning |
|---------|---------|
| `✅ VX Feed background thread started` | Successfully connected |
| `VX Feed: Selected front-month contract: VXH6` | Contract identified |
| `📈 VX Volatility Adjustment: VX=25.50 → 0.7x` | Multiplier applied |
| `⚠️ VX Feed: Data is stale` | No updates for >120s |

### Telegram Alerts

When VX multiplier affects a trade, it's included in the Telegram alert metadata:

```
🤖 Signal: BUY MES
📊 Confidence: 0.525 (VX adj: 0.7x)
📈 VX: 25.50
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Signal Processor                         │
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐ │
│  │   Hybrid    │───▶│  Sentiment  │───▶│  VX Multiplier  │ │
│  │  Pipeline   │    │  Modifier   │    │    (NEW)        │ │
│  └─────────────┘    └─────────────┘    └─────────────────┘ │
│         │                  │                   │           │
│         ▼                  ▼                   ▼           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Final Signal Confidence                │   │
│  │  base_conf × sentiment_mod × low_vol × vx_mult     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    VxFuturesFeed                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Background  │  │   Thread-    │  │   Multiplier     │  │
│  │   Thread     │──│   Safe       │──│   Calculator     │  │
│  │  (asyncio)   │  │   State      │  │                  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              IB Gateway / TWS                        │  │
│  │  - reqContractDetails (front month selection)        │  │
│  │  - reqMktData (real-time price subscription)         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Related Files

- `shree/data/vx_futures_feed.py` - VX feed implementation
- `shree/execution/components/signal_processor.py` - Integration point
- `config.yaml` - Configuration (`vix_feed` section)
- `tests/test_vx_futures_feed.py` - Unit tests
