# Data Provider Options for MyTrader

## Overview

Your system is **already designed** to separate data collection from order execution! This architecture allows you to:

- ✅ Use **TradingView** (or others) for market data
- ✅ Use **IBKR** only for order execution
- ✅ Avoid IBKR rate limits
- ✅ Reduce costs
- ✅ Increase reliability

---

## Current Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA LAYER (Multiple Sources)                 │
├─────────────────────────────────────────────────────────────────┤
│  TradingView │ Polygon.io │ Alpha Vantage │ Yahoo Finance        │
│  (Primary)   │  (Backup)  │  (Alternative)│  (Free Fallback)    │
└──────────────┴────────────┴───────────────┴─────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  MarketDataPipeline │
                    │  (Combines Sources) │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Strategy Engine    │
                    │  (Generate Signals) │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Risk Manager      │
                    │   (Validate Trade)  │
                    └──────────┬──────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EXECUTION LAYER (IBKR Only)                   │
├─────────────────────────────────────────────────────────────────┤
│                    Interactive Brokers                           │
│                    (Orders & Fills Only)                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Recommended Data Providers

### 1. **TradingView** (Best for Retail)

**Pros:**
- ✅ Excellent charting and webhooks
- ✅ Real-time data for most markets
- ✅ Webhook alerts (can trigger your bot)
- ✅ Affordable ($15-60/month)

**Cons:**
- ❌ Requires TradingView subscription for real-time
- ❌ Rate limits on API calls

**Setup:**
```yaml
# config.yaml
data:
  tradingview_webhook_url: "http://your-server.com:8000/tradingview"
  tradingview_symbol: "ES1!"  # E-mini S&P 500
  tradingview_interval: "1m"
```

**How to use:**
- Set up webhook alerts in TradingView
- Webhook sends data to your server
- Your bot processes and trades

---

### 2. **Polygon.io** (Best for Professional)

**Pros:**
- ✅ Real-time and historical data
- ✅ WebSocket streaming
- ✅ High rate limits
- ✅ Comprehensive coverage (stocks, options, crypto, forex)
- ✅ Free tier available (delayed data)

**Cons:**
- ❌ Real-time data costs $99+/month
- ❌ More complex integration

**Pricing:**
- Free: 5 API calls/minute (15-min delayed)
- Starter: $99/month (real-time, 100 calls/min)
- Developer: $249/month (unlimited)

**API Example:**
```python
import requests

url = "https://api.polygon.io/v2/aggs/ticker/ES/range/1/minute/2024-01-09/2024-01-09"
params = {"apiKey": "YOUR_API_KEY"}
response = requests.get(url, params=params)
data = response.json()
```

---

### 3. **Alpha Vantage** (Good Free Option)

**Pros:**
- ✅ **Free tier** (500 calls/day)
- ✅ No credit card required
- ✅ Easy to use
- ✅ Good for stocks, forex, crypto

**Cons:**
- ❌ Rate limits (5 calls/minute on free tier)
- ❌ Limited real-time features
- ❌ No futures data

**API Example:**
```python
import requests

url = "https://www.alphavantage.co/query"
params = {
    "function": "TIME_SERIES_INTRADAY",
    "symbol": "SPY",
    "interval": "1min",
    "apikey": "YOUR_API_KEY"
}
response = requests.get(url, params=params)
```

---

### 4. **Yahoo Finance** (Free Fallback)

**Pros:**
- ✅ Completely free
- ✅ No API key required
- ✅ Simple to use
- ✅ Good coverage

**Cons:**
- ❌ Unofficial API (can break)
- ❌ Rate limits not documented
- ❌ No official support
- ❌ Delayed data

**Library:**
```python
import yfinance as yf

# Download 1-minute data
data = yf.download("ES=F", period="1d", interval="1m")
```

---

### 5. **Finnhub** (Good Alternative)

**Pros:**
- ✅ Free tier (60 calls/minute)
- ✅ WebSocket streaming
- ✅ Real-time data
- ✅ Good documentation

**Cons:**
- ❌ Free tier has limitations
- ❌ Premium is $50+/month

---

## Implementation Guide

### Option A: Use TradingView + IBKR (Recommended for You)

**Step 1: Set up TradingView webhook endpoint**

The webhook receiver is already in your `dashboard_api.py`:

```python
@app.post("/tradingview")
async def tradingview_webhook(data: dict):
    """Receive TradingView webhook alerts"""
    # Process the alert and trigger trades
    pass
```

**Step 2: Create TradingView alerts**

In TradingView:
1. Add your indicators (RSI, MACD)
2. Set up alert conditions
3. Configure webhook URL: `http://your-ip:8000/tradingview`
4. Set message format:
```json
{
  "symbol": "{{ticker}}",
  "price": {{close}},
  "time": "{{time}}",
  "signal": "{{strategy.order.action}}"
}
```

**Step 3: Update main.py to use TradingView instead of IBKR for data**

```python
# Instead of IBKRCollector for data:
collectors = [
    TradingViewCollector(
        base_url="http://localhost:8000",
        symbol="ES1!",
        interval="1m"
    ),
    # Keep IBKR only for execution
]
```

---

### Option B: Use Polygon.io + IBKR (Best Professional Setup)

**Step 1: Create Polygon.io collector**

Create `mytrader/data/polygon.py`:

```python
"""Polygon.io real-time data integration."""
import asyncio
import json
from typing import AsyncIterator
import websockets
import pandas as pd
from .base import DataCollector

class PolygonCollector(DataCollector):
    def __init__(self, api_key: str, symbol: str):
        self.api_key = api_key
        self.symbol = symbol
        self.ws_url = f"wss://socket.polygon.io/stocks"
    
    async def collect(self) -> pd.DataFrame:
        # Implementation for historical data
        pass
    
    async def stream(self) -> AsyncIterator[dict]:
        async with websockets.connect(self.ws_url) as ws:
            # Authenticate
            await ws.send(json.dumps({"action": "auth", "params": self.api_key}))
            
            # Subscribe to symbol
            await ws.send(json.dumps({
                "action": "subscribe",
                "params": f"A.{self.symbol}"  # Aggregate bars
            }))
            
            async for message in ws:
                data = json.loads(message)
                if data[0]["ev"] == "A":  # Aggregate bar
                    yield {
                        "timestamp": pd.to_datetime(data[0]["s"], unit="ms"),
                        "open": data[0]["o"],
                        "high": data[0]["h"],
                        "low": data[0]["l"],
                        "close": data[0]["c"],
                        "volume": data[0]["v"],
                        "source": "polygon"
                    }
```

**Step 2: Update config.yaml**

```yaml
data:
  polygon_api_key: "YOUR_API_KEY"
  polygon_symbol: "ES"
  
  # Still keep IBKR for execution only
  ibkr_host: "127.0.0.1"
  ibkr_port: 4002
```

**Step 3: Update main.py**

```python
from mytrader.data.polygon import PolygonCollector

collectors = [
    PolygonCollector(
        api_key=settings.data.polygon_api_key,
        symbol=settings.data.polygon_symbol
    ),
]

# IBKR used ONLY for execution
executor = TradeExecutor(ib, settings.trading, settings.data.ibkr_symbol)
```

---

### Option C: Multiple Data Sources (Most Reliable)

Use a **fallback chain** for maximum reliability:

```python
collectors = [
    # Primary: TradingView
    TradingViewCollector(base_url="http://localhost:8000", symbol="ES1!"),
    
    # Backup: Polygon.io
    PolygonCollector(api_key=polygon_key, symbol="ES"),
    
    # Fallback: Yahoo Finance
    YahooFinanceCollector(symbol="ES=F"),
]

pipeline = MarketDataPipeline(collectors)
```

The pipeline automatically combines data from all sources and handles failures.

---

## Updated Configuration Example

Here's a complete `config.yaml` for TradingView + IBKR setup:

```yaml
# Data Collection (TradingView)
data:
  # TradingView for market data
  tradingview_webhook_url: "http://localhost:8000/tradingview"
  tradingview_symbol: "ES1!"
  tradingview_interval: "1m"
  
  # IBKR for execution ONLY
  ibkr_host: "127.0.0.1"
  ibkr_port: 4002
  ibkr_client_id: 1
  ibkr_symbol: "ES"
  ibkr_exchange: "GLOBEX"
  ibkr_currency: "USD"
  
  # Optional: Sentiment (if using)
  twitter_bearer_token: ""
  news_api_keys: []

# Trading settings (unchanged)
trading:
  max_position_size: 2
  max_daily_loss: 1500.0
  max_daily_trades: 20
  initial_capital: 100000.0
  stop_loss_ticks: 20.0
  take_profit_ticks: 40.0

# Strategy (unchanged)
strategies:
  - name: rsi_macd_sentiment
    enabled: true
    params:
      rsi_buy: 35.84
      rsi_sell: 54.83
```

---

## Cost Comparison

| Provider | Free Tier | Real-Time Cost | Best For |
|----------|-----------|----------------|----------|
| **TradingView** | Limited | $15-60/month | Retail traders |
| **Polygon.io** | 5 calls/min | $99-249/month | Professionals |
| **Alpha Vantage** | 500/day | $50/month | Light usage |
| **Yahoo Finance** | Unlimited | Free | Fallback/testing |
| **Finnhub** | 60/min | $50/month | Alternative |
| **IBKR Data** | Complex limits | Included* | Not recommended |

*IBKR data is "free" but has strict rate limits and requires market data subscriptions.

---

## Recommended Setup for You

Based on your needs, I recommend:

### **TradingView + IBKR** (Best bang for buck)

**Monthly Cost:** ~$30
- TradingView Essential: $15/month
- IBKR: Free for execution

**Why:**
1. ✅ Real-time data without IBKR rate limits
2. ✅ Excellent charting for analysis
3. ✅ Webhook alerts (automated)
4. ✅ Can test strategies visually
5. ✅ Affordable for retail

**Setup Time:** 1-2 hours

---

## Quick Migration Steps

### 1. Keep Current Setup (IBKR) for Testing
```bash
# No changes needed, works as-is
./start.sh
```

### 2. Add TradingView (Parallel)
```python
# In main.py, add TradingView to collectors
collectors = [
    TradingViewCollector(...),  # New
    IBKRCollector(...),          # Keep as fallback
]
```

### 3. Test Both Sources
```bash
# Monitor which source is providing data
tail -f logs/backend.log | grep "source"
```

### 4. Remove IBKR Data (Keep Execution)
```python
# Only use IBKR for execution
executor = TradeExecutor(ib, settings.trading, "ES")

# Use TradingView for all market data
collectors = [TradingViewCollector(...)]
```

---

## Benefits of This Approach

### Cost Savings
- **Before:** IBKR data fees + execution
- **After:** TradingView $15/month + execution (free)
- **Savings:** $50-100/month

### Reliability
- No rate limit issues from IBKR
- Can add multiple data sources as fallbacks
- Execution stays fast and reliable

### Performance
- Lower latency (direct data feed)
- Less load on IBKR connection
- Can scale to more symbols

### Flexibility
- Easy to switch providers
- Can compare data sources
- Test strategies on multiple feeds

---

## Next Steps

1. **Choose your data provider** (I recommend TradingView)
2. **Sign up and get API key/webhook URL**
3. **Test data feed** separately
4. **Integrate with existing system**
5. **Monitor for a few days**
6. **Remove IBKR data collection** (keep execution)

Would you like me to:
1. Implement Polygon.io collector?
2. Set up TradingView webhook handler?
3. Create Yahoo Finance fallback?
4. Build a multi-source pipeline?

Let me know which data provider you prefer, and I'll help you set it up! 🚀
