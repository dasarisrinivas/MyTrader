# MyTrader - Automated Trading System

## Overview

MyTrader is an automated trading system that connects to Interactive Brokers (IBKR) to execute trades based on technical indicators and sentiment analysis. The system uses a combination of RSI, MACD, and sentiment signals to make trading decisions on E-mini S&P 500 futures (ES).

---

## What Happens When You Click "Start Trading"?

### 🚀 High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     USER CLICKS "START TRADING"                          │
│                     (Dashboard Web Interface)                            │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        PRE-FLIGHT CHECKS                                 │
│  ✓ IBKR connection configuration validated                              │
│  ✓ Risk limits configured (max loss, position size, etc.)               │
│  ✓ Strategy configuration loaded (RSI/MACD/Sentiment)                   │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   COMPONENT INITIALIZATION                               │
│  • IBKR Data Collector (connects to IB Gateway)                         │
│  • Strategy Engine (loads trading strategies)                           │
│  • Risk Manager (enforces position/loss limits)                         │
│  • Trade Executor (places orders)                                       │
│  • Performance Tracker (monitors P&L)                                   │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    MARKET DATA STREAM STARTS                             │
│  • Subscribe to real-time ES futures bars (5-second bars)               │
│  • Collect OHLCV data continuously                                      │
│  • Build historical context (last 100 bars minimum)                     │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     TRADING LOOP BEGINS                                  │
│                     (Runs Continuously)                                  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                   ┌─────────────┴─────────────┐
                   │                           │
                   ▼                           ▼
    ┌──────────────────────────┐  ┌──────────────────────────┐
    │   NEW BAR RECEIVED       │  │  PERIODIC UPDATES         │
    │   (Every 5 seconds)      │  │  (Every 5 minutes)        │
    └──────────┬───────────────┘  └──────────┬───────────────┘
               │                              │
               ▼                              ▼
    ┌──────────────────────────┐  ┌──────────────────────────┐
    │  FEATURE ENGINEERING     │  │  STATUS LOGGING          │
    │  • Calculate RSI         │  │  • Current P&L           │
    │  • Calculate MACD        │  │  • Win Rate              │
    │  • Calculate ATR         │  │  • Sharpe Ratio          │
    │  • Add Sentiment         │  │  • Drawdown              │
    └──────────┬───────────────┘  └──────────────────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │  STRATEGY EVALUATION     │
    │  (RSI MACD Sentiment)    │
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │    SIGNAL GENERATION     │
    │    BUY / SELL / HOLD     │
    └──────────┬───────────────┘
               │
               ▼
         ┌─────┴─────┐
         │  HOLD?    │────── YES ──────► Continue monitoring
         └─────┬─────┘
               │ NO (BUY or SELL)
               ▼
    ┌──────────────────────────┐
    │   RISK MANAGEMENT        │
    │   • Check daily loss     │
    │   • Check max trades     │
    │   • Check position size  │
    │   • Can we trade?        │
    └──────────┬───────────────┘
               │
         ┌─────┴─────┐
         │  PASSED?  │────── NO ──────► Skip trade, continue
         └─────┬─────┘
               │ YES
               ▼
    ┌──────────────────────────┐
    │  POSITION SIZING         │
    │  • Calculate contracts   │
    │  • Based on confidence   │
    │  • Apply Kelly Criterion │
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │  CALCULATE STOPS         │
    │  • Stop Loss (ATR-based) │
    │  • Take Profit           │
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │   EXECUTE ORDER          │
    │   • Send to IBKR         │
    │   • Bracket order        │
    │   • Wait for fill        │
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │   RECORD TRADE           │
    │   • Update tracker       │
    │   • Log to history       │
    │   • Update dashboard     │
    └──────────┬───────────────┘
               │
               ▼
         Continue Loop ──────────────┐
                                     │
                                     ▼
                        ┌──────────────────────┐
                        │  USER CLICKS STOP    │
                        │  or Ctrl+C           │
                        └──────────┬───────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │  GRACEFUL SHUTDOWN   │
                        │  • Cancel orders     │
                        │  • Save report       │
                        │  • Disconnect IBKR   │
                        └──────────────────────┘
```

---

## 📊 Trading Strategy: RSI MACD Sentiment

### Strategy Components

The system uses a **combined signal approach** with three main indicators:

#### 1. **RSI (Relative Strength Index)**
- **Period**: 14 bars
- **Buy Signal**: RSI < 35.84 (oversold condition)
- **Sell Signal**: RSI > 54.83 (overbought condition)
- **Purpose**: Identifies momentum reversals

#### 2. **MACD (Moving Average Convergence Divergence)**
- **Fast Period**: 12
- **Slow Period**: 26
- **Signal Period**: 9
- **Buy Signal**: MACD crosses above signal line (bullish)
- **Sell Signal**: MACD crosses below signal line (bearish)
- **Purpose**: Confirms trend direction

#### 3. **Sentiment Analysis** (Optional)
- **Sources**: Twitter, News APIs
- **Range**: -1.0 (very negative) to +1.0 (very positive)
- **Buy Threshold**: Sentiment > -0.82 (not too negative)
- **Sell Threshold**: Sentiment < 0.22 (not too positive)
- **Purpose**: Filters out trades during extreme sentiment

---

## 🎯 When Does the System BUY?

The system generates a **BUY signal** when:

```
✅ RSI < 35.84 (Market is oversold)
AND
✅ MACD crosses above Signal Line (Bullish momentum)
AND
✅ Sentiment > -0.82 (Not extremely bearish)
AND
✅ Risk checks pass:
   • Daily loss limit not exceeded
   • Maximum daily trades not reached
   • Position size within limits
```

**What happens on BUY:**
1. Calculate position size (1-2 contracts based on confidence)
2. Calculate entry price (current market price)
3. Set stop-loss: 20 ticks below entry (approximately $250 risk per contract)
4. Set take-profit: 40 ticks above entry (approximately $500 profit per contract)
5. Place bracket order to IBKR
6. Wait for fill confirmation
7. Monitor position until exit

---

## 🎯 When Does the System SELL?

The system generates a **SELL signal** when:

```
✅ RSI > 54.83 (Market is overbought)
AND
✅ MACD crosses below Signal Line (Bearish momentum)
AND
✅ Sentiment < 0.22 (Not extremely bullish)
AND
✅ Risk checks pass
```

**What happens on SELL:**
1. Calculate position size (1-2 contracts based on confidence)
2. Calculate entry price (current market price)
3. Set stop-loss: 20 ticks above entry (risk protection)
4. Set take-profit: 40 ticks below entry (profit target)
5. Place bracket order to IBKR
6. Wait for fill confirmation
7. Monitor position until exit

---

## 💰 Position Exit Conditions

Positions are automatically closed when:

1. **Stop Loss Hit**: Price moves against you by 20 ticks
   - Limits loss to ~$250 per contract
   
2. **Take Profit Hit**: Price moves in your favor by 40 ticks
   - Locks in profit of ~$500 per contract
   
3. **Reverse Signal**: Strategy generates opposite signal
   - System may flatten position and reverse
   
4. **Daily Loss Limit**: Total daily loss exceeds $1,500
   - All positions closed, trading stops for the day
   
5. **Manual Stop**: User clicks "Stop Trading"
   - Graceful shutdown, all positions closed

---

## 🛡️ Risk Management

### Position Sizing
- **Method**: Kelly Criterion (optional) or Fixed Size
- **Max Position**: 2 contracts (configurable)
- **Based On**: Signal confidence (0.0 - 1.0)

### Risk Limits
- **Max Daily Loss**: $1,500
- **Max Daily Trades**: 20
- **Stop Loss**: 20 ticks ($250 per contract)
- **Take Profit**: 40 ticks ($500 per contract)
- **Risk/Reward Ratio**: 1:2

### Dynamic Stops (ATR-Based)
If ATR (Average True Range) is available:
- **Stop Distance**: 2.0 × ATR
- **Adjusts to market volatility**
- More room in volatile markets, tighter in calm markets

---

## 📈 Performance Tracking

The system continuously monitors:

- **Equity Curve**: Real-time account value
- **P&L**: Realized and unrealized profit/loss
- **Win Rate**: Percentage of winning trades
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Daily Performance**: Today's profit/loss
- **Trade History**: All executed trades with timestamps

All metrics are displayed live on the dashboard and updated every 5 seconds.

---

## 🔧 Configuration

### Key Settings (config.yaml)

```yaml
trading:
  max_position_size: 2          # Maximum contracts per trade
  max_daily_loss: 1500.0        # Stop trading if loss exceeds this
  max_daily_trades: 20          # Maximum trades per day
  stop_loss_ticks: 20.0         # Distance to stop loss
  take_profit_ticks: 40.0       # Distance to take profit

strategies:
  - name: rsi_macd_sentiment
    enabled: true
    params:
      rsi_buy: 35.84              # RSI threshold for buying
      rsi_sell: 54.83             # RSI threshold for selling
      sentiment_buy: -0.82        # Sentiment floor for buying
      sentiment_sell: 0.22        # Sentiment ceiling for selling
```

---

## 🚦 Getting Started

### Prerequisites
1. **IB Gateway** running on port 4002 (paper trading)
2. **Node.js** and **Python 3.12** installed
3. **Virtual environment** activated

### Start the System

```bash
./start.sh
```

This will:
1. Start the FastAPI backend (port 8000)
2. Start the React frontend (port 5173)
3. Open your browser automatically

### Start Trading

1. Navigate to `http://localhost:5173`
2. Click **"Start Trading"** button
3. Monitor the dashboard for:
   - Live P&L
   - Trade signals
   - Position status
   - Performance metrics

### Stop Trading

- Click **"Stop Trading"** button, or
- Press `Ctrl+C` in terminal, or
- Run `./stop.sh`

---

## 📁 Project Structure

```
MyTrader/
├── config.yaml                    # Main configuration
├── start.sh                       # Start all services
├── stop.sh                        # Stop all services
├── dashboard/
│   ├── backend/
│   │   └── dashboard_api.py       # FastAPI REST API
│   └── frontend/
│       └── src/
│           └── components/        # React dashboard
├── mytrader/
│   ├── strategies/
│   │   └── rsi_macd_sentiment.py  # Main trading strategy
│   ├── data/
│   │   └── ibkr.py                # IBKR data collector
│   ├── execution/
│   │   └── ib_executor.py         # Order execution
│   ├── risk/
│   │   └── manager.py             # Risk management
│   └── monitoring/
│       └── live_tracker.py        # Performance tracking
└── scripts/
    └── paper_trade.py             # Paper trading session manager
```

---

## 📊 Example Trade Flow

### Real Example

```
Time: 23:30:00
Price: $4,950.00 (ES futures)
RSI: 32.5 (oversold)
MACD: Bullish crossover
Sentiment: -0.5 (neutral-negative)

✅ BUY SIGNAL GENERATED

Risk Check:
✓ Daily loss: $-200 (below $1,500 limit)
✓ Trades today: 8 (below 20 limit)
✓ Position size: 0 (can add 2 contracts)

Position Sizing:
• Signal confidence: 0.75
• Contracts: 2

Order Placement:
• Entry: $4,950.00
• Stop Loss: $4,945.00 (20 ticks = $250 risk)
• Take Profit: $4,960.00 (40 ticks = $500 profit)

✅ ORDER FILLED: Long 2 contracts @ $4,950.00

...monitoring position...

Time: 23:45:00
Price: $4,960.00

🎯 TAKE PROFIT HIT!
✅ Position closed: +$500 profit (2 contracts × $250 gain)

Trade recorded:
• P&L: +$500
• Win rate: 65%
• Total trades: 9
```

---

## ⚠️ Important Notes

### Risk Warnings
- This system trades real money in paper trading mode
- Always verify you're connected to **paper trading** account
- Never run on live account without extensive testing
- Past performance does not guarantee future results

### System Requirements
- Stable internet connection
- IB Gateway must remain running
- Sufficient margin in IBKR account
- Market hours: ES futures trade nearly 24/5

### Troubleshooting
- If dashboard shows errors: Check backend logs at `logs/backend.log`
- If trades not executing: Verify IB Gateway connection on port 4002
- If pre-flight checks fail: Review configuration in `config.yaml`

---

## 📞 Support

For issues or questions:
1. Check `logs/backend.log` for detailed error messages
2. Review configuration in `config.yaml`
3. Ensure IB Gateway is running and connected

---

## 📜 License

This is a proprietary trading system. Use at your own risk.

**Last Updated**: November 1, 2025
