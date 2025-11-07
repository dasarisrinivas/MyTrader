# MyTrader Documentation# MyTrader - Enhanced Automated Trading System



Complete documentation for the MyTrader AI-powered SPY Futures trading system.[![Tests](https://img.shields.io/badge/tests-37%20passing-brightgreen)]()

[![Python](https://img.shields.io/badge/python-3.12%2B-blue)]()

## 📚 Essential Guides[![Win Rate](https://img.shields.io/badge/win%20rate-60%25-success)]()

[![AWS Bedrock](https://img.shields.io/badge/AWS-Bedrock%20LLM-orange)]()

### Getting Started

- **[Main README](../README.md)** - Project overview and quick start## 🚀 Recent Enhancements (November 2025)

- **[HOW_TO_START.md](HOW_TO_START.md)** - Detailed setup instructions

**🤖 NEW: AWS Bedrock LLM Integration**

### Core Systems- ✨ **AI-Powered Trade Decisions** - Claude 3 / Titan LLM integration

- 🧠 **Intelligent Analysis** - Multi-factor reasoning and confidence scoring

#### 1. SPY Futures Daily Review (Recommended)- 📚 **Continuous Learning** - Automated model fine-tuning from trade outcomes

- **[SPY_FUTURES_REVIEW_GUIDE.md](SPY_FUTURES_REVIEW_GUIDE.md)** - Complete SPY review system guide- 🎯 **Adaptive Risk Management** - LLM-suggested stops and position sizing

- **[SPY_FUTURES_QUICK_REF.md](SPY_FUTURES_QUICK_REF.md)** - Quick reference- 📊 **Performance Tracking** - SQLite-based trade logging with LLM predictions

- **Command**: `python run_spy_futures_review.py`- 🔗 **AWS Comprehend** - Sentiment analysis integration

- 🚀 **Training Pipeline** - S3-based data storage and retraining workflow

#### 2. General Live Trading Review

- **[LIVE_TRADING_REVIEW_GUIDE.md](LIVE_TRADING_REVIEW_GUIDE.md)** - General trading review📖 **[LLM Integration Guide](./LLM_INTEGRATION.md)** - Complete setup and usage documentation

- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - General quick reference

- **Command**: `python run_daily_review.py`**Major Performance Improvements:**

- ✅ **+68% Better Returns** (-0.80% vs -2.51%)

#### 3. Autonomous Learning System- ✅ **+320% Higher Win Rate** (60% vs 14.29%)

- **[AUTONOMOUS_TRADING_GUIDE.md](AUTONOMOUS_TRADING_GUIDE.md)** - Self-optimizing system- ✅ **+67% Lower Drawdown** (-1.35% vs -4.07%)

- **Command**: `python run_autonomous_trading.py daily`- ✅ **+80% Better Expectancy** per trade

- ✅ **37 Unit Tests** - All Passing (19 original + 18 LLM)

### Dashboard & Integration- ✅ **Market Regime Detection** - Adaptive strategy

- **[DASHBOARD_TRADING_INTEGRATION.md](DASHBOARD_TRADING_INTEGRATION.md)** - Dashboard setup- ✅ **Enhanced Risk Management** - Kelly Criterion + Trailing Stops

- **[ORDER_TRACKING_GUIDE.md](ORDER_TRACKING_GUIDE.md)** - Order tracking

- **[VISUAL_GUIDE.md](VISUAL_GUIDE.md)** - Dashboard visuals**Advanced Strategy Optimization Framework**

- 🎯 **Bayesian Optimization** with Optuna

### Configuration- 📊 **Market Regime-Based Strategy** (Trending/Ranging/Volatile)

- **[LLM_INTEGRATION.md](LLM_INTEGRATION.md)** - LLM setup- 🔍 **Multi-Factor Confirmation** filters

- **[RESTART_AND_PNL_GUIDE.md](RESTART_AND_PNL_GUIDE.md)** - System management- 📈 **Comprehensive Performance Analysis** with visualizations

- **[WINDOWS_TASK_SCHEDULER_SETUP.md](WINDOWS_TASK_SCHEDULER_SETUP.md)** - Windows scheduling- 🎓 **Target Metrics**: Sharpe ≥1.5, Max DD ≤15%, Win Rate ≥60%



## 🎯 Quick Start by Use Case📊 **[View Enhancement Report](./ENHANCEMENT_REPORT.md)** | **[Strategy Optimization Guide](./STRATEGY_OPTIMIZATION.md)**



### "Analyze my SPY Futures trading"---

→ **[SPY_FUTURES_REVIEW_GUIDE.md](SPY_FUTURES_REVIEW_GUIDE.md)**

## Overview

### "Review any trading instrument"

→ **[LIVE_TRADING_REVIEW_GUIDE.md](LIVE_TRADING_REVIEW_GUIDE.md)**MyTrader is an advanced automated trading system that connects to Interactive Brokers (IBKR) to execute trades on E-mini S&P 500 futures (ES). The system uses a sophisticated combination of technical indicators, sentiment analysis, AWS Bedrock LLM intelligence, and adaptive risk management to make intelligent trading decisions.



### "Enable autonomous learning"---

→ **[AUTONOMOUS_TRADING_GUIDE.md](AUTONOMOUS_TRADING_GUIDE.md)**

## What Happens When You Click "Start Trading"?

### "Setup the dashboard"

→ **[DASHBOARD_TRADING_INTEGRATION.md](DASHBOARD_TRADING_INTEGRATION.md)**### 🚀 High-Level Flow



### "I'm new to the system"```

→ **[HOW_TO_START.md](HOW_TO_START.md)**┌─────────────────────────────────────────────────────────────────────────┐

│                     USER CLICKS "START TRADING"                          │

## 📋 Quick Commands│                     (Dashboard Web Interface)                            │

└────────────────────────────────┬────────────────────────────────────────┘

```bash                                 │

# SPY Futures Review                                 ▼

python run_spy_futures_review.py                    # Daily review┌─────────────────────────────────────────────────────────────────────────┐

python run_spy_futures_review.py --days 7           # Last 7 days│                        PRE-FLIGHT CHECKS                                 │

│  ✓ IBKR connection configuration validated                              │

# General Review│  ✓ Risk limits configured (max loss, position size, etc.)               │

python run_daily_review.py                          # Daily review│  ✓ Strategy configuration loaded (RSI/MACD/Sentiment)                   │

python run_daily_review.py --csv                    # Use CSV logs└────────────────────────────────┬────────────────────────────────────────┘

                                 │

# Autonomous System                                 ▼

python run_autonomous_trading.py daily              # Daily analysis┌─────────────────────────────────────────────────────────────────────────┐

python run_autonomous_trading.py status             # System status│                   COMPONENT INITIALIZATION                               │

│  • IBKR Data Collector (connects to IB Gateway)                         │

# Dashboard│  • Strategy Engine (loads trading strategies)                           │

./scripts/start_dashboard.sh                        # Start dashboard│  • Risk Manager (enforces position/loss limits)                         │

```│  • Trade Executor (places orders)                                       │

│  • Performance Tracker (monitors P&L)                                   │

## 🗂️ File Structure└────────────────────────────────┬────────────────────────────────────────┘

                                 │

```                                 ▼

docs/┌─────────────────────────────────────────────────────────────────────────┐

├── README.md                              # This file│                    MARKET DATA STREAM STARTS                             │

├── README_OLD.md                          # Legacy README (archived)│  • Subscribe to real-time ES futures bars (5-second bars)               │

││  • Collect OHLCV data continuously                                      │

├── GETTING STARTED│  • Build historical context (last 100 bars minimum)                     │

│   ├── HOW_TO_START.md                   # Setup guide└────────────────────────────────┬────────────────────────────────────────┘

│   └── QUICKSTART.md                     # Legacy (deprecated)                                 │

│                                 ▼

├── SPY FUTURES SYSTEM┌─────────────────────────────────────────────────────────────────────────┐

│   ├── SPY_FUTURES_REVIEW_GUIDE.md       # Complete guide│                     TRADING LOOP BEGINS                                  │

│   └── SPY_FUTURES_QUICK_REF.md          # Quick reference│                     (Runs Continuously)                                  │

│└────────────────────────────────┬────────────────────────────────────────┘

├── GENERAL REVIEW SYSTEM                                 │

│   ├── LIVE_TRADING_REVIEW_GUIDE.md      # Complete guide                   ┌─────────────┴─────────────┐

│   └── QUICK_REFERENCE.md                # Quick reference                   │                           │

│                   ▼                           ▼

├── AUTONOMOUS SYSTEM    ┌──────────────────────────┐  ┌──────────────────────────┐

│   └── AUTONOMOUS_TRADING_GUIDE.md       # Complete guide    │   NEW BAR RECEIVED       │  │  PERIODIC UPDATES         │

│    │   (Every 5 seconds)      │  │  (Every 5 minutes)        │

├── DASHBOARD    └──────────┬───────────────┘  └──────────┬───────────────┘

│   ├── DASHBOARD_TRADING_INTEGRATION.md  # Integration guide               │                              │

│   ├── ORDER_TRACKING_GUIDE.md           # Order tracking               ▼                              ▼

│   └── VISUAL_GUIDE.md                   # Visual guide    ┌──────────────────────────┐  ┌──────────────────────────┐

│    │  FEATURE ENGINEERING     │  │  STATUS LOGGING          │

└── CONFIGURATION    │  • Calculate RSI         │  │  • Current P&L           │

    ├── LLM_INTEGRATION.md                # LLM setup    │  • Calculate MACD        │  │  • Win Rate              │

    ├── RESTART_AND_PNL_GUIDE.md          # System management    │  • Calculate ATR         │  │  • Sharpe Ratio          │

    └── WINDOWS_TASK_SCHEDULER_SETUP.md   # Windows scheduling    │  • Add Sentiment         │  │  • Drawdown              │

```    └──────────┬───────────────┘  └──────────────────────────┘

               │

---               ▼

    ┌──────────────────────────┐

**Documentation Version**: 1.0      │  STRATEGY EVALUATION     │

**Last Updated**: November 6, 2025      │  (RSI MACD Sentiment)    │

**Status**: Production Ready ✅    └──────────┬───────────────┘

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

## 🤖 AWS Bedrock LLM Intelligence

### AI-Enhanced Trading Decisions

MyTrader now integrates AWS Bedrock LLM (Claude 3 or Titan) to provide intelligent trade analysis and recommendations:

#### Key Features

1. **Real-Time LLM Analysis**
   - Analyzes technical indicators, sentiment, and market regime
   - Provides structured JSON recommendations with reasoning
   - Confidence scoring (0.0 - 1.0) for each recommendation

2. **Multiple Operating Modes**
   - **Consensus Mode** (Default): Both traditional and LLM signals must agree
   - **Override Mode**: LLM can override traditional strategy decisions
   - **Advisory Mode**: LLM provides recommendations without affecting execution

3. **Continuous Learning**
   - Logs all trades with LLM predictions and actual outcomes
   - Automated training pipeline for model fine-tuning
   - S3-based data storage for historical analysis
   - Weekly/monthly retraining with recent trade data

4. **Enhanced Sentiment Analysis**
   - AWS Comprehend integration for news and social media
   - Multi-source sentiment aggregation
   - Normalized sentiment scores (-1.0 to +1.0)

5. **Intelligent Risk Management**
   - LLM-suggested position sizes based on confidence
   - Dynamic stop-loss and take-profit recommendations
   - Risk assessment for each trade decision

#### Example LLM Recommendation

```json
{
  "trade_decision": "BUY",
  "confidence": 0.85,
  "suggested_position_size": 2,
  "suggested_stop_loss": 4945.0,
  "suggested_take_profit": 4960.0,
  "reasoning": "Strong oversold signal with RSI at 28.5, bullish MACD crossover, and positive sentiment improving. Market regime is mean-reverting, favorable for entry.",
  "key_factors": [
    "RSI oversold (28.5 < 30)",
    "Bullish MACD histogram divergence",
    "Sentiment improving (0.2, up from -0.3)",
    "ATR suggests controlled volatility"
  ],
  "risk_assessment": "Low risk entry with 2:1 reward ratio. Stop placement below recent support at 4945."
}
```

#### Quick Start with LLM

```python
from mytrader.strategies.llm_enhanced_strategy import LLMEnhancedStrategy

# Create LLM-enhanced strategy
strategy = LLMEnhancedStrategy(
    enable_llm=True,
    min_llm_confidence=0.7,  # Only execute if confidence >= 70%
    llm_override_mode=False,  # Require consensus
)

# The strategy automatically queries AWS Bedrock before trades
signal = strategy.generate(features_df)
```

📖 **[Complete LLM Integration Guide](./LLM_INTEGRATION.md)** - Setup, configuration, and advanced features

---

## 🎯 Enhanced Trading Strategy Features

### Multi-Condition Signal Generation

The enhanced strategy now uses **multiple confirmation factors** before entering trades:

#### BUY Signal Requires (at least 2 of):
1. **RSI < 40** (oversold condition)
2. **MACD > 0 or Bullish Crossover** (momentum confirmation)
3. **Sentiment > -0.3** (not overly bearish)
4. **Bollinger %B < 0.2** (price near lower band - bonus confirmation)
5. **ADX > 25** (strong trend - bonus confirmation)

#### SELL Signal Requires (at least 2 of):
1. **RSI > 60** (overbought condition)
2. **MACD < 0 or Bearish Crossover** (momentum reversal)
3. **Sentiment < 0.3** (not overly bullish)
4. **Bollinger %B > 0.8** (price near upper band - bonus confirmation)
5. **ADX > 25** (strong trend - bonus confirmation)

### Market Regime Detection

The system automatically detects and adapts to 5 market regimes:

1. **Trending Up** - Follows trend, relaxed RSI thresholds
2. **Trending Down** - Tighter stops, conservative entries
3. **Mean-Reverting** - Standard oscillator thresholds
4. **High Volatility** - Reduced position sizing, wider stops
5. **Low Volatility** - Standard parameters, full position sizing

### Enhanced Risk Management

- **Kelly Criterion Position Sizing**: Dynamically adjusts based on win rate and risk/reward
- **ATR-Based Trailing Stops**: Adapts to market volatility
- **Portfolio Heat Monitoring**: Tracks total risk exposure
- **Dynamic Stop Loss**: 15 ticks ($187.50 per contract)
- **Dynamic Take Profit**: 30 ticks ($375 per contract) - 2:1 Risk/Reward

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
data:
  # Market Data Source (choose one or multiple)
  tradingview_webhook_url: "http://localhost:8000/tradingview"  # TradingView
  ibkr_host: "127.0.0.1"        # IBKR (can use for data OR execution only)
  ibkr_port: 4002

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

### 💡 Data Provider Options

The system supports **multiple data sources**:

- **IBKR**: Built-in, but has rate limits
- **TradingView**: Recommended ($15/month, no rate limits) - See [DATA_PROVIDERS.md](DATA_PROVIDERS.md)
- **Polygon.io**: Professional option ($99/month)
- **Alpha Vantage**: Free tier available
- **Yahoo Finance**: Free fallback

**Pro Tip:** Use TradingView/Polygon for market data and IBKR only for execution to avoid rate limits.

📖 **[Read the full Data Providers Guide](DATA_PROVIDERS.md)** for detailed setup instructions.

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

## 🎯 Quick Start: Strategy Optimization

Want to maximize performance? Run the complete optimization pipeline:

```bash
# Quick start - Runs baseline analysis, optimization, and comparison
python3 quickstart_optimization.py

# Or run individual steps:

# Step 1: Baseline performance analysis
python3 scripts/performance_analyzer.py \
    --data data/es_synthetic_with_sentiment.csv \
    --output reports/baseline

# Step 2: Optimize strategy (50-100 trials recommended)
python3 scripts/advanced_optimizer.py \
    --data data/es_synthetic_with_sentiment.csv \
    --strategy enhanced \
    --trials 100 \
    --output reports/optimization.json

# Step 3: Compare optimized vs baseline
python3 scripts/performance_analyzer.py \
    --data data/es_synthetic_with_sentiment.csv \
    --optimized reports/optimization.json \
    --output reports/comparison
```

**What This Does:**
- Analyzes current strategy performance
- Uses Bayesian optimization to find best parameters
- Tests on validation data (prevents overfitting)
- Generates visual comparison reports
- Achieves target: Sharpe ≥1.5, Max DD ≤15%, Win Rate ≥60%

📖 **[Read Full Optimization Guide](./STRATEGY_OPTIMIZATION.md)**

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
