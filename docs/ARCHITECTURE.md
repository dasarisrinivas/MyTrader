# System Architecture

## High-Level Overview

MyTrader is an autonomous trading bot designed for futures markets (specifically ES/MES - S&P 500 E-mini and Micro E-mini futures). It operates as a **hybrid intelligent system** that combines:

1.  **LOCAL ENGINE (Fast, Real-Time)**: Ingests real-time market data from Interactive Brokers and executes trades using technical analysis strategies.
2.  **BEDROCK ENGINE (Intelligent, Event-Driven)**: Uses AWS Bedrock LLM for market analysis on specific events (NOT every tick).
3.  **RAG Memory System**: Recalls similar historical setups to enhance decision confidence.
4.  **Strict Risk Management**: Automated exit brackets that CANNOT be overridden by AI.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          HYBRID ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────┐         ┌──────────────────────┐                  │
│  │   LOCAL ENGINE       │         │   BEDROCK ENGINE     │                  │
│  │   (Fast Path)        │         │   (Smart Path)       │                  │
│  ├──────────────────────┤         ├──────────────────────┤                  │
│  │ • IB Gateway Ticks   │         │ • AWS Bedrock LLM    │                  │
│  │ • Technical Analysis │         │ • Event-Triggered    │                  │
│  │ • Signal Generation  │◄───────►│ • Bias Modifier      │                  │
│  │ • Order Execution    │         │ • NOT Trade Override │                  │
│  │ • Risk Management    │         │ • Cached Results     │                  │
│  └──────────────────────┘         └──────────────────────┘                  │
│           │                                  │                               │
│           │                                  │                               │
│           ▼                                  ▼                               │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │                      DATA LAYER                               │           │
│  │  • SQLite (bedrock_hybrid.db) - LLM call logs & costs        │           │
│  │  • SQLite (rag_storage.db) - Historical trade memory         │           │
│  │  • In-Memory - Price bars, indicators, position state        │           │
│  └──────────────────────────────────────────────────────────────┘           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Instrument: MES / ES Futures

The bot trades **SPY-equivalent futures exposure**:
- **MES (Micro E-mini S&P 500)**: Default instrument, $5 per point, lower margin
- **ES (E-mini S&P 500)**: Full contract, $50 per point

Symbol configuration in `config.yaml`:
```yaml
data:
  ibkr_symbol: "MES"  # or "ES"
  ibkr_exchange: "CME"
```

---

## Integrations

### Interactive Brokers (IBKR)
- **Library**: `ib_insync`
- **Connection**: IB Gateway on port `4002` (Paper) or `4001` (Live)
- **Function**: Real-time price bars (5-second polling), order management, position tracking

### AWS Bedrock (LLM)
- **Library**: `boto3` with `bedrock-runtime`
- **Model**: Claude 3 Sonnet (default) or Claude 3 Haiku
- **Function**: Event-driven market analysis, NOT real-time trading decisions
- **Cost Tracking**: All calls logged with token counts and cost estimates

### Telegram (Optional)
- **Function**: Real-time alerts for trade entries, exits, errors, and LLM insights

### Local Storage
- **SQLite**: Long-term memory for RAG and Bedrock call logging

---

## Data Persistence

### 1. In-Memory (Short-Term)
| Data | Purpose |
|------|---------|
| Price History (500 bars) | Technical indicator calculation |
| Current Position | Trade decision logic |
| Bedrock Bias Modifier | Event-driven LLM guidance |
| Active Orders | Order management |

### 2. SQLite Databases (Long-Term)

#### `data/bedrock_hybrid.db` (NEW)
```sql
-- Bedrock API call logging
CREATE TABLE bedrock_calls (
    id INTEGER PRIMARY KEY,
    ts TIMESTAMP,
    trigger TEXT,           -- 'market_open', 'volatility_spike', 'news', 'manual'
    prompt TEXT,
    response TEXT,
    model TEXT,
    tokens_in INTEGER,
    tokens_out INTEGER,
    cost_estimate REAL,
    latency_ms REAL,
    bias_result TEXT,       -- 'BULLISH', 'BEARISH', 'NEUTRAL'
    confidence REAL
);

-- Daily quota tracking
CREATE TABLE daily_quota (
    date TEXT PRIMARY KEY,
    call_count INTEGER,
    total_cost REAL
);
```

#### `data/rag_storage.db` (Existing)
- **Trades Table**: Historical trades with market context buckets
- **Snapshots Table**: Indicator states at trade decisions

### 3. File System
| Path | Purpose |
|------|---------|
| `logs/live_trading.log` | Execution logs |
| `config.yaml` | User settings |
| `reports/` | Backtest and performance reports |

---

## Core Logic: End-to-End Flow

### Startup Sequence

```
1. Load config.yaml
2. Initialize Multi-Strategy Engine (RSI/MACD, Momentum Reversal)
3. Initialize Hybrid Bedrock System:
   ├── HybridBedrockClient (AWS connection, caching, logging)
   ├── EventDetector (trigger conditions)
   ├── RAGContextBuilder (prompt construction)
   └── Background Worker Thread (non-blocking calls)
4. Connect to IB Gateway (async)
5. Subscribe to MES/ES market data
6. Enter main trading loop
```

### Main Trading Loop (Every 5 Seconds)

```python
while True:
    # ============= FAST PATH (Local Engine) =============
    
    # 1. Get current price from IB
    current_price = await executor.get_current_price()
    
    # 2. Update price history (last 500 bars)
    price_history.append(price_bar)
    
    # 3. Engineer features (RSI, MACD, ATR, momentum)
    features = engineer_features(price_history)
    
    # 4. Check market regime (tradable conditions)
    regime_result = regime_filter.check_regime(features)
    if not regime_result.tradable:
        continue
    
    # ============= SMART PATH (Bedrock Engine) =============
    
    # 5. Check for Bedrock trigger events (NON-BLOCKING)
    snapshot = build_market_snapshot(features, position)
    trigger, reason, payload = event_detector.should_call_bedrock(snapshot)
    
    if trigger:
        # Queue Bedrock request to background thread
        context = rag_context_builder.build_context(payload)
        bedrock_worker_queue.put((context, trigger))
    
    # 6. Check for Bedrock results (NON-BLOCKING)
    if not bedrock_result_queue.empty():
        bedrock_bias_modifier = bedrock_result_queue.get()
    
    # ============= SIGNAL GENERATION =============
    
    # 7. Generate trading signal (local strategies)
    action, confidence, risk_params = multi_strategy.generate_signal(features)
    
    # 8. Apply Bedrock bias modifier (DOES NOT OVERRIDE ACTION)
    if bedrock_bias_modifier["bias"] != "NEUTRAL":
        if agreement:
            confidence += min(0.10, bedrock_confidence * 0.15)  # Boost
        else:
            confidence -= min(0.15, bedrock_confidence * 0.20)  # Reduce
    
    # ============= EXECUTION (Risk Rules ALWAYS Apply) =============
    
    # 9. Execute trade if confidence threshold met
    if action != "HOLD" and confidence > threshold:
        await executor.place_order(
            action=action,
            stop_loss=calculated_stop,      # NEVER overridden by LLM
            take_profit=calculated_target,  # NEVER overridden by LLM
        )
```

---

## Bedrock Event Triggers

The Bedrock LLM is called ONLY on specific events, NOT every tick:

| Trigger | Condition | Purpose |
|---------|-----------|---------|
| `market_open` | 5 min after RTH open | Daily outlook and bias |
| `market_close` | 5 min before RTH close | Session review |
| `volatility_spike` | ATR > 2x baseline | Assess unusual volatility |
| `news` | Keywords detected (CPI, FOMC, Fed, etc.) | Macro event analysis |
| `manual` | Flask API `/trigger-bedrock` | On-demand analysis |

### Bedrock Output (Bias Modifier)

```json
{
    "bias": "BULLISH",
    "confidence": 0.75,
    "action": "BUY",
    "rationale": "Strong momentum with supportive technicals",
    "key_factors": ["Rising RSI", "Positive MACD crossover"],
    "risk_notes": "Watch resistance at 5400"
}
```

**CRITICAL**: The Bedrock output is a **BIAS MODIFIER ONLY**:
- It adjusts signal confidence by ±10-15%
- It DOES NOT override the trading action
- It DOES NOT change stop-loss or take-profit
- It DOES NOT bypass risk management rules

---

## Entry Logic

### Signal Generation Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Technical      │     │  RAG Memory     │     │  Bedrock Bias   │
│  Strategies     │────►│  Adjustment     │────►│  Modifier       │
│  (Local)        │     │  (Historical)   │     │  (Event-Driven) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FINAL CONFIDENCE SCORE                        │
│         (Must exceed threshold to execute trade)                 │
└─────────────────────────────────────────────────────────────────┘
```

1. **Technical Strategies Vote**:
   - RSI/MACD/Sentiment: Momentum alignment
   - Momentum Reversal: Mean reversion opportunities
   
2. **RAG Adjustment**:
   - Classify market into buckets (e.g., "High Volatility", "Morning")
   - Query: "How did we perform in these conditions?"
   - Boost or reduce confidence based on history

3. **Bedrock Bias Modifier** (if event triggered):
   - If Bedrock agrees with signal → confidence +5-10%
   - If Bedrock disagrees → confidence -10-15%

4. **Execution Gate**:
   - Final confidence must exceed `min_weighted_confidence` (default: 0.50)
   - Risk manager must approve (`can_trade()`)
   - No existing position in same direction

---

## Exit Logic

Exits are **strictly automated** and CANNOT be overridden by Bedrock:

### Bracket Orders (Set at Entry)
| Order Type | Calculation | Purpose |
|------------|-------------|---------|
| Stop Loss | ATR × multiplier (min 10 points) | Limit downside |
| Take Profit | Stop Distance × R:R ratio | Lock in gains |

### Dynamic Adjustments
- **Trailing Stop**: Moves stop in favor as price advances
- **Time-Based Exit**: Close after max duration (configurable)
- **Disaster Stop**: Emergency exit if loss exceeds threshold

---

## Flask API Endpoints (Dashboard)

### Bedrock Endpoints (NEW)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/trigger-bedrock` | POST | Manual Bedrock analysis trigger |
| `/api/bedrock-status` | GET | Daily calls, cost, quota, last 10 calls |
| `/api/bedrock-detector-status` | GET | Event detector state |

### Example: Manual Trigger

```bash
curl -X POST http://localhost:8000/api/trigger-bedrock \
  -H "Content-Type: application/json" \
  -d '{"notes": "Pre-FOMC analysis", "context": {"current_price": 5375.00}}'
```

Response:
```json
{
    "request_id": "a1b2c3d4",
    "status": "success",
    "message": "Bedrock analysis complete: BULLISH",
    "result": {
        "bias": "BULLISH",
        "confidence": 0.72,
        "rationale": "..."
    }
}
```

---

## Cost Management

### Bedrock API Costs (Approximate)

| Model | Input (per 1K tokens) | Output (per 1K tokens) |
|-------|----------------------|------------------------|
| Claude 3 Sonnet | $0.003 | $0.015 |
| Claude 3 Haiku | $0.00025 | $0.00125 |

### Daily Limits (Configurable)
- **Daily Quota**: 1000 calls/day (default)
- **Daily Cost Limit**: $50/day (default)
- **Warning**: Logged when quota approaches limit

### Cost Tracking Query
```sql
SELECT date, call_count, total_cost 
FROM daily_quota 
ORDER BY date DESC 
LIMIT 7;
```

---

## Running the Bot

### Start Trading (Live)
```bash
# Ensure IB Gateway is running on port 4002
python main.py live --config config.yaml
```

### Start Dashboard
```bash
cd dashboard/backend
uvicorn dashboard_api:app --host 0.0.0.0 --port 8000 --reload
```

### Environment Variables (AWS)
```bash
export AWS_REGION=us-east-1
export AWS_BEDROCK_MODEL=anthropic.claude-3-sonnet-20240229-v1:0
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
```

---

## Safety Guarantees

| Guarantee | Implementation |
|-----------|----------------|
| Bedrock never overrides stop-loss | Code enforces at execution layer |
| Bedrock never increases position size | Risk manager validates all orders |
| Bedrock calls are non-blocking | Background thread with queue |
| Bedrock is event-driven, not tick-driven | EventDetector with cooldowns |
| All Bedrock calls are logged | SQLite with cost tracking |
| Daily cost limits enforced | Quota check before each call |

---

## Module Reference

### New Modules (Hybrid Bedrock)

| Module | Location | Purpose |
|--------|----------|---------|
| `HybridBedrockClient` | `mytrader/llm/bedrock_hybrid_client.py` | AWS Bedrock API with caching & logging |
| `EventDetector` | `mytrader/llm/event_detector.py` | Trigger condition detection |
| `RAGContextBuilder` | `mytrader/llm/rag_context_builder.py` | Structured prompt construction |
| `BedrockSQLiteManager` | `mytrader/llm/sqlite_manager.py` | Call logging & quota tracking |

### Existing Modules

| Module | Location | Purpose |
|--------|----------|---------|
| `TradeExecutor` | `mytrader/execution/ib_executor.py` | IB order management |
| `MultiStrategy` | `mytrader/strategies/multi_strategy.py` | Signal generation |
| `RiskManager` | `mytrader/risk/manager.py` | Position sizing & limits |
| `RAGEngine` | `mytrader/llm/rag_engine.py` | Historical memory retrieval |
