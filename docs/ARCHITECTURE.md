# MyTrader Architecture Documentation

## Overview

MyTrader is an autonomous SPY futures trading bot enhanced with Retrieval-Augmented Generation (RAG) for AI-powered decision-making. The system combines traditional technical analysis with large language model (LLM) intelligence to generate, validate, and execute trading signals.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend Dashboard                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │  Trading │  │   RAG    │  │  Signal  │  │    Error     │   │
│  │ Controls │  │  Status  │  │ Display  │  │  Handling    │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
│         React + Vite + Tailwind CSS + WebSocket                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ WebSocket + REST API
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Backend API (FastAPI)                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ Trading  │  │   RAG    │  │ Backtest │  │  WebSocket   │   │
│  │   API    │  │   API    │  │   API    │  │   Manager    │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Trading Engine                           │
│  ┌──────────────────────┐  ┌──────────────────────────────┐    │
│  │  Multi-Strategy      │  │    LLM-Enhanced Strategy     │    │
│  │  ┌───────────────┐   │  │  ┌──────────────────────┐   │    │
│  │  │ Trend Follow  │   │  │  │   RAG Trade Advisor  │   │    │
│  │  │ Breakout      │   │  │  │  ┌────────────────┐  │   │    │
│  │  │ Mean Reversion│   │  │  │  │  RAG Engine    │  │   │    │
│  │  └───────────────┘   │  │  │  │  ┌──────────┐  │  │   │    │
│  │                       │  │  │  │  │  FAISS   │  │  │   │    │
│  │  ┌───────────────┐   │  │  │  │  │ Vector DB│  │  │   │    │
│  │  │ RAG Validator │   │  │  │  │  └──────────┘  │  │   │    │
│  │  └───────────────┘   │  │  │  └────────────────┘  │   │    │
│  └──────────────────────┘  │  └──────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Risk Management Layer                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ Stop-Loss│  │  Take    │  │ Position │  │  Circuit     │   │
│  │  Config  │  │  Profit  │  │  Sizing  │  │  Breakers    │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Execution & Data Layer                        │
│  ┌──────────────────────┐  ┌──────────────────────────────┐    │
│  │   IBKR Executor      │  │      Data Pipeline           │    │
│  │  ┌───────────────┐   │  │  ┌────────────────────────┐ │    │
│  │  │ Order Mgmt    │   │  │  │  Market Data (IBKR)    │ │    │
│  │  │ Position Mgmt │   │  │  │  Feature Engineering   │ │    │
│  │  │ Bracket Orders│   │  │  │  Technical Indicators  │ │    │
│  │  └───────────────┘   │  │  └────────────────────────┘ │    │
│  └──────────────────────┘  └──────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    External Services                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│  │  AWS Bedrock     │  │  Interactive     │  │   Market     │ │
│  │  (Claude 3)      │  │   Brokers (IB)   │  │    Data      │ │
│  │  - LLM           │  │  - Trading API   │  │  Providers   │ │
│  │  - Embeddings    │  │  - TWS/Gateway   │  │              │ │
│  └──────────────────┘  └──────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Frontend Dashboard (React)

**Location:** `/dashboard/frontend/`

#### Key Components:

- **Dashboard.jsx**: Main dashboard layout with tabs
- **TradingSignalDisplay.jsx**: Real-time signal visualization
- **RAGStatusIndicator.jsx**: RAG engine health monitoring
- **ErrorStates.jsx**: Error notifications and connection status
- **BotOverview.jsx**: Performance metrics and bot status
- **DecisionIntelligence.jsx**: AI decision reasoning display

**Features:**
- Real-time updates via WebSocket
- Error handling with auto-retry
- Responsive design with Tailwind CSS
- Trading signal history with confidence scores
- RAG knowledge base health monitoring

### 2. Backend API (FastAPI)

**Location:** `/dashboard/backend/`

#### Main APIs:

**dashboard_api.py:**
- `/api/trading/status` - Get bot status
- `/api/trading/start` - Start trading bot
- `/api/trading/stop` - Stop trading bot
- `/api/trading/emergency-exit` - Emergency position closure
- WebSocket endpoint for real-time updates

**rag_api.py:**
- `/rag/ingest` - Ingest documents into knowledge base
- `/rag/ask` - Query with RAG (retrieval + generation)
- `/rag/retrieve` - Retrieve documents only
- `/rag/stats` - Get RAG engine statistics
- `/rag/clear-cache` - Clear query cache
- `/rag/health` - Health check endpoint

### 3. Trading Engine

**Location:** `/mytrader/`

#### Core Modules:

**Strategies (`/strategies/`):**
- `multi_strategy.py`: Main strategy orchestrator
  - Trend following (MA crossovers)
  - Breakout detection (price + volume)
  - Mean reversion (Bollinger Bands + RSI)
  - Auto-strategy selection based on market conditions
  
- `llm_enhanced_strategy.py`: LLM wrapper for base strategies
- `rag_validator.py`: RAG-based signal validation

**LLM/RAG (`/llm/`):**
- `rag_engine.py`: Core RAG implementation
  - AWS Titan embeddings
  - FAISS vector store
  - Query caching
  - Retry logic with exponential backoff
  - Latency monitoring
  
- `multi_instrument_rag.py`: Multi-instrument support
- `bedrock_client.py`: AWS Bedrock integration
- `rag_trade_advisor.py`: RAG-enhanced trading advice

**Risk Management (`/risk/`):**
- `manager.py`: Risk manager with Kelly criterion
- `config.py`: Configurable risk parameters
  - Stop-loss (fixed, ATR-based, percentage, trailing)
  - Take-profit (fixed, risk:reward, ATR-based)
  - Position sizing (fixed, Kelly, risk parity)
  - Circuit breakers and time restrictions

**Execution (`/execution/`):**
- `ib_executor.py`: Interactive Brokers execution
- `live_trading_manager.py`: Live trading orchestration

### 4. RAG System Architecture

#### Document Ingestion Flow:

```
Documents → Validation → Embedding Generation → Vector Storage
    ↓           ↓              ↓                      ↓
 Filter      Check        AWS Titan              FAISS Index
 Invalid     Length      Embeddings             + Persistence
```

#### Query Flow:

```
Query → Cache Check → Embedding → Vector Search → LLM Generation
   ↓        ↓            ↓            ↓              ↓
 Input   Return If    AWS Titan    Top-K Docs    Claude 3
 Valid   Cached      Embeddings   + Scores       Response
```

#### Error Handling:

- **Retry Logic**: Exponential backoff (1s, 2s, 4s)
- **Fallback**: Zero vectors for failed embeddings
- **Validation**: Input validation before processing
- **Monitoring**: Latency tracking and error counting
- **Health Status**: healthy/degraded/unhealthy states

### 5. Risk Management Configuration

**Location:** `/data/risk_config.json`

#### Configurable Parameters:

**Stop-Loss:**
```json
{
  "type": "atr_based",           // fixed_ticks, atr_based, percentage
  "atr_multiplier": 2.0,         // 2x ATR for stop
  "use_trailing": true,          // Enable trailing stops
  "trailing_atr_multiplier": 1.5 // 1.5x ATR for trailing
}
```

**Take-Profit:**
```json
{
  "type": "risk_reward_ratio",   // fixed_ticks, risk_reward_ratio, atr_based
  "risk_reward_ratio": 2.0,      // 2:1 reward:risk
  "use_partial_exits": false     // Partial profit taking
}
```

**Position Sizing:**
```json
{
  "method": "kelly",             // fixed, kelly, risk_parity
  "kelly_fraction": 0.5,         // Half-Kelly for safety
  "risk_per_trade_pct": 1.0,     // 1% risk per trade
  "scale_by_confidence": true    // Scale by signal confidence
}
```

**Limits:**
```json
{
  "max_daily_loss": 2000.0,      // Max loss per day
  "max_daily_trades": 20,        // Max trades per day
  "enable_circuit_breaker": true, // Stop on excessive loss
  "circuit_breaker_loss_pct": 5.0 // 5% daily loss triggers circuit breaker
}
```

## Data Flow

### Signal Generation Flow:

```
1. Market Data → Feature Engineering → Technical Indicators
                         ↓
2. Multi-Strategy Analysis → Generate Raw Signal
                         ↓
3. RAG Validator → Retrieve Trading Rules → Validate Signal
                         ↓
4. LLM Enhancement → Generate AI Recommendation → Merge Signals
                         ↓
5. Risk Manager → Calculate Position Size → Apply Risk Limits
                         ↓
6. IBKR Executor → Place Bracket Order → Monitor Position
```

### Signal Validation with RAG:

```
Raw Signal (BUY/SELL/HOLD) + Market Context
            ↓
Query RAG: "What are best practices for [action] in [market] conditions?"
            ↓
Retrieve: Top-3 relevant trading rules/documents
            ↓
Analyze: Check for warning patterns in retrieved rules
            ↓
Adjust: Reduce confidence or reject signal if warnings found
            ↓
Validated Signal with adjusted confidence
```

## Configuration

### Main Config (`config.example.yaml`):

```yaml
# Data sources
data:
  ibkr_host: "127.0.0.1"
  ibkr_port: 4002
  ibkr_symbol: "ES"
  ibkr_exchange: "CME"

# Trading parameters
trading:
  max_position_size: 4
  max_daily_loss: 2000.0
  initial_capital: 100000.0
  stop_loss_ticks: 10.0
  take_profit_ticks: 20.0

# LLM configuration
llm:
  enabled: true
  model_id: "anthropic.claude-3-sonnet-20240229-v1:0"
  min_confidence_threshold: 0.7
  override_mode: false

# RAG configuration
rag:
  enabled: true
  embedding_model_id: "amazon.titan-embed-text-v1"
  vector_store_path: "data/rag_index"
  top_k_results: 3
  score_threshold: 0.5
  cache_enabled: true
  cache_ttl_seconds: 3600
```

## Deployment

### Prerequisites:

```bash
# Python dependencies
pip install -r requirements.txt

# Frontend dependencies
cd dashboard/frontend
npm install
```

### Running the System:

```bash
# Start backend
python -m uvicorn dashboard.backend.dashboard_api:app --reload --port 8000

# Start frontend (separate terminal)
cd dashboard/frontend
npm run dev
```

### Trading Bot:

```bash
# Live trading
python main.py live --config config.yaml

# Backtesting
python main.py backtest --data data/historical.parquet
```

## Monitoring & Observability

### Key Metrics:

**RAG Engine:**
- Document count in knowledge base
- Average embedding latency
- Cache hit rate
- Error count and last error time
- Health status (healthy/degraded/unhealthy)

**Trading Bot:**
- Current position (size, PnL, entry price)
- Win rate and profit factor
- Daily PnL and drawdown
- Signal confidence distribution
- LLM override frequency

**Risk Management:**
- Current exposure (% of capital at risk)
- Trades today vs. daily limit
- Loss today vs. daily limit
- Circuit breaker status

### Health Checks:

- Backend API: `GET /health`
- RAG Engine: `GET /rag/health`
- WebSocket: Connection status indicator

## Security Considerations

### API Keys & Secrets:

- AWS credentials stored in environment variables or IAM roles
- IBKR credentials in secure config (not in repository)
- No hardcoded secrets in code

### Risk Controls:

- Circuit breakers prevent runaway losses
- Position size limits prevent over-leverage
- Daily trade limits prevent excessive activity
- Emergency exit button for manual intervention

## Testing

### Unit Tests:

```bash
# Run all tests
python -m pytest tests/ -v

# Specific test files
python -m pytest tests/test_rag_engine.py -v
python -m pytest tests/test_risk_config.py -v
```

### Test Coverage:

- RAG engine: Embedding generation, retrieval, caching, error handling
- Risk config: Validation, serialization, parameter calculation
- Signal validation: Data validation, NaN handling, extreme values

## Future Enhancements

### Planned Features:

1. **Multi-Instrument Support**: Trade beyond SPY (NQ, QQQ, etc.)
2. **Advanced RAG**: Semantic chunking, hybrid search
3. **Model Fine-tuning**: Custom model training on trade history
4. **Portfolio Management**: Multiple positions, correlation analysis
5. **Backtesting UI**: Interactive backtest parameter tuning
6. **Alert System**: SMS/Email notifications for critical events
7. **Performance Analytics**: Advanced metrics and visualizations

### Scalability Improvements:

- Async embeddings for faster ingestion
- Distributed vector store for larger knowledge bases
- Rate limiting and load balancing for API
- Database for persistent storage (currently file-based)

## Troubleshooting

### Common Issues:

**RAG Engine Not Initializing:**
- Check AWS credentials are configured
- Verify boto3 and faiss-cpu are installed
- Check RAG enabled in config: `rag.enabled: true`

**WebSocket Disconnects:**
- Check backend is running on port 8000
- Verify CORS settings in dashboard_api.py
- Check browser console for connection errors

**Trading Bot Not Starting:**
- Verify IB Gateway/TWS is running
- Check port configuration (4002 for Gateway)
- Ensure API connections are enabled in IB settings

**Signals Not Generating:**
- Check minimum data requirements (50+ bars)
- Verify technical indicators are calculating correctly
- Review logs for NaN or validation errors

## Contributing

When contributing to this project:

1. Follow existing code structure and naming conventions
2. Add tests for new functionality
3. Update documentation for API changes
4. Use type hints for Python code
5. Follow ESLint rules for React code
6. Test changes locally before submitting PR

## License

[Add your license information here]

## Contact & Support

[Add contact information here]
