# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Live Trading
```bash
python run_bot.py                          # Start live trading (requires config.yaml)
python run_bot.py --simulation             # Dry-run (no real orders)
python run_bot.py --reset-state            # Clear cooldowns/locks on startup
python run_bot.py --cooldown 10            # Override cooldown in minutes
./start_bot.sh                             # Production startup wrapper
./stop.sh                                  # Graceful shutdown
```

### Backtesting
```bash
python -m backtest.run --symbol MES --start 2024-01-09 --end 2026-01-09
python -m backtest.run --symbol MES --start 2024-01-09 --end 2026-01-09 --data-source databento
python -m backtest.run --symbol MES --data-source file --data-file data/es_historical.csv
```

### Testing
```bash
pytest tests/
pytest tests/test_mes_one_minute_strategy.py -v
pytest --cov=shree tests/
python test_ib_connections.py              # Integration test (requires IB Gateway)
```

### Docker
```bash
docker build -t shreebot:latest .
docker run -e IBKR_HOST=127.0.0.1 -v $(pwd)/config.yaml:/app/config.yaml shreebot:latest
```

## Architecture

ShreeBot is an algorithmic MES (Micro E-mini S&P 500) futures trading bot using Interactive Brokers. The flow is:

**`run_bot.py` → `LiveTradingManager` → [Strategy Engine → Risk Gate → IB Executor]**

### Layers

**Data Layer** (`shree/data/`)
- `ibkr.py`: Real-time bar collection via `ib_insync`, with reconnect logic
- `live_data_manager.py`: Candle aggregation + technical indicators
- `candle_aggregator.py`: Rolls 1m bars into 5m/15m/30m

**Strategy Layer** (`shree/strategies/`)
- `engine.py`: Multi-strategy voting ensemble. Aggregates signals from 6+ strategies, adjusts confidence thresholds dynamically via Sharpe ratio feedback
- `mes_one_minute.py`: Primary 1m strategy with 26+ named signal types (A–G with variants)
- `scoring_entry.py`: Confidence-weighted scoring entry module
- `trading_filters.py`: Price level filters (PDH, PDL, weekly H/L)
- `rag_validator.py`: RAG knowledge base validation against historical trades

**Hybrid/Deterministic Layer** (`shree/hybrid/`)
- `d_engine.py`: RSI pullback filters, session-aware thresholds
- `h_engine.py`: Aggregates deterministic + ML/LLM signals
- `coordination.py`: Agent bus for inter-module communication

**RAG + LLM Layer** (`shree/rag/`, `shree/llm/`)
- FAISS-backed vector store; retrieves similar historical trades as context
- AWS Bedrock integration for optional reasoning overlay (commentary, not decision-making)
- Enabled/disabled via `config.yaml` (`rag.enabled`, `llm.enabled`)

**Risk Layer** (`shree/risk/`)
- `risk_gate.py`: Hard pre-trade checks (margin, daily loss, drawdown cap)
- `manager.py`: Position sizing (Kelly / fixed-fraction), daily trade counting
- `dynamic_support.py`: Tightens SL floor as profit accumulates during a session

**Execution Layer** (`shree/execution/`)
- `live_trading_manager.py`: Main trading loop; manages candle close events, cooldown, simulation mode, WebSocket broadcasting
- `ib_executor.py`: Places bracket orders (entry + SL stop + TP target), monitors fills, reconciles with IB state; uses `OrderLockManager` to prevent duplicate orders

**Monitoring** (`shree/monitoring/`, `shree/observability/`)
- `live_tracker.py`: Real-time equity, drawdown, Sharpe, win-rate
- `order_tracker.py`: Order lifecycle tracking
- `prometheus.py`: Metrics export on port 8000

### Configuration

Copy `config.example.yaml` → `config.yaml` before running. Key sections:

| Section | Purpose |
|---------|---------|
| `data:` | IBKR host/port/client-id, symbol |
| `trading:` | Max position size, daily loss limit, initial capital |
| `one_minute:` | RTH-only flag, overnight trading, max trades/day |
| `risk_gate:` | Hard limits: `max_contracts`, `daily_max_loss_usd`, `margin_buffer_usd` |
| `llm:` | Bedrock `model_id`, `region_name`, `enabled` |
| `rag:` | Backend (`local_faiss`/`opensearch`), `top_k_results`, `enabled` |
| `telegram:` | Bot token, chat ID, notification triggers |
| `observability:` | `prometheus_enabled`, `prometheus_port` |

IBKR Gateway must be running on `127.0.0.1:7497` (paper) or `127.0.0.1:4001` (live) before starting the bot.

### Known Critical Issues (from `docs/review.md`)

Before scaling beyond 1 contract or going fully live, be aware of:
1. **Forced trades**: After 10 consecutive HOLD signals, the bot forces a BUY/SELL — dangerous in choppy markets
2. **Fail-open risk checks**: Position/margin check failures do not halt trading
3. **Inconsistent daily loss limits**: `RiskGate` ($150/day) and `RiskManager` ($2000/day) are mismatched
4. **Bracket order race condition**: SL and TP child orders placed sequentially, leaving a window with no hedge
5. **Hardcoded parameters**: RSI bounds, ADX thresholds in signal logic may overfit to backtested data

### Key Reference Docs

- `docs/review.md`: Detailed code review with risk flaws and red-team scenarios
- `DEPLOYMENT_OPS.md`: Live validation metrics and performance thresholds (50-trade minimum before scaling)
- `docs/BACKTESTING.md`: Backtest guide and data sources (Databento, Polygon, IB historical)
