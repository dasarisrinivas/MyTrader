# MyTrader System Design & Safety Architecture

## Overview
This document outlines the architecture and safety mechanisms implemented for the MyTrader SPY-futures bot. The system has been enhanced to enforce strict risk controls, integrate Retrieval-Augmented Generation (RAG) for decision support, and improve observability.

## Safety Mechanisms

### 1. Hard Position Cap (`PositionManager`)
- **Constraint**: `MAX_CONTRACTS = 5` (Absolute hard cap on net open contracts).
- **Implementation**: The `PositionManager` class acts as the authoritative gatekeeper for all order placements.
- **Logic**:
  - Before any order is submitted, `PositionManager.can_place_order(requested_qty)` is called.
  - It queries the Interactive Brokers API (source of truth) for the current net position.
  - It calculates `available_capacity = MAX_CONTRACTS - abs(current_net_position)`.
  - If `requested_qty > available_capacity`, the order is either reduced or rejected based on policy.
- **Concurrency**: Uses `asyncio.Lock` to prevent race conditions where multiple concurrent signals could breach the cap.

### 2. Margin Safety
- **Constraint**: `MARGIN_LIMIT_PCT = 0.80` (80%).
- **Implementation**: `PositionManager` estimates post-trade margin usage.
- **Logic**:
  - Fetches current account equity and margin maintenance from IB.
  - Estimates incremental margin for the new order (using a conservative per-contract margin estimate).
  - Rejects orders if `(current_margin + incremental_margin) / equity > MARGIN_LIMIT_PCT`.

### 3. Execution Safety
- **Limit Orders**: The system prefers Limit orders with a small marketable buffer (e.g., 2 ticks) to ensure fills without paying full spread or suffering unlimited slippage.
- **Slippage Protection**: Monitors execution prices and alerts if slippage exceeds thresholds.
- **Reconciliation**: On startup and reconnection, the system reconciles its internal state with IB positions.

## RAG Integration (Retrieval-Augmented Generation)

### Purpose
To leverage historical trade data and market context to improve decision confidence, without letting an LLM directly control trading.

### Architecture
1.  **Storage (`RAGStorage`)**:
    -   **Database**: SQLite (`data/rag_storage.db`).
    -   **Schema**:
        -   `trades`: Stores completed trades with entry/exit details, PnL, hold time, and decision features.
        -   `market_snapshots`: Stores OHLCV and indicators.
    -   **Bucketing**: Trades are indexed by `volatility_bucket` (LOW, MEDIUM, HIGH) and `time_of_day` (MORNING, MIDDAY, CLOSE) for fast retrieval.

2.  **Retrieval & Decision**:
    -   During the decision cycle, the system identifies the current market bucket.
    -   It retrieves similar historical trades from `RAGStorage`.
    -   It calculates aggregate stats (Win Rate, Avg PnL) for these similar trades.
    -   **Adjustment**:
        -   If historical Win Rate > 60% (high confidence), signal confidence is boosted (+0.1).
        -   If historical Win Rate < 40% (low confidence), signal confidence is penalized (-0.2).

3.  **Persistence**:
    -   Trade entries are saved to `RAGStorage` upon order fill.
    -   Trade exits update the record with realized PnL and hold duration via execution listeners.

## Backtesting & Verification

### Backtest Engine
-   The `BacktestingEngine` has been updated to strictly enforce `MAX_CONTRACTS`.
-   It tracks `max_concurrent_contracts` as a metric to verify compliance during simulation.

### Tests
-   **Unit Tests**: `tests/test_position_manager.py` verifies cap enforcement and margin logic.
-   **Integration Tests**: `tests/test_rag_integration.py` verifies RAG storage and retrieval.
-   **Safety Tests**: `tests/test_backtest_safety.py` confirms that the backtest engine respects constraints even with aggressive strategies.

## Operational Runbook

### Configuration
All sensitive configuration is read from environment variables.
-   `MAX_CONTRACTS`: Default 5.
-   `MARGIN_LIMIT_PCT`: Default 0.8.
-   `IBKR_HOST` / `IBKR_PORT`: Connection details.

### Deployment
1.  **Environment Setup**:
    ```bash
    export IBKR_HOST=127.0.0.1
    export IBKR_PORT=4002
    export MAX_CONTRACTS=5
    ```
2.  **Run Tests**:
    ```bash
    pytest tests/
    ```
3.  **Start Bot**:
    ```bash
    python main.py
    ```

### Monitoring
-   **Logs**: Structured logs are output to console and file (if configured).
-   **Alerts**: Critical issues (cap breach attempts, margin rejections) trigger alerts via configured channels (Slack/Email).
