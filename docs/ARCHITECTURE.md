# System Architecture

## High-Level Overview

MyTrader is an autonomous trading bot designed for futures markets (specifically ES/MES). It operates as a closed-loop system that:
1.  **Ingests** real-time market data from Interactive Brokers.
2.  **Analyzes** price action using multiple technical strategies.
3.  **Enhances** decisions using a Retrieval-Augmented Generation (RAG) memory system that recalls similar historical setups.
4.  **Executes** trades with strict risk management and automated exit brackets.

The system is built in Python and uses an asynchronous event loop to handle data processing and execution concurrently.

## Integrations

The bot interacts with the following external systems:

*   **Interactive Brokers (IBKR)**:
    *   **Library**: `ib_insync`
    *   **Connection**: Connects to IB Gateway or TWS on port `4002` (Paper) or `4001` (Live).
    *   **Function**: Receives real-time price bars (5-second updates) and manages orders/positions.
*   **Telegram (Optional)**:
    *   **Function**: Sends real-time alerts for trade entries, exits, and errors to a configured chat ID.
*   **Local Storage**:
    *   **SQLite**: Acts as the long-term memory for the RAG system.

## Data Persistence

The bot uses a hybrid approach for data management:

### 1. In-Memory (Short-Term)
*   **Price History**: The last 500 price bars are kept in RAM for calculating technical indicators (RSI, MACD, ATR).
*   **State**: Current position, active orders, and daily PnL are tracked in memory during the session.

### 2. SQLite Database (Long-Term)
*   **Location**: `data/rag_storage.db`
*   **Trades Table**: Stores every executed trade with context "buckets" (e.g., "High Volatility", "Morning Session").
*   **Snapshots Table**: Stores market indicator states at the time of every trade decision.

### 3. File System
*   **Logs**: Detailed execution logs are written to `logs/live_trading.log`.
*   **Config**: User settings are read from `config.yaml`.

## Core Logic

### Entry Logic
The entry decision process follows this pipeline:

1.  **Data Collection**: The bot waits until it has collected enough bars (default: 50) to calculate indicators.
2.  **Strategy Vote**: The `StrategyEngine` asks multiple strategies for a decision:
    *   *RSI/MACD/Sentiment*: Checks for momentum alignment.
    *   *Momentum Reversal*: Checks for mean reversion opportunities.
3.  **RAG Adjustment**:
    *   The system classifies the current market into buckets (e.g., "High Volatility", "Midday").
    *   It queries the SQLite database: *"How did we perform in these conditions previously?"*
    *   If past performance was good, confidence is boosted. If bad, confidence is reduced.
4.  **Execution**: If the final confidence exceeds the threshold and no position is open, a trade is placed.

### Exit Logic
Exits are automated and strictly defined at the moment of entry:

1.  **Bracket Orders**: Every entry is submitted with a hard **Stop Loss** and **Take Profit** attached.
    *   *Stop Loss*: Calculated based on ATR (Average True Range) or fixed ticks.
    *   *Take Profit*: Set at a fixed multiple of the risk (e.g., 2x Stop Loss).
2.  **Trailing Stop**: As the price moves in favor, the bot dynamically moves the Stop Loss up (for Longs) or down (for Shorts) to lock in profits.
