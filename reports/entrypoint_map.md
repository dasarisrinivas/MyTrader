# Entrypoint Map - MyTrader Trading Bot

## Overview
This document maps the execution flow from `start_bot.sh` through the trading loop to identify the true entrypoint and control flow.

## Entrypoint Chain

### 1. Shell Script: `start_bot.sh`
- **Location**: `/Users/svss/Documents/code/MyTrader/start_bot.sh`
- **Invocation**: `python run_bot.py [BOT_ARGS]`
- **Environment Variables Set**:
  - `MAX_CONTRACTS` (default: 5)
  - `IBKR_HOST` (default: 127.0.0.1)
  - `IBKR_PORT` (default: 4002)
  - `ENABLE_GUARDRAILS` (default: 1)
  - `MYTRADER_SIMULATION` (optional, adds `--simulation` flag)
  - Feature Flags (when guardrails enabled):
    - `FF_ENTRY_RISK_GUARDS=1`
    - `FF_WAIT_BLOCKING=1`
    - `FF_EXIT_GUARDS=1`
    - `FF_LEARNING_HOOKS=1`
- **Working Directory**: Project root (`/Users/svss/Documents/code/MyTrader`)
- **Logs**: `logs/bot.log` (stdout/stderr redirected)

### 2. Python Entry: `run_bot.py`
- **Location**: `/Users/svss/Documents/code/MyTrader/run_bot.py`
- **Main Function**: `async def main()`
- **Key Actions**:
  1. Parses CLI args (`--simulation`, `--config`, `--cooldown`)
  2. Configures logging to `logs/live_trading.log`
  3. Loads settings from `config.yaml` (or specified config)
  4. Creates `LiveTradingManager(settings, simulation_mode=args.simulation)`
  5. Calls `await manager.start()`

### 3. Trading Manager: `LiveTradingManager.start()`
- **Location**: `mytrader/execution/live_trading_manager.py:736`
- **Main Loop**: `async def start()`
- **Flow**:
  1. Calls `await self.initialize()` to set up IB connection, executor, strategies
  2. Enters `while self.running and not self.stop_requested:` loop
  3. Polls every 5 seconds (`poll_interval = 5`)
  4. For each cycle:
     - Fetches current price via `executor.get_current_price()`
     - Appends to `price_history` (max 500 bars)
     - Waits for warmup (`min_bars_needed = 50`)
     - Calls `await self._process_trading_cycle(current_price)`

### 4. Trading Cycle: `_process_trading_cycle()`
- **Location**: `mytrader/execution/live_trading_manager.py:814`
- **Key Steps**:
  1. **Cooldown Check**: If `_last_trade_time` < `_cooldown_seconds` ago, skip
  2. **Order Lock Check**: If `executor.is_order_locked()`, skip
  3. **Candle Close Validation**: Only evaluate at start of new candle (60s period)
  4. **Feature Engineering**: Converts price history to DataFrame, engineers features
  5. **Signal Generation**: 
     - If hybrid pipeline enabled → `hybrid_pipeline.generate_signal()`
     - Else → legacy `engine.generate_signal()`
  6. **Signal Processing**:
     - If hybrid → `await self._process_hybrid_signal(signal, pipeline_result, current_price, features)`
     - Else → `await self._place_order(signal, current_price, features)`

### 5. Order Placement Paths

#### A. Hybrid Signal Path: `_process_hybrid_signal()`
- **Location**: `mytrader/execution/live_trading_manager.py:1363`
- **Flow**:
  1. Gets current position
  2. Updates trailing stops if position exists
  3. **AWS Agents Consultation** (if enabled):
     - Builds market snapshot
     - Calls `aws_agent_invoker.get_trading_decision()`
     - Checks `should_block_on_wait()` if decision is WAIT
     - Applies confidence adjustments
  4. **Exit Logic**: If position exists and signal is opposite direction → `_place_exit_order()`
  5. **Entry Logic**: If no position → `_place_hybrid_order()` or `_place_order()`

#### B. Standard Order Path: `_place_order()`
- **Location**: `mytrader/execution/live_trading_manager.py:2184`
- **Flow**:
  1. Position sizing via `risk.position_size()`
  2. Calculates stop_loss/take_profit from ATR or fallback
  3. **Validation**: Calls `_validate_entry_guard()` if `FF_ENTRY_RISK_GUARDS` enabled
  4. **Simulation Check**: If `simulation_mode`, logs and returns
  5. **Order Submission**: Calls `executor.place_order()` with bracket orders

#### C. Exit Order Path: `_place_exit_order()`
- **Location**: `mytrader/execution/live_trading_manager.py:1581`
- **Flow**:
  1. Gets current position
  2. Determines exit action (SELL if long, BUY if short)
  3. Calls `executor.place_order()` with `stop_loss=None, take_profit=None, reduce_only=True`

### 6. Executor: `TradeExecutor.place_order()`
- **Location**: `mytrader/execution/ib_executor.py:634`
- **Flow**:
  1. Validates bracket prices via `validate_bracket_prices()` if SL/TP provided
  2. Builds parent order (LimitOrder or MarketOrder)
  3. Builds bracket children (StopLimitOrder for SL, LimitOrder for TP)
  4. Engages order lock
  5. Places order via `ib.placeOrder()`
  6. Registers callbacks for fills

## Key Configuration Files

1. **`config.yaml`**: Main configuration
   - Trading parameters (max_position_size, risk_per_trade_pct, etc.)
   - AWS agents settings (`aws_agents.block_on_wait`, `aws_agents.wait_override_confidence`)
   - Feature flags (`features.enforce_entry_risk_checks`, etc.)
   - RAG settings (`rag.backend`, `rag.opensearch_enabled`)

2. **Environment Variables** (from `start_bot.sh`):
   - Feature flags: `FF_ENTRY_RISK_GUARDS`, `FF_WAIT_BLOCKING`, `FF_EXIT_GUARDS`, `FF_LEARNING_HOOKS`
   - IB connection: `IBKR_HOST`, `IBKR_PORT`
   - Risk: `MAX_CONTRACTS`

## Log Locations

- **Main Bot Log**: `logs/bot.log` (from `start_bot.sh` stdout/stderr)
- **Trading Log**: `logs/live_trading.log` (from `run_bot.py` logger)
- **Structured Events**: Via `log_structured_event()` (may write to separate files)

## Trade Cycle ID Flow

- **Generation**: `_current_cycle_id = str(uuid.uuid4())` at start of `_process_trading_cycle()`
- **Propagation**: Passed via `metadata={"trade_cycle_id": self._current_cycle_id}` to orders
- **Storage**: Tracked in `_register_trade_entry()` and learning hooks
- **Issue**: Exit orders may not always resolve back to entry `trade_cycle_id` if correlation is lost

## Critical Control Points

1. **Entry Risk Guards**: `_validate_entry_guard()` (line 2402) - only called if `FF_ENTRY_RISK_GUARDS=1`
2. **WAIT Blocking**: `should_block_on_wait()` (line 1488) - only blocks if `FF_WAIT_BLOCKING=1` and `block_on_wait=True`
3. **Bracket Validation**: `validate_bracket_prices()` in `order_builder.py` - validates SL/TP are on correct side
4. **Exit Protection**: `_place_exit_order()` sets `stop_loss=None, take_profit=None` - no validation
5. **Emergency Stop**: **MISSING** - no automatic stop placement when fill occurs without bracket

