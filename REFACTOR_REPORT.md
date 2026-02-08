# MyTrader Codebase Refactor Report

**Date:** 2026-02-07  
**Author:** Senior Trading Systems Engineer (handoff document)  
**Branch:** `fix/optimize-spy-strategy-20251216`

---

## 1. Execution Flow — Call Graph

```
start_bot.sh
│
├── Pre-flight checks (IB Gateway alive, config.yaml exists)
├── Parse config.yaml via inline Python to extract IBKR_HOST/PORT
├── Set env vars (PROMETHEUS, MAX_CONTRACTS, guardrails)
├── Optional: guardrail tests via pytest + replay_trade_from_logs.py
│
└── nohup python run_bot.py [--simulation] &
    │
    ├── run_bot.py  (production entry — 95 lines)
    │   ├── parse_args() → --simulation, --config, --cooldown, --reset-state
    │   ├── configure_logging() → logs/live_trading.log, logs/bot.log
    │   ├── load_settings(config_path) → Settings dataclass
    │   ├── LiveTradingManager(settings, simulation_mode, reset_state)
    │   │   ├── __init__(): builds ~20 component helpers
    │   │   │   ├── RiskGate, CooldownManager, StatusBroadcaster
    │   │   │   ├── ContextManager, OrderCoordinator, RiskController
    │   │   │   ├── TradingSessionManager, MarketDataCoordinator
    │   │   │   ├── SignalProcessor, TradeDecisionEngine
    │   │   │   ├── SystemHealthMonitor, TrendContinuationOptimizer
    │   │   │   └── TradeLearningRecorder, AgentBus
    │   │   │
    │   │   └── .start() → delegates to TradingSessionManager.start()
    │   │       │
    │   │       ├── initialize()
    │   │       │   ├── LivePerformanceTracker
    │   │       │   ├── Strategy selection (15m ES / 1m MES / 1m Scoring)
    │   │       │   ├── StrategyEngine([strategy])
    │   │       │   ├── RiskManager(config, sizing_method)
    │   │       │   ├── TelegramNotifier (optional)
    │   │       │   ├── IB() + TradeExecutor + .connect()
    │   │       │   ├── force_order_reconciliation()
    │   │       │   ├── RAGStorage (S3-backed)
    │   │       │   ├── DecisionMetricsLogger (optional)
    │   │       │   ├── Hybrid RAG+LLM Pipeline (optional)
    │   │       │   ├── AWS Bedrock Agents (optional)
    │   │       │   ├── _load_historical_context() → PDH/PDL/WH/WL
    │   │       │   └── _bootstrap_price_history(warmup_bars)
    │   │       │
    │   │       └── _trading_loop() [infinite async loop]
    │   │           │
    │   │           ├── market_data_coordinator.wait_for_next_bar()
    │   │           │   └── LiveDataManager / poll-based bar collection
    │   │           │
    │   │           └── _process_trading_cycle(price, bar_ts)
    │   │               ├── get_current_position()
    │   │               ├── If has position → _check_position_exit_signals()
    │   │               │   ├── Trend-flip exit
    │   │               │   ├── Stop/TP bracket protection
    │   │               │   ├── Trailing stop update
    │   │               │   └── Opposite-signal exit
    │   │               │
    │   │               └── If flat → signal_processor.process_trading_cycle()
    │   │                   ├── engineer_features(price_history)
    │   │                   ├── StrategyEngine.evaluate(features)
    │   │                   ├── MTF Trend Gate filter
    │   │                   ├── Sentiment checks (Stocktwits/multi-source)
    │   │                   ├── RAG validation (knowledge base)
    │   │                   ├── Scoring system evaluation
    │   │                   ├── Hybrid pipeline (3-layer decision)
    │   │                   ├── risk_controller.evaluate_trade()
    │   │                   └── order_coordinator.submit_order()
    │   │                       └── TradeExecutor.place_order()
    │   │                           └── IB bracket order (parent + SL + TP)
    │   │
    │   └── asyncio.run(main())
    │
    └── main.py  (legacy entry — 1306 lines, NOT used by run_bot.py)
        ├── run_live(): monolithic 1200-line trading loop
        └── run_backtest(): backtesting via BacktestingEngine
```

### Key Insight: Two Parallel Entry Points

| Entry Point | Used By | Lines | Status |
|-------------|---------|-------|--------|
| `run_bot.py` | `start_bot.sh` (production) | 95 | **Active** — clean, delegates to LiveTradingManager |
| `main.py` | Manual CLI (`python main.py live`) | 1,306 | **Legacy** — monolithic, duplicates run_bot.py logic |

**Recommendation:** `main.py` should be deprecated or reduced to a thin CLI wrapper over the same `LiveTradingManager`.

---

## 2. Refactor Candidates — Files Flagged

### Critical (>1000 lines, mixed responsibilities)

| File | Lines | Responsibilities | Severity |
|------|-------|-----------------|----------|
| `mytrader/strategies/entry_modules.py` | **5,135** | Session time management, BUY continuation, SHORT continuation, SELL exhaustion, Evening BUY patterns, Evening SELL patterns, IntegratedEntryManager | 🔴 Critical |
| `mytrader/execution/live_trading_manager.py` | **3,843** | Session lifecycle, position tracking, exit logic, trade context, bracket fill inference, status updates, learning hooks, RAG wiring, AWS agents | 🔴 Critical |
| `mytrader/execution/ib_executor.py` | **3,019** | IB connection, contract management, order locking, dedup, event handlers, bracket orders, emergency orders, position queries, trailing stops, PnL tracking, keepalive | 🔴 Critical |
| `mytrader/rag/hybrid_rag_pipeline.py` | **2,761** | RAG retrieval, signal enhancement, vector search, caching | 🟡 High |
| `mytrader/execution/components/signal_processor.py` | **2,529** | Feature engineering orchestration, strategy evaluation, sentiment checks, MTF gating, scoring, hybrid pipeline, VIX feed | 🟡 High |
| `mytrader/data/sentiment_aggregator.py` | **1,756** | Multi-source sentiment collection, caching, scoring, entry/position evaluation | 🟡 High |
| `mytrader/execution/reconcile.py` | **1,377** | Position reconciliation, order matching, state recovery | 🟡 High |
| `main.py` | **1,306** | Legacy entry point — full trading loop (duplicates LiveTradingManager) | 🔴 Critical (dead code risk) |
| `mytrader/strategies/range_reversion_module.py` | **1,179** | Range detection, reversion logic, evening session support | 🟡 Medium |
| `mytrader/strategies/mes_one_minute.py` | **1,017** | 1-minute MES strategy with trend detection, entries, exits | 🟡 Medium |
| `mytrader/config.py` | **945** | 20+ dataclass configs in one file | 🟡 Medium |

### Moderate (400-1000 lines, some mixed concerns)

| File | Lines | Notes |
|------|-------|-------|
| `execution/components/trend_continuation_optimizer.py` | 978 | Bracket modification logic |
| `strategies/market_state.py` | 922 | Market phase detection |
| `monitoring/order_tracker.py` | 883 | SQLite-based order tracking |
| `strategies/scoring_entry.py` | 832 | Scoring system |
| `data/live_data_manager.py` | 803 | Bar collection and streaming |
| `llm/rag_engine.py` | 733 | Vector search + embedding |
| `execution/components/order_coordinator.py` | 713 | Order submission coordination |
| `strategies/multi_strategy.py` | 691 | Multi-strategy voting |
| `data/candle_aggregator.py` | 679 | Multi-timeframe candle building |

---

## 3. Proposed Split Plan

### 3a. `entry_modules.py` (5,135 → 7 files)

```
mytrader/strategies/entry/
  __init__.py                    # Re-exports for backward compat
  session_time.py                # SessionWindow, SessionTimeManager (~200 lines)
  signals.py                     # PullbackAnalysis, EntrySignal dataclasses (~30 lines)
  buy_continuation.py            # BuyContinuationModule (~690 lines)
  short_continuation.py          # ShortContinuationModule (~550 lines)
  sell_exhaustion.py             # SellExhaustionModule (~270 lines)
  evening_continuation.py        # EveningPatternAnalysis, EveningContinuationModule (~980 lines)
  evening_sell_continuation.py   # EveningSellPatternAnalysis, EveningSellContinuationModule (~1000 lines)
  integrated_manager.py          # IntegratedEntryManager (~1350 lines → further split later)
```

### 3b. `config.py` (945 → 5 files)

```
mytrader/config/
  __init__.py           # Re-exports Settings and all configs
  data_sources.py       # DataSourceConfig
  trading.py            # TradingConfig, EntryFilterConfig, RiskGateConfig
  strategy.py           # OneMinuteStrategyConfig, ThirtyMinuteStrategyConfig, StrategyConfig
  integrations.py       # LLMConfig, RAGConfig, TelegramConfig, StockwitsSentimentConfig, etc.
  settings.py           # Settings master dataclass + validate()
```

### 3c. `ib_executor.py` (3,019 → 5 files)

```
mytrader/execution/
  ib_executor.py         # TradeExecutor core class (~500 lines) — init, connect, contract
  executor_models.py     # OrderResult, PositionInfo, CloseFill dataclasses (~80 lines)
  order_lock.py          # Order locking, dedup, submission signatures (~300 lines)
  bracket_orders.py      # Bracket order placement, emergency orders, protection (~700 lines)
  position_tracker.py    # Position queries, trailing stops, PnL tracking (~600 lines)
  event_handlers.py      # _on_order_status, _on_execution callbacks (~250 lines)
  reconnection.py        # Keepalive, reconnect logic (~200 lines)
```

### 3d. `main.py` (1,306 → 2 files)

```
main.py                  # Thin CLI dispatcher (~30 lines) — argparse + delegates
bot/
  legacy_live.py         # run_live() for backward compat (moved, NOT actively used)
  backtest_runner.py     # run_backtest() extracted (~60 lines)
```

### 3e. `live_trading_manager.py` (3,843 → already partially decomposed)

The file already delegates to 10+ component helpers. Remaining in-file logic:
- Exit signal generation (~400 lines) → `exit_signal_handler.py`
- Bracket fill inference (~150 lines) → keep in trade context
- Position transition detection (~100 lines) → keep in `_process_trading_cycle`
- Status/helper methods (~200 lines) → various component files

---

## 4. Risky Assumptions & Hidden Coupling Discovered

### 🔴 Critical Findings

1. **Two entry points with divergent logic**: `run_bot.py` (production) uses `LiveTradingManager` with proper component decomposition. `main.py` has a 1200-line `run_live()` that duplicates this logic with **different behavior** (e.g., inline Bedrock hybrid architecture, different strategy initialization). If anyone runs `python main.py live`, they get different risk management.

2. **Global state in sentiment modules**: `stocktwits_sentiment.py` and `sentiment_aggregator.py` use module-level caches (`_cache`, `_last_fetch_time`). Multiple imports or test harnesses can silently share state.

3. **Import-time side effects**: Several modules catch `ImportError` and set `*_AVAILABLE = False` flags at import time. This means import order matters and test isolation is fragile.

4. **`config.py` is imported everywhere**: The 945-line monolith is imported by ~50 files. Splitting it requires careful re-export via `__init__.py`.

5. **`ib_executor.py` class-level global cache**: `TradeExecutor._global_contract_cache` is a class variable shared across all instances. Safe in production (single instance) but dangerous in tests.

### 🟡 Moderate Findings

6. **Circular import risk**: `live_trading_manager.py` imports from `strategies/`, `execution/components/`, `rag/`, `llm/`, `hybrid/`, `learning/`, `aws/`, `monitoring/`, `risk/`, `data/`. Any of those importing back into `execution/` would break.

7. **Inline imports scattered throughout**: `main.py` has `from datetime import datetime` inside the while loop body (line ~472), and `from dataclasses import dataclass` repeated twice (lines ~775 and ~800).

8. **No explicit interface contracts**: Strategy classes don't share a common protocol/ABC for `generate_signal()`. `MesOneMinuteTrendStrategy`, `EsFifteenMinStrategy`, and `MultiStrategy` all have slightly different signatures.

---

## 5. Refactor Execution Plan

### Phase 1: Safe structural splits (no logic changes)

1. ✅ Split `entry_modules.py` (5,136 lines) → `strategies/entry/` package (8 files + `__init__.py`)
2. ✅ Split `config.py` (946 lines) → `config/` package (8 files + `__init__.py`)
3. ✅ Mark `main.py` as legacy with deprecation header
4. ✅ Backward-compatible re-export shims (entry_modules.py → 46-line shim, config/__init__.py)
5. ✅ Extract `ib_executor.py` dataclasses → `execution/models.py` (54 lines)

### Phase 2: Execution layer cleanup

6. 🔲 Extract order lock/dedup logic from `ib_executor.py` into `execution/order_lock.py`
7. 🔲 Extract exit signal handling from `live_trading_manager.py` (~400 lines)
8. 🔲 Split `signal_processor.py` (2,529 lines) sentiment/MTF helpers into separate modules

### Phase 3: Future opportunities (not in this PR)

- Unify strategy interfaces with a Protocol/ABC
- Replace module-level sentiment caches with injected state
- Add integration tests for the refactored import graph
- Consider `main.py` removal (after confirming no users)

---

## 6. Session Awareness — RTH / Overnight Isolation

The codebase already has clear session separation:

| Module | Session | Purpose |
|--------|---------|---------|
| `BuyContinuationModule` | RTH (9:30-10:45 CST) | Buy pullbacks in bullish acceptance |
| `ShortContinuationModule` | RTH | Short continuation in bearish acceptance |
| `SellExhaustionModule` | RTH | Sell on exhaustion confirmation |
| `EveningContinuationModule` | Overnight (16:00-09:30 CST) | Evening buy patterns |
| `EveningSellContinuationModule` | Overnight | Evening sell patterns |
| `SessionTimeManager` | Both | Centralized window classification |
| `IntegratedEntryManager` | Both | Routes to correct module by session |

**Action:** ✅ Done — each is now in its own file within `strategies/entry/`.

---

## 7. Refactor Summary — What Changed

### Files created (new)

| File | Lines | Purpose |
|------|-------|---------|
| `mytrader/strategies/entry/__init__.py` | 44 | Re-export shim for backward compat |
| `mytrader/strategies/entry/session_time.py` | 163 | SessionWindow enum + SessionTimeManager |
| `mytrader/strategies/entry/signals.py` | 49 | PullbackAnalysis + EntrySignal dataclasses |
| `mytrader/strategies/entry/buy_continuation.py` | 483 | BuyContinuationModule |
| `mytrader/strategies/entry/short_continuation.py` | 574 | ShortContinuationModule |
| `mytrader/strategies/entry/sell_exhaustion.py` | 298 | SellExhaustionModule |
| `mytrader/strategies/entry/evening_buy.py` | 1,027 | EveningPatternAnalysis + EveningContinuationModule |
| `mytrader/strategies/entry/evening_sell.py` | 1,046 | EveningSellPatternAnalysis + EveningSellContinuationModule |
| `mytrader/strategies/entry/integrated_manager.py` | 1,399 | IntegratedEntryManager + create_entry_manager |
| `mytrader/config/__init__.py` | 64 | Re-export shim for backward compat |
| `mytrader/config/data_sources.py` | 28 | DataSourceConfig |
| `mytrader/config/strategy.py` | 300 | EntryFilterConfig, 1m, 30m, StrategyConfig |
| `mytrader/config/risk.py` | 152 | RiskGateConfig, TradingConfig |
| `mytrader/config/backtest.py` | 36 | BacktestConfig, OptimizationConfig |
| `mytrader/config/llm_rag.py` | 157 | LLMConfig, RAGConfig, HybridConfig, AWSAgentsConfig |
| `mytrader/config/integrations.py` | 215 | Sentiment, VIX feed, Telegram configs |
| `mytrader/config/misc.py` | 39 | LearningConfig, FeatureFlagsConfig, ObservabilityConfig |
| `mytrader/config/settings.py` | 113 | Settings root dataclass |
| `mytrader/execution/models.py` | 54 | OrderResult, PositionInfo, CloseFill |

### Files replaced with shims

| File | Before | After | Notes |
|------|--------|-------|-------|
| `mytrader/strategies/entry_modules.py` | 5,136 | 46 | Thin re-export shim |
| `mytrader/config.py` | 946 | *deleted* | Replaced by `config/` package |

### Files modified

| File | Change |
|------|--------|
| `mytrader/execution/ib_executor.py` | Removed inline dataclasses (3,020 → 2,983), imports from `models.py` |
| `main.py` | Added deprecation header pointing to `run_bot.py` |

### Backward compatibility

- **Zero consumer changes required** — all existing imports continue to work
- Identity check passes: objects imported via shim `is` same as via new package
- No circular imports detected (tested with different import orders)

### Safety validation results

- ✅ 81 import checks passed (all names, all paths, all consumers)
- ✅ No circular imports
- ✅ `Settings().validate()` works
- ✅ Key downstream consumers load successfully

---
