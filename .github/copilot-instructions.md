# Copilot Instructions — ShreeBot (MES Futures Trading Bot)

## Architecture Overview

ShreeBot is an autonomous **MES (Micro E-mini S&P 500) futures trading bot** connecting to Interactive Brokers via `ib_insync`. All timestamps use **Central Time (America/Chicago)**—use `shree.utils.timezone_utils.now_cst()`, never `datetime.now()`.

**Entry point:** `run_bot.py` → `LiveTradingManager` (the 3,200-line orchestrator in `shree/execution/live_trading_manager.py`). The legacy `main.py` is **deprecated**—do not add new logic there.

### Core Decision Pipeline (3-layer hybrid)

1. **D-Engine** (`shree/hybrid/d_engine.py`) — deterministic technical signals (EMA pullback, OR breakout, trend continuation)
2. **H-Engine** (`shree/hybrid/h_engine.py`) — LLM + RAG confirmation via AWS Bedrock (`shree/llm/bedrock_client.py`, `shree/rag/`)
3. **Confidence Scorer** (`shree/hybrid/confidence.py`) — merges D+H signals, applies penalties, produces final confidence

Trades are gated by `RiskGate` (`shree/risk/risk_gate.py`) which enforces hard dollar limits, margin checks, and session rules **before** any order reaches IBKR.

### Key Module Map

| Directory | Purpose |
|---|---|
| `shree/strategies/` | Signal generators. **Active strategy: `es_fifteen_min.py`** (EMA21 pullback + OR breakout on 15m bars). Legacy 1m strategies (`mes_one_minute*.py`) are retired. |
| `shree/execution/` | Order lifecycle. `LiveTradingManager` delegates to `components/` submodules (cooldown, risk, signal processing, order coordination). |
| `shree/risk/` | `RiskGate` (hard limits), `RiskManager` (position sizing), `atr_module.py` (ATR-based stop/target). |
| `shree/hybrid/` | D-engine + H-engine orchestration, `AgentBus` for cross-agent messaging. |
| `shree/rag/` | Local RAG pipeline: `local_knowledge_base.py`, `embedding_builder.py`, `pipeline_integration.py`. |
| `shree/data/` | `live_data_manager.py` (tick subscriptions), `candle_aggregator.py` (multi-timeframe: 1m→5m/15m/30m). |
| `shree/config/` | Split dataclass configs. `Settings` in `settings.py` aggregates all sub-configs loaded from `config.yaml`. |
| `shree/features/` | `feature_engineer.py` — computes EMA, RSI, ATR, ADX, Stochastic on OHLCV DataFrames. Strategies import `_ema`, `_atr`, `_adx`, `_rsi` directly. |
| `backtest/` | Top-level backtest framework: `engine.py` reuses the same strategy/risk logic as live. Run via `python -m backtest.run`. |

## Configuration

All runtime config lives in `config.yaml` (never committed with secrets). Config dataclasses are in `shree/config/` submodules. Key patterns:
- **Environment variable overrides** — `RiskGateConfig` fields read from env (e.g., `MAX_MES_CONTRACTS`, `DAILY_MAX_LOSS_USD`, `RISK_PER_TRADE_USD`). `settings_loader.py` also supports env overrides for RAG/trading thresholds.
- **Position limits are consolidated** — `Settings.validate()` takes the most conservative value across `risk_gate.max_contracts`, `trading.max_position_size`, and `trading.max_contracts_limit`.
- When adding a new config field, add it to the appropriate `shree/config/*.py` dataclass, then expose it in `config.yaml` and `config.example.yaml`.

## Coding Conventions

- **Logging:** Use `from shree.utils.logger import logger` (loguru). Structured events via `log_structured_event()` from `shree/utils/structured_logging.py`.
- **Strategy pattern:** Subclass `BaseStrategy` (`shree/strategies/base.py`), implement `generate(features: pd.DataFrame) -> Signal`. The `Signal` dataclass has `action` (BUY/SELL/HOLD), `confidence` (0–1), `metadata` (dict).
- **Risk values are in points, ticks, or USD** — comments always clarify which unit. MES = $5/point, $1.25/tick, tick_size=0.25.
- **Feature flags** control experimental features via env vars (`FF_ENTRY_RISK_GUARDS`, `FF_WAIT_BLOCKING`, `FF_EXIT_GUARDS`, `FF_LEARNING_HOOKS`). Check `FeatureFlagsConfig` in `shree/config/misc.py`.
- **Graceful imports** — external integrations (AWS, hybrid pipeline) use `try/except ImportError` with fallback flags like `HYBRID_PIPELINE_AVAILABLE`, `AWS_AGENTS_AVAILABLE`.

## Commands

```bash
# Live trading (requires IB Gateway on port 4001)
. start_bot.sh                # Production start (sources .venv, checks IB Gateway, backgrounds process)
. stop.sh                     # Graceful shutdown

# Simulation (no real orders)
SHREE_SIMULATION=1 . start_bot.sh
python run_bot.py --simulation

# Backtesting
python -m backtest.run --symbol MES --start 2025-02-01 --end 2026-01-31 --bar 15m
./start_backtest.sh           # Wrapper with defaults

# Tests
python -m pytest tests/ -x    # Run all tests (no live IB needed)
ENABLE_GUARDRAILS=1 . start_bot.sh  # Runs guardrail tests before starting
```

## Critical Safety Rules

- **Max 1 contract** for automated MES trading — enforced at config validation, `RiskGate`, and executor level.
- **CME maintenance window 4–5 PM CT** — hard block on all entries. Respect `avoid_close_window_minutes` before maintenance.
- **Cooldowns are layered:** base cooldown → loss cooldown → consecutive-loss cooldown. See `CooldownManager` in `shree/execution/components/cooldown_manager.py`.
- **Never bypass `RiskGate`** — it is the last line of defense. If adding a new entry path, it must call `RiskGate.evaluate_entry()`.
- **Stop-loss bounds:** min 6 points, max 25 points ($30–$125 risk). The 15m strategy uses 1.5×ATR for stops.

## Testing Patterns

Tests are in `tests/` using `pytest` + `pytest-mock`. Most tests construct config objects directly (no YAML loading). Example pattern from `test_risk_gate.py`:
```python
cfg = RiskGateConfig()
cfg.min_stop_points = 2.0
gate = RiskGate(cfg)
account = {"available_funds": 5000.0, "realized_pnl_today": 0.0}
result = gate.evaluate_entry(action="BUY", entry_price=5000.0, ...)
assert not result.allowed
```

## Things to Watch Out For

- `LiveTradingManager` is 3,200+ lines — changes there should be surgical. Prefer adding logic to `shree/execution/components/` submodules.
- The `backtest/engine.py` imports strategy classes directly — if you rename or restructure a strategy, update the backtest imports.
- Config has **duplicate risk parameters** in `trading` and `risk_gate` sections; `Settings.validate()` reconciles them. Don't assume one is authoritative without checking.
- `main.py` is legacy but still importable — don't break its imports even though it's deprecated.
