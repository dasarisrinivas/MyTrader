# Copilot Instructions — ShreeBot (Multi-Strategy Futures Trading Platform)

## Architecture — Three Independent Bots

ShreeBot runs **three independent trading bots** sharing one IB Gateway (`127.0.0.1`). Each has its own entry point, IB client_id, log files, and state:

| Bot | Entry Point | IB client_id | Contract | Status |
|---|---|---|---|---|
| **MES** | `run_bot.py` → `LiveTradingManager` | 11 (exec), 71 (VX), 1 (data) | MES (Micro E-mini S&P 500) | Live — places orders |
| **Gold** | `run_gold.py` → `GoldTradingManager` | 3 | MGC (Micro Gold) / GC | Live — places orders |
| **SPY Options** | `run_spy_options.py` → `SpyOptionsManager` | 5 | SPY options chain | Signal-only — Telegram alerts, no orders |

All connect via `ib_insync`. All timestamps must use `shree.utils.timezone_utils.now_cst()` — **never** `datetime.now()`.

### MES Pipeline (15m bars)
```
es_fifteen_min.generate() → signal_processor.py → hybrid_rag_pipeline.py
  → live_trading_manager._process_hybrid_signal() → order_coordinator.py
  → ib_executor.py (bracket order) → exit_manager.py (breakeven/profit locks)
```
Signals A–G + A-prime/D-prime. See `docs/SIGNAL_OPTIMIZATION_REMAINING.md` for full signal reference.

### Gold Pipeline (1m bars aggregated from 5s)
```
GoldIntradayStrategy.generate() → GoldRiskManager → GoldTradingManager
  → IB bracket order → staged time stop (20/40/60 bar)
```
Signal families: VWAP/EMA pullback + ORB. Session-aware 5-bucket system (OVERNIGHT/LONDON_OPEN/COMEX_OPEN/MIDDAY/PRE_CLOSE). See `docs/GOLD_TRADING_IMPROVEMENTS_ROADMAP.md`.

---

## Quick Commands

```bash
# MES bot
. start_bot.sh               # Live (port 4001)
. start_paper_bot.sh          # Paper (port 4002)
. stop.sh                     # Graceful shutdown

# Gold bot
. start_gold.sh               # Paper by default (port 4002)
GOLD_SIMULATION=0 . start_gold.sh  # Live
. stop_gold.sh

# SPY Options
. start_spy_options.sh

# Tests
python3 -m pytest tests/ -x                  # All (no IB needed)
python3 -m pytest tests/gold/ -v             # Gold-specific (18 test files)
python3 -m pytest tests/ -k "overnight"      # By keyword

# Backtesting
python3 -m backtest.run --symbol MES --start 2025-02-01 --end 2026-01-31 --bar 15m

# Trade journal (auto-ingested daily at 3 PM CT via launchd)
python3 scripts/daily_journal.py --report YYYY-MM-DD YYYY-MM-DD
sqlite3 data/trade_journal.db "SELECT date, signals_generated, trades_taken, realized_pnl FROM daily_summary ORDER BY date DESC LIMIT 10"
```

**Runtime selection rule:** During **RTH** → use live bot scripts. **Outside RTH** → use paper bot scripts. When user says "restart the bot" without specifying, infer from session time.

---

## Project Conventions

- **Always `python3`** — never bare `python`. Python 3.11+.
- **Logging:** `from shree.utils.logger import logger` (loguru). Structured events via `log_structured_event()`.
- **Strategy pattern:** Subclass `BaseStrategy` (`shree/strategies/base.py`), implement `generate(features: pd.DataFrame) -> Signal`. Signal = `{action, confidence, metadata}`.
- **Config:** Dataclasses in `shree/config/*.py`, loaded from `config.yaml` via `settings_loader.py`. Root: `Settings` in `settings.py`. Paper overrides: `config.paper.yaml`.
- **Risk values** in points, ticks, or USD — comments always clarify unit. MES=$5/point, MGC=$1/point, tick_size=0.25.
- **Feature flags** via env vars: `FF_ENTRY_RISK_GUARDS`, `FF_EXIT_GUARDS`, `ENABLE_CHOP_EXCEPTION`. See `FeatureFlagsConfig` in `shree/config/misc.py`.
- **Graceful imports** — external integrations use `try/except ImportError` with fallback flags (`HYBRID_PIPELINE_AVAILABLE`, `AWS_AGENTS_AVAILABLE`).
- **Tests** construct config objects directly (no YAML loading). Pattern: `cfg = RiskGateConfig(); cfg.field = value; gate = RiskGate(cfg)`.

### Adding a New Config Field
1. Add to appropriate `shree/config/*.py` dataclass
2. Add to `config.yaml` + `config.example.yaml` + `config.paper.yaml` if applicable
3. If risk-related: ensure `Settings.validate()` reconciles it (duplicate params exist in `trading:` and `risk_gate:`)

---

## Key Module Map

| Directory | Purpose |
|---|---|
| `shree/strategies/es_fifteen_min.py` | MES signal generator (Signals A–G, 15m bars) |
| `shree/strategies/gold/` | Gold signal generator (`signals.py`), regime detector (`regime.py`), strategy wiring (`strategy.py`) |
| `shree/spy_options/` | SPY Options signal engine, IB client, chain builder, sweep tracker |
| `shree/execution/live_trading_manager.py` | MES orchestrator (3,200+ lines — prefer surgical edits, add to `components/`) |
| `shree/execution/components/` | Signal processor, order coordinator, exit manager, cooldown, sentiment |
| `shree/execution/gold/` | Gold manager, risk, state, journal, contract rollover |
| `shree/risk/` | `risk_gate.py` (9-layer gate), `dynamic_support.py`, `trade_math.py` |
| `shree/config/` | All config dataclasses. `settings.py` (root), `gold.py` (527 lines), `spy_options.py` |
| `shree/data/` | Candle aggregation, sentiment (Stocktwits+Reddit), VX futures feed |
| `shree/rag/` | Hybrid RAG pipeline (advisory only, max -0.05 confidence dampen) |
| `backtest/` | Engine reuses live strategy/risk. `walk_forward.py` for WFO grid search |
| `shree/utils/news_calendar.py` | ForexFactory feed → news lockout windows for Gold |

---

## Critical Safety Rules

- **Max 1 contract** per bot — enforced at config validation, RiskGate, and executor level
- **Never bypass `RiskGate`** — every new entry path must call `RiskGate.evaluate_entry()`
- **CME maintenance 4–5 PM CT** — hard block on all entries
- **Daily loss cap:** MES $250/day (5% of $5K), Gold has separate caps in `gold:` config
- **Cooldowns are layered:** base 10min → loss 20min → consecutive-loss (3 in a row) 30min
- **Overnight guards** (MES): RSI extreme block, MACD divergence block, ATR floor — all in `es_fifteen_min.generate()`, all RTH-exempt. Thresholds use strict inequality (`<` not `<=`)
- **Gold session buckets** control per-session entry enable/disable, confidence offsets, and SL/TP multipliers. Config: `shree/config/gold.py :: GoldSessionBucketConfig`

---

## ⛔ Date Awareness Rule

Copilot MUST NOT state a day-of-week from a date without verifying:
```bash
python3 -c "from datetime import date; d=date(YYYY,M,D); print(d.strftime('%A %b %d %Y'))"
```
Hallucinating "Monday" vs "Tuesday" causes real operational errors (missed contract rolls, wrong event prep).

---

## Diagnosing Issues

| Question | Command |
|---|---|
| Is the MES bot running? | `pgrep -f "python.*run_bot.py"` or `cat logs/bot.pid` |
| Is the Gold bot running? | `pgrep -f "python.*run_gold.py"` |
| Why no MES trades? | `grep -E "NO_SIGNAL diag\|BLOCKED\|CHOP" logs/live_trading.log \| tail -10` |
| Why no Gold trades? | `grep -E "HOLD\|cooldown\|BLOCKED" logs/gold_trading.log \| tail -10` |
| Recent MES trades? | `grep "order_placed" logs/reconcile.log \| tail -10` |
| Performance history? | `sqlite3 data/trade_journal.db` (prefer DB over log grep) |
| Contract roll needed? | `grep "Qualified contract" logs/live_trading.log \| tail -2` |
| VX feed alive? | `grep "VX Feed: Price" logs/live_trading.log \| tail -1` |

When user asks about performance/P&L/trades — **always query `data/trade_journal.db` first**, not logs.

---

## Pitfalls & Gotchas

- The `one_minute:` config section holds all `ft_*` (fifteen-minute) params — historical naming, do not rename
- `LiveTradingManager` is 3,200+ lines — add new logic to `shree/execution/components/` submodules instead
- Config has **duplicate risk params** in `trading:` and `risk_gate:` — `Settings.validate()` reconciles to most conservative
- `archive/main.py` is legacy — still importable but deprecated, don't add logic there
- Opening Range computed from first 2 bars after **ETH open (17:00 CT)**, not RTH open
- VX feed uses separate IB connection (client_id=71) — if disconnected, MES continues with zero VX adjustment
- Gold bot's `GoldTradingManager` is fully independent (1,320 lines) — shares no runtime state with MES bot
- MES and Gold bots can run simultaneously on the same IB Gateway (different client_ids)
- `backtest/engine.py` imports strategy classes directly — update imports if renaming strategies
- See `docs/SIGNAL_OPTIMIZATION_REMAINING.md` for MES optimization roadmap and pending fixes
