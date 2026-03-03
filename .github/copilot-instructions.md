# Copilot Instructions — ShreeBot (MES Futures Trading Bot)

## Quick Reference — "How do I check…?"

| Question | Command / Location |
|---|---|
| Is the bot running? | `pgrep -f "python.*run_bot.py"` or `cat logs/bot.pid` |
| Current price / market state? | `tail -50 logs/live_trading.log \| grep -E "Heartbeat\|price="` |
| Why is it not trading? | `grep -E "NO_SIGNAL diag\|BLOCKED\|CHOP_REGIME\|HOLD" logs/live_trading.log \| tail -20` |
| Today's signals & decisions? | `grep -E "Signal:\|Pipeline result\|SCORE_DEBUG\|SCORING" logs/live_trading.log \| tail -30` |
| Sentiment right now? | `grep "MULTI-SOURCE SENTIMENT SUMMARY" -A8 logs/live_trading.log \| tail -10` |
| VX / volatility multiplier? | `grep "VX Feed: Price" logs/live_trading.log \| tail -1` |
| Today's Opening Range? | `cat data/or_$(date +%Y-%m-%d).json` |
| Dynamic support floor? | `grep "Dynamic support floor" logs/live_trading.log \| tail -1` |
| Historical context (PDH/PDL)? | `grep "Historical context loaded" -A4 logs/live_trading.log \| tail -5` |
| Active orders / position? | `grep -E "Sync:.*active orders\|Position:" logs/live_trading.log \| tail -3` |
| Recent trades? | `grep "order_placed" logs/reconcile.log \| tail -10` |
| Risk gate blocks? | `grep -E "RiskGate\|BLOCKED\|evaluate_entry" logs/live_trading.log \| tail -10` |
| Today's P&L? | `grep "realized_pnl\|Daily P&L" logs/live_trading.log \| tail -5` |
| Cooldown status? | `grep "cooldown\|Cooldown" logs/live_trading.log \| tail -5` |
| Start / stop the bot? | `. start_bot.sh` / `. stop.sh` |
| Run tests? | `python3 -m pytest tests/ -x` |
| Backtest? | `python3 -m backtest.run --symbol MES --start 2025-02-01 --end 2026-01-31 --bar 15m` |
| Optimization roadmap? | `docs/SIGNAL_OPTIMIZATION_REMAINING.md` |

---

## Optimization Roadmap (Phase 3+4 Timeline)

See `docs/SIGNAL_OPTIMIZATION_REMAINING.md` for full details. Key decision gates:

| Date | Gate | Action |
|---|---|---|
| **Mar 10** | Touch-band near-misses ≥ 10? | → Implement Fix #6 (ATR touch band for A/D signals) |
| **Mar 17** | 2-week review: trades/day < 2? | → Start Fix #3 (day-type classifier) |
| **Mar 24** | Phase 3 checkpoint | → Deploy Fix #7 (doji) + #8 (F cap) if #3 is done |
| **Apr 1** | Signal variety insufficient? | → Start Fix #9 (OR continuation pattern) |

When the current date matches or passes a gate date, **proactively suggest** checking the gate criteria and starting the corresponding work.

---

## Architecture Overview

ShreeBot is an autonomous **MES (Micro E-mini S&P 500) futures trading bot** connecting to Interactive Brokers via `ib_insync`. All timestamps use **Central Time (America/Chicago)** — use `shree.utils.timezone_utils.now_cst()`, never `datetime.now()`.

**Entry point:** `run_bot.py` → `LiveTradingManager` (the 3,200-line orchestrator in `shree/execution/live_trading_manager.py`). The legacy `main.py` is **deprecated** — do not add new logic there.

### Core Decision Pipeline (every 15m bar)

```
15m Bar Close
  → es_fifteen_min.py :: generate()        # Signals A–F (deterministic)
  → signal_processor.py                    # Hybrid overlay, sentiment, VX scaling, CHOP guard
  → hybrid_rag_pipeline.py :: process()    # Rule engine + RAG retriever + LLM (advisory)
  → live_trading_manager._process_hybrid_signal()  # Confidence threshold gate
  → order_coordinator.py                   # Support floor, risk gate, order building
  → ib_executor.py                         # IBKR bracket order (SL + TP)
  → exit_manager.py                        # Time stops, profit locks, emergency exits
```

### Signal Types (A–F)

| Signal | Type | Direction | Entry Condition |
|---|---|---|---|
| **A** | EMA21 Pullback | LONG | EMA21 > EMA50, bar low touches EMA21, bullish close, ADX 18–45 |
| **B** | OR Breakout | LONG | Close crosses above OR High, EMA9 > EMA21, ADX > 18 |
| **C** | EMA9 Pullback | LONG | Shallow dip to EMA9 in uptrend, MACD > 0, tighter stop |
| **D** | EMA21 Pullback | SHORT | Mirror of A in downtrend (EMA21 < EMA50), ADX 18–45 |
| **E** | OR Breakdown | SHORT | Close crosses below OR Low, EMA9 < EMA21, ADX > 18 |
| **F** | Trend Continuation | LONG/SHORT | Price runs from EMA21, EMA stack aligned, 2+ ascending/descending closes, ADX >= 25, MACD confirms |

**Note (Mar 2026):** MACD and RSI filters removed from A/B/D/E signals to reduce over-filtering. MACD intentionally kept on C (shallow pullback needs momentum confirmation) and F (trend continuation needs momentum). RSI kept on C only.

**Priority:** A > C > B > F_long > D > E > F_short

**Stop/Target (fixed-point system, Feb 2026):**
- A/B/D/E: SL=6pts ($30), TP=8pts ($40) — R:R 1.33:1
- C: SL=ATR-adaptive (8–20pt), TP=SL x 1.25 — R:R 1.25:1
- F: SL=ATR-adaptive (8–20pt), TP=SL x 1.25 — R:R 1.25:1

### Confidence Flow

```
Strategy base confidence (0.70)
  → Hybrid overlay: agree (+0), uncertain (-0.05), oppose (-0.05 max)
  → Sentiment adjustment: REDUCE_SIZE (x0.7), BLOCK (→HOLD), PROCEED (x1.0–1.1)
  → VX additive scaling (signal-type-aware, MAR 2026):
      VX < 16:  continuation -0.08, others -0.03
      VX 16-22: neutral (0.0)
      VX 22-28: breakout +0.03, continuation +0.05, pullback -0.05
      VX 28-35: breakout 0.0, others -0.08
      VX 35+:   breakout -0.03, others -0.15
  → CHOP guard (block-all): ALL pullback/trend_cont in CHOP → HOLD
    Optional exception (OFF by default, ENABLE_CHOP_EXCEPTION=1):
      LONG + ADX≥25 + BULLISH bias + conf≥0.70 + ATR expanding → dampen −0.05
  → Final confidence must be >= 0.40 (min_confidence_for_trade in LTM)
```

### Risk Layers (9 gates, evaluated in order)

1. **Session gate** — CME maintenance 4–5 PM CT blocks all entries
2. **Daily loss cap** — $250/day (5% of $5K capital)
3. **Weekly loss cap** — $500/week (10% of capital)
4. **Margin check** — available_funds > initial_margin + $1000 buffer
5. **Stop-loss bounds** — min 6pts, max 25pts
6. **Risk per trade** — $25 min, $125 max (stop_distance x $5)
7. **Consecutive losses** — 5 in a row → halt
8. **Peak drawdown** — 4% intraday → halt + flatten
9. **Max contracts** — hard cap at 1 MES contract

---

## Key Module Map

| Directory | Purpose |
|---|---|
| `shree/strategies/` | Signal generators. **Active: `es_fifteen_min.py`** (Signals A–F on 15m bars). Legacy 1m strategies are retired. |
| `shree/execution/` | Order lifecycle. `LiveTradingManager` → `components/` submodules. |
| `shree/execution/components/` | `signal_processor.py` (hybrid overlay, sentiment, scoring), `order_coordinator.py` (floor checks, order building), `exit_manager.py` (time stops, profit locks), `cooldown_manager.py`, `sentiment_evaluator.py`, `trading_session_manager.py` |
| `shree/risk/` | `risk_gate.py` (9-layer gate), `dynamic_support.py` (auto-computed floor), `atr_module.py`, `trade_math.py` |
| `shree/rag/` | `hybrid_rag_pipeline.py` (rule engine + RAG + LLM), `pipeline_integration.py` (wiring to LTM), `embedding_builder.py` (FAISS), `s3_storage.py` |
| `shree/data/` | `candle_aggregator.py` (15m bar construction), `sentiment_aggregator.py` (Stocktwits + Reddit + VIX), `vx_futures_feed.py` (real-time VX from IBKR) |
| `shree/config/` | Split dataclass configs in `settings.py` / `risk.py` / `strategy.py` / `misc.py`. Loaded from `config.yaml` via `settings_loader.py`. |
| `shree/features/` | `feature_engineer.py` — computes EMA, RSI, ATR, ADX, MACD on OHLCV DataFrames. |
| `shree/hybrid/` | `d_engine.py` (deterministic), `h_engine.py` (LLM+RAG), `confidence.py` (merger), `multi_factor_scorer.py` |
| `shree/monitoring/` | `order_tracker.py` (SQLite order log), `pnl_calculator.py` |
| `backtest/` | `engine.py` reuses same strategy/risk logic as live. Run via `python3 -m backtest.run`. |
| `agent/` | Autonomous analysis agent (separate process, `start_analyst.sh`). |
| `tools/` | Trade replay, order checking, historical data download utilities. |
| `scripts/` | Ops scripts: backup, IB status check, metrics, data download. |

---

## Configuration

All runtime config lives in **`config.yaml`** (~1000 lines, never committed with secrets). Config dataclasses in `shree/config/` submodules.

### Config Sections Quick Map

| Section | Key Fields | Notes |
|---|---|---|
| `data:` | `ibkr_host`, `ibkr_port` (4001=live, 4002=paper), `ibkr_client_id` | IBKR connection |
| `trading:` | `max_position_size`, `max_daily_loss`, `min_confidence_for_trade`, `entry_filters:`, `ft_*` params | Strategy + risk params |
| `risk_gate:` | `risk_per_trade_usd`, `daily_max_loss_usd`, `weekly_max_loss_usd`, `min/max_stop_points` | Hard dollar limits |
| `dynamic_support:` | `buffer_points`, `use_pdl`, `use_weekly_low`, `use_or_low` | Auto-computed support floor |
| `one_minute:` | `use_15m_strategy: true`, all `ft_*` params | 15m strategy params (despite section name) |
| `multi_source_sentiment:` | Stocktwits/Reddit weights, thresholds | Sentiment aggregation |
| `vix_feed:` | `client_id: 71`, thresholds (20=elevated, 30=extreme) | VX futures feed |
| `rag:` | `backend: local_faiss`, FAISS settings | RAG pipeline |
| `llm:` | `enabled: false` (disabled — too conservative) | AWS Bedrock LLM |
| `hybrid_rag_pipeline:` | Session-specific params (RTH, evening, overnight, premarket) | Pipeline tuning |

### Environment Variable Overrides

```bash
MAX_MES_CONTRACTS=1          # Risk gate hard cap
DAILY_MAX_LOSS_USD=250       # Daily loss limit
RISK_PER_TRADE_USD=100       # Per-trade risk
SHREE_SIMULATION=1           # Simulation mode (no real orders)
HIGH_IMPACT_DATES=2026-03-15,2026-03-20  # Block entries on event dates
FF_ENTRY_RISK_GUARDS=1       # Feature flag: entry risk guards
FF_EXIT_GUARDS=1             # Feature flag: exit guards
```

### Adding a New Config Field

1. Add field to appropriate `shree/config/*.py` dataclass
2. Add to `config.yaml` and `config.example.yaml`
3. If it affects risk: make sure `Settings.validate()` considers it
4. Test with direct config construction (no YAML loading needed in tests)

---

## Log Files & Databases

### Log Files (`logs/`)

| File | Content |
|---|---|
| `live_trading.log` | **Primary runtime log** — signals, decisions, orders, errors, heartbeats |
| `bot.log` | Audit log (same content, tee'd) |
| `bot.pid` | PID of running bot process |
| `reconcile.log` | Structured JSON: every `order_placed`, `order_inserted` event |
| `decisions.csv` | Historical decision log |
| `live_trading.YYYY-MM-DD_HH-MM-SS.log` | Rotated live trading logs from previous runs |
| `backtest.*.log` | Backtest run logs |

### Databases (`data/`)

| File | Content |
|---|---|
| `orders.db` | SQLite: order tracker (all orders placed, fills, cancels) |
| `llm_trades.db` | SQLite: LLM trade logger (decisions, outcomes) |
| `rag_storage.db` | SQLite: RAG trade patterns and outcomes |

### Market Data Files (`data/`)

| File | Content |
|---|---|
| `or_YYYY-MM-DD.json` | Today's Opening Range: `{"date": "...", "or_high": N, "or_low": N}` |
| `es_historical.csv` | Historical OHLCV data for backtesting |

---

## Reading Log Output — Key Patterns

### Signal Diagnostics (NO_SIGNAL)
```
NO_SIGNAL diag: A:ema21(6812.2)<=ema50(6815.4) | B:no_cross(c=6810.0,prev=6808.8,OR_H=6833.0) | D:adx(17)out[18.0-45.0] | E:no_cross(...) | F:stack(...)
```
Each signal shows WHY it didn't fire. Common reasons:
- `A:ema21<=ema50` — no uptrend (EMA alignment wrong)
- `A:low(N)>touch(N)` — bar didn't pull back to EMA21
- `A:bearish(c,o)` — close <= open (not a bullish candle)
- `A:adx(N)out[min-max]` — ADX outside 18–45 range
- `B:no_cross(c,prev,OR_H)` — price didn't cross OR High this bar
- `B:maxed(N)` — max OR breakouts per day reached
- `D:ema21>=ema50` — no downtrend for short pullback
- `D:bullish(c,o)` — close >= open (not bearish candle, needed for short)
- `E:no_cross(c,prev,OR_L)` — price didn't cross OR Low
- `F:stack(e9,e21,e50)` — EMAs not stacked for trend continuation
- `F:adx(N)<25` — ADX too low for trend continuation
- `F:asc(c1,c2,c3)` — not enough ascending/descending closes

### Pipeline Result
```
SCORE_DEBUG: buy=12.0, sell=0.0, threshold=15, scalp_mode=True, daily_bias=BEARISH
SCORE_BREAKDOWN: TREND_SCORE:-30(...) | RANGE_RSI<48:+12.0 | RSI_NEUTRAL(45.9) | ...
Pipeline result: SCALP_SELL (conf=13%, time=377ms)
```

### Confidence Adjustments
```
Hybrid OPPOSES (SELL vs BUY): conf dampen -0.050 -> 0.650
SENTIMENT-TREND: bearish sentiment (+0.03)
VX Adjustment: VX=24.5 | 0.700 +0.030 = 0.730 (breakout)
VX Adjustment: VX=30.0 | 0.700 -0.080 = 0.620 (pullback)
VX Neutral: VX=19.0 — no adjustment (signal_type=other)
CHOP Block-All Guard Activated: blocking BUY pullback signal (EMA21_PB_LONG) in CHOP regime
CHOP Block-All Guard Activated: blocking SELL trend_cont signal (TREND_CONT_SHORT) in CHOP regime
CHOP Exception Activated: LONG allowed | ADX=28 | conf=0.700→0.650 | bias=BULLISH | ATR_expanding=True
BLOCKED: Signal confidence 0.11 < threshold 0.15
```

### Dynamic Support Floor
```
Dynamic support floor updated: 6823.50 (lowest=WL=6828.50 - 5.0pt buffer) [sources: PDL=6870.75, WL=6828.50] trigger=historical_context
```

### Heartbeat (normal operation)
```
Heartbeat: waiting for 15m bar | bars=63 | price=6810.0
```

---

## Common User Questions — How to Answer

### "How is the market performing?" / "What's happening?"
1. `tail -50 logs/live_trading.log` — latest heartbeat shows price + bar count
2. `grep "VX Feed: Price" logs/live_trading.log | tail -1` — volatility regime
3. `grep "Historical context loaded" -A4 logs/live_trading.log | tail -5` — PDH/PDL/weekly range
4. `cat data/or_$(date +%Y-%m-%d).json` — today's Opening Range
5. `grep "Dynamic support floor" logs/live_trading.log | tail -1` — current floor
6. `grep "NO_SIGNAL diag" logs/live_trading.log | tail -3` — why no signals
7. `grep "MULTI-SOURCE SENTIMENT SUMMARY" -A8 logs/live_trading.log | tail -10` — sentiment

### "Why isn't the bot trading?"
Check in this order:
1. **HOLD signals**: `grep "NO_SIGNAL diag" logs/live_trading.log | tail -5` — signal conditions not met
2. **Confidence too low**: `grep "BLOCKED.*confidence" logs/live_trading.log | tail -5`
3. **CHOP regime**: `grep "CHOP Block-All Guard\|CHOP Exception" logs/live_trading.log | tail -5`
4. **Risk gate block**: `grep "RiskGate" logs/live_trading.log | tail -5`
5. **Cooldown active**: `grep -i "cooldown" logs/live_trading.log | tail -5`
6. **Startup grace**: `grep "waiting for.*completed bars" logs/live_trading.log | tail -3`
7. **Sentiment block**: `grep "BLOCK\|REDUCE_SIZE" logs/live_trading.log | tail -5`
8. **Maintenance window**: check if 4–5 PM CT

### "What trades happened today?"
```bash
grep "order_placed" logs/reconcile.log | tail -20
grep "order_inserted" logs/reconcile.log | tail -5
```

### "Is the bot connected to IBKR?"
```bash
grep "Connected to IBKR\|connection keepalive\|Sync:.*active orders" logs/live_trading.log | tail -5
```

---

## Coding Conventions

- **Always use `python3`** — never bare `python` in commands, scripts, shebangs, or documentation. Python 3.11+.
- **Logging:** `from shree.utils.logger import logger` (loguru). Structured events via `log_structured_event()` from `shree/utils/structured_logging.py`.
- **Timezone:** Always `from shree.utils.timezone_utils import now_cst`. Never `datetime.now()`.
- **Strategy pattern:** Subclass `BaseStrategy` (`shree/strategies/base.py`), implement `generate(features: pd.DataFrame) -> Signal`. The `Signal` dataclass: `action` (BUY/SELL/HOLD), `confidence` (0–1), `metadata` (dict).
- **Risk values are in points, ticks, or USD** — comments always clarify which unit. MES = $5/point, $1.25/tick, tick_size=0.25.
- **Feature flags** control experimental features via env vars (`FF_ENTRY_RISK_GUARDS`, `FF_WAIT_BLOCKING`, `FF_EXIT_GUARDS`, `FF_LEARNING_HOOKS`). Check `FeatureFlagsConfig` in `shree/config/misc.py`.
- **Graceful imports** — external integrations use `try/except ImportError` with fallback flags like `HYBRID_PIPELINE_AVAILABLE`, `AWS_AGENTS_AVAILABLE`.

---

## Commands

```bash
# Bot Management
. start_bot.sh                # Production start (sources .venv, checks IB Gateway, backgrounds)
. stop.sh                     # Graceful shutdown (SIGTERM -> SIGKILL after 5s)

# Simulation (no real orders)
SHREE_SIMULATION=1 . start_bot.sh
python3 run_bot.py --simulation

# Backtesting
python3 -m backtest.run --symbol MES --start 2025-02-01 --end 2026-01-31 --bar 15m
./start_backtest.sh           # Wrapper with defaults

# Tests
python3 -m pytest tests/ -x                    # All tests (no live IB needed)
python3 -m pytest tests/test_risk_gate.py -v   # Single test file
python3 -m pytest tests/ -k "dynamic_support"  # By keyword
ENABLE_GUARDRAILS=1 . start_bot.sh             # Runs guardrail tests before starting

# Debugging / Inspection
python3 check_dbs.py          # Inspect SQLite databases
python3 inspect_parquet.py    # Inspect parquet data files
python3 analyze_trades.py     # Analyze recent trade outcomes
python3 test_ib_connections.py # Test IBKR connectivity
```

---

## Critical Safety Rules

- **Max 1 contract** for automated MES trading — enforced at config validation, `RiskGate`, and executor level.
- **CME maintenance window 4–5 PM CT** — hard block on all entries. Respect `avoid_close_window_minutes` (20 min) before maintenance.
- **Cooldowns are layered:** base 10min → loss 20min → consecutive-loss 45min. See `CooldownManager` in `shree/execution/components/cooldown_manager.py`.
- **Never bypass `RiskGate`** — it is the last line of defense. If adding a new entry path, it must call `RiskGate.evaluate_entry()`.
- **Stop-loss bounds:** min 6 points ($30), max 25 points ($125). The 15m strategy uses fixed-point or ATR-adaptive stops.
- **Dynamic support floor** auto-computes from `min(PDL, weekly_low, OR_low) - buffer`. It replaces the old hardcoded `structural_support_floor`.
- **Account:** $5K capital, single MES contract. $250 daily loss cap, $500 weekly cap. $100 risk per trade default.

---

## IBKR Connection Details

| Parameter | Value |
|---|---|
| Host | `127.0.0.1` |
| Port | `4001` (IB Gateway LIVE), `4002` (paper) |
| Executor client_id | `11` (primary order execution) |
| VX feed client_id | `71` (volatility data — separate to avoid conflicts) |
| Data client_id | `1` (historical bar requests) |
| Contract | `MES` (Micro E-mini S&P 500), exchange `CME`, currency `USD` |
| Front month | Auto-qualified via `ib_executor.get_qualified_contract()` |

---

## Testing Patterns

Tests in `tests/` using `pytest` + `pytest-mock`. Most tests construct config objects directly (no YAML loading).

```python
cfg = RiskGateConfig()
cfg.min_stop_points = 2.0
gate = RiskGate(cfg)
account = {"available_funds": 5000.0, "realized_pnl_today": 0.0}
result = gate.evaluate_entry(action="BUY", entry_price=5000.0, ...)
assert not result.allowed
```

Key test files:
- `test_risk_gate.py` — risk gate logic (9 layers)
- `test_dynamic_support_floor.py` — dynamic support floor (32 tests)
- `test_chop_regime_guard.py` — CHOP regime guard (block-all + optional 5-gate exception framework, 24 tests)
- `test_or_break_counter.py` — OR breakout counting
- `test_trend_cont_guards.py` — trend continuation guards
- `test_sentiment_aggregator.py` — multi-source sentiment
- `test_vx_or_breakout_exempt.py` — VX additive signal-type-aware scaling (12 tests)
- `test_vx_futures_feed.py` — VX volatility feed
- `test_indicators.py` — EMA/RSI/ATR/ADX calculations

Known pre-existing failures (~29): `test_backtest.py`, `test_exhaustion_dampening.py`, `test_jan2026_audit_fixes.py`, `test_vx_futures_feed.py`, `test_mes_one_minute_strategy.py`, `test_rsi_pullback_filter.py`, `test_trend_pullback_enhancer.py`, `test_reconciliation_and_exits.py` — unrelated to recent work.

---

## Things to Watch Out For

- `LiveTradingManager` is 3,200+ lines — changes should be surgical. Prefer adding logic to `shree/execution/components/` submodules.
- The `backtest/engine.py` imports strategy classes directly — if you rename or restructure a strategy, update the backtest imports.
- Config has **duplicate risk parameters** in `trading` and `risk_gate` sections; `Settings.validate()` reconciles them. Don't assume one is authoritative without checking.
- `main.py` is legacy but still importable — don't break its imports even though it's deprecated.
- The `one_minute:` config section contains all `ft_*` (fifteen-minute) parameters — historical naming, don't rename.
- `structural_support_floor: 0` in config means static fallback is disabled; dynamic floor handles it now.
- Opening Range is computed from first 2 bars after ETH open (17:00 CT), not RTH open. Check `_compute_opening_range()` in `es_fifteen_min.py`.
- The hybrid pipeline (RAG) is **advisory only** — it dampens confidence but capped at -0.05. It does NOT have veto power.
- VX futures use a separate IBKR client connection (client_id=71) — if VX feed disconnects, bot continues with no VX adjustment (additive 0.0).
- Sentiment sources: Stocktwits (55%) + Reddit (45%). Twitter is disabled (0%). VIX regime blended in separately.
