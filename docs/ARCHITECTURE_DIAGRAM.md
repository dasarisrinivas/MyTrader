# Hybrid LLM/RAG Decision Architecture

## Overview

The Hybrid Decision Engine combines **deterministic rules** (D-engine) with **heuristic LLM/RAG analysis** (H-engine) to make trading decisions. This architecture addresses the core issue: calling LLM on every tick is too expensive and slow.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HYBRID DECISION FLOW                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────────┐     ┌─────────────────────────┐  │
│  │  Market Data │────▶│  Feature Engine  │────▶│  D-Engine (Deterministic)│  │
│  │   (1m bars)  │     │  (TA Indicators) │     │  - RSI/MACD/EMA scoring  │  │
│  └──────────────┘     └──────────────────┘     │  - Entry/Exit rules      │  │
│                                                │  - PDH/PDL levels        │  │
│                                                └────────────┬────────────┘  │
│                                                             │               │
│                                              ┌──────────────▼──────────────┐│
│                                              │     CANDIDATE SIGNAL?       ││
│                                              │  (technical_score >= 0.55)  ││
│                                              └──────────────┬──────────────┘│
│                                                             │               │
│                              NO: HOLD (skip H-engine) ◀─────┼─────▶ YES     │
│                                                             │               │
│                                              ┌──────────────▼──────────────┐│
│                                              │  H-Engine (Heuristic)       ││
│                                              │  ┌──────────────────────┐   ││
│                                              │  │  RAG: Similar Trades │   ││
│                                              │  │  (FAISS vector search)│   ││
│                                              │  └──────────┬───────────┘   ││
│                                              │             │               ││
│                                              │  ┌──────────▼───────────┐   ││
│                                              │  │  LLM: Trade Advisor  │   ││
│                                              │  │  (Bedrock Claude)    │   ││
│                                              │  └──────────┬───────────┘   ││
│                                              └──────────────┼──────────────┘│
│                                                             │               │
│                                              ┌──────────────▼──────────────┐│
│                                              │   CONFIDENCE SCORER         ││
│                                              │   final = 0.5*tech +        ││
│                                              │          0.3*model +        ││
│                                              │          0.2*rag            ││
│                                              └──────────────┬──────────────┘│
│                                                             │               │
│                                              ┌──────────────▼──────────────┐│
│                                              │   SAFETY MANAGER            ││
│                                              │   - Cooldown (5min)         ││
│                                              │   - Order throttle (3/15m)  ││
│                                              │   - P&L limits              ││
│                                              └──────────────┬──────────────┘│
│                                                             │               │
│                                              ┌──────────────▼──────────────┐│
│                                              │   FINAL DECISION            ││
│                                              │   ┌────────────────────┐    ││
│                                              │   │ DRY-RUN: Log only  │    ││
│                                              │   │ LIVE: Execute IBKR │    ││
│                                              │   └────────────────────┘    ││
│                                              └─────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. D-Engine (Deterministic)

**File:** `mytrader/hybrid/d_engine.py`

**Purpose:** Fast, rule-based signal generation running on every candle close.

```
Inputs:
  - RSI(14)
  - MACD / MACD histogram
  - EMA(9) / EMA(20)
  - ATR(14)
  - Volume
  - Higher-timeframe levels (PDH/PDL, WH/WL)

Outputs:
  - DEngineSignal:
    - action: BUY / SELL / HOLD
    - is_candidate: bool (whether to call H-engine)
    - technical_score: 0.0 - 1.0
    - entry_price, stop_loss, take_profit
    - trend_aligned: bool
```

**Entry Rules:**
- BUY: RSI < 35, MACD_hist > 0, EMA_9 > EMA_20, price near support
- SELL: RSI > 65, MACD_hist < 0, EMA_9 < EMA_20, price near resistance

### 2. H-Engine (Heuristic / LLM + RAG)

**File:** `mytrader/hybrid/h_engine.py`

**Purpose:** Context-aware confirmation using historical patterns and LLM reasoning.

```
Inputs:
  - DEngineSignal (candidate)
  - RAG context (similar historical trades)
  - Market conditions

Outputs:
  - HEngineAdvisory:
    - recommendation: LONG / SHORT / HOLD
    - model_confidence: 0.0 - 1.0
    - rag_confidence: 0.0 - 1.0
    - explanation: str
    - cached: bool
```

**Rate Limiting:**
- Max 10 LLM calls per hour
- Min 60 seconds between calls
- Cache TTL: 5 minutes

### 3. Confidence Scorer

**File:** `mytrader/hybrid/confidence.py`

**Formula:**
```
final_confidence = (0.5 * technical_score) +
                   (0.3 * model_confidence) +
                   (0.2 * rag_confidence)
```

**Trade Decision:**
- `should_trade = final_confidence >= 0.60 AND consensus_reached`
- Consensus: D-engine and H-engine agree on direction

### 4. Safety Manager

**File:** `mytrader/hybrid/safety.py`

**Guards:**
| Check | Threshold | Action |
|-------|-----------|--------|
| Cooldown | 5 minutes | Block rapid trades |
| Order Throttle | 3 per 15 min | Limit frequency |
| P&L Limit | -2.5% from peak | Emergency stop |

### 5. Decision Logger

**File:** `mytrader/hybrid/decision_logger.py`

**Outputs:**
- `logs/decisions/decision_log.json` - Full audit trail
- `logs/decisions/decisions.csv` - Tabular export

## Data Flow Example

```
12:00:00 - Candle close detected
  └─▶ D-Engine evaluates: RSI=32, MACD=0.3, EMA bullish
       └─▶ technical_score = 0.72 (> 0.55 threshold)
            └─▶ is_candidate = True
                 └─▶ H-Engine called
                      ├─▶ RAG: Found 3 similar bullish setups
                      │    └─▶ Win rate 67%, avg gain +$180
                      └─▶ LLM: "LONG recommended, strong support at PDL"
                           └─▶ model_confidence = 0.78
                                └─▶ Confidence Scorer:
                                     final = 0.5*0.72 + 0.3*0.78 + 0.2*0.67 = 0.72
                                     └─▶ 0.72 >= 0.60 ✓ CONSENSUS ✓
                                          └─▶ Safety Manager: OK
                                               └─▶ EXECUTE BUY
```

## Configuration

**File:** `config.yaml`

```yaml
hybrid:
  enabled: true
  
  d_engine:
    candidate_threshold: 0.55
    rsi_oversold: 30.0
    rsi_overbought: 70.0
    ema_fast: 9
    ema_slow: 20
  
  h_engine:
    max_calls_per_hour: 10
    min_interval_seconds: 60
    cache_ttl_seconds: 300
  
  confidence:
    weights:
      technical: 0.5
      model: 0.3
      rag: 0.2
    threshold: 0.60
    require_consensus: true
  
  safety:
    order_cooldown_minutes: 5
    max_orders_per_15min: 3
    emergency_pnl_drop_pct: 2.5
  
  dry_run: false
```

## Files Changed

| File | Purpose |
|------|---------|
| `mytrader/hybrid/__init__.py` | Module exports |
| `mytrader/hybrid/d_engine.py` | Deterministic engine |
| `mytrader/hybrid/h_engine.py` | Heuristic LLM/RAG engine |
| `mytrader/hybrid/confidence.py` | Confidence scoring |
| `mytrader/hybrid/safety.py` | Safety guards |
| `mytrader/hybrid/decision_logger.py` | Audit logging |
| `mytrader/hybrid/hybrid_decision.py` | Main orchestrator |
| `config.yaml` | Configuration section |
| `simulate_replay.py` | Historical replay testing |
| `tests/test_hybrid_decision.py` | Unit and integration tests |

## Integration Points

### IBKR Execution
```python
from mytrader.hybrid import HybridDecisionEngine

engine = HybridDecisionEngine.from_config("config.yaml")

# In trading loop:
decision = engine.evaluate(features, current_price, candle_time)

if decision.should_execute and not engine.is_dry_run():
    ib_client.place_order(
        action=decision.action,
        quantity=1,
        entry_price=decision.entry_price,
        stop_loss=decision.stop_loss,
        take_profit=decision.take_profit,
    )
```

### Existing RAG Integration
The H-engine uses the existing RAG infrastructure:
- `mytrader/llm/rag_storage.py` - SQLite storage
- `mytrader/llm/rag_engine.py` - FAISS vector search

### Existing LLM Integration
The H-engine uses the existing LLM clients:
- `mytrader/llm/bedrock_hybrid_client.py` - AWS Bedrock
- `mytrader/llm/trade_advisor.py` - Trade analysis prompts

## Testing

### Unit Tests
```bash
pytest tests/test_hybrid_decision.py -v
```

### Simulation Replay
```bash
# Replay Dec 8, 2025 in dry-run mode
python simulate_replay.py --date 2025-12-08 --dry-run

# Output: reports/simulation/replay_2025-12-08_*.json
```

### Live Dry-Run
```bash
# Start with dry-run flag
python run_bot.py --dry-run
```

### Live Trading
```bash
# Requires explicit approval
python run_bot.py --approve-live
```

## Why This Architecture?

### Problem
The original implementation called LLM on every tick (60 calls/minute), which was:
1. Too slow (500ms+ latency per call)
2. Too expensive ($0.003+ per call × 60/min = $180/day)
3. Rate-limited (Bedrock throttling)

### Solution
Event-triggered H-engine only on candidate signals:
- Average 2-5 candidates per hour
- ~10 LLM calls/hour max
- Cost: ~$0.30/day
- Latency: Amortized over fewer calls

## Monitoring

### Decision Metrics
```python
stats = engine.get_stats()
print(f"D-engine evaluations: {stats['d_engine_calls']}")
print(f"H-engine calls: {stats['h_engine_calls']}")
print(f"Trades executed: {stats['trades_executed']}")
```

### Safety Status
```python
if engine.safety_manager.is_emergency_stopped:
    print("⚠️ EMERGENCY STOP ACTIVE")
```

---

**Branch:** `feature/hybrid-llm-rag`  
**Created:** December 9, 2025  
**Status:** Ready for dry-run testing
