# MES Strategy Refactor - Acceptance vs Exhaustion Framework
## January 27, 2026

### Problem Statement
The MES trading strategy had a **SELL-side bias** where bullish acceptance signals
(EMA_STACK_UP, MACD_POS, RSI_HIGH) were incorrectly interpreted as SHORT signals,
causing repeated losses by shorting strength.

### Root Cause
The old logic treated high RSI (>60-70) as a contrarian sell signal regardless of
trend context. This caused the system to short strong uptrends at exactly the wrong
time.

### Solution: Acceptance vs Exhaustion Framework

#### Core Definitions

**ACCEPTANCE (Continuation)**
- EMA_STACK_UP = true (Price > EMA9 > EMA21 > EMA50)
- MACD_POS > 0.20
- RSI between 55 and 70
- Price above VWAP or EMA50
- **Meaning**: Market accepting direction, continuation likely
- **Action**: BUY pullbacks, NEVER short

**EXHAUSTION (Reversal)**
- RSI >= 65 AND RSI falling
- OR MACD slope <= 0 (flattening/divergence)
- OR failed high (rejection candle)
- **Meaning**: Market rejecting extension, reversal possible
- **Action**: SHORT only with exhaustion confirmation

### Hard Block Rules (Never Violated)

1. **NEVER SHORT ACCEPTANCE**
   - If EMA_STACK_UP AND MACD_POS > 0.20 → SHORT = BLOCKED

2. **MORNING RTH PROTECTION**
   - If SESSION = RTH AND TIME < 10:15 CST
   - AND EMA_STACK_UP AND MACD_POS > 0.15 → SHORT = BLOCKED

3. **SQUEEZE PROTECTION**
   - If RSI > 75 AND MACD_POS > 0.30 → SHORT = BLOCKED

### BUY CONTINUATION MODULE (Morning RTH Optimized)

#### Session Windows
- **PRE_MARKET**: Before 9:30 CST
- **MORNING_OPEN**: 9:30-10:00 CST (volatile, careful)
- **MORNING_PRIME**: 10:00-11:00 CST (**BEST for continuation**)
- **MIDDAY**: 11:00-14:00 CST (chop, avoid)
- **AFTERNOON**: 14:00-15:00 CST (possible trend)
- **CLOSE**: 15:00-16:00 CST (position squaring)

#### Entry Requirements
1. `is_acceptance = TRUE` (or strong trend direction)
2. RSI in zone [55-70], ideal [58-67]
3. ADX >= 25 (trending)
4. MACD histogram > 0.20
5. Valid pullback to:
   - EMA9 touch (shallow pullback, highest score)
   - EMA21 touch (deeper pullback)
   - VWAP touch (anchor pullback)
   - Shallow consolidation (< 30% retracement)

#### Hard Blocks (Module-Level)
1. RSI > 72 → BLOCKED (overextension chase risk)
2. Price > VWAP + 2.0 ATR → BLOCKED (too extended)
3. TIME < 10:15 CST AND not MORNING_OPEN → careful

#### Stop Placement
- Below swing low (recent 3-5 bars)
- OR below EMA21 - 0.5 ATR
- OR below VWAP - 0.3 ATR
- Minimum: 3.25 points

#### Target Placement
- PDH if within 1.5x risk
- OR measured move (1.5-2.0 R)
- Extended target: 3.0R (for strong continuation)
- Minimum R:R = 2.0

### EVENING CONTINUATION MODULE (Late RTH Momentum)

#### Purpose
Captures ONLY strong, high-momentum continuation during late RTH (14:00-16:00 CST)
when pullbacks are unreliable and chop risk is elevated.

**Design Philosophy:**
- Trade LESS frequently than morning continuation
- Require STRONGER confirmation (ADX >= 32 vs 25, MACD >= 0.30 vs 0.20)
- Use TIGHTER stops (max 2.75 pts) and SMALLER targets (2.0-2.5R)
- Skip most setups by design — DISCIPLINE over frequency

#### Session Windows
- **AFTERNOON**: 14:00-15:00 CST
- **CLOSE**: 15:00-16:00 CST
- **CUTOFF**: No new entries after 15:40 CST

#### Market State Requirements (ALL REQUIRED)
- `is_acceptance == True`
- `is_exhaustion == False`
- `EMA_STACK_UP == True`
- Price ABOVE VWAP and EMA21
- ADX >= 32
- MACD >= 0.30
- trend_score >= +70

#### RSI Rules (STRICT)
- Valid zone: 55-68
- RSI > 68 → **HARD BLOCK** (distribution risk)
- RSI < 55 → **HARD BLOCK** (momentum loss)

#### Entry Patterns (ONLY TWO)

**TYPE A — Break-and-Hold:**
- Break above PDH / range high / last impulse high
- Next candle HOLDS above breakout
- Close in top 30% of candle
- Volume >= session median

**TYPE B — Micro Pullback:**
- Pullback depth <= 20% of last impulse
- Does NOT touch EMA21
- Holds above VWAP
- Strong bullish close

#### Explicitly Disallowed
- EMA21 or VWAP pullbacks (morning module territory)
- Deep retracements (> 20%)
- RSI divergence plays
- Inside-bar compression without volume
- Any SELL or reversal logic

#### Risk Management
- Max stop distance: 2.75 points
- Mandatory partial at 1.5R
- Final target: 2.0R-2.5R (NO extended targets)

#### Confidence Scoring
- Base: 50%
- ADX >= 35: +8%
- MACD >= 0.35: +6%
- Break-and-hold pattern: +10%
- Volume expansion: +6%
- Minimum to trade: 65%
- Maximum cap: 78%

### Files Modified

1. **NEW: `shree/strategies/market_state.py`**
   - `MarketStateDetector` class
   - `MarketStateResult` dataclass with boolean flags
   - Implements all acceptance/exhaustion detection
   - Hard block rules implementation

2. **ENHANCED: `shree/strategies/entry_modules.py`**
   - `SessionWindow` enum for session classification
   - `PullbackAnalysis` dataclass for pullback scoring
   - `EveningPatternAnalysis` dataclass for evening patterns
   - `EntrySignal` dataclass for signal output
   - `BuyContinuationModule` - **Morning RTH optimized**
     - Session window detection
     - Multi-level pullback analysis (EMA9/EMA21/VWAP)
     - Confirmation patterns (engulfing, higher low)
     - Stop/target calculation with R:R validation
   - `EveningContinuationModule` - **Late RTH momentum** (NEW)
     - Strict session filtering (14:00-15:40 CST only)
     - Break-and-hold pattern detection
     - Micro pullback pattern detection
     - Tighter stop/target rules
     - Higher confidence thresholds
   - `SellExhaustionModule` - SELL only on exhaustion
   - `IntegratedEntryManager` - Unified interface with priority ordering

3. **MODIFIED: `shree/strategies/mes_one_minute.py`**
   - Added imports for new modules
   - Added `_entry_manager` initialization
   - Modified `generate()` to use new logic (config flag: `use_acceptance_exhaustion_logic`)
   - Added `_prepare_entry_data()` helper
   - Legacy code preserved with fallback option

4. **MODIFIED: `shree/rag/enhanced_signal_engine.py`**
   - Fixed RSI scoring to respect trend context
   - High RSI in uptrend now boosts buy, not sell
   - Added RSI continuation zone detection (55-70 bullish, 30-45 bearish)

5. **MODIFIED: `shree/rag/hybrid_rag_pipeline.py`**
   - Added `is_bullish_acceptance` detection
   - RSI > 60 in acceptance → boosts BUY, not SELL
   - RSI 55-60 in acceptance → RSI_BULLISH_ZONE boost
   - Mean reversion override respects acceptance (won't flip BUY→SELL)
   - Daily bullish bias now strongly penalizes shorts in acceptance

### IntegratedEntryManager Priority Order

```
1. BuyContinuationModule     → Morning/Prime continuation (HIGHEST)
2. EveningContinuationModule → Late RTH momentum (ONLY if #1 returns NO_SIGNAL)
3. SellExhaustionModule      → Reversal on exhaustion (LOWEST)
```

### New Metadata Fields

Signals now include:
- `is_acceptance` - Boolean: in acceptance phase
- `is_exhaustion` - Boolean: exhaustion signals present
- `allow_buy` - Boolean: BUY trades permitted
- `allow_short` - Boolean: SHORT trades permitted
- `market_phase` - Enum: ACCEPTANCE_UP, EXHAUSTION_UP, CHOP, etc.
- `trend_score` - Float: Directional strength (-100 to +100)
- `exhaustion_score` - Float: Exhaustion signal strength (0-100)
- `continuation_score` - Float: Continuation setup quality (0-100)
- `pattern_type` - String: BREAK_HOLD, MICRO_PULLBACK (evening module)

### Configuration Options

```yaml
# In config.yaml or strategy config
use_acceptance_exhaustion_logic: true  # Enable new logic (default: true)
macd_acceptance_threshold: 0.20        # MACD level for acceptance
rsi_acceptance_min: 55                 # RSI floor for acceptance
rsi_acceptance_max: 70                 # RSI ceiling for acceptance
rsi_exhaustion_threshold: 65           # RSI level for exhaustion

# Evening module config (optional)
evening_config:
  adx_min: 32
  macd_min: 0.30
  trend_score_min: 70
  max_stop_distance: 2.75
```

### Trade Logic Summary

| Market Phase      | BUY Action           | SELL Action           | Evening Module      |
|-------------------|----------------------|-----------------------|---------------------|
| ACCEPTANCE_UP     | ✅ Buy pullbacks     | ❌ BLOCKED            | ✅ Break/Micro only |
| ACCEPTANCE_DOWN   | ❌ BLOCKED           | ✅ Sell rallies       | ❌ Not applicable   |
| EXHAUSTION_UP     | ⚠️ Cautious buys     | ✅ Sell with confirm  | ❌ BLOCKED          |
| EXHAUSTION_DOWN   | ✅ Buy with confirm  | ⚠️ Cautious sells     | ❌ Not applicable   |
| SQUEEZE           | ❌ Avoid             | ❌ Avoid              | ❌ Avoid            |
| CHOP              | ⚠️ Low confidence    | ⚠️ Low confidence     | ❌ BLOCKED          |

### Testing

Run the strategy with:
```bash
cd /Users/svss/Documents/code/Shree
./start_bot.sh
```

Check logs for acceptance logic:
```bash
tail -f logs/bot.log | grep -E "(ACCEPTANCE|RSI_BULLISH|SCORE_BREAKDOWN)"
```

### Rollback

To disable new logic and use legacy behavior:
```yaml
use_acceptance_exhaustion_logic: false
```
