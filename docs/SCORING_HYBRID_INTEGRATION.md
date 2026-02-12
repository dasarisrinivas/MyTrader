# Scoring + Hybrid/RAG Integration

**Date:** February 2, 2026  
**Status:** ✅ Active in Production

## Overview

The Shree bot now uses a **dual-validation architecture** that combines:

1. **Hybrid/RAG Pipeline** - Generates primary trading signals using LLM-enhanced market context
2. **Scoring System** - Validates signal quality using objective technical metrics

This creates a 2-layer decision process:
- **Layer 1 (Hybrid):** Determines DIRECTION (buy/sell/hold) using RAG intelligence
- **Layer 2 (Scoring):** Determines QUALITY (should we act? how much size?)

## Architecture

```
Market Data (1-min bars)
    ↓
Hybrid/RAG Pipeline
    ↓
Primary Signal (BUY/SELL/HOLD + confidence)
    ↓
Scoring Validation ← [Scores 0-105 points]
    ↓
Final Decision:
    - Score ≥60: CONFIRM @ 1.0x (full size)
    - Score 45-59: CONFIRM @ 0.5x (half size)
    - Score <45: DOWNGRADE to HOLD (block trade)
```

## Configuration

### Enable Scoring Validation

In `config.yaml`:

```yaml
one_minute:
  use_scoring_system: true  # Enable scoring validation
  scoring_full_size_threshold: 60.0  # Full size if score >= 60
  scoring_half_size_threshold: 45.0  # Half size if score >= 45
```

### Disable Scoring (Hybrid Only)

```yaml
one_minute:
  use_scoring_system: false  # Scoring validation disabled
```

## How It Works

### 1. Hybrid Signal Generation

The hybrid pipeline processes market data through:
- Multi-timeframe candle aggregation (5m, 15m, 30m)
- Sentiment analysis (multi-source)
- VX futures volatility assessment
- RAG/LLM contextual reasoning

**Output:** Signal with action (BUY/SELL/HOLD) and confidence

### 2. Scoring Validation

If scoring is enabled, the signal passes through quality assessment:

**Scoring Components:**
- **Trend Alignment** (0-40 pts): EMA alignment, price position, ADX strength
- **Momentum Quality** (0-25 pts): MACD strength, RSI positioning
- **Regime Fitness** (0-20 pts): Volatility conditions, spread quality
- **Entry Timing** (0-20 pts): VWAP position, recent price action
- **Penalties** (negative): Overextension, poor timing, conflicting signals

**Total Score Range:** 0-105 points

### 3. Decision Logic

```python
if score >= 60:
    # High quality - CONFIRM with full size
    action = hybrid_signal.action
    position_size = 1.0
    
elif score >= 45:
    # Marginal quality - CONFIRM with half size
    action = hybrid_signal.action
    position_size = 0.5
    confidence *= 0.75  # Reduce confidence
    
else:
    # Poor quality - BLOCK trade
    action = "HOLD"
    position_size = 0.0
    log_downgrade_reason()
```

## Log Output

### When Scoring Confirms Signal

```
🎯 SCORING VALIDATION: SELL → Score=67.5/105
   Components: Trend=32.0 Mom=18.5 Regime=15.0 Entry=14.0 Penalty=-12.0
   Decision: SELL @ 1.0x (Reason: High quality setup)
✅ SCORING CONFIRMED: SELL @ 1.0x (score 67.5)
```

### When Scoring Reduces Size

```
🎯 SCORING VALIDATION: BUY → Score=52.3/105
   Components: Trend=24.0 Mom=14.5 Regime=12.0 Entry=10.8 Penalty=-9.0
   Decision: BUY @ 0.5x (Reason: Marginal quality - half size)
✅ SCORING HALF-SIZE: BUY @ 0.5x (score 52.3, conf 0.85 → 0.64)
```

### When Scoring Blocks Signal

```
🎯 SCORING VALIDATION: SELL → Score=38.2/105
   Components: Trend=16.0 Mom=10.2 Regime=8.0 Entry=12.0 Penalty=-8.0
   Decision: HOLD @ 0.0x (Reason: Score below minimum threshold)
⚠️  SCORING DOWNGRADE: SELL blocked (score 38.2 < 45.0)
```

## Implementation Details

### Code Location

**File:** `shree/execution/components/signal_processor.py`

**Integration Point:** Line ~1117 in `generate_trading_signal()`:

```python
# Generate hybrid signal
hybrid_signal = self.pipeline_integration.process(...)

# Apply scoring validation
hybrid_signal = self._apply_scoring_validation(
    hybrid_signal=hybrid_signal,
    features=features,
    current_price=current_price,
)

return hybrid_signal
```

**Validation Method:** `_apply_scoring_validation()` at line ~2050

### Key Features

1. **Fail-Safe Design:**
   - If scoring unavailable → uses hybrid signal as-is
   - If scoring errors → uses hybrid signal as-is
   - If scoring disabled → uses hybrid signal as-is

2. **Metadata Enrichment:**
   - Original signal metadata preserved
   - Scoring breakdown added to metadata
   - Downgrade reasons logged for analysis

3. **Structured Logging:**
   - Emits `scoring_validation` events
   - Includes full score breakdown
   - Tracks confidence adjustments

## Benefits

### 1. Quality Filtering
- Blocks poor-quality setups that RAG might flag
- Reduces drawdown from low-probability trades
- Improves win rate by rejecting marginal entries

### 2. Position Sizing Intelligence
- Full size for high-conviction setups (score ≥60)
- Half size for moderate setups (score 45-59)
- No position for poor setups (score <45)

### 3. Risk Management
- Independent technical validation layer
- Prevents overtrading in poor conditions
- Aligns position size with signal quality

### 4. Transparency
- Detailed score breakdowns in logs
- Clear decision rationale
- Easy to analyze and tune

## Tuning Guide

### If Too Restrictive (Blocking Good Trades)

**Problem:** Scoring blocks trades that perform well

**Solutions:**
1. Lower thresholds:
   ```yaml
   scoring_full_size_threshold: 55.0  # Was 60
   scoring_half_size_threshold: 40.0  # Was 45
   ```

2. Adjust scoring weights in `scoring_entry.py`:
   - Reduce penalty severity
   - Increase credit for trend/momentum
   - Relax regime requirements

### If Not Selective Enough (Taking Bad Trades)

**Problem:** Scoring confirms trades that lose

**Solutions:**
1. Raise thresholds:
   ```yaml
   scoring_full_size_threshold: 65.0  # Was 60
   scoring_half_size_threshold: 50.0  # Was 45
   ```

2. Tighten scoring criteria:
   - Increase penalty weights
   - Require stronger trend alignment
   - Add stricter entry timing rules

### Analyzing Performance

**Compare win rates by score range:**
```bash
# Extract score and P&L from structured logs
grep "scoring_validation" logs/bot.log | \
  jq -r '[.payload.score_total, .payload.final_action] | @csv'
```

**Find trades blocked by scoring:**
```bash
grep "SCORING DOWNGRADE" logs/bot.log
```

**Track position sizing:**
```bash
grep "position_size" logs/bot.log | \
  jq -r '.payload.position_size' | \
  sort | uniq -c
```

## Testing

### 1. Verify Integration

Start bot and check logs:
```bash
tail -f logs/bot.log | grep -E "(SCORING|🎯)"
```

**Expected output:**
- `🎯 SCORING VALIDATION:` on every hybrid signal
- Score breakdowns with components
- Decision logged (CONFIRM/DOWNGRADE)

### 2. Monitor Score Distribution

After 1 hour of trading:
```bash
grep "score_total" logs/bot.log | \
  jq -r '.payload.score_total' | \
  awk '{sum+=$1; n++} END {print "Avg:", sum/n, "Count:", n}'
```

### 3. Check Downgrade Rate

```bash
downgrade=$(grep -c "SCORING DOWNGRADE" logs/bot.log)
total=$(grep -c "SCORING VALIDATION" logs/bot.log)
echo "Downgrade rate: $(( downgrade * 100 / total ))%"
```

**Healthy range:** 30-60% downgrade rate
- <30%: Scoring may be too permissive
- >60%: Scoring may be too restrictive

## Troubleshooting

### Scoring Not Running

**Symptom:** No `🎯 SCORING VALIDATION:` logs

**Checks:**
1. Verify config: `use_scoring_system: true`
2. Check imports: `SCORING_AVAILABLE` should be True
3. Restart bot to load new code

### All Signals Blocked

**Symptom:** Every signal gets downgraded to HOLD

**Checks:**
1. Thresholds too high? Lower to 50/40
2. Scoring weights too strict? Review `scoring_entry.py`
3. Market conditions poor? Check score components

### Scores Always Similar

**Symptom:** All scores in narrow range (e.g., 45-50)

**Checks:**
1. Increase scoring sensitivity
2. Add more weight to differentiating factors
3. Review market data quality (missing indicators?)

## Rollback Plan

To disable scoring validation without code changes:

```yaml
one_minute:
  use_scoring_system: false
```

Bot will immediately revert to hybrid-only mode.

## Performance Expectations

### Baseline (Hybrid Only)
- Signal frequency: Every 5-10 bars
- Win rate: ~55-60%
- Average trade: Moderate quality

### With Scoring (Dual Validation)
- Signal frequency: ~40% reduction (due to blocking)
- Win rate: **Expected +5-10%** (quality filtering)
- Average trade: Higher quality entries
- Position sizing: Mixed (full/half based on quality)

### Key Metrics to Track

1. **Win Rate by Score Range:**
   - Score ≥60: Target 65%+ win rate
   - Score 45-59: Target 55-60% win rate
   - Score <45: Should be blocked

2. **Profit Factor:**
   - Hybrid only: Baseline
   - Dual validation: **Target +15-25%** improvement

3. **Max Drawdown:**
   - Should reduce vs hybrid-only
   - Half-sizing reduces risk on marginal setups

## Next Steps

1. **Monitor for 24-48 hours:**
   - Collect score distribution data
   - Track win rate by score range
   - Analyze downgrade decisions

2. **Tune thresholds:**
   - Adjust based on observed performance
   - A/B test different threshold values
   - Document optimal ranges

3. **Enhance scoring:**
   - Add market regime detection
   - Incorporate session characteristics
   - Weight by recent performance

## References

- **Scoring System:** `docs/SCORING_TECHNICAL.md`
- **Live Trading Guide:** `docs/SCORING_LIVE_TRADING.md`
- **Backtest Results:** `docs/SCORING_BACKTEST_RESULTS.md`
- **Quick Reference:** `docs/SCORING_QUICKREF.md`

## Version History

- **v1.0** (Feb 1, 2026): Initial scoring system implementation
- **v2.0** (Feb 2, 2026): Hybrid + Scoring integration (this document)

---

**Status:** ✅ Production Ready  
**Confidence:** High - Both systems independently validated  
**Risk:** Low - Fail-safe design with config-based rollback
