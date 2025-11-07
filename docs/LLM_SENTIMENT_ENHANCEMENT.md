# LLM Sentiment & Reasoning Enhancement

## Overview

This document describes the enhancements made to extract and track **LLM sentiment scores**, **reasoning text**, and **AI confidence levels** throughout the trading system.

## Goals Achieved

✅ **Extract sentiment score** from LLM output (-1.0 = bearish, 0.0 = neutral, +1.0 = bullish)  
✅ **Extract LLM reasoning** text for each trade decision  
✅ **Include AI confidence** levels in all outputs  
✅ **Store sentiment** in database for historical analysis  
✅ **Surface sentiment** in real-time dashboard  

---

## Implementation Details

### 1. Data Schema Enhancement

**File**: `mytrader/llm/data_schema.py`

**Changes**:
- Added `sentiment_score: float = 0.0` field to `TradeRecommendation` dataclass
- Updated `to_dict()` method to include sentiment score
- Sentiment range: **-1.0 (very bearish)** to **+1.0 (very bullish)**, **0.0 (neutral)**

```python
@dataclass
class TradeRecommendation:
    trade_decision: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0.0 to 1.0
    sentiment_score: float = 0.0  # -1.0 (bearish) to +1.0 (bullish)
    reasoning: str = ""
    key_factors: list[str] = field(default_factory=list)
    # ... other fields
```

### 2. Database Schema Update

**File**: `mytrader/llm/trade_logger.py`

**Changes**:
- Upgraded schema version from 1 to 2
- Added `sentiment_score REAL DEFAULT 0.0` column to `llm_recommendations` table
- Updated `log_trade_entry()` to store sentiment with each trade
- Updated `get_recent_trades()` to retrieve sentiment from database

**Database Structure**:
```sql
CREATE TABLE llm_recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_outcome_id INTEGER,
    trade_decision TEXT NOT NULL,
    confidence REAL NOT NULL,
    sentiment_score REAL DEFAULT 0.0,  -- NEW COLUMN
    reasoning TEXT,
    key_factors TEXT,
    -- ... other columns
);
```

### 3. LLM Prompt Enhancement

**File**: `mytrader/llm/bedrock_client.py`

**Changes**:
- Updated `_build_prompt()` to explicitly request sentiment score from Claude AI
- Enhanced prompt with clear instructions:
  - "Extract the overall market sentiment from your analysis"
  - "sentiment_score should be -1.0 (very bearish) to +1.0 (very bullish) to 0.0 (neutral)"
  - "This should reflect YOUR interpretation of market conditions"
  
**LLM Response Format**:
```json
{
    "trade_decision": "BUY",
    "confidence": 0.85,
    "sentiment_score": 0.65,  // NEW FIELD
    "reasoning": "Brief explanation...",
    "key_factors": ["factor1", "factor2"],
    "risk_assessment": "Brief risk analysis"
}
```

### 4. Response Parsing

**File**: `mytrader/llm/bedrock_client.py`

**Changes**:
- Updated `get_trade_recommendation()` to extract `sentiment_score` from LLM JSON response
- Added sentiment to log output:
  ```python
  logger.info(
      f"LLM Recommendation: {recommendation.trade_decision} "
      f"(confidence: {recommendation.confidence:.2f}, "
      f"sentiment: {recommendation.sentiment_score:+.2f}) - {recommendation.reasoning}"
  )
  ```

### 5. Signal Metadata Enhancement

**File**: `mytrader/strategies/llm_enhanced_strategy.py`

**Changes**:
- Updated `generate()` to include LLM sentiment in signal metadata:
  ```python
  enhanced_signal.metadata.update({
      "llm_decision": llm_rec.trade_decision,
      "llm_confidence": llm_rec.confidence,
      "llm_sentiment": llm_rec.sentiment_score,  # NEW
      "llm_reasoning": llm_rec.reasoning,
      "strategy": "llm_enhanced"
  })
  ```

### 6. Trade Advisor Updates

**File**: `mytrader/llm/trade_advisor.py`

**Changes**:
- Updated all signal generation paths to include `llm_sentiment` in metadata
- Enhanced logging to show sentiment in all decision points:
  - Confidence threshold checks
  - LLM override mode
  - Consensus mode (agreement)
  - Signal conflicts

**Example Log Output**:
```
LLM override: HOLD -> BUY (confidence: 0.80, sentiment: +0.65)
Signals agree: BUY (confidence boosted to 0.88, sentiment: +0.72)
Signal conflict: Traditional=HOLD, LLM=BUY (sentiment: +0.55). Defaulting to HOLD.
```

### 7. Dashboard API Enhancement

**File**: `dashboard/backend/dashboard_api.py`

**Changes**:
- Updated WebSocket broadcast to include sentiment score:
  ```python
  await manager.broadcast({
      "type": "llm_analysis",
      "action": llm_rec.get('trade_decision'),
      "confidence": llm_rec.get('confidence'),
      "sentiment_score": llm_rec.get('sentiment_score', 0.0),  # NEW
      "reasoning": llm_rec.get('reasoning', '')[:200],
      "timestamp": datetime.now().isoformat()
  })
  ```

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         REAL-TIME DATA FLOW                         │
└─────────────────────────────────────────────────────────────────────┘

IB Gateway (Live Market Data)
    ↓
Feature Engineering (41 Technical Indicators)
    ↓
Traditional Strategy Analysis
    ↓
AWS Bedrock Claude AI ──────────────────┐
    ├─ Analyzes market conditions        │
    ├─ Generates trade recommendation    │
    ├─ Extracts sentiment score         │ NEW EXTRACTION
    ├─ Provides reasoning text          │
    └─ Outputs confidence level         │
    ↓                                    ↓
Trade Advisor (Consensus Logic)         │
    ├─ Compares Traditional vs AI       │
    ├─ Includes sentiment in metadata   │ ← SENTIMENT FLOW
    └─ Generates enhanced signal        │
    ↓                                    ↓
LLM Enhanced Strategy                   │
    ├─ Adds LLM metadata to signal      │
    ├─ Surfaces sentiment, confidence   │ ← METADATA ENRICHMENT
    └─ Returns Signal object            │
    ↓                                    ↓
Signal Metadata:                        │
{                                       │
  "llm_decision": "BUY",               │
  "llm_confidence": 0.80,              │
  "llm_sentiment": +0.65,  ←──────────┘ NEW FIELD
  "llm_reasoning": "The extremely oversold RSI..."
}
    ↓
Trade Logger Database
    ├─ Stores sentiment_score in llm_recommendations table
    ├─ Persists reasoning text
    └─ Links to trade outcomes
    ↓
Dashboard WebSocket API
    ├─ Broadcasts sentiment_score to frontend
    ├─ Shows reasoning in UI
    └─ Displays confidence levels
    ↓
React Dashboard UI
    ├─ Live Trading Tab (real-time signals)
    ├─ SPY AI Insights Tab (historical analysis)
    └─ Visual sentiment indicators
```

---

## Sentiment Score Interpretation

| Range | Label | Meaning | Example Conditions |
|-------|-------|---------|-------------------|
| **+0.8 to +1.0** | Very Bullish | Strong upward bias | RSI > 70, MACD bullish crossover, positive news |
| **+0.4 to +0.8** | Bullish | Moderate upward bias | RSI 60-70, uptrend, positive sentiment |
| **+0.0 to +0.4** | Slightly Bullish | Weak upward bias | RSI 50-60, mixed signals |
| **0.0** | Neutral | No clear direction | RSI 45-55, conflicting indicators |
| **-0.4 to 0.0** | Slightly Bearish | Weak downward bias | RSI 40-50, mixed signals |
| **-0.8 to -0.4** | Bearish | Moderate downward bias | RSI 30-40, downtrend, negative sentiment |
| **-1.0 to -0.8** | Very Bearish | Strong downward bias | RSI < 30, MACD bearish crossover, negative news |

**Important Notes**:
- Sentiment is LLM's **interpretation** of all available data
- Not just a copy of input sentiment_score from features
- Combines technicals, sentiment data, price action, volatility
- Provides context for why LLM made its decision

---

## Example LLM Output

**Input Context**:
- RSI: 15.2 (extremely oversold)
- MACD: Histogram positive but weak
- ADX: 45.8 (strong downtrend)
- Price: $6,756.50
- Input sentiment_score: -0.3 (slightly bearish news)

**LLM Output**:
```json
{
  "trade_decision": "BUY",
  "confidence": 0.80,
  "sentiment_score": -0.45,
  "reasoning": "The extremely oversold RSI (15.2) combined with strong ADX (45.8) suggests a potential reversal is imminent. While sentiment is slightly bearish, the technical setup indicates capitulation selling may be exhausted.",
  "key_factors": [
    "Extremely oversold RSI indicates potential bounce",
    "High ADX confirms strong trend (reversal opportunity)",
    "MACD histogram starting to turn positive"
  ],
  "risk_assessment": "Medium risk - counter-trend trade requires tight stops"
}
```

**Interpretation**:
- **trade_decision**: BUY (reversal play)
- **confidence**: 0.80 (high confidence in technical setup)
- **sentiment_score**: -0.45 (still bearish but less extreme than -0.3 input)
  - LLM sees oversold conditions as bearish sentiment starting to shift
  - Not blindly copying input sentiment
  - Reflects technical analysis context
- **reasoning**: Clear explanation of the trade thesis

---

## Testing the Enhancement

### 1. Start the Live Trading Bot

```bash
# Kill any existing processes
pkill -f "main.py live"

# Start fresh
source .venv/bin/activate
python main.py live
```

### 2. Monitor the Logs

Look for enhanced log output:

```
🤖 LLM Recommendation: BUY (confidence: 0.80, sentiment: +0.65) - The extremely oversold...
✅ Signals agree: BUY (confidence boosted to 0.88, sentiment: +0.72)
⚠️ Signal conflict: Traditional=HOLD, LLM=BUY (sentiment: +0.55). Defaulting to HOLD.
```

### 3. Check Database Storage

```bash
sqlite3 data/llm_trades.db "SELECT trade_decision, confidence, sentiment_score, reasoning FROM llm_recommendations ORDER BY id DESC LIMIT 5;"
```

### 4. Verify Dashboard Display

1. Open dashboard: `http://localhost:5173`
2. Navigate to "Live Trading" tab
3. Look for **LLM Analysis** section showing:
   - Trade decision (BUY/SELL/HOLD)
   - Confidence level (0.0 - 1.0)
   - **Sentiment score** (-1.0 to +1.0) ← NEW
   - Reasoning text

### 5. Check WebSocket Messages

Open browser console and watch for:
```javascript
{
  type: "llm_analysis",
  action: "BUY",
  confidence: 0.80,
  sentiment_score: 0.65,  // NEW FIELD
  reasoning: "The extremely oversold RSI...",
  timestamp: "2025-11-06T10:30:45.123Z"
}
```

---

## Database Migration

**Existing databases** will be automatically migrated:
- Schema version bumped from 1 → 2
- `sentiment_score` column added with DEFAULT 0.0
- Existing records get 0.0 sentiment (neutral)
- New records will have actual LLM sentiment

**No manual intervention required** - migration happens on first run.

---

## API Changes

### WebSocket Broadcast (NEW)

**Type**: `llm_analysis`

**Fields**:
```json
{
  "type": "llm_analysis",
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": 0.0 - 1.0,
  "sentiment_score": -1.0 to +1.0,  // ← NEW
  "reasoning": "text",
  "timestamp": "ISO 8601"
}
```

### Signal Metadata (ENHANCED)

```python
signal.metadata = {
  "llm_decision": "BUY",
  "llm_confidence": 0.80,
  "llm_sentiment": 0.65,  // ← NEW
  "llm_reasoning": "The extremely oversold RSI...",
  "traditional_action": "HOLD",
  "traditional_confidence": 0.60,
  "mode": "consensus"
}
```

---

## Benefits

### 1. Enhanced Decision Transparency
- See **why** LLM made each decision (reasoning text)
- Understand LLM's market **sentiment interpretation** (not just action)
- Track **confidence evolution** over time

### 2. Historical Analysis
- Analyze correlation between sentiment and trade outcomes
- Identify patterns: "When LLM sentiment is > 0.7, win rate is X%"
- Backtest sentiment-based filters

### 3. Risk Management
- High confidence + negative sentiment = possible trap (extra caution)
- Low confidence + extreme sentiment = wait for clarity
- Consensus + aligned sentiment = higher quality signal

### 4. Model Training
- Store sentiment for LLM fine-tuning
- Train on: (market context, sentiment, outcome) tuples
- Improve sentiment calibration over time

### 5. Dashboard Visualization
- Real-time sentiment gauge
- Sentiment vs confidence scatter plots
- Historical sentiment trends

---

## Future Enhancements

### Potential Additions:
1. **Sentiment Breakdown**: Bull factors vs bear factors
2. **Sentiment Confidence**: How confident is LLM in its sentiment?
3. **Sentiment Change**: Track delta from previous analysis
4. **Multi-timeframe Sentiment**: 5min vs 1hr vs daily sentiment
5. **Sentiment Alerts**: Notify when sentiment shifts dramatically
6. **Sentiment Filters**: Only trade when sentiment aligns with technicals

---

## Files Modified

| File | Changes | Lines Changed |
|------|---------|--------------|
| `mytrader/llm/data_schema.py` | Added sentiment_score field | +2 lines |
| `mytrader/llm/trade_logger.py` | Schema v2, sentiment column | +10 lines |
| `mytrader/llm/bedrock_client.py` | Prompt enhancement, parsing | +15 lines |
| `mytrader/strategies/llm_enhanced_strategy.py` | Metadata enhancement | +3 lines |
| `mytrader/llm/trade_advisor.py` | Logging enhancement | +15 lines |
| `dashboard/backend/dashboard_api.py` | WebSocket broadcast update | +5 lines |

**Total**: ~50 lines of code changes across 6 files

---

## Testing Checklist

- [x] Schema migration completes without errors
- [x] LLM prompt includes sentiment extraction request
- [x] LLM response parsing extracts sentiment_score
- [ ] Sentiment stored in database correctly
- [ ] Sentiment appears in signal metadata
- [ ] Dashboard receives sentiment via WebSocket
- [ ] Dashboard UI displays sentiment score
- [ ] Log output shows sentiment in decision messages
- [ ] Historical queries include sentiment data

---

## Support

For questions or issues:
1. Check logs in `logs/live_trading.log`
2. Query database: `sqlite3 data/llm_trades.db`
3. Review this document for expected behavior
4. Check dashboard console for WebSocket messages

---

**Last Updated**: November 6, 2025  
**Schema Version**: 2  
**Status**: ✅ Implementation Complete, Testing In Progress
