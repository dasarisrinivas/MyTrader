# RAG Integration with Trading Bot

## Overview

This document explains how RAG (Retrieval-Augmented Generation) enhances the trading bot's BUY/SELL/HOLD decisions by retrieving relevant trading knowledge before making recommendations.

## How It Works

### Current Bot Flow (Without RAG)

```
Market Data → Technical Indicators → Strategy Signal → Risk Check → Execute
                                           ↓
                                      LLM Enhancement (optional)
```

### Enhanced Bot Flow (With RAG)

```
Market Data → Technical Indicators → Strategy Signal
                                           ↓
                            ┌──────────────────────────┐
                            │  RAG-Enhanced Advisor    │
                            └──────────────────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    │                                              │
              Query Market                                   Traditional
              Conditions                                       Signal
                    │                                              │
                    ▼                                              ▼
           ┌──────────────────┐                          ┌────────────────┐
           │ RAG Knowledge Base│                          │ Strategy Logic │
           │ • RSI strategies  │                          │ • RSI          │
           │ • Risk mgmt       │                          │ • MACD         │
           │ • Position sizing │                          │ • Sentiment    │
           │ • Market regimes  │                          └────────────────┘
           └──────────────────┘                                   │
                    │                                              │
                    ▼                                              │
           Retrieve Relevant                                       │
           Trading Knowledge                                       │
                    │                                              │
                    └────────────────┬─────────────────────────────┘
                                     │
                                     ▼
                          Augmented Prompt with
                          Context + Knowledge
                                     │
                                     ▼
                          AWS Bedrock (Claude)
                                     │
                                     ▼
                          Enhanced BUY/SELL/HOLD
                                     │
                                     ▼
                          Risk Check → Execute
```

## Integration Points

### 1. Bot Startup (main.py)

When the bot starts:

```python
# Current: Standard LLM Enhancement
multi_strategy = LLMEnhancedStrategy(
    base_strategy=multi_strategy,
    enable_llm=True,
    min_llm_confidence=0.55
)

# NEW: RAG-Enhanced Strategy
from mytrader.llm.rag_trade_advisor import RAGEnhancedTradeAdvisor
from mytrader.llm.rag_engine import RAGEngine

# Initialize RAG engine with knowledge base
rag_engine = RAGEngine(
    bedrock_client=bedrock_client,
    vector_store_path="data/rag_index"
)

# Create RAG-enhanced advisor
rag_advisor = RAGEnhancedTradeAdvisor(
    bedrock_client=bedrock_client,
    rag_engine=rag_engine,
    enable_rag=True,
    rag_top_k=3
)

# Use in strategy
multi_strategy = LLMEnhancedStrategy(
    base_strategy=multi_strategy,
    trade_advisor=rag_advisor  # Use RAG-enhanced advisor
)
```

### 2. Signal Generation

Every time the bot evaluates market conditions:

**Step 1: Traditional Analysis**
- RSI, MACD, sentiment indicators → BUY/SELL/HOLD signal

**Step 2: RAG Retrieval** (NEW)
- Build query based on current conditions:
  - "Trading strategy for ES with RSI oversold below 30, MACD bullish crossover, positive sentiment"
- Retrieve relevant documents from knowledge base:
  ```
  [Document 1] (relevance: 0.87)
  RSI below 30 indicates oversold conditions. Consider buying when RSI crosses back above 30...
  
  [Document 2] (relevance: 0.82)
  MACD bullish crossover confirmed when histogram turns positive...
  
  [Document 3] (relevance: 0.76)
  Position sizing with Kelly Criterion: calculate based on win rate and risk-reward...
  ```

**Step 3: Augmented LLM Decision**
- LLM receives:
  - Current market data (price, indicators)
  - Traditional signal (BUY/SELL/HOLD)
  - Retrieved trading knowledge (from RAG)
- LLM generates decision based on knowledge + data

**Step 4: Final Decision**
Three modes:

1. **Consensus Mode** (default):
   - If Traditional = BUY and LLM = BUY → Execute BUY
   - If they disagree → HOLD
   
2. **Override Mode**:
   - LLM decision takes precedence
   - Traditional signal ignored if LLM is confident
   
3. **Threshold Mode**:
   - Only execute if LLM confidence >= threshold (e.g., 0.7)

## Configuration

### Enable RAG in config.yaml

```yaml
# RAG Configuration
rag:
  enabled: true
  embedding_model_id: "amazon.titan-embed-text-v1"
  region_name: "us-east-1"
  vector_store_path: "data/rag_index"
  top_k_results: 3  # Retrieve top 3 relevant documents
  score_threshold: 0.5  # Minimum similarity score
  cache_enabled: true
  knowledge_base_path: "data/knowledge_base"

# LLM Configuration
llm:
  enabled: true
  model_id: "anthropic.claude-3-sonnet-20240229-v1:0"
  min_confidence_threshold: 0.70  # Only execute if confidence >= 70%
  override_mode: false  # Use consensus mode
```

### Knowledge Base Setup

Create trading knowledge documents:

```bash
mkdir -p data/knowledge_base

# Add trading documents
cat > data/knowledge_base/rsi_strategy.txt << 'EOF'
RSI Trading Strategy:
- RSI < 30: Oversold, potential buy signal
- RSI > 70: Overbought, potential sell signal
- Wait for confirmation: RSI crossing back
- Divergence: Price makes new low but RSI doesn't = bullish
EOF

cat > data/knowledge_base/risk_management.txt << 'EOF'
Risk Management Principles:
- Never risk more than 1-2% per trade
- Use stop losses on every trade
- Position size based on ATR
- Daily loss limit: $1,500
EOF

# Ingest into RAG
python bin/test_rag.py --ingest
```

## Example Scenario

### Market Conditions
- **Symbol**: ES (E-mini S&P 500)
- **Price**: $4,550.25
- **RSI**: 28 (oversold)
- **MACD**: Histogram turning positive
- **Sentiment**: +0.6 (bullish)
- **ATR**: 15 points

### Without RAG (Traditional + LLM)

1. **Traditional Strategy**: "BUY" (RSI oversold, MACD bullish)
2. **LLM**: Analyzes data → "BUY" (confidence: 0.75)
3. **Decision**: BUY (consensus)

### With RAG (Knowledge-Enhanced)

1. **Traditional Strategy**: "BUY" (RSI oversold, MACD bullish)

2. **RAG Retrieval**:
   Query: "Trading strategy for ES with RSI 28 oversold, MACD bullish, sentiment positive"
   
   Retrieved:
   - "RSI below 30 indicates oversold. Wait for RSI to cross back above 30 for confirmation..."
   - "MACD histogram turning positive is strong buy signal when combined with oversold RSI..."
   - "Position size calculation: With ATR 15, use 2x ATR stop loss = 30 points..."

3. **RAG-Enhanced LLM**:
   ```
   Prompt includes:
   - Market data (RSI 28, MACD bullish, etc.)
   - Retrieved knowledge about RSI oversold strategies
   - Retrieved knowledge about position sizing
   
   LLM response:
   {
     "trade_decision": "BUY",
     "confidence": 0.82,
     "reasoning": "RSI at 28 is oversold with MACD confirmation. Based on 
                   established RSI principles, this is a high-probability buy 
                   setup. Position size should be conservative given ATR.",
     "suggested_position_size": 2,
     "suggested_stop_loss": 4520.25,  # 30 points = 2x ATR
     "suggested_take_profit": 4610.25  # 60 points = 2:1 risk-reward
   }
   ```

4. **Decision**: BUY with 2 contracts, stop at 4520.25, target 4610.25

### Benefit of RAG

**Without RAG**: LLM makes decision based only on raw data  
**With RAG**: LLM makes decision grounded in trading principles and strategies

## Operational Flow

### Bot Startup Sequence

```python
1. Initialize Configuration
   ├─ Load config.yaml
   ├─ Check if RAG enabled
   └─ Check if LLM enabled

2. Initialize RAG Engine (if enabled)
   ├─ Create BedrockClient for LLM
   ├─ Create RAGEngine for embeddings + retrieval
   ├─ Load existing vector store (if available)
   └─ OR ingest knowledge base documents

3. Initialize Strategy
   ├─ Create MultiStrategy (base strategy)
   ├─ Create RAGEnhancedTradeAdvisor
   │  ├─ Pass BedrockClient
   │  ├─ Pass RAGEngine
   │  └─ Set RAG parameters (top_k, threshold)
   └─ Wrap in LLMEnhancedStrategy

4. Start Trading Loop
   └─ Every 5 seconds:
      ├─ Fetch market data
      ├─ Calculate indicators
      ├─ Generate signal (with RAG enhancement)
      └─ Execute trades
```

### Per-Trade Decision Flow

```
Market Data Available
         │
         ▼
Calculate Indicators (RSI, MACD, ATR, etc.)
         │
         ▼
Traditional Strategy Signal
(MultiStrategy.generate_signal)
         │
         ▼
RAG-Enhanced Advisor
         │
         ├─ Build query from market conditions
         │  ("ES with RSI 28, MACD bullish...")
         │
         ├─ Retrieve knowledge (top 3 docs)
         │  - Document 1: RSI strategy
         │  - Document 2: MACD confirmation
         │  - Document 3: Position sizing
         │
         ├─ Build augmented prompt
         │  (Market data + Retrieved knowledge)
         │
         ├─ Invoke Bedrock (Claude)
         │  → Returns: BUY/SELL/HOLD + reasoning
         │
         ├─ Check confidence threshold
         │  (e.g., >= 0.70)
         │
         └─ Apply decision mode
            ├─ Consensus: Both agree?
            ├─ Override: LLM decides
            └─ Threshold: Confidence check
         │
         ▼
Final Signal: BUY/SELL/HOLD
         │
         ▼
Risk Check (position limits, daily loss)
         │
         ▼
Execute Trade (if approved)
```

## Benefits

### 1. **Factual Grounding**
- LLM decisions backed by established trading principles
- Reduces hallucinations and speculative recommendations
- Consistent application of risk management rules

### 2. **Contextual Intelligence**
- Retrieves only relevant knowledge for current conditions
- Different knowledge for different market regimes
- Adapts reasoning based on volatility, sentiment, etc.

### 3. **Explainable Decisions**
- Can trace decision back to specific trading principles
- Shows which knowledge influenced the decision
- Easier to audit and improve

### 4. **Continuous Improvement**
- Add new trading strategies to knowledge base
- Update principles based on performance
- No need to retrain models

## Monitoring

### Log Output

```
2025-11-10 10:30:15 INFO: Market data: ES @ $4,550.25
2025-11-10 10:30:15 INFO: Indicators: RSI=28, MACD=+0.15, ATR=15
2025-11-10 10:30:15 INFO: Traditional signal: BUY (conf: 0.70)
2025-11-10 10:30:16 INFO: RAG Query: Trading strategy for ES with RSI 28 oversold...
2025-11-10 10:30:17 INFO: Retrieved 3 relevant documents for decision
2025-11-10 10:30:18 INFO: LLM Recommendation: BUY (confidence: 0.82)
2025-11-10 10:30:18 INFO:    Reasoning: RSI oversold with MACD confirmation...
2025-11-10 10:30:18 INFO:    Position: 2 contracts, Stop: 4520.25, Target: 4610.25
2025-11-10 10:30:18 INFO: CONSENSUS: BUY (combined conf: 0.76)
2025-11-10 10:30:19 INFO: Placing BUY order for 2 contracts...
```

### Performance Tracking

Track RAG impact:
- Decisions with RAG vs without RAG
- Win rate comparison
- Average confidence scores
- Knowledge retrieval quality

## Testing

Test RAG integration:

```bash
# 1. Test RAG engine standalone
python bin/test_rag.py

# 2. Test with backtest
python main.py backtest --data data/es_historical.csv

# 3. Test live (paper trading)
python main.py live
```

## Troubleshooting

### Issue: "No knowledge retrieved"
- Check if documents are ingested: `curl http://localhost:8000/rag/stats`
- Lower `score_threshold` in config.yaml (e.g., 0.3)
- Verify knowledge base has relevant content

### Issue: "RAG not being used"
- Check `rag.enabled: true` in config.yaml
- Verify RAG engine initialized: Check logs for "RAG enhancement ENABLED"
- Ensure RAGEnhancedTradeAdvisor is used instead of standard TradeAdvisor

### Issue: "Slow performance"
- Enable caching: `cache_enabled: true`
- Reduce `top_k_results` (e.g., 2 instead of 3)
- Use cached vector store (auto-saved to disk)

## Next Steps

1. **Populate Knowledge Base**: Add your trading strategies and principles
2. **Test in Backtest**: Verify RAG improves decision quality
3. **Monitor Live**: Watch RAG impact on real decisions
4. **Iterate**: Update knowledge base based on performance

---

**Status**: Ready for integration  
**Impact**: Enhanced decision-making with knowledge-grounded AI  
**Benefit**: More consistent, explainable, and reliable trading decisions
