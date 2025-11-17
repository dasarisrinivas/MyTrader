# RAG Pipeline and Decision Flow Diagrams

## 1. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         MYTRADER ARCHITECTURE                            │
└──────────────────────────────────────────────────────────────────────────┘

┌─────────────┐
│   IBKR API  │  ← Live market data (SPY futures price, volume)
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ Market Data      │  ← Collects OHLCV bars every 5 seconds
│ Pipeline         │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Feature          │  ← RSI, MACD, ATR, Bollinger Bands, ADX
│ Engineering      │
└──────┬───────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                   MULTI-STRATEGY ENGINE                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Trend        │  │ Breakout     │  │ Mean         │      │
│  │ Following    │  │ Detection    │  │ Reversion    │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │               │
│         └──────────────────┴──────────────────┘               │
│                            │                                  │
│                            ▼                                  │
│                 ┌──────────────────────┐                     │
│                 │  RAG Signal          │  ← Validates with   │
│                 │  Validator           │     trading rules   │
│                 └──────────┬───────────┘                     │
└────────────────────────────┼──────────────────────────────────┘
                             │ Traditional Signal
                             │ (BUY/SELL/HOLD + confidence)
                             ▼
┌────────────────────────────────────────────────────────────────┐
│              LLM-ENHANCED STRATEGY (Optional)                  │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │     RAG-ENHANCED TRADE ADVISOR                        │    │
│  │                                                        │    │
│  │  1. Build query from market context                   │    │
│  │  2. Retrieve relevant docs from RAG engine            │    │
│  │     (AWS Titan Embeddings + FAISS)                    │    │
│  │  3. Augment prompt with retrieved knowledge           │    │
│  │  4. Call AWS Bedrock Claude LLM                       │    │
│  │  5. Apply consensus/override logic                    │    │
│  │                                                        │    │
│  │  ┌──────────────┐  ┌──────────────┐                  │    │
│  │  │  Override    │  │  Consensus   │                  │    │
│  │  │  Mode        │  │  Mode        │                  │    │
│  │  │  (LLM wins)  │  │  (Must agree)│                  │    │
│  │  └──────────────┘  └──────────────┘                  │    │
│  └───────────────────────────┬────────────────────────────┘    │
└──────────────────────────────┼───────────────────────────────┘
                               │ Enhanced Signal
                               │ (action, confidence, reasoning)
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                    RISK MANAGEMENT                           │
│  • Kelly Criterion position sizing                          │
│  • ATR-based stop-loss/take-profit                         │
│  • Daily loss limits                                        │
│  • Max position size checks                                │
└──────────┬───────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────┐
│ Trade Executor   │  → Places orders with IBKR
│ (IBKR Orders)    │
└──────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────┐
│                    DASHBOARD (WebSocket)                      │
│  • Real-time signal updates                                  │
│  • LLM reasoning display                                     │
│  • Performance metrics                                       │
│  • Trade history                                             │
└──────────────────────────────────────────────────────────────┘
```

## 2. RAG Engine Detailed Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                       RAG ENGINE                                │
└─────────────────────────────────────────────────────────────────┘

INITIALIZATION:
┌────────────────┐
│ Load Config    │  ← vector_store_path, embedding_model_id
└───────┬────────┘
        │
        ▼
┌────────────────────────┐
│ Initialize AWS Bedrock │  ← boto3.client("bedrock-runtime")
│ Runtime Client         │
└───────┬────────────────┘
        │
        ▼
┌────────────────────────┐
│ Initialize FAISS Index │  ← IndexFlatIP(dimension=1536)
│ (IndexFlatIP)          │     Cosine similarity search
└───────┬────────────────┘
        │
        ▼
┌────────────────────────┐
│ Load Existing Index    │  ← If vector_store_path exists
│ from Disk (if present) │     Load .faiss and .pkl files
└────────────────────────┘


DOCUMENT INGESTION:
┌──────────────────┐
│ Input: Documents │  ← List of trading knowledge texts
└────────┬─────────┘
         │
         ▼
┌──────────────────────────┐
│ For each document:       │
│   1. Call AWS Titan      │  ← invoke_model(amazon.titan-embed-text-v1)
│      Embeddings API      │
│   2. Get 1536-dim vector │
│   3. Normalize vector    │  ← L2 normalization for cosine similarity
│   4. Sleep 0.1s          │  ← Rate limiting
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Add to FAISS Index       │  ← index.add(embeddings)
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Store Document Texts     │  ← self.documents.append(text)
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Save to Disk             │  ← .faiss + .pkl files
└──────────────────────────┘


RETRIEVAL (Query Time):
┌──────────────────┐
│ Input: Query     │  ← "Trading strategy for SPY with RSI oversold..."
└────────┬─────────┘
         │
         ▼
┌──────────────────────────┐
│ Check Cache              │  ← Hash query, check cache TTL
│ (if cache_enabled=true)  │
└────────┬─────────────────┘
         │
         │ Cache MISS
         ▼
┌──────────────────────────┐
│ Generate Query Embedding │  ← AWS Titan Embeddings
│ and Normalize            │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ FAISS Similarity Search  │  ← index.search(query_embedding, top_k)
│ (top_k documents)        │     Returns: (scores, indices)
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Filter by Threshold      │  ← Keep only score >= score_threshold
│ (default: 0.5)           │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Return (document, score) │  ← Top K results sorted by score
│ tuples                   │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Cache Results            │  ← Store with timestamp
└──────────────────────────┘
```

## 3. Signal Generation Decision Tree

```
┌──────────────────────────────────────────────────────────────────┐
│                    SIGNAL GENERATION FLOW                        │
└──────────────────────────────────────────────────────────────────┘

START: New Price Bar Received
│
├─> STEP 1: Feature Engineering
│   ├─ Calculate RSI(14)
│   ├─ Calculate MACD(12,26,9)
│   ├─ Calculate ATR(14)
│   ├─ Calculate ADX(14)
│   └─ Calculate Bollinger Bands(20,2)
│
├─> STEP 2: Market Context Analysis
│   ├─ Trend Bias: price vs MA(50)
│   │  ├─ Bullish: price > MA(50) * 1.01
│   │  ├─ Bearish: price < MA(50) * 0.99
│   │  └─ Neutral: otherwise
│   │
│   └─ Volatility Level: ATR vs Avg ATR
│      ├─ High: ATR > Avg * 1.3
│      ├─ Low: ATR < Avg * 0.7
│      └─ Medium: otherwise
│
├─> STEP 3: Strategy Selection
│   ├─ IF mode = "auto":
│   │  ├─ High volatility + trending → Trend Following
│   │  ├─ Low volatility + neutral → Mean Reversion
│   │  └─ Medium volatility → Breakout
│   │
│   ├─ ELSE: Use configured strategy
│   │
│   └─ IF position open: LOCK strategy (no switching)
│
├─> STEP 4: Generate Traditional Signal
│   │
│   ├─> TREND FOLLOWING:
│   │   ├─ Detect MA crossover
│   │   │  ├─ MA(10) crosses above MA(50) → BUY
│   │   │  └─ MA(10) crosses below MA(50) → SELL
│   │   │
│   │   ├─ Momentum confirmation
│   │   │  ├─ 5-bar drop > 3 points → SELL
│   │   │  └─ 5-bar rise > 3 points → BUY
│   │   │
│   │   └─ Confidence boosters:
│   │      ├─ Price momentum aligned: +0.1
│   │      ├─ Market bias aligned: +0.1
│   │      └─ MA separation > 0.5%: +0.1
│   │
│   ├─> BREAKOUT:
│   │   ├─ Price > 20-bar high → BUY
│   │   ├─ Price < 20-bar low → SELL
│   │   │
│   │   └─ Confidence boosters:
│   │      ├─ Volume surge > 1.2x avg: +0.15
│   │      ├─ Trend aligned: +0.1
│   │      └─ Strong breakout > 0.05%: +0.1
│   │
│   └─> MEAN REVERSION:
│       ├─ Price < BB lower + RSI < 30 → BUY (oversold)
│       ├─ Price > BB upper + RSI > 70 → SELL (overbought)
│       │
│       └─ Confidence adjustments:
│          ├─ Extreme RSI (<25 or >75): +0.1
│          ├─ Far from band (>1%): +0.1
│          └─ Against strong trend: -0.1
│
├─> STEP 5: Apply Minimum Confidence Filter
│   ├─ IF confidence < 0.65:
│   │  └─> OUTPUT: HOLD (insufficient confidence)
│   │
│   └─ ELSE: Continue to RAG validation
│
├─> STEP 6: RAG Signal Validation (if RAG enabled)
│   │
│   ├─ Build validation query:
│   │  "Trading strategy for SPY with [market conditions]"
│   │
│   ├─ Retrieve top-2 trading rules (threshold=0.6)
│   │
│   ├─ Apply rule penalties:
│   │  ├─ High volatility warning → -0.15
│   │  ├─ Counter-trend trade → -0.20
│   │  └─ Poor risk:reward (<1.5) → -0.10
│   │
│   ├─ Adjusted confidence = original - penalties
│   │
│   └─ IF adjusted < 0.5:
│      └─> OUTPUT: HOLD (RAG rejected signal)
│
├─> STEP 7: LLM Enhancement (if LLM enabled)
│   │
│   ├─ Check rate limit:
│   │  ├─ IF too soon since last call:
│   │  │  └─> Use cached recommendation
│   │  │
│   │  └─ ELSE: Make new LLM call
│   │
│   ├─ IF RAG enabled:
│   │  ├─ Build context query from market conditions
│   │  ├─ Retrieve top-3 relevant docs (threshold=0.5)
│   │  ├─ Build augmented prompt with docs
│   │  └─ Call AWS Bedrock Claude
│   │
│   ├─ ELSE:
│   │  └─ Call Claude with standard prompt
│   │
│   ├─ Parse LLM response:
│   │  ├─ Extract: trade_decision, confidence, sentiment
│   │  ├─ Validate: confidence in [0, 1]
│   │  └─ Store: reasoning, key_factors
│   │
│   ├─ IF LLM confidence < threshold (0.7):
│   │  └─> OUTPUT: HOLD (LLM not confident)
│   │
│   ├─> CONSENSUS LOGIC:
│   │   │
│   │   ├─ IF override_mode = True:
│   │   │  └─> OUTPUT: LLM decision (final)
│   │   │
│   │   ├─ ELSE IF traditional = LLM:
│   │   │  ├─ Combined confidence = (trad + LLM) / 2
│   │   │  └─> OUTPUT: Agreed decision
│   │   │
│   │   └─ ELSE (disagreement):
│   │      └─> OUTPUT: HOLD (no consensus)
│   │
│   └─ Cache recommendation for rate limiting
│
├─> STEP 8: Position Rules
│   ├─ IF already long + signal=BUY:
│   │  └─> OUTPUT: HOLD
│   │
│   └─ IF already short + signal=SELL:
│      └─> OUTPUT: HOLD
│
└─> FINAL OUTPUT:
    ├─ Action: BUY | SELL | HOLD
    ├─ Confidence: 0.0 to 1.0
    ├─ Risk Params: stop_loss, take_profit, ATR
    └─ Metadata:
       ├─ Strategy used
       ├─ Market bias & volatility
       ├─ LLM reasoning (if applicable)
       ├─ RAG documents (if applicable)
       └─ Traditional + LLM confidence
```

## 4. Data Flow: Backend → Frontend

```
┌─────────────────────────────────────────────────────────────────┐
│                   BACKEND → FRONTEND DATA FLOW                  │
└─────────────────────────────────────────────────────────────────┘

SIGNAL GENERATION (main.py):
│
├─ Signal generated in trading loop (line 305-365)
│  ├─ Signal contains: action, confidence, metadata
│  │  └─ Metadata includes:
│  │     ├─ risk_params (stop_loss, take_profit, ATR)
│  │     ├─ llm_recommendation (if LLM enabled)
│  │     │  ├─ trade_decision
│  │     │  ├─ confidence
│  │     │  ├─ sentiment_score
│  │     │  └─ reasoning (TRUNCATED to 200 chars!)
│  │     └─ rag_enabled flag
│  │
│  └─ Stored in local variable (NOT persisted)
│
├─> WEBSOCKET BROADCAST (dashboard_api.py line 346-352)
│   │
│   ├─ Type: "signal"
│   ├─ Data:
│   │  ├─ signal: action (BUY/SELL/HOLD)
│   │  ├─ confidence: float
│   │  ├─ market_bias: string
│   │  └─ volatility: string
│   │
│   └─ IF LLM enabled (line 314-321):
│      ├─ Type: "llm_analysis"
│      ├─ Data:
│      │  ├─ action: LLM decision
│      │  ├─ confidence: LLM confidence
│      │  ├─ sentiment_score: float
│      │  └─ reasoning: TRUNCATED to 200 chars ❌
│      │
│      └─ ISSUE: RAG documents NOT included ❌
│
├─> REST API: /api/trading/status (line 789-812)
│   │
│   ├─ Returns:
│   │  ├─ is_running: boolean
│   │  ├─ mode: "integrated"
│   │  └─ message: string
│   │
│   └─ ISSUE: Last signal NOT stored ❌
│      (Frontend expects status.last_signal)
│
├─> REST API: /api/market/status (line 815-965)
│   │
│   ├─ Parses logs/live_trading.log
│   ├─ Extracts:
│   │  ├─ current_price
│   │  ├─ last_signal (from log)
│   │  ├─ signal_confidence (from log)
│   │  ├─ market_bias (from log)
│   │  └─ active_strategy (from log)
│   │
│   └─ ISSUE: Log parsing can fail or be stale ❌
│
└─> REST API: /api/trades (line 968-1096)
    │
    ├─ Parses logs/live_trading.log
    ├─ Returns:
    │  ├─ trades: array of signals
    │  ├─ executions: array of order fills
    │  └─ orders: array of close events
    │
    └─ ISSUE: No entry_reason or exit_reason fields ❌


FRONTEND POLLING (DecisionIntelligence.jsx):
│
├─ useEffect hook: Poll every 3 seconds (line 13)
│
├─> Fetch /api/trading/status
│   ├─ Updates: latestSignal
│   │  ├─ action: status.last_signal ❌ (undefined)
│   │  ├─ confidence: status.signal_confidence ❌ (undefined)
│   │  └─ reason: status.signal_reason ❌ (undefined)
│   │
│   └─ ISSUE: Status endpoint doesn't expose signal data
│
├─> Fetch /api/trades?limit=1
│   ├─ Updates: lastTrade
│   │  ├─ action: trade.action
│   │  ├─ pnl: trade.pnl
│   │  ├─ entry_reason: trade.entry_reason ❌ (undefined)
│   │  └─ exit_reason: trade.exit_reason ❌ (undefined)
│   │
│   └─ ISSUE: Trades endpoint doesn't parse reasons from logs
│
└─> Sentiment from status.sentiment_score
    └─ ISSUE: Sentiment not updated in status endpoint ❌


DISPLAY (DecisionIntelligence.jsx):
│
├─ Current Signal Card (line 90-136)
│  ├─ Action badge (BUY/SELL/HOLD)
│  ├─ Confidence percentage
│  ├─ Confidence bar with color coding
│  └─ AI Reasoning: latestSignal.reason
│     └─ ISSUE: Always shows "Analyzing market conditions..." ❌
│
├─ Market Sentiment Gauge (line 138-178)
│  ├─ Sentiment score: -1 to +1
│  ├─ Label: Bullish/Bearish/Neutral
│  └─ Visual gauge with marker
│     └─ ISSUE: Sentiment not propagated from LLM ❌
│
└─ Last Trade Decision (line 180-223)
   ├─ Action and P&L
   ├─ Entry reason (hardcoded placeholder)
   └─ Exit reason (hardcoded placeholder)
      └─ ISSUE: Reasons not extracted from logs ❌


DATA CONSISTENCY ISSUES:
│
├─ ISSUE 1: Signal stored in local variable
│  └─ Not accessible to REST API endpoints
│
├─ ISSUE 2: Log parsing for status
│  └─ Prone to format changes and failures
│
├─ ISSUE 3: WebSocket vs REST inconsistency
│  └─ WebSocket has signal, REST doesn't
│
├─ ISSUE 4: LLM reasoning truncation
│  └─ Full reasoning lost after 200 characters
│
└─ ISSUE 5: No RAG visibility
   └─ Frontend can't see retrieved documents
```

## 5. Error Propagation Paths

```
┌─────────────────────────────────────────────────────────────────┐
│                   ERROR HANDLING ANALYSIS                        │
└─────────────────────────────────────────────────────────────────┘

SCENARIO 1: AWS Bedrock Embedding Failure
│
├─ Occurs in: RAGEngine._get_embedding() (line 100-137)
│
├─> TRY:
│   └─ invoke_model(amazon.titan-embed-text-v1)
│
├─> CATCH ClientError:
│   ├─ Log error: "AWS Bedrock embedding error"
│   └─ Re-raise exception ✓ (good)
│
├─> CATCH Exception:
│   ├─ Log error: "Error generating embedding"
│   └─ Re-raise exception ✓ (good)
│
├─> CALLER: _batch_embeddings() (line 139-170)
│   │
│   ├─> CATCH Exception:
│   │   ├─ Log error ✓
│   │   └─ Use ZERO VECTOR as fallback ❌ (BAD)
│   │      └─ Silent failure → incorrect similarity scores
│   │
│   └─> RETURN: embeddings array (may contain zeros)
│
└─> IMPACT:
    ├─ Documents with zero embeddings will match every query
    ├─ Retrieval quality degraded silently
    └─ No user notification ❌


SCENARIO 2: LLM JSON Parse Failure
│
├─ Occurs in: BedrockClient._parse_response() (line 158-174)
│
├─> TRY:
│   ├─ Find JSON in response text
│   └─ json.loads(json_text)
│
├─> CATCH JSONDecodeError:
│   ├─ Log error: "Failed to parse JSON response" ✓
│   ├─ Log response text for debugging ✓
│   └─ RETURN: {} (empty dict) ✓
│
├─> CALLER: get_trade_recommendation() (line 337-406)
│   │
│   ├─ IF parsed is empty:
│   │  ├─ Log error: "Empty response from LLM" ✓
│   │  └─ RETURN: None ✓
│   │
│   └─> CALLER: RAGEnhancedTradeAdvisor.enhance_signal() (line 203-332)
│       │
│       ├─ IF llm_rec is None:
│       │  ├─ Log warning ✓
│       │  └─ RETURN: traditional_signal (fallback) ✓
│       │
│       └─> CALLER: LLMEnhancedStrategy.generate() (line 80-142)
│           │
│           └─> RETURN: traditional_signal ✓
│
└─> IMPACT:
    ├─ Graceful degradation to traditional strategy ✓
    ├─ Logged for debugging ✓
    └─ Frontend doesn't know LLM failed ❌


SCENARIO 3: RAG Retrieval Returns Empty
│
├─ Occurs in: RAGEngine.retrieve_context() (line 218-283)
│
├─> IF len(documents) == 0:
│   ├─ Log warning: "No documents in index" ✓
│   └─ RETURN: [] (empty list) ✓
│
├─> IF no results meet threshold:
│   ├─ RETURN: [] (empty list)
│   └─ NO LOGGING ❌ (should log)
│
├─> CALLER: RAGEnhancedTradeAdvisor.enhance_signal() (line 244-273)
│   │
│   ├─ IF not retrieved_docs:
│   │  ├─ Log warning: "No relevant knowledge retrieved" ✓
│   │  └─ SET retrieved_knowledge = "No specific trading knowledge retrieved"
│   │
│   └─> Continue with LLM call (uses standard prompt)
│
└─> IMPACT:
    ├─ LLM still provides recommendation ✓
    ├─ Quality may be lower without RAG context
    └─ Frontend doesn't know RAG failed ❌


SCENARIO 4: Signal Generation Exception
│
├─ Occurs in: main.py trading loop (line 301-366)
│
├─> TRY:
│   └─ multi_strategy.generate(features)
│
├─> CATCH Exception:
│   ├─ Log error: "Error generating signal" ✓
│   ├─ Log traceback ✓
│   └─ Continue to next iteration (skip trade) ✓
│
├─> NOT broadcasted to WebSocket ❌
│
└─> IMPACT:
    ├─ Trade skipped (safe) ✓
    ├─ Error logged ✓
    └─ Frontend shows stale signal ❌


SCENARIO 5: Dashboard API /api/trading/status Called But No Signal Stored
│
├─ Occurs in: dashboard_api.py (line 789-812)
│
├─> IF live_trading_manager exists:
│   └─ RETURN: { is_running: True, mode: "integrated" }
│
├─> ELSE:
│   └─ RETURN: { is_running: False }
│
├─> ISSUE: No signal data in response ❌
│
├─> Frontend expects:
│   ├─ status.last_signal
│   ├─ status.signal_confidence
│   └─ status.signal_reason
│
└─> IMPACT:
    ├─ Frontend shows undefined values ❌
    ├─ Breaks DecisionIntelligence component
    └─ User sees "Analyzing market conditions..." forever


RECOMMENDATIONS:
│
├─ 1. Add structured error broadcasting
│    └─ WebSocket event: { type: "error", component: "RAG", message: ... }
│
├─ 2. Store signal in shared state
│    └─ Use Redis or in-memory store for latest signal
│
├─ 3. Add health check endpoint
│    └─ /api/health: Check RAG, LLM, IBKR connectivity
│
├─ 4. Log RAG failures with more context
│    └─ Include query, threshold, num_docs in log
│
└─ 5. Frontend error display
   └─ Show warning banner when component fails
```

## Summary

This document provides a comprehensive view of the RAG pipeline and decision flow in MyTrader. Key insights:

1. **RAG Integration**: Well-designed but needs robustness (retries, validation)
2. **Decision Flow**: Complex but logical, with proper fallbacks
3. **Data Propagation**: Inconsistent - WebSocket has data, REST doesn't
4. **Error Handling**: Graceful degradation works, but visibility lacking
5. **Frontend**: Expects data that backend doesn't provide

**Next Steps**: Implement fixes from CODEBASE_ANALYSIS.md Priority P0 items.
