#!/usr/bin/env python3
"""Test script for RAG (Retrieval-Augmented Generation) pipeline.

This script tests the RAG implementation including:
- Document ingestion
- Embedding generation with AWS Titan
- FAISS vector storage and retrieval
- Context-augmented generation with Bedrock models
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
from typing import List

from mytrader.config import RAGConfig
from mytrader.llm.bedrock_client import BedrockClient
from mytrader.llm.rag_engine import RAGEngine
from mytrader.utils.logger import logger


# Sample trading knowledge base documents
SAMPLE_DOCUMENTS = [
    """
    ES Futures Trading Basics:
    E-mini S&P 500 (ES) futures are popular derivatives contracts that track the S&P 500 index.
    Key specifications:
    - Contract multiplier: $50 per point
    - Tick size: 0.25 points ($12.50 per tick)
    - Trading hours: Nearly 24 hours, Sunday-Friday
    - Margin requirements: Typically $13,200 for day trading
    - Settlement: Cash settled
    """,
    
    """
    RSI (Relative Strength Index) Trading Strategy:
    RSI is a momentum oscillator that measures the speed and magnitude of price changes.
    - Range: 0 to 100
    - Oversold: RSI < 30 (potential buy signal)
    - Overbought: RSI > 70 (potential sell signal)
    - Divergence: When price makes new high/low but RSI doesn't, indicating potential reversal
    - Period: Standard is 14 periods, but can be adjusted for different timeframes
    Best used in ranging markets, less effective in strong trends.
    """,
    
    """
    MACD (Moving Average Convergence Divergence) Indicator:
    MACD is a trend-following momentum indicator showing relationship between two EMAs.
    Components:
    - MACD Line: 12-period EMA minus 26-period EMA
    - Signal Line: 9-period EMA of MACD line
    - Histogram: MACD minus Signal line
    Trading signals:
    - Bullish: MACD crosses above signal line
    - Bearish: MACD crosses below signal line
    - Divergence: Price diverges from MACD indicating potential trend change
    Works best in trending markets.
    """,
    
    """
    Risk Management in Futures Trading:
    Essential principles for managing risk:
    1. Position Sizing: Never risk more than 1-2% of capital per trade
    2. Stop Loss: Always use stops to limit downside (e.g., 20 ticks for ES)
    3. Risk-Reward Ratio: Target at least 2:1 reward to risk
    4. Portfolio Heat: Limit total portfolio risk to 6-8% maximum
    5. Daily Loss Limit: Set maximum daily loss (e.g., $1,500)
    6. Diversification: Don't concentrate all risk in one position
    Remember: Preservation of capital is the first rule of trading.
    """,
    
    """
    Market Sentiment Analysis:
    Sentiment measures market psychology and can be quantified through:
    - News sentiment: Analyzing financial news with NLP
    - Social media: Twitter, Reddit sentiment scores
    - VIX (Fear Index): High VIX indicates fear, low VIX indicates complacency
    - Put/Call Ratio: High ratio suggests bearish sentiment
    - Commitment of Traders (COT) reports
    Combining technical analysis with sentiment provides edge:
    - Strong technical signal + confirming sentiment = high probability trade
    - Technical signal contradicting sentiment = proceed with caution
    """,
    
    """
    Optimal Trading Times for ES Futures:
    Market activity varies throughout the trading day:
    - Pre-market (7:00-9:30 AM ET): Lower volume, wider spreads
    - Market Open (9:30-10:30 AM ET): Highest volatility and volume
    - Mid-day (10:30 AM-2:00 PM ET): Lower activity, choppy moves
    - Power Hour (2:00-3:00 PM ET): Increased activity as positions adjust
    - Market Close (3:00-4:00 PM ET): High volume, trend resolution
    - After Hours (4:00-5:00 PM ET): Lower liquidity
    Best opportunities typically during open and close periods.
    """,
    
    """
    Kelly Criterion for Position Sizing:
    The Kelly Criterion is a mathematical formula for optimal position sizing:
    Kelly % = (Win Rate × Average Win - Loss Rate × Average Loss) / Average Win
    
    Example:
    - Win rate: 60% (0.60)
    - Average win: $500
    - Loss rate: 40% (0.40)
    - Average loss: $300
    Kelly % = (0.60 × 500 - 0.40 × 300) / 500 = 0.36 or 36%
    
    Important: Use fractional Kelly (e.g., 0.25 × Kelly) for safety.
    Never use full Kelly as it can lead to significant drawdowns.
    """,
    
    """
    Understanding ATR (Average True Range):
    ATR measures market volatility and is crucial for setting stops and targets.
    - Calculation: Average of true ranges over N periods (typically 14)
    - True Range: Max of (High-Low, |High-Close|, |Low-Close|)
    - Higher ATR = Higher volatility
    - Lower ATR = Lower volatility
    
    Applications:
    - Stop Loss: Set stops at 2-3 × ATR from entry
    - Position Sizing: Reduce size in high ATR environments
    - Profit Targets: Set targets based on ATR multiples
    - Breakout Confirmation: High ATR breakouts more reliable
    """,
]


def create_rag_engine(config: RAGConfig) -> RAGEngine:
    """Create and initialize RAG engine.
    
    Args:
        config: RAG configuration
        
    Returns:
        Initialized RAG engine
    """
    logger.info("Initializing RAG engine...")
    
    # Create Bedrock client for generation
    bedrock_client = BedrockClient(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        region_name=config.region_name,
        max_tokens=2048,
        temperature=0.3
    )
    
    # Create RAG engine
    rag_engine = RAGEngine(
        bedrock_client=bedrock_client,
        embedding_model_id=config.embedding_model_id,
        region_name=config.region_name,
        vector_store_path=config.vector_store_path,
        dimension=config.embedding_dimension,
        cache_enabled=config.cache_enabled,
        cache_ttl_seconds=config.cache_ttl_seconds
    )
    
    return rag_engine


def test_ingestion(rag_engine: RAGEngine, documents: List[str]) -> None:
    """Test document ingestion.
    
    Args:
        rag_engine: RAG engine instance
        documents: Documents to ingest
    """
    logger.info("=" * 80)
    logger.info("TEST 1: Document Ingestion")
    logger.info("=" * 80)
    
    print(f"\nIngesting {len(documents)} documents...")
    rag_engine.ingest_documents(
        documents=documents,
        clear_existing=True,
        batch_size=5
    )
    
    stats = rag_engine.get_stats()
    print(f"\n✓ Successfully ingested {stats['num_documents']} documents")
    print(f"  Embedding dimension: {stats['embedding_dimension']}")
    print(f"  Vector store: {stats['vector_store_path']}")


def test_retrieval(rag_engine: RAGEngine) -> None:
    """Test document retrieval.
    
    Args:
        rag_engine: RAG engine instance
    """
    logger.info("=" * 80)
    logger.info("TEST 2: Document Retrieval")
    logger.info("=" * 80)
    
    # Test queries
    queries = [
        "What is RSI and how do I use it for trading?",
        "How should I manage risk in futures trading?",
        "What is the Kelly Criterion?",
        "When is the best time to trade ES futures?",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n--- Query {i} ---")
        print(f"Query: {query}")
        
        # Retrieve context
        results = rag_engine.retrieve_context(
            query=query,
            top_k=3,
            score_threshold=0.3
        )
        
        if results:
            print(f"\n✓ Retrieved {len(results)} relevant documents:")
            for j, (doc, score) in enumerate(results, 1):
                print(f"\n  [{j}] Relevance Score: {score:.4f}")
                print(f"  Preview: {doc[:150].strip()}...")
        else:
            print("  ✗ No relevant documents found")


def test_generation(rag_engine: RAGEngine) -> None:
    """Test RAG generation.
    
    Args:
        rag_engine: RAG engine instance
    """
    logger.info("=" * 80)
    logger.info("TEST 3: RAG Generation")
    logger.info("=" * 80)
    
    # Test queries
    queries = [
        "I'm seeing RSI at 28 on ES futures. What should I do?",
        "How do I set my stop loss for an ES futures trade?",
        "What's the optimal position size if I have $100,000 capital and 55% win rate?",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'=' * 80}")
        print(f"Query {i}: {query}")
        print('=' * 80)
        
        # Generate with RAG
        result = rag_engine.generate_with_rag(
            query=query,
            top_k=3,
            score_threshold=0.3,
            include_scores=True
        )
        
        print(f"\n📊 Generation Statistics:")
        print(f"  Documents retrieved: {result['num_documents_retrieved']}")
        print(f"  Generation time: {result['generation_time_seconds']:.2f}s")
        print(f"  Model: {result['model_id']}")
        
        if result.get('retrieval_scores'):
            print(f"  Relevance scores: {[f'{s:.3f}' for s in result['retrieval_scores']]}")
        
        print(f"\n📝 Retrieved Context:")
        for j, doc in enumerate(result['retrieved_documents'], 1):
            print(f"\n  [Context {j}]")
            print(f"  {doc[:200].strip()}...")
        
        print(f"\n💡 Generated Response:")
        print(f"  {result['response']}")
        print()


def test_cache(rag_engine: RAGEngine) -> None:
    """Test query caching.
    
    Args:
        rag_engine: RAG engine instance
    """
    logger.info("=" * 80)
    logger.info("TEST 4: Query Caching")
    logger.info("=" * 80)
    
    query = "What is MACD and how does it work?"
    
    print(f"\nQuery: {query}")
    
    # First query (cache miss)
    print("\n1st Query (should be cache miss):")
    import time
    start = time.time()
    results1 = rag_engine.retrieve_context(query, top_k=3)
    time1 = time.time() - start
    print(f"  Time: {time1:.3f}s")
    print(f"  Results: {len(results1)} documents")
    
    # Second query (cache hit)
    print("\n2nd Query (should be cache hit):")
    start = time.time()
    results2 = rag_engine.retrieve_context(query, top_k=3)
    time2 = time.time() - start
    print(f"  Time: {time2:.3f}s")
    print(f"  Results: {len(results2)} documents")
    
    # Compare
    speedup = time1 / time2 if time2 > 0 else float('inf')
    print(f"\n✓ Cache speedup: {speedup:.1f}x faster")
    
    # Clear cache
    rag_engine.clear_cache()
    print("\n✓ Cache cleared")


def test_persistence(config: RAGConfig) -> None:
    """Test index persistence.
    
    Args:
        config: RAG configuration
    """
    logger.info("=" * 80)
    logger.info("TEST 5: Index Persistence")
    logger.info("=" * 80)
    
    print("\nTesting index persistence...")
    
    # Create new engine (should load existing index)
    rag_engine = create_rag_engine(config)
    
    stats = rag_engine.get_stats()
    
    if stats['num_documents'] > 0:
        print(f"✓ Successfully loaded existing index with {stats['num_documents']} documents")
        
        # Test retrieval on loaded index
        query = "What is risk management?"
        results = rag_engine.retrieve_context(query, top_k=2)
        
        if results:
            print(f"✓ Retrieval works on loaded index ({len(results)} results)")
        else:
            print("✗ Retrieval failed on loaded index")
    else:
        print("✗ No documents found in loaded index")


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test RAG pipeline")
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip document ingestion (use existing index)"
    )
    parser.add_argument(
        "--test",
        choices=["ingestion", "retrieval", "generation", "cache", "persistence", "all"],
        default="all",
        help="Which test to run"
    )
    parser.add_argument(
        "--vector-store",
        default="data/rag_index",
        help="Path to vector store"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = RAGConfig(
        enabled=True,
        embedding_model_id="amazon.titan-embed-text-v1",
        region_name="us-east-1",
        vector_store_path=args.vector_store,
        embedding_dimension=1536,
        top_k_results=3,
        score_threshold=0.5,
        cache_enabled=True,
        cache_ttl_seconds=3600,
        batch_size=5
    )
    
    try:
        # Create RAG engine
        rag_engine = create_rag_engine(config)
        
        print("\n" + "=" * 80)
        print("RAG PIPELINE TEST SUITE")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Embedding Model: {config.embedding_model_id}")
        print(f"  Region: {config.region_name}")
        print(f"  Vector Store: {config.vector_store_path}")
        print(f"  Cache Enabled: {config.cache_enabled}")
        
        # Run tests
        if args.test in ["ingestion", "all"] and not args.skip_ingestion:
            test_ingestion(rag_engine, SAMPLE_DOCUMENTS)
        
        if args.test in ["retrieval", "all"]:
            test_retrieval(rag_engine)
        
        if args.test in ["generation", "all"]:
            test_generation(rag_engine)
        
        if args.test in ["cache", "all"]:
            test_cache(rag_engine)
        
        if args.test in ["persistence", "all"]:
            test_persistence(config)
        
        # Final stats
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        stats = rag_engine.get_stats()
        print(f"\nRAG Engine Statistics:")
        print(json.dumps(stats, indent=2))
        
        print("\n✓ All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
