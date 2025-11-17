#!/usr/bin/env python3
"""
Simple RAG Example
Demonstrates basic usage of the RAG system
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mytrader.llm.bedrock_client import BedrockClient
from mytrader.llm.rag_engine import RAGEngine


def main():
    """Run simple RAG example."""
    
    print("=" * 80)
    print("RAG Example: Trading Q&A System")
    print("=" * 80)
    
    # Sample trading knowledge
    trading_knowledge = [
        """
        E-mini S&P 500 (ES) Futures Contract Specifications:
        - Symbol: ES
        - Exchange: CME
        - Contract Size: $50 × S&P 500 Index
        - Tick Size: 0.25 index points ($12.50)
        - Trading Hours: Nearly 24/5 (Sunday-Friday)
        - Typical Margin: $13,200 (day trading)
        - Settlement: Cash settled monthly
        """,
        
        """
        RSI (Relative Strength Index) Trading Rules:
        - RSI measures momentum from 0 to 100
        - Below 30: Oversold (potential buy signal)
        - Above 70: Overbought (potential sell signal)
        - Best used in ranging markets
        - Can stay extreme during strong trends
        - Combine with other indicators for confirmation
        """,
        
        """
        Risk Management Best Practices:
        - Risk only 1-2% of capital per trade
        - Always use stop losses
        - Maintain 2:1 reward-to-risk ratio minimum
        - Set daily loss limits ($1,500 for $100K account)
        - Never add to losing positions
        - Keep detailed trade logs
        """
    ]
    
    print("\n1. Initializing RAG Engine...")
    
    # Initialize Bedrock client
    bedrock_client = BedrockClient(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        region_name="us-east-1",
        temperature=0.3
    )
    
    # Initialize RAG engine
    rag_engine = RAGEngine(
        bedrock_client=bedrock_client,
        embedding_model_id="amazon.titan-embed-text-v1",
        region_name="us-east-1",
        vector_store_path="data/rag_example_index",
        cache_enabled=True
    )
    
    print("✓ RAG Engine initialized")
    
    print("\n2. Ingesting Trading Knowledge...")
    rag_engine.ingest_documents(trading_knowledge, clear_existing=True)
    print(f"✓ Ingested {len(trading_knowledge)} documents")
    
    # Example queries
    queries = [
        "What is the tick size for ES futures?",
        "When should I use RSI as a buy signal?",
        "How much should I risk per trade?"
    ]
    
    print("\n3. Running Q&A Examples...")
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'=' * 80}")
        print(f"Question {i}: {query}")
        print('=' * 80)
        
        # Generate answer with RAG
        result = rag_engine.generate_with_rag(
            query=query,
            top_k=2,
            score_threshold=0.4,
            include_scores=True
        )
        
        print(f"\n📚 Retrieved {result['num_documents_retrieved']} relevant documents")
        
        if result.get('retrieval_scores'):
            print(f"   Relevance: {[f'{s:.2f}' for s in result['retrieval_scores']]}")
        
        print(f"\n💡 Answer:")
        print(f"   {result['response'][:300]}...")
        
        print(f"\n⏱️  Generation time: {result['generation_time_seconds']:.2f}s")
    
    # Show statistics
    print(f"\n{'=' * 80}")
    print("RAG Statistics")
    print('=' * 80)
    stats = rag_engine.get_stats()
    print(f"Documents in knowledge base: {stats['num_documents']}")
    print(f"Embedding dimension: {stats['embedding_dimension']}")
    print(f"Cache size: {stats['cache_size']}")
    print(f"Vector store: {stats['vector_store_path']}")
    
    print(f"\n✓ Example completed successfully!")
    print(f"\nThe RAG index has been saved to: {stats['vector_store_path']}")
    print("You can now query it without re-ingesting documents.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure you have:")
        print("1. AWS credentials configured (aws configure)")
        print("2. Required packages installed (pip install boto3 faiss-cpu)")
        print("3. Access to AWS Bedrock models")
        sys.exit(1)
