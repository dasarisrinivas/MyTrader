#!/usr/bin/env python3
"""Quick verification that RAG is ready for bot integration."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mytrader.utils.settings_loader import load_settings
from mytrader.utils.logger import configure_logging

configure_logging(level="INFO")

def verify_rag_setup():
    """Verify RAG is properly configured and ready."""
    print("=" * 80)
    print("RAG Integration Verification")
    print("=" * 80)
    print()
    
    # Load settings
    print("1. Checking configuration...")
    try:
        settings = load_settings("config.yaml")
        
        # Check RAG enabled
        if hasattr(settings, 'rag') and settings.rag.enabled:
            print("   ✅ RAG is ENABLED in config.yaml")
            print(f"      - Model: {settings.rag.embedding_model_id}")
            print(f"      - Region: {settings.rag.region_name}")
            print(f"      - Vector Store: {settings.rag.vector_store_path}")
            print(f"      - Top-K: {settings.rag.top_k_results}")
        else:
            print("   ❌ RAG is NOT enabled in config.yaml")
            return False
        
        # Check LLM enabled
        if hasattr(settings, 'llm') and settings.llm.enabled:
            print("   ✅ LLM is ENABLED in config.yaml")
            print(f"      - Model: {settings.llm.model_id}")
            print(f"      - Min Confidence: {settings.llm.min_confidence_threshold}")
            print(f"      - Override Mode: {settings.llm.override_mode}")
        else:
            print("   ⚠️  LLM is NOT enabled - RAG will not be used")
        
    except Exception as e:
        print(f"   ❌ Configuration error: {e}")
        return False
    
    # Check vector store exists
    print()
    print("2. Checking vector store...")
    vector_store_path = Path(settings.rag.vector_store_path)
    faiss_file = vector_store_path.with_suffix(".faiss")
    pkl_file = vector_store_path.with_suffix(".pkl")
    
    if faiss_file.exists() and pkl_file.exists():
        print(f"   ✅ Vector store files exist")
        print(f"      - FAISS index: {faiss_file} ({faiss_file.stat().st_size / 1024:.1f} KB)")
        print(f"      - Documents: {pkl_file} ({pkl_file.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"   ❌ Vector store files NOT found")
        print(f"      Run: python bin/test_rag.py")
        return False
    
    # Test RAG engine initialization
    print()
    print("3. Testing RAG engine initialization...")
    try:
        from mytrader.llm.bedrock_client import BedrockClient
        from mytrader.llm.rag_engine import RAGEngine
        
        bedrock = BedrockClient(
            model_id=settings.llm.model_id,
            region_name=settings.llm.region_name
        )
        
        rag_engine = RAGEngine(
            bedrock_client=bedrock,
            embedding_model_id=settings.rag.embedding_model_id,
            region_name=settings.rag.region_name,
            vector_store_path=settings.rag.vector_store_path,
            dimension=settings.rag.embedding_dimension,
            cache_enabled=settings.rag.cache_enabled
        )
        
        stats = rag_engine.get_stats()
        print(f"   ✅ RAG engine initialized successfully")
        print(f"      - Documents: {stats['num_documents']}")
        print(f"      - Dimension: {stats['embedding_dimension']}")
        print(f"      - Cache: {'enabled' if stats['cache_enabled'] else 'disabled'}")
        
        if stats['num_documents'] == 0:
            print(f"   ⚠️  No documents loaded - run: python bin/test_rag.py")
            return False
        
    except Exception as e:
        print(f"   ❌ RAG engine initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test retrieval
    print()
    print("4. Testing document retrieval...")
    try:
        results = rag_engine.retrieve_context(
            query="What is RSI indicator?",
            top_k=2
        )
        
        if results:
            print(f"   ✅ Retrieval working: {len(results)} documents retrieved")
            for i, (doc, score) in enumerate(results, 1):
                print(f"      [{i}] Score: {score:.3f}, Preview: {doc[:80]}...")
        else:
            print(f"   ⚠️  No documents retrieved (try lowering score_threshold)")
        
    except Exception as e:
        print(f"   ❌ Retrieval test failed: {e}")
        return False
    
    # Check AWS credentials
    print()
    print("5. Checking AWS credentials...")
    try:
        import boto3
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"   ✅ AWS credentials valid")
        print(f"      - Account: {identity['Account']}")
        print(f"      - User: {identity['Arn'].split('/')[-1]}")
    except Exception as e:
        print(f"   ⚠️  AWS credentials not configured or invalid: {e}")
        print(f"      Bot will fail when trying to call Bedrock")
        print(f"      Run: aws configure")
        return False
    
    print()
    print("=" * 80)
    print("✅ RAG Integration Verification PASSED")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Ensure IB Gateway is running (port 4002)")
    print("  2. Start the bot: python main.py live")
    print("  3. Watch for RAG-enhanced decisions in logs")
    print()
    print("Expected log output:")
    print('  🤖 RAG + LLM Enhancement ENABLED - Knowledge-grounded AI decisions')
    print('     ✅ Knowledge base loaded: 8 documents')
    print('     RAG Query: Trading strategy for ES with RSI 28...')
    print('     🤖 LLM Recommendation: BUY (confidence: 0.82)')
    print()
    
    return True


if __name__ == "__main__":
    success = verify_rag_setup()
    sys.exit(0 if success else 1)
