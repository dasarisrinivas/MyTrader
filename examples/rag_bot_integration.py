"""
Example: Integrate RAG with Trading Bot

This script shows how to modify main.py to use RAG-enhanced decision making.
"""

# Add these imports at the top of main.py
from mytrader.llm.rag_engine import RAGEngine
from mytrader.llm.rag_trade_advisor import RAGEnhancedTradeAdvisor

# In the run_live() function, replace the LLM enhancement section with:

def setup_rag_enhanced_strategy(settings: Settings, multi_strategy):
    """Set up RAG-enhanced trading strategy.
    
    Args:
        settings: Application settings
        multi_strategy: Base multi-strategy instance
        
    Returns:
        RAG-enhanced strategy if enabled, otherwise returns input strategy
    """
    from mytrader.llm.bedrock_client import BedrockClient
    from mytrader.llm.rag_engine import RAGEngine
    from mytrader.llm.rag_trade_advisor import RAGEnhancedTradeAdvisor
    from mytrader.strategies.llm_enhanced_strategy import LLMEnhancedStrategy
    
    # Check if RAG is enabled
    rag_enabled = settings.rag.enabled if hasattr(settings, 'rag') else False
    llm_enabled = settings.llm.enabled
    
    if not llm_enabled:
        logger.info("⚠️  LLM Enhancement DISABLED - using traditional signals only")
        return multi_strategy
    
    try:
        # Initialize Bedrock client
        bedrock_client = BedrockClient(
            model_id=settings.llm.model_id,
            region_name=settings.llm.region_name,
            max_tokens=settings.llm.max_tokens,
            temperature=settings.llm.temperature
        )
        
        if rag_enabled:
            logger.info("🤖 RAG + LLM Enhancement ENABLED")
            logger.info("   📚 Loading trading knowledge base...")
            
            # Initialize RAG engine
            rag_engine = RAGEngine(
                bedrock_client=bedrock_client,
                embedding_model_id=settings.rag.embedding_model_id,
                region_name=settings.rag.region_name,
                vector_store_path=settings.rag.vector_store_path,
                dimension=settings.rag.embedding_dimension,
                cache_enabled=settings.rag.cache_enabled,
                cache_ttl_seconds=settings.rag.cache_ttl_seconds
            )
            
            # Get stats
            stats = rag_engine.get_stats()
            logger.info(f"   ✅ Knowledge base loaded: {stats['num_documents']} documents")
            
            if stats['num_documents'] == 0:
                logger.warning("   ⚠️  No documents in knowledge base!")
                logger.warning("   Run: python bin/test_rag.py to populate")
            
            # Create RAG-enhanced advisor
            rag_advisor = RAGEnhancedTradeAdvisor(
                bedrock_client=bedrock_client,
                rag_engine=rag_engine,
                min_confidence_threshold=settings.llm.min_confidence_threshold,
                enable_llm=True,
                enable_rag=True,
                llm_override_mode=settings.llm.override_mode,
                rag_top_k=settings.rag.top_k_results,
                rag_score_threshold=settings.rag.score_threshold
            )
            
            # Wrap strategy
            enhanced_strategy = LLMEnhancedStrategy(
                base_strategy=multi_strategy,
                enable_llm=True,
                min_llm_confidence=settings.llm.min_confidence_threshold,
                llm_override_mode=settings.llm.override_mode,
            )
            # Override the trade_advisor with RAG-enhanced one
            enhanced_strategy.trade_advisor = rag_advisor
            
            logger.info("   📋 RAG Config:")
            logger.info(f"      Model: {settings.llm.model_id}")
            logger.info(f"      Min Confidence: {settings.llm.min_confidence_threshold}")
            logger.info(f"      Override Mode: {settings.llm.override_mode}")
            logger.info(f"      RAG Top-K: {settings.rag.top_k_results}")
            logger.info(f"      RAG Threshold: {settings.rag.score_threshold}")
            
            return enhanced_strategy
            
        else:
            logger.info("🤖 LLM Enhancement ENABLED (without RAG)")
            logger.info("   Using direct LLM calls - no knowledge retrieval")
            
            # Standard LLM enhancement without RAG
            enhanced_strategy = LLMEnhancedStrategy(
                base_strategy=multi_strategy,
                enable_llm=True,
                min_llm_confidence=settings.llm.min_confidence_threshold,
                llm_override_mode=settings.llm.override_mode,
            )
            
            logger.info(f"   📋 LLM Config: model={settings.llm.model_id}, "
                       f"min_confidence={settings.llm.min_confidence_threshold}, "
                       f"mode={'override' if settings.llm.override_mode else 'consensus'}")
            
            return enhanced_strategy
            
    except Exception as e:
        logger.error(f"Failed to initialize LLM/RAG enhancement: {e}")
        logger.exception("Traceback:")
        logger.info("Falling back to traditional strategy")
        return multi_strategy


# USAGE: In run_live() function in main.py, replace:
#
#   if llm_enabled:
#       multi_strategy = LLMEnhancedStrategy(...)
#
# With:
#
#   multi_strategy = setup_rag_enhanced_strategy(settings, multi_strategy)


# Example: Test RAG integration before starting bot
def test_rag_integration():
    """Test RAG integration with a sample query."""
    from mytrader.utils.settings_loader import load_settings
    from mytrader.llm.bedrock_client import BedrockClient
    from mytrader.llm.rag_engine import RAGEngine
    
    settings = load_settings()
    
    if not settings.rag.enabled:
        print("❌ RAG is not enabled in config.yaml")
        print("   Set rag.enabled: true to enable")
        return False
    
    try:
        # Initialize
        bedrock = BedrockClient(
            model_id=settings.llm.model_id,
            region_name=settings.llm.region_name
        )
        
        rag_engine = RAGEngine(
            bedrock_client=bedrock,
            vector_store_path=settings.rag.vector_store_path
        )
        
        stats = rag_engine.get_stats()
        print(f"✅ RAG Engine initialized: {stats['num_documents']} documents")
        
        if stats['num_documents'] == 0:
            print("⚠️  No documents in knowledge base")
            print("   Run: python bin/test_rag.py to populate")
            return False
        
        # Test retrieval
        results = rag_engine.retrieve_context(
            query="What is the RSI oversold signal?",
            top_k=2
        )
        
        print(f"✅ Retrieval test: {len(results)} documents retrieved")
        for i, (doc, score) in enumerate(results, 1):
            print(f"   [{i}] Score: {score:.2f}, Preview: {doc[:100]}...")
        
        # Test generation
        result = rag_engine.generate_with_rag(
            query="Should I buy when RSI is 28?",
            top_k=2
        )
        
        print(f"✅ Generation test:")
        print(f"   Response: {result['response'][:200]}...")
        print(f"   Retrieved: {result['num_documents_retrieved']} docs")
        print(f"   Time: {result['generation_time_seconds']:.2f}s")
        
        print("\n✅ RAG integration test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ RAG integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 80)
    print("RAG Integration Test")
    print("=" * 80)
    print()
    
    success = test_rag_integration()
    
    if success:
        print("\n" + "=" * 80)
        print("✅ Ready to start bot with RAG enhancement!")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Ensure RAG is enabled in config.yaml")
        print("2. Populate knowledge base: python bin/test_rag.py")
        print("3. Start bot: python main.py live")
    else:
        print("\n" + "=" * 80)
        print("❌ RAG integration not ready")
        print("=" * 80)
        print("\nTroubleshooting:")
        print("1. Check config.yaml: rag.enabled: true")
        print("2. Ingest documents: python bin/test_rag.py")
        print("3. Verify AWS credentials: aws sts get-caller-identity")
