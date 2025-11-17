# RAG (Retrieval-Augmented Generation) Integration Guide

## Overview

This guide explains how to use the Retrieval-Augmented Generation (RAG) system in MyTrader. RAG enhances LLM responses by retrieving relevant information from a knowledge base before generating answers, improving accuracy and factual grounding.

## Architecture

The RAG pipeline consists of three main components:

1. **Document Ingestion & Embedding**: Documents are converted to embeddings using AWS Titan Embeddings model
2. **Vector Storage & Retrieval**: FAISS stores and retrieves embeddings based on semantic similarity
3. **Context Injection & Generation**: Retrieved context is added to prompts before invoking Bedrock models

```
User Query → Titan Embedding → FAISS Retrieval → Augmented Prompt → Bedrock Model → Response
```

## Components

### RAGEngine Class

Main class implementing the RAG pipeline (`mytrader/llm/rag_engine.py`):

- `ingest_documents(docs: List[str])` - Creates and stores embeddings
- `retrieve_context(query: str, top_k: int = 3)` - Retrieves relevant passages
- `generate_with_rag(query: str)` - Retrieves context and generates augmented response

### Configuration

RAG settings in `config.yaml`:

```yaml
rag:
  enabled: true  # Enable/disable RAG
  embedding_model_id: "amazon.titan-embed-text-v1"
  region_name: "us-east-1"
  vector_store_path: "data/rag_index"  # Persistent storage
  embedding_dimension: 1536  # Titan v1 dimension
  top_k_results: 3  # Number of documents to retrieve
  score_threshold: 0.5  # Minimum similarity score (0-1)
  cache_enabled: true  # Enable query caching
  cache_ttl_seconds: 3600  # Cache TTL (1 hour)
  batch_size: 10  # Batch size for embedding
  knowledge_base_path: "data/knowledge_base"
```

## Quick Start

### 1. Install Dependencies

```bash
pip install boto3 faiss-cpu
```

For GPU support (optional):
```bash
pip install faiss-gpu
```

### 2. Configure AWS Credentials

Ensure your AWS credentials are configured:

```bash
aws configure
# OR set environment variables:
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"
```

### 3. Enable RAG in Configuration

Edit `config.yaml`:

```yaml
rag:
  enabled: true
  region_name: "us-east-1"
  # ... other settings
```

### 4. Test the RAG Pipeline

Run the test script:

```bash
python bin/test_rag.py
```

This will:
- Ingest sample trading documents
- Test document retrieval
- Test RAG generation
- Validate caching
- Test index persistence

#### Test Options

```bash
# Run all tests
python bin/test_rag.py

# Run specific test
python bin/test_rag.py --test retrieval

# Skip ingestion (use existing index)
python bin/test_rag.py --skip-ingestion

# Custom vector store path
python bin/test_rag.py --vector-store data/my_index
```

## Using the RAG Engine

### Python API

```python
from mytrader.llm.bedrock_client import BedrockClient
from mytrader.llm.rag_engine import RAGEngine

# Initialize Bedrock client
bedrock_client = BedrockClient(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    region_name="us-east-1"
)

# Initialize RAG engine
rag_engine = RAGEngine(
    bedrock_client=bedrock_client,
    embedding_model_id="amazon.titan-embed-text-v1",
    region_name="us-east-1",
    vector_store_path="data/rag_index",
    cache_enabled=True
)

# Ingest documents
documents = [
    "ES futures have a contract multiplier of $50...",
    "RSI below 30 indicates oversold conditions...",
    # ... more documents
]

rag_engine.ingest_documents(documents, clear_existing=True)

# Query with RAG
result = rag_engine.generate_with_rag(
    query="How should I use RSI for trading ES futures?",
    top_k=3,
    score_threshold=0.5
)

print(f"Response: {result['response']}")
print(f"Retrieved {result['num_documents_retrieved']} documents")
```

### REST API Endpoints

The RAG functionality is exposed via FastAPI endpoints:

#### 1. Ingest Documents

```bash
curl -X POST http://localhost:8000/rag/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      "ES futures contract specifications...",
      "RSI trading strategy..."
    ],
    "clear_existing": false,
    "batch_size": 10
  }'
```

Response:
```json
{
  "success": true,
  "num_documents_ingested": 2,
  "total_documents": 10,
  "message": "Successfully ingested 2 documents"
}
```

#### 2. Ask with RAG

```bash
curl -X POST http://localhost:8000/rag/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the optimal stop loss for ES futures?",
    "top_k": 3,
    "score_threshold": 0.5,
    "include_scores": true
  }'
```

Response:
```json
{
  "query": "What is the optimal stop loss for ES futures?",
  "response": "Based on the context, for ES futures trading...",
  "retrieved_documents": [
    "Risk management principles...",
    "Stop loss guidelines..."
  ],
  "retrieval_scores": [0.87, 0.76],
  "num_documents_retrieved": 2,
  "generation_time_seconds": 1.23,
  "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
  "timestamp": "2025-11-10T12:34:56Z"
}
```

#### 3. Retrieve Documents Only

```bash
curl -X POST http://localhost:8000/rag/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "MACD indicator",
    "top_k": 3,
    "score_threshold": 0.5
  }'
```

#### 4. Get RAG Statistics

```bash
curl http://localhost:8000/rag/stats
```

Response:
```json
{
  "num_documents": 10,
  "embedding_dimension": 1536,
  "cache_size": 5,
  "cache_enabled": true,
  "vector_store_path": "data/rag_index",
  "embedding_model": "amazon.titan-embed-text-v1",
  "llm_model": "anthropic.claude-3-sonnet-20240229-v1:0",
  "status": "active"
}
```

#### 5. Clear Query Cache

```bash
curl -X POST http://localhost:8000/rag/clear-cache
```

#### 6. Health Check

```bash
curl http://localhost:8000/rag/health
```

## Knowledge Base Management

### Creating a Knowledge Base

Create documents in your knowledge base directory:

```bash
mkdir -p data/knowledge_base
```

Add text files with trading knowledge:

```bash
# data/knowledge_base/es_futures.txt
E-mini S&P 500 (ES) Futures
Contract Specifications:
- Multiplier: $50 per point
- Tick Size: 0.25 points ($12.50)
...

# data/knowledge_base/risk_management.txt
Risk Management Principles
1. Never risk more than 1-2% per trade
2. Use stop losses on every trade
...
```

### Loading Knowledge Base

```python
from pathlib import Path

# Load all text files from knowledge base
knowledge_base_path = Path("data/knowledge_base")
documents = []

for file_path in knowledge_base_path.glob("*.txt"):
    with open(file_path, 'r') as f:
        documents.append(f.read())

# Ingest into RAG
rag_engine.ingest_documents(documents, clear_existing=True)
```

## Advanced Features

### Query Caching

RAG automatically caches query results to improve performance:

```python
# First query (cache miss) - takes ~1.5s
result1 = rag_engine.retrieve_context("What is RSI?")

# Second identical query (cache hit) - takes ~0.01s
result2 = rag_engine.retrieve_context("What is RSI?")

# Clear cache
rag_engine.clear_cache()
```

Cache settings:
- `cache_enabled`: Enable/disable caching
- `cache_ttl_seconds`: Cache time-to-live (default: 3600s)

### Custom Prompt Templates

Modify the augmented prompt format:

```python
def custom_prompt(context: str, query: str) -> str:
    return f"""You are a trading expert. Answer based on this context:

TRADING KNOWLEDGE:
{context}

USER QUESTION:
{query}

Provide a detailed, actionable answer:"""

# Override in RAGEngine._build_augmented_prompt()
```

### Batch Document Processing

For large document collections:

```python
# Process 100 documents in batches of 10
rag_engine.ingest_documents(
    documents=large_document_list,
    batch_size=10  # Process 10 at a time
)
```

### Similarity Score Threshold

Control retrieval quality:

```python
# Strict matching (high threshold)
results = rag_engine.retrieve_context(
    query="What is MACD?",
    score_threshold=0.8  # Only very relevant docs
)

# Loose matching (low threshold)
results = rag_engine.retrieve_context(
    query="What is MACD?",
    score_threshold=0.3  # Include more docs
)
```

## Performance Optimization

### 1. Embedding Batch Size

Larger batches = faster ingestion (but more memory):

```yaml
rag:
  batch_size: 20  # Default: 10
```

### 2. Vector Store Persistence

FAISS index is automatically saved to disk:

```python
# Index saved to: data/rag_index.faiss
# Documents saved to: data/rag_index.pkl
```

On restart, the index is automatically loaded.

### 3. GPU Support (Optional)

For large-scale deployments:

```bash
pip install faiss-gpu

# RAG automatically uses GPU if available
```

### 4. OpenSearch Integration (Future)

For production deployments, consider AWS OpenSearch:

```yaml
rag:
  vector_store_type: "opensearch"  # Instead of FAISS
  opensearch_endpoint: "https://..."
  opensearch_index: "trading-knowledge"
```

## Monitoring & Logging

RAG provides detailed logging:

```python
from mytrader.utils.logger import logger

# Logs include:
# - Document ingestion progress
# - Retrieval statistics
# - Generation times
# - Cache hits/misses
# - Error details
```

View logs:
```bash
tail -f logs/trading.log | grep RAG
```

## Troubleshooting

### Issue: Import Error for faiss

```bash
pip install faiss-cpu
# OR for GPU:
pip install faiss-gpu
```

### Issue: AWS Authentication Error

Verify credentials:
```bash
aws sts get-caller-identity
```

### Issue: Empty Retrieval Results

- Check if documents are ingested: `rag_engine.get_stats()`
- Lower `score_threshold` (e.g., 0.3)
- Verify query relevance to documents

### Issue: Slow Ingestion

- Reduce `batch_size` if memory constrained
- Check AWS rate limits
- Consider caching embeddings

### Issue: Vector Store Not Persisting

- Verify `vector_store_path` is writable
- Check disk space
- Ensure proper shutdown (calls `_save_index()`)

## Best Practices

1. **Curate Quality Documents**: High-quality, focused documents improve retrieval
2. **Regular Updates**: Update knowledge base as market conditions change
3. **Monitor Cache**: Clear cache after document updates
4. **Tune Thresholds**: Adjust `score_threshold` based on use case
5. **Batch Operations**: Use batching for large document sets
6. **Error Handling**: Always handle RAG failures gracefully
7. **Cost Management**: Monitor AWS Bedrock usage and costs

## Cost Considerations

### AWS Titan Embeddings Pricing

- ~$0.0001 per 1,000 input tokens
- Average document: ~500 tokens
- 1,000 documents ≈ $0.05

### Bedrock Model Pricing

- Claude 3 Sonnet: ~$0.003/1K input, ~$0.015/1K output
- Query with 3 retrieved docs (~1,500 tokens): ~$0.0045

### Cost Optimization

- Use caching to reduce embedding calls
- Persist vector store to avoid re-embedding
- Use Claude Haiku for lower costs
- Batch document ingestion

## Example Use Cases

### 1. Trading Strategy Q&A

```python
result = rag_engine.generate_with_rag(
    query="Should I trade when RSI is 35 and MACD histogram is positive?"
)
```

### 2. Risk Management Assistant

```python
result = rag_engine.generate_with_rag(
    query="What position size should I use with $100K capital?"
)
```

### 3. Market Context Analysis

```python
result = rag_engine.generate_with_rag(
    query="What are the best trading hours for ES futures?"
)
```

## API Reference

See complete API documentation in:
- `mytrader/llm/rag_engine.py` - Core RAG implementation
- `dashboard/backend/rag_api.py` - REST API endpoints

## Support

For issues or questions:
- Check logs: `logs/trading.log`
- Review test results: `python bin/test_rag.py`
- Verify configuration: `config.yaml`

## Future Enhancements

Planned features:
- [ ] Multi-modal embeddings (text + charts)
- [ ] Incremental document updates
- [ ] AWS OpenSearch integration
- [ ] Document versioning
- [ ] Real-time knowledge base updates
- [ ] Advanced retrieval strategies (hybrid search)
- [ ] Fine-tuned retrieval models
