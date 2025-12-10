# S3 Storage Migration for RAG System

This document describes the migration of all RAG-related data from local filesystem to AWS S3.

## Overview

All RAG data is now stored in AWS S3:
- **Bucket:** `rag-bot-storage-897729113303`
- **Prefix:** `spy-futures-bot/`

## S3 Key Structure

```
spy-futures-bot/
├── trade-logs/
│   └── {YYYY-MM-DD}/
│       └── {HHMMSS}_{trade_id}.json
├── daily-summaries/
│   └── {YYYY-MM-DD}/
│       └── market_summary.json
├── weekly-summaries/
│   └── {YYYY-Www}/
│       └── weekly_summary.json
├── mistake-notes/
│   └── {YYYY-MM-DD}/
│       └── {HHMMSS}_{trade_id}.md
├── llm-reasoning/
│   └── {YYYY-MM-DD}/
│       └── {HHMMSS}.json
├── docs-static/
│   ├── market_structure/
│   ├── contract_specs/
│   ├── glossary/
│   ├── indicator_definitions/
│   └── strategy_rules/
├── docs-dynamic/
│   ├── daily/
│   ├── weekly/
│   └── mistakes/
└── vectors/
    ├── index.faiss
    └── metadata.pkl
```

## Updated Modules

### 1. `s3_storage.py` (NEW)
Core S3 storage module providing:
- `S3Storage` class - Full S3 backend for RAG data
- `save_to_s3(key, data)` - Upload data to S3
- `read_from_s3(key)` - Download data from S3
- `get_s3_storage()` - Get singleton instance

### 2. `rag_storage_manager.py` (UPDATED)
- Removed all local filesystem operations (`pathlib.Path`, `mkdir`, `open()`)
- Now uses `S3Storage` backend
- Updated constructor: `RAGStorageManager(bucket_name, prefix)`
- All methods now return S3 keys instead of file paths

### 3. `embedding_builder.py` (UPDATED)
- FAISS index stored in S3: `vectors/index.faiss`
- Metadata stored in S3: `vectors/metadata.pkl`
- Updated constructor: `EmbeddingBuilder(bucket_name, prefix, ...)`
- Uses `io.BytesIO` for FAISS serialization

### 4. Other modules (UNCHANGED)
- `trade_logger.py` - Uses `get_rag_storage()` (now S3-backed)
- `mistake_analyzer.py` - Uses `get_rag_storage()` (now S3-backed)
- `rag_daily_updater.py` - Uses `get_rag_storage()` (now S3-backed)

## Configuration

### AWS Credentials
Ensure AWS credentials are configured:
```bash
# Option 1: Environment variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1

# Option 2: AWS credentials file (~/.aws/credentials)
[default]
aws_access_key_id = your_access_key
aws_secret_access_key = your_secret_key
region = us-east-1
```

### Bucket Setup
Create the S3 bucket if it doesn't exist:
```bash
aws s3 mb s3://rag-bot-storage-897729113303 --region us-east-1
```

## Usage Examples

### Direct S3 Storage
```python
from mytrader.rag import get_s3_storage, save_to_s3, read_from_s3

# Get storage instance
storage = get_s3_storage()

# Save data
storage.save_to_s3("test/data.json", {"key": "value"})

# Read data
data = storage.read_from_s3("test/data.json")
```

### RAG Storage Manager
```python
from mytrader.rag import get_rag_storage, TradeRecord

# Get storage instance (now S3-backed)
storage = get_rag_storage()

# Save a trade
trade = TradeRecord(
    trade_id="abc123",
    timestamp="2025-01-15T10:30:00",
    action="BUY",
    entry_price=6000.0,
)
s3_key = storage.save_trade(trade)
# Returns: spy-futures-bot/trade-logs/2025-01-15/103000_abc123.json

# Load trades
trades = storage.load_trades(limit=50)
```

### Embedding Builder
```python
from mytrader.rag import create_embedding_builder

# Create builder with S3 backend
builder = create_embedding_builder()

# Build index (saves to S3)
documents = storage.load_all_documents()
builder.build_index(documents)
# Saves: spy-futures-bot/vectors/index.faiss
# Saves: spy-futures-bot/vectors/metadata.pkl
```

## Migration Notes

### Breaking Changes
1. `RAGStorageManager` constructor signature changed:
   - Old: `RAGStorageManager(base_path="rag_data")`
   - New: `RAGStorageManager(bucket_name="rag-bot-storage", prefix="spy-futures-bot/")`

2. `EmbeddingBuilder` constructor signature changed:
   - Old: `EmbeddingBuilder(vectors_path="rag_data/vectors", ...)`
   - New: `EmbeddingBuilder(bucket_name="rag-bot-storage", prefix="spy-futures-bot/", ...)`

3. Factory functions updated:
   - `create_embedding_builder(bucket_name, prefix, use_bedrock)`

### Removed
- All local folder creation (`mkdir`)
- All local file reads/writes (`open()`, `Path.read_text()`, etc.)
- `pathlib.Path` imports from storage modules
- `shutil` imports from storage modules

## Benefits

1. **Durability:** S3 provides 99.999999999% durability
2. **Scalability:** No local storage limits
3. **Accessibility:** Access data from any compute instance
4. **Cost-effective:** Pay only for storage used
5. **Backup:** Easy versioning and cross-region replication
