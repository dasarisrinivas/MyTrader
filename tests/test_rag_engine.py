"""
Unit tests for RAG Engine
Tests error handling, retry logic, and data validation
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from pathlib import Path
import tempfile

# Mock dependencies before importing
with patch('mytrader.llm.rag_engine.BOTO3_AVAILABLE', True):
    with patch('mytrader.llm.rag_engine.FAISS_AVAILABLE', True):
        from mytrader.llm.rag_engine import (
            RAGEngine, 
            RAGEngineError, 
            EmbeddingError, 
            RetrievalError
        )


@pytest.fixture
def mock_bedrock_client():
    """Create a mock Bedrock client"""
    client = Mock()
    client.model_id = "test-model"
    return client


@pytest.fixture
def mock_bedrock_runtime():
    """Create a mock Bedrock runtime client"""
    with patch('boto3.client') as mock_boto_client:
        mock_runtime = Mock()
        mock_boto_client.return_value = mock_runtime
        yield mock_runtime


@pytest.fixture
def rag_engine(mock_bedrock_client, mock_bedrock_runtime):
    """Create RAG engine instance with mocked dependencies"""
    with patch('mytrader.llm.rag_engine.faiss') as mock_faiss:
        mock_index = Mock()
        mock_faiss.IndexFlatIP.return_value = mock_index
        
        engine = RAGEngine(
            bedrock_client=mock_bedrock_client,
            vector_store_path=None,  # Don't persist during tests
            cache_enabled=True
        )
        engine.index = mock_index
        
        return engine


class TestRAGEngineInitialization:
    """Test RAG engine initialization"""
    
    def test_init_without_dependencies(self):
        """Test initialization fails without required dependencies"""
        with patch('mytrader.llm.rag_engine.BOTO3_AVAILABLE', False):
            with pytest.raises(ImportError, match="boto3 is required"):
                RAGEngine(bedrock_client=Mock())
    
    def test_init_success(self, mock_bedrock_client, mock_bedrock_runtime):
        """Test successful initialization"""
        with patch('mytrader.llm.rag_engine.faiss'):
            engine = RAGEngine(
                bedrock_client=mock_bedrock_client,
                cache_enabled=True,
                cache_ttl_seconds=1800
            )
            
            assert engine.cache_enabled is True
            assert engine.cache_ttl_seconds == 1800
            assert engine.max_retries == 3
            assert engine.documents == []


class TestEmbeddingGeneration:
    """Test embedding generation with retry logic"""
    
    def test_get_embedding_success(self, rag_engine, mock_bedrock_runtime):
        """Test successful embedding generation"""
        # Mock successful API response
        mock_response = {
            'body': Mock(read=lambda: '{"embedding": [0.1, 0.2, 0.3]}')
        }
        mock_bedrock_runtime.invoke_model.return_value = mock_response
        
        embedding = rag_engine._get_embedding("test text")
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (3,)
        # Should be normalized
        assert np.allclose(np.linalg.norm(embedding), 1.0)
    
    def test_get_embedding_with_retry(self, rag_engine, mock_bedrock_runtime):
        """Test embedding generation with retry on failure"""
        # Fail first two times, succeed on third
        from botocore.exceptions import ClientError
        
        mock_bedrock_runtime.invoke_model.side_effect = [
            ClientError({'Error': {'Code': 'ThrottlingException'}}, 'invoke_model'),
            ClientError({'Error': {'Code': 'ThrottlingException'}}, 'invoke_model'),
            {'body': Mock(read=lambda: '{"embedding": [0.1, 0.2, 0.3]}')}
        ]
        
        with patch('time.sleep'):  # Speed up test
            embedding = rag_engine._get_embedding("test text")
            
        assert isinstance(embedding, np.ndarray)
        assert mock_bedrock_runtime.invoke_model.call_count == 3
    
    def test_get_embedding_max_retries_exceeded(self, rag_engine, mock_bedrock_runtime):
        """Test embedding fails after max retries"""
        from botocore.exceptions import ClientError
        
        mock_bedrock_runtime.invoke_model.side_effect = ClientError(
            {'Error': {'Code': 'ThrottlingException'}}, 
            'invoke_model'
        )
        
        with patch('time.sleep'):  # Speed up test
            with pytest.raises(EmbeddingError, match="Failed to generate embedding"):
                rag_engine._get_embedding("test text")
    
    def test_latency_tracking(self, rag_engine, mock_bedrock_runtime):
        """Test that embedding latency is tracked"""
        mock_bedrock_runtime.invoke_model.return_value = {
            'body': Mock(read=lambda: '{"embedding": [0.1, 0.2, 0.3]}')
        }
        
        rag_engine._get_embedding("test text")
        
        assert len(rag_engine.embedding_latencies) > 0
        assert rag_engine.get_avg_latency() >= 0


class TestDocumentIngestion:
    """Test document ingestion with validation"""
    
    def test_ingest_empty_documents(self, rag_engine):
        """Test ingestion with empty document list"""
        result = rag_engine.ingest_documents([])
        
        assert result['success'] is False
        assert result['num_documents'] == 0
    
    def test_ingest_validates_documents(self, rag_engine, mock_bedrock_runtime):
        """Test that invalid documents are filtered out"""
        mock_bedrock_runtime.invoke_model.return_value = {
            'body': Mock(read=lambda: '{"embedding": [0.1, 0.2, 0.3]}')
        }
        
        documents = [
            "Valid document with enough text",
            "",  # Empty - should be filtered
            None,  # None - should be filtered
            "Short",  # Too short - should be filtered
            "Another valid document with sufficient length"
        ]
        
        with patch('time.sleep'):  # Speed up test
            result = rag_engine.ingest_documents(documents)
        
        assert result['success'] is True
        assert result['num_documents'] == 2  # Only 2 valid documents
        assert result['num_errors'] == 3  # 3 invalid documents
    
    def test_ingest_handles_embedding_errors(self, rag_engine, mock_bedrock_runtime):
        """Test ingestion continues even if some embeddings fail"""
        from botocore.exceptions import ClientError
        
        # First document succeeds, second fails completely
        mock_bedrock_runtime.invoke_model.side_effect = [
            {'body': Mock(read=lambda: '{"embedding": [0.1, 0.2, 0.3]}')},
            ClientError({'Error': {'Code': 'ServiceError'}}, 'invoke_model'),
            ClientError({'Error': {'Code': 'ServiceError'}}, 'invoke_model'),
            ClientError({'Error': {'Code': 'ServiceError'}}, 'invoke_model'),
        ]
        
        documents = [
            "Valid document one with enough text",
            "Valid document two with enough text"
        ]
        
        with patch('time.sleep'):
            result = rag_engine.ingest_documents(documents, batch_size=1)
        
        assert result['success'] is True
        assert result['num_documents'] == 2
        assert result['num_errors'] > 0  # Should report embedding errors


class TestDocumentRetrieval:
    """Test document retrieval with error handling"""
    
    def test_retrieve_empty_query(self, rag_engine):
        """Test retrieval with empty query"""
        results = rag_engine.retrieve_context("")
        assert results == []
    
    def test_retrieve_invalid_query_type(self, rag_engine):
        """Test retrieval with invalid query type"""
        results = rag_engine.retrieve_context(None)
        assert results == []
    
    def test_retrieve_no_documents(self, rag_engine, mock_bedrock_runtime):
        """Test retrieval when no documents are indexed"""
        mock_bedrock_runtime.invoke_model.return_value = {
            'body': Mock(read=lambda: '{"embedding": [0.1, 0.2, 0.3]}')
        }
        
        results = rag_engine.retrieve_context("test query")
        assert results == []
    
    def test_retrieve_with_documents(self, rag_engine, mock_bedrock_runtime):
        """Test successful retrieval"""
        # Setup documents
        rag_engine.documents = ["Doc 1", "Doc 2", "Doc 3"]
        
        # Mock embedding for query
        mock_bedrock_runtime.invoke_model.return_value = {
            'body': Mock(read=lambda: '{"embedding": [0.1, 0.2, 0.3]}')
        }
        
        # Mock FAISS search results
        rag_engine.index.search.return_value = (
            np.array([[0.9, 0.7, 0.5]]),  # scores
            np.array([[0, 1, 2]])  # indices
        )
        
        results = rag_engine.retrieve_context("test query", top_k=2)
        
        assert len(results) == 2
        assert results[0][0] == "Doc 1"  # Document text
        assert results[0][1] == 0.9  # Score
    
    def test_retrieve_filters_by_threshold(self, rag_engine, mock_bedrock_runtime):
        """Test that low-scoring results are filtered"""
        rag_engine.documents = ["Doc 1", "Doc 2", "Doc 3"]
        
        mock_bedrock_runtime.invoke_model.return_value = {
            'body': Mock(read=lambda: '{"embedding": [0.1, 0.2, 0.3]}')
        }
        
        # Low scores that should be filtered
        rag_engine.index.search.return_value = (
            np.array([[0.3, 0.2, 0.1]]),
            np.array([[0, 1, 2]])
        )
        
        results = rag_engine.retrieve_context(
            "test query",
            top_k=3,
            score_threshold=0.5
        )
        
        assert len(results) == 0  # All filtered out


class TestCaching:
    """Test query caching functionality"""
    
    def test_cache_hit(self, rag_engine, mock_bedrock_runtime):
        """Test that cached queries return immediately"""
        rag_engine.documents = ["Doc 1"]
        
        # First query - cache miss
        mock_bedrock_runtime.invoke_model.return_value = {
            'body': Mock(read=lambda: '{"embedding": [0.1, 0.2, 0.3]}')
        }
        rag_engine.index.search.return_value = (
            np.array([[0.9]]),
            np.array([[0]])
        )
        
        results1 = rag_engine.retrieve_context("test query")
        call_count_1 = mock_bedrock_runtime.invoke_model.call_count
        
        # Second query - should hit cache
        results2 = rag_engine.retrieve_context("test query")
        call_count_2 = mock_bedrock_runtime.invoke_model.call_count
        
        assert results1 == results2
        assert call_count_1 == call_count_2  # No additional API call
    
    def test_cache_expiration(self, rag_engine, mock_bedrock_runtime):
        """Test that cache entries expire"""
        rag_engine.cache_ttl_seconds = 1  # Short TTL
        rag_engine.documents = ["Doc 1"]
        
        mock_bedrock_runtime.invoke_model.return_value = {
            'body': Mock(read=lambda: '{"embedding": [0.1, 0.2, 0.3]}')
        }
        rag_engine.index.search.return_value = (
            np.array([[0.9]]),
            np.array([[0]])
        )
        
        # First query
        rag_engine.retrieve_context("test query")
        
        # Wait for cache expiration
        import time
        time.sleep(1.1)
        
        # Second query - cache should be expired
        call_count_before = mock_bedrock_runtime.invoke_model.call_count
        rag_engine.retrieve_context("test query")
        call_count_after = mock_bedrock_runtime.invoke_model.call_count
        
        assert call_count_after > call_count_before  # New API call made


class TestHealthStatus:
    """Test health status monitoring"""
    
    def test_healthy_status(self, rag_engine):
        """Test healthy status with documents"""
        rag_engine.documents = ["Doc 1", "Doc 2"]
        rag_engine.error_count = 0
        
        stats = rag_engine.get_stats()
        assert stats['health_status'] == 'healthy'
    
    def test_unhealthy_status_no_documents(self, rag_engine):
        """Test unhealthy status with no documents"""
        rag_engine.documents = []
        
        stats = rag_engine.get_stats()
        assert stats['health_status'] == 'unhealthy'
    
    def test_degraded_status_recent_errors(self, rag_engine):
        """Test degraded status with recent errors"""
        import time
        
        rag_engine.documents = ["Doc 1"]
        rag_engine.error_count = 5
        rag_engine.last_error_time = time.time()
        
        stats = rag_engine.get_stats()
        assert stats['health_status'] == 'degraded'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
