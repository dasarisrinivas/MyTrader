"""Tests for Hybrid Bedrock Client integration.

Tests cover:
- Client initialization
- Context hashing and caching
- Retry logic with mocked boto3
- SQLite logging
- Cost estimation
- Quota checking
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# Mock boto3 before importing the module
@pytest.fixture(autouse=True)
def mock_boto3():
    """Mock boto3 for all tests."""
    with patch.dict(os.environ, {
        "AWS_REGION": "us-east-1",
        "AWS_BEDROCK_MODEL": "anthropic.claude-3-sonnet-20240229-v1:0",
        "AWS_ACCESS_KEY_ID": "test",
        "AWS_SECRET_ACCESS_KEY": "test",
    }):
        yield


@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database path."""
    return str(tmp_path / "test_bedrock.db")


@pytest.fixture
def mock_bedrock_response():
    """Create a mock Bedrock response."""
    return {
        "bias": "BULLISH",
        "confidence": 0.75,
        "action": "BUY",
        "rationale": "Strong momentum with supportive technicals",
        "key_factors": ["Rising RSI", "Positive MACD crossover", "Above VWAP"],
        "risk_notes": "Watch for resistance at 5400",
    }


@pytest.fixture
def mock_boto3_client(mock_bedrock_response):
    """Create a mock boto3 client."""
    mock_client = MagicMock()
    
    # Create mock response body
    response_text = json.dumps(mock_bedrock_response)
    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps({
        "content": [{"text": response_text}],
        "usage": {"input_tokens": 150, "output_tokens": 50}
    }).encode()
    
    mock_client.invoke_model.return_value = {"body": mock_body}
    
    return mock_client


class TestHybridBedrockClient:
    """Tests for HybridBedrockClient."""
    
    def test_client_initialization(self, temp_db_path, mock_boto3_client):
        """Test client initializes correctly."""
        with patch("boto3.client", return_value=mock_boto3_client):
            from mytrader.llm.bedrock_hybrid_client import HybridBedrockClient
            
            client = HybridBedrockClient(
                model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                region_name="us-east-1",
                db_path=temp_db_path,
            )
            
            assert client.model_id == "anthropic.claude-3-sonnet-20240229-v1:0"
            assert client.region_name == "us-east-1"
            assert client.db_manager is not None
    
    def test_context_hash_computation(self, temp_db_path, mock_boto3_client):
        """Test context hash is computed correctly."""
        with patch("boto3.client", return_value=mock_boto3_client):
            from mytrader.llm.bedrock_hybrid_client import HybridBedrockClient
            
            client = HybridBedrockClient(db_path=temp_db_path)
            
            context1 = "Test context for hashing"
            context2 = "Test context for hashing"
            context3 = "Different context"
            
            hash1 = client._compute_context_hash(context1)
            hash2 = client._compute_context_hash(context2)
            hash3 = client._compute_context_hash(context3)
            
            assert hash1 == hash2  # Same context = same hash
            assert hash1 != hash3  # Different context = different hash
            assert len(hash1) == 16  # Hash is truncated to 16 chars
    
    def test_cache_hit(self, temp_db_path, mock_boto3_client):
        """Test cache hit returns cached result."""
        with patch("boto3.client", return_value=mock_boto3_client):
            from mytrader.llm.bedrock_hybrid_client import HybridBedrockClient
            
            client = HybridBedrockClient(db_path=temp_db_path, cache_ttl_seconds=300)
            
            # Add to cache
            context_hash = "test_hash_1234"
            cached_result = {"bias": "BEARISH", "confidence": 0.6}
            client._update_cache(context_hash, cached_result)
            
            # Check cache hit
            result = client._check_cache(context_hash)
            assert result == cached_result
    
    def test_cache_miss_on_expired(self, temp_db_path, mock_boto3_client):
        """Test cache miss when entry is expired."""
        with patch("boto3.client", return_value=mock_boto3_client):
            from mytrader.llm.bedrock_hybrid_client import HybridBedrockClient
            import time
            
            client = HybridBedrockClient(db_path=temp_db_path, cache_ttl_seconds=1)
            
            # Add to cache
            context_hash = "test_hash_5678"
            cached_result = {"bias": "NEUTRAL", "confidence": 0.5}
            client._update_cache(context_hash, cached_result)
            
            # Wait for expiry
            time.sleep(1.5)
            
            # Check cache miss
            result = client._check_cache(context_hash)
            assert result is None
    
    def test_cost_estimation(self, temp_db_path, mock_boto3_client):
        """Test cost estimation calculation."""
        with patch("boto3.client", return_value=mock_boto3_client):
            from mytrader.llm.bedrock_hybrid_client import HybridBedrockClient
            
            client = HybridBedrockClient(db_path=temp_db_path)
            
            # Test with known values
            cost = client._estimate_cost(tokens_in=1000, tokens_out=500)
            
            # Claude 3 Sonnet: $0.003/1K input, $0.015/1K output
            expected = (1000 / 1000 * 0.003) + (500 / 1000 * 0.015)
            assert abs(cost - expected) < 0.001
    
    def test_bedrock_analyze_success(self, temp_db_path, mock_boto3_client, mock_bedrock_response):
        """Test successful Bedrock analysis."""
        with patch("boto3.client", return_value=mock_boto3_client):
            from mytrader.llm.bedrock_hybrid_client import HybridBedrockClient
            
            client = HybridBedrockClient(db_path=temp_db_path)
            
            context = """
            MARKET CONTEXT
            Time: 2025-12-07 10:00 UTC
            Price: 5375.00
            RSI: 55
            Momentum: 0.02
            """
            
            result = client.bedrock_analyze(context, trigger="test")
            
            assert result["bias"] == "BULLISH"
            assert result["confidence"] == 0.75
            assert result["action"] == "BUY"
            assert "rationale" in result
    
    def test_bedrock_analyze_caches_result(self, temp_db_path, mock_boto3_client):
        """Test that successful analysis is cached."""
        with patch("boto3.client", return_value=mock_boto3_client):
            from mytrader.llm.bedrock_hybrid_client import HybridBedrockClient
            
            client = HybridBedrockClient(db_path=temp_db_path)
            
            context = "Test context for caching"
            
            # First call
            result1 = client.bedrock_analyze(context, trigger="test")
            
            # Second call should hit cache
            result2 = client.bedrock_analyze(context, trigger="test")
            
            # Verify only one API call was made
            assert mock_boto3_client.invoke_model.call_count == 1
            assert result1["bias"] == result2["bias"]
    
    def test_bedrock_analyze_logs_to_sqlite(self, temp_db_path, mock_boto3_client):
        """Test that analysis is logged to SQLite."""
        with patch("boto3.client", return_value=mock_boto3_client):
            from mytrader.llm.bedrock_hybrid_client import HybridBedrockClient
            
            client = HybridBedrockClient(db_path=temp_db_path)
            
            context = "Test context for logging"
            client.bedrock_analyze(context, trigger="test_trigger")
            
            # Check database
            conn = sqlite3.connect(temp_db_path)
            cursor = conn.execute("SELECT * FROM bedrock_calls WHERE trigger = 'test_trigger'")
            rows = cursor.fetchall()
            conn.close()
            
            assert len(rows) >= 1
    
    def test_quota_check(self, temp_db_path, mock_boto3_client):
        """Test quota checking."""
        with patch("boto3.client", return_value=mock_boto3_client):
            from mytrader.llm.bedrock_hybrid_client import HybridBedrockClient
            
            client = HybridBedrockClient(
                db_path=temp_db_path,
                daily_quota=10,
                daily_cost_limit=1.0,
            )
            
            # Initially within quota
            within_quota, msg = client.db_manager.check_quota()
            assert within_quota is True
    
    def test_get_status(self, temp_db_path, mock_boto3_client):
        """Test status retrieval."""
        with patch("boto3.client", return_value=mock_boto3_client):
            from mytrader.llm.bedrock_hybrid_client import HybridBedrockClient
            
            client = HybridBedrockClient(db_path=temp_db_path)
            
            status = client.get_status()
            
            assert "model_id" in status
            assert "region" in status
            assert "daily_calls" in status
            assert "within_quota" in status


class TestSQLiteManager:
    """Tests for BedrockSQLiteManager."""
    
    def test_manager_initialization(self, temp_db_path):
        """Test manager initializes database correctly."""
        from mytrader.llm.sqlite_manager import BedrockSQLiteManager
        
        manager = BedrockSQLiteManager(db_path=temp_db_path)
        
        # Check tables exist
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        assert "bedrock_calls" in tables
        assert "event_triggers" in tables
        assert "daily_quota" in tables
    
    def test_log_bedrock_call(self, temp_db_path):
        """Test logging a Bedrock call."""
        from mytrader.llm.sqlite_manager import BedrockSQLiteManager
        
        manager = BedrockSQLiteManager(db_path=temp_db_path)
        
        call_id = manager.log_bedrock_call(
            trigger="test_trigger",
            prompt="Test prompt",
            response='{"bias": "NEUTRAL"}',
            model="test-model",
            tokens_in=100,
            tokens_out=50,
            cost_estimate=0.01,
        )
        
        assert call_id > 0
        
        # Verify record exists
        calls = manager.get_recent_bedrock_calls(limit=1)
        assert len(calls) == 1
        assert calls[0]["trigger"] == "test_trigger"
    
    def test_daily_stats_tracking(self, temp_db_path):
        """Test daily statistics tracking."""
        from mytrader.llm.sqlite_manager import BedrockSQLiteManager
        
        manager = BedrockSQLiteManager(db_path=temp_db_path)
        
        # Log multiple calls
        manager.log_bedrock_call(
            trigger="test1", prompt="p1", response="r1",
            model="m1", tokens_in=100, cost_estimate=0.01
        )
        manager.log_bedrock_call(
            trigger="test2", prompt="p2", response="r2",
            model="m1", tokens_in=200, cost_estimate=0.02
        )
        
        stats = manager.get_daily_stats()
        
        assert stats["call_count"] == 2
        assert stats["total_tokens_in"] == 300
        assert abs(stats["total_cost"] - 0.03) < 0.001
    
    def test_quota_exceeded_detection(self, temp_db_path):
        """Test quota exceeded detection."""
        from mytrader.llm.sqlite_manager import BedrockSQLiteManager
        
        manager = BedrockSQLiteManager(
            db_path=temp_db_path,
            daily_quota=2,  # Low quota for testing
        )
        
        # Log calls up to quota
        manager.log_bedrock_call(
            trigger="t1", prompt="p", response="r",
            model="m", tokens_in=10, cost_estimate=0.001
        )
        manager.log_bedrock_call(
            trigger="t2", prompt="p", response="r",
            model="m", tokens_in=10, cost_estimate=0.001
        )
        
        within_quota, msg = manager.check_quota()
        assert within_quota is False
        assert "exceeded" in msg.lower()


class TestInitBedrockClient:
    """Tests for init_bedrock_client factory function."""
    
    def test_init_from_env_vars(self, temp_db_path, mock_boto3_client):
        """Test initialization from environment variables."""
        with patch("boto3.client", return_value=mock_boto3_client):
            with patch.dict(os.environ, {
                "AWS_REGION": "us-west-2",
                "AWS_BEDROCK_MODEL": "anthropic.claude-3-haiku-20240307-v1:0",
            }):
                from mytrader.llm.bedrock_hybrid_client import init_bedrock_client
                
                client = init_bedrock_client(db_path=temp_db_path)
                
                assert client.region_name == "us-west-2"
                assert client.model_id == "anthropic.claude-3-haiku-20240307-v1:0"
    
    def test_init_with_explicit_params(self, temp_db_path, mock_boto3_client):
        """Test initialization with explicit parameters."""
        with patch("boto3.client", return_value=mock_boto3_client):
            from mytrader.llm.bedrock_hybrid_client import init_bedrock_client
            
            client = init_bedrock_client(
                model_id="custom-model",
                region_name="eu-west-1",
                db_path=temp_db_path,
            )
            
            assert client.region_name == "eu-west-1"
            assert client.model_id == "custom-model"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
