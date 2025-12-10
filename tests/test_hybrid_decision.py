"""Unit and Integration Tests for Hybrid Decision Engine."""
import json
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import pandas as pd
import numpy as np

# Import hybrid modules
from mytrader.hybrid.d_engine import DeterministicEngine, DEngineSignal
from mytrader.hybrid.h_engine import HeuristicEngine, HEngineAdvisory, RAGContext
from mytrader.hybrid.confidence import ConfidenceScorer, ConfidenceResult
from mytrader.hybrid.safety import SafetyManager, SafetyCheck
from mytrader.hybrid.decision_logger import DecisionLogger
from mytrader.hybrid.hybrid_decision import HybridDecisionEngine, HybridDecision


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_features():
    """Create sample feature DataFrame for testing."""
    np.random.seed(42)
    n = 100
    
    # Generate price data
    base_price = 6000.0
    prices = [base_price]
    for _ in range(n - 1):
        prices.append(prices[-1] + np.random.normal(0, 2))
    
    timestamps = [datetime.now(timezone.utc) - timedelta(minutes=n-i) for i in range(n)]
    
    df = pd.DataFrame({
        "open": prices,
        "high": [p + abs(np.random.normal(0, 1)) for p in prices],
        "low": [p - abs(np.random.normal(0, 1)) for p in prices],
        "close": prices,
        "volume": [int(np.random.uniform(100, 1000)) for _ in prices],
    }, index=pd.DatetimeIndex(timestamps))
    
    # Add indicators
    df["RSI_14"] = 50 + np.random.normal(0, 15, n)
    df["MACD"] = np.random.normal(0, 0.5, n)
    df["MACD_signal"] = df["MACD"].rolling(9).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    df["EMA_9"] = df["close"].ewm(span=9).mean()
    df["EMA_20"] = df["close"].ewm(span=20).mean()
    df["ATR_14"] = 2.0 + np.random.normal(0, 0.5, n)
    
    return df


@pytest.fixture
def d_engine():
    """Create D-engine instance."""
    return DeterministicEngine()


@pytest.fixture
def h_engine():
    """Create H-engine instance with mock LLM."""
    return HeuristicEngine(
        llm_client=None,
        rag_storage=None,
        config={"max_calls_per_hour": 10},
    )


@pytest.fixture
def confidence_scorer():
    """Create confidence scorer."""
    return ConfidenceScorer(
        weights={"technical": 0.5, "model": 0.3, "rag": 0.2},
        confidence_threshold=0.60,
    )


@pytest.fixture
def safety_manager():
    """Create safety manager."""
    return SafetyManager(
        cooldown_minutes=1,
        max_orders_per_window=5,
        dry_run=True,
    )


# ============================================================================
# D-Engine Tests
# ============================================================================

class TestDeterministicEngine:
    """Tests for the deterministic engine."""
    
    def test_initialization(self, d_engine):
        """Test D-engine initializes correctly."""
        assert d_engine is not None
        assert d_engine.config["candidate_threshold"] == 0.55
    
    def test_candle_close_check(self, d_engine):
        """Test candle close detection."""
        now = datetime.now(timezone.utc)
        candle_time = now.replace(second=0, microsecond=0)
        
        # First check should return True
        assert d_engine.is_candle_closed(candle_time)
        
        # Process the candle
        d_engine._last_candle_processed = candle_time
        
        # Same candle should return False
        assert not d_engine.is_candle_closed(candle_time)
        
        # Next candle should return True
        next_candle = candle_time + timedelta(minutes=1)
        assert d_engine.is_candle_closed(next_candle)
    
    def test_evaluate_produces_signal(self, d_engine, sample_features):
        """Test D-engine evaluation produces valid signal."""
        current_price = float(sample_features.iloc[-1]["close"])
        candle_time = sample_features.index[-1]
        
        signal = d_engine.evaluate(
            features=sample_features,
            current_price=current_price,
            candle_time=candle_time,
            force=True,
        )
        
        assert isinstance(signal, DEngineSignal)
        assert signal.action in ["BUY", "SELL", "HOLD"]
        assert 0.0 <= signal.technical_score <= 1.0
        assert signal.entry_price == current_price
    
    def test_evaluate_with_bullish_trend(self, d_engine, sample_features):
        """Test D-engine produces BUY on bullish trend."""
        # Force bullish indicators
        sample_features["EMA_9"] = sample_features["close"] + 5
        sample_features["EMA_20"] = sample_features["close"] - 5
        sample_features["MACD_hist"] = 0.3
        sample_features["RSI_14"] = 35
        
        current_price = float(sample_features.iloc[-1]["close"])
        candle_time = sample_features.index[-1]
        
        signal = d_engine.evaluate(
            features=sample_features,
            current_price=current_price,
            candle_time=candle_time,
            force=True,
        )
        
        assert signal.action == "BUY"
        assert signal.trend_aligned
    
    def test_evaluate_with_bearish_trend(self, d_engine, sample_features):
        """Test D-engine produces SELL on bearish trend."""
        # Force bearish indicators
        sample_features["EMA_9"] = sample_features["close"] - 5
        sample_features["EMA_20"] = sample_features["close"] + 5
        sample_features["MACD_hist"] = -0.3
        sample_features["RSI_14"] = 75
        
        current_price = float(sample_features.iloc[-1]["close"])
        candle_time = sample_features.index[-1]
        
        signal = d_engine.evaluate(
            features=sample_features,
            current_price=current_price,
            candle_time=candle_time,
            force=True,
        )
        
        assert signal.action == "SELL"
        assert signal.trend_aligned
    
    def test_insufficient_data_returns_hold(self, d_engine):
        """Test D-engine returns HOLD with insufficient data."""
        # Only 10 bars
        small_df = pd.DataFrame({
            "close": [100] * 10,
            "RSI_14": [50] * 10,
        })
        
        signal = d_engine.evaluate(
            features=small_df,
            current_price=100,
            candle_time=datetime.now(timezone.utc),
            force=True,
        )
        
        assert signal.action == "HOLD"
        assert not signal.is_candidate


# ============================================================================
# H-Engine Tests
# ============================================================================

class TestHeuristicEngine:
    """Tests for the heuristic (LLM + RAG) engine."""
    
    def test_initialization(self, h_engine):
        """Test H-engine initializes correctly."""
        assert h_engine is not None
        assert h_engine.config["max_calls_per_hour"] == 10
    
    def test_should_call_respects_rate_limit(self, h_engine):
        """Test rate limiting is respected."""
        # Initially should be able to call
        can_call, reason = h_engine.should_call()
        assert can_call
        
        # Simulate reaching limit
        h_engine._call_timestamps = [datetime.now(timezone.utc).timestamp()] * 10
        h_engine._last_call_time = datetime.now(timezone.utc).timestamp()
        
        can_call, reason = h_engine.should_call()
        assert not can_call
        assert "limit" in reason.lower() or "interval" in reason.lower()
    
    def test_evaluate_without_llm_client(self, h_engine):
        """Test H-engine returns conservative advisory without LLM."""
        d_signal = DEngineSignal(
            action="BUY",
            is_candidate=True,
            technical_score=0.7,
            entry_price=6000.0,
        )
        
        advisory = h_engine.evaluate(d_signal)
        
        assert isinstance(advisory, HEngineAdvisory)
        assert advisory.recommendation == "HOLD"  # Conservative without LLM
        assert advisory.model_confidence == 0.0
    
    def test_context_hash_generation(self, h_engine):
        """Test context hash is consistent."""
        d_signal = DEngineSignal(
            action="BUY",
            is_candidate=True,
            technical_score=0.7,
            entry_price=6000.0,
            candle_close_time=datetime(2025, 12, 9, 12, 0, 0, tzinfo=timezone.utc),
        )
        
        hash1 = h_engine._generate_context_hash(d_signal)
        hash2 = h_engine._generate_context_hash(d_signal)
        
        assert hash1 == hash2  # Same input = same hash
    
    def test_cache_hit(self, h_engine):
        """Test caching works correctly."""
        d_signal = DEngineSignal(
            action="BUY",
            is_candidate=True,
            technical_score=0.7,
            entry_price=6000.0,
            candle_close_time=datetime.now(timezone.utc),
        )
        
        # First call - cache miss
        advisory1 = h_engine.evaluate(d_signal)
        
        # Second call - should hit cache
        advisory2 = h_engine.evaluate(d_signal)
        
        assert advisory2.cached


# ============================================================================
# Confidence Scorer Tests
# ============================================================================

class TestConfidenceScorer:
    """Tests for confidence scoring."""
    
    def test_initialization(self, confidence_scorer):
        """Test confidence scorer initializes correctly."""
        assert confidence_scorer is not None
        assert sum(confidence_scorer.weights.values()) == pytest.approx(1.0)
    
    def test_calculate_with_d_engine_only(self, confidence_scorer):
        """Test scoring with only D-engine signal."""
        d_signal = DEngineSignal(
            action="BUY",
            is_candidate=True,
            technical_score=0.8,
        )
        
        result = confidence_scorer.calculate(d_signal, h_advisory=None)
        
        assert isinstance(result, ConfidenceResult)
        assert 0.0 <= result.final_confidence <= 1.0
        assert result.technical_score == 0.8
    
    def test_calculate_with_consensus(self, confidence_scorer):
        """Test scoring requires consensus."""
        d_signal = DEngineSignal(
            action="BUY",
            is_candidate=True,
            technical_score=0.8,
        )
        
        h_advisory = HEngineAdvisory(
            recommendation="LONG",  # Matches BUY
            model_confidence=0.7,
            explanation="Test",
        )
        
        result = confidence_scorer.calculate(d_signal, h_advisory)
        
        assert result.action == "BUY"  # Consensus
    
    def test_no_consensus_returns_hold(self, confidence_scorer):
        """Test no consensus returns HOLD."""
        d_signal = DEngineSignal(
            action="BUY",
            is_candidate=True,
            technical_score=0.8,
        )
        
        h_advisory = HEngineAdvisory(
            recommendation="SHORT",  # Disagrees with BUY
            model_confidence=0.7,
            explanation="Test",
        )
        
        result = confidence_scorer.calculate(d_signal, h_advisory)
        
        assert result.action == "HOLD"  # No consensus
        assert not result.should_trade
    
    def test_below_threshold_returns_no_trade(self, confidence_scorer):
        """Test below threshold returns should_trade=False."""
        d_signal = DEngineSignal(
            action="BUY",
            is_candidate=True,
            technical_score=0.3,  # Low score
        )
        
        result = confidence_scorer.calculate(d_signal, h_advisory=None)
        
        assert not result.should_trade


# ============================================================================
# Safety Manager Tests
# ============================================================================

class TestSafetyManager:
    """Tests for safety manager."""
    
    def test_initialization(self, safety_manager):
        """Test safety manager initializes correctly."""
        assert safety_manager is not None
        assert safety_manager.dry_run
    
    def test_check_cooldown_first_trade(self, safety_manager):
        """Test cooldown passes on first trade."""
        check = safety_manager.check_cooldown()
        assert check.is_safe
    
    def test_check_cooldown_blocks_rapid_trades(self, safety_manager):
        """Test cooldown blocks rapid trading."""
        # Record a trade
        safety_manager.record_trade("BUY", 1, 6000.0)
        
        # Check immediately - should be blocked
        check = safety_manager.check_cooldown()
        assert not check.is_safe
        assert "remaining" in check.reason.lower()
    
    def test_check_order_limit(self, safety_manager):
        """Test order limit enforcement."""
        # Record max orders
        for _ in range(5):
            safety_manager.record_trade("BUY", 1, 6000.0)
            safety_manager._last_trade_time = None  # Reset cooldown
        
        check = safety_manager.check_order_limit()
        assert not check.is_safe
        assert "limit" in check.reason.lower()
    
    def test_emergency_stop(self, safety_manager):
        """Test emergency stop works."""
        safety_manager.trigger_emergency_stop("Test emergency")
        
        check = safety_manager.check_all()
        assert not check.is_safe
        assert check.check_type == "emergency_stop"
    
    def test_pnl_limit_triggers_emergency(self, safety_manager):
        """Test P&L limit triggers emergency stop."""
        safety_manager._peak_pnl = 1000.0
        safety_manager._current_pnl = -2000.0  # 3% drop
        
        check = safety_manager.check_pnl_limit()
        assert not check.is_safe
        assert safety_manager._emergency_stop


# ============================================================================
# Decision Logger Tests
# ============================================================================

class TestDecisionLogger:
    """Tests for decision logger."""
    
    def test_log_decision(self, tmp_path):
        """Test logging a decision."""
        logger = DecisionLogger(
            log_dir=str(tmp_path),
            json_file="test_decisions.json",
            csv_file="test_decisions.csv",
        )
        
        d_signal = DEngineSignal(
            action="BUY",
            is_candidate=True,
            technical_score=0.7,
            entry_price=6000.0,
            stop_loss=5980.0,
            take_profit=6040.0,
        )
        
        confidence_result = ConfidenceResult(
            final_confidence=0.65,
            should_trade=True,
            action="BUY",
        )
        
        safety_check = SafetyCheck(
            is_safe=True,
            reason="OK",
            check_type="all",
        )
        
        logger.log_decision(
            d_signal=d_signal,
            h_advisory=None,
            confidence_result=confidence_result,
            safety_check=safety_check,
        )
        
        # Check files were created
        assert (tmp_path / "test_decisions.json").exists()
        assert (tmp_path / "test_decisions.csv").exists()
        
        # Check JSON content
        with open(tmp_path / "test_decisions.json") as f:
            data = json.load(f)
        
        assert len(data["decisions"]) == 1
        assert data["decisions"][0]["d_engine"]["action"] == "BUY"


# ============================================================================
# Hybrid Decision Engine Integration Tests
# ============================================================================

class TestHybridDecisionEngine:
    """Integration tests for the full hybrid engine."""
    
    def test_full_evaluation_flow(self, sample_features, tmp_path):
        """Test full evaluation flow."""
        engine = HybridDecisionEngine(
            config={
                "dry_run": True,
                "d_engine": {"candidate_threshold": 0.5},
                "confidence": {"threshold": 0.5},
            }
        )
        engine.decision_logger = DecisionLogger(log_dir=str(tmp_path))
        
        current_price = float(sample_features.iloc[-1]["close"])
        candle_time = sample_features.index[-1]
        
        decision = engine.evaluate(
            features=sample_features,
            current_price=current_price,
            candle_time=candle_time,
        )
        
        assert isinstance(decision, HybridDecision)
        assert decision.action in ["BUY", "SELL", "HOLD"]
        assert 0.0 <= decision.final_confidence <= 1.0
    
    def test_llm_calls_under_limit(self, sample_features, tmp_path):
        """Test LLM calls stay under limit during batch processing."""
        engine = HybridDecisionEngine(
            config={
                "dry_run": True,
                "d_engine": {"candidate_threshold": 0.4},  # More candidates
                "h_engine": {"max_calls_per_hour": 5},
            }
        )
        engine.decision_logger = DecisionLogger(log_dir=str(tmp_path))
        
        # Process 50 bars
        for i in range(50, len(sample_features)):
            window = sample_features.iloc[:i+1]
            current_price = float(window.iloc[-1]["close"])
            candle_time = window.index[-1]
            
            engine.evaluate(
                features=window,
                current_price=current_price,
                candle_time=candle_time,
            )
        
        stats = engine.get_stats()
        
        # H-engine should be rate limited
        assert stats["h_engine_calls"] <= 5 or stats["h_engine_stats"]["calls_last_hour"] <= 5
    
    def test_dry_run_prevents_execution(self, sample_features, tmp_path):
        """Test dry-run mode prevents actual execution."""
        engine = HybridDecisionEngine(
            config={
                "dry_run": True,
                "d_engine": {"candidate_threshold": 0.3},
                "confidence": {"threshold": 0.3},
            }
        )
        engine.decision_logger = DecisionLogger(log_dir=str(tmp_path))
        
        # Force bullish signal
        sample_features["EMA_9"] = sample_features["close"] + 10
        sample_features["EMA_20"] = sample_features["close"] - 10
        sample_features["MACD_hist"] = 0.5
        
        current_price = float(sample_features.iloc[-1]["close"])
        candle_time = sample_features.index[-1]
        
        decision = engine.evaluate(
            features=sample_features,
            current_price=current_price,
            candle_time=candle_time,
        )
        
        # Even with high confidence, dry-run should prevent execution
        assert not decision.should_execute
        assert engine.is_dry_run()


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
