import tempfile
from pathlib import Path

from mytrader.learning.trade_learning import TradeLearningPayload
from mytrader.rag.local_knowledge_base import LocalKnowledgeBase
from mytrader.rag.local_vector_store import LocalVectorStore


def test_local_vector_store_similarity(tmp_path):
    store_path = tmp_path / "kb.sqlite"
    store = LocalVectorStore(db_path=str(store_path), dimension=64)
    store.upsert_document("buy1", "BUY trend up volatility low", {"side": "BUY"})
    store.upsert_document("sell1", "SELL trend down volatility high", {"side": "SELL"})
    results = store.similarity_search("BUY trend up", top_k=1)
    assert results
    assert results[0][0] == "buy1"


def test_local_kb_returns_adjustment(tmp_path):
    kb_path = tmp_path / "kb.sqlite"
    kb = LocalKnowledgeBase(store_path=str(kb_path), embedding_dim=64)
    for idx in range(3):
        payload = TradeLearningPayload(
            trade_cycle_id=f"cycle_{idx}",
            symbol="ES",
            contract="ES",
            quantity=1,
            side="BUY",
            entry_time="2025-12-12T10:00:00Z",
            entry_price=100.0 + idx,
            exit_time="2025-12-12T10:30:00Z",
            exit_price=101.0 + idx,
            stop_loss=99.0,
            take_profit=103.0,
            signal_type="SCALP_BUY",
            signal_confidence=0.7,
            regime="UPTREND",
            volatility="LOW",
            outcome="WIN",
            pnl=150.0,
        )
        kb.record_trade(payload)

    result = kb.query({
        "action": "BUY",
        "trend": "UPTREND",
        "volatility": "LOW",
        "confidence": 0.7,
    })
    assert result
    assert result["similar_patterns"] >= 3
    assert result["confidence_adjustment"] > 0.0
