import pytest

from mytrader.rag.embedding_builder import EmbeddingBuilder
from mytrader.rag.hybrid_rag_pipeline import HybridRAGPipeline, RuleEngineResult, TradeAction


def test_no_filter_block_when_no_filters(monkeypatch):
    """Pipeline should not label empty filter list as FILTER_BLOCK."""
    pipeline = HybridRAGPipeline(config={})
    no_signal = RuleEngineResult(signal=TradeAction.HOLD, score=0.0)

    monkeypatch.setattr(pipeline.rule_engine, "evaluate", lambda _: no_signal)

    result = pipeline.process({})

    assert result.hold_reason is not None
    assert result.hold_reason.reason_code == "NO_SIGNAL"
    assert result.rule_engine.filters_blocked == []


def test_embedding_builder_retains_index_and_searches(monkeypatch):
    """Smoke test: build then search without triggering empty-index log."""
    monkeypatch.setattr("mytrader.rag.embedding_builder.SENTENCE_TRANSFORMERS_AVAILABLE", False, raising=False)
    builder = EmbeddingBuilder()
    documents = [
        ("doc1", "foo bar baz", {"type": "note"}),
        ("doc2", "another foo example", {"type": "note"}),
    ]

    assert builder.build_index(documents, save=False)

    results = builder.search("foo", top_k=1)

    assert builder.has_index()
    assert builder.index_ready
    assert not builder._logged_empty_index
    assert results
