import types

import pytest

import mytrader.execution.live_trading_manager as ltm
from mytrader.config import Settings
from mytrader.execution.live_trading_manager import LiveTradingManager


def _build_manager(opensearch_enabled: bool, backend: str) -> LiveTradingManager:
    settings = Settings()
    settings.aws_agents.enabled = True
    settings.rag.enabled = True
    settings.rag.opensearch_enabled = opensearch_enabled
    settings.rag.backend = backend
    return LiveTradingManager(settings, simulation_mode=True)


def test_kill_switch_blocks_agent_invoker(monkeypatch):
    manager = _build_manager(opensearch_enabled=False, backend="local_faiss")
    manager._configure_aws_agents()

    def fake_from_config(cls, *args, **kwargs):
        raise AssertionError("AgentInvoker should not be created when OpenSearch is disabled")

    monkeypatch.setattr(
        ltm.AgentInvoker,
        "from_deployed_config",
        classmethod(fake_from_config),
        raising=False,
    )
    assert manager._ensure_aws_agent_invoker() is False


def test_opensearch_enabled_allows_agent_invoker(monkeypatch):
    manager = _build_manager(opensearch_enabled=True, backend="opensearch_serverless")

    class DummyInvoker:
        def get_trading_decision(self, *args, **kwargs):
            return {}

    def fake_from_config(cls, *args, **kwargs):
        return DummyInvoker()

    class DummySnapshotBuilder:
        def __init__(self, symbol: str):
            self.symbol = symbol

        def build(self, **kwargs):
            return {"symbol": self.symbol, **kwargs}

    monkeypatch.setattr(
        ltm.AgentInvoker,
        "from_deployed_config",
        classmethod(fake_from_config),
        raising=False,
    )
    monkeypatch.setattr(ltm, "MarketSnapshotBuilder", DummySnapshotBuilder)

    manager._configure_aws_agents()
    assert manager._ensure_aws_agent_invoker() is True
    assert isinstance(manager.aws_agent_invoker, DummyInvoker)
