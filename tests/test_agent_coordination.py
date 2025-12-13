"""Tests for AgentBus coordination utilities."""
from datetime import timedelta

from mytrader.hybrid.coordination import AgentBus


def test_agent_bus_publish_and_stale_detection():
    bus = AgentBus(default_ttl_seconds=1.0)
    message = bus.publish("features", {"value": 1}, producer="test", ttl_seconds=1.0)
    assert bus.get_fresh("features") is not None

    # Force message to appear stale by adjusting timestamp
    message.created_at -= timedelta(seconds=5)
    assert bus.get_fresh("features") is None


def test_agent_bus_sequence_increments():
    bus = AgentBus()
    msg1 = bus.publish("channel", {"v": 1}, producer="first")
    msg2 = bus.publish("channel", {"v": 2}, producer="second")
    assert msg2.sequence_id == msg1.sequence_id + 1
