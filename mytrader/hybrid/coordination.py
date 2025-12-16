"""Agent coordination utilities for cross-agent messaging."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from threading import Lock
from typing import Any, Dict, Optional


@dataclass
class AgentMessage:
    """Message envelope shared across agents."""

    channel: str
    payload: Dict[str, Any]
    producer: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sequence_id: int = 0
    ttl_seconds: float = 10.0

    def is_stale(self, max_age_seconds: Optional[float] = None) -> bool:
        """Return True if message is stale."""
        max_age = max_age_seconds if max_age_seconds is not None else self.ttl_seconds
        if max_age <= 0:
            return False
        age = datetime.now(timezone.utc) - self.created_at
        return age > timedelta(seconds=max_age)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging/testing."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data


class AgentBus:
    """In-memory coordination bus with freshness awareness."""

    def __init__(self, default_ttl_seconds: float = 10.0):
        self._messages: Dict[str, AgentMessage] = {}
        self._lock = Lock()
        self._sequence = 0
        self.default_ttl_seconds = default_ttl_seconds

    def publish(
        self,
        channel: str,
        payload: Dict[str, Any],
        producer: str,
        ttl_seconds: Optional[float] = None,
    ) -> AgentMessage:
        """Publish a message to the bus."""
        with self._lock:
            self._sequence += 1
            message = AgentMessage(
                channel=channel,
                payload=payload,
                producer=producer,
                sequence_id=self._sequence,
                ttl_seconds=ttl_seconds or self.default_ttl_seconds,
            )
            self._messages[channel] = message
            return message

    def latest(self, channel: str) -> Optional[AgentMessage]:
        """Return the most recent message for a channel."""
        with self._lock:
            return self._messages.get(channel)

    def get_fresh(
        self,
        channel: str,
        max_age_seconds: Optional[float] = None,
    ) -> Optional[AgentMessage]:
        """Return message only if it is not stale."""
        message = self.latest(channel)
        if not message:
            return None
        if message.is_stale(max_age_seconds):
            return None
        return message

    def invalidate(self, channel: str) -> None:
        """Remove message for channel."""
        with self._lock:
            self._messages.pop(channel, None)


__all__ = ["AgentBus", "AgentMessage"]
