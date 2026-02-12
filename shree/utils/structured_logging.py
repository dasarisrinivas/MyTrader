"""Structured logging utilities shared by all agents."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import uuid

from .logger import logger


@dataclass
class StructuredLogEvent:
    """Normalized log schema to keep cross-agent telemetry consistent."""

    agent: str
    event_type: str
    message: str = ""
    severity: str = "info"
    payload: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize event to dictionary."""
        return {
            "agent": self.agent,
            "event_type": self.event_type,
            "message": self.message,
            "severity": self.severity.upper(),
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
        }


def log_structured_event(
    agent: str,
    event_type: str,
    message: str = "",
    payload: Optional[Dict[str, Any]] = None,
    severity: str = "info",
    correlation_id: Optional[str] = None,
) -> StructuredLogEvent:
    """Emit a structured log event.

    Args:
        agent: Name of agent/component emitting the event
        event_type: Short event type (e.g., "decision.generated")
        message: Human-readable message
        payload: Optional structured payload
        severity: Log level name (debug/info/warning/error)
        correlation_id: Optional trace identifier

    Returns:
        StructuredLogEvent that was emitted
    """
    if payload is None:
        payload = {}

    severity = severity.lower()
    if severity not in {"debug", "info", "warning", "error"}:
        severity = "info"

    if correlation_id is None:
        correlation_id = payload.get("correlation_id") or uuid.uuid4().hex[:12]

    event = StructuredLogEvent(
        agent=agent,
        event_type=event_type,
        message=message,
        severity=severity,
        payload=payload,
        correlation_id=correlation_id,
    )

    bound_logger = logger.bind(
        agent=event.agent,
        event_type=event.event_type,
        correlation_id=event.correlation_id,
    )
    log_method = getattr(bound_logger, severity, bound_logger.info)
    log_method(event.message or event.event_type, event=event.to_dict())
    return event


__all__ = ["StructuredLogEvent", "log_structured_event"]
