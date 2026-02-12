"""Standardized representation for HOLD decisions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class HoldReason:
    """Explain why a trade cycle resulted in HOLD."""

    gate: str
    reason_code: str
    reason_detail: str
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return dictionary payload suitable for structured logging."""
        payload: Dict[str, Any] = {
            "gate": self.gate,
            "reason_code": self.reason_code,
            "reason_detail": self.reason_detail,
        }
        if self.context:
            payload["context"] = self.context
        return payload

