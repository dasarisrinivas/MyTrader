"""Miscellaneous configs — learning, feature flags, observability."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import os
from datetime import time

@dataclass
class LearningConfig:
    """Local learning/ingestion hooks."""
    enabled: bool = True
    outcomes_dir: str = "rag_data/training/trade_outcomes"
    history_dir: str = "rag_data/history_snapshots"
    history_days: int = 45
    ingest_window_days: int = 60
    reason_code_retention_days: int = 90


@dataclass
class FeatureFlagsConfig:
    """Feature flags to safely roll out guardrails."""
    enforce_entry_risk_checks: bool = field(default_factory=lambda: os.environ.get("FF_ENTRY_RISK_GUARDS", "true").lower() not in {"0", "false", "no"})
    enforce_wait_blocking: bool = field(default_factory=lambda: os.environ.get("FF_WAIT_BLOCKING", "true").lower() not in {"0", "false", "no"})
    enforce_reduce_only_exits: bool = field(default_factory=lambda: os.environ.get("FF_EXIT_GUARDS", "true").lower() not in {"0", "false", "no"})
    enable_learning_hooks: bool = field(default_factory=lambda: os.environ.get("FF_LEARNING_HOOKS", "true").lower() not in {"0", "false", "no"})
    enable_chop_exception: bool = field(default_factory=lambda: os.environ.get("ENABLE_CHOP_EXCEPTION", "false").lower() in {"1", "true", "yes"})


@dataclass
class ObservabilityConfig:
    """Observability settings (Prometheus exporter)."""
    prometheus_enabled: bool = field(default_factory=lambda: os.environ.get("PROMETHEUS_ENABLED", "False").lower() in {"1", "true", "yes"})
    prometheus_addr: str = field(default_factory=lambda: os.environ.get("PROMETHEUS_ADDR", "0.0.0.0"))
    prometheus_port: int = field(default_factory=lambda: int(os.environ.get("PROMETHEUS_PORT", "8000")))
    env_label: str = field(default_factory=lambda: os.environ.get("DEPLOY_ENV", "local"))


@dataclass
class DynamicSupportConfig:
    """Dynamic structural support floor — auto-computed from market data.

    The floor is min(PDL, weekly_low, OR_low) - buffer_points.
    No hardcoded price level needed.
    """
    buffer_points: float = 5.0
    min_sources: int = 1
    use_pdl: bool = True
    use_weekly_low: bool = True
    use_or_low: bool = True



