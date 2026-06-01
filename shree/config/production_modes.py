"""Production mode & feature-flag registry — MES Phase 1.

PURPOSE
  Central, declarative separation of production vs. research configuration.
  This module ONLY *declares* modes and flags. It does NOT wire any flag into a
  decision path, so importing/reading it cannot change strategy behavior. The
  validated Phase 1 baseline is byte-identical regardless of this module.

PROVEN STATE (locked 2026-05-31)
  Phase 1 baseline is the only production-valid configuration
  (+$900.90, 134 trades, PF 1.30, Sharpe 1.80, DD -1.32%). Every adaptive /
  gating enhancement (RR modification, sizing, ADX≥22 gate, RTH-only, ORB
  removal, bucket learning, memory decay) was tested and REJECTED — see
  docs/adaptive_architecture_implementation.md. All remain OFF by default.

USAGE
  from shree.config.production_modes import PRODUCTION, ProductionConfig
  if PRODUCTION.enable_adx_filter: ...   # always False in BASELINE
"""
from __future__ import annotations

import os
from dataclasses import dataclass


# ── Modes ────────────────────────────────────────────────────────────────────
MODE_BASELINE = "BASELINE"          # production default — validated Phase 1
MODE_RESEARCH_ADX = "RESEARCH_ADX"  # research only — never the live default
MODE_RESEARCH_RTH = "RESEARCH_RTH"  # research only
MODE_EXPERIMENTAL = "EXPERIMENTAL"  # research only
VALID_MODES = (MODE_BASELINE, MODE_RESEARCH_ADX, MODE_RESEARCH_RTH, MODE_EXPERIMENTAL)


def _envb(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class ProductionConfig:
    """Frozen registry of production mode + experimental feature flags.

    EVERY experimental flag defaults to False. In BASELINE mode (the default),
    the system runs the validated Phase 1 strategy with no gating or adaptive
    logic. Flags are read-only declarations; enabling one does nothing unless a
    research/sandbox code path explicitly consults it (production paths do not).
    """
    mode: str = os.environ.get("MES_MODE", MODE_BASELINE)

    # Experimental filters — REJECTED for production (kept for isolated research)
    enable_adx_filter: bool = _envb("ENABLE_ADX_FILTER", False)       # ADX≥22 gate (failed >30% trade-loss rule)
    enable_session_filter: bool = _envb("ENABLE_SESSION_FILTER", False)  # RTH-only (removes profitable sessions)
    enable_orb_filter: bool = _envb("ENABLE_ORB_FILTER", False)       # drop OR_BREAK (overfit, small sample)

    # Adaptive layer — REJECTED (anti-predictive / no forward signal)
    enable_adaptive_sizing: bool = _envb("ENABLE_ADAPTIVE_SIZING", False)
    enable_bucket_learning: bool = _envb("ENABLE_BUCKET_LEARNING", False)

    def __post_init__(self):
        if self.mode not in VALID_MODES:
            object.__setattr__(self, "mode", MODE_BASELINE)

    @property
    def is_baseline(self) -> bool:
        return self.mode == MODE_BASELINE

    @property
    def any_experimental_enabled(self) -> bool:
        return any((self.enable_adx_filter, self.enable_session_filter,
                    self.enable_orb_filter, self.enable_adaptive_sizing,
                    self.enable_bucket_learning))

    def summary(self) -> str:
        return (f"MES_MODE={self.mode} | adx_filter={self.enable_adx_filter} "
                f"session_filter={self.enable_session_filter} orb_filter={self.enable_orb_filter} "
                f"adaptive_sizing={self.enable_adaptive_sizing} bucket_learning={self.enable_bucket_learning}")


# Singleton — read once at import. Default = BASELINE, all flags False.
PRODUCTION = ProductionConfig()
