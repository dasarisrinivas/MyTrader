"""Tests for live GC enablement — validate_for_live_gc and risk scaling."""
from __future__ import annotations

import pytest

from shree.config.gold import GoldStrategyConfig
from shree.execution.gold.risk import DailyState, GoldRiskManager
from shree.risk.trade_math import get_contract_spec


# ── validate_for_live_gc ──────────────────────────────────────────────────────

class TestValidateForLiveGc:
    def _gc_live_cfg(self) -> GoldStrategyConfig:
        cfg = GoldStrategyConfig(enabled=True, symbol="GC", allow_gc=True, simulation=False)
        return cfg

    def test_not_gc_always_passes(self) -> None:
        cfg = GoldStrategyConfig(enabled=True, symbol="MGC", simulation=False)
        cfg.validate_for_live_gc(paper_trade_count=0, paper_win_rate=0.0)

    def test_simulation_mode_always_passes(self) -> None:
        cfg = GoldStrategyConfig(enabled=True, symbol="GC", allow_gc=True, simulation=True)
        # simulation=True → gate is bypassed even with zero track record
        cfg.validate_for_live_gc(paper_trade_count=0, paper_win_rate=0.0)

    def test_insufficient_paper_trades_raises(self) -> None:
        cfg = self._gc_live_cfg()
        cfg.risk.gc_min_paper_trades = 50
        with pytest.raises(ValueError, match="paper trades"):
            cfg.validate_for_live_gc(paper_trade_count=10, paper_win_rate=0.60)

    def test_low_win_rate_raises(self) -> None:
        cfg = self._gc_live_cfg()
        cfg.risk.gc_min_paper_trades = 50
        cfg.risk.gc_min_win_rate = 0.45
        with pytest.raises(ValueError, match="win rate"):
            cfg.validate_for_live_gc(paper_trade_count=60, paper_win_rate=0.30)

    def test_sufficient_track_record_passes(self) -> None:
        cfg = self._gc_live_cfg()
        cfg.risk.gc_min_paper_trades = 50
        cfg.risk.gc_min_win_rate = 0.45
        # Should not raise
        cfg.validate_for_live_gc(paper_trade_count=55, paper_win_rate=0.50)

    def test_exact_minimums_pass(self) -> None:
        cfg = self._gc_live_cfg()
        cfg.risk.gc_min_paper_trades = 50
        cfg.risk.gc_min_win_rate = 0.45
        cfg.validate_for_live_gc(paper_trade_count=50, paper_win_rate=0.45)


# ── gc_adjusted_risk_usd ──────────────────────────────────────────────────────

class TestGcAdjustedRisk:
    def test_mgc_returns_unmodified(self) -> None:
        cfg = GoldStrategyConfig(symbol="MGC")
        cfg.risk.max_risk_per_trade_usd = 100.0
        assert cfg.gc_adjusted_risk_usd() == 100.0

    def test_gc_applies_scale(self) -> None:
        cfg = GoldStrategyConfig(symbol="GC", allow_gc=True)
        cfg.risk.max_risk_per_trade_usd = 500.0
        cfg.risk.gc_risk_scale = 0.1
        assert abs(cfg.gc_adjusted_risk_usd() - 50.0) < 0.001

    def test_gc_custom_scale(self) -> None:
        cfg = GoldStrategyConfig(symbol="GC", allow_gc=True)
        cfg.risk.max_risk_per_trade_usd = 1000.0
        cfg.risk.gc_risk_scale = 0.2
        assert abs(cfg.gc_adjusted_risk_usd() - 200.0) < 0.001


# ── GoldRiskManager effective_max_risk ───────────────────────────────────────

class TestRiskManagerEffectiveMaxRisk:
    def test_default_uses_config_max_risk(self) -> None:
        spec = get_contract_spec("MGC")
        from shree.config.gold import GoldRiskConfig
        risk_cfg = GoldRiskConfig(max_risk_per_trade_usd=50.0)
        mgr = GoldRiskManager(risk_cfg, spec)
        # Stop of 5 pts × $10/pt = $50 risk/contract → exactly 1 contract
        result = mgr.size_position(5.0, DailyState())
        assert result.approved
        assert result.contracts == 1

    def test_effective_max_risk_override_reduces_contracts(self) -> None:
        spec = get_contract_spec("GC")    # $100/pt
        from shree.config.gold import GoldRiskConfig
        # Without override: $500 max risk → $500/$500 = 1 contract
        # With 0.1× override: $50 effective → $50/$500 = 0.1 → still 1 (floor)
        # Use a tighter stop to see the difference
        risk_cfg = GoldRiskConfig(max_risk_per_trade_usd=500.0, max_contracts_hard_cap=10,
                                   daily_loss_limit_usd=10000.0)
        mgr_no_scale = GoldRiskManager(risk_cfg, spec)
        mgr_scaled   = GoldRiskManager(risk_cfg, spec, effective_max_risk=50.0)

        # Stop of 1 pt × $100/pt = $100 risk/contract
        # no_scale: floor(500/100) = 5 contracts
        # scaled:   floor(50/100) = 0 → floor(1) = 1 contract
        r_unscaled = mgr_no_scale.size_position(1.0, DailyState())
        r_scaled = mgr_scaled.size_position(1.0, DailyState())

        assert r_unscaled.contracts == 5
        assert r_scaled.contracts == 1   # floored to minimum

    def test_gc_config_auto_scales_via_constructor(self) -> None:
        """GoldStrategyConfig.gc_adjusted_risk_usd feeds into risk manager."""
        cfg = GoldStrategyConfig(symbol="GC", allow_gc=True)
        cfg.risk.max_risk_per_trade_usd = 500.0
        cfg.risk.gc_risk_scale = 0.1
        cfg.risk.max_contracts_hard_cap = 10
        cfg.risk.daily_loss_limit_usd = 10000.0
        spec = get_contract_spec("GC")
        mgr = GoldRiskManager(cfg.risk, spec, effective_max_risk=cfg.gc_adjusted_risk_usd())
        # effective = $50, stop of 1pt × $100 = $100/contract → floor(50/100)=0 → 1
        result = mgr.size_position(1.0, DailyState())
        assert result.approved
        assert result.contracts == 1
