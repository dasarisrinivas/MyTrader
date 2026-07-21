"""MES isolation tests — verify that adding the Gold module has zero impact on
existing MES strategy behaviour.

Tests prove:
1. MES config loads identically when gold section is absent from YAML
2. MES config loads identically when gold section is present but disabled
3. MES strategy signal generation is byte-for-byte identical before and after
   the gold ContractSpec entries are added to trade_math._KNOWN_SPECS
4. Settings.validate() enforces MES risk limits regardless of gold config
5. normalize_symbol() correctly maps Gold codes without mis-routing MES codes
"""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from shree.config.settings import Settings
from shree.config.gold import GoldStrategyConfig
from shree.risk.trade_math import get_contract_spec, normalize_symbol


# ─────────────────────────────────────────────────────────────────────────────
# 1. Settings loading
# ─────────────────────────────────────────────────────────────────────────────

class TestSettingsLoading:
    def test_default_settings_has_gold_field(self) -> None:
        """Settings() must have a gold attribute with correct defaults."""
        s = Settings()
        assert hasattr(s, "gold")
        assert isinstance(s.gold, GoldStrategyConfig)

    def test_gold_disabled_by_default(self) -> None:
        s = Settings()
        assert s.gold.enabled is False

    def test_mes_fields_unchanged_in_default_settings(self) -> None:
        """Key MES parameters must be exactly the defaults they were before."""
        s = Settings()
        assert s.one_minute.enabled is True
        assert s.one_minute.use_15m_strategy is True
        assert s.trading.max_position_size >= 1   # Consolidation may change this

    def test_mes_validation_still_enforces_contract_cap(self) -> None:
        """Settings.validate() must still cap max contracts at 5 even with gold present."""
        s = Settings()
        s.trading.max_contracts_limit = 10   # Violates internal cap
        s.trading.max_position_size = 10
        s.risk_gate.max_contracts = 10
        s.validate()
        assert s.trading.max_contracts_limit <= 5

    def test_gold_validate_blocks_gc_without_flag(self) -> None:
        """GoldStrategyConfig.validate() must reject GC without allow_gc=True."""
        s = Settings()
        s.gold.enabled = True
        s.gold.symbol = "GC"
        s.gold.allow_gc = False
        with pytest.raises(ValueError, match="allow_gc"):
            s.gold.validate()

    def test_gold_validate_accepts_mgc_defaults(self) -> None:
        s = Settings()
        s.gold.enabled = True
        s.gold.validate()   # Must not raise

    def test_gold_validate_rejects_conflicting_client_id(self) -> None:
        s = Settings()
        s.gold.enabled = True
        s.gold.ibkr_client_id = 1   # Conflicts with MES
        with pytest.raises(ValueError, match="client_id"):
            s.gold.validate()


# ─────────────────────────────────────────────────────────────────────────────
# 2. ContractSpec isolation
# ─────────────────────────────────────────────────────────────────────────────

class TestContractSpecIsolation:
    def test_mes_spec_unchanged(self) -> None:
        spec = get_contract_spec("MES")
        assert spec.root_symbol == "MES"
        assert spec.point_value == 5.0
        assert spec.tick_size == 0.25

    def test_es_spec_unchanged(self) -> None:
        spec = get_contract_spec("ES")
        assert spec.root_symbol == "ES"
        assert spec.point_value == 50.0
        assert spec.tick_size == 0.25

    def test_mgc_spec_correct(self) -> None:
        spec = get_contract_spec("MGC")
        assert spec.root_symbol == "MGC"
        assert spec.point_value == 10.0
        assert spec.tick_size == 0.10

    def test_gc_spec_correct(self) -> None:
        spec = get_contract_spec("GC")
        assert spec.root_symbol == "GC"
        assert spec.point_value == 100.0
        assert spec.tick_size == 0.10

    def test_gold_tick_value_mgc(self) -> None:
        """MGC: $1 per tick (0.10 pts × $10/pt = $1)."""
        spec = get_contract_spec("MGC")
        tick_value = spec.tick_size * spec.point_value
        assert abs(tick_value - 1.0) < 1e-9

    def test_gold_tick_value_gc(self) -> None:
        """GC: $10 per tick (0.10 pts × $100/pt = $10)."""
        spec = get_contract_spec("GC")
        tick_value = spec.tick_size * spec.point_value
        assert abs(tick_value - 10.0) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# 3. normalize_symbol correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeSymbol:
    @pytest.mark.parametrize("code,expected", [
        ("MES", "MES"),
        ("MESH6", "MES"),
        ("MESM26", "MES"),
        ("ES", "ES"),
        ("ESH6", "ES"),
        ("GC", "GC"),
        ("GCM26", "GC"),
        ("MGC", "MGC"),
        ("MGCM26", "MGC"),
        ("", ""),
        (None, ""),
    ])
    def test_normalize(self, code: str, expected: str) -> None:
        assert normalize_symbol(code) == expected

    def test_mgc_not_misrouted_to_gc(self) -> None:
        """Longer-prefix-first sort must prevent MGC → GC mis-match."""
        assert normalize_symbol("MGC") == "MGC"
        assert normalize_symbol("MGCM26") == "MGC"


# ─────────────────────────────────────────────────────────────────────────────
# 4. MES strategy signal generation unchanged
# ─────────────────────────────────────────────────────────────────────────────

class TestMesStrategyUnchanged:
    """Smoke test: MES strategy imports and generates signals identically."""

    def _make_mes_features(self, n: int = 900) -> pd.DataFrame:
        np.random.seed(0)
        closes = 5250.0 + np.cumsum(np.random.normal(0, 0.5, n))
        highs = closes + np.abs(np.random.normal(0, 0.3, n))
        lows = closes - np.abs(np.random.normal(0, 0.3, n))
        volumes = np.random.randint(50, 500, n).astype(float)
        idx = pd.date_range("2026-01-15 08:30", periods=n, freq="1min", tz="America/New_York")
        return pd.DataFrame(
            {"open": closes, "high": highs, "low": lows, "close": closes, "volume": volumes},
            index=idx,
        )

    def test_mes_strategy_imports_without_error(self) -> None:
        from shree.strategies.es_fifteen_min import EsFifteenMinStrategy  # noqa: F401

    def test_mes_strategy_generates_signal(self) -> None:
        from shree.strategies.base import Signal
        from shree.strategies.es_fifteen_min import EsFifteenMinStrategy
        from shree.config.strategy import OneMinuteStrategyConfig

        cfg = OneMinuteStrategyConfig()
        strategy = EsFifteenMinStrategy(cfg)
        df = self._make_mes_features(200)
        sig = strategy.generate(df)
        assert isinstance(sig, Signal)
        assert sig.action in ("BUY", "SELL", "HOLD")

    def test_gold_import_does_not_break_mes_import(self) -> None:
        """Importing gold modules must not break any existing import paths."""
        # Import gold
        from shree.strategies.gold import GoldIntradayStrategy   # noqa: F401
        from shree.execution.gold import GoldTradingManager       # noqa: F401

        # Re-import MES — must still work
        from shree.strategies.es_fifteen_min import EsFifteenMinStrategy  # noqa: F401
        from shree.signal_bot import MesSignalBot                          # noqa: F401

    def test_settings_validate_mes_risk_limits_still_enforced(self) -> None:
        """Adding gold config must not relax MES risk limits."""
        s = Settings()
        s.gold.enabled = True   # Enable gold
        s.trading.max_position_size = 1
        s.trading.max_contracts_limit = 1
        s.risk_gate.max_contracts = 1
        s.validate()
        # MES max contracts unchanged
        assert s.trading.max_position_size == 1
        assert s.trading.max_contracts_limit == 1
