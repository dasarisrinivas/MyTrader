import types

import pytest


@pytest.mark.asyncio
async def test_place_hybrid_order_maps_is_scalp_metadata_to_scalp_action(monkeypatch):
    """Hybrid entry path bypasses OrderCoordinator.execute_trade_with_risk_checks.

    Ensure metadata.is_scalp opt-in still maps BUY/SELL -> SCALP_* before protection sizing.
    """

    from shree.execution.live_trading_manager import LiveTradingManager

    class DummySettings:
        class Trading:
            initial_capital = 10000
            max_position_size = 1
            tick_size = 0.25
            min_distance_ticks = 4

        trading = Trading()
        risk_gate = types.SimpleNamespace(min_stop_points=4.0)

    m = LiveTradingManager.__new__(LiveTradingManager)
    m.settings = DummySettings()
    m.simulation_mode = True
    m.status = types.SimpleNamespace(
        current_position=0,
        hybrid_market_trend="UPTREND",
        hybrid_volatility_regime="MEDIUM",
    )

    class DummyRisk:
        def get_statistics(self):
            return {}

        def position_size(self, *_args, **_kwargs):
            return 1

        def can_trade(self, _qty):
            return True

        def register_trade(self):
            return None

    m.risk = DummyRisk()

    async def _noop(*_args, **_kwargs):
        return None

    # Avoid side-effects
    m._broadcast_error = _noop
    m._broadcast_order_update = _noop
    m._record_submission_timestamp = lambda: None
    m._notify_position_opened = lambda *_args, **_kwargs: None
    m._add_reason_code = lambda *_args, **_kwargs: None

    # Entry gate + risk gate + guard
    async def enforce_entry_gates(_action, _metadata):
        return True, "OK", "sigkey"

    m.order_coordinator = types.SimpleNamespace(
        enforce_entry_gates=enforce_entry_gates,
        record_signal_key=lambda *_args, **_kwargs: None,
    )

    async def enforce_risk_gate(*_args, **_kwargs):
        return True, None

    m._enforce_risk_gate = enforce_risk_gate
    m._validate_entry_guard = lambda *_args, **_kwargs: True

    # Prepare metadata passthrough
    m._prepare_order_metadata = lambda base, *_args, **_kwargs: dict(base)

    # Minimal features stub with iloc
    class Features:
        def __init__(self):
            self._row = {
                "ATR_14": 2.0,
                "close": 100.0,
                "RSI_14": 50,
                "MACD": 0,
                "EMA_9": 100.0,
                "EMA_20": 100.0,
                "PDH": 0,
                "PDL": 0,
            }
            self.iloc = self

        def __getitem__(self, idx):
            assert idx == -1
            return self._row

    features = Features()

    # Signal from pipeline
    signal = types.SimpleNamespace(
        action="SELL",
        confidence=0.9,
        metadata={"is_scalp": True},
    )

    # Simulation mode: should early-return after mapping
    await LiveTradingManager._place_hybrid_order(m, signal, pipeline_result=None, current_price=100.0, features=features)

    assert signal.action == "SCALP_SELL"
    assert signal.metadata.get("original_action") == "SELL"
    assert signal.metadata.get("scalp_intent_source") == "metadata.is_scalp"
