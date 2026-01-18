import types


class DummyContractSpec:
    tick_size = 0.25


def _make_manager(mode="paper"):
    # Minimal manager stub to exercise RiskController.calculate_stop_loss
    m = types.SimpleNamespace()
    m._min_stop_distance = 1.0  # 4 ticks on MES
    m.contract_spec = DummyContractSpec()
    m.trading_mode = mode
    m._commission_per_side = 1.0  # MES default in trade_math

    # settings stubs
    m.settings = types.SimpleNamespace()
    m.settings.trading = types.SimpleNamespace()
    m.settings.trading.tick_size = 0.25
    m.settings.data = types.SimpleNamespace()
    m.settings.data.ibkr_symbol = "MES"
    return m


def test_scalp_mode_uses_tighter_multipliers_and_min_stop():
    from mytrader.execution.components.risk_controller import RiskController

    manager = _make_manager(mode="paper")
    rc = RiskController(manager)

    entry = 5000.0
    atr = 2.0
    sl, tp, meta = rc.calculate_stop_loss(
        entry_price=entry,
        action="SCALP_BUY",
        atr=atr,
        regime_params={},
    )

    # Stop offset should be max(atr*1.0, min_stop_distance=1.0) = 2.0
    assert abs((entry - sl) - 2.0) < 1e-9

    # Target offset in scalp mode uses 1.5x ATR by default => 3.0
    assert abs((tp - entry) - 3.0) < 1e-9

    assert meta.get("scalp_mode") is True


def test_non_scalp_mode_requires_tp_beyond_stop_plus_tick():
    from mytrader.execution.components.risk_controller import RiskController

    manager = _make_manager(mode="paper")
    rc = RiskController(manager)

    entry = 5000.0
    atr = 1.0
    sl, tp, meta = rc.calculate_stop_loss(
        entry_price=entry,
        action="BUY",
        atr=atr,
        regime_params={},
    )

    stop_dist = entry - sl
    target_dist = tp - entry

    # Defaults: stop = 2.0*ATR=2.0, target = max(4.0*ATR=4.0, stop+tick=2.25) => 4.0
    assert abs(stop_dist - 2.0) < 1e-9
    assert target_dist + 1e-9 >= stop_dist + 0.25
    assert meta.get("scalp_mode") is False
