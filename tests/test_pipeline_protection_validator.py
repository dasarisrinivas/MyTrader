from mytrader.risk.protection_validator import calculate_protection


def test_pipeline_scalp_sell_is_clamped_to_min_ticks():
    # Pipeline provided wide offsets (like your log 4/8 points). For MES scalps we still allow,
    # but we also enforce minimum tick safety (>= 4 ticks each).
    prot = calculate_protection(
        action="SCALP_SELL",
        entry_price=7000.0,
        stop_points=0.5,   # too tight (2 ticks)
        target_points=0.5,  # too tight (2 ticks)
        atr_value=2.0,
        tick_size=0.25,
        volatility="MED",
        trading_mode="paper",
        symbol="MES",
    )
    assert prot.stop_offset >= 1.0
    assert prot.target_offset >= 1.0


def test_pipeline_live_enforces_min_take_profit_mes():
    # In live mode MES requires a minimum TP distance (spec min_take_profit_points_live=1.25)
    prot = calculate_protection(
        action="SCALP_BUY",
        entry_price=7000.0,
        stop_points=1.0,
        target_points=0.25,  # 1 tick TP is too small
        atr_value=2.0,
        tick_size=0.25,
        volatility="MED",
        trading_mode="live",
        symbol="MES",
    )
    assert prot.target_offset >= 1.25
