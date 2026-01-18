from types import SimpleNamespace


def test_trend_pullback_enhancer_turns_hold_into_buy_in_micro_up():
    from mytrader.execution.components.signal_processor import SignalProcessor

    sp = SignalProcessor.__new__(SignalProcessor)
    sp.settings = SimpleNamespace(trading=SimpleNamespace(entry_filters={"enable_trend_pullback": True}))

    # Minimal logger dependency is module-level; method uses no other instance deps.

    hybrid_signal = SimpleNamespace(action="HOLD", confidence=0.1, metadata={})
    features = {"RSI_14": 48.0, "EMA_9": 7010.0}

    out = SignalProcessor._apply_trend_pullback_enhancer(
        sp,
        hybrid_signal=hybrid_signal,
        features=features,
        market_trend="MICRO_UP",
        current_price=7012.0,
    )

    assert out.action == "BUY"
    assert out.metadata["trend_pullback_enhanced"] is True
    assert out.metadata["trend_pullback_reason"] == "PULLBACK_LONG_1M"


def test_trend_pullback_enhancer_does_not_create_countertrend_trade():
    from mytrader.execution.components.signal_processor import SignalProcessor

    sp = SignalProcessor.__new__(SignalProcessor)
    sp.settings = SimpleNamespace(trading=SimpleNamespace(entry_filters={"enable_trend_pullback": True}))

    hybrid_signal = SimpleNamespace(action="HOLD", confidence=0.1, metadata={})
    features = {"RSI_14": 50.0, "EMA_9": 7010.0}

    out = SignalProcessor._apply_trend_pullback_enhancer(
        sp,
        hybrid_signal=hybrid_signal,
        features=features,
        market_trend="MICRO_UP",
        current_price=7008.0,  # below EMA9, so no long reclaim
    )

    assert out.action == "HOLD"


def test_trend_pullback_enhancer_turns_hold_into_sell_in_micro_down_flush():
    from mytrader.execution.components.signal_processor import SignalProcessor

    sp = SignalProcessor.__new__(SignalProcessor)
    sp.settings = SimpleNamespace(trading=SimpleNamespace(entry_filters={"enable_trend_pullback": True}))

    hybrid_signal = SimpleNamespace(action="HOLD", confidence=0.1, metadata={})
    # RSI below 45 in a downtrend should still be eligible for a trend-follow short
    # if price is rejected below EMA9.
    features = {"RSI_14": 41.0, "EMA_9": 7017.28}

    out = SignalProcessor._apply_trend_pullback_enhancer(
        sp,
        hybrid_signal=hybrid_signal,
        features=features,
        market_trend="MICRO_DOWN",
        current_price=7017.00,
    )

    assert out.action == "SELL"
    assert out.metadata["trend_pullback_enhanced"] is True
    assert out.metadata["trend_pullback_reason"] in {"PULLBACK_SHORT_1M", "PULLBACK_SHORT_FLUSH_1M"}
