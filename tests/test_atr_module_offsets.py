import pytest

from mytrader.risk.atr_module import compute_protective_offsets


def test_compute_protective_offsets_non_scalper_tighter_stop():
    # Representative MES/ES 1m ATR in points
    atr = 2.5218
    tick_size = 0.25

    offsets = compute_protective_offsets(
        atr_value=atr,
        tick_size=tick_size,
        scalper=False,
        volatility="MEDIUM",
        current_price=6927.75,
    )

    # For high-priced instruments, atr_module uses a dynamic threshold
    # (0.05% of price) to decide if ATR is "available". With current_price
    # provided, this falls back to percentage-based sizing.
    assert offsets.fallback_used is True
    assert offsets.reason == "ATR below threshold"
    assert offsets.stop_offset == pytest.approx(6927.75 * 0.0004, rel=1e-6)
    assert offsets.target_offset == pytest.approx(6927.75 * 0.0008, rel=1e-6)


def test_compute_protective_offsets_scalper_still_tighter():
    atr = 2.5218
    tick_size = 0.25

    offsets = compute_protective_offsets(
        atr_value=atr,
        tick_size=tick_size,
        scalper=True,
        volatility="MEDIUM",
        current_price=6927.75,
    )

    # Same fallback behavior applies for scalper when ATR is below the dynamic threshold.
    assert offsets.fallback_used is True
    assert offsets.reason == "ATR below threshold"
    assert offsets.stop_offset == pytest.approx(6927.75 * 0.0004, rel=1e-6)
    assert offsets.target_offset == pytest.approx(6927.75 * 0.0008, rel=1e-6)
