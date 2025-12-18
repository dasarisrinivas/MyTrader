"""Unit tests for futures trade math helpers."""
import pytest

from mytrader.config import TradingConfig
from mytrader.risk.trade_math import calculate_realized_pnl, get_contract_spec


def test_mes_half_point_move() -> None:
    """MES should pay $5 per point → $2.50 for half-point."""
    spec = get_contract_spec("MES", TradingConfig())
    gross, points = calculate_realized_pnl(5000.0, 5000.5, 1, spec)
    assert pytest.approx(points, rel=1e-6) == 0.5
    assert pytest.approx(gross, rel=1e-6) == 2.5


def test_es_half_point_move() -> None:
    """ES should pay $50 per point → $25.00 for half-point."""
    spec = get_contract_spec("ES", TradingConfig())
    gross, points = calculate_realized_pnl(5000.0, 5000.5, 1, spec)
    assert pytest.approx(points, rel=1e-6) == 0.5
    assert pytest.approx(gross, rel=1e-6) == 25.0
