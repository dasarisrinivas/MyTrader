import pytest
from unittest.mock import MagicMock, AsyncMock
from mytrader.execution.position_manager import PositionManager, DecisionResult
from mytrader.config import TradingConfig

@pytest.fixture
def mock_ib():
    ib = MagicMock()
    ib.positions = MagicMock(return_value=[])
    
    # Mock sufficient balance by default
    val_liq = MagicMock()
    val_liq.tag = 'NetLiquidation'
    val_liq.value = '100000'
    
    val_margin = MagicMock()
    val_margin.tag = 'FullInitMarginReq'
    val_margin.value = '10000'
    
    ib.accountSummaryAsync = AsyncMock(return_value=[val_liq, val_margin])
    return ib

@pytest.fixture
def config():
    conf = TradingConfig()
    conf.max_contracts_limit = 5
    conf.margin_limit_pct = 0.8
    return conf

@pytest.mark.asyncio
async def test_position_manager_init(mock_ib, config):
    pm = PositionManager(mock_ib, config, "ES")
    assert pm.max_contracts == 5
    assert pm.margin_limit_pct == 0.8

@pytest.mark.asyncio
async def test_can_place_order_basic(mock_ib, config):
    pm = PositionManager(mock_ib, config, "ES")
    
    # Mock 0 positions
    mock_ib.positions.return_value = []
    
    # Request 1 contract
    decision = await pm.can_place_order(1)
    assert decision.allowed_contracts == 1
    assert decision.reason == "Approved"

@pytest.mark.asyncio
async def test_can_place_order_cap_breach(mock_ib, config):
    pm = PositionManager(mock_ib, config, "ES")
    
    # Mock 3 existing long positions
    pos = MagicMock()
    pos.contract.symbol = "ES"
    pos.position = 3
    mock_ib.positions.return_value = [pos]
    
    # Request 3 more (Total 6 > 5)
    decision = await pm.can_place_order(3)
    
    # Should be reduced to 2 (5 - 3)
    assert decision.allowed_contracts == 2
    assert "Reduced to 2" in decision.reason

@pytest.mark.asyncio
async def test_can_place_order_short_cap(mock_ib, config):
    pm = PositionManager(mock_ib, config, "ES")
    
    # Mock 3 existing short positions (-3)
    pos = MagicMock()
    pos.contract.symbol = "ES"
    pos.position = -3
    mock_ib.positions.return_value = [pos]
    
    # Request 3 more short (-3) -> Total -6
    decision = await pm.can_place_order(-3)
    
    # Should be reduced to -2 (-5 - (-3))
    assert decision.allowed_contracts == -2
    assert "Reduced to -2" in decision.reason

@pytest.mark.asyncio
async def test_can_place_order_reduce_position(mock_ib, config):
    pm = PositionManager(mock_ib, config, "ES")
    
    # Mock 6 existing long positions (violation state)
    pos = MagicMock()
    pos.contract.symbol = "ES"
    pos.position = 6
    mock_ib.positions.return_value = [pos]
    
    # Request sell 1 (-1) -> Total 5
    # Should be allowed because it reduces risk
    decision = await pm.can_place_order(-1)
    assert decision.allowed_contracts == -1
    assert decision.reason == "Approved"

@pytest.mark.asyncio
async def test_margin_check(mock_ib, config):
    pm = PositionManager(mock_ib, config, "ES")
    
    # Mock 0 positions
    mock_ib.positions.return_value = []
    
    # Mock Margin: NetLiq 100k, InitMargin 75k (75%)
    # Requesting 1 contract (approx 15k margin) -> 90k (90%) > 80% limit
    
    val_liq = MagicMock()
    val_liq.tag = 'NetLiquidation'
    val_liq.value = '100000'
    
    val_margin = MagicMock()
    val_margin.tag = 'FullInitMarginReq'
    val_margin.value = '75000'
    
    mock_ib.accountSummaryAsync.return_value = [val_liq, val_margin]
    
    # Request 1 contract
    decision = await pm.can_place_order(1)
    
    # Should be rejected due to margin
    assert decision.allowed_contracts == 0
    assert "Margin limit" in decision.reason
