import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone
import json
import os

from mytrader.llm.rag_storage import RAGStorage, TradeRecord
from mytrader.execution.live_trading_manager import LiveTradingManager, TradingStatus
from mytrader.config import Settings

# --- RAGStorage Tests ---

@pytest.fixture
def temp_db_path(tmp_path):
    return str(tmp_path / "test_rag.db")

@pytest.fixture
def rag_storage(temp_db_path):
    return RAGStorage(db_path=temp_db_path)

def test_rag_storage_save_and_retrieve(rag_storage):
    record = TradeRecord(
        uuid="test-uuid-1",
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        contract_month="ES",
        entry_price=4000.0,
        entry_qty=1,
        exit_price=4010.0,
        exit_qty=1,
        pnl=500.0,
        fees=2.5,
        hold_seconds=60,
        decision_features={"feature": 1},
        decision_rationale={"reason": "test"}
    )
    
    buckets = {
        "volatility": "LOW",
        "time_of_day": "MORNING",
        "signal_type": "BUY"
    }
    
    rag_storage.save_trade(record, buckets)
    
    # Retrieve
    retrieved = rag_storage.retrieve_similar_trades(buckets, limit=1)
    assert len(retrieved) == 1
    assert retrieved[0]['uuid'] == "test-uuid-1"
    assert retrieved[0]['pnl'] == 500.0
    assert retrieved[0]['decision_features'] == {"feature": 1}

def test_rag_storage_bucket_stats(rag_storage):
    # Save a winning trade
    win_record = TradeRecord(
        uuid="win-1",
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        contract_month="ES",
        entry_price=4000.0,
        entry_qty=1,
        exit_price=4010.0,
        exit_qty=1,
        pnl=500.0,
        fees=2.5,
        hold_seconds=60,
        decision_features={},
        decision_rationale={}
    )
    rag_storage.save_trade(win_record, {"volatility": "LOW"})
    
    # Save a losing trade
    loss_record = TradeRecord(
        uuid="loss-1",
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        contract_month="ES",
        entry_price=4000.0,
        entry_qty=1,
        exit_price=3990.0,
        exit_qty=1,
        pnl=-500.0,
        fees=2.5,
        hold_seconds=60,
        decision_features={},
        decision_rationale={}
    )
    rag_storage.save_trade(loss_record, {"volatility": "LOW"})
    
    stats = rag_storage.get_bucket_stats({"volatility": "LOW"})
    assert stats['count'] == 2
    assert stats['win_rate'] == 0.5
    assert stats['avg_pnl'] == 0.0

# --- LiveTradingManager Integration Tests ---

@pytest.fixture
def mock_settings():
    settings = MagicMock(spec=Settings)
    settings.trading = MagicMock()
    settings.trading.initial_capital = 100000
    settings.trading.max_position_size = 5
    settings.trading.tick_size = 0.25
    settings.trading.stop_loss_ticks = 10
    settings.trading.take_profit_ticks = 20
    settings.data = MagicMock()
    settings.data.ibkr_symbol = "ES"
    settings.data.ibkr_exchange = "GLOBEX"
    return settings

@pytest.mark.asyncio
async def test_manager_process_trading_cycle_rag_adjustment(mock_settings):
    manager = LiveTradingManager(mock_settings)
    manager.rag_storage = MagicMock()
    manager.engine = MagicMock()
    manager.executor = AsyncMock()
    manager.tracker = MagicMock()
    manager.risk = MagicMock()
    
    # Mock RAG stats to return high win rate
    manager.rag_storage.get_bucket_stats.return_value = {
        "count": 10,
        "win_rate": 0.8,
        "avg_pnl": 100.0
    }
    manager.rag_storage.retrieve_similar_trades.return_value = []
    
    # Mock signal
    signal = MagicMock()
    signal.action = "BUY"
    signal.confidence = 0.5
    signal.metadata = {}
    manager.engine.evaluate.return_value = signal
    
    # Mock price history
    manager.price_history = [
        {'timestamp': datetime.now(timezone.utc), 'open': 100, 'high': 101, 'low': 99, 'close': 100, 'volume': 100}
        for _ in range(60)
    ]
    
    # Mock feature engineering to return valid features
    with patch('mytrader.execution.live_trading_manager.engineer_features') as mock_features:
        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.iloc = [MagicMock()]
        mock_df.iloc[-1].get.side_effect = lambda k, d=None: 0.003 if k == 'volatility_5m' else 0.0
        mock_features.return_value = mock_df
        
        await manager._process_trading_cycle(current_price=100.0)
        
        # Verify RAG stats were called
        manager.rag_storage.get_bucket_stats.assert_called()
        
        # Verify confidence was adjusted (0.5 + 0.1 = 0.6 due to high win rate)
        # Note: signal object is modified in place
        assert signal.confidence == 0.6

@pytest.mark.asyncio
async def test_manager_execution_updates_rag(mock_settings):
    manager = LiveTradingManager(mock_settings)
    manager.rag_storage = MagicMock()
    
    # Setup current trade context
    manager.current_trade_id = "test-trade-id"
    manager.current_trade_entry_time = datetime.now(timezone.utc).isoformat()
    manager.current_trade_entry_price = 4000.0
    manager.current_trade_buckets = {"volatility": "LOW"}
    
    # Mock execution fill
    fill = MagicMock()
    fill.execution.price = 4010.0
    fill.execution.shares = 1
    fill.commissionReport.realizedPNL = 500.0
    fill.commissionReport.commission = 2.5
    
    trade = MagicMock()
    
    # Call handler
    manager._on_execution_details(trade, fill)
    
    # Verify save_trade was called with updated info
    manager.rag_storage.save_trade.assert_called()
    call_args = manager.rag_storage.save_trade.call_args
    record = call_args[0][0]
    
    assert record.uuid == "test-trade-id"
    assert record.pnl == 500.0
    assert record.exit_price == 4010.0
    
    # Verify context was cleared
    assert manager.current_trade_id is None
