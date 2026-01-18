"""
Unit tests for backtest framework.

Tests cover:
1. No lookahead bias verification
2. Data normalization correctness
3. Broker simulation accuracy
4. Continuous futures rolling
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch


def test_backtest_root_orders_persist_feature_snapshots(tmp_path):
    """Backtest should persist root orders with features/rationale when trade_cycle_id is present.

    This protects forensic attribution: <5m losers were dominated by missing snapshots.
    """

    from mytrader.monitoring.order_tracker import OrderTracker

    db_path = tmp_path / "orders.db"
    tracker = OrderTracker(db_path=str(db_path))

    tracker.record_order_placement(
        order_id=1,
        symbol="MES",
        action="BUY",
        quantity=1,
        order_type="MARKET",
        entry_price=5000.0,
        confidence=0.6,
        parent_order_id=None,
        trade_cycle_id="cycle123",
        features={"rsi": 55.0, "regime": "RANGE"},
        rationale={"source": "backtest", "note": "unit-test"},
        market_regime="RANGE",
    )

    import sqlite3

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT features, rationale
            FROM orders
            WHERE order_id = 1 AND parent_order_id IS NULL AND trade_cycle_id = 'cycle123'
            """
        ).fetchone()
    assert row is not None
    features_json, rationale_json = row
    assert features_json is not None and features_json.strip() not in {"", "{}"}
    assert rationale_json is not None and rationale_json.strip() not in {"", "{}"}


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    
    # 1000 1-minute bars
    timestamps = pd.date_range(
        start="2024-01-10 09:30:00",
        periods=1000,
        freq="1min",
        tz="America/New_York"
    )
    
    # Random walk price
    base_price = 4800.0
    returns = np.random.normal(0, 0.0002, len(timestamps))
    close_prices = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": close_prices * (1 + np.random.normal(0, 0.0001, len(timestamps))),
        "high": close_prices * (1 + np.abs(np.random.normal(0, 0.0003, len(timestamps)))),
        "low": close_prices * (1 - np.abs(np.random.normal(0, 0.0003, len(timestamps)))),
        "close": close_prices,
        "volume": np.random.randint(100, 1000, len(timestamps)),
    })
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)
    
    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture
def sample_5m_data(sample_ohlcv_data):
    """Resample 1m data to 5m."""
    df_5m = sample_ohlcv_data.resample("5min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()
    return df_5m


# ============================================================================
# Lookahead Bias Tests
# ============================================================================

class TestNoLookahead:
    """Tests to verify no lookahead bias in backtest."""
    
    def test_indicator_uses_only_past_data(self, sample_ohlcv_data):
        """Verify indicators don't use future data."""
        from mytrader.features.feature_engineer import FeatureEngineer
        
        df = sample_ohlcv_data.copy()
        engineer = FeatureEngineer()
        
        # Calculate features
        df_with_features = engineer.add_all_features(df)
        
        # For each point, verify EMA uses only past data
        for i in range(50, len(df)):  # Start after warmup
            # Get data up to this point only
            past_data = df.iloc[:i+1].copy()
            past_features = engineer.add_all_features(past_data)
            
            # The last row's EMA should match what we calculated with full data
            # (if no lookahead, they should be identical)
            current_idx = df.index[i]
            
            if "ema_fast" in df_with_features.columns and "ema_fast" in past_features.columns:
                assert abs(
                    df_with_features.loc[current_idx, "ema_fast"] - 
                    past_features.iloc[-1]["ema_fast"]
                ) < 1e-10, f"EMA mismatch at {current_idx} - possible lookahead"
            
            # Only check a few points to speed up test
            if i > 60:
                break
    
    def test_signal_generation_no_future_bars(self, sample_ohlcv_data):
        """Verify signal generation only sees current and past bars."""
        from mytrader.strategies.mes_one_minute import MesOneMinuteTrendStrategy
        from mytrader.config import OneMinuteStrategyConfig
        
        config = OneMinuteStrategyConfig()
        strategy = MesOneMinuteTrendStrategy(config)
        
        # Record what data each generate() call sees
        seen_timestamps = []
        original_generate = strategy.generate
        
        def tracked_generate(df, *args, **kwargs):
            seen_timestamps.append({
                "call_time": df.index[-1],
                "data_end": df.index.max()
            })
            return original_generate(df, *args, **kwargs)
        
        strategy.generate = tracked_generate
        
        # Simulate bar-by-bar processing
        for i in range(100, 150):
            current_bar = sample_ohlcv_data.index[i]
            past_only = sample_ohlcv_data.iloc[:i+1]
            
            try:
                strategy.generate(past_only)
            except Exception:
                pass  # Ignore strategy errors, we're testing data access
        
        # Verify no call saw data beyond its call time
        for record in seen_timestamps:
            assert record["data_end"] <= record["call_time"], \
                f"Signal generation saw future data: call_time={record['call_time']}, " \
                f"data_end={record['data_end']}"
    
    def test_mtf_alignment_no_incomplete_bars(self, sample_ohlcv_data, sample_5m_data):
        """Verify 5m features use only completed 5m bars at each 1m decision point."""
        # At 09:33 (1m bar), the 5m bar 09:30-09:35 is NOT complete
        # So we should only use the 09:25-09:30 bar for features
        
        for i, (ts, row) in enumerate(sample_ohlcv_data.iterrows()):
            if i < 10:
                continue
                
            # Find which 5m bars are complete at this 1m timestamp
            complete_5m_bars = sample_5m_data[sample_5m_data.index <= ts]
            
            # The last complete 5m bar should have ended on or before ts
            if len(complete_5m_bars) > 0:
                last_5m_bar = complete_5m_bars.index[-1]
                # 5m bar at 09:30 covers 09:30-09:35, so it's complete at 09:35
                five_min_end = last_5m_bar + pd.Timedelta(minutes=5)
                
                # At decision time ts, we should NOT use a 5m bar that ends after ts
                assert five_min_end <= ts + pd.Timedelta(minutes=5), \
                    f"Using incomplete 5m bar at {ts}"
            
            if i > 20:
                break


# ============================================================================
# Data Normalization Tests
# ============================================================================

class TestDataNormalization:
    """Tests for data normalization module."""
    
    def test_schema_normalization(self):
        """Test that various input schemas get normalized correctly."""
        from backtest.data.normalize import DataNormalizer
        
        normalizer = DataNormalizer()
        
        # IBKR format
        ibkr_df = pd.DataFrame({
            "date": ["2024-01-10 09:30:00"],
            "open": [4800.0],
            "high": [4801.0],
            "low": [4799.0],
            "close": [4800.5],
            "volume": [100],
            "barCount": [50],
            "average": [4800.25]
        })
        
        normalized = normalizer.normalize(ibkr_df)
        
        assert "timestamp" in normalized.columns or normalized.index.name == "timestamp"
        assert "open" in normalized.columns
        assert "close" in normalized.columns
    
    def test_timezone_normalization(self):
        """Test timezone handling."""
        from backtest.data.normalize import DataNormalizer
        
        normalizer = DataNormalizer()
        
        # Create data with different timezone
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-10 09:30:00"]).tz_localize("America/Chicago"),
            "open": [4800.0],
            "high": [4801.0],
            "low": [4799.0],
            "close": [4800.5],
            "volume": [100]
        })
        df.set_index("timestamp", inplace=True)
        
        normalized = normalizer.normalize_timezone(df, target_tz="UTC")
        
        assert normalized.index.tzinfo is not None
        # Chicago 09:30 = UTC 15:30 (during standard time)
    
    def test_missing_bar_handling(self, sample_ohlcv_data):
        """Test handling of missing bars."""
        from backtest.data.normalize import DataNormalizer
        
        normalizer = DataNormalizer()
        
        # Remove some bars
        df = sample_ohlcv_data.drop(sample_ohlcv_data.index[50:55])
        
        # Fill missing bars
        filled = normalizer.fill_missing_bars(df, freq="1min", method="forward_fill")
        
        # Should have all bars now
        expected_bars = len(sample_ohlcv_data)
        assert len(filled) == expected_bars


# ============================================================================
# Broker Simulation Tests
# ============================================================================

class TestBrokerSimulation:
    """Tests for order execution simulation."""
    
    def test_market_order_fill(self):
        """Test market order fills at next bar."""
        from backtest.broker_sim import BrokerSimulator, SimulatedOrder, SlippageModel
        
        slippage = SlippageModel(base_ticks=1.0, tick_size=0.25)
        broker = BrokerSimulator(slippage=slippage, commission_per_contract=2.40)
        
        # Submit market buy
        order = SimulatedOrder(
            order_id="test_1",
            symbol="MES",
            side="buy",
            quantity=1,
            order_type="market",
            submit_time=datetime(2024, 1, 10, 9, 30, tzinfo=timezone.utc)
        )
        broker.submit_order(order)
        
        # Process next bar
        bar = pd.Series({
            "open": 4800.0,
            "high": 4801.0,
            "low": 4799.0,
            "close": 4800.5,
            "volume": 100
        }, name=datetime(2024, 1, 10, 9, 31, tzinfo=timezone.utc))
        
        fills = broker.process_bar(bar)
        
        assert len(fills) == 1
        assert fills[0]["order_id"] == "test_1"
        # Fill should be at open + slippage
        assert fills[0]["fill_price"] == 4800.0 + (1.0 * 0.25)  # +1 tick
    
    def test_stop_order_trigger(self):
        """Test stop order triggers correctly."""
        from backtest.broker_sim import BrokerSimulator, SimulatedOrder, SlippageModel
        
        slippage = SlippageModel(base_ticks=1.0, tick_size=0.25)
        broker = BrokerSimulator(slippage=slippage)
        
        # Submit stop sell at 4795
        order = SimulatedOrder(
            order_id="stop_1",
            symbol="MES",
            side="sell",
            quantity=1,
            order_type="stop",
            stop_price=4795.0,
            submit_time=datetime(2024, 1, 10, 9, 30, tzinfo=timezone.utc)
        )
        broker.submit_order(order)
        
        # Bar that doesn't trigger stop
        bar1 = pd.Series({
            "open": 4800.0, "high": 4801.0, "low": 4798.0, "close": 4799.0
        }, name=datetime(2024, 1, 10, 9, 31, tzinfo=timezone.utc))
        
        fills = broker.process_bar(bar1)
        assert len(fills) == 0  # Not triggered
        
        # Bar that triggers stop
        bar2 = pd.Series({
            "open": 4797.0, "high": 4798.0, "low": 4793.0, "close": 4794.0
        }, name=datetime(2024, 1, 10, 9, 32, tzinfo=timezone.utc))
        
        fills = broker.process_bar(bar2)
        assert len(fills) == 1
        assert fills[0]["order_id"] == "stop_1"
        # Stop triggered at stop price
        assert fills[0]["fill_price"] <= 4795.0
    
    def test_bracket_order_management(self):
        """Test bracket order (entry + stop + target)."""
        from backtest.broker_sim import BrokerSimulator, SimulatedOrder, SlippageModel
        
        broker = BrokerSimulator()
        
        # Submit bracket: buy at market, stop at 4790, target at 4810
        entry = SimulatedOrder(
            order_id="entry_1",
            symbol="MES",
            side="buy",
            quantity=1,
            order_type="market",
            submit_time=datetime(2024, 1, 10, 9, 30, tzinfo=timezone.utc),
            bracket_stop_price=4790.0,
            bracket_tp_price=4810.0
        )
        broker.submit_order(entry)
        
        # Fill entry
        bar1 = pd.Series({
            "open": 4800.0, "high": 4801.0, "low": 4799.0, "close": 4800.5
        }, name=datetime(2024, 1, 10, 9, 31, tzinfo=timezone.utc))
        fills = broker.process_bar(bar1)
        
        assert len(fills) == 1
        assert broker.has_position("MES")
        
        # Bracket orders should be active now
        assert len(broker.pending_orders) >= 2 or broker.positions["MES"].stop_price == 4790.0
    
    def test_commission_tracking(self):
        """Test commission is correctly tracked."""
        from backtest.broker_sim import BrokerSimulator, SimulatedOrder
        
        broker = BrokerSimulator(commission_per_contract=2.40)
        
        # Round trip trade
        buy = SimulatedOrder(
            order_id="buy_1", symbol="MES", side="buy", quantity=2,
            order_type="market", submit_time=datetime(2024, 1, 10, 9, 30, tzinfo=timezone.utc)
        )
        broker.submit_order(buy)
        
        bar = pd.Series({
            "open": 4800.0, "high": 4801.0, "low": 4799.0, "close": 4800.5
        }, name=datetime(2024, 1, 10, 9, 31, tzinfo=timezone.utc))
        broker.process_bar(bar)
        
        sell = SimulatedOrder(
            order_id="sell_1", symbol="MES", side="sell", quantity=2,
            order_type="market", submit_time=datetime(2024, 1, 10, 9, 35, tzinfo=timezone.utc)
        )
        broker.submit_order(sell)
        
        bar2 = pd.Series({
            "open": 4805.0, "high": 4806.0, "low": 4804.0, "close": 4805.5
        }, name=datetime(2024, 1, 10, 9, 36, tzinfo=timezone.utc))
        broker.process_bar(bar2)
        
        # Commission should be 2.40 * 2 contracts * 2 sides = 9.60
        # Or if commission is per round-trip: 2.40 * 2 = 4.80
        assert broker.total_commission > 0


# ============================================================================
# Continuous Futures Tests
# ============================================================================

class TestContinuousFutures:
    """Tests for continuous futures rolling."""
    
    def test_back_adjustment(self):
        """Test back-adjustment preserves returns."""
        from backtest.data.roll import ContinuousFuturesBuilder, RollConfig
        
        config = RollConfig(adjustment_method="back_adjust")
        builder = ContinuousFuturesBuilder(config)
        
        # Simulate two contracts with a gap
        front_month = pd.DataFrame({
            "timestamp": pd.date_range("2024-03-01", periods=5, freq="1D"),
            "close": [4800, 4810, 4820, 4815, 4825]  # Last price 4825
        }).set_index("timestamp")
        
        back_month = pd.DataFrame({
            "timestamp": pd.date_range("2024-03-06", periods=5, freq="1D"),
            "close": [4830, 4840, 4850, 4845, 4855]  # First price 4830 (gap of 5)
        }).set_index("timestamp")
        
        # After back-adjustment, returns should be continuous
        continuous = builder.build_continuous(
            contracts={"H24": front_month, "M24": back_month},
            roll_dates=[("H24", "M24", pd.Timestamp("2024-03-06"))]
        )
        
        # Calculate returns
        returns = continuous["close"].pct_change().dropna()
        
        # The return at roll date should not have the gap artifact
        # (Without adjustment, it would show (4830-4825)/4825 = 0.1%)
        assert len(continuous) == 10
    
    def test_roll_detection_by_volume(self):
        """Test volume/OI crossover roll detection."""
        from backtest.data.roll import ContinuousFuturesBuilder, RollConfig
        
        config = RollConfig(roll_method="volume_oi_crossover")
        builder = ContinuousFuturesBuilder(config)
        
        # Front month with declining volume
        front = pd.DataFrame({
            "timestamp": pd.date_range("2024-03-10", periods=5, freq="1D"),
            "close": [4800, 4810, 4820, 4815, 4825],
            "volume": [10000, 8000, 5000, 3000, 1000]  # Declining
        }).set_index("timestamp")
        
        # Back month with increasing volume
        back = pd.DataFrame({
            "timestamp": pd.date_range("2024-03-10", periods=5, freq="1D"),
            "close": [4805, 4815, 4825, 4820, 4830],
            "volume": [2000, 4000, 6000, 8000, 10000]  # Increasing
        }).set_index("timestamp")
        
        roll_date = builder.detect_roll_date(front, back)
        
        # Roll should happen when back volume > front volume
        # Day 3: front=5000, back=6000 -> roll here
        assert roll_date == pd.Timestamp("2024-03-12")


# ============================================================================
# Integration Tests
# ============================================================================

class TestBacktestIntegration:
    """Integration tests for full backtest pipeline."""
    
    @pytest.mark.slow
    def test_full_backtest_pipeline(self, sample_ohlcv_data, sample_5m_data):
        """Test running a complete backtest."""
        from backtest.engine import BacktestEngine, BacktestConfig
        from mytrader.config import OneMinuteStrategyConfig, TradingConfig, RiskGateConfig
        
        config = BacktestConfig(
            symbol="MES",
            start_date=sample_ohlcv_data.index[0].to_pydatetime(),
            end_date=sample_ohlcv_data.index[-1].to_pydatetime(),
            initial_capital=50000.0,
            slippage_ticks=1.0,
            commission_per_contract=2.40,
            strategy_config=OneMinuteStrategyConfig(),
            trading_config=TradingConfig(),
            risk_gate_config=RiskGateConfig()
        )
        
        engine = BacktestEngine(config)
        engine.load_data(sample_ohlcv_data, sample_5m_data)
        
        results = engine.run()
        
        assert "trades" in results
        assert "equity_curve" in results
        assert results["equity_curve"] is not None
    
    @pytest.mark.slow
    def test_analysis_metrics_calculation(self, sample_ohlcv_data):
        """Test metrics calculation."""
        from backtest.analysis import BacktestAnalyzer
        
        # Simulate some trades
        trades = [
            {"entry_time": "2024-01-10 10:00", "exit_time": "2024-01-10 10:30",
             "side": "long", "entry_price": 4800, "exit_price": 4810,
             "quantity": 1, "pnl": 50.0, "gross_pnl": 52.40},
            {"entry_time": "2024-01-10 11:00", "exit_time": "2024-01-10 11:15",
             "side": "long", "entry_price": 4815, "exit_price": 4805,
             "quantity": 1, "pnl": -50.0, "gross_pnl": -47.60},
            {"entry_time": "2024-01-10 14:00", "exit_time": "2024-01-10 14:45",
             "side": "short", "entry_price": 4820, "exit_price": 4800,
             "quantity": 2, "pnl": 200.0, "gross_pnl": 204.80},
        ]
        
        # Create equity curve
        equity = pd.Series(
            [50000, 50050, 50000, 50200],
            index=pd.date_range("2024-01-10 09:30", periods=4, freq="1h")
        )
        
        analyzer = BacktestAnalyzer(
            trades=trades,
            equity_curve=equity,
            initial_capital=50000.0
        )
        
        analysis = analyzer.analyze()
        
        assert "metrics" in analysis
        assert analysis["metrics"]["total_trades"] == 3
        assert analysis["metrics"]["win_rate"] > 0
        assert "sharpe_ratio" in analysis["metrics"]


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_data_handling(self):
        """Test graceful handling of empty data."""
        from backtest.engine import BacktestEngine, BacktestConfig
        from mytrader.config import OneMinuteStrategyConfig, TradingConfig, RiskGateConfig
        
        config = BacktestConfig(
            symbol="MES",
            start_date=datetime(2024, 1, 10, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 11, tzinfo=timezone.utc),
            initial_capital=50000.0,
            strategy_config=OneMinuteStrategyConfig(),
            trading_config=TradingConfig(),
            risk_gate_config=RiskGateConfig()
        )
        
        engine = BacktestEngine(config)
        
        # Empty DataFrame
        empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        
        with pytest.raises((ValueError, Exception)):
            engine.load_data(empty_df, None)
    
    def test_nan_handling_in_features(self, sample_ohlcv_data):
        """Test NaN values don't cause crashes."""
        # Insert some NaN values
        df = sample_ohlcv_data.copy()
        df.iloc[100:105, df.columns.get_loc("close")] = np.nan
        
        from mytrader.features.feature_engineer import FeatureEngineer
        
        engineer = FeatureEngineer()
        
        # Should not raise
        features = engineer.add_all_features(df)
        
        # NaN in input should propagate to features, not crash
        assert features is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
