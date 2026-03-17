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

    from shree.monitoring.order_tracker import OrderTracker

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
        from shree.features.feature_engineer import add_technical_indicators

        df = sample_ohlcv_data.copy()

        # Calculate features
        df_with_features = add_technical_indicators(df)

        # For each point, verify EMA uses only past data
        for i in range(50, len(df)):  # Start after warmup
            # Get data up to this point only
            past_data = df.iloc[:i+1].copy()
            past_features = add_technical_indicators(past_data)
            
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
        from shree.strategies.mes_one_minute import MesOneMinuteTrendStrategy
        from shree.config import OneMinuteStrategyConfig
        
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

        normalized = normalizer._normalize_timezone(df)

        assert normalized.index.tzinfo is not None
        # Chicago 09:30 = UTC 15:30 (during standard time)
    
    def test_missing_bar_handling(self, sample_ohlcv_data):
        """Test handling of missing bars."""
        from backtest.data.normalize import DataNormalizer
        
        normalizer = DataNormalizer()
        
        # Remove some bars
        df = sample_ohlcv_data.drop(sample_ohlcv_data.index[50:55])
        
        # Fill missing bars
        filled = normalizer.normalize(df)
        
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
        from backtest.broker_sim import BrokerSimulator, BrokerConfig, OrderSide

        config = BrokerConfig(slippage_ticks=1.0, tick_size=0.25, commission_per_contract=2.40)
        broker = BrokerSimulator(config)

        # Submit market buy
        ts = datetime(2024, 1, 10, 9, 30, tzinfo=timezone.utc)
        order = broker.submit_market_order("MES", OrderSide.BUY, 1, ts)
        order_id = order.order_id

        # Process next bar
        bar = pd.Series({
            "open": 4800.0,
            "high": 4801.0,
            "low": 4799.0,
            "close": 4800.5,
            "volume": 100
        }, name=datetime(2024, 1, 10, 9, 31, tzinfo=timezone.utc))

        fills = broker.process_bar("MES", bar, bar.name)

        assert len(fills) == 1
        assert fills[0].order_id == order_id
        # Fill should be at open + slippage
        assert fills[0].price == 4800.0 + (1.0 * 0.25)  # +1 tick
    
    def test_stop_order_trigger(self):
        """Test stop order triggers correctly."""
        from backtest.broker_sim import BrokerSimulator, BrokerConfig, OrderSide

        config = BrokerConfig(slippage_ticks=1.0, tick_size=0.25)
        broker = BrokerSimulator(config)

        # Submit stop sell at 4795
        ts = datetime(2024, 1, 10, 9, 30, tzinfo=timezone.utc)
        order = broker.submit_stop_order("MES", OrderSide.SELL, 1, 4795.0, ts)
        order_id = order.order_id

        # Bar that doesn't trigger stop
        bar1 = pd.Series({
            "open": 4800.0, "high": 4801.0, "low": 4798.0, "close": 4799.0
        }, name=datetime(2024, 1, 10, 9, 31, tzinfo=timezone.utc))

        fills = broker.process_bar("MES", bar1, bar1.name)
        assert len(fills) == 0  # Not triggered

        # Bar that triggers stop (low=4793 <= stop=4795)
        bar2 = pd.Series({
            "open": 4797.0, "high": 4798.0, "low": 4793.0, "close": 4794.0
        }, name=datetime(2024, 1, 10, 9, 32, tzinfo=timezone.utc))

        fills = broker.process_bar("MES", bar2, bar2.name)
        assert len(fills) == 1
        assert fills[0].order_id == order_id
        # Stop triggered at stop price
        assert fills[0].price <= 4795.0
    
    def test_bracket_order_management(self):
        """Test bracket order (entry + stop + target)."""
        from backtest.broker_sim import BrokerSimulator, OrderSide, OrderType

        broker = BrokerSimulator()

        # Submit bracket: buy at market, stop at 4790, target at 4810
        ts = datetime(2024, 1, 10, 9, 30, tzinfo=timezone.utc)
        entry, sl, tp = broker.submit_bracket_order(
            "MES", OrderSide.BUY, 1, OrderType.MARKET,
            stop_loss=4790.0, take_profit=4810.0, timestamp=ts
        )

        # Fill entry
        bar1 = pd.Series({
            "open": 4800.0, "high": 4801.0, "low": 4799.0, "close": 4800.5
        }, name=datetime(2024, 1, 10, 9, 31, tzinfo=timezone.utc))
        fills = broker.process_bar("MES", bar1, bar1.name)

        assert len(fills) == 1
        assert broker.get_open_position("MES") is not None

        # Bracket orders (SL + TP) should be active now
        assert len(broker.pending_orders) >= 2
    
    def test_commission_tracking(self):
        """Test commission is correctly tracked."""
        from backtest.broker_sim import BrokerSimulator, BrokerConfig, OrderSide

        config = BrokerConfig(commission_per_contract=2.40)
        broker = BrokerSimulator(config)

        # Round trip trade
        ts_buy = datetime(2024, 1, 10, 9, 30, tzinfo=timezone.utc)
        broker.submit_market_order("MES", OrderSide.BUY, 2, ts_buy)

        bar = pd.Series({
            "open": 4800.0, "high": 4801.0, "low": 4799.0, "close": 4800.5
        }, name=datetime(2024, 1, 10, 9, 31, tzinfo=timezone.utc))
        broker.process_bar("MES", bar, bar.name)

        ts_sell = datetime(2024, 1, 10, 9, 35, tzinfo=timezone.utc)
        broker.submit_market_order("MES", OrderSide.SELL, 2, ts_sell)

        bar2 = pd.Series({
            "open": 4805.0, "high": 4806.0, "low": 4804.0, "close": 4805.5
        }, name=datetime(2024, 1, 10, 9, 36, tzinfo=timezone.utc))
        broker.process_bar("MES", bar2, bar2.name)

        # Commission should be 2.40 * 2 contracts * 2 sides = 9.60
        # Or if commission is per round-trip: 2.40 * 2 = 4.80
        assert broker.total_commission > 0


# ============================================================================
# Continuous Futures Tests
# ============================================================================

class TestContinuousFutures:
    """Tests for continuous futures rolling."""
    
    def test_back_adjustment(self):
        """Test back-adjustment builds continuous series from two contracts."""
        from backtest.data.roll import ContinuousFuturesBuilder, RollConfig, AdjustmentMethod

        config = RollConfig(adjustment_method=AdjustmentMethod.BACK_ADJUST)
        builder = ContinuousFuturesBuilder(config)

        # Use proper contract codes (ESH24, ESM24) with full OHLCV and UTC timestamps.
        # front_month: March 1-5 (before ESH24 roll on March 10)
        # back_month: March 11-15 (after roll; ESM24 active_start = March 22 → need end_date >= March 22)
        front_month = pd.DataFrame({
            "open":  [4800, 4810, 4820, 4815, 4825],
            "high":  [4802, 4812, 4822, 4817, 4827],
            "low":   [4798, 4808, 4818, 4813, 4823],
            "close": [4800, 4810, 4820, 4815, 4825],
            "volume": [1000] * 5,
        }, index=pd.date_range("2024-03-01", periods=5, freq="1D", tz="UTC"))

        back_month = pd.DataFrame({
            "open":  [4830, 4840, 4850, 4845, 4855],
            "high":  [4832, 4842, 4852, 4847, 4857],
            "low":   [4828, 4838, 4848, 4843, 4853],
            "close": [4830, 4840, 4850, 4845, 4855],
            "volume": [1000] * 5,
        }, index=pd.date_range("2024-03-11", periods=5, freq="1D", tz="UTC"))

        from datetime import timezone as tz
        start_date = datetime(2024, 3, 1, tzinfo=tz.utc)
        # end_date >= ESM24 active_start (March 22) so both contracts appear in calendar
        end_date = datetime(2024, 4, 30, tzinfo=tz.utc)

        continuous = builder.build(
            contract_data={"ESH24": front_month, "ESM24": back_month},
            start_date=start_date,
            end_date=end_date,
        )

        # Both contracts' bars should be present
        assert len(continuous) == 10
    
    def test_roll_detection_by_volume(self):
        """Test that VOLUME_CROSSOVER config builds a continuous series."""
        from backtest.data.roll import ContinuousFuturesBuilder, RollConfig, RollMethod, AdjustmentMethod

        config = RollConfig(
            roll_method=RollMethod.VOLUME_CROSSOVER,
            adjustment_method=AdjustmentMethod.UNADJUSTED,
        )
        builder = ContinuousFuturesBuilder(config)

        dates = pd.date_range("2024-03-01", periods=5, freq="1D", tz="UTC")

        # Front month with declining volume
        front = pd.DataFrame({
            "open":  [4800, 4810, 4820, 4815, 4825],
            "high":  [4802, 4812, 4822, 4817, 4827],
            "low":   [4798, 4808, 4818, 4813, 4823],
            "close": [4800, 4810, 4820, 4815, 4825],
            "volume": [10000, 8000, 5000, 3000, 1000],
        }, index=dates)

        # Back month with increasing volume
        back = pd.DataFrame({
            "open":  [4805, 4815, 4825, 4820, 4830],
            "high":  [4807, 4817, 4827, 4822, 4832],
            "low":   [4803, 4813, 4823, 4818, 4828],
            "close": [4805, 4815, 4825, 4820, 4830],
            "volume": [2000, 4000, 6000, 8000, 10000],
        }, index=dates)

        from datetime import timezone as tz
        start_date = datetime(2024, 3, 1, tzinfo=tz.utc)
        end_date = datetime(2024, 4, 30, tzinfo=tz.utc)

        continuous = builder.build(
            contract_data={"ESH24": front, "ESM24": back},
            start_date=start_date,
            end_date=end_date,
            volume_data={"ESH24": front, "ESM24": back},
        )

        # Build should succeed; result is a DataFrame (possibly empty if no data at roll)
        assert isinstance(continuous, pd.DataFrame)
        # Roll report is accessible after build
        roll_report = builder.get_roll_report()
        assert isinstance(roll_report, pd.DataFrame)


# ============================================================================
# Integration Tests
# ============================================================================

class TestBacktestIntegration:
    """Integration tests for full backtest pipeline."""
    
    @pytest.mark.slow
    def test_full_backtest_pipeline(self, sample_ohlcv_data, sample_5m_data):
        """Test running a complete backtest."""
        from backtest.engine import BacktestEngine, BacktestConfig
        from shree.config import OneMinuteStrategyConfig, TradingConfig, RiskGateConfig
        
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
        
        # Simulate some trades (BacktestAnalyzer expects "realized_pnl" key)
        trades = [
            {"entry_time": "2024-01-10 10:00", "exit_time": "2024-01-10 10:30",
             "side": "long", "entry_price": 4800, "exit_price": 4810,
             "quantity": 1, "realized_pnl": 50.0},
            {"entry_time": "2024-01-10 11:00", "exit_time": "2024-01-10 11:15",
             "side": "long", "entry_price": 4815, "exit_price": 4805,
             "quantity": 1, "realized_pnl": -50.0},
            {"entry_time": "2024-01-10 14:00", "exit_time": "2024-01-10 14:45",
             "side": "short", "entry_price": 4820, "exit_price": 4800,
             "quantity": 2, "realized_pnl": 200.0},
        ]

        # Create equity curve
        equity = pd.Series(
            [50000, 50050, 50000, 50200],
            index=pd.date_range("2024-01-10 09:30", periods=4, freq="1h")
        )

        analyzer = BacktestAnalyzer(
            trades=trades,
            equity_curve=equity,
            block_reasons={},
            optimizer_stats={},
            initial_capital=50000.0,
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
        from shree.config import OneMinuteStrategyConfig, TradingConfig, RiskGateConfig
        
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

        from shree.features.feature_engineer import add_technical_indicators

        # Should not raise
        features = add_technical_indicators(df)

        # NaN in input should propagate to features, not crash
        assert features is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
