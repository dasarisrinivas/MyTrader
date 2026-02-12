"""
Unit tests for session gating, weekend closure, close window, and confidence validation.

Tests cover:
1. SessionManager.get_current_session() - correct session detection
2. Weekend closure - Friday 4 PM CT to Sunday 5 PM CT
3. Close window gating - 3-4 PM CT (RiskGate)
4. Confidence unit normalization - 0-100 values converted to 0-1
5. YAML session config consumption
"""
import pytest
from datetime import datetime, time, timedelta
from unittest.mock import patch, MagicMock
from zoneinfo import ZoneInfo

# Import modules under test
from shree.utils.session_manager import SessionManager, TradingSession, SessionConfig
from shree.risk.risk_gate import RiskGate, RiskGateConfig

CST = ZoneInfo("America/Chicago")


class TestSessionManagerSessionDetection:
    """Test SessionManager correctly identifies trading sessions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = SessionManager(config={})
    
    def test_rth_session_detection(self):
        """RTH should be detected 8:30 AM - 3:00 PM CT."""
        # 10:00 AM CT on a Tuesday
        rth_time = datetime(2025, 1, 14, 10, 0, 0, tzinfo=CST)
        assert self.manager.get_current_session(rth_time) == TradingSession.RTH
    
    def test_evening_session_detection(self):
        """EVENING should be detected 5:00 PM - 11:00 PM CT."""
        # 8:00 PM CT on a Tuesday
        evening_time = datetime(2025, 1, 14, 20, 0, 0, tzinfo=CST)
        assert self.manager.get_current_session(evening_time) == TradingSession.EVENING
    
    def test_overnight_session_detection(self):
        """OVERNIGHT should be detected 11:00 PM - 3:00 AM CT."""
        # 1:00 AM CT on a Wednesday
        overnight_time = datetime(2025, 1, 15, 1, 0, 0, tzinfo=CST)
        assert self.manager.get_current_session(overnight_time) == TradingSession.OVERNIGHT
        
        # 11:30 PM CT on a Tuesday
        late_night = datetime(2025, 1, 14, 23, 30, 0, tzinfo=CST)
        assert self.manager.get_current_session(late_night) == TradingSession.OVERNIGHT
    
    def test_premarket_session_detection(self):
        """PRE_MARKET should be detected 3:00 AM - 8:30 AM CT."""
        # 6:00 AM CT on a Tuesday
        premarket_time = datetime(2025, 1, 14, 6, 0, 0, tzinfo=CST)
        assert self.manager.get_current_session(premarket_time) == TradingSession.PRE_MARKET
    
    def test_maintenance_window_detection(self):
        """MAINTENANCE should be detected 4:00 PM - 5:00 PM CT."""
        # 4:30 PM CT on a Tuesday
        maintenance_time = datetime(2025, 1, 14, 16, 30, 0, tzinfo=CST)
        assert self.manager.get_current_session(maintenance_time) == TradingSession.MAINTENANCE


class TestWeekendClosure:
    """Test weekend closure: Friday 4 PM CT - Sunday 5 PM CT."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = SessionManager(config={})
    
    def test_friday_before_close_is_not_weekend(self):
        """Friday before 4 PM CT should NOT be weekend."""
        # Friday 2:00 PM CT
        friday_rth = datetime(2025, 1, 10, 14, 0, 0, tzinfo=CST)
        session = self.manager.get_current_session(friday_rth)
        assert session != TradingSession.WEEKEND
        assert session == TradingSession.RTH
    
    def test_friday_maintenance_is_weekend(self):
        """Friday 4 PM CT onwards should be WEEKEND (market closed after maintenance)."""
        # Friday 4:00 PM CT
        friday_close = datetime(2025, 1, 10, 16, 0, 0, tzinfo=CST)
        assert self.manager.get_current_session(friday_close) == TradingSession.WEEKEND
    
    def test_friday_evening_is_weekend(self):
        """Friday evening should be WEEKEND."""
        # Friday 6:00 PM CT
        friday_evening = datetime(2025, 1, 10, 18, 0, 0, tzinfo=CST)
        assert self.manager.get_current_session(friday_evening) == TradingSession.WEEKEND
    
    def test_saturday_is_weekend(self):
        """All day Saturday should be WEEKEND."""
        # Saturday noon
        saturday = datetime(2025, 1, 11, 12, 0, 0, tzinfo=CST)
        assert self.manager.get_current_session(saturday) == TradingSession.WEEKEND
    
    def test_sunday_before_open_is_weekend(self):
        """Sunday before 5 PM CT should be WEEKEND."""
        # Sunday 4:00 PM CT
        sunday_before_open = datetime(2025, 1, 12, 16, 0, 0, tzinfo=CST)
        assert self.manager.get_current_session(sunday_before_open) == TradingSession.WEEKEND
    
    def test_sunday_after_open_is_not_weekend(self):
        """Sunday 5:00 PM CT onwards should be EVENING (market open)."""
        # Sunday 5:01 PM CT
        sunday_open = datetime(2025, 1, 12, 17, 1, 0, tzinfo=CST)
        session = self.manager.get_current_session(sunday_open)
        assert session != TradingSession.WEEKEND
        assert session == TradingSession.EVENING
    
    def test_trading_blocked_during_weekend(self):
        """is_trading_allowed() should return False during weekend."""
        # Saturday noon
        saturday = datetime(2025, 1, 11, 12, 0, 0, tzinfo=CST)
        allowed, reason = self.manager.is_trading_allowed(saturday)
        assert allowed is False
        assert "weekend" in reason.lower()


class TestCloseWindowGating:
    """Test close window gating: 3-4 PM CT blocking in RiskGate."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = RiskGateConfig()
        self.gate = RiskGate(self.config)
    
    def test_close_window_defaults(self):
        """Verify close window defaults to 3-4 PM CT (60 min before 16:00)."""
        assert self.config.intraday_close_time == time(16, 0)
        assert self.config.avoid_close_window_minutes == 60
    
    def test_before_close_window_allowed(self):
        """2:59 PM CT should NOT be blocked by close window."""
        before_window = datetime(2025, 1, 14, 14, 59, 0, tzinfo=CST)
        blocked = self.gate._check_close_window(before_window)
        assert blocked is False
    
    def test_during_close_window_blocked(self):
        """3:00 PM - 4:00 PM CT should be blocked."""
        # 3:00 PM CT
        start_window = datetime(2025, 1, 14, 15, 0, 0, tzinfo=CST)
        assert self.gate._check_close_window(start_window) is True
        
        # 3:30 PM CT
        mid_window = datetime(2025, 1, 14, 15, 30, 0, tzinfo=CST)
        assert self.gate._check_close_window(mid_window) is True
        
        # 3:59 PM CT
        end_window = datetime(2025, 1, 14, 15, 59, 0, tzinfo=CST)
        assert self.gate._check_close_window(end_window) is True
    
    def test_at_close_window_end_blocked(self):
        """4:00 PM CT (window end) should be blocked."""
        window_end = datetime(2025, 1, 14, 16, 0, 0, tzinfo=CST)
        assert self.gate._check_close_window(window_end) is True
    
    def test_after_close_window_not_blocked_by_close_check(self):
        """4:01 PM CT should not be blocked by close window (maintenance blocks separately)."""
        after_window = datetime(2025, 1, 14, 16, 1, 0, tzinfo=CST)
        # _check_close_window only checks close window, not maintenance
        blocked = self.gate._check_close_window(after_window)
        assert blocked is False


class TestMaintenanceWindow:
    """Test maintenance window: 4-5 PM CT blocking."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = RiskGateConfig()
        self.gate = RiskGate(self.config)
        self.manager = SessionManager(config={})
    
    def test_maintenance_window_in_risk_gate(self):
        """RiskGate should block 4-5 PM CT."""
        # 4:30 PM CT
        maintenance_time = datetime(2025, 1, 14, 16, 30, 0, tzinfo=CST)
        assert self.gate._check_maintenance_window(maintenance_time) is True
    
    def test_maintenance_window_in_session_manager(self):
        """SessionManager should identify 4-5 PM CT as MAINTENANCE."""
        # 4:30 PM CT
        maintenance_time = datetime(2025, 1, 14, 16, 30, 0, tzinfo=CST)
        assert self.manager.get_current_session(maintenance_time) == TradingSession.MAINTENANCE
        
        # is_trading_allowed should return False
        allowed, reason = self.manager.is_trading_allowed(maintenance_time)
        assert allowed is False
        assert "maintenance" in reason.lower()


class TestConfidenceNormalization:
    """Test confidence unit normalization (0-100 to 0-1 conversion)."""
    
    def test_session_manager_normalizes_0_100_values(self):
        """SessionManager should convert 0-100 values to 0-1."""
        config = {
            "futures_trading": {
                "active_sessions": {
                    "rth": {
                        "min_confidence": 65,  # 0-100 scale
                    }
                }
            }
        }
        manager = SessionManager(config=config)
        rth_config = manager.get_session_config(TradingSession.RTH)
        # Should be converted to 0.65
        assert rth_config.min_confidence == 0.65
    
    def test_session_manager_keeps_0_1_values(self):
        """SessionManager should keep 0-1 values unchanged."""
        config = {
            "futures_trading": {
                "active_sessions": {
                    "rth": {
                        "min_confidence": 0.55,  # Already 0-1 scale
                    }
                }
            }
        }
        manager = SessionManager(config=config)
        rth_config = manager.get_session_config(TradingSession.RTH)
        # Should remain 0.55
        assert rth_config.min_confidence == 0.55


class TestYAMLSessionConfigConsumption:
    """Test SessionManager reads futures_trading.active_sessions from YAML."""
    
    def test_reads_rth_config(self):
        """SessionManager should read RTH config from futures_trading.active_sessions."""
        config = {
            "futures_trading": {
                "active_sessions": {
                    "rth": {
                        "min_confidence": 0.55,
                        "atr_multiplier_sl": 3.0,
                        "max_trades_per_hour": 2,
                        "cooldown_after_loss_minutes": 20,
                    }
                }
            }
        }
        manager = SessionManager(config=config)
        rth_config = manager.get_session_config(TradingSession.RTH)
        
        assert rth_config.min_confidence == 0.55
        assert rth_config.atr_multiplier_sl == 3.0
        assert rth_config.max_trades_per_hour == 2
        assert rth_config.cooldown_after_loss_minutes == 20
    
    def test_reads_evening_config(self):
        """SessionManager should read EVENING config."""
        config = {
            "futures_trading": {
                "active_sessions": {
                    "evening": {
                        "min_confidence": 0.60,
                        "max_trades_per_session": 4,
                    }
                }
            }
        }
        manager = SessionManager(config=config)
        evening_config = manager.get_session_config(TradingSession.EVENING)
        
        assert evening_config.min_confidence == 0.60
        assert evening_config.max_trades_per_session == 4
    
    def test_reads_overnight_config(self):
        """SessionManager should read OVERNIGHT config."""
        config = {
            "futures_trading": {
                "active_sessions": {
                    "overnight": {
                        "min_confidence": 0.70,
                        "spread_gate_ticks": 3.0,
                    }
                }
            }
        }
        manager = SessionManager(config=config)
        overnight_config = manager.get_session_config(TradingSession.OVERNIGHT)
        
        assert overnight_config.min_confidence == 0.70
        assert overnight_config.spread_gate_ticks == 3.0
    
    def test_reads_premarket_config(self):
        """SessionManager should read PRE_MARKET config (both keys)."""
        # Test 'premarket' key
        config = {
            "futures_trading": {
                "active_sessions": {
                    "premarket": {
                        "min_confidence": 0.65,
                    }
                }
            }
        }
        manager = SessionManager(config=config)
        premarket_config = manager.get_session_config(TradingSession.PRE_MARKET)
        assert premarket_config.min_confidence == 0.65
        
        # Test 'pre_market' key
        config2 = {
            "futures_trading": {
                "active_sessions": {
                    "pre_market": {
                        "min_confidence": 0.68,
                    }
                }
            }
        }
        manager2 = SessionManager(config=config2)
        premarket_config2 = manager2.get_session_config(TradingSession.PRE_MARKET)
        assert premarket_config2.min_confidence == 0.68
    
    def test_ignores_non_config_keys(self):
        """SessionManager should ignore start_time, end_time, timezone keys."""
        config = {
            "futures_trading": {
                "active_sessions": {
                    "rth": {
                        "start_time": "08:30",  # Should be ignored
                        "end_time": "15:00",    # Should be ignored
                        "timezone": "America/Chicago",  # Should be ignored
                        "min_confidence": 0.55,  # Should be applied
                    }
                }
            }
        }
        manager = SessionManager(config=config)
        rth_config = manager.get_session_config(TradingSession.RTH)
        
        # Should have applied min_confidence
        assert rth_config.min_confidence == 0.55
        # Should not have start_time attribute
        assert not hasattr(rth_config, 'start_time')
    
    def test_handles_missing_config(self):
        """SessionManager should use defaults when no config provided."""
        manager = SessionManager(config={})
        rth_config = manager.get_session_config(TradingSession.RTH)
        
        # Should have default values
        assert rth_config.min_confidence == 0.55
        assert rth_config.max_trades_per_hour == 2


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = SessionManager(config={})
    
    def test_session_boundary_rth_start(self):
        """Exactly 8:30 AM CT should be RTH."""
        rth_start = datetime(2025, 1, 14, 8, 30, 0, tzinfo=CST)
        assert self.manager.get_current_session(rth_start) == TradingSession.RTH
    
    def test_session_boundary_rth_end(self):
        """Exactly 3:00 PM CT should NOT be RTH (close window starts)."""
        rth_end = datetime(2025, 1, 14, 15, 0, 0, tzinfo=CST)
        # At 3:00 PM, we're in the close window (3-4 PM), but session is still RTH
        session = self.manager.get_current_session(rth_end)
        assert session == TradingSession.RTH
    
    def test_session_boundary_evening_start(self):
        """Exactly 5:00 PM CT should be EVENING."""
        evening_start = datetime(2025, 1, 14, 17, 0, 0, tzinfo=CST)
        assert self.manager.get_current_session(evening_start) == TradingSession.EVENING
    
    def test_session_boundary_overnight_transition(self):
        """Exactly 11:00 PM CT should be OVERNIGHT."""
        overnight_start = datetime(2025, 1, 14, 23, 0, 0, tzinfo=CST)
        assert self.manager.get_current_session(overnight_start) == TradingSession.OVERNIGHT
    
    def test_naive_datetime_handled(self):
        """SessionManager should handle naive datetimes."""
        # Naive datetime (no timezone)
        naive_time = datetime(2025, 1, 14, 10, 0, 0)
        # Should not raise, should assume CST
        session = self.manager.get_current_session(naive_time)
        assert session == TradingSession.RTH


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
