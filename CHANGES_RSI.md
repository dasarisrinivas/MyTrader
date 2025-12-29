# RSI Trend-Aware Pullback Integration

- Added reusable RSI computation helper (`mytrader/features/indicators.py`) with tests for short history and trend behaviour.
- Introduced `RSITrendPullbackFilter` (`mytrader/hybrid/rsi_trend_filter.py`) to align RSI signals with EMA50 trend, ATR bounds, and session-aware adjustments.
- Refactored `DeterministicEngine` to use the pullback filter as confirmation (never standalone), gate BUY/SELL when trend/RSI/ATR conflict, and derive ATR-based SL/TP from configurable multipliers with session tweaks.
- Logged session/trend context and RSI pullback outcomes inside D-engine metadata/reasons for auditability.
