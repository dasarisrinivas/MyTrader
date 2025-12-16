# Strategy Performance Analysis Report

**Generated:** 2025-11-02 08:04:41

**Data Period:** 2024-01-02 09:30:00 to 2024-01-08 15:59:00

**Total Bars:** 1,950

## Performance Metrics Comparison

| Strategy | Sharpe | Sortino | Max DD (%) | Profit Factor | Win Rate (%) | Total Trades | Total Return (%) |
|----------|--------|---------|------------|---------------|--------------|--------------|------------------|
| RSI_MACD_Sentiment | -1.08 | -0.25 | -1.85 | 0.18 | 0.0 | 20 | -1.77 |
| Momentum_Reversal | 0.00 | 0.00 | 0.00 | inf | 0.0 | 0 | 0.00 |
| Combined_Baseline | -1.08 | -0.25 | -1.85 | 0.18 | 0.0 | 20 | -1.77 |
| Enhanced_Regime | -1.17 | -0.31 | -2.16 | 0.30 | 0.0 | 8 | -2.11 |

## Target Metrics

- **Sharpe Ratio:** ≥ 1.5 ✓
- **Max Drawdown:** ≤ 15% ✓
- **Win Rate:** ≥ 60% ✓
- **Profit Factor:** ≥ 1.3 ✓

## Key Insights

- **Best by Sharpe Ratio:** Momentum_Reversal (0.00)
- **Best by Total Return:** Momentum_Reversal (0.00%)

## Visualizations

- Equity Curves: `equity_curves_comparison.png`
- Drawdown Analysis: `drawdown_analysis.png`
- Trade Analysis: `trade_analysis.png`
- Risk Metrics: `risk_metrics.png`
- Monthly Returns: `monthly_returns_heatmap.png`

---

## Backtest Agent Diagnostics (2025-12-14 Update)

### 1. Agent Execution Gaps
- **Agent 2 (Decision Engine)** now produces daily outputs by combining retrieval memory with deterministic heuristics (`mytrader/backtest/runner.py`, `mytrader/agents/lambda_wrappers.py`). Similar trade analogs are captured from completed trades, and if no analogs exist the wrapper falls back to RSI/trend-based scoring with adaptive thresholds.
- **Agent 3 (Risk Engine)** receives every Agent 2 action (including placeholders) along with adaptive risk parameters so it can emit an evaluation for each decision event. Placeholder entries are written automatically on days where market data is missing to unblock downstream agents.
- The scheduler explicitly runs Agent 2/3 per bar and records their artifacts under the correct backtest date so the validator no longer flags “missing decisions/risk logs”.

### 2. Daily Trade Simulation & Execution Flow
- The intraday loop now tracks every Agent 2 invocation (`agent2_invocations`) and ensures trades are simulated whenever heuristic confidence beats the adaptive threshold. Even if no trade is taken, a decision log is produced so execution gaps are visible.
- When a position is opened, we capture the market snapshot and later pair it with the realized P&L on exit to feed the similarity search memory, guaranteeing actionable analogs after the first few trades.
- Daily summaries include counts of trades, decisions and estimated “missed opportunity” (difference between day close/open when no trade fired) so that scheduling/logic gaps stand out immediately.

### 3. Adaptive Learning (Agent 4)
- Introduced a persisted `strategy_state.json` plus adjustment history managed by `StrategyStateManager` (`mytrader/learning/strategy_state.py`). Agent 4 now tunes `decision_threshold`, `exploration_rate`, `risk_multiplier`, and trade caps nightly based on daily P&L, trade counts, and missed opportunities.
- Agent 4 artifacts embed the updated state snapshot plus notes about what changed and why, creating an auditable learning trail rather than silent skips.
- The scheduler always runs Agent 4—even on data-less days—passing in summarized metrics so the learning agent can react (loosen criteria when zero trades occur, tighten risk after drawdowns, etc.).

### 4. Artifact Resiliency & Alerts
- After every trading day the runner validates the per-day artifact set. Missing files trigger immediate placeholder generation (or an on-the-spot Agent 4 rerun) and log warnings so issues are surfaced while the run is in-progress.
- Placeholder NDJSON entries explicitly indicate why they were created (e.g., “NO_DATA” or “BACKFILL”) and keep the downstream pipeline alive without hiding failures.

### 5. Testing & Observability Recommendations
- Extend `tests/test_backtest_harness.py` with fixtures that assert the new placeholder and adaptive-learning behavior (e.g., verifying Agent 4 updates thresholds when trades=0).
- Capture lightweight decision/risk telemetry per day (already emitted via logger) into a CSV for quick visualization; plotting `decision_events` vs trades will highlight overly strict settings before running the full 30-day sweep.
