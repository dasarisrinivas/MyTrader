# Backtest Entrypoint Documentation

## Overview

The backtest harness runs a 30-day backtest that verifies all four agents are invoked correctly, produce expected artifacts, and integrate end-to-end. The backtest runs offline using historical data and does not require Interactive Brokers connectivity.

## Entry Point

### Shell Script: `start_backtest.sh`

```bash
./start_backtest.sh [OPTIONS]
```

**Options:**
- `--start-date YYYY-MM-DD` - Start date (default: 30 days ago)
- `--end-date YYYY-MM-DD` - End date (default: yesterday)
- `--symbol SYMBOL` - Trading symbol (default: ES for SPY futures)
- `--data-source local|s3` - Data source (default: local)
- `--config PATH` - Path to config file (default: config.yaml)

### Python Module: `mytrader.backtest.runner`

```bash
python -m mytrader.backtest.runner \
    --start-date 2024-11-01 \
    --end-date 2024-11-30 \
    --symbol ES \
    --data-source local \
    --config config.yaml
```

## Configuration

The backtest uses the same `config.yaml` as live trading, with additional feature flags:

```yaml
backtest:
  enabled: true
  mode: backtest  # Ensures no live trades
  data_path: "data/historical_spy_es.parquet"
  artifacts_dir: "artifacts/backtest"
  local_lambda_mode: true  # Use local Lambda wrappers instead of AWS

features:
  enable_backtest_mode: true  # Feature flag to enable backtest mode
```

## Architecture

### Decision Pipeline

The backtest uses the **exact same decision pipeline** as live trading:

```
Market Data → Agent 2 (Decision) → Agent 3 (Risk) → Simulated Execution
```

The only difference is the broker adapter is swapped for a simulated executor.

### Agent Invocation

1. **Agent 1: Data Ingestion & Feature Builder** (Nightly)
   - Triggered once per day at end-of-day
   - Processes raw trade data and generates features
   - Writes to `artifacts/backtest/YYYY-MM-DD/agent1_features_manifest.json`

2. **Agent 2: RAG + Similarity Search Decision Engine** (Real-time)
   - Invoked during trading loop when decision is needed
   - Uses local vector store or stubbed KB client in backtest mode
   - Logs decisions to `artifacts/backtest/YYYY-MM-DD/agent2_decisions.ndjson`

3. **Agent 3: Risk & Position Sizing Agent** (Real-time)
   - Invoked immediately after Agent 2 decision
   - Enforces SL/TP validity and position sizing rules
   - Writes to `artifacts/backtest/YYYY-MM-DD/agent3_risk.ndjson`

4. **Agent 4: Strategy Optimization & Learning Agent** (Nightly 11PM CST)
   - Triggered once per day at 11 PM CST emulation
   - Consumes day's outcomes and writes learning updates
   - Writes to `artifacts/backtest/YYYY-MM-DD/agent4_learning_update.json`

## Artifacts

Each backtest day produces artifacts in:
```
artifacts/backtest/YYYY-MM-DD/
├── agent1_features_manifest.json
├── agent2_decisions.ndjson
├── agent3_risk.ndjson
├── trades.ndjson
└── agent4_learning_update.json
```

## Summary Report

After completion, generates:
```
reports/backtest_last30_summary.md
```

Includes:
- Days processed
- #decisions, #trades, win rate, total PnL
- #Agent1 runs, #Agent4 runs
- Missing artifact check results (must be zero)

## Feature Flags

The backtest uses feature flags to ensure it doesn't affect production:

- `FF_BACKTEST_MODE=1` - Enables backtest mode (disables IB connection)
- `FF_LOCAL_LAMBDA=1` - Uses local Lambda wrappers instead of AWS
- `FF_ARTIFACT_LOGGING=1` - Enables strict artifact logging

## Data Requirements

### Historical Data Format

Expected format: Parquet or CSV with columns:
- `timestamp` (datetime index)
- `open`, `high`, `low`, `close` (float)
- `volume` (int, optional)

### Data Location

- **Local**: `data/historical_spy_es.parquet` or `data/es_YYYY-MM-DD_to_YYYY-MM-DD.parquet`
- **S3**: `s3://bucket-name/historical/YYYY-MM-DD/...`

If data is missing, the backtest will:
1. Check for local files
2. Attempt S3 download (if configured)
3. Fail with clear error message if data unavailable

## Execution Flow

1. **Initialization**
   - Load config
   - Set feature flags
   - Initialize local Lambda wrappers
   - Set up artifact directories

2. **Day-by-Day Processing**
   For each day in date range:
   - Load market data for the day
   - Run Agent 1 (data ingestion) at start-of-day
   - Simulate trading loop:
     - Get market data snapshot
     - Invoke Agent 2 (decision)
     - Invoke Agent 3 (risk)
     - Simulate execution if approved
   - Run Agent 4 (learning) at end-of-day (11 PM CST)

3. **Artifact Generation**
   - Write all agent outputs to artifacts directory
   - Generate per-day manifests
   - Validate all artifacts present

4. **Summary Generation**
   - Aggregate statistics across all days
   - Generate summary report
   - Validate no missing artifacts

## Verification

The backtest verifies:
- ✅ Agent 1 runs exactly once per day
- ✅ Agent 2 runs for every decision attempt
- ✅ Agent 3 runs for every Agent 2 decision
- ✅ Agent 4 runs exactly once per day at 11 PM CST
- ✅ All artifacts are present and valid
- ✅ No missing artifacts (fails if any missing)

## Example Usage

```bash
# Run 30-day backtest with defaults
./start_backtest.sh

# Run custom date range
./start_backtest.sh --start-date 2024-11-01 --end-date 2024-11-30

# Run with S3 data source
./start_backtest.sh --data-source s3

# Run with custom config
./start_backtest.sh --config config.backtest.yaml
```

## Troubleshooting

### Missing Historical Data
- Check `data/` directory for parquet/csv files
- Verify date range matches available data
- Use `--data-source s3` if data is in S3

### Missing Artifacts
- Check `artifacts/backtest/` directory
- Verify feature flags are set correctly
- Check logs for agent invocation errors

### Agent Not Invoked
- Verify scheduler emulator is running
- Check agent invocation logs
- Ensure feature flags are enabled
