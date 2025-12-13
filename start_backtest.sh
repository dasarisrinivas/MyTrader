#!/bin/bash

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#          🤖 MyTrader - Backtest Runner 🤖                      
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Usage: ./start_backtest.sh [OPTIONS]
#
# Runs a 30-day backtest that verifies all four agents are invoked
# correctly, produce expected artifacts, and integrate end-to-end.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BLUE}          🤖 MyTrader - Backtest Runner 🤖${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Parse arguments
START_DATE=""
END_DATE=""
SYMBOL="ES"
DATA_SOURCE="local"
CONFIG="config.yaml"

while [[ $# -gt 0 ]]; do
    case $1 in
        --start-date)
            START_DATE="$2"
            shift 2
            ;;
        --end-date)
            END_DATE="$2"
            shift 2
            ;;
        --symbol)
            SYMBOL="$2"
            shift 2
            ;;
        --data-source)
            DATA_SOURCE="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./start_backtest.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --start-date YYYY-MM-DD    Start date (default: 30 days ago)"
            echo "  --end-date YYYY-MM-DD      End date (default: yesterday)"
            echo "  --symbol SYMBOL            Trading symbol (default: ES)"
            echo "  --data-source local|s3|ib Data source (default: local)"
            echo "                            'ib' downloads from IBKR automatically"
            echo "  --config PATH              Config file (default: config.yaml)"
            echo ""
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check if config file exists
if [ ! -f "$CONFIG" ]; then
    echo -e "${RED}❌ Config file not found: $CONFIG${NC}"
    echo ""
    echo "Copy config.example.yaml to config.yaml and configure it:"
    echo "  cp config.example.yaml config.yaml"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment if it exists
VENV_PATH="$PWD/.venv"
if [ -d "$VENV_PATH" ]; then
    echo -e "${BLUE}[INFO]${NC} Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
fi

# Set feature flags for backtest mode
export FF_BACKTEST_MODE=1
export FF_LOCAL_LAMBDA=1
export FF_ARTIFACT_LOGGING=1

# Use python3
PYTHON_BIN="python3"

# Set up data directory and file
DATA_DIR="data/ib"
DATA_FILE="$DATA_DIR/${SYMBOL}_1m_last30d.parquet"

mkdir -p "$DATA_DIR"

# Handle IBKR data download
if [ "$DATA_SOURCE" = "ib" ]; then
    echo -e "${BLUE}[INFO]${NC} Downloading last 30 days from IBKR into $DATA_FILE ..."
    if ! $PYTHON_BIN tools/ib_download_last30d.py --symbol "$SYMBOL" --out "$DATA_FILE"; then
        echo -e "${RED}❌ Failed to download data from IBKR${NC}"
        exit 1
    fi
    DATA_SOURCE="local"
    echo -e "${GREEN}✅ Data download complete${NC}"
fi

# Auto-download if local data is missing
if [ "$DATA_SOURCE" = "local" ] && [ ! -f "$DATA_FILE" ]; then
    echo -e "${YELLOW}[WARN]${NC} Missing $DATA_FILE. Auto-downloading from IBKR..."
    if ! $PYTHON_BIN tools/ib_download_last30d.py --symbol "$SYMBOL" --out "$DATA_FILE"; then
        echo -e "${RED}❌ Failed to download data from IBKR${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Data download complete${NC}"
fi

# Build Python command
PYTHON_CMD="$PYTHON_BIN -m mytrader.backtest.runner"

if [ -n "$START_DATE" ]; then
    PYTHON_CMD="$PYTHON_CMD --start-date $START_DATE"
fi

if [ -n "$END_DATE" ]; then
    PYTHON_CMD="$PYTHON_CMD --end-date $END_DATE"
fi

PYTHON_CMD="$PYTHON_CMD --symbol $SYMBOL"
PYTHON_CMD="$PYTHON_CMD --data-source $DATA_SOURCE"
PYTHON_CMD="$PYTHON_CMD --config $CONFIG"

echo -e "${BLUE}[INFO]${NC} Starting backtest..."
echo -e "${BLUE}[INFO]${NC} Command: $PYTHON_CMD"
echo ""

# Run backtest
if $PYTHON_CMD; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${GREEN}                   ✅ Backtest Complete! ✅${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo -e "${BLUE}📊 Summary Report:${NC} reports/backtest_last30_summary.md"
    echo -e "${BLUE}📁 Artifacts:${NC} artifacts/backtest/"
    echo ""
else
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${RED}                   ❌ Backtest Failed ❌${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo -e "${YELLOW}Check logs/backtest.log for details${NC}"
    exit 1
fi
