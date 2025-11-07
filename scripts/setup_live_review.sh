#!/bin/bash
# Setup script for live paper trading review system

set -e

echo "=============================================="
echo "Live Paper Trading Review System Setup"
echo "=============================================="
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check Python installation
echo "[1/6] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "✓ Python $PYTHON_VERSION found"

# Check virtual environment
echo ""
echo "[2/6] Checking virtual environment..."
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Creating..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment exists"
fi

# Activate venv
source venv/bin/activate

# Install dependencies
echo ""
echo "[3/6] Checking dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "✓ Dependencies installed"

# Create required directories
echo ""
echo "[4/6] Creating directories..."
mkdir -p data
mkdir -p logs
mkdir -p reports/daily_reviews
mkdir -p reports/ai_insights
mkdir -p data/config_backups
echo "✓ Directories created"

# Check config file
echo ""
echo "[5/6] Checking configuration..."
if [ ! -f "config.yaml" ]; then
    if [ -f "config.example.yaml" ]; then
        echo "⚠️  config.yaml not found. Copying from example..."
        cp config.example.yaml config.yaml
        echo "✓ config.yaml created from example"
        echo "⚠️  Please edit config.yaml with your settings"
    else
        echo "❌ Neither config.yaml nor config.example.yaml found"
        exit 1
    fi
else
    echo "✓ config.yaml exists"
fi

# Check AWS credentials
echo ""
echo "[6/6] Checking AWS credentials..."
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "⚠️  AWS credentials not found in environment"
    echo "   Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
    echo "   Or configure via: aws configure"
else
    echo "✓ AWS credentials found"
fi

# Test run
echo ""
echo "=============================================="
echo "Setup complete! Running test..."
echo "=============================================="
echo ""

# Check if database has data
if [ -f "data/llm_trades.db" ]; then
    echo "Testing with existing database..."
    python3 run_daily_review.py --days 1
else
    echo "⚠️  No trade database found (data/llm_trades.db)"
    echo "   Trade logging must be active during paper trading"
    echo "   See docs/LIVE_TRADING_REVIEW_GUIDE.md for setup"
fi

echo ""
echo "=============================================="
echo "Next Steps:"
echo "=============================================="
echo ""
echo "1. Configure IBKR trade logging"
echo "   See: docs/LIVE_TRADING_REVIEW_GUIDE.md"
echo ""
echo "2. Run manual review:"
echo "   python run_daily_review.py"
echo ""
echo "3. Setup scheduled execution:"
echo "   See: scripts/setup_cron.sh (Linux/macOS)"
echo "   Or use Task Scheduler (Windows)"
echo ""
echo "4. Review reports in:"
echo "   reports/daily_reviews/"
echo ""
echo "Documentation: docs/LIVE_TRADING_REVIEW_GUIDE.md"
echo "=============================================="
