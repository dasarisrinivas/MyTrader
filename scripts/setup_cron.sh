#!/bin/bash
# Setup cron job for daily paper trading review
# Run this script to schedule automatic daily reviews at 6:00 PM

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "Cron Setup for Daily Trading Review"
echo "=============================================="
echo ""

# Detect Python path
if [ -f "$PROJECT_DIR/venv/bin/python" ]; then
    PYTHON_PATH="$PROJECT_DIR/venv/bin/python"
    echo "✓ Using virtual environment Python: $PYTHON_PATH"
elif command -v python3 &> /dev/null; then
    PYTHON_PATH=$(which python3)
    echo "✓ Using system Python: $PYTHON_PATH"
else
    echo "❌ Python not found. Please install Python 3.8+"
    exit 1
fi

# Create cron command
CRON_CMD="0 18 * * * cd $PROJECT_DIR && $PYTHON_PATH run_daily_review.py >> logs/cron_review.log 2>&1"

echo ""
echo "Cron job will run:"
echo "  • Daily at 6:00 PM (18:00)"
echo "  • Command: python run_daily_review.py"
echo "  • Output: logs/cron_review.log"
echo ""

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "run_daily_review.py"; then
    echo "⚠️  Cron job already exists!"
    echo ""
    echo "Current crontab:"
    crontab -l | grep "run_daily_review.py"
    echo ""
    read -p "Replace existing job? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
    
    # Remove existing job
    crontab -l | grep -v "run_daily_review.py" | crontab -
    echo "✓ Removed existing job"
fi

# Add new cron job
(crontab -l 2>/dev/null; echo "$CRON_CMD") | crontab -

echo ""
echo "✓ Cron job installed successfully!"
echo ""
echo "Verify installation:"
echo "  crontab -l | grep run_daily_review"
echo ""
echo "Test manual execution:"
echo "  cd $PROJECT_DIR && python run_daily_review.py"
echo ""
echo "View cron logs:"
echo "  tail -f $PROJECT_DIR/logs/cron_review.log"
echo ""
echo "Remove cron job:"
echo "  crontab -e  # Delete the line with run_daily_review.py"
echo ""
echo "=============================================="
echo "Schedule Configuration:"
echo "=============================================="
echo ""
echo "Current: Daily at 6:00 PM (18:00)"
echo ""
echo "To change schedule, edit crontab:"
echo "  crontab -e"
echo ""
echo "Cron format: MIN HOUR DAY MONTH WEEKDAY"
echo "Examples:"
echo "  0 18 * * *     - Daily at 6:00 PM"
echo "  0 17 * * 1-5   - Weekdays at 5:00 PM"
echo "  30 18 * * *    - Daily at 6:30 PM"
echo "  0 20 * * 0     - Sundays at 8:00 PM"
echo ""
echo "=============================================="
