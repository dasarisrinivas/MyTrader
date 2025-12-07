#!/bin/bash
# =============================================================================
# Local Test Runner for MyTrader
# =============================================================================
# Runs all unit tests and integration tests locally
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
RESULTS_FILE="${PROJECT_ROOT}/reconcile_test_results.txt"
LOG_DIR="${PROJECT_ROOT}/logs"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Ensure log directory exists
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "  MyTrader Local Test Runner"
echo "=========================================="
echo ""
echo "Project root: $PROJECT_ROOT"
echo "Results file: $RESULTS_FILE"
echo ""

# Start results file
echo "MyTrader Test Results" > "$RESULTS_FILE"
echo "Run at: $(date -u +%Y-%m-%dT%H:%M:%S.000Z)" >> "$RESULTS_FILE"
echo "=========================================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Function to run a test file
run_test() {
    local test_file=$1
    local test_name=$(basename "$test_file" .py)
    
    echo -e "${BLUE}Running: $test_name${NC}"
    echo "" >> "$RESULTS_FILE"
    echo "Test: $test_name" >> "$RESULTS_FILE"
    echo "-----------------------------------------" >> "$RESULTS_FILE"
    
    # Run the test
    if python -m pytest "$test_file" -v --tb=short 2>&1 | tee -a "$RESULTS_FILE"; then
        echo -e "${GREEN}✅ $test_name: PASSED${NC}"
        echo "Status: PASSED" >> "$RESULTS_FILE"
        return 0
    else
        echo -e "${RED}❌ $test_name: FAILED${NC}"
        echo "Status: FAILED" >> "$RESULTS_FILE"
        return 1
    fi
}

# Track overall status
FAILED_TESTS=0
TOTAL_TESTS=0

# Change to project directory
cd "$PROJECT_ROOT"

# Check Python environment
echo -e "${BLUE}Checking Python environment...${NC}"
python --version
echo "Python: $(python --version)" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Install test dependencies if needed
if ! python -c "import pytest" 2>/dev/null; then
    echo -e "${YELLOW}Installing pytest...${NC}"
    pip install pytest pytest-asyncio --quiet
fi

# Run reconciliation tests
echo ""
echo -e "${BLUE}=== Running Reconciliation Tests ===${NC}"
echo "" >> "$RESULTS_FILE"
echo "=== Reconciliation Tests ===" >> "$RESULTS_FILE"

TOTAL_TESTS=$((TOTAL_TESTS + 1))
if ! run_test "tests/test_reconcile.py"; then
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

# Run live data tests
echo ""
echo -e "${BLUE}=== Running Live Data Tests ===${NC}"
echo "" >> "$RESULTS_FILE"
echo "=== Live Data Tests ===" >> "$RESULTS_FILE"

TOTAL_TESTS=$((TOTAL_TESTS + 1))
if ! run_test "tests/test_live_data.py"; then
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

# Run mock IB server tests
echo ""
echo -e "${BLUE}=== Running Mock IB Server Tests ===${NC}"
echo "" >> "$RESULTS_FILE"
echo "=== Mock IB Server Tests ===" >> "$RESULTS_FILE"

if [ -f "tests/mock_ib_server.py" ]; then
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    if python tests/mock_ib_server.py 2>&1 | tee -a "$RESULTS_FILE"; then
        echo -e "${GREEN}✅ mock_ib_server: PASSED${NC}"
        echo "Status: PASSED" >> "$RESULTS_FILE"
    else
        echo -e "${RED}❌ mock_ib_server: FAILED${NC}"
        echo "Status: FAILED" >> "$RESULTS_FILE"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
fi

# Run existing tests if they exist
echo ""
echo -e "${BLUE}=== Running Existing Tests ===${NC}"

for test_file in tests/test_sync_logic.py tests/test_position_manager.py tests/test_orders.py; do
    if [ -f "$test_file" ]; then
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
        if ! run_test "$test_file"; then
            FAILED_TESTS=$((FAILED_TESTS + 1))
        fi
    fi
done

# Summary
echo ""
echo "=========================================="
echo -e "${BLUE}  Test Summary${NC}"
echo "=========================================="
echo ""

PASSED_TESTS=$((TOTAL_TESTS - FAILED_TESTS))

echo "" >> "$RESULTS_FILE"
echo "=========================================" >> "$RESULTS_FILE"
echo "SUMMARY" >> "$RESULTS_FILE"
echo "=========================================" >> "$RESULTS_FILE"
echo "Total tests: $TOTAL_TESTS" >> "$RESULTS_FILE"
echo "Passed: $PASSED_TESTS" >> "$RESULTS_FILE"
echo "Failed: $FAILED_TESTS" >> "$RESULTS_FILE"

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}All $TOTAL_TESTS tests passed!${NC}"
    echo "Status: ALL PASSED" >> "$RESULTS_FILE"
    exit 0
else
    echo -e "${RED}$FAILED_TESTS of $TOTAL_TESTS tests failed${NC}"
    echo "Status: $FAILED_TESTS FAILED" >> "$RESULTS_FILE"
    exit 1
fi
