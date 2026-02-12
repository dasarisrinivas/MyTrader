#!/bin/bash
# Script to check if Shree metrics endpoint is reachable
# Usage: ./check_metrics_reachable.sh [MAC_MINI_IP] [PORT]

set -e

MAC_IP="${1:-localhost}"
PORT="${2:-8000}"
ENDPOINT="http://${MAC_IP}:${PORT}/metrics"

echo "🔍 Checking Shree metrics endpoint..."
echo "   Endpoint: ${ENDPOINT}"
echo ""

# Test 1: Basic connectivity
echo "Test 1: Basic connectivity (curl)"
if curl -s --connect-timeout 5 --max-time 10 "${ENDPOINT}" > /dev/null; then
    echo "   ✅ PASS: Endpoint is reachable"
else
    echo "   ❌ FAIL: Cannot connect to endpoint"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check bot is running: ps aux | grep run_bot.py"
    echo "  2. Check PROMETHEUS_ENABLED=true in bot environment"
    echo "  3. Check firewall allows connections to port ${PORT}"
    echo "  4. Verify IP address: ifconfig | grep inet"
    exit 1
fi

# Test 2: Valid Prometheus format
echo ""
echo "Test 2: Valid Prometheus metrics format"
METRICS=$(curl -s --connect-timeout 5 --max-time 10 "${ENDPOINT}")
if echo "$METRICS" | grep -q "# HELP"; then
    echo "   ✅ PASS: Valid Prometheus metrics format"
else
    echo "   ❌ FAIL: Response doesn't look like Prometheus metrics"
    exit 1
fi

# Test 3: Shree-specific metrics present
echo ""
echo "Test 3: Shree metrics present"
EXPECTED_METRICS=(
    "shree_live_bar_age_seconds"
    "shree_stale_episode_active"
    "shree_stale_live_bars_blocks_total"
    "shree_decisions_total"
)

MISSING=0
for metric in "${EXPECTED_METRICS[@]}"; do
    if echo "$METRICS" | grep -q "^${metric}"; then
        echo "   ✅ Found: ${metric}"
    else
        echo "   ⚠️  Missing: ${metric} (may not be emitted yet)"
        MISSING=$((MISSING + 1))
    fi
done

if [ $MISSING -eq ${#EXPECTED_METRICS[@]} ]; then
    echo ""
    echo "   ❌ FAIL: No Shree metrics found at all"
    echo "   This likely means PROMETHEUS_ENABLED=false or bot hasn't started metrics yet"
    exit 1
fi

# Test 4: Sample metric values
echo ""
echo "Test 4: Sample metric values"
echo "$METRICS" | grep "^shree_" | head -10

echo ""
echo "✅ All checks passed! Metrics endpoint is healthy."
echo ""
echo "Next steps:"
echo "  1. Get your Mac mini IP: ifconfig | grep 'inet ' | grep -v 127.0.0.1"
echo "  2. Update deploy/k8s-observability/prometheus-scrape-config.yaml with your IP"
echo "  3. Apply k8s manifests: kubectl apply -f deploy/k8s-observability/"
echo "  4. Verify Prometheus can scrape: check Prometheus UI -> Status -> Targets"
