#!/bin/bash

echo "🔍 Checking IB Gateway Connection Status..."
echo ""

# Check if IB Gateway is running
if lsof -i :4001 | grep -q JavaAppli; then
    echo "✅ IB Gateway is running"
    echo ""
    echo "Current connections:"
    lsof -i :4001 | grep -v "COMMAND"
    echo ""
    
    # Count connections
    CLIENT_CONN=$(lsof -i :4001 | grep ESTABLISHED | grep -v JavaAppli | wc -l | tr -d ' ')
    echo "Active client connections: $CLIENT_CONN"
    
    if [ "$CLIENT_CONN" -gt 0 ]; then
        echo "⚠️  WARNING: There are $CLIENT_CONN active client connections"
        echo "   These should be cleared before starting trading"
        echo ""
        echo "Run: ./restart_clean.sh"
    else
        echo "✅ No active client connections - ready for trading"
    fi
else
    echo "❌ IB Gateway is NOT running"
    echo ""
    echo "Please start IB Gateway:"
    echo "  1. Open IB Gateway application"
    echo "  2. Login to Live Trading"  
    echo "  3. Wait for 'Connected' status"
fi

echo ""
echo "================================"
echo "If you see Error 162:"
echo "================================"
echo "This means IB Gateway has stale session data."
echo ""
echo "SOLUTION:"
echo "  1. Close IB Gateway COMPLETELY (Cmd+Q)"
echo "  2. Wait 60 seconds (important!)"
echo "  3. Restart IB Gateway"
echo "  4. Login fresh"
echo "  5. Run: ./restart_clean.sh"
echo "  6. Run: ./start_trading.sh"
echo ""
