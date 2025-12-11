#!/usr/bin/env python3
"""
Test AWS Agent Integration

Quick test to verify AWS agents are properly connected to the trading bot.

Usage:
    python aws/scripts/test_live_integration.py
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger


def setup_logging():
    """Configure logging."""
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")


async def test_config_loader():
    """Test loading AWS configuration."""
    print("\n" + "=" * 60)
    print("Test 1: AWS Config Loader")
    print("=" * 60)
    
    try:
        from mytrader.aws import load_aws_config, get_aws_config
        
        config = load_aws_config()
        
        print(f"✅ Loaded configuration from deployed_resources.yaml")
        print(f"   Region: {config.get('region_name')}")
        print(f"   S3 Bucket: {config.get('s3_bucket')}")
        print(f"   Knowledge Base ID: {config.get('knowledge_base_id')}")
        print(f"   Agent IDs: {json.dumps(config.get('agent_ids', {}), indent=6)}")
        print(f"   Signal Flow ARN: {config.get('signal_flow_arn', 'Not configured')[:50]}...")
        
        return True
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False


async def test_agent_invoker():
    """Test AgentInvoker initialization."""
    print("\n" + "=" * 60)
    print("Test 2: AgentInvoker Initialization")
    print("=" * 60)
    
    try:
        from mytrader.aws import AgentInvoker
        
        invoker = AgentInvoker.from_deployed_config()
        
        print(f"✅ AgentInvoker initialized successfully")
        print(f"   Agent client region: {invoker.agent_client.region_name}")
        print(f"   Use Step Functions: {invoker.use_step_functions}")
        print(f"   Configured agents: {list(invoker.agent_client.agent_ids.keys())}")
        
        return invoker
    except Exception as e:
        print(f"❌ AgentInvoker initialization failed: {e}")
        return None


async def test_market_snapshot():
    """Test market snapshot builder."""
    print("\n" + "=" * 60)
    print("Test 3: Market Snapshot Builder")
    print("=" * 60)
    
    try:
        from mytrader.aws import MarketSnapshotBuilder
        
        builder = MarketSnapshotBuilder(symbol="ES")
        
        # Test mock snapshot
        mock_snapshot = builder.build_mock()
        print(f"✅ Mock snapshot created")
        print(f"   Symbol: {mock_snapshot.get('symbol')}")
        print(f"   Price: ${mock_snapshot.get('price'):,.2f}")
        print(f"   Trend: {mock_snapshot.get('trend')}")
        print(f"   Volatility: {mock_snapshot.get('volatility')}")
        print(f"   RSI: {mock_snapshot.get('rsi'):.1f}")
        
        # Test custom snapshot
        custom_snapshot = builder.build(
            symbol="ES",
            price=6050.00,
            trend="UPTREND",
            volatility="MED",
            rsi=55,
            atr=12.5,
        )
        print(f"\n✅ Custom snapshot created")
        print(f"   Price: ${custom_snapshot.get('price'):,.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Snapshot builder failed: {e}")
        return False


async def test_decision_agent(invoker):
    """Test invoking the Decision Agent."""
    print("\n" + "=" * 60)
    print("Test 4: Decision Agent Invocation")
    print("=" * 60)
    
    if not invoker:
        print("⚠️ Skipped - AgentInvoker not available")
        return None
    
    try:
        from mytrader.aws import MarketSnapshotBuilder
        
        builder = MarketSnapshotBuilder()
        snapshot = builder.build(
            symbol="ES",
            price=6000.00,
            trend="UPTREND",
            volatility="MED",
            rsi=42,  # Oversold-ish
            atr=10,
        )
        
        print(f"📊 Invoking Decision Agent with snapshot...")
        print(f"   Price: ${snapshot.get('price'):,.2f}, RSI: {snapshot.get('rsi'):.1f}")
        
        response = invoker.agent_client.invoke_decision_agent(
            market_snapshot=snapshot,
        )
        
        print(f"\n✅ Decision Agent Response:")
        print(f"   Decision: {response.get('decision')}")
        print(f"   Confidence: {response.get('confidence', 0):.1%}")
        print(f"   Reason: {response.get('reason', 'N/A')[:100]}...")
        
        return response
        
    except Exception as e:
        print(f"❌ Decision Agent invocation failed: {e}")
        return None


async def test_risk_agent(invoker):
    """Test invoking the Risk Agent."""
    print("\n" + "=" * 60)
    print("Test 5: Risk Agent Invocation")
    print("=" * 60)
    
    if not invoker:
        print("⚠️ Skipped - AgentInvoker not available")
        return None
    
    try:
        # Create a mock trade decision
        trade_decision = {
            'decision': 'BUY',
            'confidence': 0.75,
            'symbol': 'ES',
            'proposed_size': 1,
        }
        
        account_metrics = {
            'current_pnl_today': 250.00,
            'current_position': 0,
            'losing_streak': 0,
            'trades_today': 2,
            'account_balance': 100000.00,
            'open_risk': 0,
        }
        
        market_conditions = {
            'volatility': 'MED',
            'regime': 'UPTREND',
            'atr': 10,
            'vix': 15,
        }
        
        print(f"📊 Invoking Risk Agent with trade decision...")
        print(f"   Proposed: {trade_decision.get('decision')} (conf: {trade_decision.get('confidence'):.1%})")
        
        response = invoker.agent_client.invoke_risk_agent(
            trade_decision=trade_decision,
            account_metrics=account_metrics,
            market_conditions=market_conditions,
        )
        
        print(f"\n✅ Risk Agent Response:")
        print(f"   Allowed: {response.get('allowed_to_trade')}")
        print(f"   Size Multiplier: {response.get('size_multiplier', 0):.2f}")
        print(f"   Adjusted Size: {response.get('adjusted_size', 0)}")
        print(f"   Risk Flags: {response.get('risk_flags', [])}")
        
        return response
        
    except Exception as e:
        print(f"❌ Risk Agent invocation failed: {e}")
        return None


async def test_full_trading_decision(invoker):
    """Test full trading decision flow."""
    print("\n" + "=" * 60)
    print("Test 6: Full Trading Decision Flow")
    print("=" * 60)
    
    if not invoker:
        print("⚠️ Skipped - AgentInvoker not available")
        return None
    
    try:
        from mytrader.aws import MarketSnapshotBuilder
        
        builder = MarketSnapshotBuilder()
        snapshot = builder.build(
            symbol="ES",
            price=6020.00,
            trend="UPTREND",
            volatility="MED",
            rsi=38,  # Oversold
            atr=12,
        )
        
        account_metrics = {
            'current_pnl_today': 100.00,
            'current_position': 0,
            'losing_streak': 1,
            'trades_today': 3,
            'account_balance': 100000.00,
            'open_risk': 0,
        }
        
        print(f"📊 Getting full trading decision...")
        print(f"   Snapshot: price=${snapshot.get('price'):,.2f}, RSI={snapshot.get('rsi'):.1f}")
        
        decision = invoker.get_trading_decision(
            market_snapshot=snapshot,
            account_metrics=account_metrics,
        )
        
        print(f"\n✅ Full Trading Decision:")
        print(f"   Final Decision: {decision.get('decision')}")
        print(f"   Confidence: {decision.get('confidence', 0):.1%}")
        print(f"   Allowed to Trade: {decision.get('allowed_to_trade')}")
        print(f"   Adjusted Size: {decision.get('adjusted_size', 0)}")
        print(f"   Risk Flags: {decision.get('risk_flags', [])}")
        print(f"   Latency: {decision.get('latency_ms', 0)}ms")
        
        return decision
        
    except Exception as e:
        print(f"❌ Full trading decision failed: {e}")
        return None


async def main():
    """Run all tests."""
    setup_logging()
    
    print("\n" + "=" * 60)
    print("AWS Agent Integration Test Suite")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Config Loader
    results['config'] = await test_config_loader()
    
    # Test 2: AgentInvoker
    invoker = await test_agent_invoker()
    results['invoker'] = invoker is not None
    
    # Test 3: Market Snapshot
    results['snapshot'] = await test_market_snapshot()
    
    # Test 4: Decision Agent
    decision_response = await test_decision_agent(invoker)
    results['decision_agent'] = decision_response is not None
    
    # Test 5: Risk Agent
    risk_response = await test_risk_agent(invoker)
    results['risk_agent'] = risk_response is not None
    
    # Test 6: Full Flow
    full_decision = await test_full_trading_decision(invoker)
    results['full_flow'] = full_decision is not None
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"   {status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! AWS agents are ready for live trading.")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
