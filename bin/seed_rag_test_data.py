#!/usr/bin/env python3
"""
Seed RAG storage with test trade data for testing retrieval.
"""

import sys
sys.path.insert(0, '/Users/svss/Documents/code/MyTrader')

from datetime import datetime, timedelta
from mytrader.rag.s3_storage import S3Storage
from mytrader.rag.rag_storage_manager import RAGStorageManager, TradeRecord
import json

def create_test_trades():
    """Create realistic test trade records."""
    
    base_time = datetime.now() - timedelta(days=7)
    
    trades = [
        # Winning BUY trade - oversold bounce at PDL
        TradeRecord(
            trade_id="test_001",
            timestamp=base_time.isoformat(),
            action="BUY",
            entry_price=6050.25,
            exit_price=6058.50,
            quantity=1,
            stop_loss=6045.0,
            take_profit=6060.0,
            pnl=8.25 * 50,  # $412.50 profit
            pnl_pct=0.14,
            market_trend="DOWNTREND",
            volatility_regime="LOW",
            rsi=38.5,
            macd_hist=-0.15,
            atr=2.5,
            ema_9=6052.0,
            ema_20=6055.0,
            pdh=6080.0,
            pdl=6045.0,
            llm_action="BUY",
            llm_confidence=0.72,
            llm_reasoning="Oversold bounce near PDL with RSI < 40. MACD histogram turning positive.",
            result="WIN",
            duration_minutes=7.0,
            exit_reason="TP_HIT",
            time_of_day="MIDDAY",
            day_of_week="Monday",
        ),
        
        # Winning SELL trade - overbought rejection at PDH
        TradeRecord(
            trade_id="test_002",
            timestamp=(base_time + timedelta(days=1)).isoformat(),
            action="SELL",
            entry_price=6095.75,
            exit_price=6088.00,
            quantity=1,
            stop_loss=6100.0,
            take_profit=6085.0,
            pnl=7.75 * 50,  # $387.50 profit
            pnl_pct=0.13,
            market_trend="UPTREND",
            volatility_regime="MEDIUM",
            rsi=62.3,
            macd_hist=0.25,
            atr=3.2,
            ema_9=6092.0,
            ema_20=6088.0,
            pdh=6098.0,
            pdl=6065.0,
            llm_action="SELL",
            llm_confidence=0.68,
            llm_reasoning="Overbought rejection near PDH with RSI > 60. Mean reversion setup.",
            result="WIN",
            duration_minutes=9.0,
            exit_reason="TP_HIT",
            time_of_day="OPEN",
            day_of_week="Tuesday",
        ),
        
        # Losing BUY trade - caught in downtrend
        TradeRecord(
            trade_id="test_003",
            timestamp=(base_time + timedelta(days=2)).isoformat(),
            action="BUY",
            entry_price=6070.00,
            exit_price=6065.50,
            quantity=1,
            stop_loss=6065.0,
            take_profit=6080.0,
            pnl=-4.50 * 50,  # -$225 loss
            pnl_pct=-0.07,
            market_trend="DOWNTREND",
            volatility_regime="HIGH",
            rsi=45.0,
            macd_hist=-0.35,
            atr=4.5,
            ema_9=6068.0,
            ema_20=6075.0,
            pdh=6100.0,
            pdl=6050.0,
            llm_action="BUY",
            llm_confidence=0.55,
            llm_reasoning="Attempted bounce but strong downtrend continued. MACD still negative.",
            result="LOSS",
            duration_minutes=3.0,
            exit_reason="SL_HIT",
            time_of_day="MIDDAY",
            day_of_week="Wednesday",
        ),
        
        # Winning scalp BUY in range-bound market
        TradeRecord(
            trade_id="test_004",
            timestamp=(base_time + timedelta(days=3)).isoformat(),
            action="SCALP_BUY",
            entry_price=6082.25,
            exit_price=6084.75,
            quantity=1,
            stop_loss=6080.0,
            take_profit=6085.0,
            pnl=2.50 * 50,  # $125 profit
            pnl_pct=0.04,
            market_trend="RANGE",
            volatility_regime="LOW",
            rsi=46.5,
            macd_hist=0.05,
            atr=1.8,
            ema_9=6083.0,
            ema_20=6082.5,
            pdh=6090.0,
            pdl=6075.0,
            llm_action="SCALP_BUY",
            llm_confidence=0.65,
            llm_reasoning="Range-bound scalp with RSI slightly below 50. Quick 2.5pt target.",
            result="WIN",
            duration_minutes=2.0,
            exit_reason="TP_HIT",
            time_of_day="CLOSE",
            day_of_week="Thursday",
        ),
        
        # Winning SELL trade - trend continuation
        TradeRecord(
            trade_id="test_005",
            timestamp=(base_time + timedelta(days=4)).isoformat(),
            action="SELL",
            entry_price=6055.00,
            exit_price=6048.25,
            quantity=1,
            stop_loss=6060.0,
            take_profit=6045.0,
            pnl=6.75 * 50,  # $337.50 profit
            pnl_pct=0.11,
            market_trend="DOWNTREND",
            volatility_regime="MEDIUM",
            rsi=42.0,
            macd_hist=-0.28,
            atr=3.0,
            ema_9=6058.0,
            ema_20=6065.0,
            pdh=6085.0,
            pdl=6040.0,
            llm_action="SELL",
            llm_confidence=0.75,
            llm_reasoning="Trend continuation short. Price below all EMAs, MACD confirming.",
            result="WIN",
            duration_minutes=10.0,
            exit_reason="TP_HIT",
            time_of_day="OPEN",
            day_of_week="Friday",
        ),
        
        # Losing SELL trade - false breakdown
        TradeRecord(
            trade_id="test_006",
            timestamp=(base_time + timedelta(days=5)).isoformat(),
            action="SCALP_SELL",
            entry_price=6078.50,
            exit_price=6081.00,
            quantity=1,
            stop_loss=6081.0,
            take_profit=6075.0,
            pnl=-2.50 * 50,  # -$125 loss
            pnl_pct=-0.04,
            market_trend="RANGE",
            volatility_regime="LOW",
            rsi=52.0,
            macd_hist=-0.02,
            atr=1.5,
            ema_9=6079.0,
            ema_20=6078.0,
            pdh=6085.0,
            pdl=6070.0,
            llm_action="SCALP_SELL",
            llm_confidence=0.58,
            llm_reasoning="Attempted breakdown failed. Market reversed quickly - false signal in choppy conditions.",
            result="LOSS",
            duration_minutes=1.5,
            exit_reason="SL_HIT",
            time_of_day="MIDDAY",
            day_of_week="Monday",
        ),
        
        # Winning BUY - strong uptrend momentum
        TradeRecord(
            trade_id="test_007",
            timestamp=(base_time + timedelta(days=6)).isoformat(),
            action="BUY",
            entry_price=6120.00,
            exit_price=6132.50,
            quantity=1,
            stop_loss=6110.0,
            take_profit=6135.0,
            pnl=12.50 * 50,  # $625 profit
            pnl_pct=0.21,
            market_trend="UPTREND",
            volatility_regime="HIGH",
            rsi=58.0,
            macd_hist=0.45,
            atr=5.0,
            ema_9=6118.0,
            ema_20=6110.0,
            pdh=6100.0,
            pdl=6060.0,
            llm_action="BUY",
            llm_confidence=0.80,
            llm_reasoning="Strong uptrend continuation. Price above all EMAs, MACD strongly positive.",
            result="WIN",
            duration_minutes=15.0,
            exit_reason="TP_HIT",
            time_of_day="MIDDAY",
            day_of_week="Monday",
        ),
    ]
    
    return trades


def main():
    print("🔄 Seeding RAG storage with test data...")
    
    # Initialize storage manager
    storage = RAGStorageManager()
    
    # Create and save test trades
    trades = create_test_trades()
    
    for trade in trades:
        key = storage.save_trade(trade)
        print(f"  ✅ Saved trade {trade.trade_id}: {trade.action} {trade.result} (PnL: ${trade.pnl:.2f}) -> {key}")
    
    # Create a daily summary
    print("\n🔄 Creating daily summary...")
    summary_data = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "total_trades": len(trades),
        "wins": sum(1 for t in trades if t.result == "WIN"),
        "losses": sum(1 for t in trades if t.result == "LOSS"),
        "total_pnl": sum(t.pnl for t in trades),
        "avg_win": sum(t.pnl for t in trades if t.result == "WIN") / max(1, sum(1 for t in trades if t.result == "WIN")),
        "avg_loss": sum(t.pnl for t in trades if t.result == "LOSS") / max(1, sum(1 for t in trades if t.result == "LOSS")),
        "best_trade": max(trades, key=lambda t: t.pnl).trade_id,
        "worst_trade": min(trades, key=lambda t: t.pnl).trade_id,
        "market_conditions": "Mixed - trending and range-bound periods",
        "key_insights": [
            "PDL bounces worked well in oversold conditions",
            "PDH rejections provided good short entries",
            "Avoid scalps in choppy/range conditions without clear RSI extremes",
            "Trend continuation trades had best risk/reward"
        ]
    }
    
    summary_key = storage.save_daily_summary(summary_data)
    print(f"  ✅ Saved daily summary -> {summary_key}")
    
    # Verify data was saved
    print("\n🔍 Verifying S3 data...")
    s3 = S3Storage()
    
    # List trades
    trades_prefix = "spy-futures-bot/trades/"
    print(f"\nTrades in S3:")
    # Use boto3 to list objects
    import boto3
    s3_client = boto3.client('s3', region_name='us-east-1')
    response = s3_client.list_objects_v2(
        Bucket=s3.bucket,
        Prefix=trades_prefix,
        MaxKeys=10
    )
    
    if 'Contents' in response:
        for obj in response['Contents']:
            print(f"  📄 {obj['Key']} ({obj['Size']} bytes)")
    else:
        print("  ⚠️ No trades found")
    
    # List summaries
    summaries_prefix = "spy-futures-bot/daily_summaries/"
    response = s3_client.list_objects_v2(
        Bucket=s3.bucket,
        Prefix=summaries_prefix,
        MaxKeys=10
    )
    
    print(f"\nDaily Summaries in S3:")
    if 'Contents' in response:
        for obj in response['Contents']:
            print(f"  📄 {obj['Key']} ({obj['Size']} bytes)")
    else:
        print("  ⚠️ No summaries found")
    
    print("\n✅ RAG test data seeded successfully!")
    print(f"\nTotal test trades: {len(trades)}")
    print(f"Win rate: {sum(1 for t in trades if t.result == 'WIN') / len(trades) * 100:.1f}%")
    print(f"Total PnL: ${sum(t.pnl for t in trades):.2f}")


if __name__ == "__main__":
    main()
