#!/usr/bin/env python3
"""Analyze LLM trading performance and generate insights."""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mytrader.config import Settings
from mytrader.llm.trade_logger import TradeLogger
from mytrader.utils.logger import configure_logging, logger
import pandas as pd


async def analyze_llm_performance(days: int = 1):
    """Analyze LLM performance over recent trading days."""
    
    configure_logging(level="INFO")
    settings = Settings()
    
    logger.info(f"📊 Analyzing LLM performance for last {days} day(s)")
    
    # Get trade logger
    trade_logger = TradeLogger()
    
    # Get recent trades
    trades = trade_logger.get_recent_trades(limit=1000)
    
    if not trades:
        logger.warning("No trades found!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(trades)
    
    # Filter to last N days
    cutoff_time = datetime.now() - timedelta(days=days)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[df['timestamp'] >= cutoff_time]
    
    if df.empty:
        logger.warning(f"No trades in last {days} day(s)")
        return
    
    logger.info(f"\n{'='*60}")
    logger.info(f"LLM TRADING PERFORMANCE - Last {days} Day(s)")
    logger.info(f"{'='*60}")
    
    # Overall stats
    total_trades = len(df)
    wins = len(df[df['realized_pnl'] > 0])
    losses = len(df[df['realized_pnl'] < 0])
    breakeven = len(df[df['realized_pnl'] == 0])
    
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    total_pnl = df['realized_pnl'].sum()
    avg_pnl = df['realized_pnl'].mean()
    
    logger.info(f"\n📈 OVERALL STATISTICS:")
    logger.info(f"  Total Trades: {total_trades}")
    logger.info(f"  Wins: {wins} ({win_rate:.1f}%)")
    logger.info(f"  Losses: {losses}")
    logger.info(f"  Breakeven: {breakeven}")
    logger.info(f"  Total P&L: ${total_pnl:,.2f}")
    logger.info(f"  Average P&L: ${avg_pnl:.2f}")
    
    # LLM vs Traditional comparison
    llm_trades = df[df['trade_decision'].isin(['BUY', 'SELL', 'HOLD'])]
    if not llm_trades.empty:
        logger.info(f"\n🤖 LLM RECOMMENDATIONS:")
        
        for decision in ['BUY', 'SELL', 'HOLD']:
            decision_trades = llm_trades[llm_trades['trade_decision'] == decision]
            if not decision_trades.empty:
                dec_wins = len(decision_trades[decision_trades['realized_pnl'] > 0])
                dec_total = len(decision_trades)
                dec_winrate = (dec_wins / dec_total * 100) if dec_total > 0 else 0
                dec_pnl = decision_trades['realized_pnl'].sum()
                
                logger.info(f"  {decision}:")
                logger.info(f"    Trades: {dec_total}")
                logger.info(f"    Win Rate: {dec_winrate:.1f}%")
                logger.info(f"    Total P&L: ${dec_pnl:,.2f}")
    
    # Confidence analysis
    if 'confidence' in df.columns:
        logger.info(f"\n🎯 CONFIDENCE ANALYSIS:")
        
        high_conf = df[df['confidence'] >= 0.7]
        med_conf = df[(df['confidence'] >= 0.55) & (df['confidence'] < 0.7)]
        low_conf = df[df['confidence'] < 0.55]
        
        for name, subset in [('High (≥0.7)', high_conf), 
                            ('Medium (0.55-0.7)', med_conf), 
                            ('Low (<0.55)', low_conf)]:
            if not subset.empty:
                wins = len(subset[subset['realized_pnl'] > 0])
                total = len(subset)
                winrate = (wins / total * 100) if total > 0 else 0
                pnl = subset['realized_pnl'].sum()
                
                logger.info(f"  {name}:")
                logger.info(f"    Trades: {total}")
                logger.info(f"    Win Rate: {winrate:.1f}%")
                logger.info(f"    Total P&L: ${pnl:,.2f}")
    
    # Best and worst trades
    logger.info(f"\n💰 TOP TRADES:")
    best_trades = df.nlargest(3, 'realized_pnl')
    for idx, trade in best_trades.iterrows():
        logger.info(f"  {trade['timestamp']}: {trade['trade_decision']} "
                   f"@ ${trade['entry_price']:.2f} → ${trade['exit_price']:.2f} "
                   f"= ${trade['realized_pnl']:,.2f}")
    
    logger.info(f"\n❌ WORST TRADES:")
    worst_trades = df.nsmallest(3, 'realized_pnl')
    for idx, trade in worst_trades.iterrows():
        logger.info(f"  {trade['timestamp']}: {trade['trade_decision']} "
                   f"@ ${trade['entry_price']:.2f} → ${trade['exit_price']:.2f} "
                   f"= ${trade['realized_pnl']:,.2f}")
    
    # Recommendations
    logger.info(f"\n💡 RECOMMENDATIONS:")
    
    if win_rate < 50:
        logger.info("  ⚠️  Win rate below 50% - Consider adjusting confidence threshold")
    
    if 'confidence' in df.columns and not high_conf.empty:
        high_conf_winrate = (len(high_conf[high_conf['realized_pnl'] > 0]) / len(high_conf) * 100)
        if high_conf_winrate > 60:
            logger.info(f"  ✅ High confidence trades performing well ({high_conf_winrate:.1f}% win rate)")
            logger.info("  💡 Consider only taking trades with confidence ≥ 0.7")
    
    if total_pnl < 0:
        logger.info("  ⚠️  Negative P&L - Review entry/exit conditions")
    
    logger.info(f"\n{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze LLM trading performance")
    parser.add_argument("--days", type=int, default=1, help="Number of days to analyze")
    
    args = parser.parse_args()
    
    asyncio.run(analyze_llm_performance(days=args.days))
