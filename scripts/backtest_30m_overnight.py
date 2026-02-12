#!/usr/bin/env python3
"""
Run 30m-only backtest for overnight trading strategy.

This uses native 30m data directly (not resampled from 1m) to test
overnight trading when 1m data quality is poor.

Usage:
    python scripts/backtest_30m_overnight.py
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.engine import BacktestEngine, BacktestConfig
from backtest.analysis import BacktestAnalyzer
from backtest.report import ReportGenerator, ReportConfig
from shree.config import OneMinuteStrategyConfig, RiskGateConfig, TradingConfig
from loguru import logger


def run_30m_overnight_backtest():
    """Run backtest using only 30m data for overnight trading."""
    
    # Load native 30m data
    data_path = Path("data/raw/ES/ES_30min_60D.parquet")
    if not data_path.exists():
        logger.error(f"30m data not found: {data_path}")
        logger.info("Run: python scripts/download_30m_data.py")
        return None
    
    df_30m = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df_30m)} 30m bars")
    logger.info(f"Date range: {df_30m.index.min()} to {df_30m.index.max()}")
    
    # Check data quality
    zero_range = (df_30m['high'] == df_30m['low']).sum()
    logger.info(f"Zero-range bars: {zero_range} ({zero_range/len(df_30m)*100:.1f}%)")
    
    # Configure strategy for 30m bars
    # 30m ATR is typically 6-10 pts vs 1-2 pts for 1m
    # Adjust all parameters accordingly
    strategy_config = OneMinuteStrategyConfig(
        warmup_bars=30,  # 30 bars * 30min = 15 hours warmup
        rth_only=False,  # Allow overnight trading
        allow_overnight_trading=True,
        stop_atr_multiplier=1.5,  # Reduced from 6.5 (30m ATR is already larger)
        take_profit_multiple=2.0,  # 2R target
        trend_adx_threshold=20.0,
        breakout_adx_threshold=18.0,
        cooldown_minutes=60,  # 2 bars cooldown
        max_trades_per_hour=2,
        max_trades_per_day=6,
        max_hold_minutes=240,  # 4 hours max hold (8 bars)
        # Disable filters that were tuned for 1m data
        min_bar_volume=0,  # Volume filter tuned for 1m (disable for 30m)
        require_high_atr=False,  # ATR regime filter was tuned for 1m
        high_atr_percentile=0.90,  # If enabled, only block very high ATR
        # MTF doesn't make sense for 30m-only data
        use_mtf_regime=False,
        require_5m_trend_alignment=False,
    )
    
    # RiskGate config for 30m - much wider stops allowed
    # 30m median ATR = 6.46, 67th = 9.34, so 1.5x = 14 pts typical stop
    risk_gate_config = RiskGateConfig(
        min_stop_points=6.0,   # Min 6 pts stop (reasonable for 30m)
        max_stop_points=25.0,  # Allow up to 25 pts stop for 30m
        risk_per_trade_usd=75.0,  # $75 max risk = 15 pts stop at $5/pt
    )
    
    trading_config = TradingConfig()
    
    # Set date range from data
    start_date = df_30m.index.min()
    end_date = df_30m.index.max()
    
    bt_config = BacktestConfig(
        symbol="ES",
        start_date=start_date,
        end_date=end_date,
        initial_capital=5000,
        slippage_ticks=1,
        commission_per_contract=2.4,
        session_type="full",  # Full session for overnight
        strategy_config=strategy_config,
        trading_config=trading_config,
        risk_gate_config=risk_gate_config,
        warmup_bars=30,
    )
    
    # Create and run engine
    engine = BacktestEngine(bt_config)
    engine.load_30m_only(df_30m)
    
    logger.info("Running 30m overnight backtest...")
    results = engine.run_30m_only()
    
    # Analyze results
    trades = results.get("trades", [])
    logger.info(f"\nTotal trades: {len(trades)}")
    
    if trades:
        # Simple manual analysis
        trades_df = pd.DataFrame(trades)
        print("\n" + "="*60)
        print("30M OVERNIGHT BACKTEST RESULTS")
        print("="*60)
        
        # Check available columns
        pnl_col = 'pnl' if 'pnl' in trades_df.columns else 'realized_pnl'
        
        # Calculate basic stats from trades
        total_trades = len(trades)
        if pnl_col in trades_df.columns:
            winners = trades_df[trades_df[pnl_col] > 0]
            losers = trades_df[trades_df[pnl_col] < 0]
            win_rate = len(winners) / total_trades if total_trades > 0 else 0
            total_pnl = trades_df[pnl_col].sum()
            gross_profit = winners[pnl_col].sum() if len(winners) > 0 else 0
            gross_loss = abs(losers[pnl_col].sum()) if len(losers) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            avg_win = winners[pnl_col].mean() if len(winners) > 0 else 0
            avg_loss = losers[pnl_col].mean() if len(losers) > 0 else 0
            
            print(f"Total Trades:    {total_trades}")
            print(f"Winners:         {len(winners)}")
            print(f"Losers:          {len(losers)}")
            print(f"Win Rate:        {win_rate*100:.1f}%")
            print(f"Total P&L:       ${total_pnl:.2f}")
            print(f"Gross Profit:    ${gross_profit:.2f}")
            print(f"Gross Loss:      ${gross_loss:.2f}")
            print(f"Profit Factor:   {profit_factor:.2f}")
            print(f"Avg Win:         ${avg_win:.2f}")
            print(f"Avg Loss:        ${avg_loss:.2f}")
        else:
            print(f"Total Trades: {total_trades}")
            print("Trade columns:", trades_df.columns.tolist())
        print("="*60)
        
        # Show sample trades
        print("\nSample trades:")
        print(trades_df.head(10).to_string())
        
        # Show block reasons
        block_reasons = results.get("block_reasons", {})
        if block_reasons:
            print("\nBlock reasons (top 10):")
            for reason, count in sorted(block_reasons.items(), key=lambda x: -x[1])[:10]:
                print(f"  {reason}: {count}")
    else:
        logger.warning("No trades generated!")
        
        # Show block reasons
        block_reasons = results.get("block_reasons", {})
        if block_reasons:
            print("\nBlock reasons:")
            for reason, count in sorted(block_reasons.items(), key=lambda x: -x[1])[:10]:
                print(f"  {reason}: {count}")
    
    return results


if __name__ == "__main__":
    results = run_30m_overnight_backtest()
