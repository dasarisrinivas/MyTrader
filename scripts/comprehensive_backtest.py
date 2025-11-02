#!/usr/bin/env python3
"""Comprehensive backtest with before/after comparison and visualization."""

import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from mytrader.backtesting.engine import BacktestingEngine
from mytrader.config import BacktestConfig, TradingConfig
from mytrader.strategies.rsi_macd_sentiment import RsiMacdSentimentStrategy
from mytrader.strategies.momentum_reversal import MomentumReversalStrategy
from mytrader.utils.logger import configure_logging, logger


def run_backtest(data_path: Path, strategy_config: dict, label: str) -> dict:
    """Run a single backtest with given configuration."""
    # Load data
    df = pd.read_csv(data_path, parse_dates=["timestamp"], index_col="timestamp")
    
    # Setup configuration
    trading_config = TradingConfig(
        initial_capital=100000.0,
        max_position_size=3,  # Reduced from 4 for better risk management
        max_daily_loss=2000.0,
        max_daily_trades=20,
        stop_loss_ticks=15.0,  # Increased from 10 for less whipsaw
        take_profit_ticks=30.0,  # Increased from 20 for better R:R ratio (2:1)
        tick_size=0.25,
        tick_value=12.5,
        commission_per_contract=2.4,
        contract_multiplier=50.0
    )
    
    backtest_config = BacktestConfig(
        initial_capital=100000.0,
        slippage=0.25,
        risk_free_rate=0.02
    )
    
    # Initialize strategies with provided parameters
    strategies = [
        RsiMacdSentimentStrategy(**strategy_config),
        MomentumReversalStrategy()
    ]
    
    # Run backtest
    engine = BacktestingEngine(strategies, trading_config, backtest_config)
    result = engine.run(df)
    
    return {
        "label": label,
        "metrics": result.metrics,
        "trades": result.trades,
        "equity_curve": result.equity_curve,
        "config": strategy_config
    }


def create_comparison_report(baseline: dict, enhanced: dict, output_dir: Path):
    """Create comprehensive comparison report."""
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Print to console
    print("\n" + "=" * 80)
    print("BEFORE/AFTER COMPARISON REPORT")
    print("=" * 80 + "\n")
    
    print(f"{'Metric':<30} {'BASELINE':>15} {'ENHANCED':>15} {'CHANGE':>15}")
    print("-" * 80)
    
    metrics_to_compare = [
        ('total_return', '%', 100),
        ('cagr', '%', 100),
        ('sharpe', '', 1),
        ('sortino', '', 1),
        ('max_drawdown', '%', 100),
        ('profit_factor', '', 1),
        ('calmar_ratio', '', 1),
        ('total_trades', '', 1),
        ('win_rate', '%', 100),
        ('avg_win', '$', 1),
        ('avg_loss', '$', 1),
        ('expectancy', '$', 1),
    ]
    
    comparison_data = []
    
    for metric, unit, multiplier in metrics_to_compare:
        baseline_val = baseline['metrics'].get(metric, 0)
        enhanced_val = enhanced['metrics'].get(metric, 0)
        
        if baseline_val != 0:
            change_pct = ((enhanced_val - baseline_val) / abs(baseline_val)) * 100
            change_str = f"{change_pct:+.1f}%"
        else:
            change_str = "N/A"
        
        if unit == '%':
            baseline_str = f"{baseline_val * multiplier:.2f}%"
            enhanced_str = f"{enhanced_val * multiplier:.2f}%"
        elif unit == '$':
            baseline_str = f"${baseline_val:.2f}"
            enhanced_str = f"${enhanced_val:.2f}"
        else:
            baseline_str = f"{baseline_val:.2f}" if isinstance(baseline_val, float) else str(baseline_val)
            enhanced_str = f"{enhanced_val:.2f}" if isinstance(enhanced_val, float) else str(enhanced_val)
        
        print(f"{metric.replace('_', ' ').title():<30} {baseline_str:>15} {enhanced_str:>15} {change_str:>15}")
        
        comparison_data.append({
            'Metric': metric.replace('_', ' ').title(),
            'Baseline': baseline_str,
            'Enhanced': enhanced_str,
            'Change': change_str
        })
    
    # Save comparison to CSV
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_dir / 'comparison_report.csv', index=False)
    print(f"\n📊 Comparison saved to: {output_dir / 'comparison_report.csv'}")
    
    # Create visualizations
    create_visualizations(baseline, enhanced, output_dir)


def create_visualizations(baseline: dict, enhanced: dict, output_dir: Path):
    """Create comparison visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Equity curves
    ax = axes[0, 0]
    if not baseline['equity_curve'].empty:
        ax.plot(baseline['equity_curve'].values, label='Baseline', alpha=0.7)
    if not enhanced['equity_curve'].empty:
        ax.plot(enhanced['equity_curve'].values, label='Enhanced', alpha=0.7)
    ax.set_title('Equity Curve Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Account Value ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Key metrics comparison
    ax = axes[0, 1]
    metrics = ['total_return', 'sharpe', 'win_rate', 'profit_factor']
    baseline_vals = [baseline['metrics'].get(m, 0) for m in metrics]
    enhanced_vals = [enhanced['metrics'].get(m, 0) for m in metrics]
    
    x = range(len(metrics))
    width = 0.35
    ax.bar([i - width/2 for i in x], baseline_vals, width, label='Baseline', alpha=0.7)
    ax.bar([i + width/2 for i in x], enhanced_vals, width, label='Enhanced', alpha=0.7)
    ax.set_title('Key Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n') for m in metrics], fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Drawdown comparison
    ax = axes[1, 0]
    if not baseline['equity_curve'].empty:
        baseline_dd = (baseline['equity_curve'] / baseline['equity_curve'].cummax() - 1) * 100
        ax.plot(baseline_dd.values, label='Baseline', alpha=0.7)
    if not enhanced['equity_curve'].empty:
        enhanced_dd = (enhanced['equity_curve'] / enhanced['equity_curve'].cummax() - 1) * 100
        ax.plot(enhanced_dd.values, label='Enhanced', alpha=0.7)
    ax.set_title('Drawdown Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Drawdown (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # 4. Trade distribution
    ax = axes[1, 1]
    if baseline['trades'] and enhanced['trades']:
        baseline_pnls = [t.get('realized', 0) for t in baseline['trades'] if 'realized' in t]
        enhanced_pnls = [t.get('realized', 0) for t in enhanced['trades'] if 'realized' in t]
        
        ax.hist(baseline_pnls, bins=20, alpha=0.5, label='Baseline', edgecolor='black')
        ax.hist(enhanced_pnls, bins=20, alpha=0.5, label='Enhanced', edgecolor='black')
        ax.set_title('Trade PnL Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('PnL ($)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    chart_path = output_dir / 'performance_comparison.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"📈 Visualizations saved to: {chart_path}")
    plt.close()


def main():
    """Run comprehensive backtest comparison."""
    configure_logging(level="INFO")
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE BACKTEST COMPARISON")
    print("=" * 80 + "\n")
    
    data_path = Path("data/es_comprehensive_data.csv")
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print("   Run: python scripts/generate_comprehensive_data.py")
        return
    
    # Baseline configuration (conservative)
    print("🔄 Running BASELINE backtest...")
    baseline_config = {
        "rsi_buy": 30.0,
        "rsi_sell": 70.0,
        "sentiment_buy": 0.6,
        "sentiment_sell": 0.4,
        "use_macd_crossover": False
    }
    baseline = run_backtest(data_path, baseline_config, "Baseline")
    
    # Enhanced configuration (optimized)
    print("🔄 Running ENHANCED backtest...")
    enhanced_config = {
        "rsi_buy": 40.0,
        "rsi_sell": 60.0,
        "sentiment_buy": -0.3,
        "sentiment_sell": 0.3,
        "use_macd_crossover": False  # Direct MACD signal
    }
    enhanced = run_backtest(data_path, enhanced_config, "Enhanced")
    
    # Generate comparison report
    output_dir = Path("reports")
    create_comparison_report(baseline, enhanced, output_dir)
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ BACKTEST COMPARISON COMPLETE")
    print("=" * 80)
    print(f"\nBaseline: {baseline['metrics'].get('total_trades', 0)} trades, "
          f"{baseline['metrics'].get('total_return', 0) * 100:.2f}% return")
    print(f"Enhanced: {enhanced['metrics'].get('total_trades', 0)} trades, "
          f"{enhanced['metrics'].get('total_return', 0) * 100:.2f}% return")
    
    # Improvement summary
    baseline_return = baseline['metrics'].get('total_return', 0)
    enhanced_return = enhanced['metrics'].get('total_return', 0)
    if baseline_return != 0:
        improvement = ((enhanced_return - baseline_return) / abs(baseline_return)) * 100
        print(f"\n{'🎉 IMPROVEMENT' if improvement > 0 else '⚠️  DECLINE'}: {improvement:+.1f}%")


if __name__ == "__main__":
    main()
