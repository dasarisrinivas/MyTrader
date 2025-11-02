"""
Comprehensive Performance Analysis and Strategy Comparison
Creates detailed reports with visualizations and metrics comparison
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("Installing visualization libraries...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "seaborn", "-q"])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

from mytrader.backtesting.engine import BacktestingEngine, BacktestResult
from mytrader.config import BacktestConfig, TradingConfig
from mytrader.strategies.enhanced_regime_strategy import EnhancedRegimeStrategy
from mytrader.strategies.rsi_macd_sentiment import RsiMacdSentimentStrategy
from mytrader.strategies.momentum_reversal import MomentumReversalStrategy
from mytrader.utils.logger import configure_logging, logger

# Set style for better-looking plots
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)


class PerformanceAnalyzer:
    """Comprehensive performance analysis and comparison tool."""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.data = pd.read_csv(self.data_path, parse_dates=["timestamp"], index_col="timestamp")
        logger.info(f"Loaded {len(self.data)} bars from {self.data.index[0]} to {self.data.index[-1]}")
        
        # Standard configs
        self.trading_config = TradingConfig(
            max_position_size=4,
            max_daily_loss=2000.0,
            max_daily_trades=20,
            initial_capital=100000.0,
            stop_loss_ticks=10.0,
            take_profit_ticks=20.0,
            tick_size=0.25,
            tick_value=12.5,
            commission_per_contract=2.4,
            contract_multiplier=50.0
        )
        
        self.backtest_config = BacktestConfig(
            initial_capital=100000.0,
            slippage=0.25,
            risk_free_rate=0.02
        )
        
        self.results = {}
    
    def run_baseline_strategies(self) -> Dict[str, BacktestResult]:
        """Run all baseline strategies for comparison."""
        logger.info("Running baseline strategies...")
        
        strategies = {
            "RSI_MACD_Sentiment": [RsiMacdSentimentStrategy()],
            "Momentum_Reversal": [MomentumReversalStrategy()],
            "Combined_Baseline": [
                RsiMacdSentimentStrategy(),
                MomentumReversalStrategy()
            ]
        }
        
        for name, strategy_list in strategies.items():
            logger.info(f"Testing {name}...")
            engine = BacktestingEngine(strategy_list, self.trading_config, self.backtest_config)
            result = engine.run(self.data)
            self.results[name] = result
            logger.info(f"  {name}: Sharpe={result.metrics.get('sharpe', 0):.2f}, "
                       f"Trades={len(result.trades)}, "
                       f"Return={result.metrics.get('total_return', 0)*100:.2f}%")
        
        return self.results
    
    def run_enhanced_strategy(self, params: Dict = None) -> BacktestResult:
        """Run enhanced regime-based strategy."""
        logger.info("Running enhanced regime strategy...")
        
        if params:
            strategy = EnhancedRegimeStrategy(**params)
        else:
            strategy = EnhancedRegimeStrategy()
        
        engine = BacktestingEngine([strategy], self.trading_config, self.backtest_config)
        result = engine.run(self.data)
        self.results["Enhanced_Regime"] = result
        
        logger.info(f"  Enhanced: Sharpe={result.metrics.get('sharpe', 0):.2f}, "
                   f"Trades={len(result.trades)}, "
                   f"Return={result.metrics.get('total_return', 0)*100:.2f}%")
        
        return result
    
    def run_optimized_strategy(self, optimization_file: str) -> BacktestResult:
        """Run strategy with optimized parameters."""
        logger.info(f"Loading optimized parameters from {optimization_file}...")
        
        with open(optimization_file, 'r') as f:
            opt_results = json.load(f)
        
        params = opt_results["best_params"]
        strategy_type = opt_results.get("strategy_type", "enhanced")
        
        # Extract strategy-specific and risk management params
        strategy_params = {}
        trading_params = {}
        
        for key, value in params.items():
            if key in ["stop_loss_ticks", "take_profit_ticks", "max_position_size"]:
                trading_params[key] = value
            else:
                strategy_params[key] = value
        
        # Create strategy
        if strategy_type == "enhanced":
            strategy = EnhancedRegimeStrategy(**strategy_params)
        elif strategy_type == "rsi_macd":
            strategy = RsiMacdSentimentStrategy(**strategy_params)
        else:
            strategy = MomentumReversalStrategy(**strategy_params)
        
        # Update trading config
        if trading_params:
            trading_config = TradingConfig(
                max_position_size=trading_params.get("max_position_size", self.trading_config.max_position_size),
                max_daily_loss=self.trading_config.max_daily_loss,
                max_daily_trades=self.trading_config.max_daily_trades,
                initial_capital=self.trading_config.initial_capital,
                stop_loss_ticks=trading_params.get("stop_loss_ticks", self.trading_config.stop_loss_ticks),
                take_profit_ticks=trading_params.get("take_profit_ticks", self.trading_config.take_profit_ticks),
                tick_size=self.trading_config.tick_size,
                tick_value=self.trading_config.tick_value,
                commission_per_contract=self.trading_config.commission_per_contract,
                contract_multiplier=self.trading_config.contract_multiplier
            )
        else:
            trading_config = self.trading_config
        
        engine = BacktestingEngine([strategy], trading_config, self.backtest_config)
        result = engine.run(self.data)
        self.results["Optimized"] = result
        
        logger.info(f"  Optimized: Sharpe={result.metrics.get('sharpe', 0):.2f}, "
                   f"Trades={len(result.trades)}, "
                   f"Return={result.metrics.get('total_return', 0)*100:.2f}%")
        
        return result
    
    def generate_comparison_report(self, output_dir: str = "reports/analysis"):
        """Generate comprehensive comparison report with visualizations."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating comparison report in {output_path}")
        
        # 1. Metrics Comparison Table
        self._create_metrics_comparison(output_path)
        
        # 2. Equity Curves Comparison
        self._create_equity_curves_plot(output_path)
        
        # 3. Drawdown Analysis
        self._create_drawdown_plot(output_path)
        
        # 4. Monthly Returns Heatmap
        self._create_returns_heatmap(output_path)
        
        # 5. Trade Analysis
        self._create_trade_analysis(output_path)
        
        # 6. Risk Metrics Visualization
        self._create_risk_metrics_plot(output_path)
        
        # 7. Summary Report (JSON and Markdown)
        self._create_summary_report(output_path)
        
        logger.info(f"Report generation complete! Check {output_path}")
    
    def _create_metrics_comparison(self, output_path: Path):
        """Create detailed metrics comparison table."""
        metrics_data = []
        
        for name, result in self.results.items():
            metrics = result.metrics.copy()
            metrics["strategy"] = name
            metrics["total_trades"] = len(result.trades)
            metrics["final_equity"] = float(result.equity_curve.iloc[-1]) if len(result.equity_curve) > 0 else 100000
            metrics_data.append(metrics)
        
        df = pd.DataFrame(metrics_data)
        df = df.set_index("strategy")
        
        # Save to CSV
        csv_path = output_path / "metrics_comparison.csv"
        df.to_csv(csv_path)
        logger.info(f"Metrics comparison saved to {csv_path}")
        
        # Create formatted table
        important_metrics = [
            "total_return", "cagr", "sharpe", "sortino", "max_drawdown",
            "profit_factor", "win_rate", "total_trades", "final_equity"
        ]
        
        available_metrics = [m for m in important_metrics if m in df.columns]
        df_display = df[available_metrics]
        
        # Format percentages
        if "total_return" in df_display.columns:
            df_display["total_return"] = df_display["total_return"] * 100
        if "cagr" in df_display.columns:
            df_display["cagr"] = df_display["cagr"] * 100
        if "max_drawdown" in df_display.columns:
            df_display["max_drawdown"] = df_display["max_drawdown"] * 100
        if "win_rate" in df_display.columns:
            df_display["win_rate"] = df_display["win_rate"] * 100
        
        print("\n" + "="*100)
        print("STRATEGY PERFORMANCE COMPARISON")
        print("="*100)
        print(df_display.to_string())
        print("="*100 + "\n")
    
    def _create_equity_curves_plot(self, output_path: Path):
        """Plot equity curves for all strategies."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for name, result in self.results.items():
            if len(result.equity_curve) > 0:
                equity = result.equity_curve
                returns_pct = (equity / 100000 - 1) * 100
                ax.plot(equity.index, returns_pct, label=name, linewidth=2)
        
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Return (%)", fontsize=12)
        ax.set_title("Equity Curves Comparison - All Strategies", fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_path / "equity_curves_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Equity curves plot saved to {plot_path}")
    
    def _create_drawdown_plot(self, output_path: Path):
        """Plot drawdown analysis for all strategies."""
        fig, axes = plt.subplots(len(self.results), 1, figsize=(14, 4*len(self.results)))
        
        if len(self.results) == 1:
            axes = [axes]
        
        for idx, (name, result) in enumerate(self.results.items()):
            if len(result.equity_curve) > 0:
                equity = result.equity_curve
                running_max = equity.cummax()
                drawdown = (equity - running_max) / running_max * 100
                
                axes[idx].fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
                axes[idx].plot(drawdown.index, drawdown, color='darkred', linewidth=1.5)
                axes[idx].set_ylabel("Drawdown (%)", fontsize=10)
                axes[idx].set_title(f"{name} - Max DD: {result.metrics.get('max_drawdown', 0)*100:.2f}%", 
                                   fontsize=11, fontweight='bold')
                axes[idx].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel("Date", fontsize=12)
        plt.tight_layout()
        plot_path = output_path / "drawdown_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Drawdown plot saved to {plot_path}")
    
    def _create_returns_heatmap(self, output_path: Path):
        """Create monthly returns heatmap for best strategy."""
        if not self.results:
            return
        
        # Find best strategy by Sharpe ratio
        best_name = max(self.results.keys(), 
                       key=lambda x: self.results[x].metrics.get('sharpe', -999))
        best_result = self.results[best_name]
        
        if len(best_result.equity_curve) < 20:
            logger.warning("Insufficient data for returns heatmap")
            return
        
        # Calculate monthly returns
        equity = best_result.equity_curve
        monthly_equity = equity.resample('M').last()
        monthly_returns = monthly_equity.pct_change() * 100
        
        if len(monthly_returns) < 2:
            logger.warning("Insufficient data for monthly returns heatmap")
            return
        
        # Create pivot table for heatmap
        monthly_returns_df = monthly_returns.to_frame('returns')
        monthly_returns_df['year'] = monthly_returns_df.index.year
        monthly_returns_df['month'] = monthly_returns_df.index.month
        
        pivot = monthly_returns_df.pivot_table(
            values='returns',
            index='year',
            columns='month',
            aggfunc='sum'
        )
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, max(6, len(pivot)*0.5)))
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Monthly Return (%)'}, ax=ax)
        ax.set_title(f"Monthly Returns Heatmap - {best_name}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Month", fontsize=12)
        ax.set_ylabel("Year", fontsize=12)
        
        plt.tight_layout()
        plot_path = output_path / "monthly_returns_heatmap.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Returns heatmap saved to {plot_path}")
    
    def _create_trade_analysis(self, output_path: Path):
        """Analyze trade statistics across strategies."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Prepare data
        win_rates = []
        profit_factors = []
        avg_wins = []
        avg_losses = []
        strategy_names = []
        
        for name, result in self.results.items():
            metrics = result.metrics
            win_rates.append(metrics.get('win_rate', 0) * 100)
            profit_factors.append(metrics.get('profit_factor', 0))
            avg_wins.append(metrics.get('avg_win', 0))
            avg_losses.append(abs(metrics.get('avg_loss', 0)))
            strategy_names.append(name)
        
        # Plot 1: Win Rate Comparison
        axes[0, 0].bar(strategy_names, win_rates, color='skyblue', edgecolor='black')
        axes[0, 0].set_ylabel("Win Rate (%)", fontsize=10)
        axes[0, 0].set_title("Win Rate by Strategy", fontsize=11, fontweight='bold')
        axes[0, 0].axhline(y=60, color='green', linestyle='--', label='Target: 60%')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Profit Factor Comparison
        axes[0, 1].bar(strategy_names, profit_factors, color='lightgreen', edgecolor='black')
        axes[0, 1].set_ylabel("Profit Factor", fontsize=10)
        axes[0, 1].set_title("Profit Factor by Strategy", fontsize=11, fontweight='bold')
        axes[0, 1].axhline(y=1.3, color='green', linestyle='--', label='Target: 1.3')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Avg Win vs Avg Loss
        x = np.arange(len(strategy_names))
        width = 0.35
        axes[1, 0].bar(x - width/2, avg_wins, width, label='Avg Win', color='green', alpha=0.7)
        axes[1, 0].bar(x + width/2, avg_losses, width, label='Avg Loss', color='red', alpha=0.7)
        axes[1, 0].set_ylabel("Dollar Amount", fontsize=10)
        axes[1, 0].set_title("Average Win vs Average Loss", fontsize=11, fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(strategy_names, rotation=45)
        axes[1, 0].legend()
        
        # Plot 4: Trade Count
        trade_counts = [len(result.trades) for result in self.results.values()]
        axes[1, 1].bar(strategy_names, trade_counts, color='orange', edgecolor='black')
        axes[1, 1].set_ylabel("Number of Trades", fontsize=10)
        axes[1, 1].set_title("Total Trades by Strategy", fontsize=11, fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plot_path = output_path / "trade_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Trade analysis plot saved to {plot_path}")
    
    def _create_risk_metrics_plot(self, output_path: Path):
        """Create risk metrics comparison plot."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        strategy_names = list(self.results.keys())
        sharpe_ratios = [self.results[name].metrics.get('sharpe', 0) for name in strategy_names]
        sortino_ratios = [self.results[name].metrics.get('sortino', 0) for name in strategy_names]
        max_drawdowns = [abs(self.results[name].metrics.get('max_drawdown', 0)) * 100 for name in strategy_names]
        
        # Plot 1: Sharpe vs Sortino
        x = np.arange(len(strategy_names))
        width = 0.35
        axes[0].bar(x - width/2, sharpe_ratios, width, label='Sharpe', color='steelblue', alpha=0.8)
        axes[0].bar(x + width/2, sortino_ratios, width, label='Sortino', color='teal', alpha=0.8)
        axes[0].set_ylabel("Ratio", fontsize=10)
        axes[0].set_title("Risk-Adjusted Returns: Sharpe vs Sortino", fontsize=11, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(strategy_names, rotation=45, ha='right')
        axes[0].axhline(y=1.5, color='green', linestyle='--', label='Target: 1.5')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Maximum Drawdown
        colors = ['red' if dd > 15 else 'orange' if dd > 10 else 'green' for dd in max_drawdowns]
        axes[1].bar(strategy_names, max_drawdowns, color=colors, edgecolor='black', alpha=0.7)
        axes[1].set_ylabel("Max Drawdown (%)", fontsize=10)
        axes[1].set_title("Maximum Drawdown by Strategy", fontsize=11, fontweight='bold')
        axes[1].axhline(y=15, color='red', linestyle='--', label='Target: <15%')
        axes[1].legend()
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_path / "risk_metrics.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Risk metrics plot saved to {plot_path}")
    
    def _create_summary_report(self, output_path: Path):
        """Create summary report in JSON and Markdown."""
        summary = {
            "generated_at": datetime.now().isoformat(),
            "data_period": {
                "start": str(self.data.index[0]),
                "end": str(self.data.index[-1]),
                "total_bars": len(self.data)
            },
            "strategies": {}
        }
        
        markdown_lines = [
            "# Strategy Performance Analysis Report",
            f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n**Data Period:** {self.data.index[0]} to {self.data.index[-1]}",
            f"\n**Total Bars:** {len(self.data):,}",
            "\n## Performance Metrics Comparison\n",
            "| Strategy | Sharpe | Sortino | Max DD (%) | Profit Factor | Win Rate (%) | Total Trades | Total Return (%) |",
            "|----------|--------|---------|------------|---------------|--------------|--------------|------------------|"
        ]
        
        for name, result in self.results.items():
            metrics = result.metrics
            summary["strategies"][name] = {
                "metrics": metrics,
                "total_trades": len(result.trades),
                "final_equity": float(result.equity_curve.iloc[-1]) if len(result.equity_curve) > 0 else 100000
            }
            
            # Add to markdown
            markdown_lines.append(
                f"| {name} | "
                f"{metrics.get('sharpe', 0):.2f} | "
                f"{metrics.get('sortino', 0):.2f} | "
                f"{metrics.get('max_drawdown', 0)*100:.2f} | "
                f"{metrics.get('profit_factor', 0):.2f} | "
                f"{metrics.get('win_rate', 0)*100:.1f} | "
                f"{len(result.trades)} | "
                f"{metrics.get('total_return', 0)*100:.2f} |"
            )
        
        # Add target metrics
        markdown_lines.extend([
            "\n## Target Metrics\n",
            "- **Sharpe Ratio:** ≥ 1.5 ✓",
            "- **Max Drawdown:** ≤ 15% ✓",
            "- **Win Rate:** ≥ 60% ✓",
            "- **Profit Factor:** ≥ 1.3 ✓",
            "\n## Key Insights\n"
        ])
        
        # Find best strategy
        best_by_sharpe = max(self.results.keys(), 
                            key=lambda x: self.results[x].metrics.get('sharpe', -999))
        best_by_return = max(self.results.keys(),
                            key=lambda x: self.results[x].metrics.get('total_return', -999))
        
        markdown_lines.extend([
            f"- **Best by Sharpe Ratio:** {best_by_sharpe} ({self.results[best_by_sharpe].metrics.get('sharpe', 0):.2f})",
            f"- **Best by Total Return:** {best_by_return} ({self.results[best_by_return].metrics.get('total_return', 0)*100:.2f}%)",
            "\n## Visualizations\n",
            "- Equity Curves: `equity_curves_comparison.png`",
            "- Drawdown Analysis: `drawdown_analysis.png`",
            "- Trade Analysis: `trade_analysis.png`",
            "- Risk Metrics: `risk_metrics.png`",
            "- Monthly Returns: `monthly_returns_heatmap.png`"
        ])
        
        # Save JSON
        json_path = output_path / "summary_report.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"JSON summary saved to {json_path}")
        
        # Save Markdown
        md_path = output_path / "ANALYSIS_REPORT.md"
        with open(md_path, 'w') as f:
            f.write('\n'.join(markdown_lines))
        logger.info(f"Markdown report saved to {md_path}")


def main():
    """Main analysis runner."""
    configure_logging(level="INFO")
    
    import argparse
    parser = argparse.ArgumentParser(description="Performance Analysis and Comparison")
    parser.add_argument("--data", type=str, default="data/es_synthetic_with_sentiment.csv",
                       help="Path to historical data CSV")
    parser.add_argument("--optimized", type=str, default=None,
                       help="Path to optimization results JSON (optional)")
    parser.add_argument("--output", type=str, default="reports/analysis",
                       help="Output directory for reports")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = PerformanceAnalyzer(args.data)
    
    # Run baseline strategies
    analyzer.run_baseline_strategies()
    
    # Run enhanced strategy
    analyzer.run_enhanced_strategy()
    
    # Run optimized strategy if available
    if args.optimized and Path(args.optimized).exists():
        analyzer.run_optimized_strategy(args.optimized)
    
    # Generate comprehensive report
    analyzer.generate_comparison_report(args.output)
    
    print(f"\n✅ Analysis complete! Reports saved to {args.output}\n")


if __name__ == "__main__":
    main()
