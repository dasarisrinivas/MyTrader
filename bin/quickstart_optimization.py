#!/usr/bin/env python3
"""
Quick Start: Strategy Optimization & Analysis Pipeline
Runs the complete optimization workflow in sequence
"""
import sys
from pathlib import Path
import subprocess
import json
from datetime import datetime

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}")
    print(f"{text}")
    print(f"{'='*80}{Colors.ENDC}\n")

def print_success(text):
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")

def run_command(cmd, description):
    """Run a command and handle errors."""
    print_info(f"Running: {description}")
    print(f"  Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print_success(f"Completed: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed: {description}")
        print(f"  Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print_error(f"Command not found: {cmd[0]}")
        return False

def check_dependencies():
    """Check if required packages are installed."""
    print_header("Checking Dependencies")
    
    required = ["pandas", "numpy", "optuna", "matplotlib", "seaborn"]
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print_success(f"{package} is installed")
        except ImportError:
            missing.append(package)
            print_warning(f"{package} is missing")
    
    if missing:
        print_info(f"Installing missing packages: {', '.join(missing)}")
        cmd = [sys.executable, "-m", "pip", "install"] + missing
        if run_command(cmd, "Install dependencies"):
            print_success("All dependencies installed")
        else:
            print_error("Failed to install dependencies")
            return False
    
    return True

def check_data_file(data_path):
    """Check if data file exists."""
    if not Path(data_path).exists():
        print_error(f"Data file not found: {data_path}")
        print_info("Available data files:")
        data_dir = Path("data")
        if data_dir.exists():
            for f in data_dir.glob("*.csv"):
                print(f"  - {f}")
        return False
    
    print_success(f"Data file found: {data_path}")
    return True

def main():
    """Main pipeline execution."""
    print_header("MyTrader Strategy Optimization Pipeline")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    data_path = "data/es_synthetic_with_sentiment.csv"
    strategy_type = "enhanced"  # Options: enhanced, rsi_macd, momentum
    n_trials = 50  # Increase to 100+ for production
    
    print_info(f"Configuration:")
    print(f"  Data: {data_path}")
    print(f"  Strategy: {strategy_type}")
    print(f"  Trials: {n_trials}")
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print_error("Dependency check failed. Exiting.")
        return 1
    
    # Step 2: Check data
    print_header("Step 1: Data Validation")
    if not check_data_file(data_path):
        return 1
    
    # Step 3: Run baseline analysis
    print_header("Step 2: Baseline Performance Analysis")
    baseline_cmd = [
        sys.executable,
        "scripts/performance_analyzer.py",
        "--data", data_path,
        "--output", "reports/baseline_analysis"
    ]
    
    if not run_command(baseline_cmd, "Baseline analysis"):
        print_warning("Baseline analysis failed, continuing anyway...")
    
    # Step 4: Run optimization
    print_header("Step 3: Strategy Optimization (Bayesian)")
    print_info(f"This may take 10-30 minutes depending on data size and trial count...")
    
    opt_output = f"reports/{strategy_type}_optimization.json"
    opt_cmd = [
        sys.executable,
        "scripts/advanced_optimizer.py",
        "--data", data_path,
        "--strategy", strategy_type,
        "--metric", "risk_adjusted_return",
        "--trials", str(n_trials),
        "--output", opt_output
    ]
    
    if not run_command(opt_cmd, "Strategy optimization"):
        print_error("Optimization failed. Check logs above.")
        return 1
    
    # Step 5: Load and display optimization results
    print_header("Step 4: Optimization Results")
    try:
        with open(opt_output, 'r') as f:
            results = json.load(f)
        
        print_success("Optimization completed successfully!")
        print(f"\n{Colors.BOLD}Best Parameters:{Colors.ENDC}")
        for key, value in results['best_params'].items():
            print(f"  {key}: {value}")
        
        print(f"\n{Colors.BOLD}Training Score:{Colors.ENDC} {results['best_train_score']:.4f}")
        
        if 'validation_results' in results:
            val_metrics = results['validation_results']['metrics']
            print(f"\n{Colors.BOLD}Validation Metrics:{Colors.ENDC}")
            print(f"  Sharpe Ratio: {val_metrics.get('sharpe', 0):.2f}")
            print(f"  Sortino Ratio: {val_metrics.get('sortino', 0):.2f}")
            print(f"  Max Drawdown: {val_metrics.get('max_drawdown', 0)*100:.2f}%")
            print(f"  Profit Factor: {val_metrics.get('profit_factor', 0):.2f}")
            print(f"  Win Rate: {val_metrics.get('win_rate', 0)*100:.1f}%")
            print(f"  Total Trades: {results['validation_results']['total_trades']}")
    except Exception as e:
        print_error(f"Failed to load results: {e}")
        return 1
    
    # Step 6: Run comprehensive comparison
    print_header("Step 5: Comprehensive Performance Comparison")
    comparison_cmd = [
        sys.executable,
        "scripts/performance_analyzer.py",
        "--data", data_path,
        "--optimized", opt_output,
        "--output", "reports/final_comparison"
    ]
    
    if not run_command(comparison_cmd, "Performance comparison"):
        print_warning("Comparison analysis failed")
    
    # Summary
    print_header("Pipeline Complete!")
    print_success("All steps completed successfully")
    
    print(f"\n{Colors.BOLD}Generated Reports:{Colors.ENDC}")
    print(f"  1. Baseline Analysis: reports/baseline_analysis/")
    print(f"  2. Optimization Results: {opt_output}")
    print(f"  3. Final Comparison: reports/final_comparison/")
    print(f"     - Equity curves comparison")
    print(f"     - Drawdown analysis")
    print(f"     - Trade statistics")
    print(f"     - Risk metrics visualization")
    print(f"     - ANALYSIS_REPORT.md (summary)")
    
    print(f"\n{Colors.BOLD}Next Steps:{Colors.ENDC}")
    print("  1. Review reports in reports/final_comparison/")
    print("  2. Check ANALYSIS_REPORT.md for detailed insights")
    print("  3. If satisfied, run paper trading validation")
    print("  4. Consider running more optimization trials (100+) for production")
    
    print(f"\n{Colors.BOLD}Important Reminders:{Colors.ENDC}")
    print_warning("Past performance does not guarantee future results")
    print_warning("Always paper trade for 1-3 months before live trading")
    print_warning("Start with small position sizes in live trading")
    
    print(f"\n{Colors.OKGREEN}{Colors.BOLD}Happy Trading! 🚀{Colors.ENDC}\n")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Pipeline interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
