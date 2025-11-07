"""
Quick verification script to ensure all optimization components are in place
"""
import sys
from pathlib import Path
from datetime import datetime

def check_file(filepath, description):
    """Check if a file exists and return status."""
    path = Path(filepath)
    exists = path.exists()
    size = path.stat().st_size if exists else 0
    status = "✅" if exists else "❌"
    print(f"{status} {description}")
    if exists:
        print(f"   Path: {filepath}")
        print(f"   Size: {size:,} bytes")
    return exists

def main():
    print("\n" + "="*80)
    print("MyTrader Optimization Framework - Installation Verification")
    print("="*80 + "\n")
    
    components = {
        "Core Strategy": [
            ("mytrader/strategies/enhanced_regime_strategy.py", "Enhanced Regime-Based Strategy"),
        ],
        "Optimization Scripts": [
            ("scripts/advanced_optimizer.py", "Bayesian Optimizer (Optuna)"),
            ("scripts/performance_analyzer.py", "Performance Analysis & Visualization"),
            ("quickstart_optimization.py", "Quick Start Pipeline"),
        ],
        "Documentation": [
            ("STRATEGY_OPTIMIZATION.md", "Strategy Optimization Guide"),
            ("OPTIMIZATION_SUMMARY.md", "Optimization Summary Report"),
            ("README.md", "Updated README with Quick Start"),
        ],
        "Data Files": [
            ("data/es_synthetic_with_sentiment.csv", "Sample ES Futures Data"),
        ]
    }
    
    all_good = True
    
    for category, files in components.items():
        print(f"\n{category}:")
        print("-" * 80)
        for filepath, description in files:
            if not check_file(filepath, description):
                all_good = False
        print()
    
    # Check Python packages
    print("\nRequired Python Packages:")
    print("-" * 80)
    
    packages = ["pandas", "numpy", "optuna", "matplotlib", "seaborn"]
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            print(f"❌ {package} NOT installed (run: pip install {package})")
            all_good = False
    
    print("\n" + "="*80)
    if all_good:
        print("✅ ALL COMPONENTS VERIFIED - READY TO OPTIMIZE!")
        print("\nQuick Start:")
        print("  python3 quickstart_optimization.py")
        print("\nOr run individual steps:")
        print("  python3 scripts/advanced_optimizer.py --help")
        print("  python3 scripts/performance_analyzer.py --help")
    else:
        print("⚠️  SOME COMPONENTS MISSING - CHECK ABOVE")
        print("\nMissing packages? Run:")
        print("  pip install pandas numpy optuna matplotlib seaborn")
    print("="*80 + "\n")
    
    # Summary statistics
    print("Summary:")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Platform: {sys.platform}")
    print("\nTarget Metrics:")
    print("  • Sharpe Ratio ≥ 1.5")
    print("  • Max Drawdown ≤ 15%")
    print("  • Win Rate ≥ 60%")
    print("  • Profit Factor ≥ 1.3")
    print("\nDocumentation:")
    print("  • STRATEGY_OPTIMIZATION.md - Complete guide")
    print("  • OPTIMIZATION_SUMMARY.md - Executive summary")
    print("  • README.md - Quick reference")
    print()

if __name__ == "__main__":
    main()
