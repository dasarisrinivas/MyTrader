#!/usr/bin/env python3
"""
Live Trading Monitor Dashboard

Real-time monitoring dashboard for active trading sessions with:
1. Live P&L tracking
2. Performance metrics
3. Position monitoring
4. Risk alerts
5. Trade history

Usage:
    python scripts/monitor.py --report reports/paper_trade_latest.json
    python scripts/monitor.py --live  # Monitor active session
"""

import sys
from pathlib import Path
import json
from datetime import datetime, timedelta
import time

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mytrader.utils.logger import configure_logging, logger


class TradingMonitor:
    """Real-time trading monitor."""
    
    def __init__(self, report_path=None):
        """Initialize monitor."""
        self.report_path = report_path
        self.last_update = None
        
    def load_report(self):
        """Load performance report."""
        if not self.report_path or not Path(self.report_path).exists():
            return None
        
        try:
            with open(self.report_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Failed to load report: {e}")
            return None
    
    def display_dashboard(self, data):
        """Display trading dashboard."""
        # Clear screen
        print("\033[2J\033[H")  # ANSI clear screen
        
        print("=" * 100)
        print(f"{'MyTrader - Live Trading Monitor':^100}")
        print("=" * 100)
        print(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 100)
        
        if not data:
            print("\n❌ No data available\n")
            return
        
        # Session info
        print(f"\n📊 SESSION INFORMATION")
        print("-" * 100)
        
        session_info = data.get('session_info', {})
        print(f"Start Time:      {session_info.get('start_time', 'N/A')}")
        print(f"Duration:        {session_info.get('duration', 'N/A')}")
        print(f"Strategy:        {session_info.get('strategy', 'N/A')}")
        
        # Performance metrics
        print(f"\n💰 PERFORMANCE METRICS")
        print("-" * 100)
        
        snapshot = data.get('snapshot', {})
        
        total_pnl = snapshot.get('total_pnl', 0)
        total_return = snapshot.get('total_return', 0) * 100
        
        # Color code P&L
        pnl_color = '\033[92m' if total_pnl >= 0 else '\033[91m'  # Green/Red
        reset_color = '\033[0m'
        
        print(f"Total P&L:       {pnl_color}${total_pnl:+,.2f}{reset_color}")
        print(f"Total Return:    {pnl_color}{total_return:+.2f}%{reset_color}")
        print(f"Sharpe Ratio:    {snapshot.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown:    {snapshot.get('max_drawdown', 0) * 100:.2f}%")
        print(f"Win Rate:        {snapshot.get('win_rate', 0) * 100:.1f}%")
        
        # Trade statistics
        print(f"\n📈 TRADE STATISTICS")
        print("-" * 100)
        
        print(f"Total Trades:    {snapshot.get('trade_count', 0)}")
        print(f"Winning Trades:  {snapshot.get('winning_trades', 0)}")
        print(f"Losing Trades:   {snapshot.get('losing_trades', 0)}")
        print(f"Avg Trade P&L:   ${snapshot.get('avg_trade_pnl', 0):+.2f}")
        
        # Recent trades
        trades = data.get('trades', [])
        if trades:
            print(f"\n📋 RECENT TRADES (Last 10)")
            print("-" * 100)
            print(f"{'Time':<20} {'Action':<6} {'Qty':<4} {'Price':<10} {'P&L':<12} {'Status':<10}")
            print("-" * 100)
            
            for trade in trades[-10:]:
                timestamp = trade.get('timestamp', 'N/A')[:19]
                action = trade.get('action', 'N/A')
                qty = trade.get('qty', 0)
                price = trade.get('price', 0)
                pnl = trade.get('realized', 0)
                status = trade.get('status', 'N/A')
                
                pnl_str = f"${pnl:+.2f}"
                if pnl > 0:
                    pnl_str = f"\033[92m{pnl_str}\033[0m"
                elif pnl < 0:
                    pnl_str = f"\033[91m{pnl_str}\033[0m"
                
                print(f"{timestamp:<20} {action:<6} {qty:<4} ${price:<9.2f} {pnl_str:<20} {status:<10}")
        
        # Risk alerts
        print(f"\n⚠️  RISK ALERTS")
        print("-" * 100)
        
        alerts = []
        
        # Check drawdown
        max_dd = snapshot.get('max_drawdown', 0)
        if max_dd > 0.05:  # 5% drawdown
            alerts.append(f"🔴 High drawdown: {max_dd * 100:.1f}%")
        
        # Check win rate
        win_rate = snapshot.get('win_rate', 0)
        trade_count = snapshot.get('trade_count', 0)
        if trade_count >= 10 and win_rate < 0.4:
            alerts.append(f"🔴 Low win rate: {win_rate * 100:.1f}% (10+ trades)")
        
        # Check daily loss
        if total_pnl < -1000:  # $1000 loss
            alerts.append(f"🔴 Daily loss exceeds $1000: ${total_pnl:.2f}")
        
        if alerts:
            for alert in alerts:
                print(f"   {alert}")
        else:
            print("   ✅ No active alerts")
        
        # Instructions
        print(f"\n" + "-" * 100)
        print("Press Ctrl+C to exit monitor")
        print("=" * 100)
    
    def monitor_live(self, refresh_interval=5):
        """Monitor live trading session."""
        print("🔍 Starting live monitor...")
        print(f"   Refresh interval: {refresh_interval} seconds")
        print(f"   Report path: {self.report_path}\n")
        
        try:
            while True:
                data = self.load_report()
                self.display_dashboard(data)
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            print("\n\n✅ Monitor stopped")
    
    def display_summary(self):
        """Display session summary (one-time)."""
        data = self.load_report()
        self.display_dashboard(data)
        
        # Equity curve
        if data and 'equity_curve' in data:
            equity = data['equity_curve']
            if equity:
                print("\n\n📈 EQUITY CURVE")
                print("-" * 100)
                
                # Simple ASCII chart
                values = [e['equity'] for e in equity[-50:]]  # Last 50 points
                min_val = min(values)
                max_val = max(values)
                range_val = max_val - min_val if max_val > min_val else 1
                
                chart_height = 10
                for i in range(chart_height, -1, -1):
                    line = ""
                    threshold = min_val + (i / chart_height) * range_val
                    
                    for val in values:
                        if val >= threshold:
                            line += "█"
                        else:
                            line += " "
                    
                    if i == chart_height:
                        print(f"${max_val:>8,.0f} │{line}")
                    elif i == 0:
                        print(f"${min_val:>8,.0f} │{line}")
                    else:
                        print(f"{'':>9} │{line}")
                
                print(f"{'':>9} └{'─' * len(values)}")
                print(f"{'':>11}{' ' * (len(values)//2 - 5)}Time →")


def watch_directory(directory, pattern='paper_trade_*.json'):
    """Watch directory for latest report file."""
    from glob import glob
    
    reports_dir = Path(directory)
    if not reports_dir.exists():
        return None
    
    files = list(reports_dir.glob(pattern))
    if not files:
        return None
    
    # Get most recent file
    latest = max(files, key=lambda p: p.stat().st_mtime)
    return latest


def main():
    """Main entry point."""
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Monitor trading session')
    parser.add_argument('--report', help='Path to performance report JSON')
    parser.add_argument('--live', action='store_true', help='Monitor live session (auto-refresh)')
    parser.add_argument('--interval', type=int, default=5, help='Refresh interval in seconds (default: 5)')
    parser.add_argument('--watch-dir', default='reports', help='Directory to watch for reports (default: reports)')
    
    args = parser.parse_args()
    
    # Configure logging
    configure_logging(level="WARNING")  # Less verbose for dashboard
    
    # Determine report path
    if args.report:
        report_path = args.report
    else:
        # Find latest report
        report_path = watch_directory(args.watch_dir)
        if report_path:
            print(f"📊 Found latest report: {report_path}\n")
        else:
            print(f"❌ No reports found in {args.watch_dir}/")
            print(f"\nTo generate a report, run:")
            print(f"   python scripts/paper_trade.py --config config.yaml")
            sys.exit(1)
    
    # Create monitor
    monitor = TradingMonitor(report_path)
    
    # Run monitor
    if args.live:
        monitor.monitor_live(refresh_interval=args.interval)
    else:
        monitor.display_summary()
        print()


if __name__ == "__main__":
    main()
