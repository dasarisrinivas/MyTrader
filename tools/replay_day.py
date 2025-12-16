#!/usr/bin/env python3
"""Dry-run replay tool for a specific trading day.

Usage:
    python tools/replay_day.py --date 2025-12-12 --dry-run
    python tools/replay_day.py --date 2025-12-12 --order-id 14812
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.replay_trade_from_logs import parse_order_context, replay_guardrails, format_bool


def find_orders_for_date(log_path: Path, date_str: str) -> List[int]:
    """Find all order IDs mentioned in logs for a specific date."""
    date_pattern = re.compile(rf"{date_str}")
    order_pattern = re.compile(r"orderId=(\d+)")
    orders = set()
    
    if not log_path.exists():
        return []
    
    lines = log_path.read_text().splitlines()
    for line in lines:
        if date_pattern.search(line):
            for match in order_pattern.finditer(line):
                orders.add(int(match.group(1)))
    
    return sorted(orders)


def analyze_trade_timeline(log_path: Path, date_str: str, dry_run: bool = True) -> Dict:
    """Analyze all trades for a specific date and verify guardrails."""
    orders = find_orders_for_date(log_path, date_str)
    
    if not orders:
        print(f"⚠️  No orders found for date {date_str}")
        return {"orders": [], "blocked": 0, "allowed": 0}
    
    print(f"📅 Analyzing {len(orders)} orders for {date_str}\n")
    
    blocked_count = 0
    allowed_count = 0
    order_details = []
    
    for order_id in orders:
        try:
            ctx = parse_order_context(log_path, order_id)
            guard_triggered, details = replay_guardrails(
                context=ctx,
                block_on_wait=True,  # Always enforce WAIT blocking
                override_confidence=0.75,
                contract_multiplier=50.0,
                max_loss=1250.0,
                tick_size=0.25,
                min_distance_ticks=4,
            )
            
            if guard_triggered:
                blocked_count += 1
            else:
                allowed_count += 1
            
            order_details.append({
                "order_id": order_id,
                "action": ctx.signal_action,
                "confidence": ctx.signal_confidence,
                "entry_price": ctx.entry_price,
                "stop_loss": ctx.stop_loss,
                "take_profit": ctx.take_profit,
                "blocked": guard_triggered,
                "reasons": details["reasons"],
            })
            
            status = "🛑 BLOCKED" if guard_triggered else "✅ ALLOWED"
            print(f"{status} Order {order_id}: {ctx.signal_action} @ {ctx.entry_price or 'N/A'}")
            if details["reasons"]:
                print(f"   Reasons: {', '.join(details['reasons'])}")
            print()
            
        except Exception as e:
            print(f"⚠️  Error analyzing order {order_id}: {e}")
            continue
    
    summary = {
        "orders": order_details,
        "blocked": blocked_count,
        "allowed": allowed_count,
        "total": len(orders),
    }
    
    print("=" * 60)
    print(f"📊 Summary for {date_str}:")
    print(f"   Total orders: {summary['total']}")
    print(f"   Would be blocked: {summary['blocked']}")
    print(f"   Would be allowed: {summary['allowed']}")
    print("=" * 60)
    
    if dry_run:
        print("\n✅ Dry-run complete. No trades were actually placed.")
        print("   This analysis shows what would happen with current guardrails.")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replay trading day and verify guardrails would block unsafe trades."
    )
    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="Date to analyze (YYYY-MM-DD format, e.g., 2025-12-12)",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("logs/bot.log"),
        help="Path to bot log file",
    )
    parser.add_argument(
        "--order-id",
        type=int,
        help="Analyze specific order ID only",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Dry-run mode (default: True)",
    )
    
    args = parser.parse_args()
    
    # Validate date format
    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"[ERROR] Invalid date format: {args.date}. Use YYYY-MM-DD format.")
        sys.exit(1)
    
    if not args.log.exists():
        print(f"[ERROR] Log file not found: {args.log}")
        sys.exit(2)
    
    if args.order_id:
        # Analyze single order
        try:
            ctx = parse_order_context(args.log, args.order_id)
            guard_triggered, details = replay_guardrails(
                context=ctx,
                block_on_wait=True,
                override_confidence=0.75,
                contract_multiplier=50.0,
                max_loss=1250.0,
                tick_size=0.25,
                min_distance_ticks=4,
            )
            
            print(f"🔍 Order {args.order_id} Analysis:")
            print(f"   Action: {ctx.signal_action}")
            print(f"   Entry: {ctx.entry_price}")
            print(f"   SL: {ctx.stop_loss}, TP: {ctx.take_profit}")
            print(f"   Guardrails would block: {format_bool(guard_triggered)}")
            if details["reasons"]:
                print(f"   Reasons: {', '.join(details['reasons'])}")
        except ValueError as e:
            print(f"[ERROR] {e}")
            sys.exit(2)
    else:
        # Analyze all orders for date
        summary = analyze_trade_timeline(args.log, args.date, dry_run=args.dry_run)
        
        if summary["blocked"] > 0:
            print(f"\n✅ Guardrails would have blocked {summary['blocked']} unsafe trade(s).")
            sys.exit(0)
        else:
            print(f"\n⚠️  No trades would be blocked. Review guardrail thresholds if needed.")
            sys.exit(0)

