#!/usr/bin/env python3
"""Replay historical log snippets with the new guardrails."""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mytrader.execution.guards import (  # pylint: disable=wrong-import-position
    WaitDecisionContext,
    should_block_on_wait,
    compute_trade_risk_dollars,
)
from mytrader.execution.order_builder import validate_bracket_prices  # pylint: disable=wrong-import-position


@dataclass
class OrderContext:
    order_id: int
    log_line: str
    signal_action: Optional[str] = None
    signal_confidence: Optional[float] = None
    decision: Optional[str] = None
    decision_confidence: Optional[float] = None
    advisory_only: bool = False
    position_qty: Optional[float] = None
    request_action: Optional[str] = None
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


def parse_order_context(log_path: Path, order_id: int) -> OrderContext:
    """Extract relevant context from logs for the given order."""
    lines = log_path.read_text().splitlines()
    signal_re = re.compile(r"Hybrid Pipeline: ([A-Z_]+) \(conf=([0-9.]+)")
    decision_re = re.compile(r"Decision Agent response: ([A-Z]+) \(confidence: ([0-9.]+)%\)")
    position_re = re.compile(r"Reconciled position: ES qty=([\-0-9.]+)")
    order_re = re.compile(r"orderId=(\d+)")
    telegram_re = re.compile(
        r"Telegram alert: (\w+) [0-9.]+ @ ([0-9.]+), SL=([A-Za-z0-9.]+), TP=([A-Za-z0-9.]+)"
    )
    context = OrderContext(order_id=order_id, log_line="")
    last_signal_action: Optional[str] = None
    last_signal_conf: Optional[float] = None
    last_decision: Optional[str] = None
    last_decision_conf: Optional[float] = None
    last_advisory = False
    last_position: Optional[float] = None
    last_telegram: Optional[tuple[str, float, Optional[float], Optional[float]]] = None

    for idx, line in enumerate(lines):
        if "Hybrid Pipeline:" in line:
            match = signal_re.search(line)
            if match:
                last_signal_action = match.group(1)
                last_signal_conf = float(match.group(2))
        if "Decision Agent response:" in line:
            match = decision_re.search(line)
            if match:
                last_decision = match.group(1)
                last_decision_conf = float(match.group(2)) / 100.0
        if "AWS WAIT advisory only" in line:
            last_advisory = True
        if "Reconciled position:" in line:
            match = position_re.search(line)
            if match:
                try:
                    last_position = float(match.group(1))
                except ValueError:
                    last_position = None
        if "Telegram alert:" in line:
            match = telegram_re.search(line)
            if match:
                action = match.group(1)
                entry_price = float(match.group(2))
                sl_token = match.group(3)
                tp_token = match.group(4)
                stop = None if sl_token == "None" else float(sl_token)
                target = None if tp_token == "None" else float(tp_token)
                last_telegram = (action, entry_price, stop, target)

        if f"orderId={order_id}" in line:
            context.log_line = line
            context.signal_action = last_signal_action
            context.signal_confidence = last_signal_conf
            context.decision = last_decision
            context.decision_confidence = last_decision_conf
            context.advisory_only = last_advisory
            context.position_qty = last_position
            if "action='" in line:
                action_match = re.search(r"action='([A-Z]+)'", line)
                if action_match:
                    context.request_action = action_match.group(1)
            if last_telegram:
                context.entry_price = last_telegram[1]
                context.stop_loss = last_telegram[2]
                context.take_profit = last_telegram[3]
            break

    if not context.log_line:
        raise ValueError(f"Order {order_id} not found in {log_path}")
    return context


def replay_guardrails(
    context: OrderContext,
    block_on_wait: bool,
    override_confidence: float,
    contract_multiplier: float,
    max_loss: float,
    tick_size: float = 0.25,
    min_distance_ticks: int = 4,
) -> tuple[bool, dict]:
    """Run guardrail logic over the parsed context.
    
    Returns:
        (blocked, details): Tuple of (whether trade would be blocked, details dict)
    """
    details = {
        "wait_blocked": False,
        "missing_protection": False,
        "invalid_bracket": False,
        "insufficient_distance": False,
        "excessive_risk": False,
        "reasons": [],
    }
    
    # Check 1: WAIT blocking
    if (
        context.decision == "WAIT"
        and context.signal_confidence is not None
        and context.decision_confidence is not None
    ):
        wait_ctx = WaitDecisionContext(
            decision=context.decision,
            advisory_only=context.advisory_only,
            confidence=context.decision_confidence,
            size_multiplier=None,
        )
        wait_blocked = should_block_on_wait(
            wait_ctx,
            block_on_wait=block_on_wait,
            override_confidence=override_confidence,
            signal_confidence=context.signal_confidence,
        )
        if wait_blocked:
            details["wait_blocked"] = True
            details["reasons"].append("AWS_WAIT")
    
    # Check 2: Missing protection
    missing_protection = context.stop_loss is None or context.take_profit is None
    if missing_protection:
        details["missing_protection"] = True
        details["reasons"].append("MISSING_PROTECTION")
    
    # Check 3: Bracket validation (if protection exists)
    invalid_bracket = False
    insufficient_distance = False
    if not missing_protection and context.entry_price is not None:
        action = context.request_action or context.signal_action or "BUY"
        bracket_result = validate_bracket_prices(
            action=action,
            entry_price=context.entry_price,
            stop_loss=context.stop_loss,
            take_profit=context.take_profit,
            tick_size=tick_size,
            min_distance_ticks=min_distance_ticks,
        )
        if not bracket_result.valid:
            invalid_bracket = True
            details["invalid_bracket"] = True
            details["reasons"].append(f"INVALID_BRACKET: {bracket_result.reason}")
        else:
            # Check if distances were adjusted (indicating insufficient distance)
            if bracket_result.adjusted_fields:
                insufficient_distance = True
                details["insufficient_distance"] = True
                details["reasons"].append(f"INSUFFICIENT_DISTANCE: {bracket_result.adjusted_fields}")
    
    # Check 4: Excessive risk
    excessive_risk = False
    if not missing_protection and context.entry_price is not None and context.stop_loss is not None:
        risk = compute_trade_risk_dollars(
            entry_price=context.entry_price,
            stop_loss=context.stop_loss,
            contract_multiplier=contract_multiplier,
        )
        if risk > max_loss:
            excessive_risk = True
            details["excessive_risk"] = True
            details["reasons"].append(f"MAX_LOSS_CAP: ${risk:.2f} > ${max_loss:.2f}")
    
    blocked = (
        details["wait_blocked"]
        or details["missing_protection"]
        or details["invalid_bracket"]
        or details["excessive_risk"]
    )
    
    return blocked, details


def format_bool(value: bool) -> str:
    return "✅" if value else "❌"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dry-run guardrail replay using logs.")
    parser.add_argument("--log", type=Path, default=Path("logs/bot.log"), help="Path to bot log file")
    parser.add_argument("--order-id", type=int, required=True, help="Order ID to analyze")
    parser.add_argument("--block-on-wait", type=int, default=1, help="Whether WAIT should block (1/0)")
    parser.add_argument("--override-confidence", type=float, default=0.75, help="Override threshold when WAIT is advisory-only")
    parser.add_argument("--max-loss", type=float, default=1250.0, help="Max per-trade dollar loss cap")
    parser.add_argument("--contract-multiplier", type=float, default=50.0, help="Contract multiplier for ES")
    parser.add_argument("--tick-size", type=float, default=0.25, help="Tick size for ES (default 0.25)")
    parser.add_argument("--min-distance-ticks", type=int, default=4, help="Minimum distance in ticks (default 4 = 1 point)")
    parser.add_argument("--verify-reduce-only", action="store_true", help="Also verify reduce-only exit logic based on position size")
    args = parser.parse_args()

    if not args.log.exists():
        print(f"[ERROR] Log file not found: {args.log}")
        sys.exit(2)

    try:
        ctx = parse_order_context(args.log, args.order_id)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(2)

    print(f"🔍 Analyzing order {args.order_id} ({ctx.signal_action}, decision={ctx.decision})")
    if ctx.signal_confidence is not None:
        print(f"   Signal confidence: {ctx.signal_confidence:.2f}")
    if ctx.decision_confidence is not None:
        print(f"   AWS confidence: {ctx.decision_confidence:.2f} (advisory_only={ctx.advisory_only})")
    if ctx.entry_price is not None:
        print(
            f"   Entry telemetry: price={ctx.entry_price:.2f} "
            f"SL={ctx.stop_loss} TP={ctx.take_profit}"
        )

    guard_triggered, details = replay_guardrails(
        context=ctx,
        block_on_wait=bool(args.block_on_wait),
        override_confidence=args.override_confidence,
        contract_multiplier=args.contract_multiplier,
        max_loss=args.max_loss,
        tick_size=args.tick_size,
        min_distance_ticks=args.min_distance_ticks,
    )
    
    print("\n📊 Guardrail Analysis:")
    print(f"   WAIT guard invoked: {format_bool(ctx.decision == 'WAIT' and bool(args.block_on_wait))}")
    if ctx.decision == "WAIT":
        print(f"   WAIT blocked: {format_bool(details['wait_blocked'])}")
        if details['wait_blocked']:
            print(f"      Reason: AWS WAIT decision (advisory_only={ctx.advisory_only})")
    
    print(f"   Protective levels present: {format_bool(ctx.stop_loss is not None and ctx.take_profit is not None)}")
    if ctx.stop_loss is None or ctx.take_profit is None:
        print(f"   Missing protection: {format_bool(details['missing_protection'])}")
        if details['missing_protection']:
            print(f"      SL={ctx.stop_loss}, TP={ctx.take_profit}")
    
    if ctx.entry_price and ctx.stop_loss and ctx.take_profit:
        print(f"   Bracket validation: {format_bool(not details['invalid_bracket'])}")
        if details['invalid_bracket']:
            print(f"      Invalid bracket detected")
        if details['insufficient_distance']:
            print(f"      Insufficient distance (adjusted)")
        
        # Calculate risk
        risk = compute_trade_risk_dollars(
            entry_price=ctx.entry_price,
            stop_loss=ctx.stop_loss,
            contract_multiplier=args.contract_multiplier,
        )
        print(f"   Estimated risk: ${risk:.2f} (max: ${args.max_loss:.2f})")
        print(f"   Risk within limit: {format_bool(not details['excessive_risk'])}")
    
    print(f"\n🛑 Guardrails would block trade: {format_bool(guard_triggered)}")
    if details['reasons']:
        print(f"   Blocking reasons: {', '.join(details['reasons'])}")

    exit_required_action = None
    if args.verify_reduce_only and ctx.position_qty is not None:
        exit_required_action = "SELL" if ctx.position_qty > 0 else "BUY"
        print(
            f"   Reduce-only exit would use {exit_required_action} "
            f"for position qty={ctx.position_qty}"
        )

    if not guard_triggered:
        print("[WARN] Guardrails would not block this trade – investigate further.")
        sys.exit(1)

    print("✅ Dry-run replay completed – guardrails prevent recurrence.")
