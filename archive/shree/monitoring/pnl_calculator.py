"""Enhanced P&L calculation for order tracking."""
from typing import List, Dict, Optional


def calculate_pnl_for_orders(orders: List[Dict]) -> List[Dict]:
    """
    Calculate P&L for orders by matching BUY/SELL pairs.
    
    For open positions, calculate unrealized P&L.
    For closed positions (round trips), calculate realized P&L.
    """
    # Sort orders by timestamp
    sorted_orders = sorted(orders, key=lambda x: x['timestamp'])
    
    # Track position
    position = 0
    avg_entry_price = 0.0
    total_cost = 0.0
    
    enhanced_orders = []
    
    for order in sorted_orders:
        if order['status'] != 'Filled':
            enhanced_orders.append(order)
            continue
        
        order_copy = order.copy()
        quantity = order['filled_quantity'] or order['quantity']
        fill_price = order['avg_fill_price'] or order['entry_price']
        
        if not fill_price:
            enhanced_orders.append(order_copy)
            continue
        
        if order['action'] == 'BUY':
            # Opening or adding to long position
            if position <= 0:
                # New position or closing short
                if position < 0:
                    # Closing short position
                    contracts_closed = min(quantity, abs(position))
                    pnl = contracts_closed * (avg_entry_price - fill_price) * 50  # ES multiplier
                    order_copy['calculated_pnl'] = pnl
                    position += contracts_closed
                    
                    # Remaining quantity opens new long
                    remaining = quantity - contracts_closed
                    if remaining > 0:
                        total_cost = remaining * fill_price
                        avg_entry_price = fill_price
                        position = remaining
                else:
                    # Opening new long
                    total_cost = quantity * fill_price
                    avg_entry_price = fill_price
                    position = quantity
            else:
                # Adding to long position
                total_cost += quantity * fill_price
                position += quantity
                avg_entry_price = total_cost / position
            
        elif order['action'] == 'SELL':
            # Closing long or opening short
            if position > 0:
                # Closing long position
                contracts_closed = min(quantity, position)
                pnl = contracts_closed * (fill_price - avg_entry_price) * 50  # ES multiplier
                order_copy['calculated_pnl'] = pnl
                position -= contracts_closed
                
                # Remaining quantity opens new short
                remaining = quantity - contracts_closed
                if remaining > 0:
                    total_cost = remaining * fill_price
                    avg_entry_price = fill_price
                    position = -remaining
                elif position > 0:
                    total_cost = position * avg_entry_price
            else:
                # Opening new short or adding to short
                if position == 0:
                    total_cost = quantity * fill_price
                    avg_entry_price = fill_price
                    position = -quantity
                else:
                    total_cost += quantity * fill_price
                    position -= quantity
                    avg_entry_price = total_cost / abs(position)
        
        # Store current position info
        order_copy['position_after'] = position
        order_copy['avg_entry_price'] = avg_entry_price if position != 0 else None
        
        enhanced_orders.append(order_copy)
    
    return enhanced_orders


def calculate_unrealized_pnl(position: int, avg_entry: float, current_price: float) -> float:
    """Calculate unrealized P&L for open position."""
    if position == 0:
        return 0.0
    
    # ES futures: $50 per point
    if position > 0:
        # Long position
        return position * (current_price - avg_entry) * 50
    else:
        # Short position
        return abs(position) * (avg_entry - current_price) * 50
