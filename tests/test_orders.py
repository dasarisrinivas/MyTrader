from mytrader.monitoring.order_tracker import OrderTracker

tracker = OrderTracker()
orders = tracker.get_all_orders()
print(f'Total orders: {len(orders)}')
for order in orders:
    print(f"\nOrder {order['order_id']}:")
    print(f"  {order['action']} {order['quantity']} {order['symbol']} @ {order['status']}")
    details = tracker.get_order_details(order['order_id'])
    if details:
        print(f"  Has details: events={len(details.get('events', []))}, execs={len(details.get('executions', []))}")
        print(f"  Events: {details.get('events', [])}")
    else:
        print(f"  Details is None!")
