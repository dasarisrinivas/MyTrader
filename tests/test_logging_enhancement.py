import unittest
import sqlite3
import json
import os
from mytrader.monitoring.order_tracker import OrderTracker

class TestLoggingEnhancement(unittest.TestCase):
    def setUp(self):
        self.db_path = "tests/test_orders.db"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.tracker = OrderTracker(self.db_path)
        
    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_schema_columns(self):
        """Verify new columns exist in the schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("PRAGMA table_info(orders)")
            columns = [row[1] for row in cursor.fetchall()]
            self.assertIn("rationale", columns)
            self.assertIn("features", columns)
            self.assertIn("market_regime", columns)

    def test_record_order_with_context(self):
        """Verify recording an order with rationale and features."""
        rationale = {"reason": "RSI oversold", "score": 0.8}
        features = {"rsi": 25, "close": 100.5}
        
        self.tracker.record_order_placement(
            order_id=123,
            symbol="TEST",
            action="BUY",
            quantity=1,
            rationale=json.dumps(rationale),
            features=json.dumps(features),
            market_regime="HIGH_VOLATILITY"
        )
        
        # Verify data is stored
        order = self.tracker.get_order_details(123)
        self.assertIsNotNone(order)
        self.assertEqual(order["rationale"], json.dumps(rationale))
        self.assertEqual(order["features"], json.dumps(features))
        self.assertEqual(order["market_regime"], "HIGH_VOLATILITY")

if __name__ == '__main__':
    unittest.main()
