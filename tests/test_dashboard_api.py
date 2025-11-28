import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.getcwd())

from dashboard.backend.dashboard_api import app

client = TestClient(app)

@pytest.fixture
def mock_order_tracker():
    with patch('dashboard.backend.dashboard_api.OrderTracker') as mock:
        yield mock

@pytest.fixture
def mock_bedrock():
    with patch('mytrader.llm.bedrock_client.BedrockClient') as mock:
        yield mock

@pytest.fixture
def mock_settings():
    with patch('dashboard.backend.dashboard_api.load_settings') as mock:
        mock.return_value.llm.model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
        mock.return_value.llm.region_name = "us-east-1"
        yield mock

def test_get_detailed_orders(mock_order_tracker):
    # Setup mock data
    tracker_instance = mock_order_tracker.return_value
    tracker_instance.get_all_orders.return_value = [
        {
            'order_id': 123,
            'timestamp': datetime.now().isoformat(),
            'symbol': 'ES',
            'action': 'BUY',
            'quantity': 1,
            'order_type': 'LMT',
            'status': 'Filled',
            'entry_price': 5000.0,
            'filled_quantity': 1,
            'avg_fill_price': 5000.0,
            'calculated_pnl': 100.0,
            'rationale': 'Bullish signal',
            'features': '{"rsi": 30}',
            'market_regime': 'TRENDING_UP'
        }
    ]
    tracker_instance.get_order_details.return_value = {'events': [], 'executions': []}

    # Make request
    response = client.get("/api/orders/detailed")
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert "orders" in data
    assert len(data["orders"]) == 1
    order = data["orders"][0]
    assert order["order_id"] == 123
    assert order["rationale"] == "Bullish signal"
    assert order["features"] == '{"rsi": 30}'
    assert order["market_regime"] == "TRENDING_UP"

def test_analyze_performance(mock_order_tracker, mock_bedrock, mock_settings):
    # Setup mock data
    tracker_instance = mock_order_tracker.return_value
    tracker_instance.get_all_orders.return_value = [
        {
            'order_id': 123,
            'timestamp': datetime.now().isoformat(),
            'symbol': 'ES',
            'action': 'BUY',
            'status': 'Filled',
            'quantity': 1,
            'filled_quantity': 1,
            'entry_price': 5000.0,
            'avg_fill_price': 5000.0,
            'calculated_pnl': 100.0,
            'rationale': 'Bullish',
            'market_regime': 'TRENDING'
        }
    ]
    
    bedrock_instance = mock_bedrock.return_value
    bedrock_instance.generate_text.return_value = "Great performance today!"

    # Make request
    response = client.post(
        "/api/analysis/performance",
        json={"period": "today"}
    )
    
    if response.status_code != 200:
        print(f"Response error: {response.text}")
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data["analysis"] == "Great performance today!"
    assert data["stats"]["total_pnl"] == 100.0
    assert data["stats"]["trades"] == 1
    
    # Verify LLM was called
    bedrock_instance.generate_text.assert_called_once()
