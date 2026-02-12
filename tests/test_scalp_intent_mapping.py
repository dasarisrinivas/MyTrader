import types

import pytest


@pytest.mark.asyncio
async def test_order_coordinator_maps_is_scalp_metadata_to_scalp_action():
    """Option B (Strict): BUY/SELL stays primary, but metadata.is_scalp opt-in triggers SCALP_* logic."""

    from shree.execution.components.order_coordinator import OrderCoordinator

    class DummyManager:
        def __init__(self):
            self._current_cycle_id = "cycle"

        def _add_reason_code(self, _reason: str) -> None:
            return None

        async def _broadcast_error(self, _msg: str) -> None:
            return None

    class DummyCoordinator(OrderCoordinator):
        def __init__(self):
            super().__init__(DummyManager())

        def prepare_order_metadata(self, raw_metadata, current_price, source):
            # minimal shim: keep metadata as-is
            return dict(raw_metadata or {})

        async def enforce_entry_gates(self, action, metadata):
            # stop before any deeper logic executes
            return False, "UNIT_TEST_STOP", "sigkey"

    oc = DummyCoordinator()

    signal = types.SimpleNamespace(
        action="SELL",
        confidence=0.9,
        metadata={"is_scalp": True},
    )

    # features isn't used before our test-stop gate triggers
    await oc.execute_trade_with_risk_checks(signal, current_price=100.0, features=None)

    assert signal.action == "SCALP_SELL"


@pytest.mark.asyncio
async def test_order_coordinator_does_not_map_without_is_scalp():
    from shree.execution.components.order_coordinator import OrderCoordinator

    class DummyManager:
        def __init__(self):
            self._current_cycle_id = "cycle"

        def _add_reason_code(self, _reason: str) -> None:
            return None

        async def _broadcast_error(self, _msg: str) -> None:
            return None

    class DummyCoordinator(OrderCoordinator):
        def __init__(self):
            super().__init__(DummyManager())

        def prepare_order_metadata(self, raw_metadata, current_price, source):
            return dict(raw_metadata or {})

        async def enforce_entry_gates(self, action, metadata):
            return False, "UNIT_TEST_STOP", "sigkey"

    oc = DummyCoordinator()

    signal = types.SimpleNamespace(
        action="SELL",
        confidence=0.9,
        metadata={},
    )

    await oc.execute_trade_with_risk_checks(signal, current_price=100.0, features=None)

    assert signal.action == "SELL"
