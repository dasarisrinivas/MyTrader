"""Signal processing pipeline placeholder."""


class SignalProcessor:
    """Executes strategy and feature processing."""

    def __init__(self, manager: "LiveTradingManager"):  # noqa: F821 (forward reference)
        self.manager = manager

    async def process_trading_cycle(self, current_price: float):
        """Delegate to the manager's trading-cycle logic (to be migrated here)."""
        return await self.manager._process_trading_cycle(current_price)
