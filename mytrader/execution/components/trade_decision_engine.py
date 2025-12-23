"""Trade decision evaluation placeholder."""


class TradeDecisionEngine:
    """Evaluates buy/sell/hold actions."""

    def __init__(self, manager: "LiveTradingManager"):  # noqa: F821 (forward reference)
        self.manager = manager
