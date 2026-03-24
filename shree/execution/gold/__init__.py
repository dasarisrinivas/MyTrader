"""Gold futures execution package.

Components:
    GoldContractFactory   — IBKR contract creation and qualification
    GoldRiskManager       — position sizing and daily guardrails
    GoldStateManager      — persist state across bot restarts
    GoldJournal           — trade logging and session metrics
    GoldTradingManager    — main async trading loop
"""
from .contract import GoldContractFactory
from .risk import GoldRiskManager, SizingResult
from .rollover import ContractRollMonitor, should_roll
from .state import GoldStateManager
from .journal import GoldJournal
from .manager import GoldTradingManager

__all__ = [
    "GoldContractFactory",
    "GoldRiskManager",
    "SizingResult",
    "ContractRollMonitor",
    "should_roll",
    "GoldStateManager",
    "GoldJournal",
    "GoldTradingManager",
]
