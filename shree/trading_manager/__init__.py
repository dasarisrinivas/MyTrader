"""Trading Manager — autonomous oversight layer for ShreeBot.

Runs as a third process alongside the MES Futures bot and SPY Options bot.
Watches signals, applies an institutional decision framework, and has full
veto + force-close authority over the bots.

Mandate: capital preservation first, profit second, scale third.
"""

__version__ = "0.1.0"
