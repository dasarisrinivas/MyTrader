"""Configuration for SPY options trading bot.

To override any value, set the corresponding environment variable before starting.
Example:
    LIVE_TRADING=true python main.py
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------
IBKR_HOST: str = os.getenv("IBKR_HOST", "127.0.0.1")

# IB Gateway live port: 4001
IBKR_LIVE_PORT: int = int(os.getenv("IBKR_LIVE_PORT", "4001"))

# IMPORTANT: must differ from MES bot's clientId (which is hardcoded to 11)
CLIENT_ID: int = int(os.getenv("SPY_BOT_CLIENT_ID", "20"))

LIVE_TRADING: bool = os.getenv("LIVE_TRADING", "true").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Instrument
# ---------------------------------------------------------------------------
SYMBOL: str = "SPY"
EXCHANGE: str = "SMART"
CURRENCY: str = "USD"

# ---------------------------------------------------------------------------
# Option selection
# ---------------------------------------------------------------------------
MIN_DTE: int = 5          # Minimum days to expiration
MAX_DTE: int = 7          # Maximum days to expiration (weekly)
TARGET_DELTA_PUT: float = -0.25   # Target delta for short put leg
TARGET_DELTA_CALL: float = 0.25   # Target delta for short call leg
DELTA_TOLERANCE: float = 0.07     # Accept delta within ± this of target

MIN_THETA: float = -0.05          # Minimum theta (must be at least this negative)
MIN_OPEN_INTEREST: int = 500
MIN_VOLUME: int = 200
MAX_SPREAD_PCT: float = 0.15      # Max bid/ask spread as fraction of mid-price
MAX_SPREAD_ABS: float = 0.10      # Max bid/ask spread in absolute dollars

# ---------------------------------------------------------------------------
# Risk management
# ---------------------------------------------------------------------------
PROFIT_TARGET_PCT: float = 0.50   # Close at 50% of max profit (buy back at 50% of credit)
MAX_LOSS_MULTIPLE: float = 2.0    # Close if loss reaches 2× premium received
DELTA_STOP: float = 0.50          # Close if |delta| exceeds this (position gone ITM)
MAX_ACCOUNT_RISK_PCT: float = 0.05  # Max 5% of NLV in margin for this bot
DAILY_LOSS_LIMIT_PCT: float = 0.03  # Pause new entries if account drops >3% on the day

# ---------------------------------------------------------------------------
# PDT compliance
# ---------------------------------------------------------------------------
MAX_WEEKLY_TRADES: int = 3        # Max round-trips in a rolling 5-trading-day window
MAX_CONTRACTS: int = 1            # Hard ceiling — do not change until 3+ months of live data
PDT_LOG_FILE: str = "spy_options_bot/pdt_log.json"

# ---------------------------------------------------------------------------
# VIX / trend filters
# ---------------------------------------------------------------------------
MIN_VIX: float = 12.0             # Don't sell premium in extremely low-vol environments
MAX_VIX: float = 30.0             # Circuit breaker — skip entry on panic-spike days (VIX > 30)
TREND_SMA_DAYS: int = 20          # Days for SPY trend SMA

# ---------------------------------------------------------------------------
# Order execution
# ---------------------------------------------------------------------------
ORDER_FILL_TIMEOUT: int = 60      # Seconds before adjusting limit price
ORDER_ADJUST_STEP: float = 0.01   # Step down mid-price by this amount after timeout

# ---------------------------------------------------------------------------
# Market hours (Eastern Time)
# ---------------------------------------------------------------------------
MARKET_OPEN_HOUR: int = 9
MARKET_OPEN_MINUTE: int = 35      # Enter after 9:35 AM ET (avoid open volatility)
MARKET_CLOSE_HOUR: int = 15
MARKET_CLOSE_MINUTE: int = 30     # Stop new entries at 3:30 PM ET
EOD_CLOSE_HOUR: int = 15          # Thursday EOD close: 3:45 PM ET
EOD_CLOSE_MINUTE: int = 45
ENTRY_DAYS: tuple[int, ...] = (0, 1, 2)  # Monday=0, Tuesday=1, Wednesday=2

# ---------------------------------------------------------------------------
# Polling
# ---------------------------------------------------------------------------
POLL_INTERVAL: int = 300          # Check positions every 5 minutes (seconds)

# ---------------------------------------------------------------------------
# Notifications (optional — bot runs normally if not set)
# ---------------------------------------------------------------------------
# Set these environment variables to enable Telegram alerts:
#   export TELEGRAM_BOT_TOKEN="123456:ABC-DEF..."
#   export TELEGRAM_CHAT_ID="-100123456789"
TELEGRAM_BOT_TOKEN: str | None = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID: str | None = os.getenv("TELEGRAM_CHAT_ID")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE: str = "logs/spy_options_bot.log"
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# ---------------------------------------------------------------------------
# Derived (do not edit)
# ---------------------------------------------------------------------------
IBKR_PORT: int = IBKR_LIVE_PORT
TRADING_MODE: str = "live" if LIVE_TRADING else "paper"
