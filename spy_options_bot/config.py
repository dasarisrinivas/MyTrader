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
# Roll trigger: when |delta| crosses this threshold, close current spread and
# re-enter next week rather than waiting for the hard stop at DELTA_STOP.
# Only fires during the entry window (Mon–Wed) so the bot can immediately open
# a fresh spread for next Friday in the same cycle.
ROLL_DELTA_TRIGGER: float = 0.40
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

# IV Rank — only sell when VIX is elevated relative to its 1-year range
# 0.20 = VIX must be in at least the 20th percentile of its 52-week range
MIN_IV_RANK: float = 0.20

# Skew — put IV minus call IV (at similar delta). If calls are much more expensive
# than puts (negative skew), market is pricing in upside risk → skip puts.
# -0.03 = allow slight call skew but block entry if calls >3% more expensive than puts
MIN_PUT_CALL_SKEW: float = -0.03

# Expected move buffer — strike must be at or beyond N × expected move from spot.
# 1.0 = must be at least 1 standard deviation OTM
EXPECTED_MOVE_BUFFER: float = 1.0

# Theta efficiency — theta must be at least this fraction of delta per day.
# Ensures we're collecting meaningful decay relative to the directional risk taken.
MIN_THETA_DELTA_RATIO: float = 0.08

# Vega exposure — skip if short leg's vega × 100 > this value.
# Units: dollars lost per 1-point VIX increase per contract (100 shares).
# At $10, a 5-point VIX spike costs at most $50 on the short leg before hedge offset.
MAX_VEGA_LOSS_PER_VIX_POINT: float = 10.0

# Support / resistance — block put entry when SPY is within this % of its 52-week low
# (too close to a major floor = gap-down risk). Block call entry near 52-week high.
SR_LOW_BUFFER_PCT: float = 0.05   # block puts if SPY < 52w_low × 1.05
SR_HIGH_BUFFER_PCT: float = 0.03  # block calls if SPY > 52w_high × 0.97

# Event risk — number of calendar days around FOMC/CPI/NFP where we skip entry
EVENT_BLACKOUT_DAYS: int = 1

# ---------------------------------------------------------------------------
# VIX spike guard — catches early panic before SMA200 reacts (price-based)
# ---------------------------------------------------------------------------
VIX_SPIKE_MULTIPLIER: float = 1.25   # Block if VIX > 5-day avg × this
VIX_SPIKE_SKIP_DAYS: int = 3         # Calendar days to pause after spike

# ---------------------------------------------------------------------------
# Large move guard — catches gap scenarios that VIX spike misses
# ---------------------------------------------------------------------------
LARGE_MOVE_PCT: float = 0.02         # Block if SPY moved >2% from prior close
LARGE_MOVE_SKIP_DAYS: int = 2        # Calendar days to pause after large move

# ---------------------------------------------------------------------------
# Credit spread
# ---------------------------------------------------------------------------
SPREAD_WIDTH: float = 10.0           # Dollar-width of put/call spread (hedge 10 strikes away)
MIN_NET_CREDIT: float = 0.60         # Minimum net credit after buying the hedge

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
