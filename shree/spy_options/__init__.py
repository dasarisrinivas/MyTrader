"""SPY Options signal-only bot package.

Signal flow:
    IBOptionsClient  →  ChainSnapshot  →  SignalEngine  →  SpyOptionsManager  →  Telegram
"""
