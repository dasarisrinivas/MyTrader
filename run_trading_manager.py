#!/usr/bin/env python3
"""Entry point for the Trading Manager daemon.

Usage:
  python3 run_trading_manager.py
  TM_DRY_RUN=1 python3 run_trading_manager.py    # don't actually kill bots
  TM_ACCOUNT_EQUITY=4750 python3 run_trading_manager.py
"""
from shree.trading_manager.manager import run

if __name__ == "__main__":
    raise SystemExit(run())
