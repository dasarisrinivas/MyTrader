"""Lightweight indicator utilities used across the trading stack."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi_series(series: pd.Series, period: int = 14, initial_value: float = 50.0) -> pd.Series:
    """Compute RSI using Wilder's smoothing.
    
    Designed for streaming/rolling use: returns a full series so callers can
    append new prices without recomputing the entire window. Short histories
    are handled gracefully by seeding early values with ``initial_value``.
    """
    if series.empty:
        return pd.Series([], index=series.index, dtype=float)

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # Handle monotonic runs gracefully
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(avg_gain != 0, 0.0)
    rsi = rsi.clip(lower=0, upper=100)

    if initial_value is not None and len(rsi) > 0:
        rsi.iloc[:period] = rsi.iloc[:period].fillna(initial_value)

    return rsi.ffill().fillna(initial_value).astype(float)
