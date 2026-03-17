"""Black-Scholes option pricing and Greeks reconstruction.

Used by the backtest engine to price options at any historical date given:
  - SPY spot price (from historical bars)
  - VIX level (as IV proxy — already annualized)
  - Days to expiry

No lookahead: all inputs must come from data available at the entry timestamp.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

# Risk-free rate: approximate 3-month T-bill (update periodically)
RISK_FREE_RATE: float = 0.053


@dataclass
class BSResult:
    price: float   # Option mid-price (per share)
    delta: float   # Rate of change vs underlying
    gamma: float   # Rate of change of delta
    theta: float   # Daily time decay (negative = decay per calendar day)
    vega: float    # Price change per 1% move in IV


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    right: str,
) -> BSResult:
    """Compute Black-Scholes price and all Greeks.

    Args:
        S:     Spot price (e.g., SPY last price)
        K:     Strike price
        T:     Time to expiry in years (e.g., 5 / 365)
        r:     Risk-free rate (annualized decimal, e.g., 0.053)
        sigma: Implied volatility (annualized decimal, e.g., 0.20 for 20%)
        right: 'P' for put, 'C' for call

    Returns:
        BSResult with price and Greeks.  price is floored at $0.01.
    """
    if T <= 1e-6 or sigma <= 1e-6:
        intrinsic = max(0.0, (K - S) if right == "P" else (S - K))
        delta = -1.0 if (right == "P" and S < K) else (1.0 if (right == "C" and S > K) else 0.0)
        return BSResult(price=max(intrinsic, 0.01), delta=delta, gamma=0.0, theta=0.0, vega=0.0)

    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    nd1 = norm.pdf(d1)
    disc = np.exp(-r * T)

    if right == "C":
        price = S * Nd1 - K * disc * Nd2
        delta = float(Nd1)
        theta = (-(S * nd1 * sigma) / (2 * sqrtT) - r * K * disc * Nd2) / 365.0
    else:
        price = K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = float(Nd1 - 1.0)
        theta = (-(S * nd1 * sigma) / (2 * sqrtT) + r * K * disc * norm.cdf(-d2)) / 365.0

    gamma = float(nd1 / (S * sigma * sqrtT))
    vega = float(S * nd1 * sqrtT / 100.0)  # per 1% point change in sigma

    return BSResult(
        price=max(float(price), 0.01),
        delta=round(delta, 6),
        gamma=round(gamma, 8),
        theta=round(float(theta), 6),
        vega=round(vega, 6),
    )


def find_strike_for_delta(
    S: float,
    T: float,
    r: float,
    sigma: float,
    target_delta: float,
    right: str,
) -> float:
    """Return the strike K that produces exactly target_delta (analytical solution).

    For a call: delta = N(d1) = target → d1 = N⁻¹(target)
    For a put:  delta = N(d1) - 1 = target → d1 = N⁻¹(1 + target)
    Then:       K = S · exp(−d1·σ·√T + (r + ½σ²)·T)
    """
    if right == "C":
        d1_target = float(norm.ppf(target_delta))
    else:  # Put — target_delta is negative
        d1_target = float(norm.ppf(1.0 + target_delta))

    log_SK = d1_target * sigma * np.sqrt(T) - (r + 0.5 * sigma ** 2) * T
    return float(S * np.exp(-log_SK))


def vix_to_sigma(vix: float) -> float:
    """Convert VIX level (e.g., 20.5) to annualized decimal vol (e.g., 0.205).

    VIX already represents 30-day annualized IV; no term-structure scaling applied
    so the result is conservative (uses actual VIX without forward vol adjustment).
    """
    return vix / 100.0


def dte_years(dte_calendar_days: int) -> float:
    """Convert calendar DTE to years for Black-Scholes T parameter."""
    return max(dte_calendar_days, 1) / 365.0
