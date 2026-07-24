"""Minimal Black-Scholes IV inversion + delta, shared by the IB and ThetaData
ingest paths. Used only to populate delta/iv on prints when the vendor does not
supply greeks (delta-weighted flow needs a delta). Pure functions, no deps.
"""
from __future__ import annotations

import math
from typing import Optional

R_DEFAULT = 0.045  # flat short-rate assumption; 0DTE greeks are insensitive to it


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(S: float, K: float, T: float, sigma: float, right: str,
             r: float = R_DEFAULT) -> float:
    right = right.upper()[:1]
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, (S - K) if right == "C" else (K - S))
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if right == "C":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def bs_delta(S: float, K: float, T: float, sigma: float, right: str,
             r: float = R_DEFAULT) -> Optional[float]:
    right = right.upper()[:1]
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1) if right == "C" else norm_cdf(d1) - 1.0


def implied_vol(price: float, S: float, K: float, T: float,
                right: str) -> Optional[float]:
    """Bisection IV. Returns None if not solvable in a sane range."""
    if price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None
    lo, hi = 1e-4, 5.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if bs_price(S, K, T, mid, right) > price:
            hi = mid
        else:
            lo = mid
    iv = 0.5 * (lo + hi)
    return iv if 1e-3 < iv < 4.99 else None
