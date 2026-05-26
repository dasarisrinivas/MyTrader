"""One-off verification: replay yesterday's 738P PUT_SWEEP signal through the
structure-aware, fill-realistic risk model.

The risk model uses executable pricing (long_ask − short_bid) rather than
mid-vs-mid, which under-estimates spread risk by ~20% on typical SPY chains.

Run with the project venv on PYTHONPATH so `shree.*` resolves.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from shree.trading_manager.spy_signal_watcher import _parse_spy_signal


# The exact JSON line we found in logs/spy_signals.jsonl for yesterday.
YESTERDAY_PUT_SWEEP_JSON = {
    "ts": "2026-05-18T19:37:32+00:00",
    "kind": "spy_signal",
    "signal_id": "PUT_SWEEP:738:P:MAY26:2026-05-18T19:37:32+00:00",
    "signal_type": "PUT_SWEEP",
    "strike": 738.0,
    "right": "P",
    "expiry": "MAY26",
    "expiry_date": "20260529",
    "dte": 11,
    "confidence": 0.7998,
    "confidence_tier": "HIGH",
    "spy_price": 736.74,
    "vix": 18.37,
    "iv_rank": 27.87,
    "regime": "RANGE_BOUND",
    "delta": -0.5068,
    "gamma": 0.0211,
    "theta": -0.319,
    "vega": 0.509,
    "impl_vol": 0.1517,
    "bid": 7.92,
    "ask": 7.96,
    "spread_pct": 0.5037,
    "volume": 10709,
    "open_interest": 13553,
    "sentiment_label": "NEUTRAL",
    "sentiment_score": 13.57,
    "reasoning": [],
    "suggested_trade": "Put Sweep: 738P exp MAY26\nBear Put Spread: Buy 738P / Sell 733P exp MAY26",
}


def compute(d: dict) -> dict:
    sig = _parse_spy_signal(d)
    assert sig is not None, "parser returned None"
    return {
        "structure": sig.structure or "(empty)",
        "long_bid": sig.bid, "long_ask": sig.ask, "long_mid": round(sig.mid, 3),
        "short_bid": sig.short_bid, "short_ask": sig.short_ask, "short_mid": round(sig.short_mid, 3),
        "estimated_risk_usd": round(sig.estimated_risk_per_contract_usd, 2),
    }


def show(label: str, d: dict, cap: float) -> None:
    r = compute(d)
    print(f"--- {label}")
    print(json.dumps(r, indent=2))
    verdict = "PASS" if r["estimated_risk_usd"] <= cap else "REJECT"
    print(f"  → {verdict} vs cap ${cap:.2f}")
    print()


def main() -> None:
    cap = 90.0  # SUSPECT-mode 2% cap from yesterday

    print("=" * 70)
    print("Risk model verification — 738P PUT_SWEEP from 2026-05-18 14:37 CT")
    print(f"SUSPECT-mode per-trade cap: ${cap:.2f}")
    print("=" * 70)
    print()

    # CASE A — Legacy pre-patch behaviour (structure missing)
    show("A: Legacy row, no structure tag (= yesterday's actual REJECT)",
         YESTERDAY_PUT_SWEEP_JSON, cap)

    # CASE B — Tagged LONG explicitly
    b = {**YESTERDAY_PUT_SWEEP_JSON, "structure": "LONG"}
    show("B: structure=LONG (explicit naked → unchanged from A)", b, cap)

    # CASE C — Spread tag, no short quote → width fallback (60% × 5 × 100 = $300)
    c = {**YESTERDAY_PUT_SWEEP_JSON,
         "structure": "BEAR_PUT_SPREAD", "short_strike": 733.0}
    show("C: BEAR_PUT_SPREAD, short_strike only (60% × width × 100 fallback)", c, cap)

    # CASE D — Spread with realistic short-leg quote
    # 733P mid ~$5.50 with a typical 60¢ wide chain: bid 5.20, ask 5.80
    d = {**c, "short_bid": 5.20, "short_ask": 5.80}
    show("D: BEAR_PUT_SPREAD, short quoted bid 5.20 / ask 5.80 (executable)", d, cap)

    # CASE E — Same spread, but use the OPTIMISTIC mid-vs-mid math
    # (this is what my first patch did — included here to show the gap)
    e = {**c, "short_bid": 5.50, "short_ask": 5.50}   # collapse short to mid
    e["bid"] = 7.94
    e["ask"] = 7.94                                    # collapse long to mid too
    show("E: same spread, MID-VS-MID math (over-optimistic — shows the ~22% gap)", e, cap)

    # CASE F — Tight chain (e.g. paper-trading or high-liquidity expiry)
    f = {**c, "short_bid": 5.45, "short_ask": 5.55}
    show("F: BEAR_PUT_SPREAD, tight 10¢ short chain", f, cap)


if __name__ == "__main__":
    main()
