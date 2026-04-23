"""CSV / in-memory replay harness for rules_v2.

The harness walks the provided bar timeline one 5-minute increment at a
time. At each step it:

  1. Re-classifies the regime from bars seen so far.
  2. Offers any *candidate* signals (as if the legacy engine had emitted
     them) to ``RulesV2Engine.filter(...)`` and records the verdict.
  3. Asks the engine for any NEW ``TREND_CONTINUATION`` candidate and
     records it too.

Outputs are returned as a list of dicts + a summary; if ``output_csv`` /
``output_md`` are provided, writes them to disk.
"""
from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..config import RulesV2Config
from ..engine import EngineInputs, FilterDecision, RulesV2Engine


CandidateSignal = Dict[str, Any]   # {"ts", "signal_type", "direction", "strike", "confidence"}


@dataclass
class ReplayEvent:
    ts: datetime
    kind: str                         # "LEGACY" | "CONTINUATION" | "REGIME"
    signal_type: str
    direction: str
    price: float
    confidence: float
    allowed: bool
    rule: str                         # which filter ruled, or "pass"
    reason: str
    leg_id: int = -1
    regime: str = ""


@dataclass
class ReplayResult:
    events: List[ReplayEvent] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)

    def to_rows(self) -> List[Dict[str, Any]]:
        return [asdict(e) for e in self.events]

    def count(self, **filters) -> int:
        n = 0
        for e in self.events:
            if all(getattr(e, k) == v for k, v in filters.items()):
                n += 1
        return n


def _orb_levels(bars: List[Dict]) -> Tuple[Optional[float], Optional[float]]:
    """Derive ORB high/low from the first 6 bars (09:30–10:00 ET)."""
    if len(bars) < 6:
        return None, None
    first6 = bars[:6]
    return max(b["high"] for b in first6), min(b["low"] for b in first6)


def replay(
    bars: Sequence[Dict],
    candidates: Sequence[CandidateSignal],
    cfg: Optional[RulesV2Config] = None,
    option_quotes: Optional[List] = None,
    output_csv: Optional[str] = None,
    output_md: Optional[str] = None,
) -> ReplayResult:
    """Run a deterministic replay.

    ``bars`` — chronologically ordered 5m bars. Each bar is evaluated after
    the ones preceding it.

    ``candidates`` — legacy signal candidates the replay should route through
    the v2 filter. Each carries a timestamp (``ts``) that maps it to the
    nearest bar. Fields expected: ts, signal_type, direction, strike, confidence.
    """
    cfg = cfg or RulesV2Config(enabled=True)
    engine = RulesV2Engine(cfg)
    engine.begin_session()

    result = ReplayResult()

    # Index candidates by bar position for fast lookup
    cand_by_idx: Dict[int, List[CandidateSignal]] = {}
    for c in candidates:
        ts = c["ts"]
        idx = _nearest_bar_idx(bars, ts)
        cand_by_idx.setdefault(idx, []).append(c)

    for i in range(1, len(bars) + 1):
        window = list(bars[:i])
        last = window[-1]
        spy_price = last["close"]
        now = last["date"]
        orb_hi, orb_lo = _orb_levels(window)

        regime = engine.classify_regime(window, spy_price)
        inputs = EngineInputs(
            bars=window,
            spy_price=spy_price,
            vix=last.get("vix", 19.3),
            iv_rank=last.get("iv_rank", 40.0),
            orb_high=orb_hi,
            orb_low=orb_lo,
            rsi_5m=None,
            vwap=regime.vwap,
            option_quotes=option_quotes,
            now=now,
        )

        # Continuation candidate (NEW signal)
        for cc in engine.generate_additional(window, regime, spy_price, now=now):
            decision = engine.filter(
                signal_type="TREND_CONTINUATION",
                direction=cc.direction,
                price=spy_price,
                confidence=cc.confidence,
                regime=regime,
                inputs=inputs,
            )
            result.events.append(
                ReplayEvent(
                    ts=now,
                    kind="CONTINUATION",
                    signal_type="TREND_CONTINUATION",
                    direction=cc.direction,
                    price=spy_price,
                    confidence=cc.confidence,
                    allowed=decision.allowed,
                    rule=decision.rule,
                    reason=decision.reason,
                    leg_id=decision.leg_id,
                    regime=regime.regime,
                )
            )
            if decision.allowed:
                engine.commit(cc.direction, spy_price, now=now)

        # Legacy candidates scheduled at this bar
        for c in cand_by_idx.get(i - 1, []):
            decision = engine.filter(
                signal_type=c["signal_type"],
                direction=c["direction"],
                price=spy_price,
                confidence=c.get("confidence", 0.80),
                regime=regime,
                inputs=inputs,
            )
            result.events.append(
                ReplayEvent(
                    ts=now,
                    kind="LEGACY",
                    signal_type=c["signal_type"],
                    direction=c["direction"],
                    price=spy_price,
                    confidence=c.get("confidence", 0.80),
                    allowed=decision.allowed,
                    rule=decision.rule,
                    reason=decision.reason,
                    leg_id=decision.leg_id,
                    regime=regime.regime,
                )
            )
            if decision.allowed:
                engine.commit(c["direction"], spy_price, now=now)

    _fill_summary(result)

    if output_csv:
        _write_csv(result, output_csv)
    if output_md:
        _write_markdown(result, output_md)
    return result


def _nearest_bar_idx(bars: Sequence[Dict], ts: datetime) -> int:
    best = 0
    best_delta = None
    for i, b in enumerate(bars):
        d = abs((b["date"] - ts).total_seconds())
        if best_delta is None or d < best_delta:
            best_delta = d
            best = i
    return best


def _fill_summary(result: ReplayResult) -> None:
    summary: Dict[str, int] = {}
    for e in result.events:
        summary[f"total_{e.kind.lower()}"] = summary.get(f"total_{e.kind.lower()}", 0) + 1
        if e.allowed:
            summary[f"allowed_{e.kind.lower()}"] = summary.get(f"allowed_{e.kind.lower()}", 0) + 1
        else:
            summary[f"blocked_{e.kind.lower()}"] = summary.get(f"blocked_{e.kind.lower()}", 0) + 1
            key = f"blocked_by_{e.rule}"
            summary[key] = summary.get(key, 0) + 1
    result.summary = summary


def _write_csv(result: ReplayResult, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = result.to_rows()
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_markdown(result: ReplayResult, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("# rules_v2 replay report\n\n")
        fh.write("## Summary\n\n")
        for k in sorted(result.summary):
            fh.write(f"- `{k}`: {result.summary[k]}\n")
        fh.write("\n## Events\n\n")
        fh.write("| time | kind | signal | dir | price | conf | allowed | rule | reason | regime |\n")
        fh.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for e in result.events:
            fh.write(
                f"| {e.ts.isoformat(timespec='minutes')} | {e.kind} | {e.signal_type} | "
                f"{e.direction} | {e.price:.2f} | {e.confidence:.2f} | "
                f"{'✓' if e.allowed else '✗'} | {e.rule} | {e.reason} | {e.regime} |\n"
            )
