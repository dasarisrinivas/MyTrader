#!/usr/bin/env python3
"""Generate a human-readable audit report from orders.db:trade_outcomes.

This is designed to answer questions like:
- Are losses dominated by STOP_LOSS vs SIGNAL_EXIT vs TIME_EXIT?
- Is overnight trading systematically worse than RTH?
- What are the worst trades / sessions / reasons?

By default, this reads `data/orders.db` and uses `trade_outcomes` as the canonical
closure table.

Examples:
  python3 tools/trade_outcomes_report.py --days 30
  python3 tools/trade_outcomes_report.py --symbol MES --days 90
  python3 tools/trade_outcomes_report.py --include-backfill
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    # Python 3.9+ (macOS default often 3.9)
    from zoneinfo import ZoneInfo

    CST = ZoneInfo("America/Chicago")
except Exception:  # pragma: no cover
    CST = None


@dataclass
class Row:
    trade_cycle_id: str
    symbol: str
    entry_time: Optional[str]
    exit_time: Optional[str]
    quantity: float
    exit_reason: Optional[str]
    realized_pnl: float


@dataclass
class EnrichedRow(Row):
    root_order_id: Optional[int]
    action: Optional[str]
    order_type: Optional[str]
    market_regime: Optional[str]
    features: dict[str, Any]
    rationale: dict[str, Any]
    order_timestamp: Optional[str]


def _safe_json_loads(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, (bytes, bytearray)):
        try:
            payload = payload.decode("utf-8", errors="ignore")
        except Exception:
            return {}
    if not isinstance(payload, str):
        return {}
    s = payload.strip()
    if not s:
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _first_non_empty(*values: Any) -> Optional[str]:
    for v in values:
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        return str(v)
    return None


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        # Accept both timezone-aware and naive.
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _bucket_hold_minutes(minutes: Optional[float]) -> str:
    if minutes is None or math.isnan(minutes):
        return "UNKNOWN"
    if minutes < 5:
        return "<5m"
    if minutes < 30:
        return "5-30m"
    if minutes < 120:
        return "30-120m"
    return ">=120m"


def _session_label(exit_time: Optional[datetime]) -> str:
    if not exit_time or CST is None:
        return "UNKNOWN"
    local = exit_time.astimezone(CST)
    # Simple heuristic: RTH roughly 08:30-15:00 CST for ES/MES.
    minutes = local.hour * 60 + local.minute
    if 8 * 60 + 30 <= minutes <= 15 * 60:
        return "RTH"
    return "OVERNIGHT"


def _fetch_rows(
    conn: sqlite3.Connection,
    cutoff_iso: Optional[str],
    symbol: Optional[str],
    include_backfill: bool,
    exit_reason_filter: Optional[str],
) -> list[Row]:
    clauses = []
    params: list[object] = []

    if cutoff_iso:
        # Prefer exit_time; fall back to updated_at for rows without exit_time.
        clauses.append("COALESCE(exit_time, updated_at) >= ?")
        params.append(cutoff_iso)

    if symbol:
        clauses.append("symbol = ?")
        params.append(symbol)

    # Backwards-compatible default: exclude BACKFILL unless explicitly included.
    if not include_backfill:
        clauses.append("COALESCE(exit_reason, '') <> 'BACKFILL'")

    # Optional explicit exit_reason filter.
    if exit_reason_filter:
        if exit_reason_filter.startswith("!"):
            clauses.append("COALESCE(exit_reason, '') <> ?")
            params.append(exit_reason_filter[1:])
        else:
            clauses.append("COALESCE(exit_reason, '') = ?")
            params.append(exit_reason_filter)

    where = "WHERE " + " AND ".join(clauses) if clauses else ""

    rows = conn.execute(
        f"""
        SELECT trade_cycle_id, symbol, entry_time, exit_time, quantity,
               exit_reason, COALESCE(realized_pnl, 0) AS realized_pnl
        FROM trade_outcomes
        {where}
        """,
        tuple(params),
    ).fetchall()

    out: list[Row] = []
    for r in rows:
        out.append(
            Row(
                trade_cycle_id=str(r[0]),
                symbol=str(r[1]),
                entry_time=r[2],
                exit_time=r[3],
                quantity=float(r[4] or 0),
                exit_reason=r[5],
                realized_pnl=float(r[6] or 0.0),
            )
        )
    return out


def _fetch_enriched_rows(
    conn: sqlite3.Connection,
    cutoff_iso: Optional[str],
    symbol: Optional[str],
    include_backfill: bool,
    exit_reason_filter: Optional[str],
) -> list[EnrichedRow]:
    clauses = []
    params: list[object] = []

    if cutoff_iso:
        clauses.append("COALESCE(t.exit_time, t.updated_at) >= ?")
        params.append(cutoff_iso)

    if symbol:
        clauses.append("t.symbol = ?")
        params.append(symbol)

    # Backwards-compatible default: exclude BACKFILL unless explicitly included.
    if not include_backfill:
        clauses.append("COALESCE(t.exit_reason, '') <> 'BACKFILL'")

    # Optional explicit exit_reason filter.
    if exit_reason_filter:
        if exit_reason_filter.startswith("!"):
            clauses.append("COALESCE(t.exit_reason, '') <> ?")
            params.append(exit_reason_filter[1:])
        else:
            clauses.append("COALESCE(t.exit_reason, '') = ?")
            params.append(exit_reason_filter)

    where = "WHERE " + " AND ".join(clauses) if clauses else ""

    rows = conn.execute(
        f"""
        SELECT
            t.trade_cycle_id,
            t.symbol,
            t.entry_time,
            t.exit_time,
            t.quantity,
            t.exit_reason,
            COALESCE(t.realized_pnl, 0) AS realized_pnl,
            o.order_id AS root_order_id,
            o.action,
            o.order_type,
            o.market_regime,
            o.features,
                        o.rationale,
                        o.timestamp AS order_timestamp
        FROM trade_outcomes t
        LEFT JOIN orders o
          ON o.trade_cycle_id = t.trade_cycle_id AND o.parent_order_id IS NULL
        {where}
        """,
        tuple(params),
    ).fetchall()

    out: list[EnrichedRow] = []
    for r in rows:
        out.append(
            EnrichedRow(
                trade_cycle_id=str(r[0]),
                symbol=str(r[1]),
                entry_time=r[2],
                exit_time=r[3],
                quantity=float(r[4] or 0),
                exit_reason=r[5],
                realized_pnl=float(r[6] or 0.0),
                root_order_id=(int(r[7]) if r[7] is not None else None),
                action=r[8],
                order_type=r[9],
                market_regime=r[10],
                features=_safe_json_loads(r[11]),
                rationale=_safe_json_loads(r[12]),
                order_timestamp=r[13],
            )
        )
    return out


def _sum(items: Iterable[float]) -> float:
    return float(sum(items))


def _pct(n: int, d: int) -> str:
    if d <= 0:
        return "n/a"
    return f"{(100.0 * n / d):.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db-path",
        default=str(Path(__file__).resolve().parents[1] / "data" / "orders.db"),
    )
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument(
        "--include-backfill",
        action="store_true",
        help="Include trades with exit_reason=BACKFILL (historical backfill rows).",
    )
    parser.add_argument(
        "--exit-reason",
        type=str,
        default=None,
        help=(
            "Filter by exit_reason. Examples: --exit-reason BACKFILL (only backfill), "
            "--exit-reason !BACKFILL (exclude backfill explicitly)."
        ),
    )
    parser.add_argument(
        "--feature-audit",
        action="store_true",
        help="Join outcomes to root order features/rationale and print diagnostics for <5m vs >=5m trades.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=12,
        help="How many values to show for top categories in feature audit.",
    )
    args = parser.parse_args()

    cutoff = datetime.now(timezone.utc) - timedelta(days=int(args.days))
    cutoff_iso = cutoff.replace(microsecond=0).isoformat()

    db_path = Path(args.db_path)
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if args.feature_audit:
            rows = _fetch_enriched_rows(
                conn,
                cutoff_iso=cutoff_iso,
                symbol=args.symbol,
                include_backfill=bool(args.include_backfill),
                exit_reason_filter=args.exit_reason,
            )
        else:
            rows = _fetch_rows(
                conn,
                cutoff_iso=cutoff_iso,
                symbol=args.symbol,
                include_backfill=bool(args.include_backfill),
                exit_reason_filter=args.exit_reason,
            )

    print("=" * 88)
    print(f"Trade outcomes report | db={db_path} | days={args.days} | symbol={args.symbol or 'ALL'}")
    print(
        f"Rows included: {len(rows)} (include_backfill={bool(args.include_backfill)}"
        + (f", exit_reason_filter={args.exit_reason}" if args.exit_reason else "")
        + ")"
    )
    print("=" * 88)

    if not rows:
        print("No rows matched (try --include-backfill or increase --days).")
        return 0

    pnls = [r.realized_pnl for r in rows]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    print("\n## Overall")
    print(f"Total PnL: {_sum(pnls):.2f}")
    print(f"Win rate: {len(wins)}/{len(pnls)} = {_pct(len(wins), len(pnls))}")
    print(f"Avg win: {(sum(wins)/len(wins)):.2f}" if wins else "Avg win: n/a")
    print(f"Avg loss: {(sum(losses)/len(losses)):.2f}" if losses else "Avg loss: n/a")

    if wins and losses:
        expectancy = (sum(wins) + sum(losses)) / len(pnls)
        print(f"Expectancy / trade: {expectancy:.2f}")

    # Breakdown by exit_reason
    by_reason: dict[str, list[Row]] = {}
    for r in rows:
        key = (r.exit_reason or "UNKNOWN")
        by_reason.setdefault(key, []).append(r)

    print("\n## Breakdown by exit_reason")
    for reason, rs in sorted(by_reason.items(), key=lambda kv: len(kv[1]), reverse=True):
        total = _sum([x.realized_pnl for x in rs])
        w = sum(1 for x in rs if x.realized_pnl > 0)
        l = sum(1 for x in rs if x.realized_pnl < 0)
        print(f"- {reason:14s} n={len(rs):4d} pnl={total:9.2f} win%={_pct(w, len(rs))} (w={w}, l={l})")

    # Breakdown by session
    by_session: dict[str, list[Row]] = {}
    for r in rows:
        et = _parse_iso(r.exit_time)
        by_session.setdefault(_session_label(et), []).append(r)

    print("\n## Breakdown by session (based on exit_time CST)")
    for sess, rs in sorted(by_session.items(), key=lambda kv: len(kv[1]), reverse=True):
        total = _sum([x.realized_pnl for x in rs])
        w = sum(1 for x in rs if x.realized_pnl > 0)
        print(f"- {sess:10s} n={len(rs):4d} pnl={total:9.2f} win%={_pct(w, len(rs))}")

    # Hold-time buckets
    by_hold: dict[str, list[Row]] = {}
    for r in rows:
        et = _parse_iso(r.exit_time)
        it = _parse_iso(r.entry_time)
        mins = None
        if et and it:
            mins = (et - it).total_seconds() / 60.0
        by_hold.setdefault(_bucket_hold_minutes(mins), []).append(r)

    print("\n## Hold-time buckets")
    for bucket, rs in sorted(by_hold.items(), key=lambda kv: len(kv[1]), reverse=True):
        total = _sum([x.realized_pnl for x in rs])
        print(f"- {bucket:10s} n={len(rs):4d} pnl={total:9.2f}")

    # Worst trades
    print("\n## Worst 10 trades")
    for r in sorted(rows, key=lambda x: x.realized_pnl)[:10]:
        print(
            f"- {r.trade_cycle_id} {r.symbol} pnl={r.realized_pnl:8.2f} reason={r.exit_reason or 'UNKNOWN'} exit_time={r.exit_time}"
        )

    if args.feature_audit:
        erows: list[EnrichedRow] = rows  # type: ignore[assignment]

        def hold_minutes(x: EnrichedRow) -> Optional[float]:
            et = _parse_iso(x.exit_time)
            it = _parse_iso(x.entry_time)
            if not et or not it:
                return None
            return (et - it).total_seconds() / 60.0

        short: list[EnrichedRow] = []
        longish: list[EnrichedRow] = []
        for rr in erows:
            mins = hold_minutes(rr)
            if mins is None:
                continue
            (short if mins < 5 else longish).append(rr)

        def _pnl(rs: list[EnrichedRow]) -> float:
            return _sum([x.realized_pnl for x in rs])

        print("\n## Feature audit (joined to root order)")
        print(
            f"Enriched rows: {len(erows)} | with root orders: {sum(1 for x in erows if x.root_order_id is not None)}"
        )
        print(f"<5m trades: n={len(short)} pnl={_pnl(short):.2f}")
        print(f">=5m trades: n={len(longish)} pnl={_pnl(longish):.2f}")

        # Coverage diagnostics: are we losing because we're trading without context?
        missing_feat_short = sum(1 for x in short if not x.features)
        missing_rat_short = sum(1 for x in short if not x.rationale)
        missing_feat_long = sum(1 for x in longish if not x.features)
        missing_rat_long = sum(1 for x in longish if not x.rationale)
        print(
            "Coverage: "
            f"<5m missing features={missing_feat_short}/{len(short)} | missing rationale={missing_rat_short}/{len(short)}; "
            f">=5m missing features={missing_feat_long}/{len(longish)} | missing rationale={missing_rat_long}/{len(longish)}"
        )

        def top_cats(rs: list[EnrichedRow], getter, top_n: int) -> list[tuple[str, int, float]]:
            buckets: dict[str, list[EnrichedRow]] = {}
            for x in rs:
                try:
                    key = getter(x)
                except Exception:
                    key = None
                if key is None:
                    k = "UNKNOWN"
                else:
                    k = str(key)
                buckets.setdefault(k, []).append(x)
            scored: list[tuple[str, int, float]] = []
            for k, xs in buckets.items():
                scored.append((k, len(xs), _pnl(xs)))
            scored.sort(key=lambda t: (t[1], abs(t[2])), reverse=True)
            return scored[:top_n]

        def print_cat_block(name: str, getter) -> None:
            print(f"\n### {name} (top {int(args.top_n)})")
            print("<5m:")
            for k, n, p in top_cats(short, getter, int(args.top_n)):
                print(f"- {k:24s} n={n:4d} pnl={p:9.2f}")
            print(">=5m:")
            for k, n, p in top_cats(longish, getter, int(args.top_n)):
                print(f"- {k:24s} n={n:4d} pnl={p:9.2f}")

        print_cat_block("market_regime", lambda x: x.market_regime)
        print_cat_block("market_trend (rationale)", lambda x: x.rationale.get("market_trend"))
        print_cat_block("volatility_regime (rationale)", lambda x: x.rationale.get("volatility_regime"))
        print_cat_block("confidence_band (rationale)", lambda x: x.rationale.get("confidence_band"))
        print_cat_block("filters_blocked count", lambda x: len(x.rationale.get("filters_blocked") or []))
        print_cat_block(
            "has RSI_OVERBOUGHT filter",
            lambda x: "RSI_OVERBOUGHT" in (x.rationale.get("filters_passed") or []),
        )
        print_cat_block(
            "has RSI_OVERSOLD filter",
            lambda x: "RSI_OVERSOLD" in (x.rationale.get("filters_passed") or []),
        )

        def bucket_rsi(x: EnrichedRow) -> str:
            v = x.features.get("rsi")
            try:
                fv = float(v)
            except Exception:
                return "UNKNOWN"
            if fv < 30:
                return "<30"
            if fv < 50:
                return "30-50"
            if fv < 70:
                return "50-70"
            return ">=70"

        print_cat_block("RSI bucket (features)", bucket_rsi)

        def bucket_decision_conf(x: EnrichedRow) -> str:
            v = x.rationale.get("decision_confidence")
            try:
                fv = float(v)
            except Exception:
                return "UNKNOWN"
            if fv < 0.45:
                return "<0.45"
            if fv < 0.55:
                return "0.45-0.55"
            if fv < 0.65:
                return "0.55-0.65"
            return ">=0.65"

        print_cat_block("decision_confidence bucket (rationale)", bucket_decision_conf)

        # Strategy/path fingerprinting (comes from rationale when present)
        print_cat_block(
            "strategy_name (rationale)",
            lambda x: x.rationale.get("strategy_name"),
        )
        print_cat_block(
            "protection_source (rationale)",
            lambda x: x.rationale.get("protection_source"),
        )
        print_cat_block(
            "atr_fallback_used (rationale)",
            lambda x: x.rationale.get("atr_fallback_used"),
        )

        # Time clustering: bucket by trading day (from order timestamp if present, else entry_time)
        def day_bucket(x: EnrichedRow) -> str:
            ts = _first_non_empty(getattr(x, "order_timestamp", None), x.entry_time, x.exit_time)
            if not ts:
                return "UNKNOWN"
            dt = _parse_iso(ts)
            if not dt:
                return "UNKNOWN"
            return dt.date().isoformat()

        print_cat_block("day bucket", day_bucket)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
