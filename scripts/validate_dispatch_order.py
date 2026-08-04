"""Replay validation for deterministic dispatch ordering — RESEARCH ONLY.

Replays real historical candidate batches through the dispatch cap under both
orderings and proves the two invariants that matter:

  I1  When the daily cap does NOT bind, the dispatched SET is identical.
      Ordering alone can never add or drop a signal.
  I2  When the cap DOES bind, the dispatched set differs — and the family mix
      moves from "whatever the rule order was" toward the candidate pool's
      actual composition. That difference is the point of the change.

Batches are reconstructed from logs/blocked_signals.jsonl: rows written in the
same second are one evaluate() batch, and the file order is the original
emission order (expiry-dict x rule order), i.e. the biased baseline.

No production code is modified and nothing here can place an order.
"""
from __future__ import annotations

import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from shree.spy_options.manager import dispatch_order_key  # noqa: E402

LOG = os.path.join(ROOT, "logs", "blocked_signals.jsonl")
CAP = 10


class Row:
    """Minimal signal shim exposing the fields dispatch_order_key() reads."""

    __slots__ = ("signal_type", "strike", "right", "expiry_date", "spy_price")

    def __init__(self, r):
        self.signal_type = r.get("signal_type", "")
        self.strike = float(r.get("strike") or 0.0)
        self.right = r.get("right") or ""
        self.expiry_date = r.get("expiry_date") or ""
        self.spy_price = float(r.get("spy_price") or 0.0)

    @property
    def ident(self):
        return (self.signal_type, self.strike, self.right,
                self.expiry_date, self.spy_price)


def load_batches():
    """-> {day: [ [Row, ...] batch in emission order, ... ] } chronological."""
    days = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(LOG, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            # only rows the confidence gate killed become new candidates
            if not str(r.get("gate", "")).startswith("confidence_threshold"):
                continue
            if not r.get("expiry_date"):
                continue
            days[r["ts"][:10]][r["ts"][:19]].append(Row(r))
    return {d: [b[k] for k in sorted(b)] for d, b in sorted(days.items())}


def simulate(batches, order):
    """Run one session's batches through the daily cap. Returns dispatched."""
    sent, budget = [], CAP
    for batch in batches:
        if budget <= 0:
            break
        ordered = batch if order == "source" else sorted(batch, key=dispatch_order_key)
        for s in ordered:
            if budget <= 0:
                break
            sent.append(s)
            budget -= 1
    return sent


def main():
    per_day = load_batches()
    if not per_day:
        print("no candidate batches found")
        return

    print(f"REPLAY VALIDATION — dispatch ordering   cap={CAP}/day   "
          f"sessions={len(per_day)}\n")

    # ── I1: cap does not bind -> identical dispatched set ────────────────────
    nb = ident = 0
    for day, batches in per_day.items():
        for b in batches:
            if len(b) > CAP:
                continue
            nb += 1
            a = {x.ident for x in simulate([b], "source")}
            c = {x.ident for x in simulate([b], "hash")}
            ident += (a == c)
    print(f"I1  non-binding batches: {ident}/{nb} identical dispatched set "
          f"({'PASS' if ident == nb else 'FAIL'})")

    # ── I2: cap binds -> composition shifts toward the candidate pool ────────
    print(f"\nI2  full-session replay (cap binds every session)")
    print(f"    {'session':<12}{'cands':>7}{'sent':>6}{'same':>6}   family mix "
          f"source -> hash")
    for day, batches in per_day.items():
        cands = [s for b in batches for s in b]
        src = simulate(batches, "source")
        hsh = simulate(batches, "hash")
        same = len({x.ident for x in src} & {x.ident for x in hsh})
        fs = collections.Counter(x.signal_type for x in src)
        fh_ = collections.Counter(x.signal_type for x in hsh)
        print(f"    {day:<12}{len(cands):>7}{len(src):>6}{same:>6}   "
              f"{dict(fs)} -> {dict(fh_)}")
        assert len(src) == len(hsh), "trade-count invariant violated"

    # ── family representation vs the candidate pool ──────────────────────────
    pool = collections.Counter()
    got_src = collections.Counter()
    got_hash = collections.Counter()
    for day, batches in per_day.items():
        pool.update(s.signal_type for b in batches for s in b)
        got_src.update(x.signal_type for x in simulate(batches, "source"))
        got_hash.update(x.signal_type for x in simulate(batches, "hash"))
    tot = sum(pool.values())
    n_src, n_hash = sum(got_src.values()), sum(got_hash.values())
    print(f"\n    FAMILY REPRESENTATION (share of dispatched vs share of pool)")
    print(f"    {'family':<22}{'pool%':>8}{'source%':>9}{'hash%':>8}"
          f"{'|err| src':>11}{'|err| hash':>12}")
    e_src = e_hash = 0.0
    for fam, n in pool.most_common():
        p = 100 * n / tot
        a = 100 * got_src[fam] / n_src if n_src else 0
        b = 100 * got_hash[fam] / n_hash if n_hash else 0
        e_src += abs(a - p)
        e_hash += abs(b - p)
        print(f"    {fam:<22}{p:>8.1f}{a:>9.1f}{b:>8.1f}{abs(a-p):>11.1f}"
              f"{abs(b-p):>12.1f}")
    print(f"\n    total absolute representation error: "
          f"source {e_src:.1f}pp  ->  hash {e_hash:.1f}pp"
          f"   ({'IMPROVED' if e_hash < e_src else 'NOT IMPROVED'})")


if __name__ == "__main__":
    main()
