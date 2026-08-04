"""Deterministic dispatch ordering (AUG 3 2026).

Locks the properties the confidence-gate experiment depends on: the order in
which candidates compete for `max_signals_per_day` must be deterministic,
reproducible across processes, and independent of the order the signal engine
happened to emit them in.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from shree.spy_options.manager import dispatch_identity, dispatch_order_key
from shree.spy_options.signal_engine import SignalType

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Sig:
    def __init__(self, fam, strike, right="C", expiry_date="20260804", spy=638.5):
        self.signal_type = fam
        self.strike = strike
        self.right = right
        self.expiry_date = expiry_date
        self.spy_price = spy


FAMILIES = [SignalType.PC_RATIO_EXTREME, SignalType.CALL_SWEEP,
            SignalType.PUT_SWEEP, SignalType.ORB_BREAKOUT,
            SignalType.TREND_CONTINUATION, SignalType.BULL_CALL_SPREAD]


def _batch():
    return [Sig(f, 630 + i) for i, f in enumerate(FAMILIES)]


# ── determinism ──────────────────────────────────────────────────────────────

def test_order_is_stable_across_calls():
    b = _batch()
    assert [dispatch_identity(s) for s in sorted(b, key=dispatch_order_key)] == \
           [dispatch_identity(s) for s in sorted(b, key=dispatch_order_key)]


def test_order_independent_of_input_order():
    """The whole point: emission order must not survive into dispatch order."""
    b = _batch()
    a = [dispatch_identity(s) for s in sorted(b, key=dispatch_order_key)]
    c = [dispatch_identity(s) for s in sorted(list(reversed(b)), key=dispatch_order_key)]
    assert a == c


def test_order_is_reproducible_in_a_fresh_process():
    """builtin hash() is salted by PYTHONHASHSEED — using it would make live
    and replay disagree. This test fails loudly if anyone swaps it in."""
    code = "\n".join((
        "import sys",
        "sys.path.insert(0, %r)" % REPO,
        "from shree.spy_options.manager import dispatch_order_key",
        "from shree.spy_options.signal_engine import SignalType",
        "class S:",
        "    def __init__(self, f, k):",
        "        self.signal_type = f",
        "        self.strike = k",
        "        self.right = 'C'",
        "        self.expiry_date = '20260804'",
        "        self.spy_price = 638.5",
        "b = [S(SignalType.CALL_SWEEP, 630), S(SignalType.ORB_BREAKOUT, 631)]",
        "print([dispatch_order_key(x)[0] for x in sorted(b, key=dispatch_order_key)])",
    ))
    outs = set()
    for seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, env=env, cwd=REPO)
        assert r.returncode == 0, r.stderr
        outs.add(r.stdout.strip())
    assert len(outs) == 1, f"ordering varies with PYTHONHASHSEED: {outs}"


# ── the key is a total order and is replayable ───────────────────────────────

def test_key_is_a_total_order_no_positional_fallback():
    b = _batch()
    keys = [dispatch_order_key(s) for s in b]
    assert len(set(keys)) == len(keys)


def test_identity_uses_only_persisted_fields():
    """Every component must exist on spy_signals so historical dispatch can be
    re-derived offline without extra state."""
    s = Sig(SignalType.CALL_SWEEP, 640.0)
    ident = dispatch_identity(s)
    assert ident == "CALL_SWEEP|640.00|C|20260804|638.50"


def test_ordering_varies_between_poll_cycles():
    """A key fixed across cycles would let one family win every cycle — the
    original bias in a new form. spy_price makes the permutation move."""
    b1 = [Sig(f, 630 + i, spy=638.50) for i, f in enumerate(FAMILIES)]
    b2 = [Sig(f, 630 + i, spy=639.10) for i, f in enumerate(FAMILIES)]
    fam = lambda xs: [x.signal_type for x in sorted(xs, key=dispatch_order_key)]
    assert fam(b1) != fam(b2)


# ── it is a pure permutation ─────────────────────────────────────────────────

def test_sort_adds_and_removes_nothing():
    b = _batch()
    out = sorted(b, key=dispatch_order_key)
    assert len(out) == len(b)
    assert {id(x) for x in out} == {id(x) for x in b}


def test_no_family_is_systematically_first():
    """Across many cycles the first slot should not belong to one family."""
    firsts = set()
    for i in range(60):
        b = [Sig(f, 630 + j, spy=630 + i * 0.13) for j, f in enumerate(FAMILIES)]
        firsts.add(sorted(b, key=dispatch_order_key)[0].signal_type)
    assert len(firsts) >= 4, f"ordering favours {firsts}"


# ── config wiring ────────────────────────────────────────────────────────────

def test_flag_defaults_on_and_is_revertible():
    from shree.config.spy_options import SpyOptionsSignalConfig
    assert SpyOptionsSignalConfig().deterministic_dispatch_order is True
    assert SpyOptionsSignalConfig(
        deterministic_dispatch_order=False).deterministic_dispatch_order is False
