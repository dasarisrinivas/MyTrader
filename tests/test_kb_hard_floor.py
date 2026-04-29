"""Tests for APR 29 2026 local KB hard floor.

The hard floor is wired in signal_processor.py around the existing soft overlay.
These tests validate the gate logic in isolation — the actual wiring path runs
inside _process_with_hybrid which is async and tightly coupled to many manager
attrs, so we test the *decision rule* directly to keep the test fast and
deterministic.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _decide(kb_n, kb_wr, threshold_wr=0.20, min_n=10):
    """Replicates the hard-floor predicate exactly as written in signal_processor."""
    return (
        threshold_wr > 0
        and min_n > 0
        and kb_n >= min_n
        and kb_wr < threshold_wr
    )


class TestKbHardFloorPredicate:
    """Exact rule: block iff (n >= min_n) AND (wr < threshold_wr) AND both knobs > 0."""

    def test_blocks_today_trade_6_scenario(self):
        # Trade 6 was 14% WR on 14 similar trades — the case the floor is for.
        assert _decide(kb_n=14, kb_wr=0.14) is True

    def test_passes_at_threshold_boundary_wr(self):
        # wr == threshold should NOT block (strict <)
        assert _decide(kb_n=14, kb_wr=0.20) is False

    def test_passes_just_above_threshold(self):
        assert _decide(kb_n=14, kb_wr=0.21) is False

    def test_blocks_just_below_threshold(self):
        assert _decide(kb_n=14, kb_wr=0.19) is True

    def test_passes_below_min_n_even_with_terrible_wr(self):
        # 9 trades is below the actionable sample size — noise, don't gate.
        assert _decide(kb_n=9, kb_wr=0.0) is False

    def test_blocks_at_min_n_exactly(self):
        # n == min_n should block (>=)
        assert _decide(kb_n=10, kb_wr=0.10) is True

    def test_disabled_via_zero_threshold(self):
        # Setting threshold_wr=0 disables the gate.
        assert _decide(kb_n=100, kb_wr=0.05, threshold_wr=0.0) is False

    def test_disabled_via_zero_min_n(self):
        # Setting min_n=0 disables the gate.
        assert _decide(kb_n=100, kb_wr=0.05, min_n=0) is False

    def test_perfect_history_passes(self):
        assert _decide(kb_n=50, kb_wr=1.0) is False

    def test_aggressive_threshold(self):
        # If user wants a tighter gate at 35%, the same rule applies.
        assert _decide(kb_n=20, kb_wr=0.30, threshold_wr=0.35) is True
        assert _decide(kb_n=20, kb_wr=0.40, threshold_wr=0.35) is False
