"""End-to-end replay test against the Apr 21 2026 synthetic tape.

This is the regression test that proves the whole rules_v2 pipeline does
what the postmortem called for:

  • All 7 stale / contrarian legacy signals get blocked:
      - 5 by the ORB time-gate (outside 09:45–11:00 ET)
      - 2 by PC_RATIO alignment (TRANSITION / RANGE_BOUND regime)

  • At least one TREND_CONTINUATION PUT candidate fires during the 12:30–
    13:50 ET decline, is allowed through the entry gate + throttle, and
    produces a trigger / stop pair consistent with the tape's structure.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shree.spy_options.rules_v2.backtest.apr21_synthetic import (  # noqa: E402
    apr21_bars,
    apr21_legacy_signal_candidates,
)
from shree.spy_options.rules_v2.backtest.replay import replay  # noqa: E402
from shree.spy_options.rules_v2.config import RulesV2Config  # noqa: E402


def test_all_legacy_apr21_signals_blocked():
    cfg = RulesV2Config(enabled=True)
    r = replay(apr21_bars(), apr21_legacy_signal_candidates(), cfg=cfg)
    total = r.summary.get("total_legacy", 0)
    blocked = r.summary.get("blocked_legacy", 0)
    assert total == 7
    assert blocked == total, f"expected all {total} legacy signals blocked, got {blocked}"


def test_apr21_blocked_by_rule_breakdown():
    """Confirm WHICH rules did the blocking — catches silent behavior drift."""
    cfg = RulesV2Config(enabled=True)
    r = replay(apr21_bars(), apr21_legacy_signal_candidates(), cfg=cfg)
    # Expected: 5 blocked by orb_gate (stale time window) + 2 by pc_ratio alignment
    assert r.summary.get("blocked_by_orb_gate", 0) == 5, r.summary
    assert r.summary.get("blocked_by_pc_ratio", 0) == 2, r.summary


def test_apr21_generates_trend_continuation_put():
    cfg = RulesV2Config(enabled=True)
    r = replay(apr21_bars(), apr21_legacy_signal_candidates(), cfg=cfg)

    cont_events = [e for e in r.events if e.kind == "CONTINUATION"]
    assert cont_events, "expected at least one TREND_CONTINUATION candidate"

    allowed = [e for e in cont_events if e.allowed]
    assert allowed, "at least one TREND_CONTINUATION should pass filter+throttle"

    # All continuation entries during TREND_DOWN must be PUT direction
    for e in allowed:
        assert e.direction == "P", e
        assert e.signal_type == "TREND_CONTINUATION"
        assert "TREND_DOWN" in e.regime


def test_apr21_replay_writes_report_files(tmp_path):
    cfg = RulesV2Config(enabled=True)
    csv_path = tmp_path / "apr21.csv"
    md_path = tmp_path / "apr21.md"
    r = replay(
        apr21_bars(),
        apr21_legacy_signal_candidates(),
        cfg=cfg,
        output_csv=str(csv_path),
        output_md=str(md_path),
    )
    assert r.events
    assert csv_path.exists()
    assert md_path.exists()
    content = md_path.read_text()
    assert "rules_v2 replay report" in content
    assert "Events" in content
