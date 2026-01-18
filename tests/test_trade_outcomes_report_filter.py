from tools.trade_outcomes_report import Row


def _apply_exit_reason_filter(rows: list[Row], flt: str) -> list[Row]:
    """Tiny helper mirroring the SQL behavior so we can test semantics cheaply."""
    if flt.startswith("!"):
        ex = flt[1:]
        return [r for r in rows if (r.exit_reason or "") != ex]
    return [r for r in rows if (r.exit_reason or "") == flt]


def test_exit_reason_filter_exact_and_negation():
    rows = [
        Row("a", "MES", None, None, 1, "BACKFILL", -10),
        Row("b", "MES", None, None, 1, "STOP_LOSS", -5),
        Row("c", "MES", None, None, 1, None, 0),
    ]

    only_backfill = _apply_exit_reason_filter(rows, "BACKFILL")
    assert [r.trade_cycle_id for r in only_backfill] == ["a"]

    no_backfill = _apply_exit_reason_filter(rows, "!BACKFILL")
    assert sorted([r.trade_cycle_id for r in no_backfill]) == ["b", "c"]
