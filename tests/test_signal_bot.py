"""Signal-only bot tests — evaluate() must work offline, with no IB and no orders."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shree.config import Settings
from shree.signal_bot import MesSignalBot


def _make_15m_frame(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    closes = 6000 + np.cumsum(rng.normal(0, 2.0, n))
    highs = closes + np.abs(rng.normal(0, 1.5, n))
    lows = closes - np.abs(rng.normal(0, 1.5, n))
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    volumes = rng.integers(500, 5000, n).astype(float)
    idx = pd.date_range("2026-07-06 13:30", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


@pytest.fixture()
def bot() -> MesSignalBot:
    return MesSignalBot(Settings())


class TestEvaluate:
    def test_returns_complete_record(self, bot: MesSignalBot) -> None:
        record = bot.evaluate(_make_15m_frame())
        for key in (
            "ts", "bar_ts", "signal", "confidence", "entry", "regime",
            "adx", "atr", "htf_30m_trend", "strategy",
        ):
            assert key in record, f"missing {key}"
        assert record["signal"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= record["confidence"] <= 1.0
        assert record["regime"] in ("TRENDING", "TRANSITIONAL", "RANGE")
        assert record["htf_30m_trend"] in ("UP", "DOWN", "NEUTRAL", "UNKNOWN")

    def test_warmup_returns_hold(self, bot: MesSignalBot) -> None:
        record = bot.evaluate(_make_15m_frame(30))
        assert record["signal"] == "HOLD"

    def test_enrich_adds_required_columns(self) -> None:
        enriched = MesSignalBot._enrich(_make_15m_frame())
        assert "MACDhist_12_26_9" in enriched.columns
        assert "PDH" in enriched.columns
        assert "PDL" in enriched.columns
        assert enriched.attrs["htf_30m_trend"] in ("UP", "DOWN", "NEUTRAL", "UNKNOWN")
        # PDH for the second ET day equals the first day's high
        day = pd.Series(enriched.index.tz_convert("US/Eastern").date, index=enriched.index)
        days = sorted(set(day))
        assert len(days) >= 2
        first_day_high = enriched.loc[day == days[0], "high"].max()
        second_day_pdh = enriched.loc[day == days[1], "PDH"].iloc[0]
        assert second_day_pdh == pytest.approx(first_day_high)

    def test_no_order_api_present(self, bot: MesSignalBot) -> None:
        """The bot must have no order-placement surface at all."""
        forbidden = [a for a in dir(bot) if "order" in a.lower() or "position" in a.lower()]
        assert forbidden == []
