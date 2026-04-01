"""Backtest / validation of SPY Options signal bot against docs/SPY_OPTIONS_HOW_IT_WORKS.md.

Tests every documented behavior:
  1. Regime detection (6 regimes with correct priority)
  2. Sentiment scoring (3-component weighted model, labels, clamp)
  3. Liquidity filters (OI, spread, volume gates)
  4. Volume spike detection (cumulative-to-incremental, 4× rolling avg)
  5. Sweep tracker (repeat sweep scoring: 0/0.5/1.0 at 1/2/3 hits)
  6. Signal types (all 7)
  7. Weighted confidence model (10 components, weights match docs)
  8. Confidence tiers (MEDIUM/HIGH/EXTREME at 70/80/90)
  9. Signal dedup keys
 10. End-to-end scenario: TREND_UP with call sweep → CALL_SWEEP + BULL_CALL_SPREAD

Run with:
    python3 -m pytest tests/test_spy_options_backtest.py -v
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shree.spy_options.regime_detector import RegimeDetector, RegimeContext
from shree.spy_options.sentiment_engine import SentimentEngine, SentimentContext
from shree.spy_options.chain_builder import (
    ChainSnapshot,
    OptionQuote,
    VolumeTracker,
    passes_liquidity,
)
from shree.spy_options.sweep_tracker import SweepTracker
from shree.spy_options.signal_engine import (
    SignalEngine,
    SignalContext,
    SignalType,
    SpySignal,
    _tier,
)
from shree.config.spy_options import SpyOptionsSignalConfig


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers: synthetic 5-min bars
# ═══════════════════════════════════════════════════════════════════════════════

def _make_bars(
    n: int = 30,
    base_close: float = 560.0,
    trend: float = 0.0,     # per-bar price change
    atr_factor: float = 1.0,
    volume: int = 100_000,
) -> list[dict]:
    """Generate synthetic 5-min bars for regime/sentiment testing."""
    bars = []
    base_time = datetime(2026, 3, 30, 9, 35)
    for i in range(n):
        c = base_close + trend * i
        spread = 0.5 * atr_factor
        bars.append({
            "date": base_time + timedelta(minutes=i * 5),
            "open": c - spread / 2,
            "high": c + spread,
            "low": c - spread,
            "close": c,
            "volume": volume,
        })
    return bars


def _make_quote(
    strike: float = 560.0,
    right: str = "C",
    conid: int = 1001,
    volume: int = 5000,
    oi: int = 8000,
    bid: float = 2.50,
    ask: float = 2.60,
    bid_size: int = 400,
    ask_size: int = 100,
    delta: float = 0.45,
    gamma: float = 0.03,
    theta: float = -0.08,
    vega: float = 0.15,
    impl_vol: float = 0.18,
) -> OptionQuote:
    return OptionQuote(
        conid=conid,
        symbol=f"SPY APR26 {strike:.0f}{right}",
        strike=strike,
        right=right,
        expiry_month="APR26",
        bid=bid,
        ask=ask,
        last=(bid + ask) / 2,
        bid_size=bid_size,
        ask_size=ask_size,
        volume=volume,
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
        impl_vol=impl_vol,
        open_interest=oi,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Regime Detection
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegimeDetector:
    """Validate all 6 regimes with correct priority ordering."""

    def setup_method(self):
        self.det = RegimeDetector()

    def test_trend_up(self):
        """EMA9 > EMA21, positive slope, price above VWAP → TREND_UP."""
        bars = _make_bars(30, base_close=560.0, trend=0.15, atr_factor=1.0)
        ctx = self.det.classify(bars, spy_price=564.5, vix=18.0)
        assert ctx.regime == "TREND_UP"
        assert ctx.ema9 > ctx.ema21
        assert ctx.ema_slope > 0
        assert ctx.spy_vs_vwap > 0

    def test_trend_down(self):
        """EMA9 < EMA21, negative slope, price below VWAP → TREND_DOWN."""
        bars = _make_bars(30, base_close=560.0, trend=-0.15, atr_factor=1.0)
        ctx = self.det.classify(bars, spy_price=555.5, vix=18.0)
        assert ctx.regime == "TREND_DOWN"
        assert ctx.ema9 < ctx.ema21
        assert ctx.ema_slope < 0

    def test_range_bound(self):
        """No clear trend → RANGE_BOUND (default)."""
        bars = _make_bars(30, base_close=560.0, trend=0.0, atr_factor=1.0)
        ctx = self.det.classify(bars, spy_price=560.0, vix=18.0)
        assert ctx.regime == "RANGE_BOUND"

    def test_high_vol_vix(self):
        """VIX > 26 → HIGH_VOL regardless of trend."""
        bars = _make_bars(30, base_close=560.0, trend=0.15)
        ctx = self.det.classify(bars, spy_price=564.5, vix=30.0)
        assert ctx.regime == "HIGH_VOL"

    def test_high_vol_atr_expanded(self):
        """ATR expanded 2× median → HIGH_VOL (but < 2.5× so not NEWS_DRIVEN)."""
        # 35 baseline bars (low ATR) then 8 expanded bars. The lookback median
        # stays near 0.5 (baseline) while Wilder-smoothed ATR14 reaches ~1.17
        # (ratio ≈ 2.34, inside [2.0, 2.5) → HIGH_VOL, not NEWS_DRIVEN).
        # VIX=20 (below 26) so only the ATR path can fire HIGH_VOL.
        bars = _make_bars(35, base_close=560.0, trend=0.0, atr_factor=0.5)
        for i in range(8):
            c = 560.0
            bars.append({
                "date": datetime(2026, 3, 30, 14, 0) + timedelta(minutes=i * 5),
                "open": c - 0.5,
                "high": c + 1.0,
                "low": c - 1.0,
                "close": c,
                "volume": 200_000,
            })
        ctx = self.det.classify(bars, spy_price=560.0, vix=20.0)
        assert ctx.regime == "HIGH_VOL", (
            f"Expected HIGH_VOL from ATR expansion, got {ctx.regime} "
            f"(atr14={ctx.atr14:.4f})"
        )

    def test_low_vol(self):
        """VIX < 12 AND ATR compressed < 0.5× median → LOW_VOL."""
        # Build bars where the first 22 have a normal-ish ATR (atr_factor=2.0),
        # then the final bars are extremely compressed. ATR14 uses Wilder smoothing
        # so we need many compressed bars to drag atr_last well below 0.5× median.
        # Strategy: 22 bars with atr_factor=1.0 then 20 bars of near-zero range.
        bars = _make_bars(22, base_close=560.0, atr_factor=1.0)
        for i in range(20):
            c = 560.0
            bars.append({
                "date": datetime(2026, 3, 30, 12, 0) + timedelta(minutes=i * 5),
                "open": c,
                "high": c + 0.01,
                "low": c - 0.01,
                "close": c,
                "volume": 50_000,
            })
        ctx = self.det.classify(bars, spy_price=560.0, vix=10.0)
        assert ctx.regime == "LOW_VOL", (
            f"Expected LOW_VOL (vix=10 < 12, compressed ATR), got {ctx.regime} "
            f"(atr14={ctx.atr14:.4f})"
        )

    def test_news_driven(self):
        """ATR blow-up > 2.5× recent median → NEWS_DRIVEN (highest priority)."""
        bars = _make_bars(25, base_close=560.0, atr_factor=0.3)
        for i in range(5):
            c = 560.0
            bars.append({
                "date": datetime(2026, 3, 30, 12, 0 + i * 5),
                "open": c - 5,
                "high": c + 6,
                "low": c - 6,
                "close": c,
                "volume": 500_000,
            })
        ctx = self.det.classify(bars, spy_price=560.0, vix=18.0)
        assert ctx.regime == "NEWS_DRIVEN"

    def test_insufficient_bars_fallback(self):
        """< 22 bars → RANGE_BOUND fallback."""
        bars = _make_bars(10, base_close=560.0)
        ctx = self.det.classify(bars, spy_price=560.0, vix=18.0)
        assert ctx.regime == "RANGE_BOUND"

    def test_news_driven_priority_over_high_vol(self):
        """NEWS_DRIVEN should take priority over HIGH_VOL (even with high VIX)."""
        bars = _make_bars(25, base_close=560.0, atr_factor=0.3)
        for i in range(5):
            c = 560.0
            bars.append({
                "date": datetime(2026, 3, 30, 12, 0 + i * 5),
                "open": c - 5,
                "high": c + 6,
                "low": c - 6,
                "close": c,
                "volume": 500_000,
            })
        # Even with extreme VIX, NEWS_DRIVEN should win
        ctx = self.det.classify(bars, spy_price=560.0, vix=35.0)
        assert ctx.regime == "NEWS_DRIVEN"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Sentiment Scoring
# ═══════════════════════════════════════════════════════════════════════════════

class TestSentimentEngine:
    """Validate 3-component weighted sentiment (-100 to +100)."""

    def setup_method(self):
        self.eng = SentimentEngine()

    def _regime(self, spy_vs_vwap=2.0, ema_slope=0.05, atr=1.0) -> RegimeContext:
        return RegimeContext(
            regime="TREND_UP", ema9=562.0, ema21=560.0, atr14=atr,
            vwap=560.0, spy_vs_vwap=spy_vs_vwap, ema_slope=ema_slope,
            timestamp=datetime.utcnow(),
        )

    def test_bullish_sentiment(self):
        """Falling VIX + above VWAP + positive slope → BULLISH (> +25)."""
        ctx = self.eng.score(
            vix=14.0,
            vix_history=[16.0, 15.5, 15.0, 14.8, 14.5],  # falling
            regime=self._regime(spy_vs_vwap=1.5, ema_slope=0.08, atr=1.0),
        )
        assert ctx.label == "BULLISH"
        assert ctx.score > 25.0
        assert ctx.vix_trend == "FALLING"

    def test_bearish_sentiment(self):
        """Rising VIX + below VWAP + negative slope → BEARISH (< -25)."""
        ctx = self.eng.score(
            vix=22.0,
            vix_history=[18.0, 19.0, 19.5, 20.0, 20.5],  # rising
            regime=self._regime(spy_vs_vwap=-2.0, ema_slope=-0.1, atr=1.0),
        )
        assert ctx.label == "BEARISH"
        assert ctx.score < -25.0
        assert ctx.vix_trend == "RISING"

    def test_neutral_sentiment(self):
        """Flat VIX + near VWAP + flat slope → NEUTRAL (-25 to +25)."""
        ctx = self.eng.score(
            vix=16.0,
            vix_history=[16.0, 16.1, 15.9, 16.0, 16.0],  # flat
            regime=self._regime(spy_vs_vwap=0.1, ema_slope=0.001, atr=1.0),
        )
        assert ctx.label == "NEUTRAL"
        assert -25.0 <= ctx.score <= 25.0

    def test_score_clamped(self):
        """Score must be clamped to [-100, +100]."""
        ctx = self.eng.score(
            vix=30.0,
            vix_history=[15.0, 15.0, 15.0, 15.0, 15.0],  # massive rise
            regime=self._regime(spy_vs_vwap=-10.0, ema_slope=-1.0, atr=0.1),
        )
        assert ctx.score >= -100.0
        assert ctx.score <= 100.0

    def test_vix_weight_40pct(self):
        """VIX component accounts for 40% → max ±40 points."""
        # Max bullish VIX (falling 6%+), neutralize other components
        ctx = self.eng.score(
            vix=14.0,
            vix_history=[16.0, 15.5, 15.3, 15.1, 15.0],  # strong fall
            regime=self._regime(spy_vs_vwap=0.0, ema_slope=0.0, atr=1.0),
        )
        # VIX component should dominate; score ≈ +40 ±10 from rounding
        assert 25.0 < ctx.score < 55.0

    def test_insufficient_vix_history(self):
        """< 3 VIX readings → VIX component = 0 (neutral)."""
        ctx = self.eng.score(
            vix=22.0,
            vix_history=[20.0],  # too few
            regime=self._regime(spy_vs_vwap=0.0, ema_slope=0.0, atr=1.0),
        )
        # Without VIX trend, VWAP and slope are both 0 → should be near zero
        assert abs(ctx.score) < 10.0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Liquidity Filters
# ═══════════════════════════════════════════════════════════════════════════════

class TestLiquidityFilters:
    """Validate the 3-gate liquidity filter from docs."""

    def test_passes_all(self):
        """Contract with good OI, spread, volume → passes."""
        q = _make_quote(oi=5000, bid=2.50, ask=2.60, volume=2000)
        assert passes_liquidity(q) is True

    def test_fails_low_oi(self):
        """OI < 1000 → blocked."""
        q = _make_quote(oi=500)
        assert passes_liquidity(q, min_oi=1000) is False

    def test_fails_wide_spread(self):
        """Spread > 8% of mid → blocked."""
        q = _make_quote(bid=2.00, ask=2.50)  # spread = 0.50, mid = 2.25, 22%
        assert passes_liquidity(q, max_spread_pct=8.0) is False

    def test_fails_low_volume(self):
        """Volume < 500 → blocked."""
        q = _make_quote(volume=200)
        assert passes_liquidity(q, min_volume=500) is False

    def test_zero_oi_passes(self):
        """OI=0 (unknown) should pass — only filter when OI is reported but low."""
        q = _make_quote(oi=0, volume=2000)
        assert passes_liquidity(q) is True

    def test_zero_volume_passes(self):
        """Volume=0 should pass (volume gate only activates when volume > 0)."""
        q = _make_quote(oi=5000, volume=0)
        assert passes_liquidity(q) is True


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Volume Tracker (Cumulative → Incremental)
# ═══════════════════════════════════════════════════════════════════════════════

class TestVolumeTracker:
    """Validate cumulative-to-incremental conversion and spike detection."""

    def test_incremental_deltas(self):
        """Documented example: 5000 → 5800 → 6100 → 9500."""
        vt = VolumeTracker()
        d1 = vt.update(conid=1, current_volume=5000)
        assert d1 == 0  # first poll, no prior baseline

        d2 = vt.update(conid=1, current_volume=5800)
        assert d2 == 800

        d3 = vt.update(conid=1, current_volume=6100)
        assert d3 == 300

        d4 = vt.update(conid=1, current_volume=9500)
        assert d4 == 3400  # spike!

    def test_rolling_avg_excludes_latest(self):
        """Rolling avg uses history[:-1] — excludes the current delta."""
        vt = VolumeTracker()
        vt.update(1, 5000)    # delta=0
        vt.update(1, 5800)    # delta=800
        vt.update(1, 6100)    # delta=300
        vt.update(1, 9500)    # delta=3400

        # history = [0, 800, 300, 3400]
        # rolling_avg = avg([0, 800, 300]) = 366.67
        avg = vt.rolling_avg(1)
        assert 360 < avg < 370

    def test_spike_detection(self):
        """3400 / 366.67 ≈ 9.3× → spike (≥ 4×) and ≥ 300 floor."""
        vt = VolumeTracker()
        vt.update(1, 5000)
        vt.update(1, 5800)
        vt.update(1, 6100)
        increment = vt.update(1, 9500)
        avg = vt.rolling_avg(1)

        spike_mult = increment / avg if avg > 1 else 0.0
        assert spike_mult >= 4.0  # documented threshold
        assert increment >= 300   # documented floor


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Sweep Tracker (Repeat Sweep Detection)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSweepTracker:
    """Validate flow scoring: 0.0/0.5/1.0 at 1/2/3 hits."""

    def test_single_sweep_no_boost(self):
        """1 hit → flow score 0.0."""
        st = SweepTracker(window_minutes=15)
        st.record(560.0, "C", "APR26")
        assert st.flow_score(560.0, "C", "APR26") == 0.0

    def test_double_sweep(self):
        """2 hits within 15 min → flow score 0.5."""
        st = SweepTracker(window_minutes=15)
        st.record(560.0, "C", "APR26")
        st.record(560.0, "C", "APR26")
        assert st.flow_score(560.0, "C", "APR26") == 0.5

    def test_triple_sweep(self):
        """3+ hits within 15 min → flow score 1.0."""
        st = SweepTracker(window_minutes=15)
        st.record(560.0, "C", "APR26")
        st.record(560.0, "C", "APR26")
        st.record(560.0, "C", "APR26")
        assert st.flow_score(560.0, "C", "APR26") == 1.0

    def test_different_strikes_independent(self):
        """Different strikes tracked independently."""
        st = SweepTracker(window_minutes=15)
        st.record(560.0, "C", "APR26")
        st.record(565.0, "C", "APR26")
        assert st.flow_score(560.0, "C", "APR26") == 0.0
        assert st.flow_score(565.0, "C", "APR26") == 0.0

    def test_different_rights_independent(self):
        """Call and put at same strike tracked independently."""
        st = SweepTracker(window_minutes=15)
        st.record(560.0, "C", "APR26")
        st.record(560.0, "P", "APR26")
        assert st.flow_score(560.0, "C", "APR26") == 0.0
        assert st.flow_score(560.0, "P", "APR26") == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Confidence Tiers
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfidenceTiers:
    """Validate tier boundaries from docs: MEDIUM 70-79, HIGH 80-89, EXTREME 90+."""

    def test_medium(self):
        assert _tier(0.70) == "MEDIUM"
        assert _tier(0.79) == "MEDIUM"

    def test_high(self):
        assert _tier(0.80) == "HIGH"
        assert _tier(0.89) == "HIGH"

    def test_extreme(self):
        assert _tier(0.90) == "EXTREME"
        assert _tier(0.99) == "EXTREME"
        assert _tier(1.00) == "EXTREME"

    def test_below_threshold_still_medium(self):
        """Below 70% still gets MEDIUM tier (tier is labeling, not filtering)."""
        assert _tier(0.50) == "MEDIUM"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Weighted Confidence Model
# ═══════════════════════════════════════════════════════════════════════════════

class TestWeightedConfidence:
    """Validate the 10-component weighted confidence model sums correctly."""

    def setup_method(self):
        self.cfg = SpyOptionsSignalConfig()
        self.tracker = VolumeTracker()
        self.engine = SignalEngine(self.cfg, self.tracker)

    def _ctx(self, iv_rank=25.0, sentiment_score=50.0) -> SignalContext:
        regime = RegimeContext(
            regime="TREND_UP", ema9=562.0, ema21=560.0, atr14=1.0,
            vwap=560.0, spy_vs_vwap=2.0, ema_slope=0.05,
            timestamp=datetime.utcnow(),
        )
        sentiment = SentimentContext(
            score=sentiment_score, vix_trend="FALLING",
            spy_vs_vwap=2.0, ema_slope=0.05, label="BULLISH",
        )
        return SignalContext(
            regime=regime, sentiment=sentiment,
            iv_rank=iv_rank, vix=15.0, spy_price=562.0,
        )

    def test_perfect_call_sweep(self):
        """Ideal call sweep: high spike, strong imbalance, good Greeks, low IV, bullish."""
        q = _make_quote(
            delta=0.45, gamma=0.03, theta=-0.05,
            bid_size=800, ask_size=100, oi=15000,
        )
        conf = self.engine._weighted_confidence(
            q, "C", spike_mult=14.0,  # 14× spike
            flow_score=1.0,
            signal_type=SignalType.CALL_SWEEP,
            context=self._ctx(iv_rank=15.0, sentiment_score=80.0),
            c=self.cfg,
        )
        # With perfect inputs across all 10 components, should be very high
        assert conf >= 0.80, f"Perfect call sweep should be HIGH tier, got {conf:.3f}"

    def test_poor_greeks_penalized(self):
        """Bad delta (deep ITM/OTM) should reduce confidence."""
        q_good = _make_quote(delta=0.45, gamma=0.03, theta=-0.05)
        q_bad = _make_quote(delta=0.90, gamma=0.001, theta=-0.25)

        ctx = self._ctx()
        conf_good = self.engine._weighted_confidence(
            q_good, "C", 8.0, 0.0, SignalType.CALL_SWEEP, ctx, self.cfg)
        conf_bad = self.engine._weighted_confidence(
            q_bad, "C", 8.0, 0.0, SignalType.CALL_SWEEP, ctx, self.cfg)

        assert conf_good > conf_bad, "Good Greeks should score higher than bad Greeks"

    def test_high_theta_penalty(self):
        """Theta < -0.15 should penalize directional long signals."""
        q_low_decay = _make_quote(theta=-0.05)
        q_high_decay = _make_quote(theta=-0.25)

        ctx = self._ctx()
        conf_low = self.engine._weighted_confidence(
            q_low_decay, "C", 8.0, 0.0, SignalType.CALL_SWEEP, ctx, self.cfg)
        conf_high = self.engine._weighted_confidence(
            q_high_decay, "C", 8.0, 0.0, SignalType.CALL_SWEEP, ctx, self.cfg)

        assert conf_low > conf_high, "High theta decay should penalize confidence"

    def test_iv_regime_alignment(self):
        """Low IV rank boosts debit signals, high IV rank boosts credit signals."""
        ctx_low_iv = self._ctx(iv_rank=10.0)
        ctx_high_iv = self._ctx(iv_rank=90.0)

        q = _make_quote()
        # CALL_SWEEP is debit → low IV should be better
        conf_debit_low = self.engine._weighted_confidence(
            q, "C", 8.0, 0.0, SignalType.CALL_SWEEP, ctx_low_iv, self.cfg)
        conf_debit_high = self.engine._weighted_confidence(
            q, "C", 8.0, 0.0, SignalType.CALL_SWEEP, ctx_high_iv, self.cfg)
        assert conf_debit_low > conf_debit_high, "Low IV should boost debit signals"

        # HIGH_IV_ALERT is credit → high IV should be better
        conf_credit_high = self.engine._weighted_confidence(
            q, "C", 8.0, 0.0, SignalType.HIGH_IV_ALERT, ctx_high_iv, self.cfg)
        conf_credit_low = self.engine._weighted_confidence(
            q, "C", 8.0, 0.0, SignalType.HIGH_IV_ALERT, ctx_low_iv, self.cfg)
        assert conf_credit_high > conf_credit_low, "High IV should boost credit signals"

    def test_sentiment_alignment(self):
        """Bullish sentiment should boost call confidence, bearish boosts puts."""
        q = _make_quote()

        ctx_bull = self._ctx(sentiment_score=80.0)
        ctx_bear = self._ctx(sentiment_score=-80.0)

        conf_call_bull = self.engine._weighted_confidence(
            q, "C", 8.0, 0.0, SignalType.CALL_SWEEP, ctx_bull, self.cfg)
        conf_call_bear = self.engine._weighted_confidence(
            q, "C", 8.0, 0.0, SignalType.CALL_SWEEP, ctx_bear, self.cfg)
        assert conf_call_bull > conf_call_bear, "Bullish sentiment should boost calls"

    def test_flow_score_contribution(self):
        """Flow score = 1.0 should add 5% to confidence (0.05 * 1.0)."""
        q = _make_quote()
        ctx = self._ctx()

        conf_no_flow = self.engine._weighted_confidence(
            q, "C", 8.0, 0.0, SignalType.CALL_SWEEP, ctx, self.cfg)
        conf_with_flow = self.engine._weighted_confidence(
            q, "C", 8.0, 1.0, SignalType.CALL_SWEEP, ctx, self.cfg)

        diff = conf_with_flow - conf_no_flow
        assert 0.04 <= diff <= 0.06, f"Flow score=1.0 should add ~0.05, got {diff:.4f}"

    def test_confidence_capped_at_1(self):
        """Confidence must not exceed 1.0 even with all perfect scores."""
        q = _make_quote(
            delta=0.45, gamma=0.03, theta=-0.02,
            bid_size=5000, ask_size=50, oi=50000,
        )
        conf = self.engine._weighted_confidence(
            q, "C", 100.0,  # absurd spike
            flow_score=1.0,
            signal_type=SignalType.CALL_SWEEP,
            context=self._ctx(iv_rank=5.0, sentiment_score=100.0),
            c=self.cfg,
        )
        assert conf <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Signal Dedup Keys
# ═══════════════════════════════════════════════════════════════════════════════

class TestDedupKeys:
    """Validate dedup key format: type:expiry:strike:right."""

    def test_dedup_key_format(self):
        sig = SpySignal(
            signal_type=SignalType.CALL_SWEEP,
            strike=565.0, expiry="APR26", right="C",
            confidence=0.85, spy_price=562.0, vix=15.0,
            volume=10000, volume_spike_mult=6.0,
            bid_size=400, ask_size=100,
        )
        assert sig.dedup_key == "CALL_SWEEP:APR26:565:C"

    def test_dedup_suppresses_same_signal(self):
        """Same type + strike + expiry + right → same dedup key."""
        sig1 = SpySignal(
            signal_type=SignalType.PUT_SWEEP,
            strike=550.0, expiry="APR26", right="P",
            confidence=0.75, spy_price=560.0, vix=20.0,
            volume=8000, volume_spike_mult=5.0,
            bid_size=200, ask_size=600,
        )
        sig2 = SpySignal(
            signal_type=SignalType.PUT_SWEEP,
            strike=550.0, expiry="APR26", right="P",
            confidence=0.80, spy_price=559.0, vix=20.5,
            volume=12000, volume_spike_mult=7.0,
            bid_size=300, ask_size=700,
        )
        assert sig1.dedup_key == sig2.dedup_key

    def test_different_strikes_different_keys(self):
        sig1 = SpySignal(
            signal_type=SignalType.CALL_SWEEP,
            strike=560.0, expiry="APR26", right="C",
            confidence=0.75, spy_price=560.0, vix=15.0,
            volume=5000, volume_spike_mult=5.0,
            bid_size=200, ask_size=100,
        )
        sig2 = SpySignal(
            signal_type=SignalType.CALL_SWEEP,
            strike=565.0, expiry="APR26", right="C",
            confidence=0.75, spy_price=560.0, vix=15.0,
            volume=5000, volume_spike_mult=5.0,
            bid_size=200, ask_size=100,
        )
        assert sig1.dedup_key != sig2.dedup_key


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Signal Engine — End-to-End Scenarios
# ═══════════════════════════════════════════════════════════════════════════════

class TestSignalEngineE2E:
    """End-to-end: build chain, feed to engine, validate signal output."""

    def setup_method(self):
        self.cfg = SpyOptionsSignalConfig()
        self.tracker = VolumeTracker()
        self.engine = SignalEngine(self.cfg, self.tracker)
        self.sweep = SweepTracker(window_minutes=15)

    def _build_chain(self, calls=None, puts=None) -> ChainSnapshot:
        chain = ChainSnapshot("APR26")
        chain.calls = calls or []
        chain.puts = puts or []
        return chain

    def _context(self, regime="TREND_UP", iv_rank=25.0, sentiment=50.0) -> SignalContext:
        r = RegimeContext(
            regime=regime, ema9=562.0, ema21=560.0, atr14=1.0,
            vwap=560.0, spy_vs_vwap=2.0, ema_slope=0.05,
            timestamp=datetime.utcnow(),
        )
        s = SentimentContext(
            score=sentiment, vix_trend="FALLING",
            spy_vs_vwap=2.0, ema_slope=0.05, label="BULLISH",
        )
        return SignalContext(regime=r, sentiment=s, iv_rank=iv_rank, vix=15.0, spy_price=562.0)

    def test_call_sweep_fires(self):
        """Strong call volume spike in TREND_UP → CALL_SWEEP signal."""
        # Seed tracker with baseline volume
        conid = 1001
        for v in [1000, 1200, 1400, 1600, 1800]:
            self.tracker.update(conid, v)

        # Now a big spike
        spike_volume = 1800 + 3000  # delta = 3000
        q = _make_quote(
            conid=conid, volume=spike_volume, strike=565.0,
            oi=8000, bid=1.85, ask=1.92,
            delta=0.42, gamma=0.031, theta=-0.07,
            bid_size=850, ask_size=120,
        )

        chain = self._build_chain(calls=[q])
        ctx = self._context(regime="TREND_UP", iv_rank=24.0, sentiment=62.0)
        signals = self.engine.evaluate(chain, ctx, self.sweep)

        call_sweeps = [s for s in signals if s.signal_type == SignalType.CALL_SWEEP]
        assert len(call_sweeps) >= 1, f"Expected CALL_SWEEP, got {[s.signal_type for s in signals]}"

        sig = call_sweeps[0]
        assert sig.strike == 565.0
        assert sig.right == "C"
        assert sig.confidence >= 0.70
        assert sig.spy_price == 562.0
        assert len(sig.reasoning) > 0

    def test_call_sweep_plus_bull_spread_on_low_iv(self):
        """Low IV rank + high volume call spike → both CALL_SWEEP and BULL_CALL_SPREAD."""
        conid = 2001
        for v in [500, 600, 700, 800, 900]:
            self.tracker.update(conid, v)

        spike_volume = 900 + 4000  # delta = 4000, high volume
        q = _make_quote(
            conid=conid, volume=spike_volume, strike=565.0,
            oi=12000, bid=1.80, ask=1.88,
            delta=0.45, gamma=0.03, theta=-0.05,
            bid_size=1000, ask_size=150,
        )

        chain = self._build_chain(calls=[q])
        ctx = self._context(regime="TREND_UP", iv_rank=20.0, sentiment=60.0)
        signals = self.engine.evaluate(chain, ctx, self.sweep)

        types = {s.signal_type for s in signals}
        assert SignalType.CALL_SWEEP in types
        assert SignalType.BULL_CALL_SPREAD in types, \
            f"Low IV rank ({ctx.iv_rank}) + high volume should trigger BULL_CALL_SPREAD. Got: {types}"

    def test_put_sweep_fires(self):
        """Strong put volume spike → PUT_SWEEP signal."""
        conid = 3001
        for v in [800, 1000, 1200, 1400, 1600]:
            self.tracker.update(conid, v)

        spike_volume = 1600 + 3500
        q = _make_quote(
            conid=conid, volume=spike_volume, strike=555.0, right="P",
            oi=6000, bid=1.60, ask=1.68,
            delta=-0.40, gamma=0.025, theta=-0.06,
            bid_size=100, ask_size=800,
        )

        chain = self._build_chain(puts=[q])
        ctx = self._context(regime="TREND_DOWN", iv_rank=25.0, sentiment=-40.0)
        signals = self.engine.evaluate(chain, ctx, self.sweep)

        put_sweeps = [s for s in signals if s.signal_type == SignalType.PUT_SWEEP]
        assert len(put_sweeps) >= 1

    def test_straddle_both_sides_spiking(self):
        """Both call AND put spikes at same strike → LONG_STRADDLE."""
        call_conid = 4001
        put_conid = 4002
        for v in [500, 600, 700, 800, 900]:
            self.tracker.update(call_conid, v)
            self.tracker.update(put_conid, v)

        spike = 900 + 3000
        call_q = _make_quote(
            conid=call_conid, volume=spike, strike=560.0, right="C",
            oi=8000, delta=0.50, gamma=0.04, theta=-0.06,
            bid_size=500, ask_size=200,
        )
        put_q = _make_quote(
            conid=put_conid, volume=spike, strike=560.0, right="P",
            oi=7000, delta=-0.50, gamma=0.04, theta=-0.06,
            bid_size=200, ask_size=500,
        )

        chain = self._build_chain(calls=[call_q], puts=[put_q])
        ctx = self._context(regime="RANGE_BOUND", iv_rank=30.0, sentiment=5.0)
        signals = self.engine.evaluate(chain, ctx, self.sweep)

        straddles = [s for s in signals if s.signal_type == SignalType.LONG_STRADDLE]
        assert len(straddles) >= 1, f"Expected LONG_STRADDLE. Got: {[s.signal_type for s in signals]}"

    def test_high_iv_alert(self):
        """High IV rank (>70) or VIX > 26 → HIGH_IV_ALERT."""
        chain = self._build_chain(
            calls=[_make_quote(strike=560.0, right="C", volume=1000)],
            puts=[_make_quote(strike=560.0, right="P", volume=1000, conid=5002)],
        )
        ctx = self._context(regime="HIGH_VOL", iv_rank=80.0, sentiment=-10.0)
        ctx.vix = 28.0
        signals = self.engine.evaluate(chain, ctx, self.sweep)

        hi_iv = [s for s in signals if s.signal_type == SignalType.HIGH_IV_ALERT]
        assert len(hi_iv) >= 1, f"Expected HIGH_IV_ALERT with VIX=28 and IV rank=80. Got: {[s.signal_type for s in signals]}"

    def test_pc_ratio_bearish(self):
        """P/C ratio > 1.8 → PC_RATIO_EXTREME (bearish)."""
        # Heavy put volume, light call volume — need extreme ratio to push
        # base confidence above min_confidence (0.70).
        # base = min(0.62 + (pc - 1.8) * 0.05, 0.82)
        # sent_boost = max(0.0, -sentiment/100) * 0.08
        # P/C = 5000/600 ≈ 8.3 → base = 0.82, sent_boost = 0.064 → conf = 0.884
        calls = [_make_quote(strike=560.0, right="C", volume=600, conid=6001)]
        puts = [_make_quote(strike=560.0, right="P", volume=5000, conid=6002)]

        chain = self._build_chain(calls=calls, puts=puts)
        assert chain.put_call_ratio > 1.8

        ctx = self._context(regime="TREND_DOWN", iv_rank=40.0, sentiment=-80.0)
        signals = self.engine.evaluate(chain, ctx, self.sweep)

        pc = [s for s in signals if s.signal_type == SignalType.PC_RATIO_EXTREME]
        assert len(pc) >= 1, f"Expected PC_RATIO_EXTREME with P/C={chain.put_call_ratio:.2f}"
        assert pc[0].right == "P", "Bearish P/C should produce put-side signal"

    def test_pc_ratio_bullish(self):
        """P/C ratio < 0.5 → PC_RATIO_EXTREME (bullish)."""
        # Extreme call dominance: P/C = 100/8000 = 0.0125
        # base = min(0.60 + (0.5 - 0.0125) * 0.08, 0.80) = min(0.639, 0.80) = 0.639
        # sent_boost = max(0.0, 90/100) * 0.08 = 0.072 → conf = 0.711 (above 0.70)
        calls = [_make_quote(strike=560.0, right="C", volume=8000, conid=7001)]
        puts = [_make_quote(strike=560.0, right="P", volume=100, conid=7002)]

        chain = self._build_chain(calls=calls, puts=puts)
        assert chain.put_call_ratio < 0.5

        ctx = self._context(regime="TREND_UP", iv_rank=20.0, sentiment=90.0)
        signals = self.engine.evaluate(chain, ctx, self.sweep)

        pc = [s for s in signals if s.signal_type == SignalType.PC_RATIO_EXTREME]
        assert len(pc) >= 1, f"Expected PC_RATIO_EXTREME with P/C={chain.put_call_ratio:.4f}"
        assert pc[0].right == "C", "Bullish P/C should produce call-side signal"

    def test_min_confidence_filter(self):
        """Signals below 70% threshold should be dropped."""
        # Tiny spike that won't produce high confidence
        conid = 8001
        for v in [5000, 6000, 7000, 8000, 9000]:
            self.tracker.update(conid, v)

        # Spike is only 4.1× (barely above threshold), bad Greeks
        spike_volume = 9000 + 1100  # delta = 1100
        q = _make_quote(
            conid=conid, volume=spike_volume, strike=560.0, right="C",
            oi=500, delta=0.95, gamma=0.001, theta=-0.30,
            bid_size=100, ask_size=100,
        )

        chain = self._build_chain(calls=[q])
        # Bearish sentiment + high IV → both penalize a call sweep
        ctx = self._context(regime="RANGE_BOUND", iv_rank=85.0, sentiment=-80.0)
        signals = self.engine.evaluate(chain, ctx, self.sweep)

        # With bad Greeks, wrong sentiment, wrong IV regime — should be filtered
        call_sweeps = [s for s in signals if s.signal_type == SignalType.CALL_SWEEP]
        if call_sweeps:
            assert call_sweeps[0].confidence >= 0.70, "Filtered signal should be ≥ 70%"


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Config Defaults Match Docs
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfigDefaults:
    """Validate that SpyOptionsSignalConfig defaults match documented values."""

    def test_thresholds(self):
        c = SpyOptionsSignalConfig()
        assert c.volume_spike_mult == 4.0,          "Spike multiplier: 4×"
        assert c.min_volume_for_signal == 500,       "Min session volume: 500"
        assert c.sweep_poll_volume_threshold == 300,  "Min sweep size: 300"
        assert c.min_confidence == 0.70,              "Min confidence: 70%"
        assert c.dedup_window_minutes == 90,           "Dedup window: 90 min"
        assert c.sweep_window_minutes == 15,           "Repeat sweep window: 15 min"
        assert c.pc_ratio_bearish == 1.8,              "P/C bearish threshold: 1.8"
        assert c.pc_ratio_bullish == 0.5,              "P/C bullish threshold: 0.5"
        assert c.vix_high == 26.0,                     "VIX high threshold: 26"
        assert c.bid_ask_imbalance_threshold == 3.0,   "Bid/ask imbalance: 3×"

    def test_chain_defaults(self):
        from shree.config.spy_options import SpyOptionsChainConfig
        c = SpyOptionsChainConfig()
        assert c.strike_pct_range == 0.04,           "Strike window: ±4%"
        assert c.liquidity_min_oi == 1000,             "Min OI: 1000"
        assert c.liquidity_max_spread_pct == 8.0,      "Max spread: 8%"
        assert c.liquidity_min_volume == 500,           "Min volume: 500"
        assert c.num_expiries == 2,                     "Expiries tracked: 2"

    def test_session_defaults(self):
        from shree.config.spy_options import SpyOptionsSessionConfig
        c = SpyOptionsSessionConfig()
        assert c.poll_interval_s == 60,                "Poll interval: 60s"
        assert c.rth_start_et == "09:35",              "RTH start: 9:35 ET"
        assert c.rth_stop_et == "15:45",               "RTH stop: 3:45 ET"


# ═══════════════════════════════════════════════════════════════════════════════
# 11. OptionQuote Properties
# ═══════════════════════════════════════════════════════════════════════════════

class TestOptionQuoteProperties:
    """Validate computed properties on OptionQuote."""

    def test_mid_price(self):
        q = _make_quote(bid=1.85, ask=1.92)
        assert abs(q.mid - 1.885) < 0.001

    def test_spread(self):
        q = _make_quote(bid=1.85, ask=1.92)
        assert abs(q.spread - 0.07) < 0.001

    def test_spread_pct(self):
        q = _make_quote(bid=1.85, ask=1.92)
        expected_pct = 0.07 / 1.885 * 100.0  # ~3.71%
        assert abs(q.spread_pct - expected_pct) < 0.1

    def test_bid_ask_ratio(self):
        q = _make_quote(bid_size=850, ask_size=120)
        assert abs(q.bid_ask_ratio - 850 / 120) < 0.01

    def test_ask_bid_ratio(self):
        q = _make_quote(bid_size=120, ask_size=850)
        assert abs(q.ask_bid_ratio - 850 / 120) < 0.01
