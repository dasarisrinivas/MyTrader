"""Enhanced SPY options signal engine with weighted confidence scoring.

Signal types:
    CALL_SWEEP       — Large call volume spike with bullish bid pressure
    PUT_SWEEP        — Large put volume spike with bearish pressure
    BULL_CALL_SPREAD — Low IV rank + call sweep → debit spread recommended
    BEAR_PUT_SPREAD  — Low IV rank + put sweep  → debit spread recommended
    LONG_STRADDLE    — Both call AND put volume spike simultaneously
    HIGH_IV_ALERT    — Elevated VIX / high IV rank → premium selling opportunity
    PC_RATIO_EXTREME — Chain-level put/call ratio at bullish/bearish extreme

Confidence model (replaces simple spike scoring):
    25% volume spike strength
    15% bid/ask imbalance
    10% delta quality      (ideal range: calls 0.30-0.60, puts -0.60 to -0.30)
    10% gamma quality      (ideal: 0.005-0.08 for ATM SPY options)
    10% theta penalty      (penalise rapid decay for short-dated longs)
    10% IV regime alignment (low IV rank → good for debit, high → good for credit)
    10% sentiment alignment
     5% open interest strength
     5% flow score         (repeat sweeps within 15 min)

Threshold: 0.70 (was 0.55). Tiers: MEDIUM 70-79, HIGH 80-89, EXTREME 90+.

All signals are informational only — no orders are placed.
"""
from __future__ import annotations

import html as _html
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Set

from ..config.spy_options import SpyOptionsSignalConfig
from ..utils.logger import logger
from .chain_builder import ChainSnapshot, OptionQuote, VolumeTracker
from .dynamic_confidence import DynamicConfidence, ConfidenceAdjustment
from .external import ExternalContext
from .regime_detector import RegimeContext
from .sentiment_engine import SentimentContext
from .sweep_tracker import SweepTracker


class SignalType(str, Enum):
    CALL_SWEEP       = "CALL_SWEEP"
    PUT_SWEEP        = "PUT_SWEEP"
    BULL_CALL_SPREAD = "BULL_CALL_SPREAD"
    BEAR_PUT_SPREAD  = "BEAR_PUT_SPREAD"
    LONG_STRADDLE    = "LONG_STRADDLE"
    HIGH_IV_ALERT    = "HIGH_IV_ALERT"
    PC_RATIO_EXTREME = "PC_RATIO_EXTREME"
    ORB_BREAKOUT     = "ORB_BREAKOUT"   # Opening Range Breakout — confirmed directional move


@dataclass
class SignalContext:
    """Per-poll context passed from manager to signal engine."""

    regime: RegimeContext
    sentiment: SentimentContext
    iv_rank: float                              # 0-100 based on VIX 52w range
    vix: Optional[float]
    spy_price: float
    external: Optional[ExternalContext] = None  # External signals (news/macro/social)


@dataclass
class SpySignal:
    """A single SPY options signal ready for delivery and analytics."""

    signal_type: SignalType
    strike: float
    expiry: str
    right: str              # "C", "P", or "BOTH"
    confidence: float       # 0.0 – 1.0
    spy_price: float
    vix: Optional[float]
    volume: int
    volume_spike_mult: float
    bid_size: int
    ask_size: int
    reasoning: List[str] = field(default_factory=list)
    suggested_trade: str = ""
    expiry_date: str = ""   # YYYYMMDD (e.g. "20260417") — actual contract expiration

    # Rich fields (populated by engine)
    confidence_tier: str = "MEDIUM"  # "MEDIUM" | "HIGH" | "EXTREME"
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    impl_vol: float = 0.0
    open_interest: int = 0
    spread_pct: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    iv_rank: float = 0.0
    regime: str = "RANGE_BOUND"
    sentiment_score: float = 0.0
    sentiment_label: str = "NEUTRAL"
    flow_score: float = 0.0

    # DTE (days to expiry) — populated from expiry string
    dte: int = 0

    # External signal context (news/macro/social/flow)
    external_composite: float = 0.0   # -1.0 to +1.0
    event_risk: bool = False
    event_minutes: float = 999.0
    next_event_title: str = ""
    news_score: float = 0.0
    retail_score: float = 0.0
    macro_headwind: float = 0.0
    macro_label: str = "NEUTRAL"
    tnx_trend: str = "FLAT"
    dxy_trend: str = "FLAT"
    equity_pc: Optional[float] = None

    # Flow confirmation fields
    flow_confirmation_score: float = 0.0  # -100..+100
    dark_pool_bias: str = "NEUTRAL"
    gex_bias: str = "NEUTRAL"
    intraday_pc_ratio: Optional[float] = None

    # Dynamic confidence adjustment
    dynamic_confidence_delta: float = 0.0
    confidence_time_bucket: str = "MIDDAY"
    confidence_dte_rule: str = "STANDARD"
    conflict_detected: bool = False

    @property
    def dedup_key(self) -> str:
        return f"{self.signal_type}:{self.expiry}:{self.strike:.0f}:{self.right}"


def _tier(confidence: float) -> str:
    if confidence >= 0.90:
        return "EXTREME"
    if confidence >= 0.80:
        return "HIGH"
    return "MEDIUM"


class SignalEngine:
    """Evaluates ChainSnapshot and emits SpySignal objects."""

    def __init__(self, cfg: SpyOptionsSignalConfig, tracker: VolumeTracker) -> None:
        self._cfg = cfg
        self._tracker = tracker
        self._dyn = DynamicConfidence()

    def evaluate(
        self,
        chain: ChainSnapshot,
        context: SignalContext,
        sweep_tracker: SweepTracker,
    ) -> List[SpySignal]:
        """Run all signal rules against the chain snapshot.

        Args:
            chain:         Current option chain data for one expiry.
            context:       Per-poll regime, sentiment, IV rank, VIX, SPY price.
            sweep_tracker: Shared repeat-sweep detector.

        Returns:
            List of signals passing minimum confidence threshold.
        """
        c = self._cfg
        signals: List[SpySignal] = []

        iv_low = context.iv_rank < 30          # Low IV → debit spreads cheap
        iv_high = context.iv_rank > 70         # High IV → premium selling

        # ── Rule 1: Chain-level P/C ratio ─────────────────────────────────────
        signals.extend(self._pc_ratio_signal(chain, context, c))

        # ── Rules 2-5: Per-strike volume spike signals ─────────────────────────
        spiked_call_strikes: Set[float] = set()
        spiked_put_strikes: Set[float] = set()

        for quote, right in (
            [(q, "C") for q in chain.calls] + [(q, "P") for q in chain.puts]
        ):
            if quote.volume < c.min_volume_for_signal:
                self._tracker.update(quote.conid, quote.volume)
                continue

            increment = self._tracker.update(quote.conid, quote.volume)
            avg = self._tracker.rolling_avg(quote.conid)
            spike_mult = (increment / avg) if avg > 1 else 0.0
            is_spike = (
                spike_mult >= c.volume_spike_mult
                and increment >= c.sweep_poll_volume_threshold
            )
            if not is_spike:
                continue

            # Record in sweep tracker before calculating flow score
            sweep_tracker.record(quote.strike, right, chain.expiry_month)
            fscore = sweep_tracker.flow_score(quote.strike, right, chain.expiry_month)

            if right == "C":
                spiked_call_strikes.add(quote.strike)
                signals.extend(
                    self._call_spike_signals(quote, chain.expiry_month, spike_mult, fscore, iv_low, context, c)
                )
            else:
                spiked_put_strikes.add(quote.strike)
                signals.extend(
                    self._put_spike_signals(quote, chain.expiry_month, spike_mult, fscore, iv_low, context, c)
                )

        # ── Rule 6: Straddle ──────────────────────────────────────────────────
        signals.extend(
            self._straddle_signals(spiked_call_strikes, spiked_put_strikes, chain, context, c)
        )

        # ── Rule 7: High IV alert ─────────────────────────────────────────────
        if iv_high or (context.vix is not None and context.vix > c.vix_high):
            signals.extend(self._high_iv_signal(chain, context, c))

        # ── Rule 8: Opening Range Breakout ────────────────────────────────────
        # Fires once the 30-min ORB is established and price has confirmed a
        # break above (call) or below (put) the range for ≥1 bar.
        # Only fires on the first expiry evaluated to avoid duplicates.
        ext = context.external
        if ext is not None and getattr(ext, "orb_established", False):
            orb_status    = getattr(ext, "orb_status", "INSIDE")
            orb_confirmed = getattr(ext, "orb_breakout_confirmed", False)
            if orb_confirmed and orb_status in ("ABOVE_ORB", "BELOW_ORB"):
                signals.extend(self._orb_breakout_signal(chain, context, c))

        # ── Event risk modifier ────────────────────────────────────────────────
        ext = context.external
        if ext is not None and ext.event_risk:
            for sig in signals:
                directional = sig.signal_type in {
                    SignalType.CALL_SWEEP, SignalType.PUT_SWEEP,
                    SignalType.BULL_CALL_SPREAD, SignalType.BEAR_PUT_SPREAD,
                    SignalType.PC_RATIO_EXTREME, SignalType.ORB_BREAKOUT,
                }
                if directional:
                    # 30% confidence penalty within event window
                    sig.confidence = max(0.0, sig.confidence * 0.70)
                    sig.confidence_tier = _tier(sig.confidence)
                    sig.reasoning.append(
                        f"⚠ Event risk: {ext.next_event_title} in {ext.event_minutes:.0f} min — confidence penalised"
                    )
                elif sig.signal_type == SignalType.LONG_STRADDLE:
                    # Boost straddle during event window (big move expected)
                    sig.confidence = min(1.0, sig.confidence * 1.10)
                    sig.confidence_tier = _tier(sig.confidence)
                    sig.reasoning.append(
                        f"📅 Event catalyst: {ext.next_event_title} in {ext.event_minutes:.0f} min — straddle boosted"
                    )

        # Populate expiry_date and DTE on all signals from chain metadata
        for sig in signals:
            sig.expiry_date = chain.expiry_date
            if chain.expiry_date:
                try:
                    exp_dt = datetime.strptime(chain.expiry_date, "%Y%m%d").date()
                    sig.dte = max(0, (exp_dt - datetime.now().date()).days)
                except ValueError:
                    pass

        # ── Dynamic confidence adjustment ─────────────────────────────────────
        # Applied after event-risk modifier; updates confidence, tier, and stores
        # the adjustment breakdown on the signal for logging / analytics.
        for sig in signals:
            adj = self._dyn.adjust(
                base=sig.confidence,
                right=sig.right,
                dte=sig.dte,
                ext_ctx=context.external,
                ib_sentiment_score=context.sentiment.score,
                vix=context.vix,
                regime=context.regime.regime,
            )
            sig.confidence = adj.final
            sig.confidence_tier = _tier(adj.final)
            sig.dynamic_confidence_delta = adj.delta
            sig.confidence_time_bucket = adj.time_bucket
            sig.confidence_dte_rule = adj.dte_rule
            sig.conflict_detected = adj.conflict_detected
            if adj.conflict_detected:
                sig.reasoning.append(
                    f"⚡ Conflict: flow + sentiment + macro oppose signal direction "
                    f"({adj.flow_factor}) — confidence adjusted {adj.delta:+.0%}"
                )
            elif abs(adj.delta) >= 0.05:
                sig.reasoning.append(
                    f"📊 Dynamic adj: {adj.delta:+.0%} "
                    f"[{adj.time_bucket} | {adj.dte_rule} | {adj.flow_factor}]"
                )

        filtered = [s for s in signals if s.confidence >= c.min_confidence]
        rejected = [s for s in signals if s.confidence < c.min_confidence]
        if rejected:
            for s in rejected:
                logger.info(
                    "Signal below threshold: {} {} {}{} conf={:.1f}% (need {:.0f}%)",
                    s.signal_type.value, s.expiry, s.strike, s.right,
                    s.confidence * 100, c.min_confidence * 100,
                )

        # ── Cross-signal directional conflict filter ──────────────────────
        # If both CALL and PUT directional signals passed the threshold in the
        # same cycle, the signals are contradictory.  Keep only the direction
        # with the highest single-signal confidence.
        if filtered:
            calls = [s for s in filtered if s.right == "C"]
            puts  = [s for s in filtered if s.right == "P"]
            both  = [s for s in filtered if s.right == "BOTH"]
            if calls and puts:
                best_call = max(s.confidence for s in calls)
                best_put  = max(s.confidence for s in puts)
                if best_call >= best_put:
                    logger.warning(
                        "Directional conflict: {} CALL + {} PUT signals — "
                        "keeping CALL (best={:.0f}% vs PUT best={:.0f}%)",
                        len(calls), len(puts), best_call * 100, best_put * 100,
                    )
                    filtered = calls + both
                else:
                    logger.warning(
                        "Directional conflict: {} CALL + {} PUT signals — "
                        "keeping PUT (best={:.0f}% vs CALL best={:.0f}%)",
                        len(calls), len(puts), best_put * 100, best_call * 100,
                    )
                    filtered = puts + both

        if filtered:
            logger.info(
                "SignalEngine {}: {} signals → {} passed (threshold={:.0f}%)",
                chain.expiry_month, len(signals), len(filtered),
                c.min_confidence * 100,
            )
        return filtered

    # ── Weighted confidence model ─────────────────────────────────────────────

    def _weighted_confidence(
        self,
        quote: OptionQuote,
        right: str,
        spike_mult: float,
        flow_score: float,
        signal_type: SignalType,
        context: SignalContext,
        c: SpyOptionsSignalConfig,
    ) -> float:
        """Compute weighted confidence score (0.0–1.0) for a spike-based signal."""

        # 25% — volume spike strength (normalized against threshold)
        vol_score = min(1.0, max(0.0, (spike_mult - c.volume_spike_mult) / 10.0))
        w_vol = 0.25 * vol_score

        # 15% — bid/ask imbalance
        if right == "C":
            ratio = quote.bid_ask_ratio
        else:
            ratio = quote.ask_bid_ratio
        threshold = c.bid_ask_imbalance_threshold
        imb_score = min(1.0, max(0.0, (ratio - 1.0) / max(1.0, threshold - 1.0)))
        w_imb = 0.15 * imb_score

        # 10% — delta quality
        d = quote.delta
        if d == 0.0:
            delta_score = 0.5  # unknown
        elif right == "C":
            delta_score = 1.0 if 0.30 <= d <= 0.60 else 0.3
        else:
            delta_score = 1.0 if -0.60 <= d <= -0.30 else 0.3
        w_delta = 0.10 * delta_score

        # 10% — gamma quality (ideal ATM range for SPY: 0.005–0.08)
        g = quote.gamma
        if g == 0.0:
            gamma_score = 0.5
        else:
            gamma_score = 1.0 if 0.005 <= g <= 0.08 else 0.3
        w_gamma = 0.10 * gamma_score

        # 10% — theta penalty (theta is negative; rapid decay hurts long premium)
        t = quote.theta
        if t == 0.0:
            theta_score = 0.5
        elif t < -0.15:    # very high decay
            theta_score = 0.2
        elif t < -0.08:
            theta_score = 0.6
        else:
            theta_score = 1.0
        w_theta = 0.10 * theta_score

        # 10% — IV regime alignment
        # Credit signals (HIGH_IV_ALERT, BEAR_PUT_SPREAD, BULL_CALL_SPREAD) → high rank better
        # Directional debit signals (SWEEP) → low rank better
        credit_types = {SignalType.HIGH_IV_ALERT, SignalType.BEAR_PUT_SPREAD, SignalType.BULL_CALL_SPREAD}
        if signal_type in credit_types:
            iv_score = context.iv_rank / 100.0
        else:
            iv_score = 1.0 - (context.iv_rank / 100.0)
        w_iv = 0.10 * iv_score

        # 10% — sentiment alignment
        sent = context.sentiment.score / 100.0  # -1 to +1
        if right == "C":
            sent_score = (sent + 1.0) / 2.0        # +1 bullish → 1.0
        elif right == "P":
            sent_score = (-sent + 1.0) / 2.0       # -1 bearish → 1.0
        else:
            sent_score = 0.5                        # direction-agnostic
        w_sent = 0.10 * sent_score

        # 5% — open interest strength (10k+ = max)
        oi_score = min(1.0, quote.open_interest / 10_000.0)
        w_oi = 0.05 * oi_score

        # 5% — flow score (repeat sweep confirmation)
        w_flow = 0.05 * flow_score

        base = w_vol + w_imb + w_delta + w_gamma + w_theta + w_iv + w_sent + w_oi + w_flow

        # External composite adjustment (±composite_confidence_boost)
        ext = context.external
        if ext is not None:
            boost_cap = 0.05   # max ±5% from external
            # Directional alignment: composite positive boosts calls, negative boosts puts
            if right == "C":
                ext_adj = ext.composite_score * boost_cap
            elif right == "P":
                ext_adj = -ext.composite_score * boost_cap
            else:
                ext_adj = abs(ext.composite_score) * boost_cap * 0.5  # small boost for straddle
            base += ext_adj

        return min(1.0, max(0.0, base))

    def _enrich(self, sig: SpySignal, quote: OptionQuote, context: SignalContext) -> SpySignal:
        """Copy Greek/liquidity/context fields from quote and context into signal."""
        sig.delta = quote.delta
        sig.gamma = quote.gamma
        sig.theta = quote.theta
        sig.vega = quote.vega
        sig.impl_vol = quote.impl_vol
        sig.open_interest = quote.open_interest
        sig.spread_pct = quote.spread_pct
        sig.bid = quote.bid
        sig.ask = quote.ask
        sig.iv_rank = context.iv_rank
        sig.regime = context.regime.regime
        sig.sentiment_score = context.sentiment.score
        sig.sentiment_label = context.sentiment.label
        sig.confidence_tier = _tier(sig.confidence)
        # External fields
        if context.external is not None:
            ext = context.external
            sig.external_composite = ext.composite_score
            sig.event_risk = ext.event_risk
            sig.event_minutes = ext.event_minutes
            sig.next_event_title = ext.next_event_title
            sig.news_score = ext.news_score
            sig.retail_score = ext.retail_score
            sig.macro_headwind = ext.macro_headwind
            sig.macro_label = ext.macro_label
            sig.tnx_trend = ext.tnx_trend
            sig.dxy_trend = ext.dxy_trend
            sig.equity_pc = ext.equity_pc
            # Flow confirmation
            sig.flow_confirmation_score = ext.flow_score
            sig.dark_pool_bias = ext.flow_dark_pool
            sig.gex_bias = ext.flow_gex_bias
            sig.intraday_pc_ratio = ext.flow_pc_ratio
        return sig

    # ── Rule implementations ──────────────────────────────────────────────────

    def _pc_ratio_signal(
        self,
        chain: ChainSnapshot,
        context: SignalContext,
        c: SpyOptionsSignalConfig,
    ) -> List[SpySignal]:
        signals = []
        pc = chain.put_call_ratio
        if pc is None:
            return signals

        atm = chain.atm_strike(context.spy_price)

        if pc > c.pc_ratio_bearish and chain.total_put_volume >= c.min_volume_for_signal:
            atm_put = chain.put_at(atm)
            # Confidence: base from ratio distance, adjusted by sentiment & regime
            # Scale: P/C 1.8→0.70, 2.5→0.77, 3.0→0.82, 5.0→0.90
            base = min(0.70 + (pc - c.pc_ratio_bearish) * 0.10, 0.90)
            # Sentiment adjustment: bearish sentiment boosts, bullish PENALISES
            sent_norm = context.sentiment.score / 100.0  # -1 to +1
            if sent_norm < 0:
                sent_adj = abs(sent_norm) * 0.08   # bearish aligns → boost
            else:
                sent_adj = -sent_norm * 0.10       # bullish opposes → penalty
            # Regime alignment: TREND_UP directly opposes bearish put signal
            regime = context.regime.regime
            regime_adj = 0.0
            if regime == "TREND_UP":
                regime_adj = -0.12  # strong penalty: trend opposes put signal
            elif regime == "TREND_DOWN":
                regime_adj = 0.05   # aligned: trend supports put signal
            conf = max(0.0, min(0.95, base + sent_adj + regime_adj))
            sig = SpySignal(
                signal_type=SignalType.PC_RATIO_EXTREME,
                strike=atm, expiry=chain.expiry_month, right="P",
                confidence=conf, spy_price=context.spy_price, vix=context.vix,
                volume=chain.total_put_volume, volume_spike_mult=pc,
                bid_size=atm_put.bid_size if atm_put else 0,
                ask_size=atm_put.ask_size if atm_put else 0,
                reasoning=[
                    f"P/C ratio = {pc:.2f} (bearish threshold: >{c.pc_ratio_bearish})",
                    f"Total put volume: {chain.total_put_volume:,} vs calls: {chain.total_call_volume:,}",
                    f"Sentiment: {context.sentiment.label} ({context.sentiment.score:+.0f})",
                    f"Regime: {context.regime.regime}",
                ],
                suggested_trade=(
                    f"Watch for SPY weakness near {atm:.0f}. "
                    f"Bear Put Spread or protective puts exp {chain.expiry_month}"
                ),
            )
            if atm_put:
                self._enrich(sig, atm_put, context)
            else:
                sig.confidence_tier = _tier(conf)
                sig.iv_rank = context.iv_rank
                sig.regime = context.regime.regime
                sig.sentiment_score = context.sentiment.score
                sig.sentiment_label = context.sentiment.label
            signals.append(sig)

        elif pc < c.pc_ratio_bullish and chain.total_call_volume >= c.min_volume_for_signal:
            base = min(0.70 + (c.pc_ratio_bullish - pc) * 0.15, 0.90)
            # Sentiment adjustment: bullish sentiment boosts, bearish PENALISES
            sent_norm = context.sentiment.score / 100.0  # -1 to +1
            if sent_norm > 0:
                sent_adj = sent_norm * 0.08        # bullish aligns → boost
            else:
                sent_adj = sent_norm * 0.10        # bearish opposes → penalty (sent_norm is negative)
            # Regime alignment: TREND_DOWN opposes bullish call signal
            regime = context.regime.regime
            regime_adj = 0.0
            if regime == "TREND_DOWN":
                regime_adj = -0.12  # strong penalty: trend opposes call signal
            elif regime == "TREND_UP":
                regime_adj = 0.05   # aligned: trend supports call signal
            conf = max(0.0, min(0.95, base + sent_adj + regime_adj))
            sig = SpySignal(
                signal_type=SignalType.PC_RATIO_EXTREME,
                strike=atm, expiry=chain.expiry_month, right="C",
                confidence=conf, spy_price=context.spy_price, vix=context.vix,
                volume=chain.total_call_volume, volume_spike_mult=1.0 / pc if pc > 0 else 0.0,
                bid_size=0, ask_size=0,
                reasoning=[
                    f"P/C ratio = {pc:.2f} (bullish threshold: <{c.pc_ratio_bullish})",
                    f"Total call volume: {chain.total_call_volume:,} vs puts: {chain.total_put_volume:,}",
                    f"Sentiment: {context.sentiment.label} ({context.sentiment.score:+.0f})",
                    f"Regime: {context.regime.regime}",
                ],
                suggested_trade=(
                    f"Broad call interest near {atm:.0f} exp {chain.expiry_month}. "
                    "Watch for extended rally or mean-reversion setup."
                ),
            )
            sig.confidence_tier = _tier(conf)
            sig.iv_rank = context.iv_rank
            sig.regime = context.regime.regime
            sig.sentiment_score = context.sentiment.score
            sig.sentiment_label = context.sentiment.label
            signals.append(sig)

        return signals

    def _call_spike_signals(
        self,
        quote: OptionQuote,
        expiry: str,
        spike_mult: float,
        flow_score: float,
        iv_low: bool,
        context: SignalContext,
        c: SpyOptionsSignalConfig,
    ) -> List[SpySignal]:
        signals: List[SpySignal] = []
        conf = self._weighted_confidence(quote, "C", spike_mult, flow_score, SignalType.CALL_SWEEP, context, c)

        reasoning = [
            f"Call volume spike: {quote.volume:,} contracts at {quote.strike:.0f}C",
            f"Spike: {spike_mult:.1f}× rolling avg",
        ]
        if quote.bid_ask_ratio >= c.bid_ask_imbalance_threshold:
            reasoning.append(f"Bid/Ask ratio {quote.bid_ask_ratio:.1f}× — aggressive buyer at ask")
        if iv_low:
            reasoning.append(f"Low IV rank ({context.iv_rank:.0f}) — debit strategies are cheap")
        if flow_score > 0:
            reasoning.append(f"Repeat sweep detected (flow score: {flow_score:.1f})")
        reasoning.append(f"Regime: {context.regime.regime}  Sentiment: {context.sentiment.label} ({context.sentiment.score:+.0f})")
        if quote.delta != 0.0:
            reasoning.append(f"Delta {quote.delta:+.3f}  Gamma {quote.gamma:.4f}  IV {quote.impl_vol:.1%}")

        wing = quote.strike + 5
        suggested = f"Call Sweep: {quote.strike:.0f}C exp {expiry}"
        if iv_low:
            suggested += f"\nBull Call Spread: Buy {quote.strike:.0f}C / Sell {wing:.0f}C exp {expiry}"
        suggested += f"\nRisk: Exit if SPY loses VWAP (${context.regime.vwap:.2f})"

        sig = SpySignal(
            signal_type=SignalType.CALL_SWEEP,
            strike=quote.strike, expiry=expiry, right="C",
            confidence=conf, spy_price=context.spy_price, vix=context.vix,
            volume=quote.volume, volume_spike_mult=spike_mult,
            bid_size=quote.bid_size, ask_size=quote.ask_size,
            reasoning=reasoning, suggested_trade=suggested,
            flow_score=flow_score,
        )
        self._enrich(sig, quote, context)
        signals.append(sig)

        # BULL_CALL_SPREAD — low IV rank bonus
        if iv_low and quote.volume >= c.min_volume_for_signal * 2:
            conf2 = self._weighted_confidence(quote, "C", spike_mult, flow_score, SignalType.BULL_CALL_SPREAD, context, c)
            sig2 = SpySignal(
                signal_type=SignalType.BULL_CALL_SPREAD,
                strike=quote.strike, expiry=expiry, right="C",
                confidence=conf2, spy_price=context.spy_price, vix=context.vix,
                volume=quote.volume, volume_spike_mult=spike_mult,
                bid_size=quote.bid_size, ask_size=quote.ask_size,
                reasoning=[
                    f"Low IV rank ({context.iv_rank:.0f}) makes debit spreads cost-effective",
                    f"Call spike {spike_mult:.1f}× at {quote.strike:.0f}C",
                    "Capped-risk: buy lower call, sell higher call",
                ],
                suggested_trade=(
                    f"Buy {quote.strike:.0f}C / Sell {wing:.0f}C exp {expiry}\n"
                    f"Max profit if SPY closes above {wing:.0f} at expiry"
                ),
                flow_score=flow_score,
            )
            self._enrich(sig2, quote, context)
            signals.append(sig2)

        return signals

    def _put_spike_signals(
        self,
        quote: OptionQuote,
        expiry: str,
        spike_mult: float,
        flow_score: float,
        iv_low: bool,
        context: SignalContext,
        c: SpyOptionsSignalConfig,
    ) -> List[SpySignal]:
        signals: List[SpySignal] = []
        conf = self._weighted_confidence(quote, "P", spike_mult, flow_score, SignalType.PUT_SWEEP, context, c)

        reasoning = [
            f"Put volume spike: {quote.volume:,} contracts at {quote.strike:.0f}P",
            f"Spike: {spike_mult:.1f}× rolling avg",
        ]
        if quote.ask_bid_ratio >= c.bid_ask_imbalance_threshold:
            reasoning.append(f"Ask/Bid ratio {quote.ask_bid_ratio:.1f}× — aggressive put buyer")
        if iv_low:
            reasoning.append(f"Low IV rank ({context.iv_rank:.0f}) — puts are cheap")
        if flow_score > 0:
            reasoning.append(f"Repeat sweep detected (flow score: {flow_score:.1f})")
        reasoning.append(f"Regime: {context.regime.regime}  Sentiment: {context.sentiment.label} ({context.sentiment.score:+.0f})")
        if quote.delta != 0.0:
            reasoning.append(f"Delta {quote.delta:+.3f}  Gamma {quote.gamma:.4f}  IV {quote.impl_vol:.1%}")

        wing = quote.strike - 5
        suggested = f"Put Sweep: {quote.strike:.0f}P exp {expiry}"
        if iv_low:
            suggested += f"\nBear Put Spread: Buy {quote.strike:.0f}P / Sell {wing:.0f}P exp {expiry}"
        suggested += f"\nRisk: Exit if SPY reclaims VWAP (${context.regime.vwap:.2f}) or VIX spikes"

        sig = SpySignal(
            signal_type=SignalType.PUT_SWEEP,
            strike=quote.strike, expiry=expiry, right="P",
            confidence=conf, spy_price=context.spy_price, vix=context.vix,
            volume=quote.volume, volume_spike_mult=spike_mult,
            bid_size=quote.bid_size, ask_size=quote.ask_size,
            reasoning=reasoning, suggested_trade=suggested,
            flow_score=flow_score,
        )
        self._enrich(sig, quote, context)
        signals.append(sig)

        # BEAR_PUT_SPREAD — low IV rank bonus
        if iv_low and quote.volume >= c.min_volume_for_signal * 2:
            conf2 = self._weighted_confidence(quote, "P", spike_mult, flow_score, SignalType.BEAR_PUT_SPREAD, context, c)
            sig2 = SpySignal(
                signal_type=SignalType.BEAR_PUT_SPREAD,
                strike=quote.strike, expiry=expiry, right="P",
                confidence=conf2, spy_price=context.spy_price, vix=context.vix,
                volume=quote.volume, volume_spike_mult=spike_mult,
                bid_size=quote.bid_size, ask_size=quote.ask_size,
                reasoning=[
                    f"Low IV rank ({context.iv_rank:.0f}) makes debit spreads cost-effective",
                    f"Put spike {spike_mult:.1f}× at {quote.strike:.0f}P",
                    "Capped-risk: buy higher put, sell lower put",
                ],
                suggested_trade=(
                    f"Buy {quote.strike:.0f}P / Sell {wing:.0f}P exp {expiry}\n"
                    f"Max profit if SPY closes below {wing:.0f} at expiry"
                ),
                flow_score=flow_score,
            )
            self._enrich(sig2, quote, context)
            signals.append(sig2)

        return signals

    def _straddle_signals(
        self,
        spiked_calls: Set[float],
        spiked_puts: Set[float],
        chain: ChainSnapshot,
        context: SignalContext,
        c: SpyOptionsSignalConfig,
    ) -> List[SpySignal]:
        signals: List[SpySignal] = []
        both: Set[float] = spiked_calls & spiked_puts

        # Adjacent: call and put spikes within ±5 strike
        if not both:
            for cs in spiked_calls:
                for ps in spiked_puts:
                    if abs(cs - ps) <= 5:
                        both.add(round((cs + ps) / 2 / 5) * 5)

        for strike in both:
            atm = chain.atm_strike(context.spy_price)
            # Straddle confidence: sentiment neutral is good (direction-agnostic)
            neutrality = 1.0 - abs(context.sentiment.score) / 100.0
            conf = min(0.78 + neutrality * 0.10, 0.90)
            sig = SpySignal(
                signal_type=SignalType.LONG_STRADDLE,
                strike=strike, expiry=chain.expiry_month, right="BOTH",
                confidence=conf, spy_price=context.spy_price, vix=context.vix,
                volume=chain.total_call_volume + chain.total_put_volume,
                volume_spike_mult=c.straddle_spike_mult,
                bid_size=0, ask_size=0,
                reasoning=[
                    "Both call AND put volume spiking simultaneously",
                    "Smart money buying both sides → large move expected",
                    f"Total flow: {chain.total_call_volume + chain.total_put_volume:,} contracts",
                    f"Regime: {context.regime.regime}  IV rank: {context.iv_rank:.0f}",
                ],
                suggested_trade=(
                    f"Long Straddle: Buy {atm:.0f}C + Buy {atm:.0f}P exp {chain.expiry_month}\n"
                    f"Profit if SPY moves more than combined premium in either direction\n"
                    f"VWAP: ${context.regime.vwap:.2f}"
                ),
            )
            sig.confidence_tier = _tier(conf)
            sig.iv_rank = context.iv_rank
            sig.regime = context.regime.regime
            sig.sentiment_score = context.sentiment.score
            sig.sentiment_label = context.sentiment.label
            signals.append(sig)
        return signals

    def _orb_breakout_signal(
        self,
        chain: ChainSnapshot,
        context: SignalContext,
        c: SpyOptionsSignalConfig,
    ) -> List[SpySignal]:
        """Generate an ORB_BREAKOUT signal on confirmed 30-min range break.

        Confidence model for ORB:
          - Base is 0.72 (reasonable prior — ORB breakouts are high-probability
            but not infallible; ~60-65% historical follow-through on SPY)
          - Boosted by: regime alignment, sentiment, flow confirmation
          - Penalised by: EDR exhaustion, max pain proximity
        The dynamic confidence engine will then apply ORB-specific modifiers
        on top of these via adjustment block 13.
        """
        ext = context.external
        if ext is None:
            return []

        orb_status = getattr(ext, "orb_status", "INSIDE")
        orb_high   = getattr(ext, "orb_high", None)
        orb_low    = getattr(ext, "orb_low", None)
        orb_width  = getattr(ext, "orb_width_pct", 0.0)

        if orb_status == "ABOVE_ORB":
            right = "C"
        elif orb_status == "BELOW_ORB":
            right = "P"
        else:
            return []

        atm = chain.atm_strike(context.spy_price)
        atm_quote = chain.call_at(atm) if right == "C" else chain.put_at(atm)

        # Base confidence: ORB breakouts on SPY have strong follow-through stats.
        # Sentiment, flow, regime alignment are handled by DynamicConfidence engine
        # (blocks 3, 4, 13) to avoid double-counting — only ORB-specific width
        # adjustment is applied here.
        base_conf = 0.72

        # Narrow ORB = more reliable breakout (tight consolidation then range expansion)
        if orb_width < 0.20:   # < 0.20% is a tight range
            base_conf += 0.03
        elif orb_width > 0.60:  # wide range = less reliable
            base_conf -= 0.03

        conf = max(0.0, min(0.95, base_conf))

        # Build reasoning
        orb_ref = f"${orb_high:.2f}" if orb_high else "n/a"
        orb_ref_low = f"${orb_low:.2f}" if orb_low else "n/a"
        direction_word = "ABOVE" if right == "C" else "BELOW"
        wall = orb_ref if right == "C" else orb_ref_low

        reasoning = [
            f"30-min ORB breakout: SPY closed {direction_word} {wall}",
            f"ORB range: {orb_ref_low} – {orb_ref}  ({orb_width:.2f}% width)",
            f"Regime: {context.regime.regime}  Sentiment: {context.sentiment.label} ({context.sentiment.score:+.0f})",
        ]
        if getattr(ext, "rsi_5m", 50.0) != 50.0:
            rsi_tag = (
                " [OVERBOUGHT]" if getattr(ext, "rsi_overbought", False) else
                " [OVERSOLD]"   if getattr(ext, "rsi_oversold",   False) else ""
            )
            reasoning.append(f"RSI (5m): {ext.rsi_5m:.1f}{rsi_tag}")
        if getattr(ext, "near_pivot", False):
            reasoning.append(
                f"Near pivot {ext.pivot_nearest} ({ext.pivot_bias.replace('_', ' ')})"
            )

        wing = (atm + 5) if right == "C" else (atm - 5)
        suggested = (
            f"ORB {'Call' if right == 'C' else 'Put'} Sweep: {atm:.0f}{right} exp {chain.expiry_month}\n"
            f"{'Bull' if right == 'C' else 'Bear'} {'Call' if right == 'C' else 'Put'} Spread: "
            f"Buy {atm:.0f}{right} / Sell {wing:.0f}{right} exp {chain.expiry_month}\n"
            f"Risk: Exit if SPY {'falls back below' if right == 'C' else 'reclaims'} "
            f"{wall}"
        )

        sig = SpySignal(
            signal_type=SignalType.ORB_BREAKOUT,
            strike=atm, expiry=chain.expiry_month, right=right,
            confidence=conf, spy_price=context.spy_price, vix=context.vix,
            volume=chain.total_call_volume if right == "C" else chain.total_put_volume,
            volume_spike_mult=0.0,
            bid_size=atm_quote.bid_size if atm_quote else 0,
            ask_size=atm_quote.ask_size if atm_quote else 0,
            reasoning=reasoning,
            suggested_trade=suggested,
        )

        if atm_quote:
            self._enrich(sig, atm_quote, context)
        else:
            sig.confidence_tier = _tier(conf)
            sig.iv_rank = context.iv_rank
            sig.regime = context.regime.regime
            sig.sentiment_score = context.sentiment.score
            sig.sentiment_label = context.sentiment.label

        return [sig]

    def _high_iv_signal(
        self,
        chain: ChainSnapshot,
        context: SignalContext,
        c: SpyOptionsSignalConfig,
    ) -> List[SpySignal]:
        if (chain.total_call_volume + chain.total_put_volume) < c.min_volume_for_signal:
            return []
        atm = chain.atm_strike(context.spy_price)
        # Higher IV rank = stronger signal for premium selling
        conf = min(0.65 + context.iv_rank / 100.0 * 0.20, 0.85)
        sig = SpySignal(
            signal_type=SignalType.HIGH_IV_ALERT,
            strike=atm, expiry=chain.expiry_month, right="BOTH",
            confidence=conf, spy_price=context.spy_price, vix=context.vix,
            volume=chain.total_call_volume + chain.total_put_volume,
            volume_spike_mult=0.0, bid_size=0, ask_size=0,
            reasoning=[
                f"VIX={context.vix:.1f}" if context.vix else "VIX elevated",
                f"IV rank: {context.iv_rank:.0f}/100 — options are expensive",
                "Premium selling strategies (Iron Condor, credit spreads) favoured",
                f"Regime: {context.regime.regime}",
            ],
            suggested_trade=(
                f"Iron Condor or credit spread near {atm:.0f} exp {chain.expiry_month}\n"
                "Sell OTM call + OTM put. Profit if SPY stays range-bound.\n"
                f"VWAP anchor: ${context.regime.vwap:.2f}"
            ),
        )
        sig.confidence_tier = _tier(conf)
        sig.iv_rank = context.iv_rank
        sig.regime = context.regime.regime
        sig.sentiment_score = context.sentiment.score
        sig.sentiment_label = context.sentiment.label
        return [sig]
