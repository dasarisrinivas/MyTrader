"""Rule-based SPY options signal engine.

Signal types generated:
  CALL_SWEEP       — Large call volume spike with bullish bid pressure
  PUT_SWEEP        — Large put volume spike with bearish ask pressure
  BULL_CALL_SPREAD — Low VIX + call sweep → debit spread recommended
  BEAR_PUT_SPREAD  — Low VIX + put sweep  → debit spread recommended
  LONG_STRADDLE    — Both call AND put volume spike simultaneously
  HIGH_IV_ALERT    — Elevated VIX → premium selling opportunity
  PC_RATIO_EXTREME — Chain-level put/call ratio at bullish/bearish extreme

All signals are informational only — no orders are placed.
"""
from __future__ import annotations

import html as _html
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set

from ..config.spy_options import SpyOptionsSignalConfig
from ..utils.logger import logger
from .chain_builder import ChainSnapshot, OptionQuote, VolumeTracker


class SignalType(str, Enum):
    CALL_SWEEP       = "CALL_SWEEP"
    PUT_SWEEP        = "PUT_SWEEP"
    BULL_CALL_SPREAD = "BULL_CALL_SPREAD"
    BEAR_PUT_SPREAD  = "BEAR_PUT_SPREAD"
    LONG_STRADDLE    = "LONG_STRADDLE"
    HIGH_IV_ALERT    = "HIGH_IV_ALERT"
    PC_RATIO_EXTREME = "PC_RATIO_EXTREME"


@dataclass
class SpySignal:
    """A single SPY options signal ready for delivery."""

    signal_type: SignalType
    strike: float
    expiry: str
    right: str            # "C", "P", or "BOTH"
    confidence: float     # 0.0 – 1.0
    spy_price: float
    vix: Optional[float]
    volume: int           # total session volume at the strike
    volume_spike_mult: float  # how many × the rolling avg
    bid_size: int
    ask_size: int
    reasoning: List[str] = field(default_factory=list)
    suggested_trade: str = ""

    @property
    def dedup_key(self) -> str:
        """Unique key used to suppress duplicate alerts within the dedup window."""
        return f"{self.signal_type}:{self.expiry}:{self.strike:.0f}:{self.right}"


class SignalEngine:
    """Evaluates ChainSnapshot and emits SpySignal objects."""

    def __init__(self, cfg: SpyOptionsSignalConfig, tracker: VolumeTracker) -> None:
        self._cfg = cfg
        self._tracker = tracker

    def evaluate(
        self,
        chain: ChainSnapshot,
        spy_price: float,
        vix: Optional[float] = None,
    ) -> List[SpySignal]:
        """Run all signal rules against the provided chain snapshot.

        Args:
            chain: Current option chain data for one expiry.
            spy_price: Current SPY last/mid price.
            vix: Current VIX level (None if unavailable).

        Returns:
            List of signals passing the minimum confidence threshold.
        """
        c = self._cfg
        signals: List[SpySignal] = []

        iv_low  = vix is not None and vix < c.vix_low
        iv_high = vix is not None and vix > c.vix_high
        vix_str = f"VIX={vix:.1f}" if vix is not None else "VIX unavailable"

        # ── Rule 1: Chain-level P/C ratio ─────────────────────────────────────
        signals.extend(self._pc_ratio_signal(chain, spy_price, vix, c))

        # ── Rules 2-5: Per-strike volume spike signals ─────────────────────────
        spiked_call_strikes: Set[float] = set()
        spiked_put_strikes: Set[float] = set()

        all_quotes = [(q, "C") for q in chain.calls] + [(q, "P") for q in chain.puts]

        for quote, right in all_quotes:
            if quote.volume < c.min_volume_for_signal:
                # Update tracker to keep baseline moving even for quiet strikes
                self._tracker.update(quote.conid, quote.volume)
                continue

            increment = self._tracker.update(quote.conid, quote.volume)
            avg = self._tracker.rolling_avg(quote.conid)

            # Spike check: large poll-interval delta AND meets absolute threshold
            spike_mult = (increment / avg) if avg > 1 else 0.0
            is_spike = (
                spike_mult >= c.volume_spike_mult
                and increment >= c.sweep_poll_volume_threshold
            )

            if not is_spike:
                continue

            if right == "C":
                spiked_call_strikes.add(quote.strike)
                signals.extend(
                    self._call_spike_signals(quote, chain.expiry_month, spike_mult, spy_price, vix, iv_low, c)
                )
            else:
                spiked_put_strikes.add(quote.strike)
                signals.extend(
                    self._put_spike_signals(quote, chain.expiry_month, spike_mult, spy_price, vix, iv_low, c)
                )

        # ── Rule 6: Straddle (both C and P spike at same/adjacent strikes) ────
        signals.extend(
            self._straddle_signals(
                spiked_call_strikes, spiked_put_strikes,
                chain, spy_price, vix, vix_str, c,
            )
        )

        # ── Rule 7: High IV alert ──────────────────────────────────────────────
        if iv_high:
            signals.extend(self._high_iv_signal(chain, spy_price, vix, c))

        # Filter by minimum confidence threshold
        filtered = [s for s in signals if s.confidence >= c.min_confidence]
        if filtered:
            logger.info(
                "SignalEngine: {} signals generated ({} passed confidence filter)",
                len(signals),
                len(filtered),
            )
        return filtered

    # ── Individual rule methods ───────────────────────────────────────────────

    def _pc_ratio_signal(
        self,
        chain: ChainSnapshot,
        spy_price: float,
        vix: Optional[float],
        c: SpyOptionsSignalConfig,
    ) -> List[SpySignal]:
        signals = []
        pc = chain.put_call_ratio
        if pc is None:
            return signals

        atm = chain.atm_strike(spy_price)

        if pc > c.pc_ratio_bearish and chain.total_put_volume >= c.min_volume_for_signal:
            atm_put = chain.put_at(atm)
            signals.append(SpySignal(
                signal_type=SignalType.PC_RATIO_EXTREME,
                strike=atm,
                expiry=chain.expiry_month,
                right="P",
                confidence=min(0.62 + (pc - c.pc_ratio_bearish) * 0.05, 0.85),
                spy_price=spy_price,
                vix=vix,
                volume=chain.total_put_volume,
                volume_spike_mult=pc,
                bid_size=atm_put.bid_size if atm_put else 0,
                ask_size=atm_put.ask_size if atm_put else 0,
                reasoning=[
                    f"P/C ratio = {pc:.2f} (bearish threshold: >{c.pc_ratio_bearish})",
                    f"Total put volume: {chain.total_put_volume:,} vs calls: {chain.total_call_volume:,}",
                    "Elevated put buying signals bearish hedging or directional bets",
                ],
                suggested_trade=(
                    f"Watch for SPY weakness near {atm:.0f}. "
                    f"Consider protective puts or Bear Put Spread exp {chain.expiry_month}"
                ),
            ))

        elif pc < c.pc_ratio_bullish and chain.total_call_volume >= c.min_volume_for_signal:
            signals.append(SpySignal(
                signal_type=SignalType.PC_RATIO_EXTREME,
                strike=atm,
                expiry=chain.expiry_month,
                right="C",
                confidence=min(0.60 + (c.pc_ratio_bullish - pc) * 0.08, 0.80),
                spy_price=spy_price,
                vix=vix,
                volume=chain.total_call_volume,
                volume_spike_mult=1.0 / pc if pc > 0 else 0.0,
                bid_size=0,
                ask_size=0,
                reasoning=[
                    f"P/C ratio = {pc:.2f} (bullish threshold: <{c.pc_ratio_bullish})",
                    f"Total call volume: {chain.total_call_volume:,} vs puts: {chain.total_put_volume:,}",
                    "Heavy call-to-put skew — possible speculative bull run or complacency",
                ],
                suggested_trade=(
                    f"Broad call interest near {atm:.0f} exp {chain.expiry_month}. "
                    "Watch for extended rally or mean-reversion setup."
                ),
            ))

        return signals

    def _call_spike_signals(
        self,
        quote: OptionQuote,
        expiry: str,
        spike_mult: float,
        spy_price: float,
        vix: Optional[float],
        iv_low: bool,
        c: SpyOptionsSignalConfig,
    ) -> List[SpySignal]:
        signals: List[SpySignal] = []
        buying_bias = quote.bid_ask_ratio >= c.bid_ask_imbalance_threshold

        # ── CALL SWEEP ────────────────────────────────────────────────────────
        confidence = 0.60
        reasoning = [
            f"Call volume spike: +{quote.volume:,} contracts this session at {quote.strike:.0f}C",
            f"Spike rate: {spike_mult:.1f}× rolling avg",
        ]
        if buying_bias:
            confidence += 0.10
            reasoning.append(
                f"Bid/Ask size ratio {quote.bid_ask_ratio:.1f}× — aggressive buyer at the ask"
            )
        if iv_low:
            confidence += 0.05
            reasoning.append(f"VIX={vix:.1f} (low IV) — debit strategies are cheap")

        wing = quote.strike + 5
        suggested = f"🐂 Call Sweep: {quote.strike:.0f}C exp {expiry}"
        if iv_low:
            suggested += f"\nConsider Bull Call Spread: Buy {quote.strike:.0f}C / Sell {wing:.0f}C exp {expiry}"

        signals.append(SpySignal(
            signal_type=SignalType.CALL_SWEEP,
            strike=quote.strike,
            expiry=expiry,
            right="C",
            confidence=min(confidence, 0.88),
            spy_price=spy_price,
            vix=vix,
            volume=quote.volume,
            volume_spike_mult=spike_mult,
            bid_size=quote.bid_size,
            ask_size=quote.ask_size,
            reasoning=reasoning,
            suggested_trade=suggested,
        ))

        # ── BULL CALL SPREAD (low IV bonus) ───────────────────────────────────
        if iv_low and quote.volume >= c.min_volume_for_signal * 2:
            signals.append(SpySignal(
                signal_type=SignalType.BULL_CALL_SPREAD,
                strike=quote.strike,
                expiry=expiry,
                right="C",
                confidence=min(0.62 + (spike_mult - c.volume_spike_mult) * 0.02, 0.82),
                spy_price=spy_price,
                vix=vix,
                volume=quote.volume,
                volume_spike_mult=spike_mult,
                bid_size=quote.bid_size,
                ask_size=quote.ask_size,
                reasoning=[
                    f"Low VIX ({vix:.1f}) makes debit spreads cost-effective",
                    f"Confirmed call volume spike {spike_mult:.1f}× at {quote.strike:.0f}C",
                    "Capped-risk structure: buy lower call, sell higher call",
                ],
                suggested_trade=(
                    f"Buy {quote.strike:.0f}C / Sell {wing:.0f}C exp {expiry}\n"
                    f"Max risk = net debit paid. Max profit if SPY closes above {wing:.0f}"
                ),
            ))

        return signals

    def _put_spike_signals(
        self,
        quote: OptionQuote,
        expiry: str,
        spike_mult: float,
        spy_price: float,
        vix: Optional[float],
        iv_low: bool,
        c: SpyOptionsSignalConfig,
    ) -> List[SpySignal]:
        signals: List[SpySignal] = []
        selling_bias = quote.ask_bid_ratio >= c.bid_ask_imbalance_threshold

        # ── PUT SWEEP ─────────────────────────────────────────────────────────
        confidence = 0.60
        reasoning = [
            f"Put volume spike: +{quote.volume:,} contracts this session at {quote.strike:.0f}P",
            f"Spike rate: {spike_mult:.1f}× rolling avg",
        ]
        if selling_bias:
            confidence += 0.10
            reasoning.append(
                f"Ask/Bid size ratio {quote.ask_bid_ratio:.1f}× — aggressive seller (put buyer)"
            )
        if iv_low:
            confidence += 0.05
            reasoning.append(f"VIX={vix:.1f} (low IV) — puts are cheap")

        wing = quote.strike - 5
        suggested = f"🐻 Put Sweep: {quote.strike:.0f}P exp {expiry}"
        if iv_low:
            suggested += f"\nConsider Bear Put Spread: Buy {quote.strike:.0f}P / Sell {wing:.0f}P exp {expiry}"

        signals.append(SpySignal(
            signal_type=SignalType.PUT_SWEEP,
            strike=quote.strike,
            expiry=expiry,
            right="P",
            confidence=min(confidence, 0.88),
            spy_price=spy_price,
            vix=vix,
            volume=quote.volume,
            volume_spike_mult=spike_mult,
            bid_size=quote.bid_size,
            ask_size=quote.ask_size,
            reasoning=reasoning,
            suggested_trade=suggested,
        ))

        # ── BEAR PUT SPREAD (low IV bonus) ────────────────────────────────────
        if iv_low and quote.volume >= c.min_volume_for_signal * 2:
            signals.append(SpySignal(
                signal_type=SignalType.BEAR_PUT_SPREAD,
                strike=quote.strike,
                expiry=expiry,
                right="P",
                confidence=min(0.62 + (spike_mult - c.volume_spike_mult) * 0.02, 0.82),
                spy_price=spy_price,
                vix=vix,
                volume=quote.volume,
                volume_spike_mult=spike_mult,
                bid_size=quote.bid_size,
                ask_size=quote.ask_size,
                reasoning=[
                    f"Low VIX ({vix:.1f}) makes debit spreads cost-effective",
                    f"Confirmed put volume spike {spike_mult:.1f}× at {quote.strike:.0f}P",
                    "Capped-risk structure: buy higher put, sell lower put",
                ],
                suggested_trade=(
                    f"Buy {quote.strike:.0f}P / Sell {wing:.0f}P exp {expiry}\n"
                    f"Max risk = net debit paid. Max profit if SPY closes below {wing:.0f}"
                ),
            ))

        return signals

    def _straddle_signals(
        self,
        spiked_calls: Set[float],
        spiked_puts: Set[float],
        chain: ChainSnapshot,
        spy_price: float,
        vix: Optional[float],
        vix_str: str,
        c: SpyOptionsSignalConfig,
    ) -> List[SpySignal]:
        signals: List[SpySignal] = []

        # Direct hit: same strike spiking on both sides
        both = spiked_calls & spiked_puts

        # Adjacent: call and put spikes within ±5 strike of each other
        if not both:
            for cs in spiked_calls:
                for ps in spiked_puts:
                    if abs(cs - ps) <= 5:
                        both.add(round((cs + ps) / 2 / 5) * 5)  # snap to nearest $5

        for strike in both:
            atm = chain.atm_strike(spy_price)
            signals.append(SpySignal(
                signal_type=SignalType.LONG_STRADDLE,
                strike=strike,
                expiry=chain.expiry_month,
                right="BOTH",
                confidence=0.73,
                spy_price=spy_price,
                vix=vix,
                volume=chain.total_call_volume + chain.total_put_volume,
                volume_spike_mult=c.straddle_spike_mult,
                bid_size=0,
                ask_size=0,
                reasoning=[
                    "Both call AND put volume spiking simultaneously",
                    "Smart money buying both sides → large move expected",
                    f"Total flow: {chain.total_call_volume + chain.total_put_volume:,} contracts",
                    vix_str,
                ],
                suggested_trade=(
                    f"Long Straddle: Buy {atm:.0f}C + Buy {atm:.0f}P exp {chain.expiry_month}\n"
                    f"Profit if SPY moves more than the combined premium in either direction"
                ),
            ))
        return signals

    def _high_iv_signal(
        self,
        chain: ChainSnapshot,
        spy_price: float,
        vix: Optional[float],
        c: SpyOptionsSignalConfig,
    ) -> List[SpySignal]:
        if (chain.total_call_volume + chain.total_put_volume) < c.min_volume_for_signal:
            return []
        atm = chain.atm_strike(spy_price)
        return [SpySignal(
            signal_type=SignalType.HIGH_IV_ALERT,
            strike=atm,
            expiry=chain.expiry_month,
            right="BOTH",
            confidence=0.65,
            spy_price=spy_price,
            vix=vix,
            volume=chain.total_call_volume + chain.total_put_volume,
            volume_spike_mult=0.0,
            bid_size=0,
            ask_size=0,
            reasoning=[
                f"VIX={vix:.1f} > high IV threshold ({c.vix_high})",
                "Elevated implied volatility → options are expensive",
                "Premium selling strategies (Iron Condor, credit spreads) may be favourable",
            ],
            suggested_trade=(
                f"Consider Iron Condor or credit spread on SPY near {atm:.0f} exp {chain.expiry_month}\n"
                f"Sell OTM call + OTM put. Collect premium, profit if SPY stays range-bound."
            ),
        )]
