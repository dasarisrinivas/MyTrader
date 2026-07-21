"""MES signal-only bot.

JUL 21 2026: The IB account became a Cash account (no CME futures), so the
execution stack was removed entirely.  This bot ANALYZES and SIGNALS — it never
places, modifies, or manages orders.  IB is used strictly as a market-data feed.

Pipeline per 15m bar close:
    fetch 15m MES bars (IB historical, useRTH=False — matches the validated
    backtest indicator convention) → enrich (MACD hist, PDH/PDL, HTF 30m trend)
    → EsFifteenMinStrategy.generate() → emit BUY/SELL/HOLD with confidence,
    entry/stop/target, expected R, regime and full gate evidence to
    logs/mes_signals.jsonl (+ Telegram on actionable signals).
"""
from __future__ import annotations

import asyncio
import json
from calendar import monthrange
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from ib_insync import IB, Future

from ..config import Settings
from ..features.feature_engineer import _adx, _atr, _ema, _rsi
from ..strategies.es_fifteen_min import EsFifteenMinStrategy
from ..utils.logger import logger
from ..utils.telegram_notifier import TelegramNotifier
from ..utils.timezone_utils import now_cst

SIGNAL_LOG = Path("logs/mes_signals.jsonl")
ET = "US/Eastern"


class MesSignalBot:
    """Signal-only MES bot: market data in, BUY/SELL/HOLD out. No orders."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.strategy = EsFifteenMinStrategy(settings.one_minute)
        self.ib = IB()
        self._contract = None
        self._telegram: Optional[TelegramNotifier] = None
        tg = getattr(settings, "telegram", None)
        if tg is not None and getattr(tg, "enabled", False) and getattr(tg, "bot_token", ""):
            self._telegram = TelegramNotifier(
                bot_token=tg.bot_token, chat_id=tg.chat_id, enabled=True
            )
        self._running = False
        self._last_bar_ts: Optional[pd.Timestamp] = None

    # ------------------------------------------------------------------
    #  IB market data (read-only)
    # ------------------------------------------------------------------
    async def _connect(self) -> None:
        d = self.settings.data
        host = getattr(d, "ibkr_host", "127.0.0.1")
        port = int(getattr(d, "ibkr_port", 4001))
        client_id = int(getattr(d, "ibkr_client_id", 1))
        logger.info(f"🔌 Connecting to IB {host}:{port} (clientId={client_id}, DATA ONLY)")
        await self.ib.connectAsync(host, port, clientId=client_id, timeout=30)
        logger.info("✅ IB connected — market data feed only, order placement removed")

    async def _resolve_front_month(self) -> None:
        details = await self.ib.reqContractDetailsAsync(
            Future(symbol="MES", exchange="CME", currency="USD")
        )
        if not details:
            raise RuntimeError("No MES contract details returned by IB")
        now_utc = datetime.now(timezone.utc)

        def _expiry(cd) -> datetime:
            raw = getattr(cd.contract, "lastTradeDateOrContractMonth", "") or ""
            try:
                if len(raw) >= 8:
                    return datetime.strptime(raw[:8], "%Y%m%d").replace(tzinfo=timezone.utc)
                if len(raw) == 6:
                    y, m = int(raw[:4]), int(raw[4:6])
                    return datetime(y, m, monthrange(y, m)[1], tzinfo=timezone.utc)
            except ValueError:
                pass
            return datetime.max.replace(tzinfo=timezone.utc)

        live = [cd for cd in details if _expiry(cd) > now_utc] or details
        live.sort(key=lambda cd: cd.contract.lastTradeDateOrContractMonth)
        self._contract = live[0].contract
        logger.info(
            f"📄 Front month: {self._contract.localSymbol} "
            f"(exp {self._contract.lastTradeDateOrContractMonth})"
        )

    async def _fetch_bars(self) -> Optional[pd.DataFrame]:
        """Fetch recent completed 15m bars as an OHLCV DataFrame (UTC index)."""
        bars = await self.ib.reqHistoricalDataAsync(
            self._contract,
            endDateTime="",
            durationStr="5 D",
            barSizeSetting="15 mins",
            whatToShow="TRADES",
            useRTH=False,  # matches validated backtest indicator computation
            formatDate=2,
        )
        if not bars or len(bars) < 60:
            logger.warning(f"⚠️ Insufficient bars from IB: {0 if not bars else len(bars)}")
            return None
        df = pd.DataFrame(
            {
                "open": [b.open for b in bars],
                "high": [b.high for b in bars],
                "low": [b.low for b in bars],
                "close": [b.close for b in bars],
                "volume": [b.volume for b in bars],
            },
            index=pd.DatetimeIndex(
                [pd.Timestamp(b.date).tz_localize("UTC") if pd.Timestamp(b.date).tzinfo is None
                 else pd.Timestamp(b.date).tz_convert("UTC") for b in bars]
            ),
        )
        # Drop the in-progress bar: keep bars strictly older than "now - 1s
        # aligned to 15m".  IB's last row is the forming bar during RTH.
        cutoff = pd.Timestamp.now(tz="UTC").floor("15min")
        df = df[df.index < cutoff]
        return df

    # ------------------------------------------------------------------
    #  Enrichment (replaces the deleted feature/HTF plumbing)
    # ------------------------------------------------------------------
    @staticmethod
    def _enrich(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # Indicators the strategy computes internally on its own copy — we
        # recompute them here too so the emitted record's regime/ADX/ATR match
        # what the strategy actually saw.
        out["EMA_9"] = _ema(out["close"], 9)
        out["EMA_21"] = _ema(out["close"], 21)
        out["EMA_50"] = _ema(out["close"], 50)
        out["RSI_14"] = _rsi(out["close"], 14)
        out["ATR_14"] = _atr(out["high"], out["low"], out["close"], 14)
        out["ADX_14"] = _adx(out["high"], out["low"], out["close"], 14)
        # MACD histogram (12/26/9) — strategy reads MACDhist_12_26_9
        ema12 = out["close"].ewm(span=12, adjust=False).mean()
        ema26 = out["close"].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        out["MACDhist_12_26_9"] = macd - macd.ewm(span=9, adjust=False).mean()
        # Previous-day high/low (ET calendar days, all sessions)
        day_key = pd.Series(out.index.tz_convert(ET).date, index=out.index)
        per_day_high = out["high"].groupby(day_key).max()
        per_day_low = out["low"].groupby(day_key).min()
        out["PDH"] = day_key.map(per_day_high.shift(1)).ffill().fillna(0.0)
        out["PDL"] = day_key.map(per_day_low.shift(1)).ffill().fillna(0.0)
        # HTF 30m trend — EMA(20) of 30m closes, slope over 3 bars
        closes_30m = out["close"].resample("30min", label="right", closed="right").last().dropna()
        trend = "UNKNOWN"
        if len(closes_30m) >= 5:
            ema = closes_30m.ewm(span=20, adjust=False).mean()
            slope = float(ema.iloc[-1] - ema.iloc[-3]) if len(ema) >= 3 else 0.0
            c, e = float(closes_30m.iloc[-1]), float(ema.iloc[-1])
            trend = "UP" if (c > e and slope > 0) else "DOWN" if (c < e and slope < 0) else "NEUTRAL"
        out.attrs["htf_30m_trend"] = trend
        return out

    # ------------------------------------------------------------------
    #  Core evaluation (pure — testable without IB)
    # ------------------------------------------------------------------
    def evaluate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run the strategy on an OHLCV frame and build the signal record."""
        enriched = self._enrich(df)
        signal = self.strategy.generate(enriched)
        latest = enriched.iloc[-1]
        meta = signal.metadata or {}
        entry = float(latest["close"])
        stop = meta.get("stop_loss")
        target = meta.get("take_profit")
        expected_r = None
        if stop is not None and target is not None:
            risk = abs(entry - float(stop))
            if risk > 0:
                expected_r = round(abs(float(target) - entry) / risk, 2)
        adx = float(latest.get("ADX_14", 0) or 0)
        atr = float(latest.get("ATR_14", 0) or 0)
        regime = "TRENDING" if adx >= 25 else "TRANSITIONAL" if adx >= 18 else "RANGE"
        record: Dict[str, Any] = {
            "ts": now_cst().isoformat(),
            "bar_ts": str(enriched.index[-1]),
            "signal": signal.action,
            "confidence": round(float(signal.confidence), 3),
            "entry": entry,
            "stop": float(stop) if stop is not None else None,
            "target": float(target) if target is not None else None,
            "expected_r": expected_r,
            "regime": regime,
            "adx": round(adx, 1),
            "atr": round(atr, 2),
            "htf_30m_trend": enriched.attrs.get("htf_30m_trend", "UNKNOWN"),
            "supporting_evidence": meta.get("reason"),
            "blocking_evidence": meta.get("gate_diag"),
            "strategy": "es_fifteen_min",
        }
        return record

    # ------------------------------------------------------------------
    #  Emission
    # ------------------------------------------------------------------
    def _emit(self, record: Dict[str, Any]) -> None:
        SIGNAL_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(SIGNAL_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")
        action = record["signal"]
        if action == "HOLD":
            logger.info(
                f"📊 HOLD | {record['regime']} ADX={record['adx']} | "
                f"{record.get('blocking_evidence') or record.get('supporting_evidence')}"
            )
            return
        line = (
            f"🚨 MES SIGNAL: {action} @ {record['entry']:.2f} | "
            f"SL={record['stop']} TP={record['target']} R={record['expected_r']} | "
            f"conf={record['confidence']:.2f} | {record['regime']} "
            f"(ADX={record['adx']}, HTF={record['htf_30m_trend']})\n"
            f"   {record.get('supporting_evidence')}"
        )
        logger.info(line)
        if self._telegram is not None:
            self._telegram.send_message_background(
                f"<b>MES {action}</b> @ {record['entry']:.2f}\n"
                f"SL {record['stop']} · TP {record['target']} · R {record['expected_r']}\n"
                f"conf {record['confidence']:.2f} · {record['regime']} · "
                f"HTF {record['htf_30m_trend']}\n{record.get('supporting_evidence')}"
            )

    # ------------------------------------------------------------------
    #  Main loop
    # ------------------------------------------------------------------
    async def _seconds_to_next_bar(self) -> float:
        now = pd.Timestamp.now(tz="UTC")
        nxt = (now.floor("15min") + pd.Timedelta(minutes=15)) + pd.Timedelta(seconds=10)
        return max(1.0, (nxt - now).total_seconds())

    async def start(self) -> None:
        self._running = True
        await self._connect()
        await self._resolve_front_month()
        logger.info("🟢 MES signal bot running — 15m cycle, signal-only")
        while self._running:
            try:
                df = await self._fetch_bars()
                if df is not None and len(df) >= 60:
                    last_ts = df.index[-1]
                    if last_ts != self._last_bar_ts:
                        self._last_bar_ts = last_ts
                        record = self.evaluate(df)
                        self._emit(record)
            except (ConnectionError, asyncio.TimeoutError) as exc:
                logger.error(f"IB data error: {exc} — reconnecting next cycle")
                try:
                    if not self.ib.isConnected():
                        await self._connect()
                        await self._resolve_front_month()
                except Exception as reconnect_exc:  # noqa: BLE001
                    logger.error(f"Reconnect failed: {reconnect_exc}")
            except Exception as exc:  # noqa: BLE001
                logger.exception(f"Signal cycle error: {exc}")
            # Interruptible wait: poll _running each second so SIGTERM exits
            # promptly instead of blocking up to a full 15m bar.
            remaining = await self._seconds_to_next_bar()
            while remaining > 0 and self._running:
                await asyncio.sleep(min(1.0, remaining))
                remaining -= 1.0

    async def stop(self) -> None:
        self._running = False
        if self._telegram is not None:
            await self._telegram.close()
        if self.ib.isConnected():
            self.ib.disconnect()
        logger.info("🛑 MES signal bot stopped")
