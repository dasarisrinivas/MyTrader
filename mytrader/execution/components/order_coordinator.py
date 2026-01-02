"""Order tracking, metadata, and trade lifecycle helpers."""

import contextlib
import math
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ...learning.trade_learning import ExecutionMetrics, TradeLearningPayload
# Use S3 RAGStorageManager TradeRecord instead of local SQLite
from ...rag.rag_storage_manager import TradeRecord as RAGTradeRecord
from ...risk.trade_math import compute_risk_reward, expected_target_outcome
from ...strategies.market_regime import detect_market_regime, get_regime_parameters
from ...utils.structured_logging import log_structured_event
from ...utils.logger import logger
from ...utils.timezone_utils import now_cst
from ...utils.timezone_utils import is_market_hours_cst
from .risk_controller import TradeRequest


class OrderCoordinator:
    """Encapsulates order metadata and trade lifecycle persistence."""

    def __init__(self, manager: "LiveTradingManager"):  # noqa: F821 (forward reference)
        self.manager = manager
        self._recent_signal_keys: Dict[str, datetime] = {}
        trading_cfg = getattr(getattr(manager, "settings", None), "trading", None)
        ttl = getattr(trading_cfg, "decision_min_interval_seconds", 300) if trading_cfg else 300
        self._signal_ttl_seconds: int = max(300, int(ttl))

    def prepare_order_metadata(
        self,
        base_metadata: Optional[Dict[str, Any]],
        entry_price: float,
        strategy_name: str,
    ) -> Dict[str, Any]:
        metadata = dict(base_metadata or {})
        metadata.setdefault("trade_cycle_id", self.manager._current_cycle_id)
        candle_ts = self.manager._last_candle_processed
        if isinstance(candle_ts, datetime):
            metadata.setdefault("bar_close_timestamp", candle_ts.isoformat())
        else:
            metadata.setdefault(
                "bar_close_timestamp",
                now_cst().replace(second=0, microsecond=0).isoformat(),
            )
        metadata.setdefault(
            "signal_id",
            metadata.get("signal_id") or metadata.get("signal_type") or self.manager._current_cycle_id,
        )
        metadata.setdefault("strategy_name", strategy_name)
        metadata["entry_price"] = entry_price
        metadata.setdefault("entry_price_bucket", self.bucket_entry_price(entry_price))
        return metadata

    def bucket_entry_price(self, price: float) -> float:
        tick_size = max(getattr(self.manager.settings.trading, "tick_size", 0.25), 1e-6)
        bucket_ticks = max(1, getattr(self.manager.settings.trading, "min_stop_distance_ticks", 4))
        bucket_size = tick_size * bucket_ticks
        return round(price / bucket_size) * bucket_size

    def register_trade_entry(
        self,
        cycle_id: Optional[str],
        action: str,
        quantity: int,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        metadata = metadata or {}
        cycle_id = cycle_id or self.manager._current_cycle_id or uuid.uuid4().hex[:12]

        self.manager._current_entry_cycle_id = cycle_id
        context = self.manager._cycle_context.get(cycle_id, {})
        is_long = action in ("BUY", "SCALP_BUY")
        self.manager._open_trade_context = {
            "cycle_id": cycle_id,
            "action": action,
            "is_long": is_long,
            "quantity": quantity,
            "entry_price": entry_price,
            "entry_time": now_cst().isoformat(),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "metadata": metadata,
            "regime": context.get("regime"),
            "volatility": context.get("volatility"),
            "aws": context.get("aws"),
            "signal_confidence": context.get("signal_confidence"),
            "signal_type": context.get("signal_type"),
            "features": self.manager.current_trade_features or {},
        }
        reason_codes = context.get("reason_codes", set())
        self.manager._active_reason_codes = set(reason_codes)

        logger.info(
            f"📝 Registered trade entry: cycle_id={cycle_id}, action={action}, qty={quantity}, entry={entry_price:.2f}"
        )

    async def finalize_trade(
        self,
        exit_price: float,
        exit_time: datetime,
        realized_pnl: float,
    ) -> None:
        if not self.manager._open_trade_context or not self.manager.learning_recorder:
            return
        ctx = self.manager._open_trade_context
        direction = 1 if ctx.get("is_long") else -1
        contract = getattr(self.manager.settings.data, "ibkr_symbol", "ES")
        payload = TradeLearningPayload(
            trade_cycle_id=ctx["cycle_id"],
            symbol=contract,
            contract=contract,
            quantity=ctx["quantity"],
            side=ctx["action"],
            entry_time=ctx["entry_time"],
            entry_price=ctx["entry_price"],
            exit_time=exit_time.isoformat(),
            exit_price=exit_price,
            stop_loss=ctx["stop_loss"],
            take_profit=ctx["take_profit"],
            signal_type=ctx.get("signal_type"),
            signal_confidence=ctx.get("signal_confidence"),
            regime=ctx.get("regime"),
            volatility=ctx.get("volatility"),
            advisory=ctx.get("aws"),
            risk_parameters={
                "distance_points": abs(ctx["entry_price"] - ctx["stop_loss"]),
                "target_points": abs(ctx["take_profit"] - ctx["entry_price"]),
            },
            features=ctx.get("features") or {},
            execution_metrics=ExecutionMetrics(),
            reason_codes=sorted(self.manager._active_reason_codes),
            outcome="LOSS" if realized_pnl < 0 else "WIN" if realized_pnl > 0 else "BREAKEVEN",
            pnl=realized_pnl,
        )
        self.manager.learning_recorder.record_outcome(payload)
        if self.manager._local_kb:
            try:
                self.manager._local_kb.record_trade(payload)
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"Local KB record skipped: {exc}")
        candles = self.manager.price_history[-120:]
        history_ctx = {
            "regime": ctx.get("regime"),
            "volatility": ctx.get("volatility"),
            "symbol": contract,
        }
        if candles:
            self.manager.learning_recorder.record_history_snapshot(
                payload.trade_cycle_id,
                candles,
                history_ctx,
            )
        self.manager._open_trade_context = None
        self.manager._active_reason_codes.clear()
        self.manager._cycle_context.pop(payload.trade_cycle_id, None)
        if self.manager._current_entry_cycle_id == payload.trade_cycle_id:
            self.manager._current_entry_cycle_id = None

    async def handle_order_fill(self, trade, fill) -> None:
        """Handle execution details to persist fills and outcomes."""
        m = self.manager
        try:
            if not m.rag_storage or not m.current_trade_id:
                pass

            realized_pnl = 0.0
            commission_report = getattr(fill, "commissionReport", None)
            if commission_report and hasattr(commission_report, "realizedPNL"):
                pnl_value = commission_report.realizedPNL
                if pnl_value is not None and abs(pnl_value) > 0.001:
                    realized_pnl = pnl_value

            exit_time = datetime.now(timezone.utc)
            if m.rag_storage and m.current_trade_id:
                hold_seconds = 0
                if m.current_trade_entry_time:
                    try:
                        entry_time = datetime.fromisoformat(
                            m.current_trade_entry_time.replace("Z", "+00:00")
                        )
                        if entry_time.tzinfo is None:
                            entry_time = entry_time.replace(tzinfo=timezone.utc)
                        hold_seconds = int((exit_time - entry_time).total_seconds())
                    except Exception as exc:  # noqa: BLE001
                        logger.error(f"Error calculating hold time: {exc}")

                # Build S3 TradeRecord with proper fields
                features = m.current_trade_features or {}
                rationale = m.current_trade_rationale or {}
                record = RAGTradeRecord(
                    trade_id=m.current_trade_id,
                    timestamp=m.current_trade_entry_time or now_cst().isoformat(),
                    action=m.current_trade_action or "UNKNOWN",
                    entry_price=m.current_trade_entry_price or 0.0,
                    exit_price=fill.execution.price,
                    quantity=int(fill.execution.shares),
                    stop_loss=float(features.get("stop_loss", 0.0)),
                    take_profit=float(features.get("take_profit", 0.0)),
                    rsi=float(features.get("RSI_14", features.get("rsi", 50.0))),
                    macd_hist=float(features.get("MACD_hist", features.get("macd_hist", 0.0))),
                    ema_9=float(features.get("EMA_9", features.get("ema_9", 0.0))),
                    ema_20=float(features.get("EMA_20", features.get("ema_20", 0.0))),
                    atr=float(features.get("ATR_14", features.get("atr", 0.0))),
                    result="WIN" if realized_pnl > 0 else "LOSS" if realized_pnl < 0 else "SCRATCH",
                    realized_pnl=realized_pnl,
                    hold_time_seconds=hold_seconds,
                    market_regime=str(features.get("market_regime", "")),
                    volatility_regime=str(features.get("volatility_regime", "")),
                    session=str(features.get("session", "")),
                    confidence=float(rationale.get("confidence", 0.0)),
                    signal_source=str(rationale.get("signal_source", "hybrid")),
                )
                m.rag_storage.save_trade(record)
                logger.info("✅ Saved trade %s to S3 RAG (P&L: $%.2f)", m.current_trade_id, realized_pnl)
                m.current_trade_id = None
                m.current_trade_entry_time = None
                m.current_trade_entry_price = None
                m.current_trade_action = None
                m._record_last_trade_timestamp()

            await m._finalize_trade(
                exit_price=fill.execution.price,
                exit_time=exit_time,
                realized_pnl=realized_pnl,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Error handling order fill: {exc}")

    def _build_signal_key(self, action: str, metadata: Dict[str, Any]) -> str:
        """Create a coarse idempotency key for the candle/signal."""
        ts = metadata.get("bar_close_timestamp") or metadata.get("timestamp")
        if not ts:
            ts = now_cst().replace(second=0, microsecond=0).isoformat()
        symbol = getattr(getattr(self.manager, "settings", None), "data", None)
        symbol = getattr(symbol, "ibkr_symbol", "UNKNOWN")
        return f"{symbol}|{action.upper()}|{ts}"

    def _prune_signal_keys(self, now: datetime) -> None:
        stale = [
            key for key, ts in self._recent_signal_keys.items()
            if (now - ts).total_seconds() > self._signal_ttl_seconds
        ]
        for key in stale:
            self._recent_signal_keys.pop(key, None)

    async def enforce_entry_gates(
        self,
        action: str,
        metadata: Dict[str, Any],
        allow_existing_position: bool = False,
    ) -> tuple[bool, str, Optional[str]]:
        """
        Enforce idempotency, cooldown spacing, and single-position rules before submitting an entry.
        """
        now = datetime.now(timezone.utc)
        self._prune_signal_keys(now)
        signal_key = self._build_signal_key(action, metadata)

        if signal_key in self._recent_signal_keys:
            logger.warning("Duplicate submission blocked for key=%s", signal_key)
            return False, "DUPLICATE_SIGNAL", signal_key

        # Reserve the key immediately to close race window
        self._recent_signal_keys[signal_key] = now

        # Enforce minimum spacing between orders
        cooldown_seconds = getattr(self.manager, "_cooldown_seconds", 0) or 0
        lock = getattr(self.manager, "_trade_time_lock", None)
        with lock if lock else contextlib.nullcontext():
            last_trade = getattr(self.manager, "_last_trade_time", None)
        if cooldown_seconds and last_trade:
            elapsed = (now - last_trade).total_seconds()
            remaining = cooldown_seconds - elapsed
            if remaining > 0:
                logger.info("Cooldown active (%.1fs remaining) - blocking entry", remaining)
                self._recent_signal_keys.pop(signal_key, None)
                return False, "COOLDOWN_ACTIVE", signal_key

        enforce_market_hours = getattr(
            getattr(self.manager, "settings", None),
            "trading",
            None,
        )
        enforce_market_hours = getattr(enforce_market_hours, "enforce_market_hours", True)
        if enforce_market_hours and not is_market_hours_cst():
            logger.info("Market closed per hours check - blocking entry")
            self._recent_signal_keys.pop(signal_key, None)
            return False, "MARKET_CLOSED", signal_key

        if not allow_existing_position and getattr(self.manager, "executor", None):
            try:
                position = await self.manager.executor.get_current_position()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Position lookup failed; blocking entry: %s", exc)
                self._recent_signal_keys.pop(signal_key, None)
                return False, "POSITION_UNKNOWN", signal_key
            if position and getattr(position, "quantity", 0) != 0:
                logger.info("Open position detected (%s); blocking new entry", position.quantity)
                self._recent_signal_keys.pop(signal_key, None)
                return False, "POSITION_OPEN", signal_key

        metadata.setdefault("dedupe_key", signal_key)
        return True, "OK", signal_key

    def record_signal_key(self, signal_key: Optional[str]) -> None:
        """Record or refresh a signal key after downstream validations succeed."""
        if not signal_key:
            return
        self._recent_signal_keys[signal_key] = datetime.now(timezone.utc)

    async def execute_trade_with_risk_checks(self, signal, current_price: float, features):
        """Place an order with sizing, stops, and hard guardrails."""
        m = self.manager
        try:
            logger.info("🛠️ DEBUG: Entered execute_trade_with_risk_checks for %s", signal.action)

            raw_metadata = signal.metadata if isinstance(signal.metadata, dict) else {}
            metadata = self.prepare_order_metadata(raw_metadata, current_price, "legacy")
            allowed, reason, signal_key = await self.enforce_entry_gates(signal.action, metadata)
            if not allowed:
                m._add_reason_code(reason)
                log_structured_event(
                    agent="live_manager",
                    event_type="risk.entry_blocked",
                    message=f"Entry blocked by submission gate: {reason}",
                    payload={"trade_cycle_id": m._current_cycle_id},
                )
                return
            qty = m.trade_decision_engine.calculate_position_size(signal, metadata)

            if not m.risk.can_trade(qty):
                await m._broadcast_error("Risk limits exceeded")
                return

            row = features.iloc[-1]
            m.current_trade_features = {
                "confidence": signal.confidence,
                "atr": float(row.get("ATR_14", 0.0)),
                "volatility": float(row.get("volatility_5m", 0.0)),
                "rsi": float(row.get("RSI_14", 0.0)),
                "macd": float(row.get("MACD", 0.0)),
                "close": float(row["close"]),
                "volume": int(row.get("volume", 0)),
            }

            atr = float(metadata.get("atr_value", 0.0))
            if atr <= 0:
                atr = float(row.get("ATR_14", 0.0))

            regime, regime_conf = detect_market_regime(features)
            regime_params = get_regime_parameters(regime)

            logger.info("📊 Market Regime: %s (conf=%.2f) - Using dynamic stops", regime.value, regime_conf)

            stop_loss_hint = metadata.get("stop_loss")
            take_profit_hint = metadata.get("take_profit")
            risk_meta: Dict[str, Any] = {}
            if stop_loss_hint is not None and take_profit_hint is not None:
                stop_loss = float(stop_loss_hint)
                take_profit = float(take_profit_hint)
                metadata.setdefault("atr_fallback_used", False)
            else:
                stop_loss, take_profit, risk_meta = m.risk_controller.calculate_stop_loss(
                    entry_price=current_price,
                    action=signal.action,
                    atr=atr,
                    regime_params={**regime_params, "volatility": regime.value},
                )
            metadata.update(risk_meta)

            allowed_gate, gate_levels = await m._enforce_risk_gate(
                signal.action,
                qty,
                current_price,
                atr,
                stop_loss,
                take_profit,
            )
            if not allowed_gate:
                self._recent_signal_keys.pop(signal_key, None)
                return
            if gate_levels:
                stop_loss = gate_levels.get("stop_loss", stop_loss)
                take_profit = gate_levels.get("take_profit", take_profit)
                metadata.update(gate_levels)

            for label, value in (("stop_loss", stop_loss), ("take_profit", take_profit)):
                if value is None or not math.isfinite(value) or value == 0:
                    logger.warning("⚠️ %s invalid (%s); blocking trade", label, value)
                    m._add_reason_code("INVALID_PROTECTION")
                    return

            if atr > 0:
                logger.info(
                    "🎯 Dynamic Risk: ATR=%.2f, SL=%.2f, TP=%.2f",
                    atr,
                    abs(current_price - stop_loss),
                    abs(take_profit - current_price),
                )
            else:
                logger.info(
                    "🎯 Dynamic Risk fallback: ATR=%.2f, SL=%.2f, TP=%.2f",
                    atr,
                    abs(current_price - stop_loss),
                    abs(take_profit - current_price),
                )

            await m._broadcast_order_update(
                {
                    "status": "placing",
                    "action": signal.action,
                    "quantity": qty,
                    "entry_price": current_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "regime": regime.value,
                }
            )

            logger.info(
                "📊 Order telemetry | qty=%d position=%d lock=%s entry=%.2f SL=%.2f TP=%.2f fallback=%s",
                qty,
                m.status.current_position,
                m.executor.is_order_locked() if m.executor else False,
                current_price,
                stop_loss,
                take_profit,
                "yes" if metadata.get("atr_fallback_used") else "no",
            )

            trade_request = TradeRequest(
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                quantity=qty,
                action=signal.action,
            )
            risk_result = m.risk_controller.validate_trade_risk(trade_request)
            if not risk_result.allowed:
                m._add_reason_code("RISK_BLOCKED_INVALID_PROTECTION")
                log_structured_event(
                    agent="live_manager",
                    event_type="risk.entry_blocked",
                    message="Entry blocked by hard guardrails",
                    payload={"trade_cycle_id": m._current_cycle_id},
                )
                return

            if getattr(m, "one_minute_cfg", None) and getattr(m.one_minute_cfg, "dry_run", False):
                logger.warning(
                    "🔶 DRY RUN: Would place %s order for %d @ %.2f (SL=%.2f TP=%.2f)",
                    signal.action,
                    qty,
                    current_price,
                    stop_loss,
                    take_profit,
                )
                m._record_submission_timestamp()
                self.record_signal_key(signal_key)
                return

            if m.simulation_mode:
                logger.warning(
                    "🔶 SIMULATION: Would place %s order for %d contracts @ %.2f",
                    signal.action,
                    qty,
                    current_price,
                )
                logger.warning("   SL: %.2f, TP: %.2f", stop_loss, take_profit)
                m._record_submission_timestamp()
                self.record_signal_key(signal_key)
                await m._broadcast_order_update(
                    {
                        "status": "SIMULATED",
                        "action": signal.action,
                        "quantity": qty,
                        "fill_price": current_price,
                        "filled_quantity": qty,
                        "order_id": f"SIM-{datetime.now().strftime('%H%M%S')}",
                    }
                )
                return

            _, reward_points, rr_ratio = compute_risk_reward(
                current_price,
                stop_loss,
                take_profit,
                signal.action,
            )
            projected = expected_target_outcome(
                current_price,
                take_profit,
                qty,
                m.contract_spec,
                m.trading_mode,
                m._commission_per_side,
            )
            metadata["risk_reward"] = rr_ratio
            metadata["expected_reward_points"] = reward_points
            metadata["expected_gross"] = projected.gross_pnl
            metadata["expected_net"] = projected.net_pnl

            # Final TOCTOU check: ensure position still flat before submitting entry
            try:
                current_pos = await m.executor.get_current_position()
                if current_pos and getattr(current_pos, "quantity", 0) != 0:
                    logger.info("Position changed before submit (qty=%s); skipping entry", current_pos.quantity)
                    self.record_signal_key(None)
                    return
            except Exception as exc:  # noqa: BLE001
                logger.warning("Position recheck failed; skipping entry to be safe: %s", exc)
                return

            result = await m.executor.place_order(
                action=signal.action,
                quantity=qty,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata=metadata,
                rationale=m.current_trade_rationale,
                features=m.current_trade_features,
                market_regime=regime.value,
                entry_price=current_price,
            )

            await m._broadcast_order_update(
                {
                    "status": result.status,
                    "action": signal.action,
                    "quantity": qty,
                    "fill_price": result.fill_price,
                    "filled_quantity": result.filled_quantity,
                    "order_id": result.trade.order.orderId if result.trade else None,
                }
            )

            if result.status not in {"Cancelled", "Inactive"}:
                self.record_signal_key(signal_key)
                if result.fill_price or result.filled_quantity:
                    m._record_last_trade_timestamp()
                    logger.info("⏱️ Trade fill - cooldown activated")
                else:
                    m._record_submission_timestamp()
                    logger.info("⏱️ Submission recorded (no fill yet)")

                m.risk.register_trade()
                if result.fill_price:
                    m.tracker.record_trade(
                        action=signal.action,
                        price=result.fill_price,
                        quantity=qty,
                    )
                    m._update_status_from_tracker()
                    m._register_trade_entry(
                        cycle_id=m._current_cycle_id,
                        action=signal.action,
                        quantity=qty,
                        entry_price=result.fill_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        metadata=metadata,
                    )

                    if m.rag_storage:
                        try:
                            trade_uuid = str(uuid.uuid4())
                            m.current_trade_id = trade_uuid
                            entry_time = datetime.now(timezone.utc).isoformat()

                            m.current_trade_entry_time = entry_time
                            m.current_trade_entry_price = result.fill_price
                            m.current_trade_action = signal.action
                            m.current_trade_features = {
                                "confidence": signal.confidence,
                                "atr": atr,
                                "volatility": row.get("volatility_5m", 0.0),
                                "stop_loss": stop_loss,
                                "take_profit": take_profit,
                                "RSI_14": row.get("RSI_14", 50.0),
                                "MACD_hist": row.get("MACD_hist", 0.0),
                                "EMA_9": row.get("EMA_9", 0.0),
                                "EMA_20": row.get("EMA_20", 0.0),
                                "ATR_14": atr,
                                "market_regime": getattr(m.status, "hybrid_market_regime", ""),
                                "volatility_regime": getattr(m.status, "hybrid_volatility_regime", ""),
                            }
                            m.current_trade_rationale = {
                                "confidence": signal.confidence,
                                "signal_source": signal.metadata.get("signal_source", "hybrid") if signal.metadata else "hybrid",
                            }

                            logger.info("📝 Trade entry tracking started: %s %s @ %.2f", signal.action, trade_uuid, result.fill_price)

                        except Exception as exc:  # noqa: BLE001
                            logger.error(f"Failed to setup trade tracking: {exc}")

        except Exception as exc:  # noqa: BLE001
            logger.error(f"❌ CRITICAL: execute_trade_with_risk_checks failed with exception: {exc}")
            import traceback

            logger.error("Traceback: %s", traceback.format_exc())
            await m._broadcast_error(f"Order placement failed: {exc}")

    def add_reason_code(self, code: str) -> None:
        if not code:
            return
        self.manager._active_reason_codes.add(code)
        cycle_ctx = self.manager._cycle_context.get(self.manager._current_cycle_id)
        if cycle_ctx is not None:
            if "reason_codes" not in cycle_ctx or not isinstance(cycle_ctx["reason_codes"], set):
                cycle_ctx["reason_codes"] = set()
            cycle_ctx["reason_codes"].add(code)
