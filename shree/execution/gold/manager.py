"""GoldTradingManager — async event-driven trading loop for COMEX Gold futures.

Runs *completely independently* of LiveTradingManager and the MES strategy.

Architecture:
    - One IB() connection with its own client_id (default 3)
    - Subscribes to real-time bars via reqRealTimeBars (5-sec aggregated)
      or polls reqHistoricalData every minute
    - On each completed 1-min bar: run strategy → assess risk → place/manage orders
    - Separate SQLite order tracker, state file, journal dir
    - Flatten all positions before COMEX maintenance window

Idempotency:
    - OrderLockManager prevents overlapping submissions
    - State is loaded from disk on start; in-flight orders are reconciled
      against IB's open-order list before the first trade is attempted

Simulation mode:
    - When config.simulation=True no orders are sent; signals and sizing
      are still computed and logged so paper-trading parity is easy to verify
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from ib_insync import IB, BarData, Future, LimitOrder, MarketOrder, StopOrder, Trade

from ...config.gold import GoldStrategyConfig
from ...risk.trade_math import get_contract_spec
from ...strategies.gold.strategy import GoldIntradayStrategy, compute_indicators
from ...strategies.gold.signals import GoldSignal
from ...strategies.gold.regime import GoldRegime
from ...utils.logger import logger
from ...utils.timezone_utils import now_cst
from ...utils.structured_logging import log_structured_event
from ..order_lock import OrderLockManager
from .contract import GoldContractFactory
from .journal import GoldJournal, GoldTradeRecord
from .risk import DailyState, GoldRiskManager
from .rollover import ContractRollMonitor
from .state import GoldDayState, GoldStateManager


# Seconds between bar-close processing loops (IB sends 5-sec bars; we aggregate)
_LOOP_INTERVAL_SECONDS = 5

# How often to log heartbeat when idle
_HEARTBEAT_LOG_INTERVAL_BARS = 12   # ~1 minute on 5s loop
_IDLE_DIAGNOSTIC_LOG_INTERVAL_BARS = 12
_BAR_WATCHDOG_WARN_SECONDS = 20
_HISTORICAL_POLL_LOOKBACK = "30 M"


class _OpenPosition:
    """Tracks an in-flight bracket trade from entry fill to exit fill."""

    def __init__(
        self,
        trade_id: str,
        action: str,
        signal: GoldSignal,
        contracts: int,
        entry_bar: int,
        entry_time: datetime,
    ) -> None:
        self.trade_id = trade_id
        self.action = action
        self.signal = signal
        self.contracts = contracts
        self.entry_bar = entry_bar          # Bar index at entry
        self.entry_time = entry_time        # UTC datetime at entry
        self.entry_price: Optional[float] = None   # Set when entry fills
        self.exit_price: Optional[float] = None
        self.exit_reason: str = ""
        self.exit_time: Optional[datetime] = None
        self.entry_order_id: Optional[int] = None
        self.sl_order_id: Optional[int] = None
        self.tp_order_id: Optional[int] = None
        # Trailing stop state
        self.trail_activated: bool = False
        self.best_price: Optional[float] = None


class GoldTradingManager:
    """Async trading manager for intraday COMEX Gold futures.

    Usage::

        manager = GoldTradingManager(config)
        await manager.start()   # Runs until SIGINT / stop()
    """

    def __init__(self, config: GoldStrategyConfig) -> None:
        self._cfg = config
        self._spec = get_contract_spec(config.symbol)
        self._strategy = GoldIntradayStrategy(config)
        self._state_mgr = GoldStateManager(Path(config.state_file))
        self._journal = GoldJournal(Path(config.journal_dir))
        self._lock = OrderLockManager()
        self._risk_mgr = GoldRiskManager(
            config.risk,
            self._spec,
            effective_max_risk=config.gc_adjusted_risk_usd(),
        )

        # Runtime state (populated in start())
        self._ib: Optional[IB] = None
        self._contract: Optional[Future] = None
        self._contract_factory: Optional[GoldContractFactory] = None
        self._roll_monitor = ContractRollMonitor(config)
        self._day_state: GoldDayState = GoldDayState()

        # Bar buffer
        self._raw_bars: List[Dict] = []        # List of OHLCV dicts
        self._bar_counter: int = 0
        self._last_bar_minute: Optional[int] = None
        self._bar_list = None                  # RealTimeBarList from reqRealTimeBars
        self._last_realtime_bar_at: Optional[datetime] = None
        self._last_processed_bar_time: Optional[pd.Timestamp] = None

        # Active position tracking
        self._position: Optional[_OpenPosition] = None

        # IB order maps (order_id → Trade)
        self._active_trades: Dict[int, Trade] = {}

        # Control
        self._running: bool = False
        self._shutdown_event: asyncio.Event = asyncio.Event()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Connect, load state, subscribe to data, enter the main loop."""
        logger.info(
            "GoldTradingManager: starting — symbol={} exchange={} sim={} port={} cid={}",
            self._cfg.symbol,
            self._cfg.exchange,
            self._cfg.simulation,
            self._cfg.ibkr_port,
            self._cfg.ibkr_client_id,
        )

        # Validate config safety
        self._cfg.validate()

        # Load persisted state
        self._day_state = self._state_mgr.load()

        # Connect to IB Gateway
        self._ib = IB()
        await self._ib.connectAsync(
            self._cfg.ibkr_host,
            self._cfg.ibkr_port,
            clientId=self._cfg.ibkr_client_id,
        )
        logger.info("GoldTradingManager: connected to IB Gateway")

        # Qualify the gold futures contract
        self._contract_factory = GoldContractFactory(self._cfg, self._ib)
        self._contract = await self._contract_factory.get_qualified_contract()
        logger.info(
            "GoldTradingManager: contract qualified — {} ({})",
            self._contract.localSymbol or self._contract.symbol,
            self._contract.lastTradeDateOrContractMonth,
        )
        self._roll_monitor.check_and_warn(self._contract)

        # Reconcile any open orders from a prior session
        await self._reconcile_open_orders()

        # Subscribe to 5-second real-time bars.
        # reqRealTimeBars returns a RealTimeBarList; subscribe to its updateEvent.
        self._ib.errorEvent += self._on_ib_error
        self._bar_list = self._ib.reqRealTimeBars(
            self._contract,
            barSize=5,
            whatToShow="TRADES",
            useRTH=False,
        )
        self._bar_list.updateEvent += self._on_realtime_bar

        self._running = True
        logger.info("GoldTradingManager: entering main loop")

        try:
            while self._running and not self._shutdown_event.is_set():
                await asyncio.sleep(_LOOP_INTERVAL_SECONDS)
                await self._check_realtime_bar_watchdog()
                await self._maintenance_guard()
        except asyncio.CancelledError:
            logger.info("GoldTradingManager: loop cancelled")
        finally:
            await self._shutdown()

    def stop(self) -> None:
        """Signal the manager to stop gracefully."""
        logger.info("GoldTradingManager: stop requested")
        self._running = False
        self._shutdown_event.set()

    # ── IB event handlers ─────────────────────────────────────────────────────

    def _on_realtime_bar(self, bars: "BarDataList", has_new_bar: bool) -> None:  # type: ignore[name-defined]
        """Called by ib_insync every 5 seconds with a new real-time bar."""
        if not has_new_bar or not bars:
            return
        bar: BarData = bars[-1]
        # ib_insync ≥ 0.9.70 returns bar.time as a datetime; older versions as int
        if isinstance(bar.time, datetime):
            bar_dt = bar.time if bar.time.tzinfo else bar.time.replace(tzinfo=timezone.utc)
        else:
            bar_dt = datetime.fromtimestamp(bar.time, tz=timezone.utc)
        self._last_realtime_bar_at = datetime.now(timezone.utc)
        current_minute = bar_dt.minute

        # Accumulate raw bar data for 1-min aggregation
        self._raw_bars.append(
            {
                "time": bar_dt,
                "open": bar.open_,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
        )
        # Keep last 2 hours of 5-second bars in memory
        max_raw = 1440  # 2h × 60min/h × 12 bars/min
        if len(self._raw_bars) > max_raw:
            self._raw_bars = self._raw_bars[-max_raw:]

        # Detect new minute → process completed candle
        if self._last_bar_minute is not None and current_minute != self._last_bar_minute:
            asyncio.ensure_future(self._on_minute_close(bar_dt))
        self._last_bar_minute = current_minute

    async def _check_realtime_bar_watchdog(self) -> None:
        """Warn when the real-time bar subscription has gone quiet and fall back to polling."""
        if not self._running:
            return
        if self._last_realtime_bar_at is None:
            logger.warning(
                "GoldTradingManager: no realtime bars received yet after loop start — waiting on IB data"
            )
            await self._poll_recent_1min_bar()
            return
        age = (datetime.now(timezone.utc) - self._last_realtime_bar_at).total_seconds()
        if age >= _BAR_WATCHDOG_WARN_SECONDS:
            logger.warning(
                "GoldTradingManager: realtime bar feed quiet for %.0fs — waiting on IB data/subscription",
                age,
            )
            await self._poll_recent_1min_bar()

    async def _poll_recent_1min_bar(self) -> None:
        """Fallback path: fetch recent 1-minute bars when realtime bars are unavailable."""
        if self._ib is None or self._contract is None or not self._ib.isConnected():
            return

        try:
            loop: Optional[asyncio.AbstractEventLoop]
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if hasattr(self._ib, "reqHistoricalDataAsync"):
                bars = await self._ib.reqHistoricalDataAsync(
                    self._contract,
                    endDateTime="",
                    durationStr=_HISTORICAL_POLL_LOOKBACK,
                    barSizeSetting="1 min",
                    whatToShow="TRADES",
                    useRTH=False,
                    formatDate=2,
                    keepUpToDate=False,
                    timeout=10,
                )
            elif loop is not None:
                bars = await loop.run_in_executor(
                    None,
                    lambda: self._ib.reqHistoricalData(
                        self._contract,
                        endDateTime="",
                        durationStr=_HISTORICAL_POLL_LOOKBACK,
                        barSizeSetting="1 min",
                        whatToShow="TRADES",
                        useRTH=False,
                        formatDate=2,
                        keepUpToDate=False,
                        timeout=10,
                    ),
                )
            else:
                bars = self._ib.reqHistoricalData(
                    self._contract,
                    endDateTime="",
                    durationStr=_HISTORICAL_POLL_LOOKBACK,
                    barSizeSetting="1 min",
                    whatToShow="TRADES",
                    useRTH=False,
                    formatDate=2,
                    keepUpToDate=False,
                    timeout=10,
                )
        except Exception as exc:
            logger.warning("GoldTradingManager: fallback historical poll failed: {}", exc)
            return

        if not bars or len(bars) < 2:
            return

        records = []
        for bar in bars:
            bar_time = bar.date
            if isinstance(bar_time, datetime):
                bar_dt = bar_time if bar_time.tzinfo else bar_time.replace(tzinfo=timezone.utc)
            else:
                bar_dt = pd.to_datetime(bar_time, utc=True).to_pydatetime()
            records.append(
                {
                    "time": bar_dt,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }
            )

        df = pd.DataFrame(records)
        if df.empty:
            return
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time").sort_index()

        completed_bar_time = df.index[-2]
        if self._last_processed_bar_time is not None and completed_bar_time <= self._last_processed_bar_time:
            logger.info(
                "GoldTradingManager: fallback poll saw no new completed bar (latest={})",
                completed_bar_time.isoformat(),
            )
            return

        self._last_processed_bar_time = completed_bar_time
        self._last_realtime_bar_at = datetime.now(timezone.utc)

        logger.info(
            "GoldTradingManager: using historical polling fallback for bar {}",
            completed_bar_time.isoformat(),
        )
        await self._on_minute_close(completed_bar_time.to_pydatetime())

    def _on_ib_error(self, reqId: int, errorCode: int, errorString: str, contract) -> None:
        """Log IB error/warning events so silent subscription failures are visible."""
        if errorCode in (2104, 2106, 2158, 2119):
            # Informational: market data farm connected/disconnected
            logger.debug("IB info {}: {}", errorCode, errorString)
        else:
            logger.warning("IB error reqId={} code={}: {}", reqId, errorCode, errorString)

    # ── Per-bar processing ────────────────────────────────────────────────────

    async def _on_minute_close(self, bar_dt: datetime) -> None:
        """Called when a 1-minute bar has just closed."""
        self._bar_counter += 1

        df_preview = self._build_1min_df()
        if df_preview is not None and len(df_preview) > 0:
            logger.info(
                "GoldTradingManager: processed minute bar ts={} close={:.2f} in_pos={}",
                df_preview.index[-1].isoformat(),
                float(df_preview["close"].iloc[-1]),
                self._position is not None,
            )

        # Heartbeat
        if self._bar_counter % _HEARTBEAT_LOG_INTERVAL_BARS == 0:
            pnl = self._day_state.realized_pnl_today
            logger.info(
                "GoldTradingManager: heartbeat bar={} pnl={:.2f} in_pos={}",
                self._bar_counter,
                pnl,
                self._position is not None,
            )

        # Build 1-min OHLCV DataFrame from raw bars
        df = df_preview if df_preview is not None else self._build_1min_df()
        if df is None or len(df) < 2:
            return

        # Session guard
        if not self._is_tradeable_session(bar_dt):
            if self._position is not None:
                await self._check_time_stop(df, bar_dt, force=True)
            return

        # Manage existing position exits first
        if self._position is not None:
            await self._manage_open_position(df, bar_dt)
            return

        # Signal generation and entry
        await self._attempt_entry(df, bar_dt)

    # ── Entry ─────────────────────────────────────────────────────────────────

    async def _attempt_entry(self, df: pd.DataFrame, bar_dt: datetime) -> None:
        """Evaluate signals and place entry if approved."""
        bar_ts = pd.Timestamp(bar_dt).tz_convert("America/New_York") if bar_dt.tzinfo else None
        signal: GoldSignal = self._strategy.generate_gold(df, bar_timestamp=bar_ts)

        if not signal.is_actionable:
            if self._bar_counter % _IDLE_DIAGNOSTIC_LOG_INTERVAL_BARS == 0:
                block_reason = signal.metadata.get("block_reason", "no_setup")
                orb_block = signal.metadata.get("orb_block_reason")
                pullback_block = signal.metadata.get("pullback_block_reason")
                logger.info(
                    "GoldTradingManager: no entry — regime={} bars={} close={:.2f} reason={} orb_reason={} pb_reason={}",
                    signal.regime.value,
                    len(df),
                    float(df["close"].iloc[-1]),
                    block_reason,
                    orb_block,
                    pullback_block,
                )
            return

        # Risk approval
        stop_dist = abs(signal.entry_ref_price - signal.stop_loss)
        daily = DailyState(
            realized_pnl=self._day_state.realized_pnl_today,
            trades_today=self._day_state.trades_today,
            consecutive_losses=self._day_state.consecutive_loss_count,
            cooldown_until=self._day_state.cooldown_until,
        )
        sizing = self._risk_mgr.size_position(stop_dist, daily)

        if not sizing.approved:
            logger.info(
                "GoldTradingManager: entry blocked by risk: {}", sizing.reason
            )
            return

        trade_id = str(uuid.uuid4())[:8]
        logger.info(
            "GoldTradingManager: ENTRY SIGNAL {} {} × {} — sl={:.2f} tp={:.2f} "
            "risk={:.2f} conf={:.2f} [{}]",
            signal.action,
            self._cfg.symbol,
            sizing.contracts,
            signal.stop_loss,
            signal.take_profit,
            sizing.total_risk_usd,
            signal.confidence,
            trade_id,
        )

        if self._cfg.simulation:
            logger.info("GoldTradingManager: SIMULATION — no order sent")
            return

        # Place bracket order
        await self._place_bracket_entry(signal, sizing.contracts, trade_id, bar_dt)

    # ── Position management ───────────────────────────────────────────────────

    async def _manage_open_position(self, df: pd.DataFrame, bar_dt: datetime) -> None:
        """Manage trailing stop and time stop for an open position."""
        if self._position is None or self._position.entry_price is None:
            return

        pos = self._position
        current_price = float(df["close"].iloc[-1])
        atr = float(df["atr"].iloc[-1]) if "atr" in df.columns else pos.signal.atr

        # ── Trailing stop ────────────────────────────────────────────────────
        if self._cfg.exit.trailing_stop_enabled:
            await self._update_trailing_stop(pos, current_price, atr)

        # ── Time stop ────────────────────────────────────────────────────────
        if self._cfg.exit.time_stop_enabled:
            await self._check_time_stop(df, bar_dt)

    async def _update_trailing_stop(
        self,
        pos: _OpenPosition,
        current_price: float,
        atr: float,
    ) -> None:
        """Adjust stop loss upward (long) or downward (short) as price moves favorably."""
        if pos.entry_price is None:
            return

        activation_threshold = self._cfg.exit.trailing_activation_r
        if pos.action == "BUY":
            reward_needed = (pos.signal.take_profit - pos.entry_price) * activation_threshold
            if current_price >= pos.entry_price + reward_needed:
                pos.trail_activated = True
            if pos.trail_activated:
                new_sl = current_price - atr * self._cfg.exit.trailing_atr_mult
                # Only move stop up
                if new_sl > pos.signal.stop_loss:
                    logger.info(
                        "GoldTradingManager: trailing stop moved {:.2f} → {:.2f}",
                        pos.signal.stop_loss,
                        new_sl,
                    )
                    pos.signal.stop_loss = new_sl
                    await self._modify_sl_order(pos, new_sl)
        else:  # SELL
            reward_needed = (pos.entry_price - pos.signal.take_profit) * activation_threshold
            if current_price <= pos.entry_price - reward_needed:
                pos.trail_activated = True
            if pos.trail_activated:
                new_sl = current_price + atr * self._cfg.exit.trailing_atr_mult
                # Only move stop down
                if new_sl < pos.signal.stop_loss:
                    logger.info(
                        "GoldTradingManager: trailing stop moved {:.2f} → {:.2f}",
                        pos.signal.stop_loss,
                        new_sl,
                    )
                    pos.signal.stop_loss = new_sl
                    await self._modify_sl_order(pos, new_sl)

        if pos.best_price is None:
            pos.best_price = current_price
        elif pos.action == "BUY" and current_price > pos.best_price:
            pos.best_price = current_price
        elif pos.action == "SELL" and current_price < pos.best_price:
            pos.best_price = current_price

    async def _check_time_stop(
        self,
        df: pd.DataFrame,
        bar_dt: datetime,
        force: bool = False,
    ) -> None:
        """Exit if the position has been open too long without progress."""
        if self._position is None:
            return
        pos = self._position
        if pos.entry_price is None:
            return

        bars_held = self._bar_counter - pos.entry_bar
        if not force and bars_held < self._cfg.exit.time_stop_bars:
            return

        reason = "FLATTEN_SESSION" if force else "TIME_STOP"
        current_price = float(df["close"].iloc[-1])
        logger.info(
            "GoldTradingManager: {} — bars_held={} price={:.2f}",
            reason,
            bars_held,
            current_price,
        )
        await self._flatten_position(current_price, reason)

    # ── Order placement helpers ───────────────────────────────────────────────

    async def _place_bracket_entry(
        self,
        signal: GoldSignal,
        contracts: int,
        trade_id: str,
        entry_time: datetime,
    ) -> None:
        """Submit a bracket (entry + SL + TP) order to IB."""
        if self._contract is None:
            logger.error("GoldTradingManager: no contract — cannot place order")
            return

        opposite = "SELL" if signal.action == "BUY" else "BUY"

        # Use market order for entry; SL and TP are limit/stop children
        entry_order = MarketOrder(signal.action, contracts)
        tp_order = LimitOrder(opposite, contracts, signal.take_profit)
        sl_order = StopOrder(opposite, contracts, signal.stop_loss)

        # Link bracket
        entry_order.transmit = False
        tp_order.parentId = entry_order.orderId   # Will be set after submit
        sl_order.parentId = entry_order.orderId
        tp_order.transmit = False
        sl_order.transmit = True   # Transmit on last child

        try:
            with self._lock.engage(f"gold_bracket_{trade_id}"):
                entry_trade = self._ib.placeOrder(self._contract, entry_order)
                # Attach correct parent ID
                tp_order.parentId = entry_trade.order.orderId
                sl_order.parentId = entry_trade.order.orderId
                tp_trade = self._ib.placeOrder(self._contract, tp_order)
                sl_trade = self._ib.placeOrder(self._contract, sl_order)

                self._active_trades[entry_trade.order.orderId] = entry_trade
                self._active_trades[tp_trade.order.orderId] = tp_trade
                self._active_trades[sl_trade.order.orderId] = sl_trade

                self._position = _OpenPosition(
                    trade_id=trade_id,
                    action=signal.action,
                    signal=signal,
                    contracts=contracts,
                    entry_bar=self._bar_counter,
                    entry_time=entry_time,
                )
                self._position.entry_order_id = entry_trade.order.orderId
                self._position.tp_order_id = tp_trade.order.orderId
                self._position.sl_order_id = sl_trade.order.orderId

                # Register fill callbacks
                entry_trade.fillEvent += lambda t, f: self._on_entry_fill(t, f)
                tp_trade.fillEvent += lambda t, f: self._on_tp_fill(t, f)
                sl_trade.fillEvent += lambda t, f: self._on_sl_fill(t, f)

            logger.info(
                "GoldTradingManager: bracket submitted — entry={} tp={} sl={}",
                entry_trade.order.orderId,
                tp_trade.order.orderId,
                sl_trade.order.orderId,
            )
        except Exception as exc:
            logger.opt(exception=True).error("GoldTradingManager: order placement failed: {}", exc)
            self._position = None

    async def _modify_sl_order(self, pos: _OpenPosition, new_sl: float) -> None:
        """Modify the SL stop price on an open order."""
        if pos.sl_order_id is None or self._cfg.simulation:
            return
        trade = self._active_trades.get(pos.sl_order_id)
        if trade is None:
            return
        try:
            trade.order.auxPrice = new_sl
            self._ib.placeOrder(self._contract, trade.order)
        except Exception as exc:
            logger.warning("GoldTradingManager: failed to modify SL: {}", exc)

    async def _flatten_position(self, current_price: float, reason: str) -> None:
        """Cancel all child orders and submit a market exit."""
        if self._position is None:
            return
        pos = self._position

        if not self._cfg.simulation:
            # Cancel TP and SL
            for oid in (pos.tp_order_id, pos.sl_order_id):
                if oid and oid in self._active_trades:
                    try:
                        self._ib.cancelOrder(self._active_trades[oid].order)
                    except Exception as exc:
                        logger.warning("GoldTradingManager: cancel order {} failed: {}", oid, exc)

            # Market exit
            opposite = "SELL" if pos.action == "BUY" else "BUY"
            flat_order = MarketOrder(opposite, pos.contracts)
            flat_trade = self._ib.placeOrder(self._contract, flat_order)
            self._active_trades[flat_order.orderId] = flat_trade

        # Record exit
        self._record_exit(pos, current_price, reason)
        self._position = None

    # ── Fill event handlers ───────────────────────────────────────────────────

    def _on_entry_fill(self, trade: Trade, fill) -> None:
        if self._position is None:
            return
        self._position.entry_price = fill.execution.price
        logger.info(
            "GoldTradingManager: ENTRY FILL — {} @ {:.2f} × {} [{}]",
            self._position.action,
            fill.execution.price,
            fill.execution.shares,
            self._position.trade_id,
        )

    def _on_tp_fill(self, trade: Trade, fill) -> None:
        if self._position is None:
            return
        logger.info(
            "GoldTradingManager: TP FILL @ {:.2f} [{}]",
            fill.execution.price,
            self._position.trade_id,
        )
        self._record_exit(self._position, fill.execution.price, "PROFIT_TARGET")
        self._position = None

    def _on_sl_fill(self, trade: Trade, fill) -> None:
        if self._position is None:
            return
        logger.info(
            "GoldTradingManager: SL FILL @ {:.2f} [{}]",
            fill.execution.price,
            self._position.trade_id,
        )
        self._strategy.notify_loss()   # Post-loss bar cooldown
        self._record_exit(self._position, fill.execution.price, "STOP_LOSS")
        self._position = None

    # ── Outcome recording ─────────────────────────────────────────────────────

    def _record_exit(
        self,
        pos: _OpenPosition,
        exit_price: float,
        reason: str,
    ) -> None:
        """Compute P&L, update daily state, journal the trade."""
        entry = pos.entry_price if pos.entry_price is not None else pos.signal.entry_ref_price

        if pos.action == "BUY":
            gross_pnl = (exit_price - entry) * self._spec.point_value * pos.contracts
        else:
            gross_pnl = (entry - exit_price) * self._spec.point_value * pos.contracts

        commission = self._spec.live_commission_per_side * 2 * pos.contracts if self._spec.live_commission_per_side else 0.0
        net_pnl = gross_pnl - commission

        # Update day state
        self._day_state.realized_pnl_today += net_pnl
        self._day_state.trades_today += 1

        is_win = net_pnl > 0
        if is_win:
            self._day_state.consecutive_loss_count = 0
        else:
            self._day_state.consecutive_loss_count += 1
            cooldown = self._risk_mgr.compute_cooldown_until(
                self._day_state.consecutive_loss_count
            )
            if cooldown is not None:
                self._day_state.cooldown_until = cooldown

        # Persist
        self._state_mgr.save(self._day_state)

        # Journal
        now_utc = datetime.now(timezone.utc)
        bars_held = self._bar_counter - pos.entry_bar
        features_last = (
            compute_indicators(self._build_1min_df() or pd.DataFrame(), self._cfg)
            if self._build_1min_df() is not None
            else pd.DataFrame()
        )
        adx = float(features_last["adx"].iloc[-1]) if "adx" in features_last.columns and len(features_last) > 0 else 0.0

        record = GoldTradeRecord(
            trade_id=pos.trade_id,
            symbol=self._cfg.symbol,
            action=pos.action,
            signal_type=pos.signal.signal_type.value,
            contracts=pos.contracts,
            entry_price=entry,
            stop_loss=pos.signal.stop_loss,
            take_profit=pos.signal.take_profit,
            exit_price=exit_price,
            realized_pnl=gross_pnl,
            commission=commission,
            net_pnl=net_pnl,
            entry_time=pos.entry_time.isoformat(),
            exit_time=now_utc.isoformat(),
            hold_bars=bars_held,
            exit_reason=reason,
            regime=pos.signal.regime.value,
            atr_at_entry=pos.signal.atr,
            adx_at_entry=adx,
        )
        self._journal.record(record)

        log_structured_event(
            "GOLD_EXIT",
            trade_id=pos.trade_id,
            symbol=self._cfg.symbol,
            action=pos.action,
            signal_type=pos.signal.signal_type.value,
            exit_price=exit_price,
            net_pnl=net_pnl,
            exit_reason=reason,
            pnl_today=self._day_state.realized_pnl_today,
        )

    # ── Session helpers ───────────────────────────────────────────────────────

    def _is_tradeable_session(self, bar_dt: datetime) -> bool:
        """Return False during maintenance window or outside configured hours."""
        try:
            local = bar_dt.astimezone(
                __import__("pytz").timezone("America/Chicago")
            )
        except Exception:
            return True   # No tz library — don't block trading

        t = local.time()

        from datetime import time as _time
        maint_start = _time(
            int(self._cfg.session.maintenance_start_ct.split(":")[0]),
            int(self._cfg.session.maintenance_start_ct.split(":")[1]),
        )
        maint_end = _time(
            int(self._cfg.session.maintenance_end_ct.split(":")[0]),
            int(self._cfg.session.maintenance_end_ct.split(":")[1]),
        )
        if maint_start <= t < maint_end:
            return False

        # Check news lockout windows (ET)
        try:
            et_local = bar_dt.astimezone(__import__("pytz").timezone("America/New_York"))
            et_time = et_local.time()
            for window in self._cfg.session.news_lockout_windows_et:
                if len(window) != 2:
                    continue
                ws_h, ws_m = int(window[0].split(":")[0]), int(window[0].split(":")[1])
                we_h, we_m = int(window[1].split(":")[0]), int(window[1].split(":")[1])
                ws = _time(ws_h, ws_m)
                we = _time(we_h, we_m)
                if ws <= et_time <= we:
                    return False
        except Exception:
            pass

        return True

    # ── Data helpers ──────────────────────────────────────────────────────────

    def _build_1min_df(self) -> Optional[pd.DataFrame]:
        """Aggregate raw 5-sec bars into completed 1-minute OHLCV bars."""
        if not self._raw_bars:
            return None

        rows = pd.DataFrame(self._raw_bars)
        rows["time"] = pd.to_datetime(rows["time"], utc=True)
        rows = rows.set_index("time").sort_index()

        # Resample to 1-min closed bars (label = bar open)
        ohlcv = rows["close"].resample("1min", closed="left", label="left").ohlc()
        ohlcv["volume"] = rows["volume"].resample("1min", closed="left", label="left").sum()
        # Replace resampled column names from ohlc()
        ohlcv.columns = ["open", "high", "low", "close", "volume"]

        # Drop the currently open (incomplete) bar — last row may be partial
        ohlcv = ohlcv.iloc[:-1]

        if len(ohlcv) == 0:
            return None

        ohlcv = ohlcv.dropna(subset=["close"])
        if len(ohlcv) > 0:
            self._last_processed_bar_time = ohlcv.index[-1]
        return ohlcv

    # ── Reconciliation ────────────────────────────────────────────────────────

    async def _reconcile_open_orders(self) -> None:
        """Check IB for open orders from a previous session and log them."""
        if self._ib is None:
            return
        try:
            open_trades = self._ib.openTrades()
            gold_orders = [
                t for t in open_trades
                if t.contract.symbol == self._cfg.symbol
            ]
            if gold_orders:
                logger.warning(
                    "GoldTradingManager: found {} open gold order(s) on IB — "
                    "review manually before trading. Order IDs: {}",
                    len(gold_orders),
                    [t.order.orderId for t in gold_orders],
                )
            else:
                logger.info("GoldTradingManager: reconciliation clean — no stale orders")
        except Exception as exc:
            logger.warning("GoldTradingManager: reconciliation error: {}", exc)

    # ── Maintenance guard ─────────────────────────────────────────────────────

    async def _maintenance_guard(self) -> None:
        """Flatten positions before maintenance window; check for contract roll."""
        bar_dt = datetime.now(timezone.utc)

        # Contract roll check (once per loop iteration)
        if self._contract is not None and self._roll_monitor.needs_roll(self._contract):
            logger.warning("GoldTradingManager: rolling contract to next front month")
            if self._position is not None:
                df = self._build_1min_df()
                price = float(df["close"].iloc[-1]) if df is not None and len(df) > 0 else 0.0
                await self._flatten_position(price, "FLATTEN_ROLL")
            if self._contract_factory is not None:
                self._contract_factory.invalidate_cache()
                try:
                    self._contract = await self._contract_factory.get_qualified_contract()
                    self._roll_monitor.reset()
                    logger.info("GoldTradingManager: contract rolled successfully")
                except Exception as exc:
                    logger.error("GoldTradingManager: contract roll failed: {}", exc)

        if self._position is None or self._position.entry_price is None:
            return
        if not self._is_tradeable_session(bar_dt):
            df = self._build_1min_df()
            if df is not None and len(df) > 0:
                current_price = float(df["close"].iloc[-1])
                await self._flatten_position(current_price, "FLATTEN_MAINTENANCE")

    # ── Shutdown ─────────────────────────────────────────────────────────────

    async def _shutdown(self) -> None:
        """Flatten positions, print summary, disconnect."""
        logger.info("GoldTradingManager: shutting down")

        if self._ib is not None and self._ib.isConnected():
            if self._bar_list is not None:
                try:
                    self._ib.cancelRealTimeBars(self._bar_list)
                except Exception:
                    pass

            try:
                self._ib.errorEvent -= self._on_ib_error
            except Exception:
                pass

            if self._position is not None:
                df = self._build_1min_df()
                if df is not None and len(df) > 0:
                    price = float(df["close"].iloc[-1])
                    await self._flatten_position(price, "FLATTEN_SHUTDOWN")

            self._journal.print_session_summary()
            self._state_mgr.save(self._day_state)

            self._ib.disconnect()
            logger.info("GoldTradingManager: disconnected from IB Gateway")
