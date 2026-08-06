"""
Telegram Notification Module for Shree
Sends trade execution alerts via Telegram Bot API
"""
from __future__ import annotations

import asyncio
import html
from datetime import datetime
from typing import Optional, Dict, Any, List
from zoneinfo import ZoneInfo
import aiohttp

from ..utils.logger import logger

# Import CST utilities for timezone-aware timestamps
try:
    from ..utils.timezone_utils import now_cst, format_cst, CST
except ImportError:
    CST = ZoneInfo("America/Chicago")
    def now_cst():
        return datetime.now(CST)
    def format_cst(dt):
        return dt.strftime("%Y-%m-%d %H:%M:%S CST")


class TelegramNotifier:
    """
    Async Telegram notifier for trade execution alerts.
    Ensures notifications never block trading operations.
    """
    
    def __init__(self, bot_token: str, chat_id: str, enabled: bool = True):
        """
        Initialize Telegram notifier.
        
        Args:
            bot_token: Telegram Bot API token
            chat_id: Telegram chat/channel ID to send messages to
            enabled: Whether notifications are enabled
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled
        self.api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self._session: Optional[aiohttp.ClientSession] = None
        
        if self.enabled:
            if not bot_token or bot_token == "your-bot-token-here":
                logger.warning("⚠️  Telegram notifications disabled: Invalid bot token")
                self.enabled = False
            elif not chat_id or chat_id == "your-chat-id-here":
                logger.warning("⚠️  Telegram notifications disabled: Invalid chat ID")
                self.enabled = False
            else:
                logger.info("✅ Telegram notifications enabled")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=10)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        Send a message via Telegram Bot API.
        
        Args:
            text: Message text (supports HTML formatting)
            parse_mode: Message formatting mode ("HTML" or "Markdown")
            
        Returns:
            True if message sent successfully, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            session = await self._get_session()
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True
            }
            
            async with session.post(self.api_url, json=payload) as response:
                if response.status == 200:
                    logger.debug("✅ Telegram message sent successfully")
                    return True
                error_text = await response.text()
                # AUG 6 2026 — plain-text fallback on an HTML parse failure.
                # Telegram rejects the whole message with 400 "can't parse
                # entities" when dynamic text contains a bare '<' (gate and
                # reason strings routinely do, e.g. "flow +0 < adaptive req
                # ±25"). On 2026-08-06 this silently dropped BOTH alerts for
                # the first live fill of the observation period:
                #   400 ... Unsupported start tag "=" at byte offset 361
                #   400 ... Unsupported start tag "=" at byte offset 132
                # Escaping every call site is fragile — any new unescaped field
                # reintroduces the bug. Re-sending once without parse_mode
                # guarantees the operator still receives the alert (formatting
                # tags appear literally, which is strictly better than silence).
                if (response.status == 400 and parse_mode
                        and "parse entities" in error_text):
                    logger.warning(
                        "⚠️  Telegram HTML parse failed — resending as plain "
                        "text: {}", error_text[:160])
                    plain = dict(payload)
                    plain.pop("parse_mode", None)
                    async with session.post(self.api_url, json=plain) as r2:
                        if r2.status == 200:
                            logger.info("✅ Telegram message delivered "
                                        "(plain-text fallback)")
                            return True
                        logger.warning(
                            "⚠️  Telegram plain-text fallback also failed "
                            f"({r2.status}): {(await r2.text())[:160]}")
                        return False
                logger.warning(f"⚠️  Telegram API error ({response.status}): {error_text}")
                return False


        except asyncio.TimeoutError:
            logger.warning("⚠️  Telegram message timeout (10s exceeded)")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to send Telegram message: {e}")
            return False
    
    def send_message_background(self, text: str, parse_mode: str = "HTML"):
        """
        Send message in background without blocking.
        Fire-and-forget - errors are logged but don't raise exceptions.
        
        Args:
            text: Message text
            parse_mode: Message formatting mode
        """
        if not self.enabled:
            return
        
        # Create task in background - don't await
        # Use ensure_future for compatibility with already-running event loops
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Loop is already running, schedule coroutine
                asyncio.ensure_future(self._send_message_safe(text, parse_mode))
            else:
                # No running loop, create task
                asyncio.create_task(self._send_message_safe(text, parse_mode))
        except RuntimeError:
            # No event loop at all - create a new one in a thread
            import threading
            threading.Thread(
                target=self._send_in_new_loop,
                args=(text, parse_mode),
                daemon=True
            ).start()
    
    def _send_in_new_loop(self, text: str, parse_mode: str):
        """Run async send in a new event loop (for threading)."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._send_message_safe(text, parse_mode))
        finally:
            loop.close()
    
    async def _send_message_safe(self, text: str, parse_mode: str):
        """Wrapper that catches all exceptions to prevent background task crashes."""
        try:
            await self.send_message(text, parse_mode)
        except Exception as e:
            logger.error(f"❌ Background Telegram send failed: {e}")
    
    @staticmethod
    def format_trade_alert(
        symbol: str,
        side: str,
        quantity: int,
        fill_price: float,
        timestamp: Optional[datetime] = None,
        current_position: Optional[int] = None,
        order_id: Optional[int] = None,
        commission: Optional[float] = None,
        realized_pnl: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        entry_price: Optional[float] = None,
        exit_price: Optional[float] = None,
        points: Optional[float] = None,
        gross_pnl: Optional[float] = None,
        net_pnl: Optional[float] = None,
        risk_reward: Optional[float] = None,
        protection_note: Optional[str] = None,
        decision_reasoning: Optional[List[str]] = None,
        market_trend: Optional[str] = None,
        volatility_regime: Optional[str] = None,
        session: Optional[str] = None,
        account_value: Optional[float] = None,  # FEB 5 2026: Add account value
    ) -> str:
        """
        Format a trade execution alert message.
        
        Args:
            symbol: Trading symbol (e.g., "ES", "SPY")
            side: BUY or SELL
            quantity: Number of contracts/shares
            fill_price: Execution price
            timestamp: Execution timestamp
            current_position: Position after trade
            order_id: Order ID from broker
            commission: Commission paid
            realized_pnl: Realized P&L if closing position
            stop_loss: Stop loss price
            take_profit: Take profit price
            entry_price: Entry price for the trade
            exit_price: Exit price (if applicable)
            points: Points gained/lost on the close
            gross_pnl: Gross profit before commissions
            net_pnl: Net profit after commissions
            risk_reward: Reported risk/reward ratio
            protection_note: Risk management note
            decision_reasoning: List of reasons for the trade decision (score breakdown)
            market_trend: Current market trend (UPTREND/DOWNTREND/RANGE)
            volatility_regime: Current volatility regime (HIGH/MEDIUM/LOW)
            session: Current trading session (RTH/EVENING/OVERNIGHT)
            
        Returns:
            Formatted HTML message
        """
        # Use CST timestamp
        if timestamp is None:
            timestamp = now_cst()
        else:
            # Convert to CST if not already
            try:
                if timestamp.tzinfo is None:
                    # Assume UTC and convert to CST
                    timestamp = timestamp.replace(tzinfo=ZoneInfo("UTC")).astimezone(CST)
                else:
                    timestamp = timestamp.astimezone(CST)
            except Exception:
                # Fallback to current CST time
                timestamp = now_cst()
        
        # Emoji and color based on side
        if side.upper() == "BUY":
            emoji = "🟢"
            action = "BOUGHT"
        elif side.upper() == "SELL":
            emoji = "🔴"
            action = "SOLD"
        else:
            emoji = "⚪"
            action = side.upper()
        
        # Build message
        lines = [
            "⚡ <b>TRADE EXECUTED</b> ⚡",
            "",
            f"{emoji} <b>{action}</b>",
            f"Symbol: <b>{symbol}</b>",
            f"Quantity: <b>{quantity}</b> contracts",
            f"Price: <b>${fill_price:.2f}</b>",
            f"Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S CST')}"
        ]

        if entry_price is not None:
            lines.append(f"Entry: ${entry_price:.2f}")
        if exit_price is not None:
            lines.append(f"Exit: ${exit_price:.2f}")
        
        # Optional fields
        if order_id is not None:
            lines.append(f"Order ID: #{order_id}")
        
        if current_position is not None:
            position_emoji = "📈" if current_position > 0 else "📉" if current_position < 0 else "⚖️"
            lines.append(f"{position_emoji} Position: <b>{current_position:+d}</b>")
        
        if commission is not None:
            lines.append(f"Commission: ${commission:.2f}")
        
        if points is not None:
            lines.append(f"Points: {points:+.2f}")

        if gross_pnl is not None:
            pnl_emoji = "💰" if gross_pnl >= 0 else "📉"
            lines.append(f"{pnl_emoji} Gross P&L: <b>${gross_pnl:+.2f}</b>")

        if net_pnl is not None:
            pnl_emoji = "💰" if net_pnl >= 0 else "📉"
            lines.append(f"{pnl_emoji} Net P&L: <b>${net_pnl:+.2f}</b>")
        elif realized_pnl is not None:
            pnl_emoji = "💰" if realized_pnl >= 0 else "📉"
            lines.append(f"{pnl_emoji} Realized P&L: <b>${realized_pnl:+.2f}</b>")

        if risk_reward is not None:
            lines.append(f"R:R Ratio: {risk_reward:.2f}")
        
        # FEB 5 2026: Add account value
        if account_value is not None:
            lines.append("")
            lines.append(f"💼 <b>Account Value: ${account_value:,.2f}</b>")
        
        # Market context section
        if market_trend or volatility_regime or session:
            lines.append("")
            lines.append("<b>📊 Market Context:</b>")
            if session:
                session_emoji = "☀️" if session == "RTH" else "🌙" if session == "EVENING" else "🌃"
                lines.append(f"{session_emoji} Session: {session}")
            if market_trend:
                trend_emoji = "📈" if market_trend == "UPTREND" else "📉" if market_trend == "DOWNTREND" else "➡️"
                lines.append(f"{trend_emoji} Trend: {market_trend}")
            if volatility_regime:
                vol_emoji = "🔥" if volatility_regime == "HIGH" else "⚡" if volatility_regime == "MEDIUM" else "😴"
                lines.append(f"{vol_emoji} Volatility: {volatility_regime}")
        
        # Decision reasoning section - WHY this trade was taken
        if decision_reasoning and len(decision_reasoning) > 0:
            lines.append("")
            lines.append("<b>🧠 Decision Factors:</b>")
            for reason in decision_reasoning[:5]:  # Limit to 5 reasons to keep message concise
                # Escape HTML entities to prevent parsing errors (e.g., RSI<48 becomes RSI&lt;48)
                safe_reason = html.escape(str(reason))
                lines.append(f"  • {safe_reason}")
        
        # Risk management levels
        if stop_loss is not None or take_profit is not None:
            lines.append("")
            lines.append("<b>Risk Management:</b>")
            if stop_loss is not None:
                lines.append(f"🛡️ Stop Loss: ${stop_loss:.2f}")
            if take_profit is not None:
                lines.append(f"🎯 Take Profit: ${take_profit:.2f}")
            if protection_note:
                safe_note = html.escape(str(protection_note))
                lines.append(f"ℹ️ {safe_note}")
        elif protection_note:
            lines.append("")
            lines.append("<b>Risk Management:</b>")
            safe_note = html.escape(str(protection_note))
            lines.append(f"⚠️ {safe_note}")
        
        return "\n".join(lines)
    
    @staticmethod
    def format_signal_alert(
        symbol: str,
        action: str,
        confidence: float,
        price: float,
        strategy: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        decision_reasoning: Optional[List[str]] = None,
        market_trend: Optional[str] = None,
        volatility_regime: Optional[str] = None,
        session: Optional[str] = None,
    ) -> str:
        """
        Format a trading signal alert (optional - for signal generation).
        
        Args:
            symbol: Trading symbol
            action: BUY, SELL, or HOLD
            confidence: Signal confidence (0-1)
            price: Current market price
            strategy: Strategy name
            metadata: Additional signal metadata
            decision_reasoning: List of reasons for the trade decision
            market_trend: Current market trend (UPTREND/DOWNTREND/RANGE)
            volatility_regime: Current volatility regime (HIGH/MEDIUM/LOW)
            session: Current trading session (RTH/EVENING/OVERNIGHT)
            
        Returns:
            Formatted HTML message
        """
        if action.upper() == "BUY":
            emoji = "🟢"
        elif action.upper() == "SELL":
            emoji = "🔴"
        else:
            emoji = "⚪"
        
        confidence_pct = confidence * 100
        
        # Use CST timestamp
        timestamp = now_cst()
        
        lines = [
            "📊 <b>TRADING SIGNAL</b>",
            "",
            f"{emoji} Action: <b>{action.upper()}</b>",
            f"Symbol: <b>{symbol}</b>",
            f"Confidence: <b>{confidence_pct:.1f}%</b>",
            f"Price: ${price:.2f}",
            f"Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S CST')}"
        ]
        
        if strategy:
            lines.append(f"Strategy: {strategy}")
        
        # Market context section
        if market_trend or volatility_regime or session:
            lines.append("")
            lines.append("<b>📊 Market Context:</b>")
            if session:
                session_emoji = "☀️" if session == "RTH" else "🌙" if session == "EVENING" else "🌃"
                lines.append(f"{session_emoji} Session: {session}")
            if market_trend:
                trend_emoji = "📈" if market_trend == "UPTREND" else "📉" if market_trend == "DOWNTREND" else "➡️"
                lines.append(f"{trend_emoji} Trend: {market_trend}")
            if volatility_regime:
                vol_emoji = "🔥" if volatility_regime == "HIGH" else "⚡" if volatility_regime == "MEDIUM" else "😴"
                lines.append(f"{vol_emoji} Volatility: {volatility_regime}")
        
        # Decision reasoning section
        if decision_reasoning and len(decision_reasoning) > 0:
            lines.append("")
            lines.append("<b>🧠 Decision Factors:</b>")
            for reason in decision_reasoning[:5]:  # Limit to 5 reasons
                lines.append(f"  • {reason}")
        
        if metadata:
            lines.append("")
            lines.append("<b>Context:</b>")
            for key, value in metadata.items():
                if isinstance(value, (int, float)):
                    if isinstance(value, float):
                        lines.append(f"  • {key}: {value:.2f}")
                    else:
                        lines.append(f"  • {key}: {value}")
                else:
                    lines.append(f"  • {key}: {value}")
        
        return "\n".join(lines)
    
    @staticmethod
    def format_error_alert(
        error_type: str,
        message: str,
        details: Optional[str] = None
    ) -> str:
        """
        Format an error/warning alert.
        
        Args:
            error_type: Type of error
            message: Error message
            details: Additional details
            
        Returns:
            Formatted HTML message
        """
        lines = [
            "⚠️ <b>ALERT</b>",
            "",
            f"<b>{error_type}</b>",
            f"{message}",
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        ]
        
        if details:
            lines.append("")
            lines.append("<b>Details:</b>")
            lines.append(details)
        
        return "\n".join(lines)
    
    async def send_trade_alert(
        self,
        symbol: str,
        side: str,
        quantity: int,
        fill_price: float,
        **kwargs
    ) -> bool:
        """
        Convenience method to send a formatted trade alert.
        
        Args:
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Number of contracts
            fill_price: Execution price
            **kwargs: Additional parameters for format_trade_alert
            
        Returns:
            True if sent successfully
        """
        message = self.format_trade_alert(
            symbol=symbol,
            side=side,
            quantity=quantity,
            fill_price=fill_price,
            **kwargs
        )
        return await self.send_message(message)
    
    def send_trade_alert_background(
        self,
        symbol: str,
        side: str,
        quantity: int,
        fill_price: float,
        **kwargs
    ):
        """Send trade alert in background without blocking."""
        message = self.format_trade_alert(
            symbol=symbol,
            side=side,
            quantity=quantity,
            fill_price=fill_price,
            **kwargs
        )
        self.send_message_background(message)

    @staticmethod
    def format_position_pnl_alert(
        symbol: str,
        quantity: int,
        entry_price: float,
        current_price: float,
        pnl_per_contract: float,
        total_pnl: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> str:
        """Format a periodic open-position P&L update alert."""
        direction = "LONG" if quantity > 0 else "SHORT"
        contracts = abs(quantity)
        pnl_emoji = "💰" if total_pnl >= 0 else "🔻"
        dir_emoji = "📈" if quantity > 0 else "📉"
        timestamp = now_cst()

        lines = [
            f"📊 <b>Position P&amp;L Update</b>",
            "",
            f"{dir_emoji} <b>{direction}</b> {contracts} × {symbol} @ ${entry_price:.2f}",
            f"Current: <b>${current_price:.2f}</b>",
            f"{pnl_emoji} P&amp;L/ct: <b>${pnl_per_contract:+.2f}</b>  |  Total: <b>${total_pnl:+.2f}</b>",
        ]

        if stop_loss is not None:
            lines.append(f"🛡️ Stop: ${stop_loss:.2f}")
        if take_profit is not None:
            lines.append(f"🎯 Target: ${take_profit:.2f}")

        lines.append(f"🕐 {timestamp.strftime('%H:%M:%S CST')}")
        return "\n".join(lines)

    def send_position_pnl_background(
        self,
        symbol: str,
        quantity: int,
        entry_price: float,
        current_price: float,
        pnl_per_contract: float,
        total_pnl: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ):
        """Send periodic position P&L alert in background without blocking."""
        message = self.format_position_pnl_alert(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            current_price=current_price,
            pnl_per_contract=pnl_per_contract,
            total_pnl=total_pnl,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        self.send_message_background(message)
