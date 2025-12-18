"""
Telegram Notification Module for MyTrader
Sends trade execution alerts via Telegram Bot API
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional, Dict, Any
import aiohttp

from ..utils.logger import logger


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
                else:
                    error_text = await response.text()
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
            
        Returns:
            Formatted HTML message
        """
        if timestamp is None:
            timestamp = datetime.now()
        
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
            f"Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}"
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
        
        # Risk management levels
        if stop_loss is not None or take_profit is not None:
            lines.append("")
            lines.append("<b>Risk Management:</b>")
            if stop_loss is not None:
                lines.append(f"🛡️ Stop Loss: ${stop_loss:.2f}")
            if take_profit is not None:
                lines.append(f"🎯 Take Profit: ${take_profit:.2f}")
        
        return "\n".join(lines)
    
    @staticmethod
    def format_signal_alert(
        symbol: str,
        action: str,
        confidence: float,
        price: float,
        strategy: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
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
        
        lines = [
            "📊 <b>TRADING SIGNAL</b>",
            "",
            f"{emoji} Action: <b>{action.upper()}</b>",
            f"Symbol: <b>{symbol}</b>",
            f"Confidence: <b>{confidence_pct:.1f}%</b>",
            f"Price: ${price:.2f}",
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        ]
        
        if strategy:
            lines.append(f"Strategy: {strategy}")
        
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
