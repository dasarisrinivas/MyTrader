# 📱 Telegram Notifications Setup Guide

Get instant notifications on your phone whenever your trading bot executes a trade!

---

## 📋 Overview

The MyTrader bot can send real-time Telegram notifications for:
- ✅ **Trade Executions** - BUY/SELL orders filled with price, quantity, P&L
- 📊 **Trading Signals** - Signal generation with confidence (optional)
- ⚠️ **Errors & Warnings** - Trading issues and alerts (optional)

**Notifications are sent asynchronously and NEVER block trading operations.**

---

## 🚀 Quick Setup (5 minutes)

### Step 1: Create a Telegram Bot

1. Open Telegram and search for `@BotFather`
2. Start a chat with BotFather
3. Send the command: `/newbot`
4. Follow the prompts:
   - **Bot name:** Choose a display name (e.g., "MyTrader Bot")
   - **Bot username:** Choose a unique username ending in "bot" (e.g., "mytrader_alerts_bot")
5. **Save the API token** - BotFather will give you a token like:
   ```
   1234567890:ABCdefGHIjklMNOpqrsTUVwxyz1234567890
   ```

### Step 2: Get Your Chat ID

**Option A: Using Your Bot (Easiest)**

1. Find your new bot in Telegram (search for the username you created)
2. Start a chat and send any message (e.g., "Hello")
3. Open this URL in your browser (replace `YOUR_BOT_TOKEN`):
   ```
   https://api.telegram.org/botYOUR_BOT_TOKEN/getUpdates
   ```
4. Look for `"chat":{"id":123456789}` in the response
5. **Save your chat ID** (the number after `"id":`)

**Option B: Using @userinfobot**

1. Search for `@userinfobot` on Telegram
2. Start a chat and it will reply with your user ID
3. Use this ID as your `chat_id`

**Option C: For Channels/Groups**

1. Add your bot to the channel/group as administrator
2. Send a message in the channel/group
3. Use the getUpdates URL from Option A
4. Look for the chat ID (will be negative for groups/channels)

### Step 3: Configure MyTrader

**Using Environment Variables (Recommended):**

Create or edit `.env` file in your project root:

```bash
# Telegram Configuration
TELEGRAM_ENABLED=true
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz1234567890
TELEGRAM_CHAT_ID=123456789
```

**OR Using config.yaml:**

Edit `config.yaml`:

```yaml
telegram:
  enabled: true
  bot_token: "1234567890:ABCdefGHIjklMNOpqrsTUVwxyz1234567890"
  chat_id: "123456789"
  notify_on_trade: true     # Alert on trade execution (recommended)
  notify_on_signal: false   # Alert on signal generation (can be noisy)
  notify_on_error: true     # Alert on errors/warnings
```

### Step 4: Test the Setup

**Quick Test Script:**

```python
import asyncio
from mytrader.utils.telegram_notifier import TelegramNotifier

async def test_telegram():
    notifier = TelegramNotifier(
        bot_token="YOUR_BOT_TOKEN",
        chat_id="YOUR_CHAT_ID",
        enabled=True
    )
    
    # Send test message
    success = await notifier.send_message("🎉 MyTrader bot connected successfully!")
    
    if success:
        print("✅ Telegram notifications working!")
    else:
        print("❌ Failed to send message - check your credentials")
    
    await notifier.close()

asyncio.run(test_telegram())
```

**Test with your bot:**

```bash
# Start your trading bot
./start_bot.sh

# OR
python run_bot.py
```

You should see:
```
✅ Telegram notifications initialized
```

---

## 📱 Notification Examples

### Trade Execution Alert

```
⚡ TRADE EXECUTED ⚡

🟢 BOUGHT
Symbol: ES
Quantity: 2 contracts
Price: $5850.50
Time: 2025-11-20 14:30:15 UTC
Order ID: #12345
📈 Position: +2
Commission: $4.80

Risk Management:
🛡️ Stop Loss: $5840.50
🎯 Take Profit: $5870.50
```

### Trade Exit with P&L

```
⚡ TRADE EXECUTED ⚡

🔴 SOLD
Symbol: ES
Quantity: 2 contracts
Price: $5865.25
Time: 2025-11-20 14:45:30 UTC
Order ID: #12346
⚖️ Position: 0
Commission: $4.80
💰 Realized P&L: +$735.00
```

### Error Alert (Optional)

```
⚠️ ALERT

Connection Error
Failed to connect to IB Gateway: Timeout after 30s
Time: 2025-11-20 14:50:00 UTC

Details:
Attempting automatic reconnection...
```

---

## 🔧 Advanced Configuration

### Notification Preferences

```yaml
telegram:
  enabled: true
  bot_token: "your-token"
  chat_id: "your-chat-id"
  
  # Customize what alerts you receive
  notify_on_trade: true      # ✅ Recommended: Get trade execution alerts
  notify_on_signal: false    # ⚠️ Can be noisy: Signal generation alerts
  notify_on_error: true      # ✅ Recommended: Error/warning alerts
```

### Environment Variables (12-Factor App)

All settings can be controlled via environment variables:

```bash
export TELEGRAM_ENABLED=true
export TELEGRAM_BOT_TOKEN="your-token"
export TELEGRAM_CHAT_ID="your-chat-id"
```

This is useful for:
- **Docker deployments**
- **CI/CD pipelines**
- **Multiple environments** (dev/staging/prod)

### Multiple Chat IDs (Advanced)

To send notifications to multiple chats/groups:

1. Create separate TelegramNotifier instances
2. Or modify the code to accept a list of chat IDs

Example (manual implementation):

```python
# In your code
telegram_personal = TelegramNotifier(bot_token=token, chat_id="123456")
telegram_team = TelegramNotifier(bot_token=token, chat_id="-987654321")

# Send to both
telegram_personal.send_trade_alert_background(...)
telegram_team.send_trade_alert_background(...)
```

---

## 🛡️ Security Best Practices

### 1. Keep Your Bot Token Secret
- ❌ **Never commit tokens to Git**
- ✅ Use environment variables or `.env` files
- ✅ Add `.env` to `.gitignore`

### 2. Restrict Bot Access
- Only share your bot link with trusted people
- Telegram bots can only message users who start them first
- For groups: Make bot an admin with "Delete Messages" permission only

### 3. Chat ID Privacy
- Your chat ID is not sensitive (it's just a number)
- Bot token is sensitive - treat like a password

### 4. Disable Bot if Compromised
1. Talk to @BotFather
2. Send `/revoke` command
3. Select your bot
4. Create a new bot and update your config

---

## 🐛 Troubleshooting

### Issue: "Telegram notifications disabled: Invalid bot token"

**Cause:** Bot token is empty, missing, or incorrect

**Solution:**
1. Check your token has no extra spaces
2. Verify format: `1234567890:ABCdefGHI...`
3. Get a fresh token from @BotFather if needed

### Issue: "Failed to send Telegram message: 401 Unauthorized"

**Cause:** Invalid bot token

**Solution:**
- Verify token is correct
- Generate new token from @BotFather: `/mybots` → select bot → API Token

### Issue: "Failed to send Telegram message: 400 Bad Request: chat not found"

**Cause:** Invalid chat ID or bot hasn't been started

**Solution:**
1. Make sure you've sent at least one message to your bot
2. Verify chat ID is correct (should be a number)
3. For groups: Add bot first, then send a message

### Issue: Messages not arriving

**Check:**
1. Bot status: Make sure bot is not blocked
2. Network: Telegram API might be blocked by firewall
3. Logs: Check `logs/live_trading.log` for error messages

**Debug:**
```python
# Test connection
import asyncio
from mytrader.utils.telegram_notifier import TelegramNotifier

async def test():
    notifier = TelegramNotifier("YOUR_TOKEN", "YOUR_CHAT_ID", True)
    result = await notifier.send_message("Test message")
    print(f"Success: {result}")
    await notifier.close()

asyncio.run(test())
```

### Issue: "Telegram message timeout (10s exceeded)"

**Cause:** Network latency or Telegram API slow

**Solution:**
- This is non-blocking - trading continues normally
- Check internet connection
- Telegram API might be experiencing issues

### Issue: Not seeing trade alerts

**Check config:**
```yaml
telegram:
  enabled: true              # Must be true
  notify_on_trade: true      # Must be true for trade alerts
```

**Check logs:**
```bash
tail -f logs/live_trading.log | grep -i telegram
```

Look for:
- `✅ Telegram notifications initialized`
- `✅ Telegram message sent successfully`
- `❌ Failed to send Telegram notification`

---

## 📚 API Reference

### TelegramNotifier Class

```python
from mytrader.utils.telegram_notifier import TelegramNotifier

# Initialize
notifier = TelegramNotifier(
    bot_token="your-token",
    chat_id="your-chat-id",
    enabled=True
)

# Send trade alert (async)
await notifier.send_trade_alert(
    symbol="ES",
    side="BUY",
    quantity=2,
    fill_price=5850.50,
    timestamp=datetime.now(),
    current_position=2,
    order_id=12345,
    commission=4.80,
    stop_loss=5840.50,
    take_profit=5870.50
)

# Send trade alert (background - non-blocking)
notifier.send_trade_alert_background(
    symbol="ES",
    side="SELL",
    quantity=2,
    fill_price=5865.25,
    realized_pnl=735.00
)

# Send custom message
await notifier.send_message("<b>Custom HTML message</b>")

# Send in background (fire-and-forget)
notifier.send_message_background("Non-blocking message")

# Close connection
await notifier.close()
```

### Message Formatting

Messages support HTML formatting:
- `<b>bold</b>` - **bold**
- `<i>italic</i>` - *italic*
- `<code>code</code>` - `code`
- `<pre>preformatted</pre>` - preformatted text

Example:
```python
message = """
<b>⚡ TRADE ALERT ⚡</b>

Symbol: <code>ES</code>
Action: <b>BUY</b>
Price: <b>$5850.50</b>
"""
await notifier.send_message(message)
```

---

## 🎯 Integration Points

Telegram notifications are automatically sent from:

1. **`ib_executor.py`** - When orders are filled
   - Triggered in `_on_execution()` callback
   - Includes all execution details

2. **`live_trading_manager.py`** - Order placement
   - Initialization of Telegram notifier
   - Passed to TradeExecutor

3. **`main.py`** - Standalone bot mode
   - Initializes Telegram on startup

4. **`dashboard_api.py`** - Dashboard trading
   - Initializes Telegram for dashboard-initiated trades

**All notifications are fire-and-forget (non-blocking) to ensure trading performance.**

---

## 🔒 Privacy & Data

### What data is sent?
- Trade execution details (symbol, price, quantity, P&L)
- Trading signals (if enabled)
- Error messages (if enabled)

### What is NOT sent?
- API keys or passwords
- Account balances
- Personal information
- Historical data

### Who can see the messages?
- Only you (if using personal chat)
- Only group members (if using group chat)
- Telegram employees (encrypted in transit, but Telegram can technically access)

### Recommendations:
- Use a private chat for sensitive trading data
- Don't share bot token with anyone
- Consider disabling signal alerts in production (noisy)

---

## 📞 Support

**Issues with this integration:**
- Check logs: `logs/live_trading.log`
- Test with the debug script above
- Review this documentation

**Issues with Telegram:**
- Telegram API Docs: https://core.telegram.org/bots/api
- @BotSupport on Telegram

**Issues with MyTrader bot:**
- Check project documentation in `docs/`
- Review `README.md`

---

## 🎉 You're All Set!

Your trading bot will now send you real-time alerts on every trade execution!

**Next Steps:**
1. ✅ Start your trading bot
2. ✅ Wait for first trade execution
3. ✅ Check your Telegram for alerts
4. ✅ Adjust `notify_on_signal` and `notify_on_error` as needed

**Happy Trading! 🚀**
