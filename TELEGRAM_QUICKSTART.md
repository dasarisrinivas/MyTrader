# 📱 Telegram Notifications - Quick Reference

## What is this?

Get instant Telegram alerts on your phone every time your trading bot executes a BUY or SELL trade!

## Features

✅ **Real-time trade alerts** with price, quantity, P&L  
✅ **Non-blocking** - never slows down trading  
✅ **Safe** - errors never crash the bot  
✅ **Easy setup** - 5 minutes with automated script  
✅ **Secure** - environment variable support  

## Quick Setup

### Option 1: Automated Setup (Recommended)

```bash
./setup_telegram.sh
```

Follow the interactive prompts to:
1. Install dependencies
2. Create your Telegram bot
3. Get your chat ID
4. Test the connection
5. Save configuration

### Option 2: Manual Setup

1. **Install dependency:**
   ```bash
   pip install aiohttp>=3.9.0
   ```

2. **Create Telegram Bot:**
   - Open Telegram, search for `@BotFather`
   - Send: `/newbot`
   - Save the API token

3. **Get Chat ID:**
   - Send a message to your bot
   - Visit: `https://api.telegram.org/botYOUR_TOKEN/getUpdates`
   - Copy your chat ID from the response

4. **Configure:**
   ```bash
   export TELEGRAM_ENABLED=true
   export TELEGRAM_BOT_TOKEN="your-token-here"
   export TELEGRAM_CHAT_ID="your-chat-id-here"
   ```

5. **Start trading:**
   ```bash
   ./start_bot.sh
   ```

## Configuration

### Environment Variables (Recommended)

```bash
export TELEGRAM_ENABLED=true
export TELEGRAM_BOT_TOKEN="1234567890:ABCdefGHI..."
export TELEGRAM_CHAT_ID="123456789"
```

### config.yaml

```yaml
telegram:
  enabled: true
  bot_token: "1234567890:ABCdefGHI..."
  chat_id: "123456789"
  notify_on_trade: true     # Trade execution alerts
  notify_on_signal: false   # Signal generation (noisy)
  notify_on_error: true     # Error/warning alerts
```

## Example Alert

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

## Documentation

- **Full Setup Guide:** [docs/TELEGRAM_SETUP.md](docs/TELEGRAM_SETUP.md)
- **Implementation Details:** [docs/TELEGRAM_INTEGRATION_SUMMARY.md](docs/TELEGRAM_INTEGRATION_SUMMARY.md)

## Troubleshooting

**"No module named 'aiohttp'"**
```bash
pip install aiohttp>=3.9.0
```

**"Invalid bot token"**
- Check token has no spaces
- Get new token from @BotFather if needed

**"Chat not found"**
- Send at least one message to your bot first
- Verify chat ID is a number (not username)

**Messages not arriving**
- Check logs: `tail -f logs/live_trading.log | grep -i telegram`
- Verify bot is not blocked
- Test with: `python3 -m mytrader.utils.telegram_notifier`

See [docs/TELEGRAM_SETUP.md](docs/TELEGRAM_SETUP.md) for complete troubleshooting guide.

## Security

- ✅ Keep your bot token secret (like a password)
- ✅ Use environment variables (don't commit to Git)
- ✅ Bot can only message users who start it first
- ✅ Revoke token at @BotFather if compromised

## Performance

- **Zero impact** on trading performance
- Notifications sent asynchronously (non-blocking)
- Failed notifications don't crash the bot
- 10-second timeout on all API calls

## Support

For issues or questions:
1. Check [docs/TELEGRAM_SETUP.md](docs/TELEGRAM_SETUP.md)
2. Review logs: `logs/live_trading.log`
3. Test configuration with setup script

---

**Happy Trading! 🚀**
