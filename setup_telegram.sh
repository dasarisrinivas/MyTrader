#!/bin/bash
# Quick Setup Script for Telegram Notifications
# This script helps you set up Telegram notifications for MyTrader

set -e

echo "================================================"
echo "MyTrader - Telegram Notifications Setup"
echo "================================================"
echo ""

# Check if aiohttp is installed
echo "Checking dependencies..."
if python3 -c "import aiohttp" 2>/dev/null; then
    echo "✅ aiohttp is installed"
else
    echo "❌ aiohttp not found"
    echo ""
    read -p "Install aiohttp now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing aiohttp..."
        pip3 install aiohttp>=3.9.0
        echo "✅ aiohttp installed"
    else
        echo "Please install aiohttp manually: pip3 install aiohttp>=3.9.0"
        exit 1
    fi
fi

echo ""
echo "================================================"
echo "Step 1: Create Telegram Bot"
echo "================================================"
echo ""
echo "1. Open Telegram and search for @BotFather"
echo "2. Send: /newbot"
echo "3. Follow the prompts to create your bot"
echo "4. Save the API token (looks like: 123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11)"
echo ""
read -p "Press Enter when you have your bot token..."

echo ""
read -p "Enter your Telegram Bot Token: " BOT_TOKEN

echo ""
echo "================================================"
echo "Step 2: Get Your Chat ID"
echo "================================================"
echo ""
echo "1. Find your bot in Telegram and start a chat"
echo "2. Send any message to your bot (e.g., 'Hello')"
echo "3. Open this URL in your browser:"
echo ""
echo "   https://api.telegram.org/bot${BOT_TOKEN}/getUpdates"
echo ""
echo "4. Look for \"chat\":{\"id\":123456789}"
echo "5. Copy the number after \"id\":"
echo ""
read -p "Press Enter when ready to continue..."

echo ""
read -p "Enter your Telegram Chat ID: " CHAT_ID

echo ""
echo "================================================"
echo "Step 3: Test Configuration"
echo "================================================"
echo ""

# Create test script
cat > /tmp/test_telegram.py << EOF
import asyncio
from mytrader.utils.telegram_notifier import TelegramNotifier

async def test():
    notifier = TelegramNotifier(
        bot_token="${BOT_TOKEN}",
        chat_id="${CHAT_ID}",
        enabled=True
    )
    
    print("Sending test message...")
    success = await notifier.send_message("🎉 <b>MyTrader</b> connected successfully!")
    
    if success:
        print("✅ Telegram notifications working!")
        print("")
        print("Check your Telegram for the test message.")
        return True
    else:
        print("❌ Failed to send message")
        print("Please check your bot token and chat ID")
        return False
    
    await notifier.close()

result = asyncio.run(test())
exit(0 if result else 1)
EOF

echo "Running test..."
if python3 /tmp/test_telegram.py; then
    echo ""
    echo "================================================"
    echo "Step 4: Save Configuration"
    echo "================================================"
    echo ""
    echo "Choose configuration method:"
    echo ""
    echo "1) Environment variables (.env file) - RECOMMENDED"
    echo "2) config.yaml file"
    echo "3) Skip (configure manually later)"
    echo ""
    read -p "Enter choice (1-3): " -n 1 -r CONFIG_METHOD
    echo ""
    echo ""
    
    case $CONFIG_METHOD in
        1)
            echo "Creating .env file..."
            if [ -f .env ]; then
                echo "# Telegram Configuration (added by setup script)" >> .env
            else
                echo "# MyTrader Environment Variables" > .env
                echo "" >> .env
                echo "# Telegram Configuration" >> .env
            fi
            echo "TELEGRAM_ENABLED=true" >> .env
            echo "TELEGRAM_BOT_TOKEN=${BOT_TOKEN}" >> .env
            echo "TELEGRAM_CHAT_ID=${CHAT_ID}" >> .env
            echo ""
            echo "✅ Configuration saved to .env"
            echo ""
            echo "⚠️  IMPORTANT: Add .env to .gitignore to keep your token secret!"
            if ! grep -q "^\.env$" .gitignore 2>/dev/null; then
                echo ".env" >> .gitignore
                echo "✅ Added .env to .gitignore"
            fi
            ;;
        2)
            echo "Updating config.yaml..."
            # Backup config
            cp config.yaml config.yaml.backup
            echo "✅ Backed up config.yaml to config.yaml.backup"
            
            # Update config
            sed -i.tmp "s/enabled: false/enabled: true/g" config.yaml
            sed -i.tmp "s/bot_token: \"\"/bot_token: \"${BOT_TOKEN}\"/g" config.yaml
            sed -i.tmp "s/chat_id: \"\"/chat_id: \"${CHAT_ID}\"/g" config.yaml
            rm config.yaml.tmp
            
            echo "✅ Updated config.yaml"
            echo ""
            echo "⚠️  WARNING: Your bot token is now in config.yaml"
            echo "   Do NOT commit this file to Git!"
            ;;
        3)
            echo "Skipping configuration..."
            echo ""
            echo "To configure manually, see: docs/TELEGRAM_SETUP.md"
            ;;
        *)
            echo "Invalid choice. Skipping configuration."
            ;;
    esac
    
    echo ""
    echo "================================================"
    echo "Setup Complete! 🎉"
    echo "================================================"
    echo ""
    echo "Your Telegram notifications are ready!"
    echo ""
    echo "Next steps:"
    echo "1. Start your trading bot: ./start_bot.sh"
    echo "2. Wait for a trade execution"
    echo "3. Check Telegram for alerts"
    echo ""
    echo "For more information:"
    echo "- Full setup guide: docs/TELEGRAM_SETUP.md"
    echo "- Implementation details: docs/TELEGRAM_INTEGRATION_SUMMARY.md"
    echo ""
else
    echo ""
    echo "❌ Test failed. Please check:"
    echo "1. Bot token is correct"
    echo "2. Chat ID is correct"
    echo "3. You've sent at least one message to your bot"
    echo ""
    echo "See docs/TELEGRAM_SETUP.md for troubleshooting"
    exit 1
fi

# Cleanup
rm -f /tmp/test_telegram.py
