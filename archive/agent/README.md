# Autonomous Trading Analyst Agent

An advisory-only agent that monitors the ShreeBot live MES futures trading system, detects anomalies (inactivity, losses), and autonomously initiates diagnostic reasoning sessions with GitHub Copilot.

## ⚠️ ADVISORY ONLY

This agent **never** modifies trading logic, parameters, or live configuration. All output is diagnostic reports and alerts for human review.

## Architecture

```
agent/
├── agent.py              # Main event loop with signal handling & graceful shutdown
├── anomaly_detector.py   # Detects NO_TRADE, TRADE_LOSS, DAILY_LOSS, CONSEC_LOSS
├── context_collector.py  # Gathers trades, positions, config, logs, regime info
├── copilot_interface.py  # Subprocess Copilot CLI interaction (iterative reasoning)
├── prompt_templates.py   # Structured diagnostic & loss-analysis prompts
├── report_builder.py     # JSON reports, Markdown summaries, severity classification
├── notifier.py           # Telegram alert delivery
├── config.json           # All configurable thresholds
├── logs/                 # Agent logs + Copilot session transcripts
└── reports/              # Generated JSON & Markdown reports
```

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                    MAIN LOOP (60s poll)                   │
│                                                           │
│  1. Check trading window (Sun 6PM ET – Fri 5PM ET)       │
│  2. Run anomaly detection (4 checks)                      │
│  3. For each anomaly:                                     │
│     a. Collect full system context                        │
│     b. Build structured diagnostic prompt                 │
│     c. Run iterative Copilot reasoning (up to 5 rounds)  │
│     d. Generate JSON + Markdown reports                   │
│     e. Send Telegram alert                                │
└─────────────────────────────────────────────────────────┘
```

### Anomaly Types

| Type | Trigger | Default Threshold |
|------|---------|-------------------|
| `NO_TRADE` | No executions within window | 60 minutes |
| `TRADE_LOSS` | Single trade loss exceeds limit | $100 |
| `DAILY_LOSS` | Cumulative daily loss exceeds limit | $200 |
| `CONSEC_LOSS` | Consecutive losing trades | 3 in a row |

### Copilot Reasoning Loop

Each diagnostic session runs up to 5 rounds:

1. **Initial Diagnosis** — Full structured prompt with all context
2. **Challenge Assumptions** — Questions the initial analysis
3. **Edge Cases** — Explores unusual scenarios and false alarms
4. **Overfitting Risk** — Evaluates if suggested changes would overfit
5. **Risk-Adjusted Recommendations** — Final ranked suggestions

### Dedup & Cooldown

- Each anomaly gets a unique hash ID
- Same anomaly won't re-trigger within the dedup window (60 min default)
- Per-type cooldowns prevent alert fatigue
- Rate limited to max 3 analyses per hour

## Prerequisites

```bash
# GitHub CLI with Copilot extension
brew install gh
gh auth login
gh extension install github/gh-copilot

# Verify Copilot CLI
gh copilot --version

# Python dependencies (already in ShreeBot requirements.txt)
pip install pyyaml
```

## Usage

### Run Continuously (Production)

```bash
# Standard — monitors and interacts with Copilot
python3 agent/agent.py

# Background with nohup
nohup python3 agent/agent.py > /dev/null 2>&1 &

# Using the start script
./start_analyst.sh
```

### Single Check (Debug)

```bash
# One detection cycle, then exit
python3 agent/agent.py --once

# Dry run — detect anomalies but skip Copilot calls
python3 agent/agent.py --once --dry-run

# Verbose logging
python3 agent/agent.py --once --dry-run -v
```

### macOS launchd Service (Run on Boot)

Create `~/Library/LaunchAgents/com.shreebot.analyst.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.shreebot.analyst</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/python3</string>
        <string>/Users/svss/Documents/code/ShreeBot/agent/agent.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/svss/Documents/code/ShreeBot</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/Users/svss/Documents/code/ShreeBot/agent/logs/launchd_stdout.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/svss/Documents/code/ShreeBot/agent/logs/launchd_stderr.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
```

Install and manage:
```bash
# Install the service
launchctl load ~/Library/LaunchAgents/com.shreebot.analyst.plist

# Check status
launchctl list | grep shreebot

# Stop
launchctl unload ~/Library/LaunchAgents/com.shreebot.analyst.plist
```

## Configuration

Edit `agent/config.json`:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `anomaly_detection` | `inactivity_minutes_threshold` | 60 | Minutes before no-trade alert |
| `anomaly_detection` | `max_loss_per_trade_usd` | 100 | Single trade loss threshold |
| `anomaly_detection` | `daily_loss_threshold_usd` | 200 | Daily cumulative loss threshold |
| `anomaly_detection` | `consecutive_losses_alert` | 3 | Consecutive loss streak threshold |
| `copilot` | `max_iterations` | 5 | Max reasoning rounds per session |
| `copilot` | `timeout_seconds` | 120 | CLI timeout per invocation |
| `cooldowns` | `analysis_cooldown_minutes` | 30 | Min time between analyses |
| `cooldowns` | `dedup_window_minutes` | 60 | Window for deduplicating same anomaly |
| `cooldowns` | `max_analyses_per_hour` | 3 | Rate limit |
| `telegram` | `enabled` | false | Enable Telegram alerts |
| `polling` | `check_interval_seconds` | 60 | Polling frequency |

### Telegram Setup

```bash
export TELEGRAM_BOT_TOKEN="your-bot-token"
export TELEGRAM_CHAT_ID="your-chat-id"
```

Then set `"enabled": true` in the telegram section of `config.json`.

## Output

### Reports Directory (`agent/reports/`)

Each anomaly generates:
- `RPT_<id>_<timestamp>.json` — Full structured data
- `report_<id>_<timestamp>.md` — Human-readable Markdown

### Logs Directory (`agent/logs/`)

- `agent.log` — Main agent activity log
- `copilot_session_<id>.json` — Full Copilot reasoning transcript
- `prompt_<id>.txt` — Generated prompts (dry-run mode)

## Safety

- **No infinite loops** — Bounded iterations, rate limits, cooldowns
- **Exception safe** — Every component wrapped in try/except
- **Graceful shutdown** — Handles SIGINT/SIGTERM
- **Idempotent** — Same anomaly won't re-trigger within dedup window
- **Advisory only** — Never touches trading config or strategy code
