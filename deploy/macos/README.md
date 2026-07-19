# MyTrader — Headless Mac Mini Automation

Fully automatic IB Gateway + trading bot stack for a dedicated Mac Mini.
Goal: reboot the Mac, walk away, everything recovers and Telegram tells you
`🟢 System Ready`.

Everything lives in `deploy/macos/` and `~/.mytrader/`. Nothing in the
existing bot code is modified; the manual scripts (`start_bot.sh`, `stop.sh`)
keep working when the automation is uninstalled.

## Architecture

```
login (auto-login, Aqua session)
 ├── com.mytrader.ibgateway   KeepAlive=always      → IBC → IB Gateway (headless login, API :4002)
 ├── com.mytrader.bot         KeepAlive=flag file   → wait_for_api → run_bot.py
 ├── com.mytrader.watchdog    every 60s             → health checks + auto-restart + Telegram
 └── com.mytrader.bootverify  one-shot at login     → verifies chain → "System Ready" Telegram
```

Startup ordering is enforced by *readiness gating*, not launchd ordering:
the bot job blocks in `wait_for_api.sh` (TCP port **and** a real API
handshake + account summary via `ib_insync`) before `run_bot.py` starts.

| File | Purpose |
|---|---|
| `bin/common.sh` | shared config/logging/health primitives |
| `bin/credentials.sh` | Keychain secret management |
| `bin/start_ibgateway.sh` | renders IBC config from Keychain, runs gateway in foreground |
| `bin/wait_for_api.sh` / `bin/check_ib_api.py` | port + deep API readiness probe |
| `bin/start_bot_service.sh` | launchd entrypoint for `run_bot.py` |
| `bin/watchdog.sh` | 60s health monitor, auto-restart, alert dedup |
| `bin/boot_verify.sh` | boot chain verification + System Ready message |
| `bin/stop_trading.sh` / `bin/resume_trading.sh` | kill switch (bot only, gateway stays up) |
| `bin/status.sh` | one-shot status of everything |
| `bin/notify.sh` | Telegram sender (best-effort, never blocks) |
| `launchd/*.plist.template` | rendered into `~/Library/LaunchAgents` by the installer |
| `ibc/config.ini.template` | IBC settings; credentials injected at runtime only |
| `install.sh` / `uninstall.sh` | idempotent install, full rollback |

## Prerequisites (one-time, on the Mac Mini)

1. **Dedicated account** (recommended): create a standard macOS user
   (e.g. `trader`), do everything below as that user.
2. **IB Gateway** (not TWS) installed — download "IB Gateway, stable" from
   IBKR. Default install path `~/Applications/ibgateway/<version>`.
3. **IBC** (headless controller for IB Gateway,
   <https://github.com/IbcAlpha/IBC/releases>):
   ```bash
   sudo mkdir -p /opt/ibc && sudo unzip IBCMacos-*.zip -d /opt/ibc
   sudo chmod +x /opt/ibc/scripts/*.sh /opt/ibc/*.sh
   ```
4. **Project venv**: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
5. **`config.yaml`** present in the project root (copy from `config.example.yaml`).
6. **Auto-login + power** (makes reboot recovery touchless):
   - System Settings → Users & Groups → *Automatically log in as* the trader
     user. **FileVault must be OFF** on the boot volume for auto-login to work.
   - `sudo pmset -a sleep 0 disksleep 0 displaysleep 5 autorestart 1 womp 1`
     (never sleep, restart after power failure, wake-on-LAN).

## Install

```bash
cd ~/MyTrader/deploy/macos
./install.sh          # prompts for IB + Telegram credentials → Keychain
./bin/status.sh       # verify
```

Re-run `install.sh` any time after editing scripts/templates — it re-renders
and reloads everything.

## Credential management

Secrets live **only** in the macOS login Keychain:

| Keychain service | Content |
|---|---|
| `mytrader.ib.username` / `mytrader.ib.password` | IB Gateway login |
| `mytrader.telegram.bot_token` / `mytrader.telegram.chat_id` | Telegram |

Manage with `bin/credentials.sh {setup|check|get|delete}`.

**Security tradeoffs considered:**

- **macOS Keychain (chosen).** Encrypted at rest with the login password,
  unlocked automatically at login (which the auto-login trading account does
  anyway), ACL'd per item, nothing readable off a stolen disk without the
  account password, nothing in the repo or in `ps` output. Weakness: any
  process running *as the logged-in trading user* can request the item —
  acceptable on a dedicated single-purpose machine.
- **Encrypted credentials file** (e.g. `openssl enc` / `age`): portable, but
  the decryption key must itself live somewhere readable at boot, which
  reduces to obfuscation unless a human types a passphrase — that breaks the
  "reboot and walk away" requirement.
- **Environment variables / plist `EnvironmentVariables`**: worst option —
  plaintext on disk, visible via `launchctl print` and inherited by child
  processes. Only offered as an explicit opt-in fallback for the *Telegram*
  token (`~/.mytrader/env`), never for IB credentials.

Runtime caveat: IBC reads credentials from an ini file, so
`start_ibgateway.sh` renders `~/.mytrader/runtime/ibc-config.ini`
(dir `700`, file `600`) from the Keychain at each start. That file exists
while the gateway runs; it is owned and readable only by the trading user.

## Kill switch

```bash
./stop_trading.sh                        # repo root — stops bot, gateway stays up
deploy/macos/bin/resume_trading.sh       # resume
```

Mechanism: the bot's launchd `KeepAlive` is gated on the flag file
`~/.mytrader/trading.enabled`. The kill switch removes the flag (so neither
launchd nor the watchdog restarts the bot) and SIGTERMs the bot for a
graceful shutdown. Survives reboots: with the flag absent, the bot stays
down after a restart while the gateway still comes up.

## Health monitoring (watchdog, every 60s)

1. Gateway process dead → restart gateway job, Telegram alert.
2. Port 4002 closed 3 consecutive checks → restart gateway.
3. Deep probe every 5th run (`ib_insync` handshake + account summary on
   dedicated clientId 97): port open but API dead twice → restart gateway.
4. Bot process dead (and kill switch not engaged) → restart bot, alert.

Alerts fire on **state transitions** only (down once, recovery once) — no
1-per-minute spam. All decisions logged to `logs/launchd/automation.log`.

## Notifications sent

- 🚀 IB Gateway starting / 🔴 IB Gateway disconnected / ✅ recovered
- 🤖 Trading bot starting / ♻️ Bot restarted
- 🛑 KILL SWITCH engaged / ▶️ Trading resumed
- 🟢 System Ready after boot (with per-step checklist)
- ❌ Boot verification FAILED at step …

## Operations cheat sheet

```bash
deploy/macos/bin/status.sh                                   # full status
launchctl print gui/$(id -u)/com.mytrader.bot                # job detail
launchctl kickstart -k gui/$(id -u)/com.mytrader.ibgateway   # force-restart gateway
launchctl kickstart -k gui/$(id -u)/com.mytrader.bot         # force-restart bot
launchctl bootout gui/$(id -u)/com.mytrader.watchdog         # stop watchdog
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.mytrader.watchdog.plist  # start again
tail -f logs/launchd/automation.log logs/bot.log             # live logs
```

Manual full stop (everything, including gateway):

```bash
deploy/macos/uninstall.sh        # unload services + kill processes
deploy/macos/install.sh          # bring it all back
```

## Testing reboot recovery

1. `deploy/macos/bin/status.sh` — everything green.
2. `sudo reboot`.
3. Do **not** touch the machine. Within ~3–10 minutes you should receive the
   `🟢 System Ready` Telegram message.
4. Verify remotely (SSH): `deploy/macos/bin/status.sh`.

Also worth testing:
- **Gateway crash:** `pkill -9 -f ibcalpha.ibc` → watchdog alert + restart
  within ~60s, then recovery message.
- **Bot crash:** `pkill -9 -f run_bot.py` → `♻️ Bot restarted`.
- **Kill switch:** `./stop_trading.sh` → bot down, no restarts; reboot →
  gateway returns, bot stays down; `resume_trading.sh` → bot back.

## Rollback

`deploy/macos/uninstall.sh` removes the launchd jobs and stops the managed
processes. The repo, your config, and Keychain items are untouched
(`--purge-credentials` / `--purge-state` to remove those too). Previous
plists are saved as `*.plist.bak` by the installer.

## Notes & limitations

- **LaunchAgents, not LaunchDaemons**: IB Gateway is a GUI (Java) app and
  must run inside a logged-in Aqua session; that's why auto-login on the
  dedicated account is required. LaunchDaemons run pre-login and cannot host
  the gateway UI.
- **2FA**: fully touchless restart requires an IB user without interactive
  2FA on this device (paper accounts, or IB Key with the "seamless weekly
  re-auth" flow). With mandatory 2FA, the Sunday weekly re-login will wait
  for your phone approval — the watchdog will alert you when the API is down.
- The gateway self-restarts daily at 23:45 (IBC `AutoRestartTime`); brief
  API downtime around then is normal and the watchdog tolerates it.
- Port 4002 is the **paper** gateway port; live is 4001. Set both
  `IBKR_PORT` and `TRADING_MODE` consistently in `~/.mytrader/env`.
