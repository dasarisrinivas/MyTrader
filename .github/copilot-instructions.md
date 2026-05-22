# Copilot Instructions — ShreeBot (MyTrader)

## Operator map (what runs where)

- MES orders: `run_bot.py` → `shree/execution/live_trading_manager.py`.
- Gold orders: `run_gold.py` → `shree/execution/gold/manager.py`.
- SPY options alerts only: `run_spy_options.py` → `shree/spy_options/manager.py` (Telegram, no order placement).
- Bots are independent; they can share one IB Gateway using different client IDs and separate logs/state.

## Preflight before starting any bot

- Use `python3` only.
- Source wrappers (do not execute): `. start_bot.sh`, `. start_paper_bot.sh`, `. start_gold.sh`, `. start_spy_options.sh`.
- Confirm config file and mode:
  - MES live/paper guardrail uses `DEPLOY_ENV` + IB port (`paper→4002`, `prod/live→4001`) in `run_bot.py`.
  - Gold defaults to simulation unless overridden (`GOLD_SIMULATION=0` for live).
- Confirm IB Gateway is listening on expected port before start scripts continue.

## Start/stop runbook (day-to-day)

- MES: `. start_bot.sh` (live) / `. start_paper_bot.sh` (paper), stop with `. stop.sh`.
- Gold: `. start_gold.sh`, stop with `. stop_gold.sh`.
- SPY options: `. start_spy_options.sh`, stop with `. stop_spy_options.sh`.
- Startup scripts maintain PID files under `logs/`; stale PID cleanup is part of normal startup.

## Fast restart checklist (market-hours copy/paste)

- Use this order: **stop → verify down → start → verify up → tail logs**.

### MES (live)

```bash
. stop.sh
pgrep -f "python.*run_bot.py" || echo "MES stopped"
. start_bot.sh
pgrep -f "python.*run_bot.py"
tail -n 40 logs/live_trading.log
```

### MES (paper)

```bash
. stop_paper_bot.sh
pgrep -f "python.*run_bot.py" || echo "MES paper stopped"
. start_paper_bot.sh
pgrep -f "python.*run_bot.py"
tail -n 40 logs/live_trading.log
```

### Gold (paper default)

```bash
. stop_gold.sh
pgrep -f "python.*run_gold.py" || echo "Gold stopped"
. start_gold.sh
pgrep -f "python.*run_gold.py"
tail -n 40 logs/gold_trading.log
```

### Gold (live)

```bash
. stop_gold.sh
pgrep -f "python.*run_gold.py" || echo "Gold stopped"
GOLD_SIMULATION=0 . start_gold.sh
pgrep -f "python.*run_gold.py"
tail -n 40 logs/gold_trading.log
```

### SPY options (signal-only)

```bash
. stop_spy_options.sh
pgrep -f "python.*run_spy_options.py" || echo "SPY options stopped"
. start_spy_options.sh
pgrep -f "python.*run_spy_options.py"
tail -n 40 logs/spy_options.log
```

### If startup fails (all bots)

```bash
lsof -i:4001 -i:4002 -i:5000
pgrep -f "python.*run_bot.py|python.*run_gold.py|python.*run_spy_options.py"
```

## 30-second go/no-go criteria after restart

- **GO** only if all are true:
  - Expected process exists (`pgrep` returns PID for the target bot).
  - Target log file is updating with fresh timestamps in the last 1–2 lines.
  - No immediate startup error lines (`CRITICAL`, `Traceback`, `Fatal error`, repeated reconnect failures).
  - Mode/port matches intent (MES paper=4002, MES live=4001, Gold live only with `GOLD_SIMULATION=0`).
- **NO-GO / escalate** if any are true:
  - Process exits within ~30 seconds after start.
  - Port check fails or wrong gateway account/port is detected.
  - Risk guardrail mismatch or config-disabled bot message appears.
  - Log loops on connection/auth failures without reaching steady heartbeat.

### Quick check snippets

```bash
pgrep -f "python.*run_bot.py|python.*run_gold.py|python.*run_spy_options.py"
tail -n 20 logs/live_trading.log
tail -n 20 logs/gold_trading.log
tail -n 20 logs/spy_options.log
```

## Market-hours incident command pack (single paste)

```bash
echo "=== Process status ==="
pgrep -fl "python.*run_bot.py|python.*run_gold.py|python.*run_spy_options.py" || echo "No bot processes found"

echo "=== IB/CP gateway ports ==="
lsof -i:4001 -i:4002 -i:5000

echo "=== MES log (last 25) ==="
tail -n 25 logs/live_trading.log

echo "=== Gold log (last 25) ==="
tail -n 25 logs/gold_trading.log

echo "=== SPY Options log (last 25) ==="
tail -n 25 logs/spy_options.log
```

## Health checks and first triage

- Process alive: `pgrep -f "python.*run_bot.py|python.*run_gold.py|python.*run_spy_options.py"`.
- Primary logs:
  - MES: `logs/live_trading.log`
  - Gold: `logs/gold_trading.log`
  - SPY: `logs/spy_options.log`
- No-trade diagnostics:
  - MES: grep for `NO_SIGNAL`, `BLOCKED`, `CHOP` in `logs/live_trading.log`.
  - Gold: grep for `HOLD`, `cooldown`, `BLOCKED` in `logs/gold_trading.log`.
- Performance/P&L/trade history: query `data/trade_journal.db` first, not log grep.

## Safety rules during incidents

- Never bypass `RiskGate` (`shree/risk/risk_gate.py`) for new entry paths.
- Respect CME maintenance lockout (4–5 PM CT); bot logic already blocks entries there.
- Time logic must use `shree/utils/timezone_utils.py` (`now_cst()`), not raw `datetime.now()`.
- Keep max-contract protections intact; risk limits are reconciled conservatively in `shree/config/settings.py::Settings.validate()`.

## Recovery playbook after failures

- Step 1: stop affected bot cleanly with matching stop script.
- Step 2: verify IB connectivity/port and `DEPLOY_ENV` or `GOLD_SIMULATION` mode.
- Step 3: restart via wrapper script (not direct python command) to re-apply env guardrails.
- Step 4: confirm fresh heartbeat/log lines and PID recreation.
- Step 5: if behavior regressed after code change, run targeted tests:
  - `python3 -m pytest tests/ -x`
  - `python3 -m pytest tests/gold/ -v`

## Repo-specific implementation notes

- `LiveTradingManager` is large; prefer extending `shree/execution/components/` and wiring in.
- `one_minute:` still contains `ft_*` keys (15m naming legacy); avoid mass renames.
- `archive/main.py` is legacy; do not add new logic there.
- If renaming strategy classes, update `backtest/engine.py` imports to keep live/backtest parity.
