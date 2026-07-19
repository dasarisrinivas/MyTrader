#!/bin/bash
# install.sh - install the MyTrader launchd automation stack (per-user).
# Idempotent: safe to re-run after edits. Rollback: ./uninstall.sh
#
# Usage:  cd deploy/macos && ./install.sh [--no-load]
#   --no-load   render + copy plists but don't bootstrap them (dry-ish run)

set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$DEPLOY_DIR/../.." && pwd)"
AGENTS_DIR="$HOME/Library/LaunchAgents"
MYTRADER_HOME="$HOME/.mytrader"
DOMAIN="gui/$(id -u)"
LABELS=(com.mytrader.ibgateway com.mytrader.bot com.mytrader.watchdog com.mytrader.bootverify)
NO_LOAD=0
[ "${1:-}" = "--no-load" ] && NO_LOAD=1

fail() { echo "ERROR: $*" >&2; exit 1; }

[ "$(uname)" = "Darwin" ] || fail "this installer is macOS-only"

echo "== MyTrader macOS automation installer =="
echo "project: $PROJECT_ROOT"

# --- 1. Preflight checks ----------------------------------------------------
command -v nc >/dev/null       || fail "nc not found"
command -v security >/dev/null || fail "security (Keychain CLI) not found"

PY="$PROJECT_ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3 || true)"
[ -n "$PY" ] || fail "no python3 found; create the project venv first"
"$PY" -c "import ib_insync" 2>/dev/null \
    || echo "WARN: ib_insync not importable by $PY — deep API probes will fail (pip install -r requirements.txt)"

[ -f "$PROJECT_ROOT/config.yaml" ] \
    || echo "WARN: $PROJECT_ROOT/config.yaml missing — bot will not start until you create it"

IBC_PATH_DEFAULT="/opt/ibc"
if [ ! -f "$IBC_PATH_DEFAULT/scripts/ibcstart.sh" ]; then
    echo "WARN: IBC not found at $IBC_PATH_DEFAULT."
    echo "      Install it:  https://github.com/IbcAlpha/IBC/releases"
    echo "      (unzip to /opt/ibc, chmod +x /opt/ibc/scripts/*.sh)"
    echo "      Or set IBC_PATH in ~/.mytrader/env after install."
fi

# --- 2. Credentials ---------------------------------------------------------
if ! "$DEPLOY_DIR/bin/credentials.sh" check >/dev/null 2>&1; then
    echo
    echo "IB credentials are not in the Keychain yet."
    if [ -t 0 ]; then
        "$DEPLOY_DIR/bin/credentials.sh" setup
    else
        echo "Non-interactive shell: run  $DEPLOY_DIR/bin/credentials.sh setup  before first boot."
    fi
fi

# --- 3. Directories, env file, kill-switch flag ------------------------------
mkdir -p "$MYTRADER_HOME/state" "$MYTRADER_HOME/runtime" "$PROJECT_ROOT/logs/launchd"
chmod 700 "$MYTRADER_HOME" "$MYTRADER_HOME/runtime"
if [ ! -f "$MYTRADER_HOME/env" ]; then
    cp "$DEPLOY_DIR/env.example" "$MYTRADER_HOME/env"
    echo "created $MYTRADER_HOME/env (edit to configure ports/mode)"
fi
# Trading enabled by default; stop_trading.sh removes this flag.
touch "$MYTRADER_HOME/trading.enabled"

chmod +x "$DEPLOY_DIR"/bin/*.sh

# --- 4. Render + install plists ---------------------------------------------
mkdir -p "$AGENTS_DIR"
for label in "${LABELS[@]}"; do
    src="$DEPLOY_DIR/launchd/$label.plist.template"
    dst="$AGENTS_DIR/$label.plist"
    [ -f "$dst" ] && cp "$dst" "$dst.bak"   # rollback aid
    sed -e "s|__DEPLOY_DIR__|$DEPLOY_DIR|g" \
        -e "s|__PROJECT_ROOT__|$PROJECT_ROOT|g" \
        -e "s|__HOME__|$HOME|g" \
        "$src" > "$dst"
    plutil -lint "$dst" >/dev/null || fail "generated plist invalid: $dst"
    echo "installed $dst"
done

# --- 5. (Re)load services ----------------------------------------------------
if [ "$NO_LOAD" = "1" ]; then
    echo "--no-load: skipping launchctl bootstrap"
else
    for label in "${LABELS[@]}"; do
        launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
    done
    sleep 1
    for label in "${LABELS[@]}"; do
        launchctl bootstrap "$DOMAIN" "$AGENTS_DIR/$label.plist" \
            || fail "bootstrap failed for $label"
        echo "loaded  $label"
    done
fi

# --- 6. Power settings (need sudo; optional) ---------------------------------
echo
echo "Recommended power settings (run once, needs sudo):"
echo "  sudo pmset -a sleep 0 disksleep 0 displaysleep 5 autorestart 1 womp 1"
echo "  (never sleep, auto-restart after power failure, wake-on-LAN)"
echo
echo "Also enable auto-login for this account so the Aqua session (and IB"
echo "Gateway's GUI) comes up without a keyboard:"
echo "  System Settings > Users & Groups > Automatically log in as: $USER"
echo "  NOTE: auto-login requires FileVault to be OFF on the boot volume."
echo
echo "== Install complete =="
echo "  status:      $DEPLOY_DIR/bin/status.sh"
echo "  kill switch: $PROJECT_ROOT/stop_trading.sh"
echo "  rollback:    $DEPLOY_DIR/uninstall.sh"
