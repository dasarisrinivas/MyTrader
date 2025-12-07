# Alerts and Staged Actions - Operator Guide

This directory contains alerts and staged actions that require operator review.

## Key Files to Monitor

### 1. `staged_actions.json` (Project Root)

This file contains all pending reconciliation actions that will be executed when `DRY_RUN_CONFIRM=true`.

**Always review this file before enabling `DRY_RUN_CONFIRM`!**

Structure:
```json
{
  "generated_at": "2025-12-07T12:00:00.000Z",
  "config": {
    "dry_run": false,
    "safe_mode": false,
    "dry_run_confirm": false
  },
  "plan": {
    "to_update": [...],
    "to_delete": [...],
    "to_insert": [...],
    "ambiguous_matches": [...],
    "spy_futures_actions": [...]
  }
}
```

### 2. `./logs/ambiguous_matches.json`

If the reconciliation finds orders that can't be clearly matched between IB and the database, they are logged here for manual review.

**DO NOT enable `DRY_RUN_CONFIRM` if ambiguous matches exist!**

### 3. `./status/reconcile_status.json`

Current status of the last reconciliation run:
```json
{
  "status": "COMPLETED",
  "last_run": "2025-12-07T12:00:00.000Z",
  "inserted": 0,
  "deleted": 0,
  "updated": 0,
  "ambiguous_count": 0
}
```

## Workflow for Enabling Live Trading

### Step 1: Verify Dry Run
```bash
# Edit config/local_reconcile.yml
DRY_RUN: true
SAFE_MODE: true
DRY_RUN_CONFIRM: false

# Run the bot
python run_bot.py
```

Check `staged_actions.json` to see what would be done.

### Step 2: Stage Actions (No Execution)
```bash
# Edit config/local_reconcile.yml
DRY_RUN: false
SAFE_MODE: false  # or keep true for read-only
DRY_RUN_CONFIRM: false  # IMPORTANT: Keep false

# Run the bot
python run_bot.py
```

Actions are staged but NOT executed. Review `staged_actions.json`.

### Step 3: Execute Staged Actions
```bash
# ONLY after reviewing staged_actions.json!
# Edit config/local_reconcile.yml
DRY_RUN: false
SAFE_MODE: false
DRY_RUN_CONFIRM: true

# Run the bot
python run_bot.py
```

Watch `./logs/reconcile.log` for execution results.

## Rollback Steps

If something goes wrong:

### 1. Immediately Enable Safe Mode
```bash
# Edit config/local_reconcile.yml
SAFE_MODE: true
```

This prevents any further trading operations.

### 2. Restore Database from Backup
```bash
# Find the latest backup
ls -la ./backups/

# Restore SQL dump
sqlite3 ./data/orders.db < ./backups/db_backup_YYYYMMDD_HHMMSS.sql

# OR restore binary copy
cp ./backups/db_backup_YYYYMMDD_HHMMSS.db ./data/orders.db
```

### 3. Verify Restoration
```bash
# Check record counts
sqlite3 ./data/orders.db "SELECT COUNT(*) FROM orders;"
sqlite3 ./data/orders.db "SELECT COUNT(*) FROM order_events;"
sqlite3 ./data/orders.db "SELECT COUNT(*) FROM executions;"
```

### 4. Reset to Dry Run Mode
```bash
# Edit config/local_reconcile.yml
DRY_RUN: true
SAFE_MODE: true
DRY_RUN_CONFIRM: false
```

## Safety Flags Reference

| Flag | Effect |
|------|--------|
| `DRY_RUN: true` | Simulates all operations, no changes made |
| `SAFE_MODE: true` | Read-only mode, no trading allowed |
| `DRY_RUN_CONFIRM: false` | Stages actions but doesn't execute |
| `DRY_RUN_CONFIRM: true` | Executes staged actions |
| `FORCE_SYNC_DELETE: false` | Soft-deletes orphaned orders |
| `FORCE_SYNC_DELETE: true` | Hard-deletes orphaned orders |

## Alert Types

### `ambiguous_matches_found`
Severity: HIGH
Action: Stop and review `./logs/ambiguous_matches.json`
Do NOT auto-resolve these matches.

### `backup_failed`
Severity: CRITICAL
Action: Do not proceed with reconciliation. Fix backup script.

### `reconcile_failed`
Severity: HIGH  
Action: System enters SAFE_MODE. Review logs and restore from backup if needed.

### `live_data_backpressure_drop`
Severity: MEDIUM
Action: Consider reducing tick subscription rate or increasing queue size.

### `order_blocked_safe_mode`
Severity: INFO
Action: Expected when SAFE_MODE is enabled. Disable SAFE_MODE to allow trading.

### `duplicate_submission_blocked`
Severity: INFO
Action: Idempotency working correctly. No action needed.

## Contact

If you encounter issues:
1. Enable SAFE_MODE immediately
2. Review ./logs/reconcile.log
3. Check staged_actions.json and ambiguous_matches.json
4. Restore from backup if necessary
