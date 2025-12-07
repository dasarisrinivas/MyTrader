#!/bin/bash
# =============================================================================
# Database Backup Script for MyTrader
# =============================================================================
# This script creates timestamped backups of the orders database
# Run before any reconciliation or destructive operations
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="${PROJECT_ROOT}/backups"
LOG_DIR="${PROJECT_ROOT}/logs"
DB_PATH="${PROJECT_ROOT}/data/orders.db"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Ensure directories exist
mkdir -p "$BACKUP_DIR"
mkdir -p "$LOG_DIR"

log_message() {
    local level=$1
    local message=$2
    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%S.000Z)
    echo "{\"timestamp\": \"$timestamp\", \"level\": \"$level\", \"message\": \"$message\"}" >> "${LOG_DIR}/reconcile.log"
    case $level in
        "ERROR") echo -e "${RED}[$level]${NC} $message" ;;
        "WARN")  echo -e "${YELLOW}[$level]${NC} $message" ;;
        "INFO")  echo -e "${GREEN}[$level]${NC} $message" ;;
        *)       echo "[$level] $message" ;;
    esac
}

log_message "INFO" "local_edit_session_started - Backup script initiated"
log_message "INFO" "Backup directory: $BACKUP_DIR"
log_message "INFO" "Database path: $DB_PATH"

# Check if database exists
if [ ! -f "$DB_PATH" ]; then
    log_message "WARN" "Database not found at $DB_PATH - creating empty backup marker"
    touch "${BACKUP_DIR}/db_backup_${TIMESTAMP}_NO_DB.marker"
    echo "No database found - this may be a fresh installation"
    exit 0
fi

# Backup filename
BACKUP_FILE="${BACKUP_DIR}/db_backup_${TIMESTAMP}.sql"
BACKUP_DB_FILE="${BACKUP_DIR}/db_backup_${TIMESTAMP}.db"

# Create SQL dump backup
log_message "INFO" "Creating SQL dump backup..."
sqlite3 "$DB_PATH" ".dump" > "$BACKUP_FILE" 2>&1
if [ $? -ne 0 ]; then
    log_message "ERROR" "backup_failed - SQL dump failed"
    exit 1
fi

# Create binary backup (direct copy)
log_message "INFO" "Creating binary database backup..."
cp "$DB_PATH" "$BACKUP_DB_FILE"
if [ $? -ne 0 ]; then
    log_message "ERROR" "backup_failed - Binary copy failed"
    exit 1
fi

# Calculate checksums
SQL_CHECKSUM=$(shasum -a 256 "$BACKUP_FILE" | cut -d' ' -f1)
DB_CHECKSUM=$(shasum -a 256 "$BACKUP_DB_FILE" | cut -d' ' -f1)

# Create backup manifest
MANIFEST_FILE="${BACKUP_DIR}/db_backup_${TIMESTAMP}.manifest.json"
cat > "$MANIFEST_FILE" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%S.000Z)",
    "backup_id": "${TIMESTAMP}",
    "source_db": "${DB_PATH}",
    "sql_dump": {
        "path": "${BACKUP_FILE}",
        "checksum_sha256": "${SQL_CHECKSUM}",
        "size_bytes": $(stat -f%z "$BACKUP_FILE" 2>/dev/null || stat -c%s "$BACKUP_FILE" 2>/dev/null || echo 0)
    },
    "binary_copy": {
        "path": "${BACKUP_DB_FILE}",
        "checksum_sha256": "${DB_CHECKSUM}",
        "size_bytes": $(stat -f%z "$BACKUP_DB_FILE" 2>/dev/null || stat -c%s "$BACKUP_DB_FILE" 2>/dev/null || echo 0)
    },
    "tables_backed_up": [
        "orders",
        "order_events",
        "executions"
    ]
}
EOF

log_message "INFO" "Backup completed successfully"
log_message "INFO" "SQL dump: $BACKUP_FILE"
log_message "INFO" "Binary copy: $BACKUP_DB_FILE"
log_message "INFO" "Manifest: $MANIFEST_FILE"
log_message "INFO" "SQL checksum: $SQL_CHECKSUM"
log_message "INFO" "DB checksum: $DB_CHECKSUM"

# Export specific tables for easier inspection
log_message "INFO" "Exporting individual tables..."

# Export orders table
sqlite3 "$DB_PATH" -header -csv "SELECT * FROM orders;" > "${BACKUP_DIR}/orders_${TIMESTAMP}.csv" 2>/dev/null || true

# Export order_events table  
sqlite3 "$DB_PATH" -header -csv "SELECT * FROM order_events;" > "${BACKUP_DIR}/order_events_${TIMESTAMP}.csv" 2>/dev/null || true

# Export executions table
sqlite3 "$DB_PATH" -header -csv "SELECT * FROM executions;" > "${BACKUP_DIR}/executions_${TIMESTAMP}.csv" 2>/dev/null || true

# Count records
ORDERS_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM orders;" 2>/dev/null || echo 0)
EVENTS_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM order_events;" 2>/dev/null || echo 0)
EXEC_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM executions;" 2>/dev/null || echo 0)

log_message "INFO" "Records backed up - orders: $ORDERS_COUNT, events: $EVENTS_COUNT, executions: $EXEC_COUNT"

# Cleanup old backups (keep last 30 days)
log_message "INFO" "Cleaning up backups older than 30 days..."
find "$BACKUP_DIR" -name "db_backup_*" -mtime +30 -delete 2>/dev/null || true
find "$BACKUP_DIR" -name "orders_*" -mtime +30 -delete 2>/dev/null || true
find "$BACKUP_DIR" -name "order_events_*" -mtime +30 -delete 2>/dev/null || true
find "$BACKUP_DIR" -name "executions_*" -mtime +30 -delete 2>/dev/null || true

echo ""
echo "=========================================="
echo "  BACKUP COMPLETED SUCCESSFULLY"
echo "=========================================="
echo "  Timestamp: $TIMESTAMP"
echo "  SQL Dump:  $BACKUP_FILE"
echo "  Binary:    $BACKUP_DB_FILE"
echo "  Manifest:  $MANIFEST_FILE"
echo ""
echo "  Records: $ORDERS_COUNT orders, $EVENTS_COUNT events, $EXEC_COUNT executions"
echo "=========================================="
echo ""
echo "To restore from this backup:"
echo "  sqlite3 $DB_PATH < $BACKUP_FILE"
echo "  OR"
echo "  cp $BACKUP_DB_FILE $DB_PATH"
echo ""

exit 0
