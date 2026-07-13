#!/usr/bin/env bash
set -e

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${JAMSYSTEM_PYTHON:-$BASE_DIR/jam_env/bin/python3}"
if [ ! -x "$PYTHON" ]; then
  PYTHON="${JAMSYSTEM_PYTHON:-python3}"
fi

SERVER_URL="${1:-${JAMSYSTEM_TRAIN_SERVER:-http://192.168.137.2:8008}}"
UPLOAD_DIR="${2:-${JAMSYSTEM_UPLOAD_DIR:-/mnt/usb/JamRecords}}"
INTERVAL="${JAMSYSTEM_SYNC_INTERVAL:-60}"

echo "[SYNC] python: $PYTHON"
echo "[SYNC] server: $SERVER_URL"
echo "[SYNC] upload dir: $UPLOAD_DIR"
echo "[SYNC] interval: $INTERVAL"

exec "$PYTHON" "$BASE_DIR/rk_server_sync.py" \
  --server "$SERVER_URL" \
  --upload-dir "$UPLOAD_DIR" \
  --interval "$INTERVAL" \
  --watch
