#!/bin/bash
set -e

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

PORT="${1:-8090}"
URL="http://127.0.0.1:${PORT}"

echo "[TX-UI] starting transmitter UI on ${URL}"
python3 transmitter_ui.py --host 0.0.0.0 --port "$PORT" &
UI_PID=$!

sleep 1
if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$URL" >/dev/null 2>&1 || true
fi

wait "$UI_PID"
