#!/bin/bash
set -e

RUN_ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG="$RUN_ROOT/autostart.log"

sleep 8
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
if [ -S "$XDG_RUNTIME_DIR/wayland-0" ]; then
    export QT_QPA_PLATFORM=wayland
else
    export DISPLAY=:0
    export QT_QPA_PLATFORM=xcb
fi

pkill -f transmitter_qt_gui || true
pkill -f ad9361_rk3588 || true
exec "$RUN_ROOT/start_tx_gui.sh" >> "$LOG" 2>&1
