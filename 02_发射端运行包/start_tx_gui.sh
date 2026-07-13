#!/bin/bash
set -e

RUN_ROOT="$(cd "$(dirname "$0")" && pwd)"
export JAMSYSTEM_BASE_PATH="${JAMSYSTEM_BASE_PATH:-$RUN_ROOT/JamSystem}"

cleanup_tx() {
  pkill -9 -f "ad9361_rk3588" 2>/dev/null || true
}

cleanup_tx
trap cleanup_tx EXIT INT TERM

cd "$RUN_ROOT/transmitter_qt_gui"
./transmitter_qt_gui
